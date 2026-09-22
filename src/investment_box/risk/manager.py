"""The risk manager: the last gate before an order exists.

Every proposed trade passes through :meth:`RiskManager.evaluate`, which returns
either a sized order or a specific refusal. There is no path around it.

The checks, in the order they run — cheapest and most absolute first, so a
refusal names the *first* rule broken rather than the last:

1. Engine paused (drawdown auto-pause, kill switch, manual)
2. Daily loss limit reached
3. Position count at maximum
4. Already holding this symbol
5. Trade frequency limits (per day, per week)
6. Sector concentration
7. Position sizing
8. Settled cash (good-faith violation guard)
9. Shariah hard constraints

The Shariah gate runs last and separately, in ``execution``, on the fully
formed order — so it sees the final quantity and price, not the proposal.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import func, select

from investment_box.config.schema import Settings
from investment_box.core.audit import AuditSink
from investment_box.core.clock import UTC, Clock, SystemClock, TradingCalendar
from investment_box.core.logging import get_logger
from investment_box.core.types import to_money
from investment_box.db.models import EquitySnapshot, Trade
from investment_box.db.session import Database
from investment_box.execution.base import AccountSnapshot, BrokerPosition
from investment_box.risk.settlement import SettlementLedger, SettlementSnapshot
from investment_box.risk.sizing import PositionSize, PositionSizer

log = get_logger(__name__)


class RiskVerdict(StrEnum):
    """Why a proposal was or was not allowed."""

    APPROVED = "approved"
    PAUSED = "engine_paused"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    MAX_POSITIONS = "max_positions"
    ALREADY_HELD = "already_held"
    TRADE_LIMIT = "trade_limit"
    SECTOR_LIMIT = "sector_limit"
    UNSIZABLE = "unsizable"
    INSUFFICIENT_SETTLED_CASH = "insufficient_settled_cash"
    MIN_HOLD_NOT_MET = "min_hold_not_met"

    @property
    def is_approved(self) -> bool:
        return self is RiskVerdict.APPROVED


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """The answer, with the reason attached either way."""

    symbol: str
    verdict: RiskVerdict
    reason: str
    size: PositionSize | None = None

    @property
    def is_approved(self) -> bool:
        return self.verdict.is_approved and self.size is not None and self.size.is_tradable


@dataclass
class RiskState:
    """A snapshot of every limit and how close it is to binding.

    Rendered in the dashboard and by ``/status``, so limits are visible before
    they bite rather than only when a trade is refused.
    """

    as_of: dt.date
    equity: Decimal
    peak_equity: Decimal
    drawdown: float
    day_pnl: Decimal | None
    day_pnl_pct: float | None
    open_positions: int
    max_positions: int
    trades_today: int
    max_trades_today: int
    trades_this_week: int
    max_trades_this_week: int
    settlement: SettlementSnapshot | None = None
    is_paused: bool = False
    pause_reason: str = ""
    breaches: list[str] = field(default_factory=list)

    @property
    def position_slots_free(self) -> int:
        return max(0, self.max_positions - self.open_positions)

    @property
    def can_open_new(self) -> bool:
        return not self.is_paused and self.position_slots_free > 0 and not self.breaches

    def summary(self) -> str:
        if self.is_paused:
            return f"PAUSED: {self.pause_reason}"
        if self.breaches:
            return "; ".join(self.breaches)
        return (
            f"{self.open_positions}/{self.max_positions} positions, "
            f"{self.trades_today}/{self.max_trades_today} trades today, "
            f"drawdown {self.drawdown:.1%}"
        )


class RiskManager:
    """Enforces every limit and sizes every position."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        ledger: SettlementLedger,
        audit: AuditSink,
        *,
        clock: Clock | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.settings = settings
        self.db = database
        self.ledger = ledger
        self.audit = audit
        self.clock = clock or SystemClock()
        self.calendar = calendar or TradingCalendar()
        self.sizer = PositionSizer(settings)
        self._paused = False
        self._pause_reason = ""

    # ---------------------------------------------------------------- pausing

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self, reason: str, *, actor: str = "engine") -> None:
        """Stop opening new positions. Exits are still permitted.

        Pausing never blocks an exit: being unable to close a losing position
        because a loss limit was hit would be the worst possible behaviour.
        """
        if self._paused:
            return
        self._paused = True
        self._pause_reason = reason
        log.warning("risk.paused", reason=reason, actor=actor)
        self.audit.record("risk.paused", f"Engine paused: {reason}", actor=actor)

    def resume(self, *, actor: str = "user") -> None:
        if not self._paused:
            return
        log.info("risk.resumed", actor=actor, was=self._pause_reason)
        self.audit.record(
            "risk.resumed", f"Engine resumed (was: {self._pause_reason})", actor=actor
        )
        self._paused = False
        self._pause_reason = ""

    # ----------------------------------------------------------------- state

    def state(
        self, account: AccountSnapshot, positions: list[BrokerPosition]
    ) -> RiskState:
        """Assemble the full risk picture, and auto-pause if a hard limit broke."""
        today = self.clock.now().astimezone(UTC).date()
        peak = self._peak_equity()
        equity = account.equity
        drawdown = float(equity / peak - 1) if peak > 0 else 0.0

        day_pnl, day_pnl_pct = self._day_pnl(equity)
        trades_today = self._trade_count(since=today)
        week_start = today - dt.timedelta(days=today.weekday())
        trades_week = self._trade_count(since=week_start)

        state = RiskState(
            as_of=today,
            equity=equity,
            peak_equity=peak,
            drawdown=drawdown,
            day_pnl=day_pnl,
            day_pnl_pct=day_pnl_pct,
            open_positions=len(positions),
            max_positions=self.settings.risk.max_open_positions,
            trades_today=trades_today,
            max_trades_today=self.settings.risk.max_trades_per_day,
            trades_this_week=trades_week,
            max_trades_this_week=self.settings.risk.max_trades_per_week,
            settlement=self.ledger.snapshot(account.cash.settled, as_of=today),
            is_paused=self._paused,
            pause_reason=self._pause_reason,
        )

        # Drawdown is a hard stop: it pauses the engine rather than merely
        # refusing the current trade.
        if drawdown <= -self.settings.risk.max_drawdown_pct:
            reason = (
                f"drawdown {drawdown:.1%} breached the "
                f"{self.settings.risk.max_drawdown_pct:.0%} limit"
            )
            state.breaches.append(reason)
            self.pause(reason, actor="risk_manager")
            state.is_paused = True
            state.pause_reason = reason

        if day_pnl_pct is not None and day_pnl_pct <= -self.settings.risk.max_daily_loss_pct:
            state.breaches.append(
                f"day loss {day_pnl_pct:.1%} reached the "
                f"{self.settings.risk.max_daily_loss_pct:.0%} daily limit"
            )

        return state

    # ------------------------------------------------------------- evaluation

    def evaluate(
        self,
        *,
        symbol: str,
        entry_price: Decimal,
        account: AccountSnapshot,
        positions: list[BrokerPosition],
        atr: Decimal | None = None,
        stop_atr_mult: float | None = None,
        take_profit_atr_mult: float | None = None,
        sector: str | None = None,
    ) -> RiskDecision:
        """Decide whether and how large to trade. Never raises."""
        state = self.state(account, positions)

        def refuse(verdict: RiskVerdict, reason: str) -> RiskDecision:
            log.info("risk.refused", symbol=symbol, verdict=verdict.value, reason=reason)
            return RiskDecision(symbol=symbol, verdict=verdict, reason=reason)

        if state.is_paused:
            return refuse(RiskVerdict.PAUSED, f"engine is paused: {state.pause_reason}")

        for breach in state.breaches:
            if "daily limit" in breach:
                return refuse(RiskVerdict.DAILY_LOSS_LIMIT, breach)
            return refuse(RiskVerdict.DRAWDOWN_LIMIT, breach)

        if any(p.symbol.upper() == symbol.upper() for p in positions):
            return refuse(
                RiskVerdict.ALREADY_HELD,
                f"already holding {symbol}; this system does not add to positions",
            )

        if state.position_slots_free <= 0:
            return refuse(
                RiskVerdict.MAX_POSITIONS,
                f"at the maximum of {state.max_positions} open positions",
            )

        if state.trades_today >= state.max_trades_today:
            return refuse(
                RiskVerdict.TRADE_LIMIT,
                f"{state.trades_today} trades today, limit {state.max_trades_today}",
            )

        if state.trades_this_week >= state.max_trades_this_week:
            return refuse(
                RiskVerdict.TRADE_LIMIT,
                f"{state.trades_this_week} trades this week, "
                f"limit {state.max_trades_this_week}",
            )

        sector_reason = self._sector_check(sector, positions, account)
        if sector_reason is not None:
            return refuse(RiskVerdict.SECTOR_LIMIT, sector_reason)

        settlement = state.settlement
        available = settlement.available if settlement else account.cash.available_for_trading
        # Hold back the configured cash buffer as well as respecting settlement.
        buffer = to_money(
            self.settings.capital.allocation_usd
            * Decimal(str(self.settings.capital.cash_buffer_pct))
        )
        deployable = to_money(max(Decimal("0"), available - buffer))

        size = self.sizer.size(
            symbol=symbol,
            entry_price=entry_price,
            available_cash=deployable,
            atr=atr,
            stop_atr_mult=stop_atr_mult,
            take_profit_atr_mult=take_profit_atr_mult,
        )
        if not size.is_tradable:
            return refuse(RiskVerdict.UNSIZABLE, size.reason)

        if settlement is not None and not self.ledger.can_afford(size.notional, settlement):
            return refuse(
                RiskVerdict.INSUFFICIENT_SETTLED_CASH,
                f"${size.notional} needed but only ${settlement.available} is settled. "
                f"{settlement.explain()}",
            )

        return RiskDecision(
            symbol=symbol,
            verdict=RiskVerdict.APPROVED,
            reason=size.reason,
            size=size,
        )

    def can_exit(self, symbol: str, entry_date: dt.date) -> tuple[bool, str]:
        """Whether the minimum holding period has elapsed.

        Returns a reason either way, because "why is it still holding X?" needs
        an answer as much as "why did it sell X?".
        """
        today = self.clock.now().astimezone(UTC).date()
        held = self.calendar.trading_days_between(entry_date, today)
        minimum = self.settings.holding.min_holding_days
        if held >= minimum:
            return True, f"held {held} trading days (minimum {minimum})"
        earliest = self.calendar.add_trading_days(entry_date, minimum)
        return False, (
            f"held {held} of {minimum} required trading days; "
            f"earliest exit is {earliest}"
        )

    # -------------------------------------------------------------- internals

    def _peak_equity(self) -> Decimal:
        with self.db.session() as session:
            peak = session.scalar(
                select(func.max(EquitySnapshot.equity)).where(
                    EquitySnapshot.trading_mode == self.settings.trading_mode.value
                )
            )
        return to_money(peak) if peak else self.settings.capital.allocation_usd

    def _day_pnl(self, equity: Decimal) -> tuple[Decimal | None, float | None]:
        today = self.clock.now().astimezone(UTC).date()
        with self.db.session() as session:
            previous = session.scalar(
                select(EquitySnapshot.equity)
                .where(
                    EquitySnapshot.snapshot_date < today,
                    EquitySnapshot.trading_mode == self.settings.trading_mode.value,
                )
                .order_by(EquitySnapshot.snapshot_date.desc())
                .limit(1)
            )
        if previous is None or previous == 0:
            return None, None
        delta = to_money(equity - previous)
        return delta, float(delta / previous)

    def _trade_count(self, *, since: dt.date) -> int:
        with self.db.session() as session:
            count = session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.entry_date >= since,
                    Trade.trading_mode == self.settings.trading_mode.value,
                )
            )
        return int(count or 0)

    def _sector_check(
        self, sector: str | None, positions: list[BrokerPosition], account: AccountSnapshot
    ) -> str | None:
        """Concentration guard.

        Sector data is not available for these ETFs from the current data
        layer, so this is a structural placeholder that returns ``None`` when
        it cannot judge -- rather than silently passing everything while
        appearing to check. Phase 6 wires real sector metadata.
        """
        if sector is None:
            return None
        limit = Decimal(str(self.settings.risk.max_sector_exposure_pct))
        allocation = self.settings.capital.allocation_usd
        if allocation <= 0:
            return None
        current = to_money(
            sum(
                (p.market_value for p in positions if getattr(p, "sector", None) == sector),
                Decimal("0"),
            )
        )
        if current / allocation >= limit:
            return (
                f"sector {sector} already at {current / allocation:.0%} of capital, "
                f"limit {limit:.0%}"
            )
        return None
