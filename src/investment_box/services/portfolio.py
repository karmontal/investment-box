"""Account and portfolio state.

The views here are presentation-ready value objects: the dashboard and the
Telegram bot render them without doing arithmetic of their own, so the two
cannot drift apart.

The distinction this module exists to make visible is **account equity vs.
allocated capital**. The bot may be allowed $500 of a larger account. Every
percentage the user sees -- risk used, position size, drawdown -- is measured
against the allocation, not the account, because that is the number the rules
are written in terms of.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select

from investment_box.config.schema import Settings
from investment_box.core.clock import UTC, Clock, SystemClock, to_display
from investment_box.core.logging import get_logger
from investment_box.core.types import Symbol, TradingMode, to_money
from investment_box.db.models import EquitySnapshot, Trade
from investment_box.db.session import Database
from investment_box.execution.base import AccountSnapshot, Broker, BrokerPosition

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PositionView:
    """One open position, ready to render."""

    symbol: Symbol
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: float
    is_fractional: bool
    days_held: int | None
    pct_of_allocation: float
    #: ``None`` until Phase 3 populates screens; rendered as "unknown" rather
    #: than silently as "compliant".
    compliance_status: str | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    #: True when the stop is held by the engine, not the broker.
    stop_is_synthetic: bool = False

    @property
    def is_protected(self) -> bool:
        """Whether a broker-side stop protects this position if the engine dies."""
        return self.stop_loss_price is not None and not self.stop_is_synthetic


@dataclass(frozen=True, slots=True)
class CapitalUsage:
    """How much of the bot's allocation is deployed, and what is left."""

    allocation: Decimal
    deployed: Decimal
    available_settled: Decimal
    unsettled: Decimal
    cash_buffer: Decimal
    open_positions: int
    max_open_positions: int

    @property
    def deployed_pct(self) -> float:
        if self.allocation <= 0:
            return 0.0
        return float(self.deployed / self.allocation)

    @property
    def position_slots_free(self) -> int:
        return max(0, self.max_open_positions - self.open_positions)

    @property
    def is_fully_invested(self) -> bool:
        return self.position_slots_free == 0 or self.available_settled <= self.cash_buffer


@dataclass(frozen=True, slots=True)
class AccountView:
    """Everything ``/balance`` and the dashboard header need, in one object."""

    trading_mode: TradingMode
    equity: Decimal
    cash_settled: Decimal
    cash_unsettled: Decimal
    positions_value: Decimal
    capital: CapitalUsage
    positions: tuple[PositionView, ...]
    day_pnl: Decimal | None
    day_pnl_pct: float | None
    all_time_pnl: Decimal | None
    taken_at: dt.datetime
    is_cash_account: bool
    trading_blocked: bool
    broker_name: str
    warnings: tuple[str, ...] = ()

    @property
    def mode_tag(self) -> str:
        """The ``[PAPER]`` / ``[LIVE]`` prefix every Telegram message carries."""
        return f"[{self.trading_mode.value.upper()}]"

    def taken_at_local(self, tz_name: str = "Asia/Jerusalem") -> dt.datetime:
        from zoneinfo import ZoneInfo

        return to_display(self.taken_at, ZoneInfo(tz_name))


class PortfolioService:
    """Read-side of account and portfolio state."""

    def __init__(
        self,
        broker: Broker,
        database: Database,
        settings: Settings,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.broker = broker
        self.db = database
        self.settings = settings
        self.clock = clock or SystemClock()

    # ------------------------------------------------------------------ reads

    def get_account_view(self) -> AccountView:
        """Assemble the full account picture.

        Broker state is the source of truth for cash and positions; the
        database supplies the history the broker does not keep (entry dates,
        strategy attribution, stop levels).
        """
        snapshot = self.broker.get_account()
        positions = self._build_position_views(snapshot)
        usage = self._capital_usage(snapshot, len(positions))

        day_pnl, day_pnl_pct = self._day_pnl(snapshot.equity)
        warnings = list(self.settings.startup_warnings())

        if not snapshot.is_cash_account:
            warnings.insert(
                0,
                "BROKER ACCOUNT IS NOT A CASH ACCOUNT. This application requires a cash "
                "account: margin is never permitted. Trading is blocked until this is fixed.",
            )
        if snapshot.trading_blocked:
            warnings.insert(0, "Broker reports trading is blocked on this account.")

        unprotected = [p.symbol for p in positions if p.stop_loss_price and p.stop_is_synthetic]
        if unprotected:
            warnings.append(
                f"{len(unprotected)} position(s) rely on an engine-managed stop "
                f"({', '.join(unprotected)}). If the engine stops, they are unprotected."
            )

        return AccountView(
            trading_mode=self.settings.trading_mode,
            equity=snapshot.equity,
            cash_settled=snapshot.cash.settled,
            cash_unsettled=snapshot.cash.unsettled,
            positions_value=snapshot.positions_value,
            capital=usage,
            positions=tuple(positions),
            day_pnl=day_pnl,
            day_pnl_pct=day_pnl_pct,
            all_time_pnl=self.realized_pnl_total(),
            taken_at=snapshot.taken_at or self.clock.now(),
            is_cash_account=snapshot.is_cash_account,
            trading_blocked=snapshot.trading_blocked,
            broker_name=self.broker.name,
            warnings=tuple(warnings),
        )

    def get_positions(self) -> list[PositionView]:
        return list(self.get_account_view().positions)

    def realized_pnl_total(self) -> Decimal:
        """Net realised P&L across every closed trade in the current mode."""
        with self.db.session() as session:
            rows = session.scalars(
                select(Trade.net_pnl).where(
                    Trade.exit_at.is_not(None),
                    Trade.trading_mode == self.settings.trading_mode.value,
                )
            ).all()
        return to_money(sum((value for value in rows if value is not None), Decimal("0")))

    def closed_trades(self, limit: int = 10) -> list[Trade]:
        """Most recently closed trades, newest first."""
        with self.db.session() as session:
            return list(
                session.scalars(
                    select(Trade)
                    .where(
                        Trade.exit_at.is_not(None),
                        Trade.trading_mode == self.settings.trading_mode.value,
                    )
                    .order_by(Trade.exit_at.desc())
                    .limit(limit)
                ).all()
            )

    def open_trades(self) -> list[Trade]:
        with self.db.session() as session:
            return list(
                session.scalars(
                    select(Trade).where(
                        Trade.exit_at.is_(None),
                        Trade.trading_mode == self.settings.trading_mode.value,
                    )
                ).all()
            )

    # ------------------------------------------------------------------ write

    def record_equity_snapshot(self, *, on_date: dt.date | None = None) -> EquitySnapshot:
        """Persist today's equity point. Idempotent per date."""
        snapshot = self.broker.get_account()
        day = on_date or self.clock.now().astimezone(UTC).date()
        previous = self._previous_equity(day)

        with self.db.session() as session:
            existing = session.scalar(
                select(EquitySnapshot).where(EquitySnapshot.snapshot_date == day)
            )
            row = existing or EquitySnapshot(snapshot_date=day)
            row.taken_at = self.clock.now()
            row.equity = snapshot.equity
            row.cash_settled = snapshot.cash.settled
            row.cash_unsettled = snapshot.cash.unsettled
            row.positions_value = snapshot.positions_value
            row.day_pnl = to_money(snapshot.equity - previous) if previous is not None else None
            row.open_positions = len(self.broker.get_positions())
            row.trading_mode = self.settings.trading_mode.value
            if existing is None:
                session.add(row)
            session.flush()
            session.expunge(row)
        return row

    def equity_curve(self, days: int = 365) -> list[EquitySnapshot]:
        cutoff = self.clock.now().astimezone(UTC).date() - dt.timedelta(days=days)
        with self.db.session() as session:
            return list(
                session.scalars(
                    select(EquitySnapshot)
                    .where(
                        EquitySnapshot.snapshot_date >= cutoff,
                        EquitySnapshot.trading_mode == self.settings.trading_mode.value,
                    )
                    .order_by(EquitySnapshot.snapshot_date)
                ).all()
            )

    # --------------------------------------------------------------- internals

    def _build_position_views(self, snapshot: AccountSnapshot) -> list[PositionView]:
        allocation = self.settings.capital.allocation_usd
        open_trades = {trade.symbol: trade for trade in self.open_trades()}
        today = self.clock.now().astimezone(UTC).date()

        views: list[PositionView] = []
        for position in self.broker.get_positions():
            trade = open_trades.get(str(position.symbol))
            views.append(
                PositionView(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_entry_price=position.avg_entry_price,
                    current_price=position.current_price,
                    market_value=position.market_value,
                    unrealized_pnl=position.unrealized_pnl,
                    unrealized_pnl_pct=position.unrealized_pnl_pct,
                    is_fractional=position.is_fractional,
                    days_held=self._days_held(position, trade, today),
                    pct_of_allocation=(
                        float(position.market_value / allocation) if allocation > 0 else 0.0
                    ),
                    compliance_status=trade.compliance_status_at_entry if trade else None,
                    stop_loss_price=trade.stop_loss_price if trade else None,
                    take_profit_price=trade.take_profit_price if trade else None,
                    stop_is_synthetic=position.is_fractional,
                )
            )
        return views

    def _days_held(
        self, position: BrokerPosition, trade: Trade | None, today: dt.date
    ) -> int | None:
        """Trading days held. Calendar days would make a weekend look like a hold."""
        entry = trade.entry_date if trade else (
            position.opened_at.astimezone(UTC).date() if position.opened_at else None
        )
        if entry is None:
            return None
        from investment_box.core.clock import TradingCalendar

        return TradingCalendar(anchor=today).trading_days_between(entry, today)

    def _capital_usage(self, snapshot: AccountSnapshot, open_positions: int) -> CapitalUsage:
        allocation = self.settings.capital.allocation_usd
        buffer_amount = to_money(
            allocation * Decimal(str(self.settings.capital.cash_buffer_pct))
        )
        # The bot may not deploy more than its allocation even if the account
        # holds more, so available cash is clipped to what the allocation leaves.
        allocation_headroom = to_money(max(Decimal("0"), allocation - snapshot.positions_value))
        available = min(snapshot.cash.available_for_trading, allocation_headroom)

        return CapitalUsage(
            allocation=allocation,
            deployed=snapshot.positions_value,
            available_settled=to_money(max(Decimal("0"), available)),
            unsettled=snapshot.cash.unsettled,
            cash_buffer=buffer_amount,
            open_positions=open_positions,
            max_open_positions=self.settings.risk.max_open_positions,
        )

    def _previous_equity(self, before: dt.date) -> Decimal | None:
        with self.db.session() as session:
            return session.scalar(
                select(EquitySnapshot.equity)
                .where(
                    EquitySnapshot.snapshot_date < before,
                    EquitySnapshot.trading_mode == self.settings.trading_mode.value,
                )
                .order_by(EquitySnapshot.snapshot_date.desc())
                .limit(1)
            )

    def _day_pnl(self, equity: Decimal) -> tuple[Decimal | None, float | None]:
        today = self.clock.now().astimezone(UTC).date()
        previous = self._previous_equity(today)
        if previous is None or previous == 0:
            return None, None
        delta = to_money(equity - previous)
        return delta, float(delta / previous)
