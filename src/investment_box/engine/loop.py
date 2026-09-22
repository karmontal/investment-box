"""The trading cycle.

One cycle, in a fixed order:

1. **Reconcile** — broker state wins over local beliefs. Always first, because
   acting on a stale belief about an open order is how a position gets doubled.
2. **Settle** — release matured proceeds into settled cash.
3. **Re-screen compliance** on everything held. A holding that turned
   non-compliant is exited regardless of profit, the regime, or the strategy.
4. **Manage exits** — synthetic stops, take-profits, minimum-hold checks.
5. **Generate candidates** through the same ResearchService the dashboard uses.
6. **Risk-check and size** each candidate.
7. **Route by autonomy level** — suggest, ask, or act.
8. **Snapshot equity** so drawdown and day-P&L have a baseline tomorrow.

Exits run before entries, always. Freeing capital and honouring a compliance
exit matter more than opening something new, and at $500 the cash from an exit
is often what makes the next entry possible at all.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import select

from investment_box.config.schema import Settings
from investment_box.core.audit import AuditSink
from investment_box.core.clock import UTC, Clock, SystemClock, TradingCalendar
from investment_box.core.logging import get_logger
from investment_box.core.types import (
    ApprovalKind,
    AutonomyLevel,
    ComplianceStatus,
    ExitReason,
    to_money,
)
from investment_box.db.models import Trade
from investment_box.db.session import Database
from investment_box.execution.base import AccountSnapshot, BrokerPosition
from investment_box.execution.order_manager import OrderManager, PlacedOrder
from investment_box.forecast.candidates import Candidate
from investment_box.risk.manager import RiskDecision, RiskManager, RiskVerdict
from investment_box.services.portfolio import PortfolioService
from investment_box.services.research import ResearchService
from investment_box.shariah.status import ComplianceTracker
from investment_box.universe.builder import Instrument

log = get_logger(__name__)


@dataclass
class CycleResult:
    """What one cycle did, and why it did not do more."""

    as_of: dt.date
    started_at: dt.datetime
    exits: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    approvals_requested: list[str] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    compliance_exits: list[str] = field(default_factory=list)

    @property
    def did_anything(self) -> bool:
        return bool(self.exits or self.entries or self.approvals_requested)

    def summary(self) -> str:
        parts = []
        if self.entries:
            parts.append(f"{len(self.entries)} entry")
        if self.exits:
            parts.append(f"{len(self.exits)} exit")
        if self.approvals_requested:
            parts.append(f"{len(self.approvals_requested)} awaiting approval")
        if self.errors:
            parts.append(f"{len(self.errors)} error")
        if not parts:
            return f"no action ({len(self.refusals)} candidate(s) refused)"
        return ", ".join(parts)


class TradingCycle:
    """Runs one cycle. Stateless between runs; all state lives in the services."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        portfolio: PortfolioService,
        risk: RiskManager,
        orders: OrderManager,
        research: ResearchService,
        audit: AuditSink,
        *,
        compliance: ComplianceTracker | None = None,
        notifier: object | None = None,
        broadcast: object | None = None,
        clock: Clock | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.settings = settings
        self.db = database
        self.portfolio = portfolio
        self.risk = risk
        self.orders = orders
        self.research = research
        self.audit = audit
        self.compliance = compliance
        self.notifier = notifier
        self.broadcast = broadcast
        self.clock = clock or SystemClock()
        self.calendar = calendar or TradingCalendar()

    async def run(self, instruments: list[Instrument]) -> CycleResult:
        """Execute one full cycle."""
        now = self.clock.now()
        result = CycleResult(as_of=now.astimezone(UTC).date(), started_at=now)

        for step, action in (
            ("reconcile", self._reconcile),
            ("settle", self._settle),
        ):
            try:
                action(result)
            except Exception as exc:  # noqa: BLE001 - one step must not kill the cycle
                log.error("cycle.step_failed", step=step, error=str(exc))
                result.errors.append(f"{step}: {exc}")

        try:
            await self._compliance_exits(result)
        except Exception as exc:  # noqa: BLE001
            log.error("cycle.compliance_exits_failed", error=str(exc))
            result.errors.append(f"compliance exits: {exc}")

        try:
            await self._manage_exits(result)
        except Exception as exc:  # noqa: BLE001
            log.error("cycle.exits_failed", error=str(exc))
            result.errors.append(f"exits: {exc}")

        try:
            await self._consider_entries(instruments, result)
        except Exception as exc:  # noqa: BLE001
            log.error("cycle.entries_failed", error=str(exc))
            result.errors.append(f"entries: {exc}")

        try:
            self.portfolio.record_equity_snapshot()
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"equity snapshot: {exc}")

        log.info("cycle.complete", summary=result.summary())
        return result

    # ----------------------------------------------------------------- steps

    def _reconcile(self, result: CycleResult) -> None:
        updated = self.orders.reconcile()
        cancelled = self.orders.cancel_stale_orders()
        if cancelled:
            result.refusals.append(f"cancelled {cancelled} stale order(s)")
        log.debug("cycle.reconciled", updated=len(updated), cancelled=cancelled)

    def _settle(self, result: CycleResult) -> None:
        released = self.risk.ledger.settle_due()
        if released:
            log.info("cycle.settled", amount=str(released))

    async def _compliance_exits(self, result: CycleResult) -> None:
        """Exit anything that is no longer permissible.

        Runs before ordinary exit management and before entries: a holding that
        turned non-compliant must leave regardless of profit or strategy.
        """
        if self.compliance is None:
            return

        held = [p.symbol for p in self.portfolio.broker.get_positions()]
        if not held:
            return

        for record in self.compliance.newly_non_compliant(list(held)):
            if record.status is not ComplianceStatus.NON_COMPLIANT:
                continue  # a stale screen is a warning, not an exit trigger

            position = self.portfolio.broker.get_position(record.symbol)
            if position is None:
                continue

            placed = self.orders.close_position(
                symbol=record.symbol,
                quantity=position.quantity,
                compliance_status=record.status,
                reason=ExitReason.COMPLIANCE_EXIT.value,
            )
            message = f"{record.symbol}: {placed.reason}"
            if placed.accepted:
                result.compliance_exits.append(message)
                self._notify_exit(record.symbol, position, ExitReason.COMPLIANCE_EXIT)
            else:
                result.errors.append(f"compliance exit failed for {message}")

            if self.broadcast is not None:
                self.broadcast.compliance_changed(  # type: ignore[attr-defined]
                    record.symbol, "compliant", "non_compliant",
                    "Exiting per the configured policy.",
                )

    async def _manage_exits(self, result: CycleResult) -> None:
        """Check stops, targets and the minimum holding period."""
        open_trades = {t.symbol: t for t in self.portfolio.open_trades()}

        for position in self.portfolio.broker.get_positions():
            symbol = str(position.symbol)
            trade = open_trades.get(symbol)
            if trade is None:
                continue

            allowed, hold_reason = self.risk.can_exit(symbol, trade.entry_date)
            exit_reason = self._exit_trigger(position, trade)
            if exit_reason is None:
                continue

            if not allowed and exit_reason is not ExitReason.COMPLIANCE_EXIT:
                result.refusals.append(f"{symbol}: {exit_reason.value} blocked — {hold_reason}")
                continue

            placed = self.orders.close_position(
                symbol=symbol,
                quantity=position.quantity,
                compliance_status=ComplianceStatus(trade.compliance_status_at_entry),
                reason=exit_reason.value,
            )
            if placed.accepted:
                result.exits.append(f"{symbol}: {exit_reason.value}")
                self._close_trade(trade, position, exit_reason, placed)
                self._notify_exit(symbol, position, exit_reason)
            else:
                result.errors.append(f"exit failed for {symbol}: {placed.reason}")

    def _exit_trigger(self, position: BrokerPosition, trade: Trade) -> ExitReason | None:
        """Which exit condition fired, if any.

        Stop is checked before target. On a bar where both were touched there
        is no way to know which came first, and assuming the favourable one is
        how a system flatters itself.
        """
        price = position.current_price
        if trade.stop_loss_price is not None and price <= trade.stop_loss_price:
            return ExitReason.STOP_LOSS
        if trade.take_profit_price is not None and price >= trade.take_profit_price:
            return ExitReason.TAKE_PROFIT

        today = self.clock.now().astimezone(UTC).date()
        held = self.calendar.trading_days_between(trade.entry_date, today)
        if held >= self.settings.holding.typical_max_holding_days:
            return ExitReason.MAX_HOLD
        return None

    async def _consider_entries(
        self, instruments: list[Instrument], result: CycleResult
    ) -> None:
        """Generate candidates, size them, and route by autonomy level."""
        account = self.portfolio.broker.get_account()
        positions = self.portfolio.broker.get_positions()
        state = self.risk.state(account, positions)

        if not state.can_open_new:
            result.refusals.append(f"no new entries: {state.summary()}")
            return

        holdings = tuple(str(p.symbol) for p in positions)
        snapshot = self.research.build(instruments, current_holdings=holdings)

        for candidate in snapshot.candidates:
            if not candidate.is_tradable:
                result.refusals.append(f"{candidate.symbol}: {candidate.reason}")
                continue
            await self._consider_one(candidate, account, positions, result)

    async def _consider_one(
        self,
        candidate: Candidate,
        account: AccountSnapshot,
        positions: list[BrokerPosition],
        result: CycleResult,
    ) -> None:
        price = self.portfolio.broker.get_last_price(candidate.symbol)
        if price is None:
            result.refusals.append(f"{candidate.symbol}: no current price")
            return

        atr = None
        entry = candidate.universe_entry
        if entry is not None and entry.last_price:
            forecast = candidate.forecast
            if forecast is not None and forecast.volatility:
                # ATR is approximated from realised volatility when the feature
                # frame is not to hand; the sizer treats it identically.
                atr = to_money(Decimal(str(forecast.volatility / 16)) * price)

        decision = self.risk.evaluate(
            symbol=candidate.symbol,
            entry_price=price,
            account=account,
            positions=positions,
            atr=atr,
        )
        if not decision.is_approved or decision.size is None:
            result.refusals.append(f"{candidate.symbol}: {decision.reason}")
            if decision.verdict is RiskVerdict.PAUSED and self.broadcast is not None:
                self.broadcast.risk_limit_hit(decision.reason)  # type: ignore[attr-defined]
            return

        autonomy = self.settings.engine.autonomy_level
        whitelisted = candidate.symbol in set(self.settings.universe.whitelist)

        if autonomy is AutonomyLevel.FULLY_AUTONOMOUS or (
            autonomy is AutonomyLevel.AUTO_WITHIN_WHITELIST and whitelisted
        ):
            await self._place(candidate, decision, price, result)
        else:
            await self._request_approval(candidate, decision, price, result)

    async def _place(
        self,
        candidate: Candidate,
        decision: RiskDecision,
        price: Decimal,
        result: CycleResult,
    ) -> None:
        assert decision.size is not None  # guaranteed by is_approved
        compliance = candidate.compliance
        placed = self.orders.open_position(
            size=decision.size,
            compliance_status=(
                compliance.display_status if compliance else ComplianceStatus.UNKNOWN
            ),
            compliance_source=compliance.source if compliance else "unscreened",
            reference_price=price,
            available_cash=self._settled_cash(),
            instrument_name=(
                candidate.universe_entry.instrument.name
                if candidate.universe_entry
                else None
            ),
        )

        if not placed.accepted:
            result.refusals.append(f"{candidate.symbol}: {placed.reason}")
            return

        result.entries.append(f"{candidate.symbol}: {decision.size.quantity} @ ~${price}")
        self._open_trade(candidate, decision, price, placed)
        self._notify_entry(candidate, decision, price, placed)

    def _settled_cash(self) -> Decimal:
        """Settled cash available right now.

        Re-read rather than reused from the risk decision: an exit earlier in
        this same cycle may have changed it, and the compliance gate compares
        the order against what is actually spendable.
        """
        state = self.risk.state(
            self.portfolio.broker.get_account(), self.portfolio.broker.get_positions()
        )
        return state.settlement.available if state.settlement else Decimal("0")

    async def _request_approval(
        self,
        candidate: Candidate,
        decision: RiskDecision,
        price: Decimal,
        result: CycleResult,
    ) -> None:
        assert decision.size is not None
        if self.notifier is None:
            result.refusals.append(
                f"{candidate.symbol}: needs approval but no approval channel is configured"
            )
            return

        forecast = candidate.forecast
        payload = {
            "symbol": candidate.symbol,
            "side": "buy",
            "quantity": str(decision.size.quantity),
            "entry_price": str(price),
            "size_pct": float(decision.size.notional / self.settings.capital.allocation_usd),
            "stop_loss": str(decision.size.stop_price) if decision.size.stop_price else None,
            "take_profit": (
                str(decision.size.take_profit_price)
                if decision.size.take_profit_price
                else None
            ),
            "strategy": self.research.strategy.name,
            "probability": forecast.direction_probability if forecast else None,
            "compliance_status": (
                candidate.compliance.display_status.value if candidate.compliance else "unknown"
            ),
            "reason": candidate.reason,
            "backtest_summary": forecast.track_record.summary() if forecast else None,
        }

        await self.notifier.request(  # type: ignore[attr-defined]
            request_key=f"entry-{candidate.symbol}-{result.as_of}",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=payload,
        )
        result.approvals_requested.append(candidate.symbol)

    # --------------------------------------------------------- record-keeping

    def _open_trade(
        self,
        candidate: Candidate,
        decision: RiskDecision,
        price: Decimal,
        placed: PlacedOrder,
    ) -> None:
        assert decision.size is not None
        compliance = candidate.compliance
        now = self.clock.now()
        with self.db.session() as session:
            session.add(
                Trade(
                    symbol=candidate.symbol,
                    strategy=self.research.strategy.name,
                    quantity=decision.size.quantity,
                    entry_price=price,
                    entry_at=now,
                    entry_date=now.astimezone(UTC).date(),
                    stop_loss_price=decision.size.stop_price,
                    take_profit_price=decision.size.take_profit_price,
                    compliance_status_at_entry=(
                        compliance.display_status.value if compliance else "unknown"
                    ),
                    compliance_source_at_entry=(
                        compliance.source if compliance else "unscreened"
                    ),
                    compliance_screened_at=compliance.screened_at if compliance else None,
                    reason=candidate.reason,
                    trading_mode=self.settings.trading_mode.value,
                )
            )

    def _close_trade(
        self,
        trade: Trade,
        position: BrokerPosition,
        reason: ExitReason,
        placed: PlacedOrder,
    ) -> None:
        now = self.clock.now()
        exit_price = position.current_price
        gross = to_money((exit_price - trade.entry_price) * trade.quantity)
        proceeds = to_money(exit_price * trade.quantity)

        with self.db.session() as session:
            row = session.scalar(select(Trade).where(Trade.id == trade.id))
            if row is None:
                return
            row.exit_price = exit_price
            row.exit_at = now
            row.exit_date = now.astimezone(UTC).date()
            row.exit_reason = reason.value
            row.gross_pnl = gross
            row.net_pnl = gross - row.costs_paid
            row.holding_trading_days = self.calendar.trading_days_between(
                row.entry_date, now.astimezone(UTC).date()
            )

        self.risk.ledger.record_sale(trade.symbol, proceeds, trade_id=trade.id)

    # ---------------------------------------------------------- notifications

    def _notify_entry(
        self,
        candidate: Candidate,
        decision: RiskDecision,
        price: Decimal,
        placed: PlacedOrder,
    ) -> None:
        assert decision.size is not None
        if self.broadcast is None:
            return
        forecast = candidate.forecast
        self.broadcast.trade_opened(  # type: ignore[attr-defined]
            symbol=candidate.symbol,
            side="buy",
            quantity=decision.size.quantity,
            entry_price=price,
            size_pct=float(decision.size.notional / self.settings.capital.allocation_usd),
            stop_loss=decision.size.stop_price,
            take_profit=decision.size.take_profit_price,
            strategy=self.research.strategy.name,
            probability=forecast.direction_probability if forecast else None,
            compliance_status=(
                candidate.compliance.display_status.value if candidate.compliance else "unknown"
            ),
            reason=candidate.reason
            + (" [engine-managed stop]" if placed.synthetic_stop else ""),
        )

    def _notify_exit(
        self, symbol: str, position: BrokerPosition, reason: ExitReason
    ) -> None:
        if self.broadcast is None:
            return
        pnl = position.unrealized_pnl
        self.broadcast.trade_closed(  # type: ignore[attr-defined]
            symbol=symbol,
            exit_price=position.current_price,
            holding_days=0,
            pnl=pnl,
            pnl_pct=position.unrealized_pnl_pct,
            exit_reason=reason.value,
        )
