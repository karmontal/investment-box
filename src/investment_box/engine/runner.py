"""Assembling and running the engine.

One place that builds every part and connects them, so a process has a single
entry point and the wiring is auditable in one file.

The safety ordering here is deliberate: the engine starts **paused** if
anything about its configuration is unsafe -- unverified universe, live mode
without confirmation, a non-cash account -- rather than starting and relying on
a later check to catch it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from investment_box.config.loader import load_universe_file
from investment_box.core.clock import UTC, TradingCalendar
from investment_box.core.logging import get_logger
from investment_box.core.types import AutonomyLevel, TradingMode
from investment_box.engine.kill_switch import KillResult, KillSwitch
from investment_box.engine.loop import CycleResult, TradingCycle
from investment_box.engine.scheduler import EngineScheduler
from investment_box.engine.state import EngineState, EngineStateMachine
from investment_box.execution.order_manager import OrderManager
from investment_box.forecast.calibration import CalibrationTracker
from investment_box.risk.manager import RiskManager
from investment_box.risk.settlement import SettlementLedger
from investment_box.services.container import ServiceContainer
from investment_box.services.research import ResearchService
from investment_box.shariah.providers.mock_external import MockExternalProvider
from investment_box.shariah.status import ComplianceTracker
from investment_box.strategies import STRATEGY_REGISTRY
from investment_box.telegram.bot import TelegramStack
from investment_box.universe.builder import Instrument, UniverseBuilder

log = get_logger(__name__)


@dataclass
class EngineRunner:
    """Everything the engine needs, assembled."""

    services: ServiceContainer
    state: EngineStateMachine
    risk: RiskManager
    orders: OrderManager
    research: ResearchService
    cycle: TradingCycle
    scheduler: EngineScheduler
    kill_switch: KillSwitch
    telegram: TelegramStack | None
    instruments: list[Instrument]
    #: Reasons the engine refused to start running. Surfaced everywhere.
    blockers: list[str]

    @property
    def can_run(self) -> bool:
        return not self.blockers

    async def run_once(self) -> CycleResult:
        """Run a single cycle, honouring the engine state."""
        if not self.state.state.can_open_positions:
            log.info("engine.cycle_skipped", state=self.state.state.value)
        result = await self.cycle.run(self.instruments)
        self.state.record_cycle(result.summary())
        for error in result.errors:
            self.state.record_error(error)
        return result

    async def start(self) -> None:
        """Start the scheduler and the Telegram stack."""
        if self.blockers:
            reason = "; ".join(self.blockers)
            self.state.pause(reason, actor="startup")
            log.warning("engine.start_blocked", reason=reason)
        else:
            self.state.start()

        if self.telegram is not None:
            await self.telegram.bot.start()
            self.telegram.broadcast.startup(
                [*self.services.startup_banner(), *(f"BLOCKED: {b}" for b in self.blockers)]
            )

        self.scheduler.build(
            on_cycle=self._scheduled_cycle,
            on_expire_approvals=self._expire_approvals,
            on_daily_summary=self._daily_summary,
            on_weekly_summary=self._weekly_summary,
        )
        self.scheduler.start()

    async def stop(self) -> None:
        self.scheduler.shutdown()
        if self.telegram is not None:
            await self.telegram.bot.stop()
        self.state.stop("shut down")

    def kill(self, reason: str, *, close_positions: bool = False) -> KillResult:
        result = self.kill_switch.activate(reason, close_positions=close_positions)
        if self.telegram is not None:
            self.telegram.broadcast.kill_switch(result.detail())
        return result

    # ------------------------------------------------------------- scheduled

    async def _scheduled_cycle(self) -> None:
        await self.run_once()

    async def _expire_approvals(self) -> None:
        if self.telegram is not None:
            await self.telegram.notifier.sweep_expired()

    async def _daily_summary(self) -> None:
        if self.telegram is None:
            return
        view = self.services.portfolio.get_account_view()
        today = self.services.clock.now().astimezone(UTC).date()
        trades_today = len(
            [t for t in self.services.portfolio.closed_trades(50) if t.exit_date == today]
        )
        self.telegram.broadcast.daily_summary(
            equity=view.equity,
            day_pnl=view.day_pnl,
            day_pnl_pct=view.day_pnl_pct,
            open_positions=len(view.positions),
            trades_today=trades_today,
            cash_settled=view.cash_settled,
            cash_unsettled=view.cash_unsettled,
            taken_at=view.taken_at,
        )

    async def _weekly_summary(self) -> None:
        if self.telegram is None:
            return
        view = self.services.portfolio.get_account_view()
        curve = self.services.portfolio.equity_curve(days=7)
        trades = self.services.portfolio.closed_trades(100)

        week_start = curve[0].equity if curve else view.equity
        week_pnl = view.equity - week_start
        week_pct = float(week_pnl / week_start) if week_start else 0.0

        closed = [t for t in trades if t.net_pnl is not None]
        wins = [t for t in closed if (t.net_pnl or 0) > 0]
        win_rate = len(wins) / len(closed) if closed else None

        equities = [float(row.equity) for row in curve] or [float(view.equity)]
        peak = max(equities)
        drawdown = min(e / peak - 1 for e in equities) if peak else None

        self.telegram.broadcast.weekly_summary(
            equity=view.equity,
            week_pnl=week_pnl,
            week_pnl_pct=week_pct,
            win_rate=win_rate,
            max_drawdown=drawdown,
            trades=len(closed),
        )


def build_engine(
    services: ServiceContainer,
    *,
    strategy_name: str = "etf_momentum_rotation",
    telegram: TelegramStack | None = None,
) -> EngineRunner:
    """Assemble the engine, refusing to start running if anything is unsafe."""
    settings = services.settings
    calendar = TradingCalendar()
    blockers: list[str] = []

    ledger = SettlementLedger(
        services.database,
        calendar,
        settings.settlement.settlement_days,
        clock=services.clock,
    )
    state = EngineStateMachine(services.audit, clock=services.clock)
    risk = RiskManager(
        settings, services.database, ledger, services.audit,
        clock=services.clock, calendar=calendar,
    )
    orders = OrderManager(
        services.broker, services.database, settings, services.audit, clock=services.clock
    )
    compliance = ComplianceTracker(
        MockExternalProvider(clock=services.clock),
        services.database,
        settings.shariah,
        services.audit,
        clock=services.clock,
    )
    research = ResearchService(
        settings,
        services.market_data,
        STRATEGY_REGISTRY[strategy_name](),
        compliance=compliance,
        calibration=CalibrationTracker(services.database),
        clock=services.clock,
        calendar=calendar,
    )

    instruments = UniverseBuilder.load_instruments(load_universe_file())

    # --- refuse to run when anything is unsafe --------------------------------
    unverified = [i.symbol for i in instruments if not i.verified]
    if len(unverified) == len(instruments):
        blockers.append(
            f"every symbol is unverified ({', '.join(unverified)}). Confirm listing "
            f"and Shariah certification, then set verified: true."
        )

    if settings.trading_mode is TradingMode.LIVE:
        blockers.append(
            "LIVE mode requires the Phase 7 pre-flight checklist, which is not "
            "implemented. Refusing to trade real money."
        )

    account = None
    try:
        account = services.broker.get_account()
    except Exception as exc:  # noqa: BLE001
        blockers.append(f"broker unreachable: {exc}")

    if account is not None and not account.is_cash_account:
        blockers.append(
            "the broker account has margin enabled; this application requires a CASH "
            "account and will not place any order until that changes. In Alpaca, paper "
            "accounts default to margin: reset the paper account and choose a cash "
            "account, or use an account configured without margin."
        )
    if account is not None and account.trading_blocked:
        blockers.append("the broker reports trading is blocked on this account")

    from investment_box.services.settings_service import SettingsService

    settings_service = SettingsService(services.database, settings, services.audit)
    if settings_service.kill_requested:
        blockers.append(
            f"a kill switch is active: {settings_service.kill_reason}. Clear it in the "
            f"dashboard before the engine will trade again."
        )

    if settings.engine.autonomy_level is AutonomyLevel.SUGGEST_ONLY and telegram is None:
        blockers.append(
            "autonomy level 1 requires an approval channel, but Telegram is not configured"
        )

    cycle = TradingCycle(
        settings,
        services.database,
        services.portfolio,
        risk,
        orders,
        research,
        services.audit,
        compliance=compliance,
        notifier=telegram.notifier if telegram else None,
        broadcast=telegram.broadcast if telegram else None,
        clock=services.clock,
        calendar=calendar,
    )

    return EngineRunner(
        services=services,
        state=state,
        risk=risk,
        orders=orders,
        research=research,
        cycle=cycle,
        scheduler=EngineScheduler(
            settings, services.audit, clock=services.clock, calendar=calendar
        ),
        kill_switch=KillSwitch(services.broker, orders, state, services.audit),
        telegram=telegram,
        instruments=instruments,
        blockers=blockers,
    )


async def run_forever(runner: EngineRunner) -> None:
    """Start the engine and keep the process alive until interrupted."""
    await runner.start()
    try:
        while runner.state.state is not EngineState.KILLED:
            await asyncio.sleep(60)
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("engine.interrupted")
    finally:
        await runner.stop()
