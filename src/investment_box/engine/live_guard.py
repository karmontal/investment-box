"""Guarding the transition to live trading, and every order after it.

Two layers:

* :class:`LiveActivation` -- the one-time transition. Requires the full
  checklist to pass *and* a typed phrase. Records who, when, and the evidence
  that was true at the time.
* :class:`LiveGuard` -- the continuous one. Re-runs the critical checks before
  every live order, because passing in January does not mean passing in March.
  An account that switched to margin, a screen that went stale, or a kill flag
  raised elsewhere must all stop the next order, not merely the next restart.

The per-order check is deliberately not cached. A cached safety check is not a
safety check.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from investment_box.config.schema import Secrets, Settings
from investment_box.core.audit import AuditSink
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.errors import InvestmentBoxError
from investment_box.core.logging import get_logger
from investment_box.core.types import TradingMode
from investment_box.db.session import Database
from investment_box.engine.preflight import PreflightChecklist, PreflightReport

log = get_logger(__name__)

#: The exact phrase required. Long and specific so it cannot be typed by
#: reflex, and unambiguous in an audit log.
LIVE_CONFIRMATION_PHRASE = "ENABLE LIVE TRADING"


class LiveTradingRefusedError(InvestmentBoxError):
    """Live trading was requested and refused. Never caught and retried."""


@dataclass
class ActivationResult:
    """The outcome of trying to go live."""

    activated: bool
    reason: str
    report: PreflightReport | None = None
    activated_at: dt.datetime | None = None

    @property
    def blockers(self) -> list[str]:
        if self.report is None:
            return []
        return [f"{c.name}: {c.detail}" for c in self.report.failures]


@dataclass
class LiveActivation:
    """The one-time transition to live trading."""

    settings: Settings
    secrets: Secrets
    database: Database
    audit: AuditSink
    clock: Clock = field(default_factory=SystemClock)

    def checklist(self) -> PreflightChecklist:
        return PreflightChecklist(
            self.settings, self.secrets, self.database, clock=self.clock
        )

    def attempt(
        self,
        *,
        typed_phrase: str,
        actor: str,
        broker: object | None = None,
        data_provider_name: str | None = None,
        compliance_provider_name: str | None = None,
        held_symbols: list[str] | None = None,
        backtest_return: float | None = None,
    ) -> ActivationResult:
        """Try to enable live trading.

        The attempt is audited whether it succeeds or fails. A refused attempt
        is worth recording: it is either a mistake worth knowing about, or
        someone testing whether the gate holds.
        """
        report = self.checklist().evaluate(
            broker=broker,
            data_provider_name=data_provider_name,
            compliance_provider_name=compliance_provider_name,
            held_symbols=held_symbols,
            backtest_return=backtest_return,
        )

        if typed_phrase.strip() != LIVE_CONFIRMATION_PHRASE:
            self._record_refusal(actor, "confirmation phrase did not match", report)
            return ActivationResult(
                activated=False,
                reason=(
                    f"The confirmation phrase did not match. Type exactly: "
                    f"{LIVE_CONFIRMATION_PHRASE}"
                ),
                report=report,
            )

        if not report.passed:
            self._record_refusal(
                actor, f"{len(report.failures)} pre-flight check(s) failed", report
            )
            return ActivationResult(
                activated=False,
                reason=(
                    f"{len(report.failures)} pre-flight check(s) failed. There is no "
                    f"override."
                ),
                report=report,
            )

        now = self.clock.now()
        self.audit.record(
            "live.activated",
            f"LIVE TRADING ENABLED by {actor}. All {len(report.checks)} pre-flight "
            f"checks passed.",
            actor=actor,
            detail={
                "checks": [
                    {"name": c.name, "status": c.status.value, "detail": c.detail}
                    for c in report.checks
                ]
            },
        )
        log.error("live.activated", actor=actor)  # error level: this must stand out
        return ActivationResult(
            activated=True,
            reason="All pre-flight checks passed. Live trading is enabled.",
            report=report,
            activated_at=now,
        )

    def _record_refusal(
        self, actor: str, reason: str, report: PreflightReport
    ) -> None:
        self.audit.record(
            "live.refused",
            f"Live trading refused for {actor}: {reason}",
            actor=actor,
            detail={"failures": [c.name for c in report.failures]},
        )
        log.warning("live.refused", actor=actor, reason=reason)


@dataclass
class LiveGuard:
    """Re-checks the critical conditions before every live order.

    Not used in paper mode -- paper orders risk nothing, and running a
    network-touching checklist per paper order would make people cache it.
    """

    settings: Settings
    secrets: Secrets
    database: Database
    audit: AuditSink
    clock: Clock = field(default_factory=SystemClock)

    @property
    def is_live(self) -> bool:
        return self.settings.trading_mode is TradingMode.LIVE

    def assert_order_permitted(
        self,
        *,
        broker: object | None = None,
        data_provider_name: str | None = None,
        compliance_provider_name: str | None = None,
        held_symbols: list[str] | None = None,
    ) -> None:
        """Refuse a live order if any critical condition has changed.

        Raises:
            LiveTradingRefusedError: If a critical check now fails. Not recoverable:
                it means a precondition that was true at activation no longer
                is, and the order must not proceed.
        """
        if not self.is_live:
            return

        report = PreflightChecklist(
            self.settings, self.secrets, self.database, clock=self.clock
        ).critical_only(
            broker=broker,
            data_provider_name=data_provider_name,
            compliance_provider_name=compliance_provider_name,
            held_symbols=held_symbols,
        )

        if report.passed:
            return

        detail = "; ".join(f"{c.name}: {c.detail}" for c in report.failures)
        self.audit.record(
            "live.order_refused",
            f"Live order refused: {detail}",
            actor="live_guard",
        )
        log.error("live.order_refused", failures=[c.name for c in report.failures])
        raise LiveTradingRefusedError(
            f"A live-trading precondition no longer holds, so this order is refused: "
            f"{detail}"
        )

    def status(
        self,
        *,
        broker: object | None = None,
        data_provider_name: str | None = None,
        compliance_provider_name: str | None = None,
        held_symbols: list[str] | None = None,
    ) -> PreflightReport:
        """The current critical-check state, for display."""
        return PreflightChecklist(
            self.settings, self.secrets, self.database, clock=self.clock
        ).critical_only(
            broker=broker,
            data_provider_name=data_provider_name,
            compliance_provider_name=compliance_provider_name,
            held_symbols=held_symbols,
        )
