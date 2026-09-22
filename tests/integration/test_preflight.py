"""The live-trading pre-flight checklist and guard.

The property this file exists to prove: **there is no way to reach live
trading without producing the evidence.** Not a flag, not a config key, not a
plausible-looking argument to a constructor.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from investment_box.config.loader import load_settings
from investment_box.config.schema import Secrets
from investment_box.core.clock import FrozenClock
from investment_box.core.types import CashLedger, TradingMode
from investment_box.db.models import EquitySnapshot, Trade
from investment_box.engine.live_guard import (
    LIVE_CONFIRMATION_PHRASE,
    LiveActivation,
    LiveGuard,
    LiveTradingRefusedError,
)
from investment_box.engine.preflight import (
    MIN_PAPER_TRADES,
    MIN_PAPER_WEEKS,
    CheckStatus,
    PreflightChecklist,
)
from investment_box.execution.base import AccountSnapshot

TODAY = dt.date(2024, 6, 12)


def live_secrets(**overrides) -> Secrets:
    base = {
        "_env_file": None,
        "alpaca_api_key": "PKLIVE",
        "alpaca_secret_key": "SECRET",
        "alpaca_base_url": "https://api.alpaca.markets",
        "telegram_bot_token": "1234567890:AAFfakeTokenForTestsOnly_NotReal12345",
        "telegram_channel_id": "-1001234567890",
        "telegram_allowed_user_ids": "555000111",
    }
    base.update(overrides)
    return Secrets(**base)  # type: ignore[arg-type]


class _Account:
    def __init__(self, cash: bool = True, blocked: bool = False) -> None:
        self.is_cash = cash
        self.blocked = blocked

    def get_account(self) -> AccountSnapshot:
        return AccountSnapshot(
            equity=Decimal("500"),
            cash=CashLedger(settled=Decimal("500"), unsettled=Decimal("0")),
            positions_value=Decimal("0"),
            buying_power=Decimal("500"),
            is_cash_account=self.is_cash,
            trading_blocked=self.blocked,
        )

    def get_positions(self) -> list:
        return []


def seed_paper_history(database, weeks: int, trades: int) -> None:
    """Write enough paper history to satisfy the duration and count checks."""
    start = TODAY - dt.timedelta(weeks=weeks)
    with database.session() as session:
        session.add(
            EquitySnapshot(
                snapshot_date=start, equity=Decimal("500"),
                cash_settled=Decimal("500"), cash_unsettled=Decimal("0"),
                positions_value=Decimal("0"), trading_mode="paper",
            )
        )
        session.add(
            EquitySnapshot(
                snapshot_date=TODAY, equity=Decimal("520"),
                cash_settled=Decimal("520"), cash_unsettled=Decimal("0"),
                positions_value=Decimal("0"), trading_mode="paper",
            )
        )
        for _ in range(trades):
            session.add(
                Trade(
                    symbol="SPUS", strategy="test", quantity=Decimal("1"),
                    entry_price=Decimal("60"),
                    entry_at=dt.datetime(2024, 5, 1, tzinfo=dt.UTC),
                    entry_date=dt.date(2024, 5, 1),
                    exit_price=Decimal("61"),
                    exit_at=dt.datetime(2024, 5, 3, tzinfo=dt.UTC),
                    exit_date=dt.date(2024, 5, 3),
                    net_pnl=Decimal("1"),
                    compliance_status_at_entry="compliant",
                    compliance_source_at_entry="test",
                    trading_mode="paper",
                )
            )


@pytest.fixture
def checklist(database, clock: FrozenClock) -> PreflightChecklist:
    return PreflightChecklist(load_settings(), live_secrets(), database, clock=clock)


class TestChecklistRefusesByDefault:
    def test_fresh_install_fails(self, checklist: PreflightChecklist) -> None:
        """Nothing about a fresh install should permit live trading."""
        report = checklist.evaluate()
        assert not report.passed
        assert report.failures

    def test_no_paper_history_fails(self, checklist: PreflightChecklist) -> None:
        report = checklist.evaluate()
        duration = next(c for c in report.checks if "weeks of paper" in c.name)
        assert duration.status is CheckStatus.FAIL
        assert "no paper-trading history" in duration.detail

    def test_insufficient_weeks_fails(
        self, checklist: PreflightChecklist, database
    ) -> None:
        seed_paper_history(database, weeks=2, trades=50)
        report = checklist.evaluate()
        duration = next(c for c in report.checks if "weeks of paper" in c.name)
        assert duration.status is CheckStatus.FAIL
        assert "2.0 weeks" in duration.detail

    def test_insufficient_trades_fails(
        self, checklist: PreflightChecklist, database
    ) -> None:
        seed_paper_history(database, weeks=MIN_PAPER_WEEKS + 1, trades=3)
        report = checklist.evaluate()
        count = next(c for c in report.checks if "closed paper trades" in c.name)
        assert count.status is CheckStatus.FAIL

    def test_enough_history_passes_those_two(
        self, checklist: PreflightChecklist, database
    ) -> None:
        seed_paper_history(database, weeks=MIN_PAPER_WEEKS + 1, trades=MIN_PAPER_TRADES)
        report = checklist.evaluate()
        assert next(c for c in report.checks if "weeks of paper" in c.name).passed
        assert next(c for c in report.checks if "closed paper trades" in c.name).passed


class TestUnknownIsFailure:
    """Missing evidence is not passing evidence."""

    def test_unknown_data_provider_fails(self, checklist: PreflightChecklist) -> None:
        check = next(
            c for c in checklist.evaluate().checks if c.name == "Market data is real"
        )
        assert check.status is CheckStatus.UNKNOWN
        assert not check.passed

    def test_unknown_broker_fails(self, checklist: PreflightChecklist) -> None:
        check = next(
            c for c in checklist.evaluate().checks if "cash account" in c.name
        )
        assert not check.passed

    def test_missing_backtest_comparison_fails(
        self, checklist: PreflightChecklist
    ) -> None:
        check = next(c for c in checklist.evaluate().checks if "backtest" in c.name)
        assert check.status is CheckStatus.UNKNOWN
        assert not check.passed


class TestIndividualChecks:
    def test_synthetic_data_is_refused(self, checklist: PreflightChecklist) -> None:
        check = next(
            c
            for c in checklist.evaluate(data_provider_name="synthetic").checks
            if c.name == "Market data is real"
        )
        assert check.status is CheckStatus.FAIL
        assert check.critical

    @pytest.mark.parametrize("provider", ["mock_external", "internal_aaoifi"])
    def test_uncertified_screening_is_refused(
        self, checklist: PreflightChecklist, provider: str
    ) -> None:
        """A mock is not a ruling, and the internal screener is an estimate."""
        check = next(
            c
            for c in checklist.evaluate(compliance_provider_name=provider).checks
            if "certified source" in c.name
        )
        assert check.status is CheckStatus.FAIL
        assert check.critical

    def test_unverified_universe_is_refused(
        self, checklist: PreflightChecklist
    ) -> None:
        check = next(
            c for c in checklist.evaluate().checks if "symbol is verified" in c.name
        )
        assert check.status is CheckStatus.FAIL
        assert check.critical

    def test_margin_account_is_refused(self, checklist: PreflightChecklist) -> None:
        check = next(
            c
            for c in checklist.evaluate(broker=_Account(cash=False)).checks
            if "cash account" in c.name
        )
        assert check.status is CheckStatus.FAIL

    def test_blocked_account_is_refused(self, checklist: PreflightChecklist) -> None:
        check = next(
            c
            for c in checklist.evaluate(broker=_Account(blocked=True)).checks
            if "cash account" in c.name
        )
        assert check.status is CheckStatus.FAIL

    def test_active_kill_flag_is_refused(
        self, checklist: PreflightChecklist, database, settings, audit
    ) -> None:
        from investment_box.services.settings_service import SettingsService

        SettingsService(database, settings, audit).request_kill("testing")
        check = next(c for c in checklist.evaluate().checks if "kill switch" in c.name)
        assert check.status is CheckStatus.FAIL

    def test_missing_alerting_is_refused(self, database, clock: FrozenClock) -> None:
        """Real money without alerts means finding out when you next look."""
        bare = PreflightChecklist(
            load_settings(), Secrets(_env_file=None), database, clock=clock  # type: ignore[call-arg]
        )
        check = next(c for c in bare.evaluate().checks if "Alerting" in c.name)
        assert check.status is CheckStatus.FAIL

    def test_paper_endpoint_is_refused_for_live(
        self, database, clock: FrozenClock
    ) -> None:
        paper = PreflightChecklist(
            load_settings(),
            live_secrets(alpaca_base_url="https://paper-api.alpaca.markets"),
            database,
            clock=clock,
        )
        check = next(c for c in paper.evaluate().checks if "endpoint" in c.name)
        assert check.status is CheckStatus.FAIL

    def test_reckless_risk_limits_are_refused(
        self, database, clock: FrozenClock
    ) -> None:
        reckless = load_settings(
            overrides={"risk": {"risk_per_trade_pct": 0.20, "max_position_pct": 0.60}}
        )
        check = next(
            c
            for c in PreflightChecklist(
                reckless, live_secrets(), database, clock=clock
            ).evaluate().checks
            if "Risk limits" in c.name
        )
        assert check.status is CheckStatus.FAIL

    def test_large_backtest_divergence_is_refused(
        self, checklist: PreflightChecklist, database
    ) -> None:
        """A large gap means one of the two has a bug."""
        seed_paper_history(database, weeks=MIN_PAPER_WEEKS + 1, trades=MIN_PAPER_TRADES)
        check = next(
            c
            for c in checklist.evaluate(backtest_return=2.00).checks
            if "backtest" in c.name
        )
        assert check.status is CheckStatus.FAIL


class TestActivation:
    @pytest.fixture
    def activation(self, database, audit, clock: FrozenClock) -> LiveActivation:
        return LiveActivation(
            load_settings(), live_secrets(), database, audit, clock=clock
        )

    def test_wrong_phrase_is_refused(self, activation: LiveActivation) -> None:
        result = activation.attempt(typed_phrase="yes", actor="test")
        assert not result.activated
        assert "did not match" in result.reason

    def test_correct_phrase_alone_is_not_enough(
        self, activation: LiveActivation
    ) -> None:
        """The phrase proves intent, not readiness."""
        result = activation.attempt(
            typed_phrase=LIVE_CONFIRMATION_PHRASE, actor="test"
        )
        assert not result.activated
        assert "pre-flight check" in result.reason
        assert result.blockers

    def test_there_is_no_override_parameter(self) -> None:
        """A checklist with a bypass is a suggestion."""
        import inspect

        signature = inspect.signature(LiveActivation.attempt)
        names = set(signature.parameters)
        for forbidden in ("force", "override", "skip_checks", "bypass", "ignore_failures"):
            assert forbidden not in names

    def test_refused_attempts_are_audited(
        self, activation: LiveActivation, audit
    ) -> None:
        """Either a mistake worth knowing about, or someone testing the gate."""
        activation.attempt(typed_phrase="nope", actor="someone")
        entries = audit.recent(event_type="live.refused")
        assert entries
        assert "someone" in entries[0].summary

    def test_report_lists_what_is_blocking(self, activation: LiveActivation) -> None:
        result = activation.attempt(typed_phrase=LIVE_CONFIRMATION_PHRASE, actor="t")
        assert result.report is not None
        assert "no override" in result.report.to_text().lower()


class TestLiveGuard:
    def _guard(self, database, audit, clock, mode: TradingMode) -> LiveGuard:
        return LiveGuard(
            load_settings(overrides={"trading_mode": mode.value}),
            live_secrets(),
            database,
            audit,
            clock=clock,
        )

    def test_paper_mode_is_not_guarded(self, database, audit, clock) -> None:
        """Paper orders risk nothing; a per-order network check would get cached."""
        guard = self._guard(database, audit, clock, TradingMode.PAPER)
        guard.assert_order_permitted()  # must not raise

    def test_live_mode_refuses_when_a_critical_check_fails(
        self, database, audit, clock
    ) -> None:
        guard = self._guard(database, audit, clock, TradingMode.LIVE)
        with pytest.raises(LiveTradingRefusedError):
            guard.assert_order_permitted(broker=_Account())

    def test_refusal_is_audited(self, database, audit, clock) -> None:
        guard = self._guard(database, audit, clock, TradingMode.LIVE)
        with pytest.raises(LiveTradingRefusedError):
            guard.assert_order_permitted(broker=_Account())
        assert audit.recent(event_type="live.order_refused")

    def test_guard_checks_only_the_critical_subset(
        self, database, audit, clock
    ) -> None:
        """A cached safety check is not a safety check, so the per-order set is
        kept small enough to run every time."""
        guard = self._guard(database, audit, clock, TradingMode.LIVE)
        report = guard.status(broker=_Account())
        assert report.checks
        assert all(c.critical for c in report.checks)
        assert len(report.checks) < 12


class TestOrderManagerHonoursTheGuard:
    def test_live_guard_refusal_rejects_the_order(
        self, broker, database, audit, clock
    ) -> None:
        from investment_box.core.types import ComplianceStatus
        from investment_box.execution.order_manager import OrderManager
        from investment_box.risk.sizing import PositionSize

        guard = LiveGuard(
            load_settings(overrides={"trading_mode": "live"}),
            live_secrets(),
            database,
            audit,
            clock=clock,
        )
        manager = OrderManager(
            broker, database, load_settings(), audit, clock=clock, live_guard=guard
        )
        size = PositionSize(
            symbol="SPUS", quantity=Decimal("1"), entry_price=Decimal("45"),
            stop_price=Decimal("40"), take_profit_price=None,
            notional=Decimal("45"), risk_amount=Decimal("5"),
            is_fractional=False, reason="test",
        )
        placed = manager.open_position(
            size=size, compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45"),
            available_cash=Decimal("450"),
        )
        assert not placed.accepted
        assert "precondition" in placed.reason
