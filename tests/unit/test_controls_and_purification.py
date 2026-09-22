"""User settings, Telegram controls, purification and zakat.

The properties under test: hard constraints cannot be reached through any
setting, destructive commands need two taps, and purification never invents a
number it does not have.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from investment_box.core.types import AutonomyLevel, TradingMode, UniverseMode
from investment_box.i18n.translator import Translator
from investment_box.services.settings_service import SettingsService, SymbolRule
from investment_box.shariah.purification import (
    PurificationMethod,
    PurificationTracker,
)
from investment_box.shariah.zakat import ZakatHolding, ZakatMethod, estimate_zakat
from investment_box.telegram.auth import AuthGuard
from investment_box.telegram.controls import (
    CONFIRMATION_TIMEOUT_SECONDS,
    ControlCommands,
)

ALLOWED_USER = 555000111


@pytest.fixture
def settings_service(database, settings, audit) -> SettingsService:
    return SettingsService(database, settings, audit)


@pytest.fixture
def purification(database, audit, clock) -> PurificationTracker:
    return PurificationTracker(database, audit, clock=clock)


@pytest.fixture
def controls(audit, clock) -> ControlCommands:
    return ControlCommands(
        guard=AuthGuard(allowed_user_ids=frozenset({ALLOWED_USER}), audit=audit),
        audit=audit,
        trading_mode=TradingMode.PAPER,
        translator=Translator("en"),
        clock=clock,
    )


class TestSettingsService:
    def test_set_and_get(self, settings_service: SettingsService) -> None:
        settings_service.set("max_trades_per_day", 3)
        assert settings_service.get("max_trades_per_day") == 3

    def test_unknown_key_is_refused(self, settings_service: SettingsService) -> None:
        """A typo must not silently create a setting nothing reads."""
        with pytest.raises(ValueError, match="unknown setting"):
            settings_service.set("max_trades_per_dya", 3)

    @pytest.mark.parametrize(
        "key",
        [
            "margin_allowed",
            "short_selling_allowed",
            "derivatives_allowed",
            "leveraged_or_inverse_allowed",
            "crypto_allowed",
            "block_unsettled_usage",
            "trading_mode",
        ],
    )
    def test_hard_constraints_cannot_be_set(
        self, settings_service: SettingsService, key: str
    ) -> None:
        """No setting may look like it relaxes a hard rule."""
        with pytest.raises(ValueError):
            settings_service.set(key, True)

    def test_changes_are_audited(self, settings_service: SettingsService, audit) -> None:
        settings_service.set("max_trades_per_day", 3, actor="dashboard")
        entries = audit.recent(event_type="settings.changed")
        assert entries
        assert entries[0].actor == "dashboard"

    def test_unruled_symbol_defaults_to_needs_approval(
        self, settings_service: SettingsService
    ) -> None:
        """A symbol you have never ruled on is not one you have approved."""
        assert settings_service.rule_for("SPUS") is SymbolRule.NEEDS_APPROVAL

    def test_symbol_rules_round_trip(self, settings_service: SettingsService) -> None:
        settings_service.set_symbol_rule("SPUS", SymbolRule.ALLOWED)
        settings_service.set_symbol_rule("UMMA", SymbolRule.FORBIDDEN)
        assert settings_service.allowed_symbols() == ["SPUS"]
        assert settings_service.forbidden_symbols() == ["UMMA"]

    def test_clear_symbol_rule(self, settings_service: SettingsService) -> None:
        settings_service.set_symbol_rule("SPUS", SymbolRule.ALLOWED)
        settings_service.clear_symbol_rule("SPUS")
        assert settings_service.rule_for("SPUS") is SymbolRule.NEEDS_APPROVAL

    def test_capital_cannot_exceed_the_account(
        self, settings_service: SettingsService
    ) -> None:
        with pytest.raises(ValueError, match="cannot allocate"):
            settings_service.set_capital_allocation(
                Decimal("10000"), account_equity=Decimal("500")
            )

    def test_capital_must_be_positive(self, settings_service: SettingsService) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            settings_service.set_capital_allocation(Decimal("0"))

    def test_autonomy_round_trip(self, settings_service: SettingsService) -> None:
        settings_service.set_autonomy_level(AutonomyLevel.FULLY_AUTONOMOUS)
        assert settings_service.autonomy_level is AutonomyLevel.FULLY_AUTONOMOUS

    def test_universe_mode_round_trip(self, settings_service: SettingsService) -> None:
        settings_service.set_universe_mode(UniverseMode.ETF_AND_SCREENED_STOCKS)
        assert settings_service.universe_mode is UniverseMode.ETF_AND_SCREENED_STOCKS

    def test_falls_back_to_config_when_unset(
        self, settings_service: SettingsService
    ) -> None:
        assert settings_service.autonomy_level is AutonomyLevel.SUGGEST_ONLY

    def test_corrupt_value_does_not_crash(self, settings_service: SettingsService) -> None:
        from investment_box.db.models import SettingOverride

        with settings_service.db.session() as session:
            session.add(SettingOverride(key="autonomy_level", value_json="{not json"))
        assert settings_service.autonomy_level is AutonomyLevel.SUGGEST_ONLY


class TestKillFlag:
    def test_kill_flag_persists(self, settings_service: SettingsService) -> None:
        """A kill must survive a restart, or restarting would undo it."""
        settings_service.request_kill("test")
        assert settings_service.kill_requested
        assert settings_service.kill_reason == "test"

    def test_clearing_is_explicit(self, settings_service: SettingsService) -> None:
        settings_service.request_kill("test")
        settings_service.clear_kill()
        assert not settings_service.kill_requested

    def test_visible_to_a_second_service_instance(
        self, database, settings, audit
    ) -> None:
        """The dashboard and engine are separate processes."""
        SettingsService(database, settings, audit).request_kill("from elsewhere")
        assert SettingsService(database, settings, audit).kill_requested


class TestTelegramControls:
    def test_pause_asks_for_confirmation(self, controls: ControlCommands) -> None:
        result = controls.handle("/pause", ALLOWED_USER)
        assert result.buttons is not None
        assert not result.changed_state

    def test_kill_states_the_exposure(self, controls: ControlCommands) -> None:
        result = controls.handle("/kill", ALLOWED_USER)
        assert "KILL SWITCH" in result.message
        assert "LEFT OPEN" in result.message
        assert not result.changed_state

    def test_kill_all_says_it_will_close_positions(
        self, controls: ControlCommands
    ) -> None:
        result = controls.handle("/kill", ALLOWED_USER, "all")
        assert "close every position" in result.message

    def test_cancelling_changes_nothing(self, controls: ControlCommands) -> None:
        controls.handle("/pause", ALLOWED_USER)
        result = controls.handle_callback("ctl:pause:no", ALLOWED_USER)
        assert result is not None
        assert not result.changed_state
        assert "Cancelled" in result.message

    def test_confirmation_expires(self, controls: ControlCommands, clock) -> None:
        """The situation that prompted a kill changes fast."""
        controls.handle("/kill", ALLOWED_USER)
        clock.advance(seconds=CONFIRMATION_TIMEOUT_SECONDS + 1)
        result = controls.handle_callback("ctl:kill:yes", ALLOWED_USER)
        assert result is not None
        assert "expired" in result.message
        assert not result.changed_state

    def test_confirmation_is_per_user(self, controls: ControlCommands) -> None:
        controls.handle("/pause", ALLOWED_USER)
        result = controls.handle_callback("ctl:pause:yes", 999999)
        assert result is not None
        assert "no longer pending" in result.message

    def test_malformed_callback_is_dropped(self, controls: ControlCommands) -> None:
        for bad in ["garbage", "ctl:pause", "ctl:pause:maybe", "apv:1:approve"]:
            assert controls.handle_callback(bad, ALLOWED_USER) is None

    @pytest.mark.parametrize(
        "action", ["set_live_mode", "switch_to_live", "enable_live_trading"]
    )
    def test_live_switching_is_refused(
        self, controls: ControlCommands, action: str
    ) -> None:
        with pytest.raises(PermissionError, match="dashboard"):
            controls.guard.assert_action_allowed(action)

    def test_messages_are_mode_tagged(self, controls: ControlCommands) -> None:
        assert controls.handle("/pause", ALLOWED_USER).message.startswith("[PAPER]")


class TestPurification:
    def test_amount_is_computed_from_the_ratio(
        self, purification: PurificationTracker
    ) -> None:
        entry = purification.record_dividend(
            symbol="SPUS", pay_date=dt.date(2024, 6, 12),
            gross_amount=Decimal("10.00"), non_permissible_ratio=0.03,
            method=PurificationMethod.ISSUER_RATE,
        )
        assert entry.amount_due == Decimal("0.30")

    def test_missing_ratio_is_flagged_not_guessed(
        self, purification: PurificationTracker
    ) -> None:
        """An invented purification figure is worse than an absent one."""
        entry = purification.record_dividend(
            symbol="SPUS", pay_date=dt.date(2024, 6, 12), gross_amount=Decimal("10.00")
        )
        assert entry.amount_due == Decimal("0.00")
        assert entry.needs_a_ratio

        report = purification.report()
        assert report.entries_without_a_ratio
        assert any("NOT zero" in w for w in report.warnings())

    def test_non_issuer_ratios_are_flagged_as_provisional(
        self, purification: PurificationTracker
    ) -> None:
        purification.record_dividend(
            symbol="SPUS", pay_date=dt.date(2024, 6, 12),
            gross_amount=Decimal("10.00"), non_permissible_ratio=0.03,
            method=PurificationMethod.PROVIDER_RATIO,
        )
        assert any("issuer" in w for w in purification.report().warnings())

    def test_outstanding_excludes_what_was_paid(
        self, purification: PurificationTracker
    ) -> None:
        purification.record_dividend(
            symbol="SPUS", pay_date=dt.date(2024, 6, 12),
            gross_amount=Decimal("10.00"), non_permissible_ratio=0.03,
            method=PurificationMethod.ISSUER_RATE,
        )
        assert purification.outstanding_total() == Decimal("0.30")
        purification.mark_purified("SPUS", dt.date(2024, 6, 12))
        assert purification.outstanding_total() == Decimal("0.00")

    def test_recording_the_same_dividend_twice_updates_it(
        self, purification: PurificationTracker
    ) -> None:
        for ratio in (0.03, 0.05):
            purification.record_dividend(
                symbol="SPUS", pay_date=dt.date(2024, 6, 12),
                gross_amount=Decimal("10.00"), non_permissible_ratio=ratio,
                method=PurificationMethod.ISSUER_RATE,
            )
        report = purification.report()
        assert len(report.entries) == 1
        assert report.total_due == Decimal("0.50")

    def test_csv_marks_unknown_ratios_explicitly(
        self, purification: PurificationTracker
    ) -> None:
        purification.record_dividend(
            symbol="SPUS", pay_date=dt.date(2024, 6, 12), gross_amount=Decimal("10.00")
        )
        csv_text = purification.to_csv()
        assert "UNKNOWN" in csv_text
        assert "are not zero" in csv_text

    def test_empty_report(self, purification: PurificationTracker) -> None:
        assert purification.report().summary() == "no dividends recorded"


class TestZakat:
    def _holdings(self) -> list[ZakatHolding]:
        return [
            ZakatHolding("SPUS", Decimal("300.00")),
            ZakatHolding("SPSK", Decimal("100.00")),
        ]

    def test_full_market_value_method(self) -> None:
        estimate = estimate_zakat(
            as_of=dt.date(2024, 6, 12), holdings=self._holdings(), cash=Decimal("100.00")
        )
        assert estimate.zakatable_base == Decimal("500.00")
        assert estimate.estimated_zakat == Decimal("12.50")  # 2.5%

    def test_liabilities_reduce_the_base(self) -> None:
        estimate = estimate_zakat(
            as_of=dt.date(2024, 6, 12), holdings=self._holdings(),
            cash=Decimal("100.00"), liabilities=Decimal("200.00"),
        )
        assert estimate.zakatable_base == Decimal("300.00")

    def test_below_nisab_yields_zero(self) -> None:
        estimate = estimate_zakat(
            as_of=dt.date(2024, 6, 12), holdings=self._holdings(),
            nisab_threshold=Decimal("5000.00"),
        )
        assert estimate.meets_nisab is False
        assert estimate.estimated_zakat == Decimal("0.00")

    def test_net_method_excludes_holdings_without_a_ratio(self) -> None:
        """Excluded and named, never silently counted as zero."""
        estimate = estimate_zakat(
            as_of=dt.date(2024, 6, 12), holdings=self._holdings(),
            method=ZakatMethod.NET_ZAKATABLE_ASSETS,
        )
        assert set(estimate.excluded) == {"SPUS", "SPSK"}
        assert any("NOT in the figure" in c for c in estimate.caveats())

    def test_always_says_it_is_an_estimate(self) -> None:
        estimate = estimate_zakat(as_of=dt.date(2024, 6, 12), holdings=self._holdings())
        assert "ESTIMATE" in estimate.caveats()[0]
        assert "not a ruling" in estimate.summary()

    def test_names_the_method_used(self) -> None:
        estimate = estimate_zakat(as_of=dt.date(2024, 6, 12), holdings=self._holdings())
        assert any("Method used" in c for c in estimate.caveats())

    def test_mentions_the_hawl_requirement(self) -> None:
        estimate = estimate_zakat(as_of=dt.date(2024, 6, 12), holdings=self._holdings())
        assert any("lunar year" in c for c in estimate.caveats())
