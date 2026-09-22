"""Screening providers and the compliance tracker.

The property under test throughout: **missing or stale evidence never produces
a pass.** Compliance is a claim about a company at a time, and the safe answer
when you cannot substantiate it is UNKNOWN, not COMPLIANT.
"""

from __future__ import annotations

import datetime as dt

import pytest

from investment_box.config.loader import load_settings
from investment_box.core.clock import FrozenClock
from investment_box.core.types import ComplianceStatus
from investment_box.shariah.providers.base import FinancialRatios
from investment_box.shariah.providers.internal_aaoifi import InternalAAOIFIProvider
from investment_box.shariah.providers.mock_external import MockExternalProvider
from investment_box.shariah.status import ComplianceTracker


def fundamentals(**overrides) -> dict:
    base = {
        "sector": "Technology",
        "industry": "Software - Infrastructure",
        "long_business_summary": "Builds software.",
        "market_cap": 1_000_000_000.0,
        "total_debt": 100_000_000.0,
        "cash_and_securities": 50_000_000.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def provider(clock: FrozenClock) -> InternalAAOIFIProvider:
    settings = load_settings()
    return InternalAAOIFIProvider(
        settings.shariah, clock=clock, fundamentals_fetcher=lambda _s: fundamentals()
    )


def with_data(clock: FrozenClock, data: dict) -> InternalAAOIFIProvider:
    return InternalAAOIFIProvider(
        load_settings().shariah, clock=clock, fundamentals_fetcher=lambda _s: data
    )


class TestFinancialRatios:
    def test_breach_detection(self) -> None:
        ratios = FinancialRatios(denominator="market_cap", debt_ratio=0.45)
        assert ratios.breaches({"debt": 0.30}) == ["debt 45.0% >= 30%"]

    def test_passing_ratios_report_nothing(self) -> None:
        ratios = FinancialRatios(denominator="market_cap", debt_ratio=0.10)
        assert ratios.breaches({"debt": 0.30}) == []

    def test_missing_ratio_is_not_a_breach(self) -> None:
        """A missing ratio is an unknown, handled separately, not a pass or a fail."""
        assert FinancialRatios(denominator="market_cap").breaches({"debt": 0.30}) == []

    def test_incomplete_is_detected(self) -> None:
        assert not FinancialRatios(denominator="market_cap", debt_ratio=0.1).is_complete


class TestInternalProvider:
    def test_excessive_debt_is_non_compliant(self, clock: FrozenClock) -> None:
        result = with_data(clock, fundamentals(total_debt=400_000_000.0)).screen("TEST")
        assert result.status is ComplianceStatus.NON_COMPLIANT
        assert "debt" in result.reason

    def test_excessive_interest_securities_is_non_compliant(self, clock: FrozenClock) -> None:
        result = with_data(clock, fundamentals(cash_and_securities=400_000_000.0)).screen("TEST")
        assert result.status is ComplianceStatus.NON_COMPLIANT

    @pytest.mark.parametrize(
        ("industry", "activity"),
        [
            ("Banks - Regional", "conventional_banking"),
            ("Insurance - Life", "conventional_insurance"),
            ("Beverages - Brewers", "alcohol"),
            ("Resorts & Casinos", "gambling"),
            ("Tobacco", "tobacco"),
            ("Aerospace & Defense", "weapons"),
        ],
    )
    def test_impermissible_activities_are_caught(
        self, clock: FrozenClock, industry: str, activity: str
    ) -> None:
        result = with_data(clock, fundamentals(industry=industry)).screen("TEST")
        assert result.status is ComplianceStatus.NON_COMPLIANT
        assert activity in result.activity_flags

    def test_activity_check_beats_good_ratios(self, clock: FrozenClock) -> None:
        """A bank with no debt is still a bank."""
        data = fundamentals(industry="Banks - Regional", total_debt=0.0)
        assert with_data(clock, data).screen("TEST").status is ComplianceStatus.NON_COMPLIANT

    def test_clean_ratios_yield_doubtful_not_compliant(
        self, provider: InternalAAOIFIProvider
    ) -> None:
        """Revenue purity cannot be measured without segment data.

        The honest verdict is DOUBTFUL, which is never auto-traded, rather than
        COMPLIANT on the strength of ratios alone.
        """
        result = provider.screen("TEST")
        assert result.status is ComplianceStatus.DOUBTFUL
        assert not result.is_tradable
        assert "non-permissible revenue could not be measured" in result.reason

    def test_missing_fundamentals_yield_unknown(self, clock: FrozenClock) -> None:
        result = with_data(clock, {}).screen("TEST")
        assert result.status is ComplianceStatus.UNKNOWN
        assert not result.is_tradable

    def test_missing_market_cap_yields_unknown(self, clock: FrozenClock) -> None:
        result = with_data(clock, fundamentals(market_cap=None)).screen("TEST")
        assert result.status is ComplianceStatus.UNKNOWN

    def test_fetch_failure_yields_unknown_not_a_crash(self, clock: FrozenClock) -> None:
        def explode(_symbol: str) -> dict:
            raise RuntimeError("network down")

        provider = InternalAAOIFIProvider(
            load_settings().shariah, clock=clock, fundamentals_fetcher=explode
        )
        assert provider.screen("TEST").status is ComplianceStatus.UNKNOWN

    def test_is_not_a_certified_source(self, provider: InternalAAOIFIProvider) -> None:
        """Computed screens are estimates, and every trade records which it was."""
        assert provider.is_certified_source is False

    def test_denominator_is_recorded(self, provider: InternalAAOIFIProvider) -> None:
        result = provider.screen("TEST")
        assert result.ratios is not None
        assert result.ratios.denominator == "market_cap"

    def test_batch_isolates_failures(self, clock: FrozenClock) -> None:
        def selective(symbol: str) -> dict:
            if symbol == "BAD":
                raise RuntimeError("boom")
            return fundamentals()

        provider = InternalAAOIFIProvider(
            load_settings().shariah, clock=clock, fundamentals_fetcher=selective
        )
        results = provider.screen_many(["GOOD", "BAD"])
        assert results["GOOD"].status is ComplianceStatus.DOUBTFUL
        assert results["BAD"].status is ComplianceStatus.UNKNOWN


class TestMockProvider:
    def test_unknown_symbol_defaults_to_unknown(self, clock: FrozenClock) -> None:
        """A mock that answered COMPLIANT for anything would make tests agree
        with a dangerous default."""
        assert MockExternalProvider(clock=clock).screen("ANYTHING").status is (
            ComplianceStatus.UNKNOWN
        )

    def test_configured_verdict_is_returned(self, clock: FrozenClock) -> None:
        provider = MockExternalProvider({"SPUS": ComplianceStatus.COMPLIANT}, clock=clock)
        assert provider.screen("spus").status is ComplianceStatus.COMPLIANT

    def test_never_claims_to_be_certified(self, clock: FrozenClock) -> None:
        assert MockExternalProvider(clock=clock).is_certified_source is False


class TestComplianceTracker:
    @pytest.fixture
    def tracker(self, database, audit, clock: FrozenClock) -> ComplianceTracker:
        provider = MockExternalProvider(
            {"SPUS": ComplianceStatus.COMPLIANT, "BAD": ComplianceStatus.NON_COMPLIANT},
            clock=clock,
        )
        return ComplianceTracker(
            provider, database, load_settings().shariah, audit, clock=clock
        )

    def test_screen_persists_and_returns(self, tracker: ComplianceTracker) -> None:
        record = tracker.screen("SPUS")
        assert record.status is ComplianceStatus.COMPLIANT
        assert record.is_tradable
        assert tracker.latest("SPUS") is not None

    def test_fresh_screen_is_not_repeated(self, tracker: ComplianceTracker) -> None:
        tracker.screen("SPUS")
        tracker.screen("SPUS")
        assert tracker.provider.calls == ["SPUS"]  # type: ignore[attr-defined]

    def test_force_rescreens(self, tracker: ComplianceTracker) -> None:
        tracker.screen("SPUS")
        tracker.screen("SPUS", force=True)
        assert len(tracker.provider.calls) == 2  # type: ignore[attr-defined]

    def test_stale_screen_reads_as_unknown(
        self, tracker: ComplianceTracker, clock: FrozenClock
    ) -> None:
        """Compliance is a fact about a company at a time; companies change."""
        tracker.screen("SPUS")
        clock.advance(days=30)
        record = tracker.latest("SPUS")
        assert record is not None
        assert record.is_stale
        assert record.display_status is ComplianceStatus.UNKNOWN
        assert not record.is_tradable

    def test_non_compliant_is_not_tradable(self, tracker: ComplianceTracker) -> None:
        assert not tracker.screen("BAD").is_tradable

    def test_unscreened_symbol_is_not_tradable(self, tracker: ComplianceTracker) -> None:
        assert not tracker.screen("NEVER_HEARD_OF").is_tradable

    def test_tradable_filters_correctly(self, tracker: ComplianceTracker) -> None:
        assert tracker.tradable(["SPUS", "BAD", "UNKNOWN_ONE"]) == ["SPUS"]

    def test_status_change_is_audited(
        self, tracker: ComplianceTracker, clock: FrozenClock, audit
    ) -> None:
        tracker.screen("SPUS")
        tracker.provider.set_verdict("SPUS", ComplianceStatus.NON_COMPLIANT)  # type: ignore[attr-defined]
        clock.advance(days=30)
        tracker.screen("SPUS")

        entries = audit.recent(event_type="compliance.status_changed")
        assert entries
        assert "COMPLIANT -> NON_COMPLIANT" in entries[0].summary

    def test_history_is_append_only(
        self, tracker: ComplianceTracker, clock: FrozenClock
    ) -> None:
        tracker.screen("SPUS")
        clock.advance(days=30)
        tracker.screen("SPUS")
        assert len(tracker.history("SPUS")) == 2

    def test_status_at_reflects_what_was_known_then(
        self, tracker: ComplianceTracker, clock: FrozenClock
    ) -> None:
        """The audit question: what did we believe when we traded?"""
        tracker.screen("SPUS")
        first_time = clock.now()

        tracker.provider.set_verdict("SPUS", ComplianceStatus.NON_COMPLIANT)  # type: ignore[attr-defined]
        clock.advance(days=30)
        tracker.screen("SPUS")

        historical = tracker.status_at("SPUS", first_time)
        assert historical is not None
        assert historical.status is ComplianceStatus.COMPLIANT

    def test_newly_non_compliant_flags_holdings(self, tracker: ComplianceTracker) -> None:
        flagged = tracker.newly_non_compliant(["SPUS", "BAD"])
        assert [r.symbol for r in flagged] == ["BAD"]

    def test_exit_deadline_uses_configured_days(self, tracker: ComplianceTracker) -> None:
        deadline = tracker.exit_deadline(dt.date(2024, 6, 12))
        assert deadline == dt.date(2024, 6, 15)  # default 3 days
