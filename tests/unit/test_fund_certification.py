"""Compliance verdicts for certified funds, and how the provider is chosen.

The bug these guard: the engine and dashboard each constructed
``MockExternalProvider`` directly, whose default verdict is UNKNOWN. No symbol
could clear the compliance gate, so the application could not trade at all --
on paper or live -- and ``shariah.provider`` was a config key nobody read.

The deeper error underneath it: a certified ETF is not a company, and screening
one with a stock screener produces noise. The internal screener calls SPUS
"non-compliant: weapons". SPUS is a fund whose entire mandate is Shariah
compliance, audited annually by Raqaba LLC.
"""

from __future__ import annotations

import datetime as dt

import pytest

from investment_box.config.loader import load_settings
from investment_box.core.clock import FrozenClock
from investment_box.core.types import ComplianceStatus, UniverseMode
from investment_box.shariah.providers.base import ScreenResult
from investment_box.shariah.providers.factory import (
    ProviderNotImplementedError,
    build_screening_provider,
)
from investment_box.shariah.providers.fund_certification import CertifiedFundProvider
from investment_box.shariah.providers.internal_aaoifi import InternalAAOIFIProvider
from investment_box.shariah.providers.mock_external import MockExternalProvider
from investment_box.shariah.status import ComplianceTracker
from investment_box.universe.builder import Instrument


def fund(symbol: str, *, verified: bool = True, board: str | None = "Raqaba LLC") -> Instrument:
    return Instrument(
        symbol=symbol,
        name=f"{symbol} Test Fund",
        issuer="Test Issuer",
        certifying_board=board,
        inception=dt.date(2020, 1, 1),
        verified=verified,
    )


class TestCertifiedFundProvider:
    def test_a_verified_fund_is_compliant_on_its_board_s_authority(
        self, clock: FrozenClock
    ) -> None:
        provider = CertifiedFundProvider([fund("SPUS")], clock=clock)
        result = provider.screen("SPUS")

        assert result.status is ComplianceStatus.COMPLIANT
        assert result.source == "fund_certification"
        assert "Raqaba LLC" in result.reason
        assert result.raw["certifying_board"] == "Raqaba LLC"

    def test_the_verdict_names_its_source_so_an_audit_can_trace_it(
        self, clock: FrozenClock
    ) -> None:
        """A pass with no attributable authority is the thing to avoid."""
        provider = CertifiedFundProvider([fund("HLAL", board="Yasaar Limited")], clock=clock)
        result = provider.screen("HLAL")
        assert "Yasaar Limited" in result.reason
        assert result.raw["verified_in_config"] is True

    def test_an_unverified_fund_is_unknown_not_compliant(self, clock: FrozenClock) -> None:
        provider = CertifiedFundProvider([fund("SPRE", verified=False)], clock=clock)
        assert provider.screen("SPRE").status is ComplianceStatus.UNKNOWN

    def test_a_verified_fund_with_no_named_board_is_unknown(self, clock: FrozenClock) -> None:
        """The UMMA case: verified, but the factsheet never named the board."""
        provider = CertifiedFundProvider([fund("UMMA", board=None)], clock=clock)
        result = provider.screen("UMMA")
        assert result.status is ComplianceStatus.UNKNOWN
        assert "certification nobody can name" in result.reason

    @pytest.mark.parametrize("board", ["", "   "])
    def test_a_blank_board_string_does_not_count_as_named(
        self, clock: FrozenClock, board: str
    ) -> None:
        provider = CertifiedFundProvider([fund("SPSK", board=board)], clock=clock)
        assert provider.screen("SPSK").status is ComplianceStatus.UNKNOWN

    def test_a_symbol_outside_the_universe_is_unknown_without_a_fallback(
        self, clock: FrozenClock
    ) -> None:
        provider = CertifiedFundProvider([fund("SPUS")], clock=clock)
        assert provider.screen("AAPL").status is ComplianceStatus.UNKNOWN

    def test_a_symbol_outside_the_universe_goes_to_the_fallback(
        self, clock: FrozenClock
    ) -> None:
        fallback = MockExternalProvider(
            {"AAPL": ComplianceStatus.COMPLIANT}, clock=clock
        )
        provider = CertifiedFundProvider([fund("SPUS")], fallback=fallback, clock=clock)

        assert provider.screen("AAPL").status is ComplianceStatus.COMPLIANT
        assert fallback.calls == ["AAPL"]

    def test_a_fund_never_reaches_the_fallback(self, clock: FrozenClock) -> None:
        """The whole point: the stock screener must not get an opinion on an ETF."""
        fallback = MockExternalProvider(
            {"SPUS": ComplianceStatus.NON_COMPLIANT}, clock=clock
        )
        provider = CertifiedFundProvider([fund("SPUS")], fallback=fallback, clock=clock)

        assert provider.screen("SPUS").status is ComplianceStatus.COMPLIANT
        assert fallback.calls == []

    def test_a_failing_fallback_yields_unknown_not_a_pass(self, clock: FrozenClock) -> None:
        class Exploding:
            name = "exploding"
            is_certified_source = False

            def is_available(self) -> bool:
                return True

            def screen(self, symbol: str, *, as_of: dt.date | None = None) -> ScreenResult:
                raise RuntimeError("vendor down")

            def screen_many(
                self, symbols: list[str], *, as_of: dt.date | None = None
            ) -> dict[str, ScreenResult]:
                return {}

        provider = CertifiedFundProvider([fund("SPUS")], fallback=Exploding(), clock=clock)
        assert provider.screen("AAPL").status is ComplianceStatus.UNKNOWN

    def test_it_is_a_certified_source_only_when_nothing_is_computed_locally(
        self, clock: FrozenClock
    ) -> None:
        assert CertifiedFundProvider([fund("SPUS")], clock=clock).is_certified_source is True

        composite = CertifiedFundProvider(
            [fund("SPUS")], fallback=MockExternalProvider(clock=clock), clock=clock
        )
        assert composite.is_certified_source is False

    def test_screen_many_covers_every_symbol(self, clock: FrozenClock) -> None:
        provider = CertifiedFundProvider(
            [fund("SPUS"), fund("SPSK", verified=False)], clock=clock
        )
        results = provider.screen_many(["SPUS", "SPSK", "AAPL"])
        assert results["SPUS"].status is ComplianceStatus.COMPLIANT
        assert results["SPSK"].status is ComplianceStatus.UNKNOWN
        assert results["AAPL"].status is ComplianceStatus.UNKNOWN


class TestTheInternalScreenerIsWrongAboutFunds:
    """Evidence for why funds bypass it. Not a defect in that provider -- a
    category error in applying it."""

    def test_the_internal_screener_does_not_pass_a_certified_etf(
        self, clock: FrozenClock
    ) -> None:
        settings = load_settings()
        internal = InternalAAOIFIProvider(
            settings.shariah,
            clock=clock,
            fundamentals_fetcher=lambda _s: {
                "sector": "Industrials",
                "industry": "Aerospace & Defense",
                "long_business_summary": "Exchange traded fund.",
                "market_cap": None,
            },
        )
        assert internal.screen("SPUS").status is not ComplianceStatus.COMPLIANT

    def test_but_the_composite_still_passes_it(self, clock: FrozenClock) -> None:
        settings = load_settings()
        internal = InternalAAOIFIProvider(settings.shariah, clock=clock)
        provider = CertifiedFundProvider([fund("SPUS")], fallback=internal, clock=clock)
        assert provider.screen("SPUS").status is ComplianceStatus.COMPLIANT


class TestFactory:
    def test_mode_a_builds_no_stock_fallback(self, settings, clock: FrozenClock) -> None:
        provider = build_screening_provider(settings, [fund("SPUS")], clock=clock)
        assert provider.name == "fund_certification"
        assert provider.is_certified_source is True
        assert provider.screen("AAPL").status is ComplianceStatus.UNKNOWN

    def test_mode_b_screens_stocks_through_the_configured_provider(
        self, settings, clock: FrozenClock
    ) -> None:
        mode_b = settings.model_copy(
            update={
                "universe": settings.universe.model_copy(
                    update={"mode": UniverseMode.ETF_AND_SCREENED_STOCKS}
                )
            }
        )
        provider = build_screening_provider(mode_b, [fund("SPUS")], clock=clock)
        assert provider.is_certified_source is False
        assert provider.screen("SPUS").status is ComplianceStatus.COMPLIANT

    def test_an_unimplemented_vendor_fails_loudly(self, settings, clock: FrozenClock) -> None:
        configured = settings.model_copy(
            update={
                "universe": settings.universe.model_copy(
                    update={"mode": UniverseMode.ETF_AND_SCREENED_STOCKS}
                ),
                "shariah": settings.shariah.model_copy(update={"provider": "zoya"}),
            }
        )
        with pytest.raises(ProviderNotImplementedError, match="zoya"):
            build_screening_provider(configured, [fund("SPUS")], clock=clock)


class TestTrackerDoesNotReuseAnotherProviderSVerdict:
    def test_a_cached_mock_verdict_is_not_served_to_a_new_provider(
        self, database, settings, audit, clock: FrozenClock
    ) -> None:
        """The exact failure seen in practice: the mock's UNKNOWN persisted, so
        swapping in a working provider changed nothing until the record aged out."""
        mock = MockExternalProvider(clock=clock)
        ComplianceTracker(mock, database, settings.shariah, audit, clock=clock).screen("SPUS")

        certified = CertifiedFundProvider([fund("SPUS")], clock=clock)
        record = ComplianceTracker(
            certified, database, settings.shariah, audit, clock=clock
        ).screen("SPUS")

        assert record.status is ComplianceStatus.COMPLIANT
        assert record.source == "fund_certification"

    def test_a_fresh_verdict_from_the_same_provider_is_reused(
        self, database, settings, audit, clock: FrozenClock
    ) -> None:
        certified = CertifiedFundProvider([fund("SPUS")], clock=clock)
        tracker = ComplianceTracker(certified, database, settings.shariah, audit, clock=clock)
        tracker.screen("SPUS")
        before = len(tracker.history("SPUS"))
        tracker.screen("SPUS")
        assert len(tracker.history("SPUS")) == before
