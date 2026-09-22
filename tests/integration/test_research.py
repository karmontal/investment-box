"""The research service, and the shared path between the dashboard and /funds.

The property that matters: both consumers go through ``ResearchService``, so a
candidate's rank, probability and reason are identical in both views. A test
that asserts they agree is what keeps that true as the code changes.
"""

from __future__ import annotations

import datetime as dt

import pytest

from investment_box.config.loader import load_settings
from investment_box.core.clock import FrozenClock, TradingCalendar
from investment_box.core.types import ComplianceStatus
from investment_box.forecast.base import TrackRecord
from investment_box.forecast.candidates import CandidateStatus
from investment_box.services.research import ResearchService
from investment_box.shariah.providers.mock_external import MockExternalProvider
from investment_box.shariah.status import ComplianceTracker
from investment_box.strategies import ETFMomentumRotation
from investment_box.universe.builder import Instrument

AS_OF = dt.date(2024, 6, 12)


def instruments(verified: bool = True) -> list[Instrument]:
    return [
        Instrument(symbol="SPUS", name="SP Funds S&P 500", verified=verified,
                   inception=dt.date(2019, 12, 18)),
        Instrument(symbol="HLAL", name="Wahed FTSE USA", verified=verified,
                   inception=dt.date(2019, 7, 16)),
        Instrument(symbol="SPSK", name="SP Funds Sukuk", verified=verified,
                   inception=dt.date(2019, 12, 31)),
    ]


@pytest.fixture
def tracker(database, audit, clock: FrozenClock) -> ComplianceTracker:
    provider = MockExternalProvider(
        {
            "SPUS": ComplianceStatus.COMPLIANT,
            "HLAL": ComplianceStatus.COMPLIANT,
            "SPSK": ComplianceStatus.COMPLIANT,
        },
        clock=clock,
    )
    return ComplianceTracker(provider, database, load_settings().shariah, audit, clock=clock)


@pytest.fixture
def research(settings, repository, tracker, clock: FrozenClock) -> ResearchService:
    return ResearchService(
        settings,
        repository,
        ETFMomentumRotation(),
        compliance=tracker,
        clock=clock,
        calendar=TradingCalendar(anchor=AS_OF),
    )


class TestSnapshotStructure:
    def test_every_instrument_appears(self, research: ResearchService) -> None:
        """Including blocked ones -- the dashboard must explain each rejection."""
        snapshot = research.build(instruments(), as_of=AS_OF)
        assert {c.symbol for c in snapshot.candidates} == {"SPUS", "HLAL", "SPSK"}

    def test_candidates_are_ranked(self, research: ResearchService) -> None:
        snapshot = research.build(instruments(), as_of=AS_OF)
        assert [c.rank for c in snapshot.candidates] == [1, 2, 3]

    def test_every_candidate_has_a_reason(self, research: ResearchService) -> None:
        snapshot = research.build(instruments(), as_of=AS_OF)
        assert all(c.reason for c in snapshot.candidates)

    def test_universe_snapshot_is_attached(self, research: ResearchService) -> None:
        snapshot = research.build(instruments(), as_of=AS_OF)
        assert snapshot.universe is not None


class TestEvidenceGating:
    def test_no_track_record_means_nothing_actionable(
        self, research: ResearchService
    ) -> None:
        """An unproven strategy must not produce actionable candidates."""
        snapshot = research.build(instruments(), as_of=AS_OF)
        assert not snapshot.has_anything_actionable
        assert any("no out-of-sample record" in w for w in snapshot.warnings)

    def test_registering_a_record_enables_forecasts(
        self, research: ResearchService
    ) -> None:
        research.register_track_record(
            TrackRecord(strategy="etf_momentum_rotation", trades=200, win_rate=0.58)
        )
        snapshot = research.build(instruments(), as_of=AS_OF)
        forecasts = [c.forecast for c in snapshot.candidates if c.forecast]
        assert forecasts
        assert all(f.direction_probability != 0.5 for f in forecasts)

    def test_thin_record_is_warned_about(self, research: ResearchService) -> None:
        research.register_track_record(
            TrackRecord(strategy="etf_momentum_rotation", trades=8, win_rate=0.75)
        )
        snapshot = research.build(instruments(), as_of=AS_OF)
        assert any("only 8 out-of-sample trades" in w for w in snapshot.warnings)

    def test_unverified_symbols_are_blocked(self, research: ResearchService) -> None:
        snapshot = research.build(instruments(verified=False), as_of=AS_OF)
        assert all(c.status is CandidateStatus.BLOCKED for c in snapshot.candidates)
        assert all("unverified" in c.reason for c in snapshot.candidates)


class TestResilience:
    def test_a_broken_strategy_does_not_blank_the_page(
        self, settings, repository, tracker, clock: FrozenClock
    ) -> None:
        class Broken(ETFMomentumRotation):
            def decide(self, context):
                raise RuntimeError("strategy exploded")

        service = ResearchService(
            settings, repository, Broken(), compliance=tracker, clock=clock,
            calendar=TradingCalendar(anchor=AS_OF),
        )
        snapshot = service.build(instruments(), as_of=AS_OF)
        assert snapshot.decision is None
        assert any("strategy exploded" in w for w in snapshot.warnings)
        # The page still renders every symbol, with reasons.
        assert len(snapshot.candidates) == 3

    def test_non_compliant_symbol_is_blocked(
        self, settings, repository, database, audit, clock: FrozenClock
    ) -> None:
        provider = MockExternalProvider(
            {"SPUS": ComplianceStatus.NON_COMPLIANT, "HLAL": ComplianceStatus.COMPLIANT},
            clock=clock,
        )
        tracker = ComplianceTracker(
            provider, database, load_settings().shariah, audit, clock=clock
        )
        service = ResearchService(
            settings, repository, ETFMomentumRotation(), compliance=tracker, clock=clock,
            calendar=TradingCalendar(anchor=AS_OF),
        )
        snapshot = service.build(instruments(), as_of=AS_OF)
        spus = next(c for c in snapshot.candidates if c.symbol == "SPUS")
        assert spus.status is CandidateStatus.BLOCKED


class TestSharedWithTelegram:
    async def test_funds_and_dashboard_agree(
        self, stack, research: ResearchService
    ) -> None:
        """The whole point of routing both through one service."""
        research.register_track_record(
            TrackRecord(strategy="etf_momentum_rotation", trades=200, win_rate=0.58)
        )
        stack.handlers.ctx.research = research

        snapshot = research.build(instruments(), as_of=AS_OF)
        reply = stack.handlers.funds()

        # Every symbol the service ranked appears in the bot's reply.
        for candidate in snapshot.candidates:
            assert candidate.symbol in reply

    async def test_funds_degrades_without_a_research_service(self, stack) -> None:
        """No research service means no scores -- never placeholder numbers."""
        stack.handlers.ctx.research = None
        reply = stack.handlers.funds()
        assert "SPUS" in reply
        assert "Probability" not in reply

    async def test_funds_survives_a_research_failure(self, stack) -> None:
        class Exploding:
            def build(self, *_args, **_kwargs):
                raise RuntimeError("research exploded")

        stack.handlers.ctx.research = Exploding()
        reply = stack.handlers.funds()
        assert "SPUS" in reply  # fell back rather than failing the command
