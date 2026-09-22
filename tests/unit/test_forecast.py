"""Forecasts, calibration and candidate ranking.

The property under test throughout: **a forecast never claims more than its
evidence supports.** Thin evidence must produce a number near 50% with low or
no confidence, not a confident number with a quiet caveat.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from investment_box.core.types import ComplianceStatus
from investment_box.features.pipeline import build_features
from investment_box.forecast import (
    Candidate,
    CandidateStatus,
    Confidence,
    Forecast,
    ForecastGenerator,
    TrackRecord,
    brier_score,
    classify,
    evaluate_calibration,
    rank_candidates,
    reliability_curve,
    shrink_toward_even,
)
from investment_box.shariah.status import ComplianceRecord
from investment_box.strategies.base import Signal
from investment_box.universe.builder import Instrument, UniverseEntry

AS_OF = dt.date(2024, 6, 12)


def features_for(symbol: str = "SPUS", seed: int = 1):
    rng = np.random.default_rng(seed)
    close = 60 * np.exp(np.cumsum(rng.normal(0.0005, 0.011, 400)))
    index = pd.date_range(end="2024-06-12", periods=400, freq="B", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(np.log(1e6), 0.3, 400),
        },
        index=index,
    )
    return build_features(frame, symbol)


def compliance(status: ComplianceStatus, *, stale: bool = False) -> ComplianceRecord:
    return ComplianceRecord(
        symbol="SPUS",
        status=status,
        source="test",
        screened_at=dt.datetime(2024, 6, 12, tzinfo=dt.UTC),
        reason="test",
        is_stale=stale,
        age_days=30 if stale else 1,
    )


class TestShrinkage:
    def test_no_sample_returns_even(self) -> None:
        assert shrink_toward_even(0.9, 0) == 0.5

    def test_thin_sample_barely_moves(self) -> None:
        """A 70% win rate over 5 trades is not a 70% forecast."""
        assert shrink_toward_even(0.70, 5) < 0.55

    def test_large_sample_approaches_observed(self) -> None:
        assert shrink_toward_even(0.70, 5000) == pytest.approx(0.70, abs=0.01)

    def test_monotonic_in_sample_size(self) -> None:
        values = [shrink_toward_even(0.70, n) for n in (5, 50, 500, 5000)]
        assert values == sorted(values)

    def test_works_below_even_too(self) -> None:
        assert 0.45 < shrink_toward_even(0.30, 10) < 0.5


class TestForecastType:
    def test_probability_must_be_a_probability(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Forecast(
                symbol="SPUS", as_of=AS_OF, strategy="s", direction_probability=1.4,
                horizon_days=5, expected_return_low=-0.01, expected_return_high=0.01,
                confidence=Confidence.LOW, track_record=TrackRecord(strategy="s"),
            )

    def test_inverted_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds high"):
            Forecast(
                symbol="SPUS", as_of=AS_OF, strategy="s", direction_probability=0.6,
                horizon_days=5, expected_return_low=0.05, expected_return_high=-0.05,
                confidence=Confidence.LOW, track_record=TrackRecord(strategy="s"),
            )

    def test_there_is_no_target_price_field(self) -> None:
        """A point target invites being read as a prediction."""
        assert not hasattr(Forecast, "target_price")
        assert "target_price" not in Forecast.__slots__

    def test_no_confidence_is_never_actionable(self) -> None:
        forecast = Forecast(
            symbol="SPUS", as_of=AS_OF, strategy="s", direction_probability=0.95,
            horizon_days=5, expected_return_low=-0.01, expected_return_high=0.01,
            confidence=Confidence.NONE, track_record=TrackRecord(strategy="s"),
        )
        assert not forecast.is_actionable

    def test_probability_at_or_below_even_is_not_actionable(self) -> None:
        forecast = Forecast(
            symbol="SPUS", as_of=AS_OF, strategy="s", direction_probability=0.50,
            horizon_days=5, expected_return_low=-0.01, expected_return_high=0.01,
            confidence=Confidence.HIGH, track_record=TrackRecord(strategy="s"),
        )
        assert not forecast.is_actionable

    def test_range_description_explains_what_it_means(self) -> None:
        forecast = Forecast(
            symbol="SPUS", as_of=AS_OF, strategy="s", direction_probability=0.6,
            horizon_days=5, expected_return_low=-0.02, expected_return_high=0.02,
            confidence=Confidence.LOW, track_record=TrackRecord(strategy="s"),
        )
        text = forecast.range_description()
        assert "1-sigma" in text
        assert "one outcome in three" in text


class TestTrackRecord:
    def test_thin_record_is_not_meaningful(self) -> None:
        assert not TrackRecord(strategy="s", trades=10).is_meaningful

    def test_summary_says_when_too_few(self) -> None:
        summary = TrackRecord(strategy="s", trades=5, win_rate=0.8).summary()
        assert "too few trades" in summary

    def test_empty_record_says_so(self) -> None:
        assert "no out-of-sample record" in TrackRecord(strategy="s").summary()

    def test_beats_coin_flip_detection(self) -> None:
        assert TrackRecord(strategy="s", brier_score=0.20).beats_coin_flip is True
        assert TrackRecord(strategy="s", brier_score=0.30).beats_coin_flip is False
        assert TrackRecord(strategy="s").beats_coin_flip is None


class TestCalibration:
    def test_brier_of_a_perfect_forecaster(self) -> None:
        assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0

    def test_brier_of_always_guessing_even(self) -> None:
        predictions = np.full(100, 0.5)
        outcomes = np.array([1.0] * 50 + [0.0] * 50)
        assert brier_score(predictions, outcomes) == pytest.approx(0.25)

    def test_reliability_curve_buckets(self) -> None:
        predictions = np.array([0.15, 0.25, 0.75, 0.85])
        outcomes = np.array([0.0, 0.0, 1.0, 1.0])
        bins = reliability_curve(predictions, outcomes, bins=4)
        assert all(b.count > 0 for b in bins)
        assert sum(b.count for b in bins) == 4

    def test_empty_buckets_are_omitted_not_reported_as_zero(self) -> None:
        """An untested probability range is not a failed one."""
        bins = reliability_curve(np.array([0.9, 0.95]), np.array([1.0, 1.0]), bins=10)
        assert len(bins) < 10

    def test_overconfidence_is_detected(self) -> None:
        # Predicts 80%, happens 40% of the time.
        predictions = [0.8] * 100
        outcomes = [True] * 40 + [False] * 60
        report = evaluate_calibration("s", predictions, outcomes)
        assert report.calibration_error < 0
        assert "over-confident" in report.bias
        assert any("over-confident" in w for w in report.warnings())

    def test_well_calibrated_is_recognised(self) -> None:
        predictions = [0.6] * 100
        outcomes = [True] * 60 + [False] * 40
        report = evaluate_calibration("s", predictions, outcomes)
        assert report.bias == "well calibrated"

    def test_worse_than_a_coin_flip_is_called_out(self) -> None:
        predictions = [0.9] * 100
        outcomes = [False] * 100
        report = evaluate_calibration("s", predictions, outcomes)
        assert report.beats_coin_flip is False
        assert any("worse than no probability" in w for w in report.warnings())

    def test_small_sample_is_flagged(self) -> None:
        report = evaluate_calibration("s", [0.6] * 10, [True] * 6 + [False] * 4)
        assert not report.is_meaningful
        assert any("resolved predictions" in w for w in report.warnings())

    def test_empty_input(self) -> None:
        assert evaluate_calibration("s", [], []).samples == 0


class TestForecastGenerator:
    def _signal(self, **kwargs) -> Signal:
        return Signal(symbol="SPUS", target_weight=0.5, reason="test", **kwargs)

    def test_no_track_record_gives_a_coin_flip(self) -> None:
        forecast = ForecastGenerator().generate(
            self._signal(), features_for(), as_of=AS_OF, strategy="s",
            track_record=TrackRecord(strategy="s"),
        )
        assert forecast.direction_probability == 0.5
        assert forecast.confidence is Confidence.NONE
        assert not forecast.is_actionable
        assert any("no out-of-sample record" in c for c in forecast.caveats)

    def test_thin_record_is_shrunk_and_flagged(self) -> None:
        forecast = ForecastGenerator().generate(
            self._signal(), features_for(), as_of=AS_OF, strategy="s",
            track_record=TrackRecord(strategy="s", trades=10, win_rate=0.80),
        )
        assert forecast.direction_probability < 0.60
        assert forecast.confidence is Confidence.LOW
        assert any("shrunk toward 50%" in c for c in forecast.caveats)

    def test_solid_record_raises_confidence(self) -> None:
        forecast = ForecastGenerator().generate(
            self._signal(), features_for(), as_of=AS_OF, strategy="s",
            track_record=TrackRecord(strategy="s", trades=200, win_rate=0.60),
        )
        assert forecast.confidence in (Confidence.MEDIUM, Confidence.HIGH)
        assert forecast.direction_probability > 0.55

    def test_probability_is_capped(self) -> None:
        """Claiming 95% on a 5-day equity move is not credible."""
        forecast = ForecastGenerator().generate(
            self._signal(), features_for(), as_of=AS_OF, strategy="s",
            track_record=TrackRecord(strategy="s", trades=10000, win_rate=0.99),
        )
        assert forecast.direction_probability <= 0.80

    def test_range_comes_from_realised_volatility(self) -> None:
        forecast = ForecastGenerator().generate(
            self._signal(suggested_holding_days=5), features_for(), as_of=AS_OF,
            strategy="s", track_record=TrackRecord(strategy="s", trades=100, win_rate=0.6),
        )
        assert forecast.expected_return_low < 0 < forecast.expected_return_high
        assert forecast.volatility is not None

    def test_range_is_symmetric_around_zero(self) -> None:
        """Centring the range on a drift would be a point forecast in disguise."""
        forecast = ForecastGenerator().generate(
            self._signal(), features_for(), as_of=AS_OF, strategy="s",
            track_record=TrackRecord(strategy="s", trades=100, win_rate=0.6),
        )
        assert forecast.expected_return_low == pytest.approx(-forecast.expected_return_high)

    def test_longer_horizon_widens_the_range(self) -> None:
        generator = ForecastGenerator()
        record = TrackRecord(strategy="s", trades=100, win_rate=0.6)
        short = generator.generate(
            self._signal(suggested_holding_days=2), features_for(), as_of=AS_OF,
            strategy="s", track_record=record,
        )
        long = generator.generate(
            self._signal(suggested_holding_days=20), features_for(), as_of=AS_OF,
            strategy="s", track_record=record,
        )
        assert long.expected_return_high > short.expected_return_high

    def test_uncalibrated_model_probability_is_flagged(self) -> None:
        forecast = ForecastGenerator().generate(
            self._signal(metadata={"probability": 0.70}), features_for(), as_of=AS_OF,
            strategy="s", track_record=TrackRecord(strategy="s", trades=100, win_rate=0.6),
        )
        assert any("never been calibrated" in c for c in forecast.caveats)

    def test_measured_bias_is_corrected(self) -> None:
        calibration = evaluate_calibration(
            "s", [0.8] * 100, [True] * 40 + [False] * 60
        )
        forecast = ForecastGenerator().generate(
            self._signal(metadata={"probability": 0.70}), features_for(), as_of=AS_OF,
            strategy="s", track_record=TrackRecord(strategy="s", trades=100, win_rate=0.6),
            calibration=calibration,
        )
        assert forecast.direction_probability < 0.70
        assert any("calibration bias" in c for c in forecast.caveats)

    def test_worse_than_coin_flip_kills_confidence(self) -> None:
        calibration = evaluate_calibration("s", [0.9] * 100, [False] * 100)
        forecast = ForecastGenerator().generate(
            self._signal(metadata={"probability": 0.70}), features_for(), as_of=AS_OF,
            strategy="s", track_record=TrackRecord(strategy="s", trades=100, win_rate=0.6),
            calibration=calibration,
        )
        assert forecast.confidence is Confidence.NONE
        assert not forecast.is_actionable


class TestCandidateClassification:
    def _forecast(self, probability: float, confidence: Confidence) -> Forecast:
        return Forecast(
            symbol="SPUS", as_of=AS_OF, strategy="s", direction_probability=probability,
            horizon_days=5, expected_return_low=-0.02, expected_return_high=0.02,
            confidence=confidence, track_record=TrackRecord(strategy="s", trades=100),
        )

    def _entry(self, included: bool, reason: str) -> UniverseEntry:
        return UniverseEntry(
            instrument=Instrument(symbol="SPUS", verified=True), included=included, reason=reason
        )

    def test_blocked_by_universe(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=None, compliance=None,
            universe_entry=self._entry(False, "unverified: not confirmed"),
        )
        assert candidate.status is CandidateStatus.BLOCKED
        assert "unverified" in candidate.reason

    def test_blocked_by_non_compliance(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=self._forecast(0.7, Confidence.HIGH),
            compliance=compliance(ComplianceStatus.NON_COMPLIANT),
            universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.BLOCKED

    @pytest.mark.parametrize(
        "status", [ComplianceStatus.DOUBTFUL, ComplianceStatus.UNKNOWN]
    )
    def test_doubtful_and_unknown_need_approval(self, status: ComplianceStatus) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=self._forecast(0.7, Confidence.HIGH),
            compliance=compliance(status), universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.NEEDS_APPROVAL

    def test_stale_screen_needs_approval(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=self._forecast(0.7, Confidence.HIGH),
            compliance=compliance(ComplianceStatus.COMPLIANT, stale=True),
            universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.NEEDS_APPROVAL

    def test_no_confidence_is_watch_not_actionable(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=self._forecast(0.9, Confidence.NONE),
            compliance=compliance(ComplianceStatus.COMPLIANT),
            universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.WATCH

    def test_weak_forecast_is_watch(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=self._forecast(0.48, Confidence.HIGH),
            compliance=compliance(ComplianceStatus.COMPLIANT),
            universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.WATCH

    def test_everything_passing_is_actionable(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=self._forecast(0.65, Confidence.HIGH),
            compliance=compliance(ComplianceStatus.COMPLIANT),
            universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.ACTIONABLE
        assert candidate.is_tradable

    def test_missing_forecast_is_watch(self) -> None:
        candidate = classify(
            "SPUS", AS_OF, forecast=None,
            compliance=compliance(ComplianceStatus.COMPLIANT),
            universe_entry=self._entry(True, "passed"),
        )
        assert candidate.status is CandidateStatus.WATCH


class TestRanking:
    def _candidate(
        self, symbol: str, status: CandidateStatus, probability: float, confidence: Confidence
    ) -> Candidate:
        forecast = Forecast(
            symbol=symbol, as_of=AS_OF, strategy="s", direction_probability=probability,
            horizon_days=5, expected_return_low=-0.02, expected_return_high=0.02,
            confidence=confidence, track_record=TrackRecord(strategy="s", trades=100),
        )
        return Candidate(symbol=symbol, as_of=AS_OF, status=status, reason="", forecast=forecast)

    def test_actionable_outranks_blocked_regardless_of_score(self) -> None:
        ranked = rank_candidates(
            [
                self._candidate("BLOCKED", CandidateStatus.BLOCKED, 0.95, Confidence.HIGH),
                self._candidate("GOOD", CandidateStatus.ACTIONABLE, 0.55, Confidence.HIGH),
            ]
        )
        assert ranked[0].symbol == "GOOD"

    def test_confidence_weights_the_score(self) -> None:
        """An unproven 70% must not outrank a well-evidenced 58%."""
        unproven = self._candidate("UNPROVEN", CandidateStatus.ACTIONABLE, 0.70, Confidence.LOW)
        proven = self._candidate("PROVEN", CandidateStatus.ACTIONABLE, 0.58, Confidence.HIGH)
        assert proven.score > unproven.score

    def test_ranks_are_assigned(self) -> None:
        ranked = rank_candidates(
            [
                self._candidate("A", CandidateStatus.ACTIONABLE, 0.60, Confidence.HIGH),
                self._candidate("B", CandidateStatus.ACTIONABLE, 0.65, Confidence.HIGH),
            ]
        )
        assert [c.rank for c in ranked] == [1, 2]
        assert ranked[0].symbol == "B"

    def test_empty_input(self) -> None:
        assert rank_candidates([]) == []
