"""Storing and loading measured performance.

The bug this closes: the forecast layer refuses to act without an
out-of-sample record, and nothing in the application ever produced one.
`register_track_record` was called only by tests -- they exercised the machine
with the wire already connected by hand, which is why a suite of 700-odd tests
said nothing about an engine that could not place a trade.

So these tests care about the two ways a record can be wrong in the dangerous
direction: counted when it should not be (in-sample, stale), rather than
missed.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from investment_box.backtest.metrics import PerformanceMetrics
from investment_box.forecast.track_record_store import TrackRecordStore

NOW = dt.datetime(2026, 9, 23, 12, 0, tzinfo=dt.UTC)


def metrics(
    name: str = "etf_momentum_rotation", *, trades: int = 40, win_rate: float = 0.47
) -> PerformanceMetrics:
    return PerformanceMetrics(
        name=name,
        start=pd.Timestamp("2020-01-02"),
        end=pd.Timestamp("2026-06-30"),
        num_trades=trades,
        win_rate=win_rate,
        avg_trade_pct=0.004,
        sharpe=0.31,
    )


@pytest.fixture
def store(database) -> TrackRecordStore:
    return TrackRecordStore(database)


class TestSaving:
    def test_a_saved_record_round_trips(self, store: TrackRecordStore) -> None:
        store.save(metrics(), source="walk_forward_backtest", out_of_sample=True,
                   measured_at=NOW)
        row = store.latest("etf_momentum_rotation")

        assert row is not None
        assert row.trades == 40
        assert row.win_rate == pytest.approx(0.47)
        assert row.out_of_sample is True
        assert row.period_start == dt.date(2020, 1, 2)

    def test_the_newest_measurement_supersedes_the_older_one(
        self, store: TrackRecordStore
    ) -> None:
        store.save(metrics(trades=10), source="walk_forward_backtest",
                   out_of_sample=True, measured_at=NOW - dt.timedelta(days=30))
        store.save(metrics(trades=99), source="walk_forward_backtest",
                   out_of_sample=True, measured_at=NOW)

        row = store.latest("etf_momentum_rotation")
        assert row is not None
        assert row.trades == 99, "re-running the backtest must supersede, not accumulate"


class TestLoading:
    def test_a_fresh_out_of_sample_record_loads(self, store: TrackRecordStore) -> None:
        store.save(metrics(), source="walk_forward_backtest", out_of_sample=True,
                   measured_at=NOW - dt.timedelta(days=5))

        outcome = store.load(max_age_days=90, now=NOW)
        assert outcome.loaded_strategies == ["etf_momentum_rotation"]
        assert outcome.records[0].win_rate == pytest.approx(0.47)
        assert not outcome.rejected

    def test_an_in_sample_record_is_never_loaded(self, store: TrackRecordStore) -> None:
        """Feeding in-sample numbers to the sceptical component defeats it."""
        store.save(metrics(), source="in_sample_fit", out_of_sample=False, measured_at=NOW)

        outcome = store.load(max_age_days=90, now=NOW)
        assert outcome.records == []
        assert any("in-sample" in r for r in outcome.rejected)

    def test_a_stale_record_is_refused_with_a_reason(self, store: TrackRecordStore) -> None:
        store.save(metrics(), source="walk_forward_backtest", out_of_sample=True,
                   measured_at=NOW - dt.timedelta(days=200))

        outcome = store.load(max_age_days=90, now=NOW)
        assert outcome.records == []
        assert any("200d ago" in r for r in outcome.rejected)

    def test_expiry_is_measured_from_when_it_was_measured_not_the_period(
        self, store: TrackRecordStore
    ) -> None:
        """A backtest over old data, run today, is current evidence about the
        strategy. Age is about the measurement, not the window it covered."""
        store.save(metrics(), source="walk_forward_backtest", out_of_sample=True,
                   measured_at=NOW)
        outcome = store.load(max_age_days=90, now=NOW)
        assert outcome.loaded_strategies == ["etf_momentum_rotation"]

    def test_an_empty_store_loads_nothing_and_says_nothing_was_wrong(
        self, store: TrackRecordStore
    ) -> None:
        outcome = store.load(max_age_days=90, now=NOW)
        assert outcome.records == []
        assert outcome.rejected == []

    def test_a_stale_record_does_not_mask_a_fresh_one_for_another_strategy(
        self, store: TrackRecordStore
    ) -> None:
        store.save(metrics("mean_reversion"), source="walk_forward_backtest",
                   out_of_sample=True, measured_at=NOW - dt.timedelta(days=400))
        store.save(metrics("etf_momentum_rotation"), source="walk_forward_backtest",
                   out_of_sample=True, measured_at=NOW)

        outcome = store.load(max_age_days=90, now=NOW)
        assert outcome.loaded_strategies == ["etf_momentum_rotation"]
        assert len(outcome.rejected) == 1


class TestTheMeasuredRecordStillRefusesABadStrategy:
    def test_a_losing_win_rate_is_loaded_faithfully(self, store: TrackRecordStore) -> None:
        """Wiring the evidence must not launder it.

        The measured rotation win rate is 47% -- below a coin flip. The point of
        connecting this is that the engine declines *for a stated reason*, not
        that it starts trading.
        """
        store.save(metrics(win_rate=0.47), source="walk_forward_backtest",
                   out_of_sample=True, measured_at=NOW)

        loaded = store.load(max_age_days=90, now=NOW).records[0]
        assert loaded.win_rate == pytest.approx(0.47)
        assert loaded.win_rate < 0.5
