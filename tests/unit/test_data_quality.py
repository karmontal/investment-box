"""Cleaning, validation and the look-ahead guard."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from investment_box.core.errors import DataQualityError
from investment_box.data.base import INDEX_NAME
from investment_box.data.clean import (
    assert_no_lookahead,
    clean_ohlcv,
    summarise_coverage,
    validate_ohlcv,
)


def frame_from(rows: list[dict], dates: list[dt.date]) -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [dt.datetime.combine(d, dt.time.min, tzinfo=dt.UTC) for d in dates], name=INDEX_NAME
    )
    return pd.DataFrame(rows, index=index)


def good_row(close: float = 100.0, volume: float = 1e6) -> dict:
    return {
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": volume,
    }


class TestCleaning:
    def test_clean_data_passes_through(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11), dt.date(2024, 6, 12)]
        frame = frame_from([good_row(100), good_row(101), good_row(102)], dates)
        out, report = clean_ohlcv(frame, "TEST")
        assert len(out) == 3
        assert report.is_clean

    def test_duplicate_timestamps_dropped_keeping_last(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 10)]
        frame = frame_from([good_row(100), good_row(105)], dates)
        out, report = clean_ohlcv(frame, "TEST")
        assert len(out) == 1
        assert out["close"].iloc[0] == pytest.approx(105.0)
        assert report.duplicates_dropped == 1

    def test_nan_rows_dropped(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        rows = [good_row(100), good_row(101)]
        rows[1]["close"] = np.nan
        out, report = clean_ohlcv(frame_from(rows, dates), "TEST")
        assert len(out) == 1
        assert report.nan_rows_dropped == 1

    def test_nan_rows_are_not_forward_filled(self) -> None:
        """A filled close is a fake zero return. Dropping is the honest choice."""
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11), dt.date(2024, 6, 12)]
        rows = [good_row(100), good_row(101), good_row(102)]
        rows[1]["close"] = np.nan
        out, _ = clean_ohlcv(frame_from(rows, dates), "TEST")
        assert list(out["close"]) == pytest.approx([100.0, 102.0])

    def test_non_positive_prices_dropped(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        rows = [good_row(100), good_row(101)]
        rows[1]["low"] = -1.0
        out, report = clean_ohlcv(frame_from(rows, dates), "TEST")
        assert len(out) == 1
        assert report.non_positive_dropped == 1

    def test_impossible_ohlc_dropped(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        rows = [good_row(100), good_row(101)]
        rows[1]["high"] = 50.0  # high below low
        out, report = clean_ohlcv(frame_from(rows, dates), "TEST")
        assert len(out) == 1
        assert report.invalid_ohlc_dropped == 1

    def test_missing_column_raises(self) -> None:
        dates = [dt.date(2024, 6, 10)]
        frame = frame_from([{"open": 1.0, "high": 2.0, "low": 0.5}], dates)
        with pytest.raises(DataQualityError, match="missing columns"):
            clean_ohlcv(frame, "TEST")

    def test_index_is_normalised_to_utc(self) -> None:
        naive = pd.DatetimeIndex([dt.datetime(2024, 6, 10)])  # noqa: DTZ001 - the point of the test
        frame = pd.DataFrame([good_row(100)], index=naive)
        out, _ = clean_ohlcv(frame, "TEST")
        assert out.index.tz is not None
        assert out.index.name == INDEX_NAME

    def test_unsorted_index_is_sorted(self) -> None:
        dates = [dt.date(2024, 6, 12), dt.date(2024, 6, 10)]
        out, _ = clean_ohlcv(frame_from([good_row(102), good_row(100)], dates), "TEST")
        assert out.index.is_monotonic_increasing

    def test_empty_frame_survives(self) -> None:
        out, report = clean_ohlcv(pd.DataFrame(), "TEST")
        assert out.empty
        assert report.rows_in == 0


class TestSplitDetection:
    def test_unadjusted_two_for_one_split_flagged(self) -> None:
        """A halving overnight is a split, not a -50% day, for a broad ETF."""
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        out, report = clean_ohlcv(frame_from([good_row(100), good_row(50)], dates), "TEST")
        assert report.suspected_unadjusted_splits
        assert "2:1" in report.suspected_unadjusted_splits[0]
        assert not report.is_clean
        assert len(out) == 2  # flagged, not silently dropped

    def test_strict_mode_refuses_suspect_data(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        with pytest.raises(DataQualityError, match="unadjusted split"):
            clean_ohlcv(frame_from([good_row(100), good_row(50)], dates), "TEST", strict=True)

    def test_large_gain_is_not_treated_as_a_split(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        _, report = clean_ohlcv(frame_from([good_row(100), good_row(145)], dates), "TEST")
        assert report.extreme_moves
        assert not report.suspected_unadjusted_splits

    def test_ordinary_volatility_is_not_flagged(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        _, report = clean_ohlcv(frame_from([good_row(100), good_row(97)], dates), "TEST")
        assert not report.extreme_moves


class TestCalendarValidation:
    def test_complete_coverage_reports_no_gaps(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11), dt.date(2024, 6, 12)]
        frame = frame_from([good_row(100)] * 3, dates)
        report = validate_ohlcv(frame, "TEST", dates)
        assert report.gaps == []

    def test_small_gap_reported_but_tolerated(self) -> None:
        present = [dt.date(2024, 6, 10), dt.date(2024, 6, 12)]
        expected = [dt.date(2024, 6, 10), dt.date(2024, 6, 11), dt.date(2024, 6, 12)]
        frame = frame_from([good_row(100)] * 2, present)
        report = validate_ohlcv(frame, "TEST", expected, max_missing_ratio=0.5)
        assert report.gaps == ["2024-06-11"]

    def test_large_gap_raises(self) -> None:
        present = [dt.date(2024, 6, 10)]
        expected = [dt.date(2024, 6, d) for d in (10, 11, 12, 13, 14)]
        frame = frame_from([good_row(100)], present)
        with pytest.raises(DataQualityError, match="expected sessions missing"):
            validate_ohlcv(frame, "TEST", expected, max_missing_ratio=0.05)


class TestLookaheadGuard:
    def test_leaked_bar_raises(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11), dt.date(2024, 6, 12)]
        frame = frame_from([good_row(100)] * 3, dates)
        cutoff = pd.Timestamp("2024-06-11", tz="UTC")
        with pytest.raises(AssertionError, match="Look-ahead"):
            assert_no_lookahead(frame, cutoff)

    def test_clean_history_passes(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        frame = frame_from([good_row(100)] * 2, dates)
        assert_no_lookahead(frame, pd.Timestamp("2024-06-12", tz="UTC"))

    def test_no_cutoff_is_a_no_op(self) -> None:
        dates = [dt.date(2024, 6, 10)]
        assert_no_lookahead(frame_from([good_row(100)], dates), None)


class TestCoverageSummary:
    def test_empty(self) -> None:
        assert summarise_coverage(pd.DataFrame())["rows"] == 0

    def test_populated(self) -> None:
        dates = [dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
        summary = summarise_coverage(frame_from([good_row(100)] * 2, dates))
        assert summary["rows"] == 2
        assert summary["start"] == "2024-06-10"
        assert summary["median_dollar_volume"] == pytest.approx(1e8)
