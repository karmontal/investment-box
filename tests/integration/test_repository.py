"""Cache and repository behaviour, end to end against the synthetic provider."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from investment_box.core.clock import UTC, TradingCalendar
from investment_box.data.cache import ParquetCache
from investment_box.data.repository import MarketDataRepository
from investment_box.data.synthetic import SyntheticDataProvider

START = dt.date(2024, 1, 2)
END = dt.date(2024, 6, 12)


class TestSyntheticProvider:
    def test_is_deterministic(self, synthetic_provider: SyntheticDataProvider) -> None:
        """Test failures must be reproducible, so the generator is seeded."""
        first = synthetic_provider.get_bars("SPUS", START, END)
        second = synthetic_provider.get_bars("SPUS", START, END)
        assert first.equals(second)

    def test_symbols_differ(self, synthetic_provider: SyntheticDataProvider) -> None:
        spus = synthetic_provider.get_bars("SPUS", START, END)
        hlal = synthetic_provider.get_bars("HLAL", START, END)
        assert not spus["close"].equals(hlal["close"])

    def test_only_trading_days(self, synthetic_provider: SyntheticDataProvider) -> None:
        frame = synthetic_provider.get_bars("SPUS", START, END)
        weekdays = {ts.weekday() for ts in frame.index}
        assert weekdays <= {0, 1, 2, 3, 4}

    def test_ohlc_relationships_hold(self, synthetic_provider: SyntheticDataProvider) -> None:
        frame = synthetic_provider.get_bars("SPUS", START, END)
        assert (frame["high"] >= frame["low"]).all()
        assert (frame["high"] >= frame["close"]).all()
        assert (frame["low"] <= frame["open"]).all()
        assert (frame[["open", "high", "low", "close"]] > 0).all().all()

    def test_inverted_range_returns_empty(self, synthetic_provider: SyntheticDataProvider) -> None:
        assert synthetic_provider.get_bars("SPUS", END, START).empty


class TestParquetCache:
    def test_round_trip(self, tmp_path: Path, synthetic_provider: SyntheticDataProvider) -> None:
        cache = ParquetCache(tmp_path)
        frame = synthetic_provider.get_bars("SPUS", START, END)
        cache.write("SPUS", frame)
        assert len(cache.read("SPUS")) == len(frame)

    def test_missing_symbol_returns_empty(self, tmp_path: Path) -> None:
        assert ParquetCache(tmp_path).read("NOPE").empty

    def test_awkward_ticker_is_sanitised(self, tmp_path: Path) -> None:
        path = ParquetCache(tmp_path).path_for("^VIX")
        assert "^" not in path.name
        assert path.name.startswith("_VIX")

    def test_merge_keeps_new_rows_on_conflict(
        self, tmp_path: Path, synthetic_provider: SyntheticDataProvider
    ) -> None:
        """A re-fetch carrying a late split adjustment must win over the cache."""
        cache = ParquetCache(tmp_path)
        original = synthetic_provider.get_bars("SPUS", START, END)
        cache.write("SPUS", original)

        corrected = original.copy()
        corrected["close"] = corrected["close"] * 0.5
        merged = cache.merge_write("SPUS", corrected)
        assert merged["close"].iloc[-1] == pytest.approx(original["close"].iloc[-1] * 0.5)

    def test_merge_extends_the_range(
        self, tmp_path: Path, synthetic_provider: SyntheticDataProvider
    ) -> None:
        cache = ParquetCache(tmp_path)
        head = synthetic_provider.get_bars("SPUS", START, dt.date(2024, 3, 1))
        tail = synthetic_provider.get_bars("SPUS", dt.date(2024, 3, 4), END)
        cache.write("SPUS", head)
        merged = cache.merge_write("SPUS", tail)
        assert merged.index[0].date() <= START
        assert merged.index[-1].date() >= dt.date(2024, 6, 11)

    def test_freshness_is_measured_from_the_last_bar(
        self, tmp_path: Path, synthetic_provider: SyntheticDataProvider
    ) -> None:
        """A recently-written file full of stale bars is stale."""
        cache = ParquetCache(tmp_path, ttl_hours=12)
        cache.write("SPUS", synthetic_provider.get_bars("SPUS", START, END))
        now = dt.datetime(2024, 7, 12, tzinfo=UTC)  # a month after the last bar
        assert not cache.is_fresh("SPUS", now=now)

    def test_recent_bars_are_fresh(
        self, tmp_path: Path, synthetic_provider: SyntheticDataProvider
    ) -> None:
        cache = ParquetCache(tmp_path, ttl_hours=48)
        cache.write("SPUS", synthetic_provider.get_bars("SPUS", START, END))
        assert cache.is_fresh("SPUS", now=dt.datetime(2024, 6, 12, 18, tzinfo=UTC))

    def test_zero_ttl_is_never_fresh(
        self, tmp_path: Path, synthetic_provider: SyntheticDataProvider
    ) -> None:
        cache = ParquetCache(tmp_path, ttl_hours=0)
        cache.write("SPUS", synthetic_provider.get_bars("SPUS", START, END))
        assert not cache.is_fresh("SPUS")

    def test_corrupt_file_does_not_raise(self, tmp_path: Path) -> None:
        cache = ParquetCache(tmp_path)
        cache.path_for("BAD").write_bytes(b"this is not parquet")
        assert cache.read("BAD").empty

    def test_invalidate_and_clear(
        self, tmp_path: Path, synthetic_provider: SyntheticDataProvider
    ) -> None:
        cache = ParquetCache(tmp_path)
        cache.write("SPUS", synthetic_provider.get_bars("SPUS", START, END))
        cache.write("HLAL", synthetic_provider.get_bars("HLAL", START, END))
        cache.invalidate("SPUS")
        assert cache.read("SPUS").empty
        assert cache.clear() == 1


class TestRepository:
    def test_fetches_and_caches(self, repository: MarketDataRepository) -> None:
        first = repository.get_bars("SPUS", START, END)
        assert not first.is_empty
        assert not first.from_cache

        second = repository.get_bars("SPUS", START, END)
        assert second.from_cache
        assert len(second.frame) == len(first.frame)

    def test_window_is_respected(self, repository: MarketDataRepository) -> None:
        result = repository.get_bars("SPUS", dt.date(2024, 3, 1), dt.date(2024, 3, 28))
        assert result.frame.index[0].date() >= dt.date(2024, 3, 1)
        assert result.frame.index[-1].date() <= dt.date(2024, 3, 28)

    def test_as_of_excludes_the_cutoff_bar(self, repository: MarketDataRepository) -> None:
        """The look-ahead guard. A strategy deciding on day D must not see day D."""
        cutoff = dt.date(2024, 5, 1)
        result = repository.get_bars("SPUS", START, END, as_of=cutoff)
        assert all(ts.date() < cutoff for ts in result.frame.index)

    def test_as_of_is_enforced_even_when_the_cache_holds_more(
        self, repository: MarketDataRepository
    ) -> None:
        repository.get_bars("SPUS", START, END)  # cache the full range
        result = repository.get_bars("SPUS", START, END, as_of=dt.date(2024, 3, 1))
        assert result.from_cache
        assert result.frame.index[-1].date() < dt.date(2024, 3, 1)

    def test_inverted_window_raises(self, repository: MarketDataRepository) -> None:
        from investment_box.core.errors import DataError

        with pytest.raises(DataError, match="precedes start"):
            repository.get_bars("SPUS", END, START)

    def test_synthetic_source_is_labelled(self, repository: MarketDataRepository) -> None:
        """Nothing should be able to mistake generated prices for real ones."""
        assert repository.get_bars("SPUS", START, END).is_synthetic

    def test_get_many_isolates_a_failing_symbol(
        self, repository: MarketDataRepository, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = repository.provider.get_bars

        def explode(symbol: str, *args: object, **kwargs: object):
            if symbol == "BAD":
                raise RuntimeError("delisted")
            return original(symbol, *args, **kwargs)

        monkeypatch.setattr(repository.provider, "get_bars", explode)
        repository.fallback = None

        results = repository.get_many(["SPUS", "BAD", "HLAL"], START, END)
        assert not results["SPUS"].is_empty
        assert results["BAD"].is_empty
        assert not results["HLAL"].is_empty

    def test_data_is_clean_after_the_repository(
        self, repository: MarketDataRepository
    ) -> None:
        frame = repository.get_bars("SPUS", START, END).frame
        assert frame.index.is_monotonic_increasing
        assert not frame.index.duplicated().any()
        assert not frame.isna().any().any()
        assert frame.index.tz is not None

    def test_only_trading_days_survive(
        self, repository: MarketDataRepository, calendar: TradingCalendar
    ) -> None:
        frame = repository.get_bars("SPUS", START, END).frame
        sessions = set(calendar.sessions)
        assert all(ts.date() in sessions for ts in frame.index)
