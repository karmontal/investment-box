"""The single entry point for market data.

Everything downstream -- features, strategies, the backtester, the dashboard --
asks the repository, never a provider directly. The repository owns caching,
incremental fetching, cleaning and validation, so those cannot be skipped by
accident.

The ``as_of`` parameter is the look-ahead guard: pass it and the repository
refuses to return any bar at or after that date, whatever is in the cache.
Backtests always pass it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import cast

import pandas as pd

from investment_box.config.schema import Settings
from investment_box.core.clock import UTC, TradingCalendar
from investment_box.core.errors import DataError
from investment_box.core.logging import get_logger
from investment_box.data.base import DataProvider, OHLCVFrame, empty_frame
from investment_box.data.cache import ParquetCache
from investment_box.data.clean import DataQualityReport, clean_ohlcv, validate_ohlcv

log = get_logger(__name__)


@dataclass(frozen=True)
class FetchResult:
    """Bars plus the provenance needed to judge whether to trust them."""

    symbol: str
    frame: OHLCVFrame
    source: str
    from_cache: bool
    report: DataQualityReport

    @property
    def is_synthetic(self) -> bool:
        return self.source == "synthetic"

    @property
    def is_empty(self) -> bool:
        return self.frame.empty


class MarketDataRepository:
    """Cached, cleaned, look-ahead-safe access to historical bars."""

    def __init__(
        self,
        provider: DataProvider,
        cache: ParquetCache,
        *,
        calendar: TradingCalendar | None = None,
        fallback: DataProvider | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache
        self.calendar = calendar or TradingCalendar()
        self.fallback = fallback

    def clock_date(self) -> dt.date:
        """Today, in UTC. Split out so tests can pin it."""
        return dt.datetime.now(tz=UTC).date()

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        provider: DataProvider | None = None,
        fallback: DataProvider | None = None,
    ) -> MarketDataRepository:
        """Build a repository from config, choosing the provider by availability.

        Falls back to the synthetic provider when nothing real is usable, so a
        fresh checkout runs end to end before any credentials exist. The
        fallback is logged loudly because synthetic data is not real data.
        """
        cache = ParquetCache(settings.resolved_cache_dir, settings.data.cache_ttl_hours)

        if provider is None:
            provider = cls._select_provider(settings)

        if fallback is None and getattr(provider, "name", "") != "synthetic":
            from investment_box.data.synthetic import SyntheticDataProvider

            fallback = SyntheticDataProvider()

        return cls(provider, cache, fallback=fallback)

    @staticmethod
    def _select_provider(settings: Settings) -> DataProvider:
        from investment_box.config.loader import get_secrets
        from investment_box.data.synthetic import SyntheticDataProvider
        from investment_box.data.yfinance_provider import YFinanceProvider

        if settings.data.provider == "alpaca":
            from investment_box.data.alpaca_provider import AlpacaDataProvider

            secrets = get_secrets()
            alpaca = AlpacaDataProvider(
                api_key=(
                    secrets.alpaca_api_key.get_secret_value()
                    if secrets.alpaca_api_key
                    else None
                ),
                secret_key=(
                    secrets.alpaca_secret_key.get_secret_value()
                    if secrets.alpaca_secret_key
                    else None
                ),
            )
            if alpaca.is_available():
                return alpaca
            log.warning("data.alpaca_unavailable", action="falling back to yfinance")

        yahoo = YFinanceProvider()
        if yahoo.is_available():
            return yahoo

        log.warning(
            "data.no_real_provider",
            action="using SYNTHETIC data",
            warning="Results from synthetic data are meaningless for real decisions.",
        )
        return SyntheticDataProvider()

    def get_bars(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        *,
        as_of: dt.date | None = None,
        force_refresh: bool = False,
        validate: bool = True,
    ) -> FetchResult:
        """Return cleaned bars for ``symbol`` over ``[start, end]``.

        Args:
            symbol: Ticker.
            start: First date wanted, inclusive.
            end: Last date wanted, inclusive.
            as_of: Look-ahead guard. Bars on or after this date are removed
                before returning, regardless of what the cache holds.
            force_refresh: Ignore cache freshness and re-fetch.
            validate: Cross-check coverage against the trading calendar.

        Raises:
            DataError: If the window is invalid or every provider failed.
        """
        if end < start:
            raise DataError(f"{symbol}: end {end} precedes start {start}")

        timeframe = "1d"
        cached = self.cache.read(symbol, timeframe)
        covered = self._cache_covers(cached, start, end)

        # A window that ended in the past is closed: no new bars will ever appear
        # in it, so the TTL -- which asks "is this current as of today?" -- is the
        # wrong question. Without this, every backtest re-hits the provider for
        # decade-old data it already has. Use force_refresh to pull in a late
        # split or dividend adjustment.
        today = self.clock_date()
        window_is_closed = end < today

        serve_from_cache = covered and not force_refresh and (
            window_is_closed or self.cache.is_fresh(symbol, timeframe)
        )

        if serve_from_cache:
            frame, source, from_cache = cached, f"{self.provider.name} (cache)", True
            report = DataQualityReport(symbol=symbol, rows_in=len(cached), rows_out=len(cached))
        else:
            fetch_start = self._incremental_start(cached, start, force_refresh=force_refresh)
            raw, source = self._fetch(symbol, fetch_start, end)
            cleaned, report = clean_ohlcv(raw, symbol)
            frame = (
                self.cache.merge_write(symbol, cleaned, timeframe)
                if not cleaned.empty
                else cached
            )
            from_cache = False

        window = self._slice(frame, start, end, as_of)

        if validate and not window.empty:
            # Validate against the range actually asked for AFTER the look-ahead
            # cutoff. Checking against `end` would report every bar the guard
            # correctly withheld as a missing session.
            effective_end = min(end, as_of - dt.timedelta(days=1)) if as_of else end
            sessions = [day for day in self.calendar.sessions if start <= day <= effective_end]
            validate_ohlcv(window, symbol, sessions)

        return FetchResult(
            symbol=symbol,
            frame=window,
            source=source,
            from_cache=from_cache,
            report=report,
        )

    def get_many(
        self,
        symbols: list[str],
        start: dt.date,
        end: dt.date,
        *,
        as_of: dt.date | None = None,
        force_refresh: bool = False,
    ) -> dict[str, FetchResult]:
        """Fetch several symbols.

        One symbol failing does not fail the batch -- a delisted or mistyped
        ticker should not take the whole universe down. Failures are logged and
        return an empty result, which the universe filter then drops.
        """
        results: dict[str, FetchResult] = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_bars(
                    symbol, start, end, as_of=as_of, force_refresh=force_refresh
                )
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failure
                log.warning("data.symbol_failed", symbol=symbol, error=str(exc))
                results[symbol] = FetchResult(
                    symbol=symbol,
                    frame=empty_frame(),
                    source="error",
                    from_cache=False,
                    report=DataQualityReport(symbol=symbol),
                )
        return results

    def _fetch(self, symbol: str, start: dt.date, end: dt.date) -> tuple[OHLCVFrame, str]:
        try:
            frame = self.provider.get_bars(symbol, start, end)
            if not frame.empty:
                return frame, self.provider.name
            log.info("data.empty_response", symbol=symbol, provider=self.provider.name)
        except Exception as exc:  # noqa: BLE001 - fall through to the backup provider
            log.warning(
                "data.provider_failed",
                symbol=symbol,
                provider=self.provider.name,
                error=str(exc),
            )

        if self.fallback is not None and self.fallback.is_available():
            log.warning("data.using_fallback", symbol=symbol, fallback=self.fallback.name)
            return self.fallback.get_bars(symbol, start, end), self.fallback.name

        return empty_frame(), self.provider.name

    @staticmethod
    def _cache_covers(cached: OHLCVFrame, start: dt.date, end: dt.date) -> bool:
        if cached.empty:
            return False
        first = cast(pd.Timestamp, cached.index[0]).date()
        last = cast(pd.Timestamp, cached.index[-1]).date()
        return first <= start and last >= end

    def _incremental_start(
        self, cached: OHLCVFrame, start: dt.date, *, force_refresh: bool
    ) -> dt.date:
        """Fetch only the missing tail when the cache already covers the head.

        "Today" comes from the injected clock, never ``date.today()``. The
        latter reads the host's local zone -- in the container that is
        Asia/Jerusalem while the market day is New York -- so on either side of
        midnight the two disagree and the tail is fetched for the wrong day.
        """
        if force_refresh or cached.empty:
            return start
        first_cached = cached.index[0].date()
        last_cached = cached.index[-1].date()
        if first_cached > start:
            return start  # a hole at the front: refetch the whole window
        # Re-fetch the final cached day too, so a late correction is picked up.
        return min(last_cached, self.clock_date()) if last_cached >= start else start

    @staticmethod
    def _slice(
        frame: OHLCVFrame, start: dt.date, end: dt.date, as_of: dt.date | None
    ) -> OHLCVFrame:
        """Restrict to the requested window, enforcing the look-ahead cutoff."""
        if frame.empty:
            return frame
        lo = pd.Timestamp(dt.datetime.combine(start, dt.time.min, tzinfo=UTC))
        hi = pd.Timestamp(dt.datetime.combine(end, dt.time.max, tzinfo=UTC))
        if as_of is not None:
            cutoff = pd.Timestamp(dt.datetime.combine(as_of, dt.time.min, tzinfo=UTC))
            hi = min(hi, cutoff - pd.Timedelta(microseconds=1))
        return frame.loc[(frame.index >= lo) & (frame.index <= hi)]
