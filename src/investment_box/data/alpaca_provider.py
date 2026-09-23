"""Alpaca market-data provider.

Kept behind the same protocol as yfinance so the switch is a config change.
Requires credentials and the ``broker`` extra (``uv sync --extra broker``); with
neither, :meth:`is_available` returns ``False`` and the repository falls back to
the configured research provider rather than failing.

Phase 1 ships the adapter and its shape handling. It is exercised against the
live API in Phase 5, when execution arrives -- until then nothing in the engine
depends on it.
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import pandas as pd

from investment_box.core.errors import DataError
from investment_box.core.logging import get_logger
from investment_box.data.base import INDEX_NAME, OHLCVFrame, empty_frame

log = get_logger(__name__)


class AlpacaDataProvider:
    """Adjusted daily bars from Alpaca's market-data API."""

    name = "alpaca"
    research_only = False

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._client: object | None = None

    def is_available(self) -> bool:
        if not (self._api_key and self._secret_key):
            return False
        try:
            import alpaca  # noqa: F401
        except ImportError:
            log.info("alpaca.package_missing", hint="uv sync --extra broker")
            return False
        return True

    def _get_client(self) -> object:
        if self._client is None:
            if not self.is_available():
                raise DataError(
                    "Alpaca provider unavailable: set ALPACA_API_KEY and "
                    "ALPACA_SECRET_KEY, and install the broker extra."
                )
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self._api_key, self._secret_key)
        return self._client

    def get_bars(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        timeframe: str = "1d",
    ) -> OHLCVFrame:
        if timeframe != "1d":
            raise DataError(f"{self.name} provider supports only '1d' bars, got {timeframe!r}")

        from alpaca.data.enums import Adjustment
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client = self._get_client()
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=dt.datetime.combine(start, dt.time.min, tzinfo=dt.UTC),
            end=dt.datetime.combine(end, dt.time.max, tzinfo=dt.UTC),
            adjustment=Adjustment.ALL,  # splits and dividends
        )
        response = client.get_stock_bars(request)  # type: ignore[attr-defined]
        frame = response.df
        if frame is None or frame.empty:
            return empty_frame()
        return self._normalise(frame, symbol)

    @staticmethod
    def _normalise(raw: pd.DataFrame, symbol: str) -> OHLCVFrame:
        """Alpaca returns a (symbol, timestamp) MultiIndex; flatten it."""
        frame = raw.copy()
        if isinstance(frame.index, pd.MultiIndex):
            frame = cast(
                pd.DataFrame,
                frame.xs(symbol, level="symbol")
                if "symbol" in frame.index.names
                else frame.droplevel(0),
            )
        frame.columns = [str(col).strip().lower() for col in frame.columns]

        wanted = ["open", "high", "low", "close", "volume"]
        missing = [col for col in wanted if col not in frame.columns]
        if missing:
            raise DataError(f"{symbol}: Alpaca response missing columns {missing}")

        frame = frame[wanted].astype("float64")
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = INDEX_NAME
        return frame.sort_index()
