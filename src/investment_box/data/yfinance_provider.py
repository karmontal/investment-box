"""yfinance data provider -- research and backtesting only.

Explicitly *not* for execution. yfinance scrapes an undocumented endpoint: it
has no SLA, changes shape without notice, and occasionally returns silently
wrong data. It is good enough for building and testing strategies, and it is
the only free source with enough history for this universe. Live prices and
fills come from the broker.
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from investment_box.core.errors import DataError
from investment_box.core.logging import get_logger
from investment_box.data.base import INDEX_NAME, OHLCVFrame, empty_frame

log = get_logger(__name__)


class YFinanceProvider:
    """Adjusted daily bars from Yahoo Finance."""

    name = "yfinance"
    research_only = True

    def __init__(self, *, max_attempts: int = 3) -> None:
        self._max_attempts = max_attempts

    def is_available(self) -> bool:
        """True if the package imports. Network reachability is not probed here."""
        try:
            import yfinance  # noqa: F401
        except ImportError:
            return False
        return True

    def get_bars(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        timeframe: str = "1d",
    ) -> OHLCVFrame:
        if timeframe != "1d":
            raise DataError(f"{self.name} provider supports only '1d' bars, got {timeframe!r}")
        if end < start:
            raise DataError(f"end {end} precedes start {start}")

        raw = self._download(symbol, start, end)
        if raw is None or raw.empty:
            log.info("yfinance.no_data", symbol=symbol, start=str(start), end=str(end))
            return empty_frame()
        return self._normalise(raw, symbol)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _download(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
        import yfinance as yf

        # yfinance treats `end` as exclusive, so add a day to make it inclusive.
        frame: pd.DataFrame | None = yf.download(
            tickers=symbol,
            start=start.isoformat(),
            end=(end + dt.timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=True,  # split- and dividend-adjusted OHLC
            actions=False,
            progress=False,
            threads=False,
            group_by="column",
        )
        return frame

    @staticmethod
    def _normalise(raw: pd.DataFrame, symbol: str) -> OHLCVFrame:
        """Reshape yfinance output into the canonical frame.

        yfinance returns a MultiIndex column frame when given a ticker list and
        a flat one otherwise -- and has changed which, between versions, for a
        single ticker. Both shapes are handled rather than assumed.
        """
        frame = raw.copy()

        if isinstance(frame.columns, pd.MultiIndex):
            level_values = frame.columns.get_level_values(-1)
            if symbol.upper() in {str(v).upper() for v in level_values}:
                # .xs is typed as possibly returning a Series; on a column
                # cross-section of a 2D frame it is always a DataFrame.
                frame = cast(pd.DataFrame, frame.xs(symbol, axis=1, level=-1, drop_level=True))
            else:
                frame.columns = frame.columns.get_level_values(0)

        frame.columns = [str(col).strip().lower().replace(" ", "_") for col in frame.columns]

        if "adj_close" in frame.columns and "close" not in frame.columns:
            frame = frame.rename(columns={"adj_close": "close"})

        wanted = ["open", "high", "low", "close", "volume"]
        missing = [col for col in wanted if col not in frame.columns]
        if missing:
            raise DataError(
                f"{symbol}: yfinance returned unexpected columns "
                f"{list(frame.columns)}; missing {missing}"
            )

        frame = frame[wanted].astype("float64")
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame.index.name = INDEX_NAME
        return frame.sort_index()
