"""The data provider contract.

Every provider returns the same shape, so swapping yfinance for Alpaca (or a
synthetic generator in tests) changes nothing downstream.

``OHLCVFrame`` is a pandas DataFrame with:

* a ``DatetimeIndex`` named ``date``, timezone-aware UTC, ascending, unique
* columns exactly ``open, high, low, close, volume``, all float
* prices already adjusted for splits and dividends

The adjustment point matters: momentum and mean-reversion signals computed on
unadjusted prices see a 2-for-1 split as a -50% return. Providers adjust, and
:func:`~investment_box.data.clean.validate_ohlcv` checks for the tell-tale
signature of unadjusted data.
"""

from __future__ import annotations

import datetime as dt
from typing import Protocol, runtime_checkable

import pandas as pd

OHLCVFrame = pd.DataFrame

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
INDEX_NAME = "date"


@runtime_checkable
class DataProvider(Protocol):
    """Source of historical bars."""

    name: str

    def get_bars(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        timeframe: str = "1d",
    ) -> OHLCVFrame:
        """Return adjusted OHLCV bars for ``symbol`` over ``[start, end]`` inclusive.

        Returns an empty, correctly-shaped frame when the symbol has no data in
        the window; raises only on transport or data-integrity failures.
        """
        ...

    def is_available(self) -> bool:
        """Whether this provider can currently serve requests (credentials, network)."""
        ...


def empty_frame() -> OHLCVFrame:
    """A correctly-typed empty OHLCV frame."""
    frame = pd.DataFrame(
        {
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
        }
    )
    frame.index = pd.DatetimeIndex([], tz="UTC", name=INDEX_NAME)
    return frame
