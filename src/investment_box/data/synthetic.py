"""Deterministic synthetic bars.

Exists so that the whole application -- tests, the dashboard, a first run --
works with no network and no credentials. Given the same symbol and date range
it always produces the same series, which makes test failures reproducible.

These are **not** real prices. Anything that consumes them is tagged so a
backtest run on synthetic data can never be mistaken for a real one.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from investment_box.core.clock import TradingCalendar
from investment_box.data.base import INDEX_NAME, OHLCVFrame, empty_frame


class SyntheticDataProvider:
    """Geometric-random-walk bars, seeded from the symbol name."""

    name = "synthetic"
    research_only = True
    is_synthetic = True

    def __init__(
        self,
        *,
        annual_drift: float = 0.07,
        annual_volatility: float = 0.18,
        base_price: float = 50.0,
        base_volume: float = 1_500_000.0,
    ) -> None:
        self.annual_drift = annual_drift
        self.annual_volatility = annual_volatility
        self.base_price = base_price
        self.base_volume = base_volume

    def is_available(self) -> bool:
        return True

    @staticmethod
    def _seed(symbol: str) -> int:
        """Stable per-symbol seed.

        ``hash()`` is randomised per process, so it would make tests flaky;
        a digest is stable across runs and machines.
        """
        digest = hashlib.sha256(symbol.upper().encode()).digest()
        return int.from_bytes(digest[:4], "big")

    def get_bars(
        self,
        symbol: str,
        start: dt.date,
        end: dt.date,
        timeframe: str = "1d",
    ) -> OHLCVFrame:
        if end < start:
            return empty_frame()

        calendar = TradingCalendar(anchor=start + (end - start) / 2)
        sessions = [day for day in calendar.sessions if start <= day <= end]
        if not sessions:
            return empty_frame()

        rng = np.random.default_rng(self._seed(symbol))
        n = len(sessions)

        daily_drift = self.annual_drift / 252.0
        daily_vol = self.annual_volatility / np.sqrt(252.0)

        # Symbol-dependent starting price so the universe is not all identical.
        start_price = self.base_price * (0.5 + (self._seed(symbol) % 1000) / 500.0)

        shocks = rng.normal(daily_drift, daily_vol, size=n)
        closes = start_price * np.exp(np.cumsum(shocks))

        # Opens gap slightly from the prior close; the first open equals its close.
        gaps = rng.normal(0.0, daily_vol * 0.3, size=n)
        opens = np.empty(n)
        opens[0] = closes[0]
        opens[1:] = closes[:-1] * np.exp(gaps[1:])

        intraday = np.abs(rng.normal(0.0, daily_vol * 0.7, size=n))
        highs = np.maximum(opens, closes) * (1.0 + intraday)
        lows = np.minimum(opens, closes) * (1.0 - intraday)

        volumes = np.abs(rng.lognormal(mean=np.log(self.base_volume), sigma=0.35, size=n))

        index = pd.DatetimeIndex(
            [dt.datetime.combine(day, dt.time.min, tzinfo=dt.UTC) for day in sessions],
            name=INDEX_NAME,
        )
        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": np.round(volumes),
            },
            index=index,
        ).astype("float64")
