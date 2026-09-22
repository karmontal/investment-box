"""Assembling the per-symbol feature frame.

One function produces every feature a strategy can see, so that no strategy
computes its own variant of "momentum" and they can be compared on equal terms.

The frame is aligned to the input bars and preserves ``NaN`` warm-up rows.
Strategies must handle ``NaN`` rather than expect them dropped -- dropping them
here would silently shorten the sample and make early backtest dates disappear.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from investment_box.features import indicators as ind

#: Momentum lookbacks in trading days: roughly 1, 3 and 6 months.
MOMENTUM_WINDOWS: tuple[int, ...] = (21, 63, 126)


def feature_value(row: pd.Series | None, name: str) -> float | None:
    """Read one feature as a finite float, or ``None``.

    Centralises the four ways a feature can be unusable -- absent row, absent
    column, NaN during warm-up, or infinite from a divide-by-zero -- so that
    strategies express "I need this value" in one line and cannot forget one
    of the checks.
    """
    if row is None:
        return None
    value = row.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class FeatureSet:
    """Features for one symbol, plus the bars they were computed from."""

    symbol: str
    frame: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return self.frame.empty

    def latest(self) -> pd.Series | None:
        """The most recent complete row, or ``None`` if there isn't one."""
        if self.frame.empty:
            return None
        return self.frame.iloc[-1]

    def at(self, moment: pd.Timestamp) -> pd.Series | None:
        """The row at or immediately before ``moment``.

        Never looks forward: if there is no bar at ``moment`` it returns the
        last one before it, which is what a strategy actually knew.
        """
        usable = self.frame.loc[self.frame.index <= moment]
        return None if usable.empty else usable.iloc[-1]

    @property
    def usable_from(self) -> pd.Timestamp | None:
        """First date on which every feature is defined."""
        complete = self.frame.dropna()
        return None if complete.empty else complete.index[0]


def build_features(frame: pd.DataFrame, symbol: str) -> FeatureSet:
    """Compute every feature for one symbol's bars.

    Args:
        frame: Cleaned OHLCV, ascending, UTC-indexed.
        symbol: For labelling.
    """
    if frame.empty:
        return FeatureSet(symbol=symbol, frame=pd.DataFrame(index=frame.index))

    close, high, low, volume = frame["close"], frame["high"], frame["low"], frame["volume"]
    out = pd.DataFrame(index=frame.index)

    out["close"] = close
    out["return_1d"] = ind.returns(close, 1)
    out["return_5d"] = ind.returns(close, 5)

    for window in MOMENTUM_WINDOWS:
        out[f"momentum_{window}d"] = ind.momentum(close, window)
        out[f"risk_adj_momentum_{window}d"] = ind.risk_adjusted_momentum(close, window)

    out["volatility_20d"] = ind.realised_volatility(close, 20)
    out["volatility_60d"] = ind.realised_volatility(close, 60)

    out["rsi_14"] = ind.rsi(close, 14)

    macd_line, signal_line, histogram = ind.macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_histogram"] = histogram

    out["atr_14"] = ind.atr(high, low, close, 14)
    # ATR as a fraction of price: comparable across a $17 sukuk fund and a $75
    # equity fund, which the raw value is not.
    out["atr_pct"] = out["atr_14"] / close

    out["bollinger_position"] = ind.bollinger_position(close, 20, 2.0)
    out["ma_distance_20"] = ind.moving_average_distance(close, 20)
    out["ma_distance_50"] = ind.moving_average_distance(close, 50)
    out["ma_distance_200"] = ind.moving_average_distance(close, 200)

    out["volume_zscore_20"] = ind.volume_zscore(volume, 20)
    out["dollar_volume_20"] = ind.dollar_volume(close, volume, 20)

    out["high_20d"] = ind.rolling_max(close, 20)
    out["low_20d"] = ind.rolling_min(close, 20)
    out["high_55d"] = ind.rolling_max(close, 55)

    return FeatureSet(symbol=symbol, frame=out)


def build_many(frames: dict[str, pd.DataFrame]) -> dict[str, FeatureSet]:
    """Features for several symbols."""
    return {symbol: build_features(frame, symbol) for symbol, frame in frames.items()}
