"""Technical indicators.

Every function here obeys one rule: **the value at index ``i`` uses only data
at indices ``<= i``**. No centred windows, no ``shift(-1)``, no ``bfill``.
A single lookahead here would flow into every strategy and every backtest, and
would show up as a suspiciously good result rather than as an error.

Warm-up periods are returned as ``NaN`` rather than filled. A 14-day RSI does
not exist on day 3, and inventing a value there is how a backtest ends up
trading on data it never had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Trading days per year, for annualising volatility.
TRADING_DAYS = 252


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Simple percentage return over ``periods`` bars."""
    return close.pct_change(periods=periods)


def log_returns(close: pd.Series) -> pd.Series:
    return pd.Series(np.log(close / close.shift(1)), index=close.index)


def momentum(close: pd.Series, lookback: int) -> pd.Series:
    """Total return over ``lookback`` bars.

    ``close[i] / close[i - lookback] - 1``, so the value at ``i`` is knowable
    at the close of bar ``i``.
    """
    return close / close.shift(lookback) - 1.0


def realised_volatility(close: pd.Series, window: int = 20, annualise: bool = True) -> pd.Series:
    """Rolling standard deviation of daily returns.

    Uses a trailing window ending at the current bar.
    """
    daily = close.pct_change()
    vol = daily.rolling(window=window, min_periods=window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualise else vol


def risk_adjusted_momentum(close: pd.Series, lookback: int, vol_window: int = 20) -> pd.Series:
    """Momentum divided by volatility.

    The ranking signal for the rotation strategy. Raw momentum favours whatever
    is most volatile; dividing by volatility asks which fund delivered the most
    return per unit of risk, which is the question a small account actually
    cares about.
    """
    mom = momentum(close, lookback)
    vol = realised_volatility(close, vol_window, annualise=True)
    # A zero or missing vol gives NaN rather than an infinite score.
    return mom / vol.replace(0.0, np.nan)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI.

    Wilder's smoothing (``ewm(alpha=1/window)``) rather than a simple rolling
    mean, which is what the standard definition uses and what every charting
    package shows.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # All-gain windows give an infinite RS, which is RSI 100 by definition.
    return out.where(avg_loss != 0.0, 100.0).where(avg_gain.notna())


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range: the largest of the three standard spans."""
    previous_close = close.shift(1)
    return pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average true range, Wilder-smoothed.

    The position sizer's input: stop distance is a multiple of ATR, so that a
    calm fund gets a tighter stop and a volatile one a wider stop for the same
    dollar risk.
    """
    return true_range(high, low, close).ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()


def bollinger(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger middle, upper and lower bands."""
    middle = close.rolling(window=window, min_periods=window).mean()
    spread = close.rolling(window=window, min_periods=window).std() * num_std
    return middle, middle + spread, middle - spread


def bollinger_position(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """Where price sits in the band: 0 at the lower band, 1 at the upper."""
    _, upper, lower = bollinger(close, window, num_std)
    width = (upper - lower).replace(0.0, np.nan)
    return (close - lower) / width


def moving_average_distance(close: pd.Series, window: int) -> pd.Series:
    """Distance from a moving average, as a fraction of the average."""
    ma = close.rolling(window=window, min_periods=window).mean()
    return close / ma - 1.0


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """How unusual today's volume is, in trailing standard deviations."""
    mean = volume.rolling(window=window, min_periods=window).mean()
    std = volume.rolling(window=window, min_periods=window).std().replace(0.0, np.nan)
    return (volume - mean) / std


def dollar_volume(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Trailing average dollar volume, the liquidity filter's input."""
    return (close * volume).rolling(window=window, min_periods=1).mean()


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    """Highest value over the trailing window, *excluding* the current bar.

    Excluding the current bar is what makes a breakout test meaningful: "is
    today's close above the prior 20-day high?" is a signal, whereas comparing
    today against a window that contains today is true by construction roughly
    one day in twenty.
    """
    return series.shift(1).rolling(window=window, min_periods=window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    """Lowest value over the trailing window, excluding the current bar."""
    return series.shift(1).rolling(window=window, min_periods=window).min()


def drawdown(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak. Zero or negative."""
    return equity / equity.cummax() - 1.0
