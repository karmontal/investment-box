"""Indicators, with look-ahead as the thing most under test.

The central property: the value at index ``i`` must depend only on data at
indices ``<= i``. A single leak here would flow into every strategy and show up
as a good backtest rather than as an error, so it is tested directly by
truncating the input and checking the prefix is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment_box.features import indicators as ind


@pytest.fixture
def series() -> pd.Series:
    rng = np.random.default_rng(42)
    values = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, 400)))
    index = pd.date_range("2023-01-02", periods=400, freq="B", tz="UTC")
    return pd.Series(values, index=index)


@pytest.fixture
def ohlc(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": series,
            "high": series * 1.01,
            "low": series * 0.99,
            "open": series.shift(1).fillna(series.iloc[0]),
            "volume": pd.Series(1e6, index=series.index),
        }
    )


class TestNoLookahead:
    """Truncating the input must not change any earlier value."""

    @pytest.mark.parametrize(
        "fn",
        [
            lambda s: ind.momentum(s, 21),
            lambda s: ind.realised_volatility(s, 20),
            lambda s: ind.risk_adjusted_momentum(s, 63),
            lambda s: ind.rsi(s, 14),
            lambda s: ind.moving_average_distance(s, 50),
            lambda s: ind.bollinger_position(s, 20),
            lambda s: ind.rolling_max(s, 20),
            lambda s: ind.rolling_min(s, 20),
        ],
    )
    def test_prefix_is_stable(self, series: pd.Series, fn) -> None:
        cutoff = 300
        full = fn(series).iloc[:cutoff]
        truncated = fn(series.iloc[:cutoff])
        pd.testing.assert_series_equal(full, truncated, check_names=False)

    def test_atr_prefix_is_stable(self, ohlc: pd.DataFrame) -> None:
        cutoff = 300
        full = ind.atr(ohlc["high"], ohlc["low"], ohlc["close"]).iloc[:cutoff]
        truncated = ind.atr(
            ohlc["high"].iloc[:cutoff], ohlc["low"].iloc[:cutoff], ohlc["close"].iloc[:cutoff]
        )
        pd.testing.assert_series_equal(full, truncated, check_names=False)

    def test_rolling_max_excludes_the_current_bar(self) -> None:
        """A breakout test comparing today against a window containing today is
        true by construction about one day in twenty."""
        values = pd.Series([1.0, 2.0, 3.0, 10.0, 4.0])
        result = ind.rolling_max(values, 3)
        # At index 4 the prior 3 bars are [2, 3, 10] -> 10, not including 4.
        assert result.iloc[4] == 10.0
        # The current bar's own value never appears in its own window.
        assert result.iloc[3] == 3.0


class TestWarmup:
    """Warm-up periods are NaN, never filled."""

    def test_rsi_warmup_is_nan(self, series: pd.Series) -> None:
        result = ind.rsi(series, 14)
        assert result.iloc[:13].isna().all()
        assert result.iloc[20:].notna().all()

    def test_momentum_warmup_is_nan(self, series: pd.Series) -> None:
        assert ind.momentum(series, 21).iloc[:21].isna().all()

    def test_volatility_warmup_is_nan(self, series: pd.Series) -> None:
        assert ind.realised_volatility(series, 20).iloc[:19].isna().all()


class TestCorrectness:
    def test_momentum_matches_definition(self) -> None:
        values = pd.Series([100.0, 105.0, 110.0, 121.0])
        assert ind.momentum(values, 2).iloc[2] == pytest.approx(0.10)
        assert ind.momentum(values, 3).iloc[3] == pytest.approx(0.21)

    def test_rsi_bounds(self, series: pd.Series) -> None:
        result = ind.rsi(series, 14).dropna()
        assert (result >= 0).all()
        assert (result <= 100).all()

    def test_rsi_is_100_when_only_gains(self) -> None:
        values = pd.Series(np.arange(1.0, 40.0))
        assert ind.rsi(values, 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_is_low_when_only_losses(self) -> None:
        values = pd.Series(np.arange(40.0, 1.0, -1.0))
        assert ind.rsi(values, 14).iloc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_atr_is_positive(self, ohlc: pd.DataFrame) -> None:
        result = ind.atr(ohlc["high"], ohlc["low"], ohlc["close"]).dropna()
        assert (result > 0).all()

    def test_true_range_handles_gaps(self) -> None:
        """A gap down makes |low - prev_close| the true range, not high - low."""
        high = pd.Series([100.0, 90.0])
        low = pd.Series([99.0, 89.0])
        close = pd.Series([100.0, 89.5])
        assert ind.true_range(high, low, close).iloc[1] == pytest.approx(11.0)

    def test_bollinger_position_bounds(self, series: pd.Series) -> None:
        result = ind.bollinger_position(series, 20).dropna()
        # Mostly inside the band, and finite everywhere.
        assert np.isfinite(result).all()
        assert (result.between(-1, 2)).mean() > 0.95

    def test_risk_adjusted_momentum_penalises_volatility(self) -> None:
        """Two series with the same total return but different paths."""
        index = pd.date_range("2023-01-02", periods=60, freq="B", tz="UTC")
        calm = pd.Series(np.linspace(100, 110, 60), index=index)
        rng = np.random.default_rng(1)
        choppy_path = np.linspace(100, 110, 60) * (1 + rng.normal(0, 0.03, 60))
        choppy = pd.Series(choppy_path, index=index)

        calm_score = ind.risk_adjusted_momentum(calm, 21).iloc[-1]
        choppy_score = ind.risk_adjusted_momentum(choppy, 21).iloc[-1]
        assert calm_score > choppy_score

    def test_drawdown_is_never_positive(self, series: pd.Series) -> None:
        assert (ind.drawdown(series) <= 1e-12).all()

    def test_volume_zscore_centres_on_zero(self) -> None:
        rng = np.random.default_rng(3)
        volume = pd.Series(rng.normal(1e6, 1e5, 300))
        result = ind.volume_zscore(volume, 20).dropna()
        assert abs(result.mean()) < 0.5


class TestEdgeCases:
    def test_empty_series(self) -> None:
        empty = pd.Series(dtype=float)
        assert ind.momentum(empty, 21).empty
        assert ind.rsi(empty, 14).empty

    def test_constant_series_gives_zero_volatility(self) -> None:
        flat = pd.Series([100.0] * 60)
        assert ind.realised_volatility(flat, 20).iloc[-1] == pytest.approx(0.0)

    def test_risk_adjusted_momentum_survives_zero_volatility(self) -> None:
        """Dividing by zero volatility must give NaN, not infinity."""
        flat = pd.Series([100.0] * 60)
        result = ind.risk_adjusted_momentum(flat, 21).iloc[-1]
        assert pd.isna(result) or np.isfinite(result)
