"""Strategy behaviour.

The invariants that matter most: no strategy can ever request leverage, short,
or act on data it should not have. Those are tested directly rather than
inferred from backtest output.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from investment_box.core.types import Side
from investment_box.features.pipeline import build_features
from investment_box.features.regime import MarketRegime, RegimeState
from investment_box.strategies import STRATEGY_REGISTRY
from investment_box.strategies.base import Signal, StrategyContext
from investment_box.strategies.defensive_core import DefensiveCore, DefensiveCoreConfig
from investment_box.strategies.etf_momentum_rotation import (
    ETFMomentumRotation,
    MomentumRotationConfig,
)
from investment_box.strategies.mean_reversion import MeanReversion
from investment_box.strategies.momentum_breakout import MomentumBreakout

AS_OF = dt.date(2024, 6, 12)


def make_bars(trend: float, volatility: float, periods: int = 400, seed: int = 1) -> pd.DataFrame:
    """Deterministic bars with a chosen drift and volatility."""
    rng = np.random.default_rng(seed)
    shocks = rng.normal(trend, volatility, periods)
    close = 100 * np.exp(np.cumsum(shocks))
    index = pd.date_range(end="2024-06-12", periods=periods, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(np.log(1e6), 0.3, periods),
        },
        index=index,
    )


def context(
    bars: dict[str, pd.DataFrame], regime: RegimeState | None = None, **kwargs
) -> StrategyContext:
    return StrategyContext(
        as_of=AS_OF,
        features={s: build_features(f, s) for s, f in bars.items()},
        regime=regime,
        **kwargs,
    )


def regime(kind: MarketRegime) -> RegimeState:
    return RegimeState(as_of=AS_OF, regime=kind, reason="test")


class TestSignalInvariants:
    def test_weight_above_one_rejected(self) -> None:
        """A weight above 1 would imply leverage, which is never permitted."""
        with pytest.raises(ValueError, match="leverage"):
            Signal(symbol="SPUS", target_weight=1.5)

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Signal(symbol="SPUS", target_weight=-0.1)

    def test_short_side_rejected(self) -> None:
        with pytest.raises(ValueError, match="only long"):
            Signal(symbol="SPUS", target_weight=0.5, side=Side.SELL)


class TestLeverageGuard:
    def test_combined_weights_above_one_rejected(self) -> None:
        strategy = ETFMomentumRotation()
        over = [Signal(symbol="A", target_weight=0.6), Signal(symbol="B", target_weight=0.6)]
        with pytest.raises(ValueError, match="cash-only"):
            strategy.validate_weights(over)

    @pytest.mark.parametrize("name", list(STRATEGY_REGISTRY))
    def test_no_strategy_exceeds_full_investment(self, name: str) -> None:
        bars = {
            "SPUS": make_bars(0.0008, 0.011, seed=1),
            "HLAL": make_bars(0.0006, 0.013, seed=2),
            "SPSK": make_bars(0.0002, 0.004, seed=3),
        }
        decision = STRATEGY_REGISTRY[name]().decide(context(bars, regime(MarketRegime.RISK_ON)))
        assert decision.total_weight <= 1.0


class TestMomentumRotation:
    def test_picks_the_strongest_risk_adjusted(self) -> None:
        bars = {
            "STRONG": make_bars(0.0015, 0.008, seed=10),
            "WEAK": make_bars(-0.0005, 0.010, seed=11),
            "SPSK": make_bars(0.0002, 0.004, seed=12),
        }
        config = MomentumRotationConfig(top_n=1, defensive_symbol="SPSK")
        decision = ETFMomentumRotation(config).decide(
            context(bars, regime(MarketRegime.RISK_ON))
        )
        assert "STRONG" in decision.target_weights
        assert "WEAK" not in decision.target_weights

    def test_defensive_symbol_never_competes_for_a_slot(self) -> None:
        """A low-volatility sukuk fund would win a risk-adjusted ranking in calm
        markets purely on its low denominator."""
        bars = {
            "SPUS": make_bars(0.0008, 0.012, seed=20),
            "SPSK": make_bars(0.0006, 0.002, seed=21),  # high score, low vol
        }
        decision = ETFMomentumRotation().decide(context(bars, regime(MarketRegime.RISK_ON)))
        assert "SPSK" not in decision.target_weights

    def test_rotates_defensive_when_regime_is_risk_off(self) -> None:
        bars = {
            "SPUS": make_bars(0.0010, 0.010, seed=30),
            "SPSK": make_bars(0.0002, 0.004, seed=31),
        }
        decision = ETFMomentumRotation().decide(context(bars, regime(MarketRegime.RISK_OFF)))
        assert decision.target_weights == {"SPSK": pytest.approx(0.90)}
        assert "defensive" in decision.rationale

    def test_holds_nothing_when_regime_is_unknown(self) -> None:
        """Not knowing the regime is not the same as the regime being fine."""
        bars = {"SPUS": make_bars(0.0010, 0.010, seed=40)}
        decision = ETFMomentumRotation().decide(context(bars, regime(MarketRegime.UNKNOWN)))
        assert decision.is_flat
        assert "unknown" in decision.rationale

    def test_all_negative_momentum_goes_defensive(self) -> None:
        # Strictly declining, so the premise is guaranteed rather than assumed:
        # a negative-drift random walk can still show positive 126-day momentum.
        def declining(seed: int) -> pd.DataFrame:
            frame = make_bars(0.0, 0.004, seed=seed)
            decay = np.exp(np.linspace(0, -0.5, len(frame)))
            for column in ("open", "high", "low", "close"):
                frame[column] = frame[column] * decay
            return frame

        bars = {
            "A": declining(50),
            "B": declining(51),
            "SPSK": make_bars(0.0002, 0.004, seed=52),
        }
        decision = ETFMomentumRotation().decide(context(bars, regime(MarketRegime.RISK_ON)))
        assert "SPSK" in decision.target_weights

    def test_holds_cash_when_defensive_symbol_is_unavailable(self) -> None:
        bars = {"A": make_bars(-0.0010, 0.010, seed=60)}
        decision = ETFMomentumRotation().decide(context(bars, regime(MarketRegime.RISK_ON)))
        assert decision.is_flat
        assert "cash" in decision.rationale

    def test_insufficient_history_produces_no_signal(self) -> None:
        bars = {"SHORT": make_bars(0.001, 0.01, periods=40, seed=70)}
        decision = ETFMomentumRotation().decide(context(bars, regime(MarketRegime.RISK_ON)))
        assert decision.is_flat

    def test_respects_top_n(self) -> None:
        bars = {f"S{i}": make_bars(0.001 * (5 - i), 0.010, seed=80 + i) for i in range(5)}
        config = MomentumRotationConfig(top_n=2, defensive_symbol=None)
        decision = ETFMomentumRotation(config).decide(
            context(bars, regime(MarketRegime.RISK_ON))
        )
        assert len(decision.target_weights) == 2

    def test_hysteresis_keeps_an_incumbent_on_a_near_tie(self) -> None:
        """Churning between two effectively tied funds pays spread twice."""
        base = make_bars(0.0010, 0.010, seed=90)
        nearly_identical = base.copy() * 1.001  # same path, negligibly better
        bars = {"HELD": base, "CHALLENGER": nearly_identical}
        config = MomentumRotationConfig(top_n=1, defensive_symbol=None, replacement_margin=0.10)
        decision = ETFMomentumRotation(config).decide(
            context(bars, regime(MarketRegime.RISK_ON), current_holdings=("HELD",))
        )
        assert "HELD" in decision.target_weights


class TestBreakout:
    def test_no_breakout_means_no_signal(self) -> None:
        flat = make_bars(0.0, 0.002, seed=100)
        decision = MomentumBreakout().decide(
            context({"FLAT": flat}, regime(MarketRegime.RISK_ON))
        )
        assert decision.is_flat

    def test_regime_blocks_entries(self) -> None:
        bars = {"SPUS": make_bars(0.001, 0.01, seed=101)}
        decision = MomentumBreakout().decide(context(bars, regime(MarketRegime.RISK_OFF)))
        assert decision.is_flat
        assert "regime" in decision.rationale


class TestMeanReversion:
    def test_regime_blocks_entries(self) -> None:
        bars = {"SPUS": make_bars(0.0005, 0.012, seed=110)}
        decision = MeanReversion().decide(context(bars, regime(MarketRegime.RISK_OFF)))
        assert decision.is_flat

    def test_nothing_oversold_means_no_signal(self) -> None:
        rising = make_bars(0.002, 0.004, seed=111)
        decision = MeanReversion().decide(
            context({"RISING": rising}, regime(MarketRegime.RISK_ON))
        )
        assert decision.is_flat


class TestContextIsolation:
    def test_context_row_never_looks_forward(self) -> None:
        """StrategyContext.at returns the last bar at or before as_of."""
        bars = make_bars(0.001, 0.01, seed=120)
        features = build_features(bars, "SPUS")
        ctx = StrategyContext(as_of=dt.date(2024, 5, 1), features={"SPUS": features})
        row = ctx.row("SPUS")
        assert row is not None
        assert row.name <= pd.Timestamp("2024-05-01", tz="UTC")

    def test_missing_symbol_returns_none(self) -> None:
        ctx = StrategyContext(as_of=AS_OF, features={})
        assert ctx.row("NOPE") is None


class TestDefensiveCore:
    """The strategy built to answer the measured problem: turnover.

    The rotation strategy paid 56% of gross profit in costs across 230 trades.
    This one is designed for a handful of trades a year, so the tests that
    matter are about *not* trading: the hysteresis band, and refusing to rotate
    into a defensive asset that is itself falling.
    """

    @staticmethod
    def _bars(core_trend: float, defensive_trend: float) -> dict[str, pd.DataFrame]:
        return {
            "SPUS": make_bars(trend=core_trend, volatility=0.004, seed=3),
            "SPSK": make_bars(trend=defensive_trend, volatility=0.002, seed=7),
        }

    def test_it_holds_the_core_in_a_clear_uptrend(self) -> None:
        strategy = DefensiveCore()
        decision = strategy.decide(context(self._bars(0.0012, 0.0002)))

        assert decision.target_weights, decision.rationale
        assert set(decision.target_weights) == {"SPUS"}
        assert "above its 200-day average" in decision.rationale

    def test_it_rotates_to_sukuk_when_the_core_breaks_and_sukuk_holds(self) -> None:
        strategy = DefensiveCore()
        decision = strategy.decide(self._downtrend_context(defensive_trend=0.0006))

        assert set(decision.target_weights) == {"SPSK"}
        assert "rotating" in decision.rationale

    def test_it_holds_cash_when_the_defensive_asset_is_also_falling(self) -> None:
        """2022: rate rises took sukuk down with equities. A rule that rotates
        into a falling 'defensive' asset is a different way to lose."""
        strategy = DefensiveCore()
        decision = strategy.decide(self._downtrend_context(defensive_trend=-0.0010))

        assert decision.is_flat
        assert "also below its 200-day average" in decision.rationale

    def test_the_guard_can_be_turned_off(self) -> None:
        strategy = DefensiveCore(DefensiveCoreConfig(require_defensive_uptrend=False))
        decision = strategy.decide(self._downtrend_context(defensive_trend=-0.0010))
        assert set(decision.target_weights) == {"SPSK"}

    def _downtrend_context(self, defensive_trend: float) -> StrategyContext:
        return context(self._bars(-0.0012, defensive_trend))

    # ------------------------------------------------------------ hysteresis

    def test_inside_the_band_an_existing_holding_is_kept_untouched(self) -> None:
        """The whole point. A single threshold at zero churns on every cross,
        and at this account size those trades cost more than the signal."""
        flat_bars = {
            "SPUS": make_bars(trend=0.0, volatility=0.0005, seed=11),
            "SPSK": make_bars(trend=0.0, volatility=0.0005, seed=12),
        }
        strategy = DefensiveCore(DefensiveCoreConfig(entry_band=0.50, exit_band=0.50))

        decision = strategy.decide(context(flat_bars, current_holdings=("SPUS",)))
        assert set(decision.target_weights) == {"SPUS"}
        assert "inside the" in decision.rationale
        assert "unchanged" in decision.rationale

    def test_inside_the_band_holding_nothing_stays_holding_nothing(self) -> None:
        flat_bars = {
            "SPUS": make_bars(trend=0.0, volatility=0.0005, seed=11),
            "SPSK": make_bars(trend=0.0, volatility=0.0005, seed=12),
        }
        strategy = DefensiveCore(DefensiveCoreConfig(entry_band=0.50, exit_band=0.50))

        decision = strategy.decide(context(flat_bars))
        assert decision.is_flat
        assert "neither" in decision.rationale

    def test_a_wider_band_never_produces_more_signals_than_a_narrow_one(self) -> None:
        """Monotonicity: widening the band must not increase trading."""
        bars = self._bars(0.0002, 0.0001)
        narrow = DefensiveCore(DefensiveCoreConfig(entry_band=0.0, exit_band=0.0))
        wide = DefensiveCore(DefensiveCoreConfig(entry_band=0.60, exit_band=0.60))

        narrow_decision = narrow.decide(context(bars))
        wide_decision = wide.decide(context(bars))

        assert len(wide_decision.signals) <= len(narrow_decision.signals) or (
            wide_decision.is_flat
        )

    # ------------------------------------------------------------- safety

    def test_missing_core_data_holds_nothing_rather_than_assuming_an_uptrend(
        self,
    ) -> None:
        strategy = DefensiveCore()
        decision = strategy.decide(context({"SPSK": make_bars(0.001, 0.002, seed=5)}))

        assert decision.is_flat
        assert "holding nothing rather than assuming an uptrend" in decision.rationale

    def test_missing_defensive_data_holds_cash_rather_than_an_unmeasured_asset(
        self,
    ) -> None:
        strategy = DefensiveCore()
        decision = strategy.decide(
            context({"SPUS": make_bars(-0.0012, 0.004, seed=3)})
        )

        assert decision.is_flat
        assert "unmeasured asset" in decision.rationale

    def test_it_never_holds_both_sleeves_at_once(self) -> None:
        strategy = DefensiveCore()
        for core, defensive in ((0.0012, 0.0006), (-0.0012, 0.0006), (0.0, 0.0)):
            decision = strategy.decide(context(self._bars(core, defensive)))
            assert len(decision.target_weights) <= 1, decision.rationale

    def test_weights_never_imply_leverage(self) -> None:
        strategy = DefensiveCore(DefensiveCoreConfig(target_weight=1.0))
        decision = strategy.decide(context(self._bars(0.0012, 0.0002)))
        assert decision.total_weight <= 1.0

    def test_a_negative_band_is_refused_at_construction(self) -> None:
        """A negative band inverts the rule into buy-high-sell-low."""
        with pytest.raises(ValueError, match="non-negative"):
            DefensiveCore(DefensiveCoreConfig(entry_band=-0.01))

    def test_every_decision_carries_a_reason(self) -> None:
        strategy = DefensiveCore()
        for core, defensive in ((0.0012, 0.0002), (-0.0012, 0.0006), (-0.0012, -0.001)):
            decision = strategy.decide(context(self._bars(core, defensive)))
            assert decision.rationale, "a silent decision cannot be audited"
