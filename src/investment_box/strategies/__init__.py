"""Pluggable trading strategies.

Every strategy returns target weights, never orders, so none of them can
bypass position sizing, the cash buffer or settlement rules.
"""

from investment_box.strategies.base import (
    Signal,
    Strategy,
    StrategyContext,
    StrategyDecision,
)
from investment_box.strategies.defensive_core import DefensiveCore, DefensiveCoreConfig
from investment_box.strategies.etf_momentum_rotation import (
    ETFMomentumRotation,
    MomentumRotationConfig,
)
from investment_box.strategies.mean_reversion import MeanReversion, MeanReversionConfig
from investment_box.strategies.ml_classifier import MLClassifier, MLConfig
from investment_box.strategies.momentum_breakout import BreakoutConfig, MomentumBreakout

#: Registry, so the backtester and the dashboard can enumerate strategies
#: without importing each one.
STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    ETFMomentumRotation.name: ETFMomentumRotation,
    DefensiveCore.name: DefensiveCore,
    MomentumBreakout.name: MomentumBreakout,
    MeanReversion.name: MeanReversion,
    MLClassifier.name: MLClassifier,
}

__all__ = [
    "STRATEGY_REGISTRY",
    "BreakoutConfig",
    "DefensiveCore",
    "DefensiveCoreConfig",
    "ETFMomentumRotation",
    "MLClassifier",
    "MLConfig",
    "MeanReversion",
    "MeanReversionConfig",
    "MomentumBreakout",
    "MomentumRotationConfig",
    "Signal",
    "Strategy",
    "StrategyContext",
    "StrategyDecision",
]
