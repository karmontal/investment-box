"""Momentum breakout.

Buy when price closes above its prior N-day high, with volume confirming.

Included mainly as a comparison. Breakout systems are the classic trend-
following entry, but they have two properties that fit this account badly:
they trade often, and they produce many small losses punctuated by rare large
wins. At ~$100 positions with ~18 bps of round-trip friction, and a two-
trading-day minimum hold that blocks fast exits, the small losses arrive on
schedule while the large wins need a holding period this application does not
permit.

It is implemented honestly rather than tuned until it looks good, and the
backtest report reflects whatever it actually does.
"""

from __future__ import annotations

from dataclasses import dataclass

from investment_box.features.pipeline import feature_value
from investment_box.strategies.base import (
    Signal,
    Strategy,
    StrategyContext,
    StrategyDecision,
)


@dataclass
class BreakoutConfig:
    lookback: int = 20
    #: Require volume above its 20-day mean by this many standard deviations.
    #: A breakout on thin volume is usually noise.
    min_volume_zscore: float = 0.5
    max_positions: int = 2
    max_total_weight: float = 0.90
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    suggested_holding_days: int = 5
    #: Skip a breakout that is already extended: buying 8% above the 20-day
    #: average is buying the move, not the breakout.
    max_extension: float = 0.08


class MomentumBreakout(Strategy):
    """Long on a close above the prior N-day high, volume-confirmed."""

    name = "momentum_breakout"
    description = "Buy closes above the prior 20-day high with volume confirmation"

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        self.config = config or BreakoutConfig()
        self.warmup_bars = max(60, self.config.lookback + 30)

    def decide(self, context: StrategyContext) -> StrategyDecision:
        regime = context.regime
        if regime is not None and not regime.allows_new_entries:
            return self.flat(
                context.as_of, f"regime blocks new entries: {regime.reason}", regime
            )

        candidates: list[tuple[str, float, str]] = []
        for symbol in context.features:
            verdict = self._evaluate(context, symbol)
            if verdict is not None:
                candidates.append(verdict)

        if not candidates:
            return self.flat(context.as_of, "no symbol broke out today", regime)

        candidates.sort(key=lambda item: item[1], reverse=True)
        selected = candidates[: self.config.max_positions]
        weight = self.config.max_total_weight / len(selected)

        signals = [
            Signal(
                symbol=symbol,
                target_weight=weight,
                score=strength,
                reason=reason,
                stop_atr_mult=self.config.stop_atr_mult,
                take_profit_atr_mult=self.config.take_profit_atr_mult,
                suggested_holding_days=self.config.suggested_holding_days,
            )
            for symbol, strength, reason in selected
        ]

        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=tuple(self.validate_weights(signals)),
            rationale=f"{len(selected)} breakout(s): "
            + ", ".join(s for s, _, _ in selected),
            regime=regime,
        )

    def _evaluate(self, context: StrategyContext, symbol: str) -> tuple[str, float, str] | None:
        row = context.row(symbol)
        if row is None:
            return None

        close = feature_value(row, "close")
        prior_high = feature_value(row, f"high_{self.config.lookback}d")
        volume_z = feature_value(row, "volume_zscore_20")
        extension = feature_value(row, "ma_distance_20")

        if close is None or prior_high is None or volume_z is None or extension is None:
            return None

        if close <= prior_high:
            return None
        if volume_z < self.config.min_volume_zscore:
            return None
        if extension > self.config.max_extension:
            return None

        strength = float(close / prior_high - 1.0)
        return (
            symbol,
            strength,
            (
                f"closed {strength:+.2%} above the prior {self.config.lookback}-day high "
                f"on volume {volume_z:+.1f} sd"
            ),
        )
