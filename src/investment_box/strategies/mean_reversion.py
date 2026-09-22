"""Mean reversion on RSI and Bollinger bands.

Buy oversold conditions in an instrument that is still in an uptrend, and exit
on reversion to the middle band.

The trend filter is not optional decoration. Mean reversion without one is
"buy things that are falling", which works until it doesn't and then loses a
large fraction of the position at once. Requiring price above its 200-day
average restricts the strategy to dips within uptrends, which is the version
with any evidence behind it.

For this account specifically: mean reversion wants to exit quickly, often
within one to three days. The two-trading-day minimum hold and T+1 settlement
both cut against that, so expect the realised version to underperform the
theoretical one. That gap is reported rather than hidden.
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
class MeanReversionConfig:
    rsi_oversold: float = 30.0
    #: Exit when RSI recovers past this. The risk manager also applies the ATR
    #: stop, whichever triggers first.
    rsi_exit: float = 55.0
    #: Only buy in the lower part of the Bollinger channel.
    max_bollinger_position: float = 0.20
    #: Require price above its 200-day average: dips within uptrends only.
    require_uptrend: bool = True
    max_positions: int = 2
    max_total_weight: float = 0.90
    #: Wider than the breakout stop: mean reversion buys into weakness, so a
    #: tight stop is hit by the noise the strategy is trying to exploit.
    stop_atr_mult: float = 2.5
    take_profit_atr_mult: float = 2.0
    suggested_holding_days: int = 3


class MeanReversion(Strategy):
    """Buy oversold dips within uptrends."""

    name = "mean_reversion"
    description = "Buy RSI-oversold dips in the lower Bollinger band, uptrend only"

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        self.config = config or MeanReversionConfig()
        self.warmup_bars = 220 if self.config.require_uptrend else 60

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
            return self.flat(context.as_of, "nothing oversold within an uptrend", regime)

        # Most oversold first.
        candidates.sort(key=lambda item: item[1])
        selected = candidates[: self.config.max_positions]
        weight = self.config.max_total_weight / len(selected)

        signals = [
            Signal(
                symbol=symbol,
                target_weight=weight,
                score=-rsi_value,  # higher score = more oversold, for comparability
                reason=reason,
                stop_atr_mult=self.config.stop_atr_mult,
                take_profit_atr_mult=self.config.take_profit_atr_mult,
                suggested_holding_days=self.config.suggested_holding_days,
                metadata={"rsi": rsi_value, "exit_rsi": self.config.rsi_exit},
            )
            for symbol, rsi_value, reason in selected
        ]

        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=tuple(self.validate_weights(signals)),
            rationale=f"{len(selected)} oversold: " + ", ".join(s for s, _, _ in selected),
            regime=regime,
        )

    def _evaluate(self, context: StrategyContext, symbol: str) -> tuple[str, float, str] | None:
        row = context.row(symbol)
        if row is None:
            return None

        rsi_value = feature_value(row, "rsi_14")
        bollinger = feature_value(row, "bollinger_position")
        trend = feature_value(row, "ma_distance_200")

        if rsi_value is None or bollinger is None:
            return None
        if self.config.require_uptrend and trend is None:
            return None

        if rsi_value > self.config.rsi_oversold:
            return None
        if bollinger > self.config.max_bollinger_position:
            return None
        if self.config.require_uptrend and trend is not None and trend <= 0:
            return None

        trend_note = (
            f", {trend:+.1%} vs 200d"
            if self.config.require_uptrend and trend is not None
            else ""
        )
        return (
            symbol,
            rsi_value,
            f"RSI {rsi_value:.1f} oversold, "
            f"{bollinger:.0%} of the Bollinger range{trend_note}",
        )
