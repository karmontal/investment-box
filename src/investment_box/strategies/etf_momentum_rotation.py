"""Shariah ETF momentum rotation -- the primary strategy.

Rank the compliant ETF universe by risk-adjusted momentum, hold the top one or
two, and rotate into sukuk (or cash) when the market regime turns negative.

Why this shape, for a ~$500 cash account:

* **Few positions.** Holding one or two funds keeps each position large enough
  that whole-share rounding and fixed frictions do not dominate. Spreading
  $500 across five funds produces $100 positions that round to one or two
  shares of a $60 ETF.
* **Weekly rebalance.** T+1 settlement plus a two-trading-day minimum hold
  means capital cannot turn over faster than roughly twice a month anyway.
  Rebalancing daily would generate signals the account cannot act on.
* **Risk-adjusted, not raw, momentum.** Raw momentum systematically selects
  whatever is most volatile, which at this account size means the largest
  drawdowns relative to capital.
* **A regime filter, not a prediction.** The filter does not forecast; it
  refuses to hold equity while the broad market is below its 200-day average.
  That is a crude rule and it will be wrong at turning points, but it is the
  kind of wrong that costs opportunity rather than capital.

Honest limitation: momentum rotation is a *medium*-term strategy. The
literature that supports it uses monthly rebalancing over decades. Applying it
to 1-10 day swings, on funds with two to three years of history, is well
outside the evidence base. The backtest report says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from investment_box.core.logging import get_logger
from investment_box.features.pipeline import feature_value
from investment_box.features.regime import MarketRegime
from investment_box.strategies.base import Signal, Strategy, StrategyContext, StrategyDecision

log = get_logger(__name__)


@dataclass
class MomentumRotationConfig:
    """Tunables. Defaults are deliberately round numbers, not fitted ones.

    Anything tuned to maximise a backtest on two years of data is curve-fitting,
    so these are chosen a priori: 1/3/6-month lookbacks are the standard
    academic windows, and equal weighting across them avoids asserting that one
    horizon matters more.
    """

    #: Momentum lookbacks, blended with equal weight.
    lookbacks: tuple[int, ...] = (21, 63, 126)
    #: How many funds to hold.
    top_n: int = 2
    #: Only hold a fund whose blended score is positive: being the best of a
    #: falling universe is not a reason to own something.
    min_score: float = 0.0
    #: Where to sit when the regime is defensive. ``None`` means cash.
    defensive_symbol: str | None = "SPSK"
    #: Fraction of allocated capital deployed when fully invested. The rest is
    #: the cash buffer the risk manager also enforces.
    max_total_weight: float = 0.90
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    suggested_holding_days: int = 5
    #: Hysteresis: an incumbent holding keeps its place unless a challenger
    #: beats it by this margin. Without it the portfolio churns between funds
    #: whose scores differ in the third decimal, paying spread each time -- at
    #: $100 positions that is a real cost.
    replacement_margin: float = 0.10


class ETFMomentumRotation(Strategy):
    """Hold the top-ranked compliant ETFs; rotate defensive when the regime turns."""

    name = "etf_momentum_rotation"
    description = "Rank compliant ETFs by blended risk-adjusted momentum; hold top N"

    def __init__(self, config: MomentumRotationConfig | None = None) -> None:
        self.config = config or MomentumRotationConfig()
        self.warmup_bars = max(self.config.lookbacks) + 30

    # ------------------------------------------------------------------ main

    def decide(self, context: StrategyContext) -> StrategyDecision:
        regime = context.regime

        if regime is not None and regime.regime is MarketRegime.UNKNOWN:
            # Not knowing the regime is not the same as the regime being fine.
            return self.flat(
                context.as_of,
                f"regime unknown ({regime.reason}); holding nothing rather than "
                f"assuming conditions are favourable",
                regime,
            )

        if regime is not None and regime.regime.is_defensive:
            return self._defensive(context, regime.reason)

        scores = self._rank(context)
        if not scores:
            return self.flat(
                context.as_of, "no symbol had enough history to score", regime
            )

        positive = [(s, v) for s, v in scores if v > self.config.min_score]
        if not positive:
            return self._defensive(
                context,
                f"no fund has positive risk-adjusted momentum "
                f"(best: {scores[0][0]} at {scores[0][1]:+.2f})",
            )

        selected = self._select_with_hysteresis(positive, context.current_holdings)
        signals = self._build_signals(selected, context)

        chosen = ", ".join(f"{s} ({v:+.2f})" for s, v in selected)
        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=tuple(self.validate_weights(signals)),
            rationale=f"top {len(selected)} by blended risk-adjusted momentum: {chosen}",
            regime=regime,
        )

    # --------------------------------------------------------------- ranking

    def _rank(self, context: StrategyContext) -> list[tuple[str, float]]:
        """Score every symbol, descending. Symbols without data are omitted."""
        scored: list[tuple[str, float]] = []
        for symbol in context.features:
            if symbol == self.config.defensive_symbol:
                # The defensive sleeve is a destination, not a competitor: a
                # sukuk fund would otherwise win the ranking in calm markets
                # purely on its low volatility.
                continue
            score = self._score(context, symbol)
            if score is not None:
                scored.append((symbol, score))
        return sorted(scored, key=lambda pair: pair[1], reverse=True)

    def _score(self, context: StrategyContext, symbol: str) -> float | None:
        """Blended risk-adjusted momentum, or ``None`` if incomplete.

        Every lookback must be present. Averaging over whichever windows happen
        to be available would rank a young fund on 1-month momentum against an
        old fund's 6-month figure, which is not a comparison.
        """
        row = context.row(symbol)
        if row is None:
            return None

        values: list[float] = []
        for lookback in self.config.lookbacks:
            value = feature_value(row, f"risk_adj_momentum_{lookback}d")
            if value is None:
                return None
            values.append(value)

        return float(np.mean(values)) if values else None

    def _select_with_hysteresis(
        self, ranked: list[tuple[str, float]], holdings: tuple[str, ...]
    ) -> list[tuple[str, float]]:
        """Pick the top N, favouring incumbents on near-ties.

        A challenger must beat the incumbent it would displace by
        ``replacement_margin`` in relative terms. Otherwise the portfolio pays
        a round trip to swap between two funds that are effectively tied.
        """
        top_n = self.config.top_n
        if not holdings:
            return ranked[:top_n]

        scores = dict(ranked)
        incumbents = [(s, scores[s]) for s, _ in ranked if s in holdings][:top_n]
        challengers = [(s, v) for s, v in ranked if s not in holdings]

        selected = list(incumbents)
        for symbol, score in challengers:
            if len(selected) >= top_n:
                break
            selected.append((symbol, score))

        if len(selected) < top_n:
            return ranked[:top_n]

        # Would swapping the weakest incumbent for the best challenger be worth
        # the round trip?
        if incumbents and challengers:
            weakest = min(selected, key=lambda pair: pair[1])
            best_challenger = challengers[0]
            # Margin applied to the MAGNITUDE of the incumbent's score, not
            # multiplicatively. `score * (1 + margin)` lowers the bar when the
            # score is negative -- so a losing incumbent would be replaced more
            # easily than a winning one, which is exactly backwards. It is also
            # unstable near zero, where a trivial absolute difference is a large
            # relative one.
            threshold = weakest[1] + self.config.replacement_margin * abs(weakest[1])
            if (
                weakest[0] in holdings
                and best_challenger[0] not in [s for s, _ in selected]
                and best_challenger[1] > threshold
                and best_challenger[1] > 0
            ):
                selected = [p for p in selected if p[0] != weakest[0]] + [best_challenger]

        return sorted(selected, key=lambda pair: pair[1], reverse=True)[:top_n]

    # --------------------------------------------------------------- signals

    def _build_signals(
        self, selected: list[tuple[str, float]], context: StrategyContext
    ) -> list[Signal]:
        """Equal-weight the selected funds.

        Equal weighting rather than score-proportional: with two positions and
        a two-year sample, score differences are well inside the noise, and
        concentrating on a marginally higher score is false precision.
        """
        if not selected:
            return []

        weight = self.config.max_total_weight / len(selected)
        signals = []
        for symbol, score in selected:
            atr_pct = feature_value(context.row(symbol), "atr_pct")

            signals.append(
                Signal(
                    symbol=symbol,
                    target_weight=weight,
                    score=score,
                    reason=(
                        f"blended risk-adjusted momentum {score:+.2f}, "
                        f"rank {selected.index((symbol, score)) + 1} of {len(selected)}"
                    ),
                    stop_atr_mult=self.config.stop_atr_mult,
                    take_profit_atr_mult=self.config.take_profit_atr_mult,
                    suggested_holding_days=self.config.suggested_holding_days,
                    metadata={"atr_pct": atr_pct},
                )
            )
        return signals

    def _defensive(self, context: StrategyContext, reason: str) -> StrategyDecision:
        """Rotate into the sukuk sleeve, or to cash if it is unavailable."""
        symbol = self.config.defensive_symbol
        regime = context.regime

        if symbol is None or symbol not in context.features:
            return self.flat(
                context.as_of,
                f"{reason}; holding cash"
                + ("" if symbol is None else f" ({symbol} unavailable)"),
                regime,
            )

        if context.row(symbol) is None:
            return self.flat(
                context.as_of, f"{reason}; {symbol} has no data, holding cash", regime
            )

        signal = Signal(
            symbol=symbol,
            target_weight=self.config.max_total_weight,
            score=None,
            reason=f"defensive rotation: {reason}",
            stop_atr_mult=None,  # the defensive sleeve is not stopped out
            suggested_holding_days=self.config.suggested_holding_days,
            metadata={"defensive": True},
        )
        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=(signal,),
            rationale=f"defensive: {reason} -> {symbol}",
            regime=regime,
        )
