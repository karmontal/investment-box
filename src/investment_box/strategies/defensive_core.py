"""Hold one core fund; step aside into sukuk when its trend breaks.

This strategy exists because of what the walk-forward measurement said about
the others. Rotating among eight highly correlated US Shariah equity ETFs
produced a real but tiny gross edge -- profit factor 1.14, $0.69 gross per
trade -- and then paid $0.39 per trade to capture it, so **56% of gross profit
went to costs**. With 98% exposure and 7.4-day average holds, that strategy was
not really trading: it was buy-and-hold with worse timing, which is why its
Sharpe was 0.35 against 1.36 for simply holding SPUS.

Two conclusions follow, and this strategy is built on both.

**Selection among these funds adds nothing.** SPUS and HLAL hold substantially
the same companies. There is not enough dispersion in the universe to rank, so
this strategy does not rank: it holds one core fund and asks a single question
about it.

**Turnover is the enemy.** At this account size the cost per round trip is a
large fraction of any plausible edge, so the design target is a handful of
trades per year, not fifty. That is what the hysteresis band below is for, and
it is the most important parameter here.

What this strategy is *not* trying to do is beat buy-and-hold on total return.
It is trying to keep most of the return while cutting the worst of the
drawdown, and it should be judged on that -- max drawdown and Sharpe first,
total return second. If it fails that test the honest answer is to hold SPUS
and turn the engine off, and the backtest report will say so.
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
class DefensiveCoreConfig:
    """Tunables. The bands matter far more than the symbols."""

    #: The fund held while the trend holds up.
    core_symbol: str = "SPUS"
    #: Where capital goes when it does not. Sukuk rather than cash so idle
    #: capital is not simply dead, but see ``require_defensive_uptrend``.
    defensive_symbol: str = "SPSK"

    #: Trend measure: distance from the 200-day average, as a fraction.
    trend_feature: str = "ma_distance_200"

    #: Re-enter the core only once it is this far ABOVE its average, and leave
    #: only once it is this far BELOW. Between the two, hold whatever is
    #: already held and place no order at all.
    #:
    #: This gap is the entire turnover-control mechanism. A single threshold at
    #: zero produces a trade every time price crosses its average, which in a
    #: choppy market is many trades a month, and at this account size the costs
    #: of those trades exceed anything the signal is worth. Widening the band
    #: trades responsiveness for cost, and cost is the measured problem.
    entry_band: float = 0.02
    exit_band: float = 0.02

    #: Fraction of allocated capital to deploy. Below 1.0 leaves a cash buffer
    #: on top of whatever the risk manager already holds back.
    target_weight: float = 0.95

    #: When the defensive fund is itself below its 200-day average, hold cash
    #: instead of rotating into it. Sukuk is not a safe haven by definition:
    #: in 2022 rate rises pushed it down alongside equities, and a rule that
    #: rotates into a falling asset because it is nominally "defensive" is
    #: just a different way to lose money.
    require_defensive_uptrend: bool = True

    #: A wide stop, or none at all. This is a regime rule, not a trade: a stop
    #: tight enough to matter would fire on noise the trend filter is meant to
    #: ignore, and would reintroduce the turnover this design exists to avoid.
    stop_atr_mult: float | None = None


class DefensiveCore(Strategy):
    """Hold the core fund in an uptrend; step aside when it breaks."""

    name = "defensive_core"
    description = (
        "Hold one core ETF while it is above its 200-day average; rotate to sukuk "
        "(or cash) when it is below. A handful of trades a year by design."
    )

    def __init__(self, config: DefensiveCoreConfig | None = None) -> None:
        self.config = config or DefensiveCoreConfig()
        # 200-day average plus room for it to be defined on the first usable bar.
        self.warmup_bars = 220

        if self.config.entry_band < 0 or self.config.exit_band < 0:
            raise ValueError(
                f"{self.name}: bands must be non-negative; got entry={self.config.entry_band}, "
                f"exit={self.config.exit_band}. A negative band inverts the rule."
            )

    def decide(self, context: StrategyContext) -> StrategyDecision:
        cfg = self.config
        core_trend = feature_value(context.row(cfg.core_symbol), cfg.trend_feature)

        if core_trend is None:
            # No trend reading means no opinion. Holding through an unknown is
            # not the same as deciding to hold, so say which one this is.
            return self.flat(
                context.as_of,
                f"{cfg.core_symbol}: no {cfg.trend_feature} yet (warm-up or missing data); "
                f"holding nothing rather than assuming an uptrend",
                regime=context.regime,
            )

        holding_core = cfg.core_symbol in context.current_holdings
        holding_defensive = cfg.defensive_symbol in context.current_holdings

        if core_trend >= cfg.entry_band:
            state = "risk_on"
        elif core_trend <= -cfg.exit_band:
            state = "risk_off"
        else:
            # Inside the band: whatever is held stays held, and nothing is
            # bought. This is where the trades that would have cost more than
            # they are worth get skipped.
            state = "hold" if (holding_core or holding_defensive) else "stand_aside"

        if state == "hold":
            held = cfg.core_symbol if holding_core else cfg.defensive_symbol
            return self._single(
                context,
                held,
                core_trend,
                f"{cfg.core_symbol} is {core_trend:+.1%} from its 200-day average, inside the "
                f"±{cfg.entry_band:.0%}/{cfg.exit_band:.0%} band -- holding {held} unchanged "
                f"rather than paying to act on an ambiguous reading",
            )

        if state == "stand_aside":
            return self.flat(
                context.as_of,
                f"{cfg.core_symbol} is {core_trend:+.1%} from its 200-day average, inside the "
                f"band and nothing is held. Waiting for a clear reading rather than entering "
                f"on one that is neither.",
                regime=context.regime,
            )

        if state == "risk_on":
            return self._single(
                context,
                cfg.core_symbol,
                core_trend,
                f"{cfg.core_symbol} is {core_trend:+.1%} above its 200-day average, clear of "
                f"the {cfg.entry_band:.0%} entry band",
            )

        return self._risk_off(context, core_trend)

    # ------------------------------------------------------------------ risk off

    def _risk_off(self, context: StrategyContext, core_trend: float) -> StrategyDecision:
        cfg = self.config
        broken = (
            f"{cfg.core_symbol} is {core_trend:+.1%} from its 200-day average, past the "
            f"{cfg.exit_band:.0%} exit band"
        )

        defensive_trend = feature_value(
            context.row(cfg.defensive_symbol), cfg.trend_feature
        )

        if defensive_trend is None:
            return self.flat(
                context.as_of,
                f"{broken}, and {cfg.defensive_symbol} has no {cfg.trend_feature} to judge it "
                f"by -- holding cash rather than rotating into an unmeasured asset",
                regime=context.regime,
            )

        if cfg.require_defensive_uptrend and defensive_trend <= 0:
            return self.flat(
                context.as_of,
                f"{broken}, but {cfg.defensive_symbol} is also below its 200-day average "
                f"({defensive_trend:+.1%}) -- holding cash. Rotating into a falling defensive "
                f"asset is a different way to lose, not a hedge.",
                regime=context.regime,
            )

        return self._single(
            context,
            cfg.defensive_symbol,
            defensive_trend,
            f"{broken}; rotating to {cfg.defensive_symbol}, which is {defensive_trend:+.1%} "
            f"from its own average",
        )

    # ------------------------------------------------------------------ helpers

    def _single(
        self, context: StrategyContext, symbol: str, score: float, reason: str
    ) -> StrategyDecision:
        """One position, at most, ever. Two sleeves at once would be a blend
        nobody asked for and would double the turnover on every switch."""
        signal = Signal(
            symbol=symbol,
            target_weight=self.config.target_weight,
            score=score,
            reason=reason,
            stop_atr_mult=self.config.stop_atr_mult,
            metadata={"trend": score, "regime_rule": "ma_200_with_hysteresis"},
        )
        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=tuple(self.validate_weights([signal])),
            rationale=reason,
            regime=context.regime,
        )
