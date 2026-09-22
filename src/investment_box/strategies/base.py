"""The strategy interface.

A strategy answers one question: given what was knowable at the close of a
bar, what should the portfolio look like? It returns *target weights*, not
orders. Turning weights into orders is the risk manager's and the execution
layer's job, and keeping that separation means a strategy cannot accidentally
bypass position sizing, the cash buffer or settlement rules.

Strategies never see the future, never see the broker, and never place a
trade. They are pure functions of history plus configuration, which is what
makes them testable and what makes a walk-forward backtest meaningful.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from investment_box.core.types import Side
from investment_box.features.pipeline import FeatureSet
from investment_box.features.regime import RegimeState


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Everything a strategy may look at on a given bar.

    Deliberately narrow. There is no broker, no cash balance and no open-order
    list here: a strategy that could see them would start making execution
    decisions, and those belong downstream where settlement and risk limits
    are enforced.
    """

    as_of: dt.date
    features: dict[str, FeatureSet]
    regime: RegimeState | None = None
    #: Symbols currently held, so a strategy can express "keep holding" rather
    #: than churning a position it would re-open next bar.
    current_holdings: tuple[str, ...] = ()
    #: Trading days each holding has been open, for minimum-hold awareness.
    holding_days: dict[str, int] = field(default_factory=dict)

    def row(self, symbol: str) -> pd.Series | None:
        """The feature row for ``symbol`` as of this bar, or ``None``."""
        feature_set = self.features.get(symbol)
        if feature_set is None:
            return None
        return feature_set.at(pd.Timestamp(self.as_of, tz="UTC"))


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's opinion about one symbol.

    ``target_weight`` is a fraction of the bot's *allocated capital*, not of
    account equity, and not of the position. Zero means "hold nothing".
    """

    symbol: str
    target_weight: float
    side: Side = Side.BUY
    #: Ranking score, comparable within a strategy but not across strategies.
    score: float | None = None
    #: A sentence explaining this signal, recorded on the trade.
    reason: str = ""
    #: Suggested stop distance in ATR multiples; the risk manager decides the
    #: actual level and may tighten it.
    stop_atr_mult: float | None = None
    take_profit_atr_mult: float | None = None
    suggested_holding_days: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.target_weight <= 1.0:
            raise ValueError(
                f"{self.symbol}: target_weight must be in [0, 1], got {self.target_weight}. "
                f"Weights above 1 would imply leverage, which is never permitted."
            )
        if self.side is not Side.BUY:
            raise ValueError(
                f"{self.symbol}: only long positions are permitted; got side={self.side}"
            )


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    """The full output for one bar."""

    as_of: dt.date
    strategy: str
    signals: tuple[Signal, ...]
    #: Why the strategy did what it did, including doing nothing.
    rationale: str = ""
    regime: RegimeState | None = None

    @property
    def target_weights(self) -> dict[str, float]:
        return {s.symbol: s.target_weight for s in self.signals if s.target_weight > 0}

    @property
    def total_weight(self) -> float:
        return sum(s.target_weight for s in self.signals)

    @property
    def is_flat(self) -> bool:
        return not self.target_weights


class Strategy(ABC):
    """Base class for every strategy."""

    #: Stable identifier, recorded on every signal and trade.
    name: str = "unnamed"
    #: One line shown in the dashboard and the backtest report.
    description: str = ""
    #: Bars of history needed before the strategy can produce a signal. The
    #: backtester uses this to skip dates it could not honestly have traded.
    warmup_bars: int = 200

    @abstractmethod
    def decide(self, context: StrategyContext) -> StrategyDecision:
        """Produce target weights for ``context.as_of``.

        Must use only data at or before ``context.as_of``. The context is
        already trimmed, but a strategy that reaches around it (for instance by
        reading ``FeatureSet.frame`` directly) breaks the guarantee.
        """

    def validate_weights(self, signals: list[Signal]) -> list[Signal]:
        """Reject a weight set that would imply leverage.

        A defence against an arithmetic slip in a strategy: no combination of
        signals may total more than 100% of allocated capital, because there is
        no margin and never will be.
        """
        total = sum(s.target_weight for s in signals)
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"{self.name}: target weights sum to {total:.2%}, which would require "
                f"leverage. This account is cash-only."
            )
        return signals

    def flat(
        self, as_of: dt.date, reason: str, regime: RegimeState | None = None
    ) -> StrategyDecision:
        """A decision to hold nothing, with the reason recorded."""
        return StrategyDecision(
            as_of=as_of, strategy=self.name, signals=(), rationale=reason, regime=regime
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
