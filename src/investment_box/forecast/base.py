"""What a forecast is, and what it is not.

Principle 4 of the build spec: forecasts are probabilities with confidence and
historical out-of-sample accuracy, never point price targets presented as
certain. This module encodes that as a type.

So :class:`Forecast` has:

* a **direction probability**, not a target price
* an **expected return range**, derived from the instrument's own realised
  volatility rather than from a guess
* a **confidence** that degrades when the evidence is thin
* the **track record** of the strategy that produced it, so the number can be
  read against how often that strategy has been right before

There is deliberately no ``target_price`` field. A single number invites being
read as a prediction, and no honest version of this system can produce one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum


class Confidence(StrEnum):
    """How much weight a forecast deserves.

    Driven by evidence quality -- sample size, calibration, data coverage --
    not by the probability itself. A 70% forecast from a strategy with 12
    out-of-sample trades is LOW confidence; the number is high, the evidence
    is not.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    #: Not enough evidence to say anything. Never auto-traded.
    NONE = "none"

    @property
    def is_actionable(self) -> bool:
        return self is not Confidence.NONE


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """A strategy's realised out-of-sample performance.

    Attached to every forecast so a probability is always read alongside how
    often this strategy has actually been right.
    """

    strategy: str
    trades: int = 0
    win_rate: float | None = None
    avg_return: float | None = None
    sharpe: float | None = None
    #: Mean squared error between predicted probability and outcome. Lower is
    #: better; 0.25 is what you get by always guessing 50%.
    brier_score: float | None = None
    #: Realised frequency minus predicted probability. Positive means the
    #: strategy is under-confident, negative means over-confident.
    calibration_error: float | None = None
    period_start: dt.date | None = None
    period_end: dt.date | None = None

    @property
    def is_meaningful(self) -> bool:
        """Whether there is enough history for these figures to mean anything."""
        return self.trades >= 30

    @property
    def beats_coin_flip(self) -> bool | None:
        """Whether the Brier score beats always guessing 50%.

        ``None`` when unmeasured. This is a low bar and failing it is
        informative: a strategy worse than a coin flip is actively misleading.
        """
        if self.brier_score is None:
            return None
        return self.brier_score < 0.25

    def summary(self) -> str:
        if self.trades == 0:
            return "no out-of-sample record yet"
        parts = [f"{self.trades} trades"]
        if self.win_rate is not None:
            parts.append(f"{self.win_rate:.0%} win rate")
        if self.avg_return is not None:
            parts.append(f"{self.avg_return:+.2%} avg")
        if self.sharpe is not None:
            parts.append(f"Sharpe {self.sharpe:.2f}")
        suffix = "" if self.is_meaningful else " (too few trades to be meaningful)"
        return ", ".join(parts) + suffix


@dataclass(frozen=True, slots=True)
class Forecast:
    """A probabilistic view of one candidate.

    Note what is absent: no target price, and no single "expected return".
    ``expected_return_low``/``high`` bracket a range, and the docstring on
    :meth:`range_description` explains what the bracket means.
    """

    symbol: str
    as_of: dt.date
    strategy: str
    #: P(price higher after `horizon_days`), in [0, 1].
    direction_probability: float
    horizon_days: int
    expected_return_low: float
    expected_return_high: float
    confidence: Confidence
    track_record: TrackRecord
    #: Why this forecast, in a sentence.
    rationale: str = ""
    #: Realised annualised volatility, the input to the range.
    volatility: float | None = None
    #: Reasons this forecast should be discounted.
    caveats: tuple[str, ...] = ()
    metadata: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.direction_probability <= 1.0:
            raise ValueError(
                f"{self.symbol}: direction_probability must be in [0, 1], "
                f"got {self.direction_probability}"
            )
        if self.expected_return_low > self.expected_return_high:
            raise ValueError(
                f"{self.symbol}: expected_return_low ({self.expected_return_low}) "
                f"exceeds high ({self.expected_return_high})"
            )

    @property
    def edge(self) -> float:
        """Distance from a coin flip. Zero means the forecast says nothing."""
        return self.direction_probability - 0.5

    @property
    def is_actionable(self) -> bool:
        """Whether this forecast may drive a trade.

        Requires both a directional view *and* enough evidence. A confident
        number with no track record behind it is not actionable.
        """
        return self.confidence.is_actionable and self.direction_probability > 0.5

    def range_description(self) -> str:
        """Plain-language description of the return range.

        The range is roughly a one-standard-deviation band from the
        instrument's own realised volatility over the horizon -- so about two
        outcomes in three should land inside it, and one in three outside. It
        is not a best case and worst case.
        """
        return (
            f"{self.expected_return_low:+.1%} to {self.expected_return_high:+.1%} "
            f"over {self.horizon_days} trading days (roughly a 1-sigma band; "
            f"about one outcome in three falls outside it)"
        )

    def honest_summary(self) -> str:
        """The forecast as it should always be shown to a human."""
        lines = [
            f"{self.symbol}: {self.direction_probability:.0%} chance of being higher "
            f"in {self.horizon_days} trading days",
            f"  Range: {self.range_description()}",
            f"  Confidence: {self.confidence.value}",
            f"  Strategy record: {self.track_record.summary()}",
        ]
        if self.caveats:
            lines.extend(f"  ! {caveat}" for caveat in self.caveats)
        return "\n".join(lines)
