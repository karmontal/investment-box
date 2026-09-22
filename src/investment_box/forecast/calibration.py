"""Measuring whether the probabilities mean anything.

A model that says "65%" is useful only if things it calls 65% happen about 65%
of the time. That property is calibration, and it is not implied by accuracy:
a model can rank candidates well and still be badly miscalibrated, which makes
its numbers actively misleading when a human reads them as odds.

This module measures it three ways:

* **Brier score** -- mean squared error between predicted probability and
  outcome. Always guessing 50% scores 0.25, so anything worse than that is
  worse than useless.
* **Reliability curve** -- bucket predictions and compare predicted against
  realised frequency in each bucket. This is what exposes systematic
  over-confidence.
* **Calibration error** -- the average gap, signed, so the *direction* of the
  bias is visible.

The dashboard shows these next to any forecast. A probability without its
calibration is a number with no units.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select

from investment_box.core.logging import get_logger
from investment_box.db.models import Signal
from investment_box.db.session import Database

log = get_logger(__name__)

#: Buckets for the reliability curve. Ten is enough resolution to see a bias
#: without slicing a small sample into empty bins.
DEFAULT_BINS = 10
#: Below this many resolved predictions, calibration figures are noise.
MIN_SAMPLES = 50


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bucket of the reliability curve."""

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_frequency: float

    @property
    def gap(self) -> float:
        """Observed minus predicted. Negative means over-confident."""
        return self.observed_frequency - self.mean_predicted

    @property
    def label(self) -> str:
        return f"{self.lower:.0%}-{self.upper:.0%}"


@dataclass
class CalibrationReport:
    """How well a strategy's probabilities match reality."""

    strategy: str
    samples: int = 0
    brier_score: float | None = None
    calibration_error: float | None = None
    bins: list[ReliabilityBin] = field(default_factory=list)
    period_start: dt.date | None = None
    period_end: dt.date | None = None

    @property
    def is_meaningful(self) -> bool:
        return self.samples >= MIN_SAMPLES

    @property
    def beats_coin_flip(self) -> bool | None:
        if self.brier_score is None:
            return None
        return self.brier_score < 0.25

    @property
    def bias(self) -> str:
        """Which way the strategy is wrong, in plain language."""
        if self.calibration_error is None:
            return "unmeasured"
        if abs(self.calibration_error) < 0.02:
            return "well calibrated"
        if self.calibration_error < 0:
            return f"over-confident by {abs(self.calibration_error):.0%}"
        return f"under-confident by {self.calibration_error:.0%}"

    def warnings(self) -> list[str]:
        """Reasons to distrust this strategy's probabilities."""
        out: list[str] = []
        if not self.is_meaningful:
            out.append(
                f"only {self.samples} resolved predictions (want >= {MIN_SAMPLES}); "
                f"these calibration figures are themselves unreliable"
            )
        if self.beats_coin_flip is False:
            out.append(
                f"Brier score {self.brier_score:.3f} is worse than always guessing 50% "
                f"(0.250). These probabilities are worse than no probability at all."
            )
        if self.calibration_error is not None and self.calibration_error < -0.05:
            out.append(
                f"systematically over-confident by {abs(self.calibration_error):.0%}: "
                f"outcomes happen less often than predicted. Discount accordingly."
            )
        return out

    def summary(self) -> str:
        if self.samples == 0:
            return "no resolved predictions yet"
        brier = f"{self.brier_score:.3f}" if self.brier_score is not None else "—"
        return f"{self.samples} predictions, Brier {brier}, {self.bias}"


def brier_score(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    return float(np.mean((predictions - outcomes) ** 2))


def reliability_curve(
    predictions: np.ndarray, outcomes: np.ndarray, bins: int = DEFAULT_BINS
) -> list[ReliabilityBin]:
    """Bucket predictions and compare predicted against realised frequency.

    Empty buckets are omitted rather than reported as 0% -- an untested range
    is not a failed one.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: list[ReliabilityBin] = []

    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        # Include the top edge in the final bucket so a prediction of exactly
        # 1.0 is counted rather than silently dropped.
        mask = (
            (predictions >= lower) & (predictions <= upper)
            if index == bins - 1
            else (predictions >= lower) & (predictions < upper)
        )
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            ReliabilityBin(
                lower=float(lower),
                upper=float(upper),
                count=count,
                mean_predicted=float(predictions[mask].mean()),
                observed_frequency=float(outcomes[mask].mean()),
            )
        )
    return out


def evaluate_calibration(
    strategy: str,
    predictions: Sequence[float],
    outcomes: Sequence[bool | int],
    *,
    bins: int = DEFAULT_BINS,
    period_start: dt.date | None = None,
    period_end: dt.date | None = None,
) -> CalibrationReport:
    """Build a calibration report from paired predictions and outcomes."""
    report = CalibrationReport(
        strategy=strategy, period_start=period_start, period_end=period_end
    )
    if not predictions or len(predictions) != len(outcomes):
        return report

    predicted = np.asarray(predictions, dtype=float)
    observed = np.asarray([int(bool(o)) for o in outcomes], dtype=float)

    report.samples = len(predicted)
    report.brier_score = brier_score(predicted, observed)
    report.calibration_error = float(observed.mean() - predicted.mean())
    report.bins = reliability_curve(predicted, observed, bins)
    return report


class CalibrationTracker:
    """Reads resolved predictions out of the signals table.

    A signal is *resolved* once its disposition records what actually happened.
    Unresolved signals are excluded rather than counted as failures -- a trade
    still open is not a wrong forecast.
    """

    #: Dispositions that mean the forecast came true / did not.
    WON = frozenset({"closed_win", "target_hit"})
    LOST = frozenset({"closed_loss", "stopped_out"})

    def __init__(self, database: Database) -> None:
        self.db = database

    def report(self, strategy: str, *, since: dt.date | None = None) -> CalibrationReport:
        with self.db.session() as session:
            statement = select(Signal).where(
                Signal.strategy == strategy,
                Signal.probability.is_not(None),
                Signal.disposition.in_([*self.WON, *self.LOST]),
            )
            if since is not None:
                statement = statement.where(Signal.as_of_date >= since)
            rows = list(session.scalars(statement).all())
            for row in rows:
                session.expunge(row)

        if not rows:
            return CalibrationReport(strategy=strategy)

        predictions = [float(r.probability) for r in rows if r.probability is not None]
        outcomes = [r.disposition in self.WON for r in rows if r.probability is not None]
        dates = [r.as_of_date for r in rows]

        return evaluate_calibration(
            strategy,
            predictions,
            outcomes,
            period_start=min(dates) if dates else None,
            period_end=max(dates) if dates else None,
        )

    def all_strategies(self) -> dict[str, CalibrationReport]:
        with self.db.session() as session:
            names = list(session.scalars(select(Signal.strategy).distinct()).all())
        return {name: self.report(name) for name in names}
