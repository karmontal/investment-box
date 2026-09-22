"""Probabilistic forecasts and candidate ranking.

Forecasts are probabilities with confidence and a measured track record, never
point price targets. There is deliberately no ``target_price`` anywhere in this
package.
"""

from investment_box.forecast.base import Confidence, Forecast, TrackRecord
from investment_box.forecast.calibration import (
    CalibrationReport,
    CalibrationTracker,
    ReliabilityBin,
    brier_score,
    evaluate_calibration,
    reliability_curve,
)
from investment_box.forecast.candidates import (
    Candidate,
    CandidateStatus,
    classify,
    rank_candidates,
)
from investment_box.forecast.generator import ForecastGenerator, shrink_toward_even

__all__ = [
    "CalibrationReport",
    "CalibrationTracker",
    "Candidate",
    "CandidateStatus",
    "Confidence",
    "Forecast",
    "ForecastGenerator",
    "ReliabilityBin",
    "TrackRecord",
    "brier_score",
    "classify",
    "evaluate_calibration",
    "rank_candidates",
    "reliability_curve",
    "shrink_toward_even",
]
