"""Turning a strategy signal into an honest forecast.

The hard part is not producing a number. It is producing one that degrades
gracefully when the evidence is thin, which is the normal case here: two to
seven years of history on eight funds, and several strategies with almost no
out-of-sample trades.

So the generator:

* derives the probability from the strategy's **realised** win rate where one
  exists, rather than from the raw model score
* shrinks it toward 50% when the sample is small -- the less evidence there is,
  the closer the forecast sits to "no opinion"
* builds the return range from the instrument's **own** realised volatility
* downgrades confidence, and records why, whenever anything is missing

A forecast whose confidence is NONE is never actionable. That is the safe
default and it is what a brand-new strategy gets.
"""

from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from investment_box.core.logging import get_logger
from investment_box.features.pipeline import FeatureSet, feature_value
from investment_box.forecast.base import Confidence, Forecast, TrackRecord
from investment_box.forecast.calibration import CalibrationReport
from investment_box.strategies.base import Signal

log = get_logger(__name__)

TRADING_DAYS = 252

#: Trades below which the win rate is shrunk hard toward 50%. Chosen so that a
#: strategy needs a real sample before its record moves the forecast much.
SHRINKAGE_PRIOR_TRADES = 50
#: Probability is never reported outside this band. Claiming 90% confidence on
#: a 5-day equity move is not credible whatever the model says.
PROBABILITY_FLOOR = 0.20
PROBABILITY_CEILING = 0.80


def shrink_toward_even(rate: float, sample_size: int, prior: int = SHRINKAGE_PRIOR_TRADES) -> float:
    """Pull an observed rate toward 50% in proportion to how thin the sample is.

    A 70% win rate over 10 trades and over 500 trades are very different
    claims. This is a beta-binomial posterior mean with a symmetric prior
    centred on 0.5: with no data it returns 0.5, and it converges to the
    observed rate as the sample grows.
    """
    if sample_size <= 0:
        return 0.5
    weight = sample_size / (sample_size + prior)
    return 0.5 + weight * (rate - 0.5)


class ForecastGenerator:
    """Builds forecasts from signals, features and measured track records."""

    def __init__(
        self,
        *,
        default_horizon_days: int = 5,
        probability_floor: float = PROBABILITY_FLOOR,
        probability_ceiling: float = PROBABILITY_CEILING,
    ) -> None:
        self.default_horizon_days = default_horizon_days
        self.probability_floor = probability_floor
        self.probability_ceiling = probability_ceiling

    def generate(
        self,
        signal: Signal,
        features: FeatureSet,
        *,
        as_of: dt.date,
        strategy: str,
        track_record: TrackRecord,
        calibration: CalibrationReport | None = None,
    ) -> Forecast:
        """Produce a forecast for one signal."""
        row = features.at(_stamp(as_of))
        volatility = feature_value(row, "volatility_20d")
        horizon = signal.suggested_holding_days or self.default_horizon_days

        caveats: list[str] = []
        probability = self._probability(signal, track_record, calibration, caveats)
        low, high = self._return_range(volatility, horizon, caveats)
        confidence = self._confidence(track_record, calibration, volatility, caveats)

        return Forecast(
            symbol=signal.symbol,
            as_of=as_of,
            strategy=strategy,
            direction_probability=probability,
            horizon_days=horizon,
            expected_return_low=low,
            expected_return_high=high,
            confidence=confidence,
            track_record=track_record,
            rationale=signal.reason,
            volatility=volatility,
            caveats=tuple(caveats),
            metadata={
                "signal_score": signal.score if signal.score is not None else float("nan"),
                "target_weight": signal.target_weight,
            },
        )

    # ---------------------------------------------------------- probability

    def _probability(
        self,
        signal: Signal,
        record: TrackRecord,
        calibration: CalibrationReport | None,
        caveats: list[str],
    ) -> float:
        """Best available estimate of P(higher after the horizon).

        Order of preference: a calibrated model probability, then the
        strategy's shrunk realised win rate, then 50%. Each fallback adds a
        caveat, so the basis for the number is always visible.
        """
        model_probability = signal.metadata.get("probability")

        if isinstance(model_probability, (int, float)) and 0 <= model_probability <= 1:
            probability = float(model_probability)
            if calibration is not None and calibration.calibration_error is not None:
                if calibration.is_meaningful:
                    # Correct for measured bias rather than trusting the model.
                    probability += calibration.calibration_error
                    caveats.append(
                        f"adjusted by {calibration.calibration_error:+.0%} for this "
                        f"strategy's measured calibration bias"
                    )
                else:
                    caveats.append(
                        "model probability is uncalibrated: too few resolved "
                        "predictions to measure its bias"
                    )
            else:
                caveats.append("model probability has never been calibrated against outcomes")
        elif record.win_rate is not None and record.trades > 0:
            probability = shrink_toward_even(record.win_rate, record.trades)
            if not record.is_meaningful:
                caveats.append(
                    f"based on only {record.trades} out-of-sample trades, shrunk "
                    f"toward 50% accordingly"
                )
        else:
            caveats.append(
                "no out-of-sample record for this strategy; reported as a coin flip"
            )
            return 0.5

        return float(min(self.probability_ceiling, max(self.probability_floor, probability)))

    # --------------------------------------------------------------- range

    def _return_range(
        self, volatility: float | None, horizon_days: int, caveats: list[str]
    ) -> tuple[float, float]:
        """A one-sigma band from the instrument's own realised volatility.

        Deliberately symmetric and centred on zero. Centring it on an expected
        drift would be asserting a point forecast through the back door.
        """
        if volatility is None or volatility <= 0:
            caveats.append("volatility unavailable; no return range could be computed")
            return 0.0, 0.0

        horizon_vol = volatility * math.sqrt(horizon_days / TRADING_DAYS)
        return -horizon_vol, horizon_vol

    # ---------------------------------------------------------- confidence

    def _confidence(
        self,
        record: TrackRecord,
        calibration: CalibrationReport | None,
        volatility: float | None,
        caveats: list[str],
    ) -> Confidence:
        """Confidence is about evidence, not about the size of the number."""
        if volatility is None:
            return Confidence.NONE
        if record.trades == 0:
            caveats.append("no track record: this forecast is not actionable")
            return Confidence.NONE
        if calibration is not None and calibration.beats_coin_flip is False:
            caveats.append(
                "this strategy's probabilities score worse than a coin flip; "
                "treat the number as noise"
            )
            return Confidence.NONE

        if record.is_meaningful and calibration is not None and calibration.is_meaningful:
            return Confidence.HIGH
        if record.trades >= 30:
            return Confidence.MEDIUM
        return Confidence.LOW


def _stamp(day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")
