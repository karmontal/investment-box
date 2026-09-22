"""A gradient-boosted classifier predicting P(return > threshold in N days).

Kept deliberately small and interpretable, and fitted strictly within each
walk-forward training window -- never once on the whole sample.

Read the limitations before reading the results:

* **The sample is tiny.** Several funds in this universe have two to three
  years of history. Ten features and a few hundred labelled examples per fund
  is a regime where a boosted tree will happily fit noise and report a good
  cross-validated score. The backtester's out-of-sample windows are the only
  numbers worth looking at, and even those come from a short sample.
* **The labels overlap.** A 5-day forward return computed daily means
  consecutive labels share four of five days. That inflates any in-sample
  metric and breaks the independence assumption behind an ordinary CV split.
  This is why only walk-forward, with a gap between train and test, is used.
* **Probabilities need calibrating before they mean anything.** A raw tree
  score of 0.62 is not a 62% chance. The model is wrapped in isotonic
  calibration fitted on a held-out slice of the training window, and the
  forecast module reports realised accuracy against predicted probability so
  the miscalibration is visible rather than assumed away.

If this strategy beats the rotation strategy on two years of data, the correct
conclusion is "the sample is too short to tell", not "the model works".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from investment_box.core.logging import get_logger
from investment_box.features.pipeline import feature_value
from investment_box.strategies.base import (
    Signal,
    Strategy,
    StrategyContext,
    StrategyDecision,
)

log = get_logger(__name__)

#: Features the model may use. Fixed and small on purpose: every extra feature
#: on a few hundred rows is another chance to fit noise.
MODEL_FEATURES: tuple[str, ...] = (
    "momentum_21d",
    "momentum_63d",
    "volatility_20d",
    "rsi_14",
    "macd_histogram",
    "bollinger_position",
    "ma_distance_20",
    "ma_distance_50",
    "volume_zscore_20",
    "atr_pct",
)


@dataclass
class MLConfig:
    #: Predict P(forward return over `horizon_days` > `return_threshold`).
    horizon_days: int = 5
    return_threshold: float = 0.01
    #: Only act above this probability. High because a marginal edge does not
    #: survive ~18 bps of round-trip friction at $100 positions.
    min_probability: float = 0.58
    max_positions: int = 2
    max_total_weight: float = 0.90
    stop_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    #: Minimum labelled rows before the model is allowed to predict at all.
    min_training_rows: int = 250
    #: Held out from the end of the training window for calibration.
    calibration_fraction: float = 0.25
    random_state: int = 7
    model_params: dict[str, Any] = field(
        default_factory=lambda: {
            # Shallow and heavily regularised: the sample cannot support more.
            "n_estimators": 120,
            "max_depth": 3,
            "learning_rate": 0.05,
            "min_samples_leaf": 20,
            "subsample": 0.8,
        }
    )


class MLClassifier(Strategy):
    """Trains per-symbol classifiers inside the walk-forward training window."""

    name = "ml_classifier"
    description = "Gradient-boosted P(return > threshold) with isotonic calibration"

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig()
        self.warmup_bars = 260
        self._models: dict[str, Any] = {}
        self._trained_through: pd.Timestamp | None = None
        #: Set when training was attempted but the data could not support it.
        self.training_notes: list[str] = []

    # -------------------------------------------------------------- training

    def fit(self, features: dict[str, Any], train_end: pd.Timestamp) -> None:
        """Fit one model per symbol on data strictly before ``train_end``.

        Called by the backtester at each walk-forward step. Any symbol without
        enough rows simply gets no model, and produces no signals -- it is not
        backfilled with another symbol's model.
        """
        self._models.clear()
        self.training_notes.clear()
        self._trained_through = train_end

        for symbol, feature_set in features.items():
            frame = feature_set.frame
            usable = frame.loc[frame.index < train_end]
            try:
                model = self._fit_one(symbol, usable)
            except Exception as exc:  # noqa: BLE001 - one symbol must not kill the fit
                log.warning("ml.fit_failed", symbol=symbol, error=str(exc))
                self.training_notes.append(f"{symbol}: fit raised {type(exc).__name__}: {exc}")
                continue
            if model is not None:
                self._models[symbol] = model

    def _fit_one(self, symbol: str, frame: pd.DataFrame) -> Any | None:
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.ensemble import GradientBoostingClassifier
        except ImportError:
            self.training_notes.append("scikit-learn not installed; ML strategy disabled")
            return None

        data = self._labelled(frame)
        if len(data) < self.config.min_training_rows:
            self.training_notes.append(
                f"{symbol}: {len(data)} labelled rows, need "
                f"{self.config.min_training_rows} -- no model fitted"
            )
            return None

        x = data[list(MODEL_FEATURES)].to_numpy()
        y = data["label"].to_numpy()

        if len(np.unique(y)) < 2:
            self.training_notes.append(f"{symbol}: only one class present -- no model fitted")
            return None

        # Calibrate on the most recent slice, keeping it strictly after the
        # fitting slice so the calibration is not fitted on rows the model saw.
        split = int(len(x) * (1 - self.config.calibration_fraction))
        if split < 50 or len(x) - split < 40:
            self.training_notes.append(
                f"{symbol}: too few rows to hold out a calibration set -- no model fitted"
            )
            return None

        base = GradientBoostingClassifier(
            random_state=self.config.random_state, **self.config.model_params
        )
        base.fit(x[:split], y[:split])

        if len(np.unique(y[split:])) < 2:
            self.training_notes.append(
                f"{symbol}: calibration slice has one class -- using uncalibrated scores"
            )
            return base

        # scikit-learn 1.6 replaced cv="prefit" with FrozenEstimator, and 1.9
        # removed the old spelling entirely. Support both so the strategy does
        # not silently stop calibrating on an older install.
        try:
            from sklearn.frozen import FrozenEstimator

            calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic")
        except ImportError:  # scikit-learn < 1.6
            calibrated = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
        calibrated.fit(x[split:], y[split:])
        return calibrated

    def _labelled(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Build (features, label) rows.

        The label is whether the forward return over ``horizon_days`` exceeded
        the threshold. The final ``horizon_days`` rows have no label yet and are
        dropped -- filling them would be inventing the future.
        """
        if frame.empty or "close" not in frame:
            return pd.DataFrame()

        data = frame.copy()
        horizon = self.config.horizon_days
        forward = data["close"].shift(-horizon) / data["close"] - 1.0
        data["label"] = (forward > self.config.return_threshold).astype(int)
        # Drop rows whose forward window runs past the end of the data.
        data = data.iloc[:-horizon] if horizon > 0 else data

        columns = [*MODEL_FEATURES, "label"]
        available = [c for c in columns if c in data.columns]
        if len(available) < len(columns):
            return pd.DataFrame()
        return data[columns].dropna()

    # ------------------------------------------------------------- inference

    def decide(self, context: StrategyContext) -> StrategyDecision:
        regime = context.regime
        if regime is not None and not regime.allows_new_entries:
            return self.flat(
                context.as_of, f"regime blocks new entries: {regime.reason}", regime
            )

        if not self._models:
            return self.flat(
                context.as_of,
                "no trained model available"
                + (f" ({self.training_notes[0]})" if self.training_notes else ""),
                regime,
            )

        scored: list[tuple[str, float]] = []
        for symbol, model in self._models.items():
            probability = self._predict(context, symbol, model)
            if probability is not None and probability >= self.config.min_probability:
                scored.append((symbol, probability))

        if not scored:
            return self.flat(
                context.as_of,
                f"no symbol reached P >= {self.config.min_probability:.0%}",
                regime,
            )

        scored.sort(key=lambda pair: pair[1], reverse=True)
        selected = scored[: self.config.max_positions]
        weight = self.config.max_total_weight / len(selected)

        signals = [
            Signal(
                symbol=symbol,
                target_weight=weight,
                score=probability,
                reason=(
                    f"P(return > {self.config.return_threshold:.0%} in "
                    f"{self.config.horizon_days}d) = {probability:.1%}"
                ),
                stop_atr_mult=self.config.stop_atr_mult,
                take_profit_atr_mult=self.config.take_profit_atr_mult,
                suggested_holding_days=self.config.horizon_days,
                metadata={"probability": probability, "calibrated": True},
            )
            for symbol, probability in selected
        ]

        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=tuple(self.validate_weights(signals)),
            rationale="; ".join(f"{s} P={p:.1%}" for s, p in selected),
            regime=regime,
        )

    def _predict(self, context: StrategyContext, symbol: str, model: Any) -> float | None:
        row = context.row(symbol)
        if row is None:
            return None

        values: list[float] = []
        for feature in MODEL_FEATURES:
            value = feature_value(row, feature)
            if value is None:
                return None
            values.append(value)

        try:
            probability = model.predict_proba(np.array([values]))[0][1]
        except Exception as exc:  # noqa: BLE001 - a prediction failure is "no signal"
            log.warning("ml.predict_failed", symbol=symbol, error=str(exc))
            return None
        return float(probability)
