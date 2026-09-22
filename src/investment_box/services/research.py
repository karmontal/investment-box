"""Candidate generation for the dashboard and the bot.

Both consumers call this. Neither computes a ranking of its own, which is what
keeps the dashboard's candidate table and Telegram's ``/funds`` from ever
disagreeing.

Everything here is read-only. Generating candidates never places an order,
never mutates a position, and never decides anything -- it assembles the
evidence a human or the engine will decide on.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from investment_box.backtest.metrics import PerformanceMetrics
from investment_box.config.schema import Settings
from investment_box.core.clock import UTC, Clock, SystemClock, TradingCalendar
from investment_box.core.logging import get_logger
from investment_box.data.repository import MarketDataRepository
from investment_box.features.pipeline import FeatureSet, build_features
from investment_box.features.regime import RegimeDetector, RegimeState
from investment_box.forecast.base import TrackRecord
from investment_box.forecast.calibration import CalibrationReport, CalibrationTracker
from investment_box.forecast.candidates import Candidate, classify, rank_candidates
from investment_box.forecast.generator import ForecastGenerator
from investment_box.shariah.status import ComplianceTracker
from investment_box.strategies.base import Strategy, StrategyContext, StrategyDecision
from investment_box.universe.builder import Instrument, UniverseBuilder, UniverseSnapshot

log = get_logger(__name__)

#: History fetched per symbol. Enough for the 200-day features plus warm-up.
LOOKBACK_DAYS = 420


@dataclass
class ResearchSnapshot:
    """Everything the dashboard's research view needs, computed once."""

    as_of: dt.date
    candidates: list[Candidate] = field(default_factory=list)
    universe: UniverseSnapshot | None = None
    regime: RegimeState | None = None
    decision: StrategyDecision | None = None
    calibration: CalibrationReport | None = None
    strategy_name: str = ""
    #: Problems with the snapshot itself, shown before the table.
    warnings: list[str] = field(default_factory=list)

    @property
    def actionable(self) -> list[Candidate]:
        return [c for c in self.candidates if c.is_tradable]

    @property
    def has_anything_actionable(self) -> bool:
        return bool(self.actionable)


class ResearchService:
    """Builds ranked candidates from the universe, features and strategy."""

    def __init__(
        self,
        settings: Settings,
        repository: MarketDataRepository,
        strategy: Strategy,
        *,
        compliance: ComplianceTracker | None = None,
        calibration: CalibrationTracker | None = None,
        generator: ForecastGenerator | None = None,
        clock: Clock | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.strategy = strategy
        self.compliance = compliance
        self.calibration = calibration
        self.generator = generator or ForecastGenerator()
        self.clock = clock or SystemClock()
        self.calendar = calendar or TradingCalendar()
        self.regime_detector = RegimeDetector(repository)
        #: Populated from backtest results; empty until one has been run.
        self._track_records: dict[str, TrackRecord] = {}

    def register_track_record(self, record: TrackRecord) -> None:
        """Attach a strategy's measured out-of-sample record.

        Without one, every forecast reports NONE confidence and nothing is
        actionable -- which is the correct default for an unproven strategy.
        """
        self._track_records[record.strategy] = record

    def load_track_records_from_metrics(self, metrics: list[PerformanceMetrics]) -> None:
        for entry in metrics:
            self.register_track_record(
                TrackRecord(
                    strategy=entry.name,
                    trades=entry.num_trades,
                    win_rate=entry.win_rate,
                    avg_return=entry.avg_trade_pct,
                    sharpe=entry.sharpe,
                    period_start=entry.start.date() if entry.start is not None else None,
                    period_end=entry.end.date() if entry.end is not None else None,
                )
            )

    # ------------------------------------------------------------- the build

    def build(
        self,
        instruments: list[Instrument],
        *,
        as_of: dt.date | None = None,
        current_holdings: tuple[str, ...] = (),
    ) -> ResearchSnapshot:
        """Assemble the ranked candidate list."""
        day = as_of or self._last_session()
        snapshot = ResearchSnapshot(as_of=day, strategy_name=self.strategy.name)

        builder = UniverseBuilder(
            self.settings, self.repository, self.compliance, clock=self.clock
        )
        universe = builder.build(instruments, as_of=day)
        snapshot.universe = universe

        if universe.is_empty:
            snapshot.warnings.append(
                "No symbol passed the universe filters. "
                + "; ".join(f"{k}: {v}" for k, v in universe.rejection_summary().items())
            )

        # Features are built for every instrument, not only the included ones,
        # so the dashboard can explain a blocked symbol with real numbers
        # rather than a blank row.
        features = self._features([i.symbol for i in instruments], day)

        snapshot.regime = self._regime(day, snapshot)
        decision = self._decide(day, features, snapshot.regime, current_holdings, snapshot)
        snapshot.decision = decision

        record = self._track_record()
        calibration = self._calibration()
        snapshot.calibration = calibration
        self._add_evidence_warnings(record, calibration, snapshot)

        signals = {s.symbol: s for s in decision.signals} if decision else {}
        entries = {e.symbol: e for e in universe.entries}

        candidates: list[Candidate] = []
        for instrument in instruments:
            symbol = instrument.symbol
            feature_set = features.get(symbol)
            signal = signals.get(symbol)

            forecast = None
            if signal is not None and feature_set is not None and not feature_set.is_empty:
                forecast = self.generator.generate(
                    signal,
                    feature_set,
                    as_of=day,
                    strategy=self.strategy.name,
                    track_record=record,
                    calibration=calibration,
                )

            entry = entries.get(symbol)
            candidates.append(
                classify(
                    symbol,
                    day,
                    forecast=forecast,
                    compliance=entry.compliance if entry else None,
                    universe_entry=entry,
                    last_price=entry.last_price if entry else None,
                )
            )

        snapshot.candidates = rank_candidates(candidates)
        return snapshot

    # -------------------------------------------------------------- internals

    def _last_session(self) -> dt.date:
        today = self.clock.now().astimezone(UTC).date()
        return self.calendar.previous_trading_day(today, inclusive=True)

    def _features(self, symbols: list[str], day: dt.date) -> dict[str, FeatureSet]:
        start = day - dt.timedelta(days=LOOKBACK_DAYS)
        fetched = self.repository.get_many(symbols, start, day)
        return {
            symbol: build_features(result.frame, symbol)
            for symbol, result in fetched.items()
            if not result.frame.empty
        }

    def _regime(self, day: dt.date, snapshot: ResearchSnapshot) -> RegimeState | None:
        try:
            regime = self.regime_detector.detect(day)
        except Exception as exc:  # noqa: BLE001 - a missing regime degrades, never crashes
            log.warning("research.regime_failed", error=str(exc))
            snapshot.warnings.append(f"market regime could not be computed: {exc}")
            return None
        if regime.degraded:
            snapshot.warnings.append(f"regime is degraded: {regime.reason}")
        return regime

    def _decide(
        self,
        day: dt.date,
        features: dict[str, FeatureSet],
        regime: RegimeState | None,
        holdings: tuple[str, ...],
        snapshot: ResearchSnapshot,
    ) -> StrategyDecision | None:
        context = StrategyContext(
            as_of=day, features=features, regime=regime, current_holdings=holdings
        )
        try:
            return self.strategy.decide(context)
        except Exception as exc:  # noqa: BLE001 - a broken strategy must not blank the page
            log.error("research.strategy_failed", strategy=self.strategy.name, error=str(exc))
            snapshot.warnings.append(
                f"strategy {self.strategy.name} raised {type(exc).__name__}: {exc}"
            )
            return None

    def _track_record(self) -> TrackRecord:
        return self._track_records.get(
            self.strategy.name, TrackRecord(strategy=self.strategy.name)
        )

    def _calibration(self) -> CalibrationReport | None:
        if self.calibration is None:
            return None
        try:
            return self.calibration.report(self.strategy.name)
        except Exception as exc:  # noqa: BLE001
            log.warning("research.calibration_failed", error=str(exc))
            return None

    @staticmethod
    def _add_evidence_warnings(
        record: TrackRecord, calibration: CalibrationReport | None, snapshot: ResearchSnapshot
    ) -> None:
        """Say plainly when the forecasts rest on nothing."""
        if record.trades == 0:
            snapshot.warnings.append(
                f"{record.strategy} has no out-of-sample record. Every forecast below "
                f"is reported as a coin flip and nothing is actionable."
            )
        elif not record.is_meaningful:
            snapshot.warnings.append(
                f"{record.strategy} has only {record.trades} out-of-sample trades. "
                f"Probabilities are shrunk toward 50% to reflect that."
            )
        if calibration is not None:
            snapshot.warnings.extend(calibration.warnings())
