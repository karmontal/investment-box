"""Persisting measured performance, and loading it back for the engine.

The forecast layer refuses to act without an out-of-sample record, which is
the right default. What was missing was any way to supply one: the backtest
produced metrics and threw them away, and nothing in ``src/`` or ``scripts/``
ever called ``register_track_record``. The engine therefore reported every
probability as a coin flip and could not place a trade, ever.

This module closes that loop, and is deliberately strict about what counts:

* **In-sample figures are never loaded.** They are stored when offered, so the
  record exists, but only ``out_of_sample`` rows reach a forecast.
* **A record expires.** Costs, spreads and the universe all move; a
  measurement from last year is not evidence about this week. A stale row is
  excluded and the reason is reported, not swallowed.
* **The newest measurement per strategy wins**, so re-running the backtest
  supersedes rather than accumulates.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy import select

from investment_box.backtest.metrics import PerformanceMetrics
from investment_box.core.logging import get_logger
from investment_box.db.models import StrategyTrackRecord
from investment_box.db.session import Database
from investment_box.forecast.base import TrackRecord

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LoadOutcome:
    """What was loaded, and what was refused and why."""

    records: list[TrackRecord]
    rejected: list[str]

    @property
    def loaded_strategies(self) -> list[str]:
        return [r.strategy for r in self.records]


class TrackRecordStore:
    """Reads and writes :class:`StrategyTrackRecord` rows."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def save(
        self,
        metrics: PerformanceMetrics,
        *,
        source: str,
        out_of_sample: bool,
        measured_at: dt.datetime | None = None,
        notes: str | None = None,
    ) -> None:
        """Record one measurement. Never overwrites: history is kept."""
        with self.db.session() as session:
            session.add(
                StrategyTrackRecord(
                    strategy=metrics.name,
                    measured_at=measured_at,
                    period_start=metrics.start.date() if metrics.start is not None else None,
                    period_end=metrics.end.date() if metrics.end is not None else None,
                    trades=int(metrics.num_trades),
                    win_rate=metrics.win_rate,
                    avg_return=metrics.avg_trade_pct,
                    sharpe=metrics.sharpe,
                    out_of_sample=out_of_sample,
                    source=source,
                    notes=notes,
                )
            )
        log.info(
            "track_record.saved",
            strategy=metrics.name,
            trades=metrics.num_trades,
            win_rate=metrics.win_rate,
            out_of_sample=out_of_sample,
            source=source,
        )

    def latest(self, strategy: str) -> StrategyTrackRecord | None:
        with self.db.session() as session:
            row = session.scalar(
                select(StrategyTrackRecord)
                .where(StrategyTrackRecord.strategy == strategy)
                .order_by(StrategyTrackRecord.measured_at.desc())
                .limit(1)
            )
            if row is None:
                return None
            session.expunge(row)
        return row

    def load(self, *, max_age_days: int, now: dt.datetime) -> LoadOutcome:
        """Every strategy's newest usable measurement.

        Args:
            max_age_days: Beyond this, a measurement is treated as absent. Zero
                or less disables expiry, which is only sensible in tests.
            now: Current time, from the injected clock.
        """
        records: list[TrackRecord] = []
        rejected: list[str] = []

        with self.db.session() as session:
            names = list(session.scalars(select(StrategyTrackRecord.strategy).distinct()).all())

        for name in names:
            row = self.latest(name)
            if row is None:
                continue

            if not row.out_of_sample:
                rejected.append(
                    f"{name}: newest measurement is in-sample, which is not evidence "
                    f"about unseen data. Re-run the walk-forward backtest."
                )
                continue

            age = (now - row.measured_at).days
            if max_age_days > 0 and age > max_age_days:
                rejected.append(
                    f"{name}: measured {age}d ago, limit {max_age_days}d. Costs and "
                    f"the universe move; re-run the backtest."
                )
                continue

            records.append(
                TrackRecord(
                    strategy=row.strategy,
                    trades=row.trades,
                    win_rate=row.win_rate,
                    avg_return=row.avg_return,
                    sharpe=row.sharpe,
                    period_start=row.period_start,
                    period_end=row.period_end,
                )
            )

        return LoadOutcome(records=records, rejected=rejected)
