"""Compliance status tracking.

Screens are persisted append-only: a symbol's history is the full sequence of
screens, never a single mutable "current status" row. That is what makes it
possible to answer "what did we know when we traded?" rather than only "what do
we believe now".

Two rules the rest of the system depends on:

* **A stale screen is not a valid screen.** Past the re-screen interval, the
  status is reported as ``UNKNOWN`` regardless of what it last said. Compliance
  is a fact about a company at a time, and companies change.
* **Only COMPLIANT is auto-tradable.** ``DOUBTFUL`` and ``UNKNOWN`` require a
  human. This is enforced here and again in ``shariah/constraints.py`` at the
  order boundary.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass

from sqlalchemy import select

from investment_box.config.schema import ShariahConfig
from investment_box.core.audit import AuditSink
from investment_box.core.clock import UTC, Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ComplianceStatus
from investment_box.db.models import ComplianceScreen
from investment_box.db.session import Database
from investment_box.shariah.providers.base import ScreeningProvider, ScreenResult

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ComplianceRecord:
    """The current view of a symbol, as callers should read it."""

    symbol: str
    status: ComplianceStatus
    source: str
    screened_at: dt.datetime
    reason: str
    is_stale: bool
    age_days: int
    debt_ratio: float | None = None
    interest_securities_ratio: float | None = None
    non_permissible_revenue_ratio: float | None = None
    ratio_denominator: str | None = None

    @property
    def is_tradable(self) -> bool:
        """Auto-tradable only if COMPLIANT *and* fresh."""
        return self.status.auto_tradable and not self.is_stale

    @property
    def display_status(self) -> ComplianceStatus:
        """What to show and act on: a stale screen reads as UNKNOWN."""
        return ComplianceStatus.UNKNOWN if self.is_stale else self.status


class ComplianceTracker:
    """Screens symbols, persists the results, and answers status questions."""

    def __init__(
        self,
        provider: ScreeningProvider,
        database: Database,
        config: ShariahConfig,
        audit: AuditSink,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.provider = provider
        self.db = database
        self.config = config
        self.audit = audit
        self.clock = clock or SystemClock()

    # ------------------------------------------------------------- screening

    def screen(self, symbol: str, *, force: bool = False) -> ComplianceRecord:
        """Return the current status, re-screening if the last one is stale.

        A cached verdict is only reused when it came from the provider now in
        use. Reusing one across a provider change is how a mock's UNKNOWN
        survives being replaced by a real source -- the verdict looks fresh, so
        nothing re-screens, and the symbol stays untradable for a reason that
        no longer exists. The same applies in reverse, which matters more: a
        COMPLIANT from a provider you have since replaced is not evidence about
        the new one.
        """
        existing = self.latest(symbol)
        if (
            existing is not None
            and not existing.is_stale
            and not force
            and existing.source == self.provider.name
        ):
            return existing

        result = self.provider.screen(symbol)
        self._persist(result)
        record = self._to_record(result)

        if existing is not None and existing.status is not result.status:
            self.audit.record(
                "compliance.status_changed",
                f"{symbol.upper()}: {existing.status.value.upper()} -> "
                f"{result.status.value.upper()} ({result.reason})",
                actor="screener",
                symbol=symbol.upper(),
                detail={
                    "previous": existing.status.value,
                    "current": result.status.value,
                    "source": result.source,
                },
            )
        return record

    def screen_all(self, symbols: list[str], *, force: bool = False) -> dict[str, ComplianceRecord]:
        return {s.upper(): self.screen(s, force=force) for s in symbols}

    def latest(self, symbol: str) -> ComplianceRecord | None:
        """Most recent persisted screen, or ``None`` if never screened."""
        with self.db.session() as session:
            row = session.scalar(
                select(ComplianceScreen)
                .where(ComplianceScreen.symbol == symbol.upper())
                .order_by(ComplianceScreen.screened_at.desc())
                .limit(1)
            )
            if row is None:
                return None
            session.expunge(row)
        return self._row_to_record(row)

    def history(self, symbol: str, limit: int = 20) -> list[ComplianceRecord]:
        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(ComplianceScreen)
                    .where(ComplianceScreen.symbol == symbol.upper())
                    .order_by(ComplianceScreen.screened_at.desc())
                    .limit(limit)
                ).all()
            )
            for row in rows:
                session.expunge(row)
        return [self._row_to_record(row) for row in rows]

    def status_at(self, symbol: str, moment: dt.datetime) -> ComplianceRecord | None:
        """What we believed about ``symbol`` at ``moment``.

        The audit question. Uses only screens that existed then, so it cannot
        be contaminated by a later re-screen.
        """
        with self.db.session() as session:
            row = session.scalar(
                select(ComplianceScreen)
                .where(
                    ComplianceScreen.symbol == symbol.upper(),
                    ComplianceScreen.screened_at <= moment,
                )
                .order_by(ComplianceScreen.screened_at.desc())
                .limit(1)
            )
            if row is None:
                return None
            session.expunge(row)
        return self._row_to_record(row, now=moment)

    # ---------------------------------------------------------------- queries

    def tradable(self, symbols: list[str]) -> list[str]:
        """Filter to symbols that may be traded automatically right now."""
        return [s.upper() for s in symbols if self.screen(s).is_tradable]

    def needs_rescreen(self, symbols: list[str]) -> list[str]:
        out = []
        for symbol in symbols:
            record = self.latest(symbol)
            if record is None or record.is_stale:
                out.append(symbol.upper())
        return out

    def newly_non_compliant(self, held_symbols: list[str]) -> list[ComplianceRecord]:
        """Held positions that are no longer permissible.

        The caller alerts and applies the exit policy. Stale screens are
        included: not knowing whether a holding is still compliant is itself a
        reason to look.
        """
        flagged = []
        for symbol in held_symbols:
            record = self.screen(symbol)
            if record.status is ComplianceStatus.NON_COMPLIANT or record.is_stale:
                flagged.append(record)
        return flagged

    def exit_deadline(self, flagged_on: dt.date, calendar: object | None = None) -> dt.date:
        """When a non-compliant holding must be out, per the configured policy."""
        days = self.config.non_compliant_exit_days
        if calendar is not None and hasattr(calendar, "add_trading_days"):
            return calendar.add_trading_days(flagged_on, days)  # type: ignore[no-any-return]
        return flagged_on + dt.timedelta(days=days)

    # -------------------------------------------------------------- internals

    def _persist(self, result: ScreenResult) -> None:
        ratios = result.ratios
        with self.db.session() as session:
            session.add(
                ComplianceScreen(
                    symbol=result.symbol,
                    status=result.status.value,
                    source=result.source,
                    screened_at=result.screened_at,
                    ratio_denominator=ratios.denominator if ratios else None,
                    debt_ratio=ratios.debt_ratio if ratios else None,
                    interest_securities_ratio=(
                        ratios.interest_securities_ratio if ratios else None
                    ),
                    non_permissible_revenue_ratio=(
                        ratios.non_permissible_revenue_ratio if ratios else None
                    ),
                    business_activity_flags=(
                        ",".join(result.activity_flags) if result.activity_flags else None
                    ),
                    notes=result.reason,
                    raw_response_json=json.dumps(result.raw, default=str) if result.raw else None,
                )
            )

    def _age_days(self, screened_at: dt.datetime, now: dt.datetime | None = None) -> int:
        reference = now or self.clock.now()
        return max(0, (reference - screened_at).days)

    def _to_record(self, result: ScreenResult) -> ComplianceRecord:
        age = self._age_days(result.screened_at)
        ratios = result.ratios
        return ComplianceRecord(
            symbol=result.symbol,
            status=result.status,
            source=result.source,
            screened_at=result.screened_at,
            reason=result.reason,
            is_stale=age > self.config.rescreen_interval_days,
            age_days=age,
            debt_ratio=ratios.debt_ratio if ratios else None,
            interest_securities_ratio=ratios.interest_securities_ratio if ratios else None,
            non_permissible_revenue_ratio=(
                ratios.non_permissible_revenue_ratio if ratios else None
            ),
            ratio_denominator=ratios.denominator if ratios else None,
        )

    def _row_to_record(
        self, row: ComplianceScreen, now: dt.datetime | None = None
    ) -> ComplianceRecord:
        screened_at = (
            row.screened_at.replace(tzinfo=UTC)
            if row.screened_at.tzinfo is None
            else row.screened_at
        )
        age = self._age_days(screened_at, now)
        return ComplianceRecord(
            symbol=row.symbol,
            status=ComplianceStatus(row.status),
            source=row.source,
            screened_at=screened_at,
            reason=row.notes or "",
            is_stale=age > self.config.rescreen_interval_days,
            age_days=age,
            debt_ratio=row.debt_ratio,
            interest_securities_ratio=row.interest_securities_ratio,
            non_permissible_revenue_ratio=row.non_permissible_revenue_ratio,
            ratio_denominator=row.ratio_denominator,
        )
