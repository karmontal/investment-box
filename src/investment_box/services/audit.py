"""Append-only audit log.

Principle 6: every signal, decision, order and refusal is recorded with the
reason. This is the write side of that.

Two rules:

* Nothing here ever updates or deletes a row. Corrections are new rows.
* An audit write must never break the thing it is recording. A failed insert
  is logged and swallowed, because losing a log line is bad but crashing the
  engine mid-trade because the disk is full is worse.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import select

from investment_box.core.logging import get_logger
from investment_box.core.types import TradingMode
from investment_box.db.models import AuditLog
from investment_box.db.session import Database

log = get_logger(__name__)


class AuditService:
    """Writes and reads the audit trail.

    Satisfies :class:`investment_box.core.audit.AuditSink`, which is what lower
    layers depend on so the dependency graph stays acyclic.
    """

    def __init__(self, database: Database, trading_mode: TradingMode) -> None:
        self.db = database
        self.trading_mode = trading_mode

    def record(
        self,
        event_type: str,
        summary: str,
        *,
        actor: str = "engine",
        symbol: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one entry.

        Never raises. An audit failure is itself logged, but it does not
        propagate into the caller's control flow.
        """
        try:
            with self.db.session() as session:
                session.add(
                    AuditLog(
                        event_type=event_type,
                        actor=actor,
                        symbol=symbol,
                        summary=summary,
                        detail_json=json.dumps(detail, default=str) if detail else None,
                        trading_mode=self.trading_mode.value,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - an audit failure must not break the caller
            log.error("audit.write_failed", event_type=event_type, error=str(exc))

    def recent(self, limit: int = 50, event_type: str | None = None) -> list[AuditLog]:
        """Most recent entries, newest first."""
        with self.db.session() as session:
            statement = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
            if event_type:
                statement = statement.where(AuditLog.event_type == event_type)
            return list(session.scalars(statement).all())

    def for_symbol(self, symbol: str, limit: int = 50) -> list[AuditLog]:
        with self.db.session() as session:
            return list(
                session.scalars(
                    select(AuditLog)
                    .where(AuditLog.symbol == symbol.upper())
                    .order_by(AuditLog.id.desc())
                    .limit(limit)
                ).all()
            )
