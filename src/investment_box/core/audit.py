"""The audit sink contract.

Lives in ``core`` rather than ``services`` to keep the dependency graph acyclic.
Lower layers -- compliance screening, risk, execution -- all need to record what
they did, but none of them should know about the service layer. They depend on
this protocol; :class:`investment_box.services.audit.AuditService` satisfies it
structurally.

Without this, ``shariah.status`` imported ``services.audit``, ``services``
imported ``research``, ``research`` imported ``forecast``, and ``forecast``
imported ``shariah`` -- a cycle that only failed depending on which module was
imported first.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AuditSink(Protocol):
    """Somewhere to record a decision, an action or a refusal."""

    def record(
        self,
        event_type: str,
        summary: str,
        *,
        actor: str = "engine",
        symbol: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one entry. Must never raise into the caller."""
        ...


class NullAuditSink:
    """Discards everything. For tests and for code paths with no database."""

    def record(
        self,
        event_type: str,
        summary: str,
        *,
        actor: str = "engine",
        symbol: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        return None
