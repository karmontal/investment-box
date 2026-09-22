"""Engine state machine.

Small on purpose. The engine has four states and the transitions between them
are explicit, because "is it running?" must have an unambiguous answer that
both the dashboard and ``/status`` read the same way.

``KILLED`` is terminal within a process: the kill switch is not a pause. It
requires a deliberate restart, so a stray ``/resume`` cannot undo it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum

from investment_box.core.audit import AuditSink
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger

log = get_logger(__name__)


class EngineState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    #: Kill switch pulled. Terminal until the process restarts.
    KILLED = "killed"

    @property
    def can_open_positions(self) -> bool:
        return self is EngineState.RUNNING

    @property
    def can_close_positions(self) -> bool:
        """Exits are always permitted except after a kill.

        Being unable to close a position because the engine is paused would be
        strictly worse than the condition that caused the pause.
        """
        return self is not EngineState.KILLED


@dataclass
class EngineStatus:
    """Current state plus the history of how it got there."""

    state: EngineState = EngineState.IDLE
    reason: str = ""
    changed_at: dt.datetime | None = None
    last_cycle_at: dt.datetime | None = None
    last_cycle_summary: str = ""
    cycles_run: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def is_running(self) -> bool:
        return self.state is EngineState.RUNNING

    def describe(self) -> str:
        base = self.state.value
        return f"{base}: {self.reason}" if self.reason else base


class EngineStateMachine:
    """Owns the engine's state and audits every transition."""

    def __init__(
        self, audit: AuditSink, *, clock: Clock | None = None
    ) -> None:
        self.audit = audit
        self.clock = clock or SystemClock()
        self.status = EngineStatus()

    @property
    def state(self) -> EngineState:
        return self.status.state

    def _transition(self, to: EngineState, reason: str, actor: str) -> bool:
        if self.status.state is EngineState.KILLED and to is not EngineState.KILLED:
            log.warning("engine.transition_refused", attempted=to.value, reason="killed")
            return False
        if self.status.state is to:
            return False

        previous = self.status.state
        self.status.state = to
        self.status.reason = reason
        self.status.changed_at = self.clock.now()

        log.info(
            "engine.state_changed",
            **{"from": previous.value, "to": to.value, "reason": reason},
        )
        self.audit.record(
            f"engine.{to.value}",
            f"{previous.value} -> {to.value}: {reason}",
            actor=actor,
        )
        return True

    def start(self, *, actor: str = "scheduler") -> bool:
        return self._transition(EngineState.RUNNING, "started", actor)

    def pause(self, reason: str, *, actor: str = "user") -> bool:
        return self._transition(EngineState.PAUSED, reason, actor)

    def resume(self, *, actor: str = "user") -> bool:
        return self._transition(EngineState.RUNNING, "resumed", actor)

    def stop(self, reason: str = "stopped", *, actor: str = "scheduler") -> bool:
        return self._transition(EngineState.IDLE, reason, actor)

    def kill(self, reason: str, *, actor: str = "user") -> bool:
        """Pull the kill switch. Terminal until restart."""
        previous = self.status.state
        self.status.state = EngineState.KILLED
        self.status.reason = reason
        self.status.changed_at = self.clock.now()
        log.error("engine.killed", reason=reason, actor=actor, was=previous.value)
        self.audit.record(
            "engine.killed", f"KILL SWITCH: {reason}", actor=actor,
            detail={"previous_state": previous.value},
        )
        return True

    def record_cycle(self, summary: str) -> None:
        self.status.cycles_run += 1
        self.status.last_cycle_at = self.clock.now()
        self.status.last_cycle_summary = summary

    def record_error(self, message: str) -> None:
        self.status.errors.append(message)
        # Keep only the recent ones; an unbounded list would grow forever in a
        # long-running process.
        self.status.errors = self.status.errors[-20:]
