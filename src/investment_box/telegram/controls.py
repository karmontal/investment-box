"""Destructive Telegram commands, behind confirmations.

``/pause``, ``/resume``, ``/kill`` and ``/purification``. Everything that
changes state or closes positions requires a second, explicit tap.

Two rules that do not bend:

* **Live trading cannot be enabled from Telegram**, by anyone, ever. It needs a
  typed confirmation in the dashboard. A chat interface is the wrong place for
  an irreversible money-at-risk decision, and a stolen phone must not be able
  to make it. ``AuthGuard.assert_action_allowed`` enforces this.
* **A confirmation expires.** A pending ``/kill`` that sits unanswered for
  minutes must not still be live when someone finally taps it -- by then the
  situation that prompted it has changed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal

from investment_box.core.audit import AuditSink
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import TradingMode
from investment_box.i18n.translator import Translator
from investment_box.telegram.auth import AuthGuard
from investment_box.telegram.formatting import esc, mode_tag, money
from investment_box.telegram.transport import InlineButton

log = get_logger(__name__)

CALLBACK_PREFIX = "ctl"
#: How long a pending confirmation stays valid. Short: the situation that
#: prompted a kill changes fast.
CONFIRMATION_TIMEOUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class PendingConfirmation:
    """A destructive action awaiting its second tap."""

    action: str
    user_id: int
    requested_at: dt.datetime
    detail: str = ""
    close_positions: bool = False

    def is_expired(self, now: dt.datetime) -> bool:
        return (now - self.requested_at).total_seconds() > CONFIRMATION_TIMEOUT_SECONDS


@dataclass
class ControlResult:
    """What a command did, and what to say back."""

    message: str
    buttons: list[list[InlineButton]] | None = None
    changed_state: bool = False


@dataclass
class ControlCommands:
    """Handlers for the commands that change something."""

    guard: AuthGuard
    audit: AuditSink
    trading_mode: TradingMode
    translator: Translator
    #: Injected so this module does not import the engine, which imports
    #: services, which would create a cycle.
    engine: object | None = None
    purification: object | None = None
    clock: Clock = field(default_factory=SystemClock)
    _pending: dict[int, PendingConfirmation] = field(default_factory=dict)

    # ------------------------------------------------------------- dispatch

    def handle(self, command: str, user_id: int, argument: str | None = None) -> ControlResult:
        """Route a control command. The caller has already authorised the user."""
        name = command.lstrip("/").split("@")[0].lower()
        match name:
            case "pause":
                return self._request(user_id, "pause", "Pause the engine?")
            case "resume":
                return self._request(user_id, "resume", "Resume the engine?")
            case "kill":
                return self._kill_prompt(user_id, argument)
            case "purification":
                return self._purification()
        return ControlResult(message=self._tag("Unknown control command."))

    def handle_callback(self, data: str, user_id: int) -> ControlResult | None:
        """Apply a confirmation tap."""
        parsed = self.parse_callback(data)
        if parsed is None:
            return None
        action, choice = parsed

        pending = self._pending.get(user_id)
        if pending is None or pending.action != action:
            return ControlResult(
                message=self._tag("That confirmation is no longer pending.")
            )

        if pending.is_expired(self.clock.now()):
            self._pending.pop(user_id, None)
            return ControlResult(
                message=self._tag(
                    "That confirmation expired. Issue the command again if you still "
                    "want it — the situation may have changed."
                )
            )

        self._pending.pop(user_id, None)

        if choice != "yes":
            self.audit.record(
                f"control.{action}_cancelled", f"{action} cancelled", actor=str(user_id)
            )
            return ControlResult(message=self._tag("Cancelled. Nothing was changed."))

        return self._execute(action, user_id, pending)

    @staticmethod
    def parse_callback(data: str) -> tuple[str, str] | None:
        """Parse ``ctl:<action>:<yes|no>``. Untrusted input, so never guessed at."""
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != CALLBACK_PREFIX:
            return None
        if parts[2] not in ("yes", "no"):
            return None
        return parts[1], parts[2]

    # -------------------------------------------------------------- commands

    def _request(self, user_id: int, action: str, question: str) -> ControlResult:
        self.guard.assert_action_allowed(action)
        self._pending[user_id] = PendingConfirmation(
            action=action, user_id=user_id, requested_at=self.clock.now()
        )
        return ControlResult(
            message=self._tag(f"<b>{esc(question)}</b>"),
            buttons=self._confirm_buttons(action),
        )

    def _kill_prompt(self, user_id: int, argument: str | None) -> ControlResult:
        """The kill switch, with the exposure it would affect stated up front."""
        self.guard.assert_action_allowed("kill")
        close_positions = bool(argument and argument.strip().lower() in ("all", "close"))

        exposure = Decimal("0")
        if self.engine is not None:
            try:
                exposure = self.engine.kill_switch.estimate_exposure()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - the prompt must still appear
                log.warning("controls.exposure_failed", error=str(exc))

        self._pending[user_id] = PendingConfirmation(
            action="kill",
            user_id=user_id,
            requested_at=self.clock.now(),
            close_positions=close_positions,
        )

        lines = [
            "<b>🚨 KILL SWITCH</b>",
            "",
            "This will cancel every open order and stop the engine.",
        ]
        if close_positions:
            lines.append(
                f"It will ALSO close every position at market "
                f"({money(exposure)} exposure). This realises losses and starts the "
                f"T+1 settlement clock."
            )
        else:
            lines.append(
                f"Positions will be LEFT OPEN ({money(exposure)} exposure). "
                f"Send <code>/kill all</code> to close them too."
            )
        lines += ["", "<i>The engine cannot be resumed without a restart.</i>"]

        return ControlResult(
            message=self._tag("\n".join(lines)), buttons=self._confirm_buttons("kill")
        )

    def _purification(self) -> ControlResult:
        if self.purification is None:
            return ControlResult(
                message=self._tag("Purification tracking is not configured.")
            )
        report = self.purification.report()  # type: ignore[attr-defined]
        lines = [
            "<b>Purification</b>",
            "",
            f"Dividends recorded: {money(report.total_dividends)}",
            f"Total to purify:    {money(report.total_due)}",
            f"<b>Outstanding:        {money(report.outstanding)}</b>",
            f"Already purified:   {money(report.already_purified)}",
        ]
        for warning in report.warnings():
            lines += ["", f"⚠️ {esc(warning)}"]
        return ControlResult(message=self._tag("\n".join(lines)))

    # ------------------------------------------------------------- execution

    def _execute(
        self, action: str, user_id: int, pending: PendingConfirmation
    ) -> ControlResult:
        if self.engine is None:
            return ControlResult(
                message=self._tag("No engine is attached; nothing to control.")
            )

        actor = str(user_id)
        match action:
            case "pause":
                self.engine.state.pause("paused from Telegram", actor=actor)  # type: ignore[attr-defined]
                self.engine.risk.pause("paused from Telegram", actor=actor)  # type: ignore[attr-defined]
                return ControlResult(
                    message=self._tag("⏸ <b>Engine paused.</b> Exits are still allowed."),
                    changed_state=True,
                )
            case "resume":
                if not self.engine.state.resume(actor=actor):  # type: ignore[attr-defined]
                    return ControlResult(
                        message=self._tag(
                            "Could not resume. If the kill switch was used, the engine "
                            "requires a restart."
                        )
                    )
                self.engine.risk.resume(actor=actor)  # type: ignore[attr-defined]
                return ControlResult(
                    message=self._tag("▶️ <b>Engine resumed.</b>"), changed_state=True
                )
            case "kill":
                result = self.engine.kill(  # type: ignore[attr-defined]
                    f"kill switch from Telegram (user {actor})",
                    close_positions=pending.close_positions,
                )
                return ControlResult(
                    message=self._tag(
                        f"🚨 <b>KILL SWITCH ACTIVATED</b>\n\n{esc(result.detail())}"
                    ),
                    changed_state=True,
                )
        return ControlResult(message=self._tag("Unknown action."))

    # --------------------------------------------------------------- helpers

    def _confirm_buttons(self, action: str) -> list[list[InlineButton]]:
        return [
            [
                InlineButton("✅ Confirm", f"{CALLBACK_PREFIX}:{action}:yes"),
                InlineButton("❌ Cancel", f"{CALLBACK_PREFIX}:{action}:no"),
            ]
        ]

    def _tag(self, body: str) -> str:
        return f"{mode_tag(self.trading_mode)}\n{body}"

    def pending_for(self, user_id: int) -> PendingConfirmation | None:
        return self._pending.get(user_id)
