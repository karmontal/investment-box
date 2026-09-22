"""Authorisation for the interactive bot.

The rule: only whitelisted numeric Telegram user IDs get a response. Everything
else is ignored and logged.

Three details that matter more than they look:

* **An empty whitelist authorises nobody.** A misconfigured or missing
  ``TELEGRAM_ALLOWED_USER_IDS`` must not mean "allow everyone". It fails closed
  and the bot simply does not answer.
* **Callback queries are checked too.** A forwarded message carries its inline
  keyboard with it, so a button can be pressed by someone who never sent a
  command. Checking only ``/commands`` would leave approvals wide open.
* **Unauthorised attempts are never answered, not even with a refusal.** A
  reply confirms the bot exists and is listening. Silence does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from investment_box.core.logging import get_logger
from investment_box.services.audit import AuditService

log = get_logger(__name__)

#: Commands that can never be issued from Telegram, whatever the user id.
#: Switching to live trading is a dashboard-only action with a typed
#: confirmation; a chat interface is the wrong place for an irreversible
#: money-at-risk decision, and a compromised phone must not be able to make it.
TELEGRAM_FORBIDDEN_ACTIONS: frozenset[str] = frozenset(
    {"set_live_mode", "switch_to_live", "enable_live_trading"}
)

#: Actions that require a second confirming tap before they take effect.
TELEGRAM_CONFIRM_REQUIRED: frozenset[str] = frozenset(
    {"kill", "close_all_positions", "set_autonomy", "pause", "resume"}
)


@dataclass
class AuthGuard:
    """Decides whether a Telegram user may interact with the bot."""

    allowed_user_ids: frozenset[int]
    audit: AuditService | None = None
    #: Unauthorised ids seen this session, to avoid logging a flood from one
    #: persistent stranger while still recording that it happened.
    _seen_unauthorised: set[int] = field(default_factory=set)

    @property
    def is_enabled(self) -> bool:
        """Whether anyone at all can use the bot."""
        return bool(self.allowed_user_ids)

    def is_authorised(self, user_id: int | None) -> bool:
        return user_id is not None and user_id in self.allowed_user_ids

    def check(
        self, user_id: int | None, *, context: str, username: str | None = None
    ) -> bool:
        """Authorise an interaction, recording any rejection.

        Args:
            user_id: The Telegram numeric id, or ``None`` for an update that
                carries no sender (channel posts, service messages).
            context: What was attempted, for the log -- e.g. ``"/balance"``.
            username: Display name if known. Recorded, never trusted: a
                username is user-settable and is not an identity.

        Returns:
            ``True`` if the interaction may proceed.
        """
        if self.is_authorised(user_id):
            return True

        first_time = user_id not in self._seen_unauthorised
        if user_id is not None:
            self._seen_unauthorised.add(user_id)

        if first_time:
            log.warning(
                "telegram.unauthorised",
                user_id=user_id,
                username=username,
                context=context,
                configured_ids=len(self.allowed_user_ids),
            )
            if self.audit is not None:
                self.audit.record(
                    "telegram.unauthorised_access",
                    f"Ignored {context} from unauthorised user {user_id}",
                    actor="telegram",
                    detail={"user_id": user_id, "username": username, "context": context},
                )
        return False

    def assert_action_allowed(self, action: str) -> None:
        """Refuse actions that Telegram may never perform.

        Raises:
            PermissionError: If the action is forbidden from this transport.
        """
        if action in TELEGRAM_FORBIDDEN_ACTIONS:
            raise PermissionError(
                f"'{action}' cannot be performed from Telegram. Switching to live "
                f"trading requires a typed confirmation in the dashboard."
            )

    @staticmethod
    def requires_confirmation(action: str) -> bool:
        return action in TELEGRAM_CONFIRM_REQUIRED
