"""Bot assembly and update routing.

This is the only module that knows about ``python-telegram-bot``'s update
objects. It extracts the few fields that matter, authorises the sender, and
hands off to the transport-agnostic handlers below it.

The ordering in :meth:`TelegramBot.handle_message` is deliberate: authorise
first, then route. There is no code path that reaches a handler before the
whitelist check.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

from investment_box.config.schema import Secrets, Settings
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.i18n.translator import Translator
from investment_box.services.approvals import ApprovalService
from investment_box.services.audit import AuditService
from investment_box.services.portfolio import PortfolioService
from investment_box.telegram.approvals import ApprovalNotifier
from investment_box.telegram.auth import AuthGuard
from investment_box.telegram.broadcast import BroadcastChannel
from investment_box.telegram.commands import CommandContext, CommandHandlers
from investment_box.telegram.queue import MessageQueue
from investment_box.telegram.transport import FakeTransport, PTBTransport, TelegramTransport

log = get_logger(__name__)

#: How often expired approvals are swept.
SWEEP_INTERVAL_SECONDS = 60


@dataclass
class TelegramStack:
    """Everything Telegram-related, constructed together."""

    transport: TelegramTransport
    queue: MessageQueue
    guard: AuthGuard
    broadcast: BroadcastChannel
    notifier: ApprovalNotifier
    handlers: CommandHandlers
    bot: TelegramBot

    @property
    def is_live_transport(self) -> bool:
        return self.transport.name == "telegram"


class TelegramBot:
    """Routes authorised updates to the command handlers."""

    def __init__(
        self,
        *,
        guard: AuthGuard,
        handlers: CommandHandlers,
        notifier: ApprovalNotifier,
        queue: MessageQueue,
        translator: Translator,
        clock: Clock | None = None,
    ) -> None:
        self.guard = guard
        self.handlers = handlers
        self.notifier = notifier
        self.queue = queue
        self.translator = translator
        self.clock = clock or SystemClock()
        self._sweeper: asyncio.Task[None] | None = None

    # ---------------------------------------------------------------- updates

    async def handle_message(
        self,
        *,
        user_id: int | None,
        chat_id: int | str,
        text: str,
        username: str | None = None,
    ) -> str | None:
        """Process one incoming message.

        Returns the reply text, or ``None`` when the message is ignored --
        which is what an unauthorised sender gets. Not a refusal: a reply of
        any kind confirms the bot is here and listening.
        """
        preview = (text or "").strip()
        if not self.guard.check(user_id, context=preview[:32] or "<empty>", username=username):
            return None

        assert user_id is not None  # guaranteed by the guard

        # A pending Modify takes precedence: the user's next message is the size.
        if self.notifier.is_awaiting_size(user_id):
            updated = await self.notifier.submit_size(user_id, preview)
            if updated is None:
                reply = self.translator.t("approval.invalid_size")
                self.queue.enqueue(str(chat_id), reply, priority=80)
                return reply
            reply = self.translator.t("approval.approved_by")
            self.queue.enqueue(str(chat_id), reply, priority=80)
            return reply

        if not preview.startswith("/"):
            return None  # not a command; stay quiet rather than chattering

        command, _, argument = preview.partition(" ")
        reply = self.handlers.dispatch(command, argument or None)
        self.queue.enqueue(str(chat_id), reply, priority=60)
        return reply

    async def handle_callback(
        self,
        *,
        user_id: int | None,
        data: str,
        callback_id: str | None = None,
        username: str | None = None,
    ) -> Any:
        """Process an inline-button press.

        Callbacks are authorised exactly like messages. A forwarded message
        carries its keyboard, so skipping this check would let anyone who
        received a forward approve a trade.
        """
        if not self.guard.check(user_id, context=f"callback:{data[:24]}", username=username):
            return None
        assert user_id is not None
        return await self.notifier.handle_callback(
            data=data, user_id=user_id, callback_id=callback_id
        )

    # -------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        await self.queue.start()
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_loop(), name="approval-sweeper")

    async def stop(self) -> None:
        if self._sweeper is not None:
            self._sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sweeper
            self._sweeper = None
        await self.queue.stop()

    async def _sweep_loop(self) -> None:
        """Expire timed-out approvals on a schedule.

        Wrapped so a failure retries on the next tick instead of killing the
        task -- a dead sweeper would leave stale requests on screen, and the
        service would still refuse to honour them, but silently.
        """
        while True:
            try:
                await asyncio.sleep(SWEEP_INTERVAL_SECONDS)
                expired = await self.notifier.sweep_expired()
                if expired:
                    log.info("approval.swept", count=len(expired))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry next tick rather than die
                log.error("approval.sweep_failed", error=str(exc))


def build_telegram_stack(
    *,
    settings: Settings,
    secrets: Secrets,
    portfolio: PortfolioService,
    approvals: ApprovalService,
    audit: AuditService,
    clock: Clock | None = None,
    transport: TelegramTransport | None = None,
    data_provider_name: str = "unknown",
    engine_state: str = "idle",
) -> TelegramStack:
    """Wire the Telegram stack from config.

    With no bot token, this builds the whole stack on :class:`FakeTransport`.
    Messages are recorded and discarded rather than sent, so every other part
    of the system behaves identically whether or not Telegram is configured.
    """
    clock = clock or SystemClock()
    translator = Translator(settings.i18n.language)

    if transport is None:
        token = (
            secrets.telegram_bot_token.get_secret_value()
            if secrets.telegram_bot_token
            else None
        )
        if token:
            transport = PTBTransport(token)
        else:
            log.warning(
                "telegram.no_token",
                action="using the in-memory transport",
                hint="set TELEGRAM_BOT_TOKEN in .env to send real messages",
            )
            transport = FakeTransport()

    queue = MessageQueue(transport)
    guard = AuthGuard(allowed_user_ids=secrets.allowed_telegram_ids, audit=audit)

    if not guard.is_enabled:
        log.warning(
            "telegram.no_allowed_users",
            action="the interactive bot will answer nobody",
            hint="set TELEGRAM_ALLOWED_USER_IDS in .env",
        )

    channel_problem = secrets.telegram_channel_problem
    if channel_problem:
        log.warning("telegram.channel_id_malformed", problem=channel_problem)

    broadcast = BroadcastChannel(
        queue=queue,
        channel_id=secrets.telegram_channel_id,
        trading_mode=settings.trading_mode,
        translator=translator,
    )

    # Approval requests go to the private chat -- the first whitelisted user --
    # never to the broadcast channel, which may have other readers.
    approval_chat = (
        str(min(secrets.allowed_telegram_ids)) if secrets.allowed_telegram_ids else None
    )
    notifier = ApprovalNotifier(
        service=approvals,
        queue=queue,
        transport=transport,
        chat_id=approval_chat,
        trading_mode=settings.trading_mode,
        translator=translator,
        clock=clock,
    )

    handlers = CommandHandlers(
        CommandContext(
            settings=settings,
            portfolio=portfolio,
            approvals=approvals,
            queue=queue,
            translator=translator,
            clock=clock,
            engine_state=engine_state,
            data_provider_name=data_provider_name,
        )
    )

    bot = TelegramBot(
        guard=guard,
        handlers=handlers,
        notifier=notifier,
        queue=queue,
        translator=translator,
        clock=clock,
    )

    return TelegramStack(
        transport=transport,
        queue=queue,
        guard=guard,
        broadcast=broadcast,
        notifier=notifier,
        handlers=handlers,
        bot=bot,
    )
