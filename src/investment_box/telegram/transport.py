"""Transport abstraction over the Telegram Bot API.

Everything above this module talks to :class:`TelegramTransport`, never to
``python-telegram-bot`` directly. That buys two things: the whole stack is
testable without a network or a token, and swapping the client library later
touches one file.

:class:`FakeTransport` is not only a test double -- it is also what runs when
no bot token is configured, so the engine behaves identically with and without
Telegram wired up.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from investment_box.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class InlineButton:
    """One inline-keyboard button.

    ``callback_data`` is capped at 64 bytes by Telegram. Exceeding it makes the
    API reject the whole message, so it is validated here rather than
    discovered in production.
    """

    text: str
    callback_data: str

    def __post_init__(self) -> None:
        encoded = self.callback_data.encode("utf-8")
        if len(encoded) > 64:
            raise ValueError(
                f"callback_data is {len(encoded)} bytes; Telegram's limit is 64. "
                f"Store the payload and reference it by id instead: {self.callback_data!r}"
            )
        if not self.callback_data:
            raise ValueError("callback_data must not be empty")


@dataclass(frozen=True, slots=True)
class SentMessage:
    """What the transport reports after a successful send."""

    chat_id: str
    message_id: int
    text: str
    sent_at: dt.datetime


@runtime_checkable
class TelegramTransport(Protocol):
    """The surface the rest of the application uses."""

    name: str

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        buttons: list[list[InlineButton]] | None = None,
        parse_mode: str | None = "HTML",
        disable_notification: bool = False,
    ) -> SentMessage: ...

    async def edit_message_buttons(
        self, chat_id: str, message_id: int, buttons: list[list[InlineButton]] | None
    ) -> bool:
        """Replace a message's inline keyboard. Used to disable spent buttons."""
        ...

    async def edit_message_text(
        self, chat_id: str, message_id: int, text: str, *, parse_mode: str | None = "HTML"
    ) -> bool: ...

    async def answer_callback(
        self, callback_id: str, text: str | None = None, *, show_alert: bool = False
    ) -> bool:
        """Acknowledge a button press so Telegram stops showing a spinner."""
        ...


class PTBTransport:
    """Real transport, backed by ``python-telegram-bot``."""

    name = "telegram"

    def __init__(self, token: str) -> None:
        from telegram import Bot
        from telegram.request import HTTPXRequest

        self._bot = Bot(
            token=token,
            request=HTTPXRequest(connection_pool_size=8, read_timeout=20, write_timeout=20),
        )

    @property
    def bot(self) -> Any:
        return self._bot

    @staticmethod
    def _markup(buttons: list[list[InlineButton]] | None) -> Any:
        if not buttons:
            return None
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(b.text, callback_data=b.callback_data) for b in row]
                for row in buttons
            ]
        )

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        buttons: list[list[InlineButton]] | None = None,
        parse_mode: str | None = "HTML",
        disable_notification: bool = False,
    ) -> SentMessage:
        message = await self._bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=parse_mode,
            reply_markup=self._markup(buttons),
            disable_notification=disable_notification,
        )
        return SentMessage(
            chat_id=str(chat_id),
            message_id=message.message_id,
            text=text,
            sent_at=dt.datetime.now(tz=dt.UTC),
        )

    async def edit_message_buttons(
        self, chat_id: str, message_id: int, buttons: list[list[InlineButton]] | None
    ) -> bool:
        from telegram.error import TelegramError

        try:
            await self._bot.edit_message_reply_markup(
                chat_id=chat_id, message_id=message_id, reply_markup=self._markup(buttons)
            )
        except TelegramError as exc:
            # "message is not modified" is benign and common on a double tap.
            if "not modified" in str(exc).lower():
                return True
            log.warning("telegram.edit_markup_failed", error=str(exc))
            return False
        return True

    async def edit_message_text(
        self, chat_id: str, message_id: int, text: str, *, parse_mode: str | None = "HTML"
    ) -> bool:
        from telegram.error import TelegramError

        try:
            await self._bot.edit_message_text(
                chat_id=chat_id, message_id=message_id, text=text, parse_mode=parse_mode
            )
        except TelegramError as exc:
            if "not modified" in str(exc).lower():
                return True
            log.warning("telegram.edit_text_failed", error=str(exc))
            return False
        return True

    async def answer_callback(
        self, callback_id: str, text: str | None = None, *, show_alert: bool = False
    ) -> bool:
        """Acknowledge a button press. Never raises.

        This call is cosmetic -- it stops the spinner on the user's screen --
        but it fails routinely in normal use: Telegram rejects a callback query
        older than about a minute with "Query is too old", which happens
        whenever someone taps a button on a message they scrolled back to, or
        after the bot restarts.

        By the time this runs the decision has already been recorded, so
        letting the failure propagate would crash the update handler *after*
        the state change it was reporting on. It is logged instead.
        """
        from telegram.error import TelegramError

        try:
            await self._bot.answer_callback_query(
                callback_query_id=callback_id, text=text, show_alert=show_alert
            )
        except TelegramError as exc:
            log.warning("telegram.answer_callback_failed", error=str(exc))
            return False
        return True


@dataclass
class FakeTransport:
    """In-memory transport.

    Records everything instead of sending it. Used by the tests, and used in
    production whenever no bot token is configured -- so a missing token
    degrades to silence rather than to a crash.
    """

    name: str = "fake"
    sent: list[SentMessage] = field(default_factory=list)
    buttons_sent: list[list[list[InlineButton]] | None] = field(default_factory=list)
    edits: list[tuple[str, int, str]] = field(default_factory=list)
    markup_edits: list[tuple[str, int, list[list[InlineButton]] | None]] = field(
        default_factory=list
    )
    answered_callbacks: list[tuple[str, str | None]] = field(default_factory=list)
    #: Set to raise on the next N sends, to exercise retry and backoff.
    fail_next: int = 0
    failures_raised: int = 0
    _next_id: int = 1000

    async def send_message(
        self,
        chat_id: str,
        text: str,
        *,
        buttons: list[list[InlineButton]] | None = None,
        parse_mode: str | None = "HTML",
        disable_notification: bool = False,
    ) -> SentMessage:
        if self.fail_next > 0:
            self.fail_next -= 1
            self.failures_raised += 1
            raise ConnectionError("simulated Telegram failure")

        self._next_id += 1
        message = SentMessage(
            chat_id=str(chat_id),
            message_id=self._next_id,
            text=text,
            sent_at=dt.datetime.now(tz=dt.UTC),
        )
        self.sent.append(message)
        self.buttons_sent.append(buttons)
        return message

    async def edit_message_buttons(
        self, chat_id: str, message_id: int, buttons: list[list[InlineButton]] | None
    ) -> bool:
        self.markup_edits.append((str(chat_id), message_id, buttons))
        return True

    async def edit_message_text(
        self, chat_id: str, message_id: int, text: str, *, parse_mode: str | None = "HTML"
    ) -> bool:
        self.edits.append((str(chat_id), message_id, text))
        return True

    async def answer_callback(
        self, callback_id: str, text: str | None = None, *, show_alert: bool = False
    ) -> bool:
        self.answered_callbacks.append((callback_id, text))
        return True

    # ------------------------------------------------------------- test helpers

    @property
    def texts(self) -> list[str]:
        return [message.text for message in self.sent]

    def last_text(self) -> str | None:
        return self.sent[-1].text if self.sent else None

    def clear(self) -> None:
        self.sent.clear()
        self.buttons_sent.clear()
        self.edits.clear()
        self.markup_edits.clear()
        self.answered_callbacks.clear()
