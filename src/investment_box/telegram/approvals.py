"""Telegram transport for approval requests.

The state machine lives in :mod:`investment_box.services.approvals`; this is
only the delivery and button-handling half. Keeping them apart means the
dashboard can answer the same request later without any Telegram involvement.

Behaviours worth naming:

* After a decision, the message's buttons are removed and its text is replaced
  with the outcome. A spent approval must not still look pressable -- the
  second tap is a no-op in the service, but a user should not have to find that
  out by tapping.
* The Modify flow parks the request in ``AWAITING_MODIFICATION`` and waits for
  a numeric reply. The original deadline still applies, so a forgotten
  modification expires like anything else.
* The message id is tracked so the message can be edited later, including by
  the expiry sweeper, which runs long after the button was drawn.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ApprovalAction, ApprovalKind, ApprovalStatus, TradingMode
from investment_box.i18n.translator import Translator
from investment_box.services.approvals import ApprovalRequest, ApprovalService
from investment_box.telegram.formatting import (
    approval_buttons,
    format_approval_request,
    format_approval_resolution,
)
from investment_box.telegram.queue import MessageQueue
from investment_box.telegram.transport import TelegramTransport

log = get_logger(__name__)

CALLBACK_PREFIX = "apv"
PRIORITY_APPROVAL = 80


@dataclass
class _Delivery:
    """Where an approval message ended up, so it can be edited later."""

    chat_id: str
    message_id: int


@dataclass
class ApprovalNotifier:
    """Sends approval requests and processes the button presses."""

    service: ApprovalService
    queue: MessageQueue
    transport: TelegramTransport
    chat_id: str | None
    trading_mode: TradingMode
    translator: Translator
    clock: Clock = field(default_factory=SystemClock)
    #: approval id -> where its message was delivered
    _deliveries: dict[int, _Delivery] = field(default_factory=dict)
    #: user id -> approval id they are currently supplying a new size for
    _awaiting_size: dict[int, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ send

    async def request(
        self,
        *,
        request_key: str,
        kind: ApprovalKind,
        payload: dict[str, Any],
        timeout_minutes: int | None = None,
    ) -> ApprovalRequest:
        """Open a request and send it with its buttons.

        Returns as soon as the request is recorded. Delivery is asynchronous --
        a Telegram outage delays the message, it does not block the caller or
        change the outcome, which defaults to rejection on timeout either way.
        """
        request = self.service.create(
            request_key=request_key,
            kind=kind,
            payload=payload,
            timeout_minutes=timeout_minutes,
        )
        if not self.chat_id:
            log.warning(
                "approval.no_chat_configured",
                approval_id=request.id,
                note="request recorded; it will expire unanswered",
            )
            return request

        text = format_approval_request(
            request, self.trading_mode, self.translator, now=self.clock.now()
        )
        future = self.queue.enqueue(
            self.chat_id,
            text,
            buttons=approval_buttons(request.id, self.translator),
            priority=PRIORITY_APPROVAL,
            want_result=True,
        )
        if future is not None:
            # Record where the message landed *when* it lands, via a callback.
            # Awaiting the future here would block the caller until Telegram
            # accepted the message -- and block forever if the queue worker is
            # not running. The engine must never wait on a notification, and a
            # request that is never delivered still expires into a rejection.
            approval_id = request.id

            def _record(fut: asyncio.Future[Any], _id: int = approval_id) -> None:
                self._on_delivered(_id, fut)

            future.add_done_callback(_record)
        return request

    def _on_delivered(self, approval_id: int, future: asyncio.Future[Any]) -> None:
        """Record the delivered message id, or log why there isn't one."""
        if future.cancelled():
            return
        error = future.exception()
        if error is not None:
            log.error("approval.delivery_failed", approval_id=approval_id, error=str(error))
            return
        sent = future.result()
        self._deliveries[approval_id] = _Delivery(sent.chat_id, sent.message_id)

    # -------------------------------------------------------------- callbacks

    @staticmethod
    def parse_callback(data: str) -> tuple[int, ApprovalAction] | None:
        """Parse ``apv:<id>:<action>``. Returns ``None`` if malformed.

        Callback data arrives from the network and is never trusted: an
        unparseable or unknown value is dropped rather than guessed at.
        """
        parts = data.split(":")
        if len(parts) != 3 or parts[0] != CALLBACK_PREFIX or not parts[1].isdigit():
            return None
        try:
            action = ApprovalAction(parts[2])
        except ValueError:
            return None
        return int(parts[1]), action

    async def handle_callback(
        self, *, data: str, user_id: int, callback_id: str | None = None
    ) -> ApprovalRequest | None:
        """Apply a button press. The caller must already have authorised ``user_id``."""
        parsed = self.parse_callback(data)
        if parsed is None:
            log.warning("approval.bad_callback", data=data[:64], user_id=user_id)
            if callback_id:
                await self.transport.answer_callback(callback_id)
            return None

        approval_id, action = parsed
        responder = str(user_id)

        current = self.service.get(approval_id)
        if current is None:
            if callback_id:
                await self.transport.answer_callback(callback_id, "Not found")
            return None

        if current.status.is_terminal:
            # Includes a request that expired while the message sat on screen.
            if callback_id:
                await self.transport.answer_callback(
                    callback_id, self.translator.t("approval.already_decided"), show_alert=True
                )
            await self._finalise_message(current)
            return current

        updated = await self._apply(approval_id, action, responder)
        if updated is None:
            if callback_id:
                await self.transport.answer_callback(callback_id)
            return None

        if callback_id:
            await self.transport.answer_callback(callback_id, self._toast(updated, action))

        if action is ApprovalAction.MODIFY and updated.status is (
            ApprovalStatus.AWAITING_MODIFICATION
        ):
            self._awaiting_size[user_id] = approval_id
            await self._replace_text(updated)
            return updated

        if action is ApprovalAction.SNOOZE and not updated.status.is_terminal:
            # Redraw with the extended deadline; the buttons stay live.
            await self._refresh(updated)
            return updated

        await self._finalise_message(updated)
        return updated

    async def _apply(
        self, approval_id: int, action: ApprovalAction, responder: str
    ) -> ApprovalRequest | None:
        match action:
            case ApprovalAction.APPROVE:
                return self.service.respond(approval_id, approved=True, responder_id=responder)
            case ApprovalAction.REJECT:
                return self.service.respond(approval_id, approved=False, responder_id=responder)
            case ApprovalAction.MODIFY:
                return self.service.request_modification(approval_id, responder)
            case ApprovalAction.SNOOZE:
                return self.service.snooze(approval_id, responder)
        # Every ApprovalAction member is handled above; parse_callback rejects
        # anything else before it reaches here.
        raise AssertionError(f"unhandled approval action: {action}")

    # ------------------------------------------------------- modification flow

    def is_awaiting_size(self, user_id: int) -> bool:
        return user_id in self._awaiting_size

    async def submit_size(self, user_id: int, text: str) -> ApprovalRequest | None:
        """Consume a numeric reply as the replacement size.

        Returns ``None`` when the text is not a usable quantity; the caller
        should then tell the user and leave the request parked.
        """
        approval_id = self._awaiting_size.get(user_id)
        if approval_id is None:
            return None

        updated = self.service.apply_modification(
            approval_id, new_quantity=text.strip(), responder_id=str(user_id)
        )
        if updated is None:
            return None  # invalid number; stay parked and let the caller reply

        self._awaiting_size.pop(user_id, None)
        await self._finalise_message(updated)
        return updated

    def cancel_size_request(self, user_id: int) -> None:
        self._awaiting_size.pop(user_id, None)

    # ----------------------------------------------------------------- expiry

    async def sweep_expired(self) -> list[ApprovalRequest]:
        """Expire due requests and update their messages.

        Scheduled. The service's read paths already treat a past-deadline
        request as expired, so this exists to make the timeout *visible* rather
        than to make it correct.
        """
        expired = self.service.expire_due()
        for request in expired:
            await self._finalise_message(request)
        return expired

    # -------------------------------------------------------------- messaging

    async def _finalise_message(self, request: ApprovalRequest) -> None:
        """Strip the buttons and replace the text with the outcome."""
        delivery = self._deliveries.get(request.id)
        if delivery is None:
            return
        text = format_approval_resolution(request, self.trading_mode, self.translator)
        await self.transport.edit_message_buttons(delivery.chat_id, delivery.message_id, None)
        await self.transport.edit_message_text(delivery.chat_id, delivery.message_id, text)
        if request.status.is_terminal:
            self._deliveries.pop(request.id, None)

    async def _replace_text(self, request: ApprovalRequest) -> None:
        """Update the text but keep the keyboard, for non-terminal transitions."""
        delivery = self._deliveries.get(request.id)
        if delivery is None:
            return
        text = format_approval_resolution(request, self.trading_mode, self.translator)
        await self.transport.edit_message_buttons(delivery.chat_id, delivery.message_id, None)
        await self.transport.edit_message_text(delivery.chat_id, delivery.message_id, text)

    async def _refresh(self, request: ApprovalRequest) -> None:
        """Redraw a still-live request, e.g. after a snooze extended it."""
        delivery = self._deliveries.get(request.id)
        if delivery is None:
            return
        text = format_approval_request(
            request, self.trading_mode, self.translator, now=self.clock.now()
        )
        await self.transport.edit_message_text(delivery.chat_id, delivery.message_id, text)

    def _toast(self, request: ApprovalRequest, action: ApprovalAction) -> str:
        key = {
            ApprovalAction.APPROVE: "approval.approved_by",
            ApprovalAction.REJECT: "approval.rejected_by",
            ApprovalAction.MODIFY: "approval.send_new_size",
            ApprovalAction.SNOOZE: "approval.snoozed",
        }[action]
        if request.status is ApprovalStatus.EXPIRED:
            key = "approval.expired"
        return Translator("en").t(key)

    # ------------------------------------------------------------- test hooks

    def record_delivery(self, approval_id: int, chat_id: str, message_id: int) -> None:
        """Register where a message landed. Used by tests and by recovery."""
        self._deliveries[approval_id] = _Delivery(chat_id, message_id)

    def delivery_for(self, approval_id: int) -> tuple[str, int] | None:
        delivery = self._deliveries.get(approval_id)
        return (delivery.chat_id, delivery.message_id) if delivery else None


def utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)
