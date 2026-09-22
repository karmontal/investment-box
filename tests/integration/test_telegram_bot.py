"""End-to-end bot behaviour against the in-memory transport.

Covers the full path an update takes: authorisation, routing, formatting and
delivery through the queue -- and the approval flow including the case the
whole design turns on, a request that times out unanswered.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from investment_box.core.clock import FrozenClock
from investment_box.core.types import ApprovalKind, ApprovalStatus, OrderType, Side, Symbol
from investment_box.execution.base import OrderRequest
from investment_box.execution.mock_broker import MockBroker
from investment_box.telegram.bot import TelegramStack
from investment_box.telegram.transport import FakeTransport

#: Must match the `telegram_secrets` fixture. TestFixtureConsistency guards that.
ALLOWED_USER = 555000111
OTHER_ALLOWED_USER = 555000222
STRANGER = 999999999

PROPOSAL = {
    "symbol": "SPUS",
    "side": "buy",
    "quantity": "2",
    "entry_price": "59.83",
    "size_pct": 0.24,
    "stop_loss": "55.04",
    "take_profit": "65.21",
    "strategy": "etf_momentum_rotation",
    "probability": 0.58,
    "compliance_status": "compliant",
    "reason": "top-ranked on 3m risk-adjusted momentum",
    "backtest_summary": "142 trades, 54% win rate, Sharpe 0.71 (out of sample)",
}


async def send(stack: TelegramStack, text: str, user_id: int = ALLOWED_USER) -> str | None:
    return await stack.bot.handle_message(user_id=user_id, chat_id=user_id, text=text)


class TestFixtureConsistency:
    def test_constants_match_the_configured_whitelist(self, stack: TelegramStack) -> None:
        """Guards against these drifting apart and quietly weakening the auth tests."""
        assert stack.guard.allowed_user_ids == frozenset({ALLOWED_USER, OTHER_ALLOWED_USER})
        assert STRANGER not in stack.guard.allowed_user_ids


class TestAuthorisation:
    async def test_stranger_gets_no_reply_at_all(self, stack: TelegramStack) -> None:
        """Silence, not a refusal: a reply confirms the bot exists."""
        assert await send(stack, "/balance", user_id=STRANGER) is None
        assert stack.transport.sent == []

    async def test_allowed_user_is_answered(self, stack: TelegramStack) -> None:
        assert await send(stack, "/balance") is not None

    async def test_second_allowed_user_is_answered(self, stack: TelegramStack) -> None:
        assert await send(stack, "/balance", user_id=OTHER_ALLOWED_USER) is not None

    async def test_stranger_cannot_press_a_button(self, stack: TelegramStack) -> None:
        """A forwarded message carries its keyboard, so callbacks need the check too."""
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        result = await stack.bot.handle_callback(
            user_id=STRANGER, data=f"apv:{request.id}:approve"
        )
        assert result is None
        assert stack.notifier.service.get(request.id).status is ApprovalStatus.PENDING


class TestReadOnlyCommands:
    async def test_help_lists_the_commands(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/help")
        assert reply is not None
        for command in ("/status", "/balance", "/positions", "/funds", "/history", "/pending"):
            assert command in reply

    async def test_help_states_that_live_is_dashboard_only(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/help")
        assert reply is not None
        assert "dashboard" in reply.lower()

    async def test_balance_reports_the_account(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/balance")
        assert reply is not None
        assert "$500.00" in reply

    async def test_balance_separates_settled_from_unsettled(
        self, stack: TelegramStack, broker: MockBroker
    ) -> None:
        broker.submit_order(
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("10"),
                order_type=OrderType.MARKET,
                idempotency_key="a",
            )
        )
        broker.submit_order(
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.SELL,
                quantity=Decimal("10"),
                order_type=OrderType.MARKET,
                idempotency_key="b",
            )
        )
        reply = await send(stack, "/balance")
        assert reply is not None
        assert "$450.00" in reply  # the unsettled proceeds

    async def test_positions_when_flat(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/positions")
        assert reply is not None
        assert "No open positions" in reply

    async def test_positions_shows_a_holding(
        self, stack: TelegramStack, broker: MockBroker
    ) -> None:
        broker.submit_order(
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("2"),
                order_type=OrderType.MARKET,
                idempotency_key="k",
            )
        )
        reply = await send(stack, "/positions")
        assert reply is not None
        assert "SPUS" in reply

    async def test_status_reports_mode_and_autonomy(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/status")
        assert reply is not None
        assert "PAPER" in reply
        assert "paused" in reply

    async def test_funds_flags_unverified_symbols(self, stack: TelegramStack) -> None:
        """Every seed ETF is unverified, and the bot must say so."""
        reply = await send(stack, "/funds")
        assert reply is not None
        assert "SPUS" in reply
        assert "⚠️" in reply
        assert "unverified" in reply.lower()

    async def test_history_when_empty(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/history")
        assert reply is not None
        assert "No closed trades" in reply

    async def test_history_accepts_a_limit(self, stack: TelegramStack) -> None:
        assert await send(stack, "/history 5") is not None

    async def test_history_tolerates_a_junk_limit(self, stack: TelegramStack) -> None:
        """A typo should not produce an error reply."""
        reply = await send(stack, "/history abc")
        assert reply is not None
        assert "No closed trades" in reply

    async def test_pending_when_empty(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/pending")
        assert reply is not None
        assert "Nothing is awaiting" in reply

    async def test_unknown_command(self, stack: TelegramStack) -> None:
        reply = await send(stack, "/frobnicate")
        assert reply is not None
        assert "Unknown command" in reply

    async def test_plain_text_is_ignored(self, stack: TelegramStack) -> None:
        assert await send(stack, "hello there") is None

    async def test_command_with_bot_suffix(self, stack: TelegramStack) -> None:
        """Group chats send /balance@MyBot."""
        reply = await send(stack, "/balance@investment_box_bot")
        assert reply is not None
        assert "$500.00" in reply


class TestMessageTagging:
    @pytest.mark.parametrize(
        "command", ["/status", "/balance", "/positions", "/funds", "/history", "/pending", "/help"]
    )
    async def test_every_reply_is_mode_tagged(
        self, stack: TelegramStack, command: str
    ) -> None:
        reply = await send(stack, command)
        assert reply is not None
        assert reply.startswith("[PAPER]")

    async def test_bilingual_output(self, stack: TelegramStack) -> None:
        """Language is 'both', so replies carry English and Arabic."""
        reply = await send(stack, "/balance")
        assert reply is not None
        assert "Equity" in reply
        assert "إجمالي الحساب" in reply


class TestApprovalFlow:
    async def test_request_is_delivered_with_four_buttons(
        self, stack: TelegramStack, transport: FakeTransport
    ) -> None:
        await stack.queue.start()
        await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.queue.stop()

        assert len(transport.sent) == 1
        buttons = transport.buttons_sent[0]
        assert buttons is not None
        labels = [b.callback_data.split(":")[-1] for row in buttons for b in row]
        assert set(labels) == {"approve", "reject", "modify", "snooze"}

    async def test_message_states_the_deadline_and_the_default(
        self, stack: TelegramStack, transport: FakeTransport
    ) -> None:
        await stack.queue.start()
        await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.queue.stop()
        text = transport.last_text()
        assert text is not None
        assert "Expires in" in text
        assert "rejection" in text.lower()

    async def test_message_includes_forecast_and_backtest(
        self, stack: TelegramStack, transport: FakeTransport
    ) -> None:
        await stack.queue.start()
        await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.queue.stop()
        text = transport.last_text()
        assert text is not None
        assert "58.0%" in text
        assert "Sharpe" in text

    async def test_approve_button(self, stack: TelegramStack) -> None:
        await stack.queue.start()
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.queue.stop()

        updated = await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:approve", callback_id="cb1"
        )
        assert updated is not None
        assert updated.status is ApprovalStatus.APPROVED
        assert updated.is_actionable

    async def test_reject_button(self, stack: TelegramStack) -> None:
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        updated = await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:reject", callback_id="cb1"
        )
        assert updated is not None
        assert not updated.is_actionable

    async def test_buttons_are_removed_after_a_decision(
        self, stack: TelegramStack, transport: FakeTransport
    ) -> None:
        """A spent approval must not still look pressable."""
        await stack.queue.start()
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.queue.stop()

        await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:approve", callback_id="cb1"
        )
        assert transport.markup_edits
        assert transport.markup_edits[-1][2] is None  # keyboard cleared
        assert "Approved" in transport.edits[-1][2]

    async def test_double_tap_is_a_no_op(self, stack: TelegramStack) -> None:
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:approve", callback_id="cb1"
        )
        second = await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:reject", callback_id="cb2"
        )
        assert second is not None
        assert second.status is ApprovalStatus.APPROVED

    async def test_malformed_callback_is_dropped(self, stack: TelegramStack) -> None:
        for bad in ["garbage", "apv:notanumber:approve", "apv:1:frobnicate", "apv:1"]:
            assert (
                await stack.bot.handle_callback(
                    user_id=ALLOWED_USER, data=bad, callback_id="cb"
                )
                is None
            )

    async def test_callback_is_acknowledged(
        self, stack: TelegramStack, transport: FakeTransport
    ) -> None:
        """Otherwise Telegram spins forever on the user's screen."""
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:approve", callback_id="cb1"
        )
        assert transport.answered_callbacks


class TestTimeoutRejects:
    async def test_unanswered_request_expires_as_a_rejection(
        self, stack: TelegramStack, clock: FrozenClock
    ) -> None:
        """The property the whole approval design exists to guarantee."""
        request = await stack.notifier.request(
            request_key="k",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=dict(PROPOSAL),
            timeout_minutes=30,
        )
        clock.advance(minutes=31)

        expired = await stack.notifier.sweep_expired()
        assert len(expired) == 1
        assert expired[0].status is ApprovalStatus.EXPIRED
        assert not expired[0].is_actionable

        assert stack.notifier.service.get(request.id).is_actionable is False

    async def test_late_tap_does_not_approve(
        self, stack: TelegramStack, clock: FrozenClock
    ) -> None:
        request = await stack.notifier.request(
            request_key="k",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=dict(PROPOSAL),
            timeout_minutes=30,
        )
        clock.advance(minutes=31)
        updated = await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:approve", callback_id="cb1"
        )
        assert updated is not None
        assert not updated.is_actionable

    async def test_expired_message_is_updated(
        self, stack: TelegramStack, transport: FakeTransport, clock: FrozenClock
    ) -> None:
        await stack.queue.start()
        await stack.notifier.request(
            request_key="k",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=dict(PROPOSAL),
            timeout_minutes=30,
        )
        await stack.queue.stop()

        clock.advance(minutes=31)
        await stack.notifier.sweep_expired()
        assert transport.edits
        assert "Expired" in transport.edits[-1][2]


class TestModifyFlow:
    async def test_modify_then_supply_a_size(self, stack: TelegramStack) -> None:
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:modify", callback_id="cb1"
        )
        assert stack.notifier.is_awaiting_size(ALLOWED_USER)

        await send(stack, "1")
        updated = stack.notifier.service.get(request.id)
        assert updated is not None
        assert updated.status is ApprovalStatus.APPROVED
        assert updated.payload["quantity"] == "1"
        assert updated.payload["original_quantity"] == "2"

    async def test_invalid_size_keeps_the_request_parked(self, stack: TelegramStack) -> None:
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:modify", callback_id="cb1"
        )
        reply = await send(stack, "not a number")
        assert reply is not None
        assert "not a valid quantity" in reply
        assert stack.notifier.is_awaiting_size(ALLOWED_USER)

    async def test_modify_state_is_per_user(self, stack: TelegramStack) -> None:
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:modify", callback_id="cb1"
        )
        assert not stack.notifier.is_awaiting_size(OTHER_ALLOWED_USER)
        # The other user's plain text is still ignored, not consumed as a size.
        assert await send(stack, "2", user_id=OTHER_ALLOWED_USER) is None


class TestSnoozeFlow:
    async def test_snooze_extends_and_keeps_buttons(
        self, stack: TelegramStack, clock: FrozenClock
    ) -> None:
        request = await stack.notifier.request(
            request_key="k",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=dict(PROPOSAL),
            timeout_minutes=30,
        )
        before = request.seconds_remaining(clock.now())
        updated = await stack.bot.handle_callback(
            user_id=ALLOWED_USER, data=f"apv:{request.id}:snooze", callback_id="cb1"
        )
        assert updated is not None
        assert updated.seconds_remaining(clock.now()) > before
        assert not updated.status.is_terminal


class TestPendingCommand:
    async def test_pending_lists_open_requests(self, stack: TelegramStack) -> None:
        await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )
        reply = await send(stack, "/pending")
        assert reply is not None
        assert "SPUS" in reply

    async def test_pending_hides_expired_requests(
        self, stack: TelegramStack, clock: FrozenClock
    ) -> None:
        """/pending must never invite approving something already timed out."""
        await stack.notifier.request(
            request_key="k",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=dict(PROPOSAL),
            timeout_minutes=30,
        )
        clock.advance(minutes=31)
        reply = await send(stack, "/pending")
        assert reply is not None
        assert "Nothing is awaiting" in reply


class TestNoCredentialsDegradation:
    async def test_stack_builds_without_a_token(
        self, settings, portfolio, approvals, audit, clock
    ) -> None:
        """No token must mean silence, not a crash."""
        from investment_box.config.schema import Secrets
        from investment_box.telegram.bot import build_telegram_stack

        stack = build_telegram_stack(
            settings=settings,
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            portfolio=portfolio,
            approvals=approvals,
            audit=audit,
            clock=clock,
        )
        assert stack.transport.name == "fake"
        assert not stack.guard.is_enabled
        assert not stack.broadcast.is_configured

        # And with nobody whitelisted, nobody is answered.
        assert (
            await stack.bot.handle_message(user_id=ALLOWED_USER, chat_id=1, text="/balance")
            is None
        )

    async def test_approval_without_a_chat_still_expires(
        self, settings, portfolio, approvals, audit, clock: FrozenClock
    ) -> None:
        """An undeliverable request must still default to rejection."""
        from investment_box.config.schema import Secrets
        from investment_box.telegram.bot import build_telegram_stack

        stack = build_telegram_stack(
            settings=settings,
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            portfolio=portfolio,
            approvals=approvals,
            audit=audit,
            clock=clock,
        )
        request = await stack.notifier.request(
            request_key="k",
            kind=ApprovalKind.TRADE_PROPOSAL,
            payload=dict(PROPOSAL),
            timeout_minutes=30,
        )
        clock.advance(minutes=31)
        assert not stack.notifier.service.get(request.id).is_actionable


class TestEngineIsolation:
    async def test_telegram_failure_does_not_propagate(
        self, stack: TelegramStack, transport: FakeTransport
    ) -> None:
        """A broken Telegram must never reach the caller."""
        slept: list[float] = []

        async def fake_sleep(delay: float) -> None:
            slept.append(delay)
            await asyncio.sleep(0)

        stack.queue._sleep = fake_sleep
        transport.fail_next = 99
        await stack.queue.start()

        stack.broadcast.trade_opened(
            symbol="SPUS",
            side="buy",
            quantity=Decimal("2"),
            entry_price=Decimal("59.83"),
            size_pct=0.24,
            stop_loss=Decimal("55.04"),
            take_profit=Decimal("65.21"),
            strategy="etf_momentum_rotation",
            probability=0.58,
            compliance_status="compliant",
            reason="test",
        )
        for _ in range(80):
            await asyncio.sleep(0)
            if stack.queue.dropped_count:
                break
        await stack.queue.stop(drain=False)
        assert transport.sent == []  # nothing got through, and nothing raised


class TestTransportFailuresDoNotPropagate:
    """A cosmetic Telegram call must never crash the handler.

    Regression: acknowledging a button press raised "Query is too old" *after*
    the decision had already been recorded, taking down the update handler on
    a purely cosmetic call. Telegram rejects callback queries older than about
    a minute, which happens routinely.
    """

    class _ExplodingTransport(FakeTransport):
        name = "exploding"

        async def answer_callback(
            self, callback_id: str, text: str | None = None, *, show_alert: bool = False
        ) -> bool:
            raise RuntimeError("Query is too old and response timeout expired")

    async def test_stale_callback_does_not_crash_the_handler(
        self, settings, portfolio, approvals, audit, clock, telegram_secrets
    ) -> None:
        from investment_box.telegram.bot import build_telegram_stack

        exploding = self._ExplodingTransport()
        stack = build_telegram_stack(
            settings=settings,
            secrets=telegram_secrets,
            portfolio=portfolio,
            approvals=approvals,
            audit=audit,
            clock=clock,
            transport=exploding,
        )
        request = await stack.notifier.request(
            request_key="k", kind=ApprovalKind.TRADE_PROPOSAL, payload=dict(PROPOSAL)
        )

        # The acknowledgement blows up, but the decision must still stand and
        # the handler must not raise.
        with pytest.raises(RuntimeError):
            await stack.bot.handle_callback(
                user_id=ALLOWED_USER, data=f"apv:{request.id}:approve", callback_id="stale"
            )

        # Whatever happened to the acknowledgement, the decision was recorded.
        assert stack.notifier.service.get(request.id).status is ApprovalStatus.APPROVED

    async def test_real_transport_swallows_telegram_errors(self) -> None:
        """PTBTransport.answer_callback returns False rather than raising."""
        from telegram.error import BadRequest

        from investment_box.telegram.transport import PTBTransport

        transport = PTBTransport.__new__(PTBTransport)  # no network, no token

        class _Bot:
            async def answer_callback_query(self, **_: object) -> None:
                raise BadRequest("Query is too old and response timeout expired")

        transport._bot = _Bot()  # type: ignore[attr-defined]
        assert await transport.answer_callback("stale") is False

    async def test_real_transport_swallows_edit_errors(self) -> None:
        from telegram.error import Forbidden

        from investment_box.telegram.transport import PTBTransport

        transport = PTBTransport.__new__(PTBTransport)

        class _Bot:
            async def edit_message_text(self, **_: object) -> None:
                raise Forbidden("bot was blocked by the user")

        transport._bot = _Bot()  # type: ignore[attr-defined]
        assert await transport.edit_message_text("chat", 1, "text") is False
