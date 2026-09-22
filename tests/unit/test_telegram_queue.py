"""Outbound queue: rate limiting, retry, backoff and back-pressure.

The property that matters most: a Telegram failure must never reach the caller.
Several tests deliberately make the transport fail and then assert that the
producer side carried on regardless.
"""

from __future__ import annotations

import asyncio

import pytest

from investment_box.telegram.queue import (
    GLOBAL_MESSAGES_PER_SECOND,
    MessageQueue,
    QueuedMessage,
    RateLimiter,
)
from investment_box.telegram.transport import FakeTransport


async def collector() -> tuple[list[float], object]:
    """A sleep stub that records delays instead of waiting."""
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)
        await asyncio.sleep(0)

    return slept, fake_sleep


class TestRateLimiter:
    def test_allows_the_first_message_immediately(self) -> None:
        assert RateLimiter(start_time=0.0).delay_for("chat", now=0.0) == 0.0

    def test_throttles_once_the_bucket_empties(self) -> None:
        limiter = RateLimiter(per_second=2.0, per_chat_per_minute=1000, start_time=0.0)
        for _ in range(2):
            limiter.record("chat", now=0.0)
        assert limiter.delay_for("chat", now=0.0) > 0

    def test_bucket_refills_over_time(self) -> None:
        limiter = RateLimiter(per_second=2.0, per_chat_per_minute=1000, start_time=0.0)
        for _ in range(2):
            limiter.record("chat", now=0.0)
        assert limiter.delay_for("chat", now=5.0) == 0.0

    def test_per_chat_window_is_enforced(self) -> None:
        """The limit that actually bites for a broadcast channel."""
        limiter = RateLimiter(per_second=1000.0, per_chat_per_minute=3, start_time=0.0)
        for _ in range(3):
            limiter.record("channel", now=0.0)
        assert limiter.delay_for("channel", now=1.0) == pytest.approx(59.0, abs=0.1)

    def test_per_chat_limits_are_independent(self) -> None:
        limiter = RateLimiter(per_second=1000.0, per_chat_per_minute=2, start_time=0.0)
        for _ in range(2):
            limiter.record("a", now=0.0)
        assert limiter.delay_for("a", now=1.0) > 0
        assert limiter.delay_for("b", now=1.0) == 0.0

    def test_window_slides(self) -> None:
        limiter = RateLimiter(per_second=1000.0, per_chat_per_minute=2, start_time=0.0)
        for _ in range(2):
            limiter.record("chat", now=0.0)
        assert limiter.delay_for("chat", now=61.0) == 0.0


class TestEnqueue:
    async def test_enqueue_never_blocks(self, transport: FakeTransport) -> None:
        queue = MessageQueue(transport)
        queue.enqueue("chat", "hello")
        assert queue.size == 1
        assert transport.sent == []  # nothing sent until the worker runs

    async def test_worker_delivers(self, transport: FakeTransport) -> None:
        queue = MessageQueue(transport)
        await queue.start()
        queue.enqueue("chat", "hello")
        await queue.drain()
        await asyncio.sleep(0.01)
        await queue.stop()
        assert transport.texts == ["hello"]

    async def test_priority_ordering(self, transport: FakeTransport) -> None:
        """An alert queued behind summaries must still go out first."""
        queue = MessageQueue(transport)
        queue.enqueue("chat", "summary", priority=10)
        queue.enqueue("chat", "trade", priority=50)
        queue.enqueue("chat", "alert", priority=100)

        await queue.start()
        await queue.drain()
        await asyncio.sleep(0.01)
        await queue.stop()
        assert transport.texts == ["alert", "trade", "summary"]

    async def test_fifo_within_a_priority(self, transport: FakeTransport) -> None:
        queue = MessageQueue(transport)
        for index in range(3):
            queue.enqueue("chat", f"m{index}", priority=10)
        await queue.start()
        await queue.drain()
        await asyncio.sleep(0.01)
        await queue.stop()
        assert transport.texts == ["m0", "m1", "m2"]


class TestBackPressure:
    async def test_full_queue_sheds_the_lowest_priority(
        self, transport: FakeTransport
    ) -> None:
        queue = MessageQueue(transport, max_queue=3)
        queue.enqueue("chat", "low", priority=1)
        queue.enqueue("chat", "mid", priority=50)
        queue.enqueue("chat", "high", priority=100)
        queue.enqueue("chat", "alert", priority=100)

        assert queue.dropped_count == 1
        assert queue.size == 3

        await queue.start()
        await queue.drain()
        await asyncio.sleep(0.01)
        await queue.stop()
        assert "low" not in transport.texts
        assert "alert" in transport.texts

    async def test_queue_never_grows_unbounded(self, transport: FakeTransport) -> None:
        """A long Telegram outage must not exhaust memory."""
        queue = MessageQueue(transport, max_queue=10)
        for index in range(100):
            queue.enqueue("chat", f"m{index}")
        assert queue.size == 10
        assert queue.dropped_count == 90


class TestRetryAndFailure:
    async def test_transient_failure_is_retried(self, transport: FakeTransport) -> None:
        slept, fake_sleep = await collector()
        queue = MessageQueue(transport, sleep=fake_sleep)
        transport.fail_next = 2

        await queue.start()
        queue.enqueue("chat", "important")
        for _ in range(60):
            await asyncio.sleep(0)
            if transport.sent:
                break
        await queue.stop()

        assert transport.texts == ["important"]
        assert queue.retry_count == 2
        assert slept[:2] == [1.0, 2.0]  # exponential backoff

    async def test_permanent_failure_drops_and_does_not_raise(
        self, transport: FakeTransport
    ) -> None:
        """A dead Telegram must not take the engine with it."""
        _, fake_sleep = await collector()
        queue = MessageQueue(transport, max_attempts=3, sleep=fake_sleep)
        transport.fail_next = 99

        await queue.start()
        queue.enqueue("chat", "doomed")
        for _ in range(80):
            await asyncio.sleep(0)
            if queue.dropped_count:
                break
        await queue.stop(drain=False)

        assert queue.dropped_count == 1
        assert transport.sent == []

    async def test_producer_is_unaffected_by_failure(self, transport: FakeTransport) -> None:
        """The whole point: enqueue returns normally while delivery is broken."""
        _, fake_sleep = await collector()
        queue = MessageQueue(transport, max_attempts=2, sleep=fake_sleep)
        transport.fail_next = 99
        await queue.start()

        for index in range(5):
            queue.enqueue("chat", f"m{index}")  # must not raise

        await asyncio.sleep(0)
        await queue.stop(drain=False)
        assert queue.is_running is False


class TestFutures:
    async def test_future_resolves_with_the_sent_message(
        self, transport: FakeTransport
    ) -> None:
        """Approval requests need the message id so they can edit it later."""
        queue = MessageQueue(transport)
        await queue.start()
        future = queue.enqueue("chat", "approval", want_result=True)
        assert future is not None
        sent = await asyncio.wait_for(future, timeout=1.0)
        await queue.stop()
        assert sent.message_id > 0
        assert sent.chat_id == "chat"

    async def test_future_receives_the_failure(self, transport: FakeTransport) -> None:
        _, fake_sleep = await collector()
        queue = MessageQueue(transport, max_attempts=1, sleep=fake_sleep)
        transport.fail_next = 99
        await queue.start()
        future = queue.enqueue("chat", "doomed", want_result=True)
        assert future is not None
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(future, timeout=1.0)
        await queue.stop(drain=False)


class TestLifecycle:
    async def test_stats(self, transport: FakeTransport) -> None:
        queue = MessageQueue(transport)
        await queue.start()
        queue.enqueue("chat", "one")
        await queue.drain()
        await asyncio.sleep(0.01)
        stats = queue.stats()
        await queue.stop()
        assert stats["sent"] == 1
        assert stats["transport"] == "fake"

    async def test_start_is_idempotent(self, transport: FakeTransport) -> None:
        queue = MessageQueue(transport)
        await queue.start()
        await queue.start()
        await queue.stop()
        assert not queue.is_running

    async def test_stop_drains_by_default(self, transport: FakeTransport) -> None:
        queue = MessageQueue(transport)
        await queue.start()
        for index in range(5):
            queue.enqueue("chat", f"m{index}")
        await queue.stop()
        assert len(transport.sent) == 5

    async def test_messages_survive_being_queued_while_stopped(
        self, transport: FakeTransport
    ) -> None:
        queue = MessageQueue(transport)
        queue.enqueue("chat", "early")
        await queue.start()
        await queue.drain()
        await asyncio.sleep(0.01)
        await queue.stop()
        assert transport.texts == ["early"]


class TestQueuedMessage:
    def test_defaults(self) -> None:
        message = QueuedMessage(chat_id="c", text="t")
        assert message.attempts == 0
        assert message.priority == 0

    def test_global_rate_constant_is_below_the_documented_ceiling(self) -> None:
        assert GLOBAL_MESSAGES_PER_SECOND < 30
