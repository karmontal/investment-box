"""Outbound message queue with rate limiting and retry.

The requirement this module exists to satisfy: *never let a Telegram failure
block or crash the trading engine*. So:

* :meth:`MessageQueue.enqueue` is synchronous, non-blocking and never raises.
  A caller in the middle of placing an order hands over a message and moves on.
* Delivery happens on a background worker. Failures retry with exponential
  backoff, and after the last attempt the message is dropped with a log entry.
  A dropped notification is bad; a trade that failed because a notification
  failed is worse.
* A bounded queue sheds the oldest messages rather than growing without limit,
  so a long Telegram outage cannot exhaust memory.

Rate limits observed: Telegram allows roughly 30 messages/second overall and
about 20 messages/minute to a single group or channel. The per-chat limit is
the one that actually bites for a broadcast channel, so both are enforced.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import datetime as dt
import time
from dataclasses import dataclass, field

from investment_box.core.logging import get_logger
from investment_box.telegram.transport import InlineButton, SentMessage, TelegramTransport

log = get_logger(__name__)

#: Telegram's documented global ceiling is ~30/s; stay under it.
GLOBAL_MESSAGES_PER_SECOND = 25.0
#: Per chat, Telegram allows ~20/minute for groups and channels.
PER_CHAT_MESSAGES_PER_MINUTE = 18
DEFAULT_MAX_QUEUE = 500
DEFAULT_MAX_ATTEMPTS = 4
#: Base for exponential backoff: 1s, 2s, 4s, 8s.
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 60.0


@dataclass
class QueuedMessage:
    """One outbound message and its delivery state."""

    chat_id: str
    text: str
    buttons: list[list[InlineButton]] | None = None
    parse_mode: str | None = "HTML"
    disable_notification: bool = False
    #: Higher priority is sent first. Alerts outrank routine summaries.
    priority: int = 0
    attempts: int = 0
    queued_at: float = field(default_factory=time.monotonic)
    #: Resolved with the SentMessage when delivery succeeds, for callers that
    #: need the message id (approval requests, which later edit their buttons).
    future: asyncio.Future[SentMessage] | None = None


class RateLimiter:
    """Token bucket for the global rate, plus a sliding window per chat."""

    def __init__(
        self,
        *,
        per_second: float = GLOBAL_MESSAGES_PER_SECOND,
        per_chat_per_minute: int = PER_CHAT_MESSAGES_PER_MINUTE,
        start_time: float | None = None,
    ) -> None:
        self.per_second = per_second
        self.per_chat_per_minute = per_chat_per_minute
        self._tokens = per_second
        # Injectable so a caller supplying its own time base (tests, replay)
        # does not start out arbitrarily far from the internal anchor.
        self._last_refill = start_time if start_time is not None else time.monotonic()
        self._chat_history: dict[str, collections.deque[float]] = collections.defaultdict(
            collections.deque
        )

    def _refill(self, now: float) -> None:
        # Clamp at zero. A negative elapsed means the caller's clock disagrees
        # with ours; draining the bucket on the strength of that would throttle
        # messages for hours over what is really a bookkeeping error.
        elapsed = max(0.0, now - self._last_refill)
        self._tokens = min(self.per_second, self._tokens + elapsed * self.per_second)
        self._last_refill = max(now, self._last_refill)

    def delay_for(self, chat_id: str, *, now: float | None = None) -> float:
        """Seconds to wait before this chat may receive another message."""
        now = now if now is not None else time.monotonic()
        self._refill(now)

        wait = 0.0
        if self._tokens < 1.0:
            wait = (1.0 - self._tokens) / self.per_second

        history = self._chat_history[chat_id]
        while history and now - history[0] >= 60.0:
            history.popleft()
        if len(history) >= self.per_chat_per_minute:
            wait = max(wait, 60.0 - (now - history[0]))

        return max(0.0, wait)

    def record(self, chat_id: str, *, now: float | None = None) -> None:
        """Account for a message that has just been sent."""
        now = now if now is not None else time.monotonic()
        self._refill(now)
        self._tokens = max(0.0, self._tokens - 1.0)
        self._chat_history[chat_id].append(now)


class MessageQueue:
    """Background sender.

    Start it with :meth:`start` and stop it with :meth:`stop`. While stopped,
    :meth:`enqueue` still accepts messages -- they are delivered when the
    worker next runs.
    """

    def __init__(
        self,
        transport: TelegramTransport,
        *,
        max_queue: int = DEFAULT_MAX_QUEUE,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        rate_limiter: RateLimiter | None = None,
        sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] | None = None,
    ) -> None:
        self.transport = transport
        self.max_queue = max_queue
        self.max_attempts = max_attempts
        self.limiter = rate_limiter or RateLimiter()
        # Injectable so tests exercise backoff without actually waiting.
        self._sleep = sleep or asyncio.sleep

        self._pending: collections.deque[QueuedMessage] = collections.deque()
        self._worker: asyncio.Task[None] | None = None
        self._wakeup = asyncio.Event()
        self._running = False

        self.sent_count = 0
        self.dropped_count = 0
        self.retry_count = 0

    # --------------------------------------------------------------- producing

    def enqueue(
        self,
        chat_id: str,
        text: str,
        *,
        buttons: list[list[InlineButton]] | None = None,
        priority: int = 0,
        disable_notification: bool = False,
        want_result: bool = False,
    ) -> asyncio.Future[SentMessage] | None:
        """Hand a message over for delivery. Never blocks, never raises.

        Args:
            want_result: Return a future resolved with the delivered message.
                Only callers that need the message id (to edit it later) should
                ask for one; an unawaited future is harmless but pointless.

        Returns:
            A future when ``want_result``, else ``None``.
        """
        future: asyncio.Future[SentMessage] | None = None
        if want_result:
            with contextlib.suppress(RuntimeError):  # no running loop yet
                future = asyncio.get_running_loop().create_future()

        message = QueuedMessage(
            chat_id=str(chat_id),
            text=text,
            buttons=buttons,
            priority=priority,
            disable_notification=disable_notification,
            future=future,
        )

        if len(self._pending) >= self.max_queue:
            # Shed the oldest low-priority message. Never drop the new one
            # silently just because the queue is full -- an alert arriving
            # during an outage is the message most worth keeping.
            victim = self._lowest_priority_index()
            dropped = self._pending[victim]
            del self._pending[victim]
            self.dropped_count += 1
            log.warning(
                "telegram.queue_full",
                dropped_preview=dropped.text[:60],
                queue_size=len(self._pending),
            )

        self._insert_by_priority(message)
        self._wakeup.set()
        return future

    def _insert_by_priority(self, message: QueuedMessage) -> None:
        """Insert keeping the queue sorted by priority, FIFO within a priority."""
        for index in range(len(self._pending)):
            if self._pending[index].priority < message.priority:
                self._pending.insert(index, message)
                return
        self._pending.append(message)

    def _lowest_priority_index(self) -> int:
        lowest = 0
        for index in range(1, len(self._pending)):
            if self._pending[index].priority <= self._pending[lowest].priority:
                lowest = index
        return lowest

    # --------------------------------------------------------------- consuming

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker = asyncio.create_task(self._run(), name="telegram-message-queue")
        log.info("telegram.queue_started", transport=self.transport.name)

    async def stop(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        """Stop the worker, optionally delivering what is still queued."""
        if drain and self._pending:
            with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(self.drain(), timeout=timeout)

        self._running = False
        self._wakeup.set()
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None
        log.info(
            "telegram.queue_stopped",
            sent=self.sent_count,
            dropped=self.dropped_count,
            remaining=len(self._pending),
        )

    async def drain(self) -> None:
        """Wait until the queue is empty. Used by tests and by shutdown."""
        while self._pending:
            await asyncio.sleep(0)

    async def _run(self) -> None:
        while self._running:
            if not self._pending:
                self._wakeup.clear()
                with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                    await asyncio.wait_for(self._wakeup.wait(), timeout=1.0)
                continue

            message = self._pending.popleft()
            delay = self.limiter.delay_for(message.chat_id)
            if delay > 0:
                self._pending.appendleft(message)
                await self._sleep(delay)
                continue

            await self._deliver(message)

    async def _deliver(self, message: QueuedMessage) -> None:
        """Send one message, retrying with backoff. Never propagates."""
        try:
            sent = await self.transport.send_message(
                message.chat_id,
                message.text,
                buttons=message.buttons,
                parse_mode=message.parse_mode,
                disable_notification=message.disable_notification,
            )
        except Exception as exc:  # noqa: BLE001 - a send failure must never escape
            message.attempts += 1
            if message.attempts >= self.max_attempts:
                self.dropped_count += 1
                log.error(
                    "telegram.send_failed_permanently",
                    attempts=message.attempts,
                    error=str(exc),
                    preview=message.text[:60],
                )
                if message.future is not None and not message.future.done():
                    message.future.set_exception(exc)
                return

            self.retry_count += 1
            backoff = min(
                BACKOFF_MAX_SECONDS, BACKOFF_BASE_SECONDS * (2 ** (message.attempts - 1))
            )
            log.warning(
                "telegram.send_failed_retrying",
                attempt=message.attempts,
                backoff_seconds=backoff,
                error=str(exc),
            )
            await self._sleep(backoff)
            self._insert_by_priority(message)
            self._wakeup.set()
            return

        self.limiter.record(message.chat_id)
        self.sent_count += 1
        if message.future is not None and not message.future.done():
            message.future.set_result(sent)

    # ------------------------------------------------------------------ status

    @property
    def size(self) -> int:
        return len(self._pending)

    @property
    def is_running(self) -> bool:
        return self._running

    def stats(self) -> dict[str, int | str]:
        return {
            "queued": len(self._pending),
            "sent": self.sent_count,
            "dropped": self.dropped_count,
            "retries": self.retry_count,
            "transport": self.transport.name,
            "running": str(self._running),
        }


def utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)
