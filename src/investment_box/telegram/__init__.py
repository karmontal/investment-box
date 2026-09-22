"""Telegram broadcast channel and interactive control bot.

Layering, from the bottom up:

``transport``   the Bot API surface, with a real and an in-memory implementation
``queue``       rate-limited, retrying, non-blocking outbound delivery
``auth``        the whitelist -- fails closed, checks callbacks as well as commands
``formatting``  bilingual message construction, ``[PAPER]``/``[LIVE]`` tagging
``broadcast``   one-way channel posts
``approvals``   delivery and button handling for approval requests
``commands``    read-only command handlers, all via the service layer
``bot``         update routing and assembly
"""

from investment_box.telegram.auth import AuthGuard
from investment_box.telegram.bot import TelegramBot, TelegramStack, build_telegram_stack
from investment_box.telegram.broadcast import BroadcastChannel
from investment_box.telegram.queue import MessageQueue, RateLimiter
from investment_box.telegram.transport import FakeTransport, InlineButton, TelegramTransport

__all__ = [
    "AuthGuard",
    "BroadcastChannel",
    "FakeTransport",
    "InlineButton",
    "MessageQueue",
    "RateLimiter",
    "TelegramBot",
    "TelegramStack",
    "TelegramTransport",
    "build_telegram_stack",
]
