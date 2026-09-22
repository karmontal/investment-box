#!/usr/bin/env python3
"""Phase 2 smoke check.

Exercises the whole Telegram stack without sending anything, unless you ask it
to. Two modes:

    uv run python scripts/telegram_smoke.py             # dry run, prints messages
    uv run python scripts/telegram_smoke.py --diagnose  # check the bot can reach the channel
    uv run python scripts/telegram_smoke.py --send      # actually posts to Telegram

``--send`` requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID in .env, and it
really does post to your channel. It sends one clearly-labelled test message,
nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.core.types import ApprovalKind
from investment_box.services.container import build_services
from investment_box.telegram.bot import build_telegram_stack

DEMO_PROPOSAL = {
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
    "reason": "DEMO ONLY -- not a real signal",
    "backtest_summary": "DEMO figures, not a real backtest",
}


async def diagnose() -> int:
    """Read-only check that the bot exists and can reach the configured channel.

    "Chat not found" from a send is ambiguous -- a wrong id, a bot that was
    never added, or a deleted channel all look identical. These read-only calls
    tell the three apart before anyone starts guessing at the id.
    """
    from telegram import Bot
    from telegram.error import TelegramError

    from investment_box.config.loader import get_secrets

    secrets = get_secrets()
    rule("TELEGRAM DIAGNOSTICS")

    if not secrets.telegram_bot_token:
        print("  TELEGRAM_BOT_TOKEN is not set.")
        return 1

    problem = secrets.telegram_channel_problem
    if problem:
        print(f"  !! {problem}\n")

    bot = Bot(token=secrets.telegram_bot_token.get_secret_value())
    try:
        me = await bot.get_me()
    except TelegramError as exc:
        print(f"  Bot token rejected: {exc}")
        print("  Check TELEGRAM_BOT_TOKEN, or regenerate it with @BotFather.")
        return 1
    print(f"  Bot:     @{me.username}  (id {me.id})")

    if not secrets.telegram_channel_id:
        print("  TELEGRAM_CHANNEL_ID is not set -- broadcasts go nowhere.")
        return 1

    try:
        chat = await bot.get_chat(secrets.telegram_channel_id)
    except TelegramError as exc:
        print(f"  Channel: NOT REACHABLE -- {exc}")
        print()
        print("  The bot cannot see that chat at all. In order of likelihood:")
        print(f"    1. @{me.username} is not a member of the channel.")
        print("       Add it as an ADMINISTRATOR with 'Post Messages'.")
        print("       (Channels require admin; plain membership is not enough.)")
        print("    2. TELEGRAM_CHANNEL_ID is wrong. To get the real one: post any")
        print("       message in the channel and forward it to @JsonDumpBot, which")
        print("       reports the id including its -100 prefix.")
        print("    3. The channel was deleted.")
        return 1

    print(f"  Channel: {chat.title!r} (type {chat.type})")
    try:
        member = await bot.get_chat_member(secrets.telegram_channel_id, me.id)
        print(f"  Status:  {member.status}")
        if member.status not in ("administrator", "creator"):
            print("  !! The bot is not an admin. Channels require admin to post.")
            return 1
    except TelegramError as exc:
        print(f"  Status:  could not read -- {exc}")

    allowed = secrets.allowed_telegram_ids
    print(f"  Allowed users: {len(allowed)}"
          f"{' -- interactive bot DISABLED' if not allowed else ''}")
    print()
    print("  All checks passed. --send should work.")
    return 0


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


async def main(send: bool) -> int:
    services = build_services()
    stack = build_telegram_stack(
        settings=services.settings,
        secrets=services.secrets,
        portfolio=services.portfolio,
        approvals=services.approvals,
        audit=services.audit,
        clock=services.clock,
        data_provider_name=services.market_data.provider.name,
        engine_state="idle (phase 2)",
    )

    rule("CONFIGURATION")
    problem = services.secrets.telegram_channel_problem
    if problem:
        print(f"  !! {problem}")
    print(f"  Transport:        {stack.transport.name}")
    print(f"  Broadcast channel: {'configured' if stack.broadcast.is_configured else 'NOT SET'}")
    print(f"  Whitelisted users: {len(stack.guard.allowed_user_ids)}")
    if not stack.guard.is_enabled:
        print("  !! No TELEGRAM_ALLOWED_USER_IDS -- the bot will answer nobody.")
    if stack.transport.name == "fake":
        print("  !! No TELEGRAM_BOT_TOKEN -- running on the in-memory transport.")

    rule("READ-ONLY COMMANDS (rendered locally)")
    for command in ("/status", "/balance", "/positions", "/funds", "/history", "/pending"):
        reply = stack.handlers.dispatch(command)
        first = reply.splitlines()[0] if reply else ""
        print(f"\n  {command}  ->  {len(reply)} chars, starts {first!r}")

    rule("/help")
    print(stack.handlers.help())

    rule("APPROVAL REQUEST (dummy proposal)")
    await stack.queue.start()
    # Unique per run: a fixed key would hit the idempotency guard on the second
    # run and hand back the previous request instead of a fresh one.
    request_key = f"smoke-demo-{int(services.clock.now().timestamp())}"
    request = await stack.notifier.request(
        request_key=request_key,
        kind=ApprovalKind.TRADE_PROPOSAL,
        payload=dict(DEMO_PROPOSAL),
        timeout_minutes=30,
    )
    await asyncio.sleep(0.1)
    print(f"  id={request.id} status={request.status.value} "
          f"expires_in={request.seconds_remaining(services.clock.now()) // 60}m")
    print(f"  actionable: {request.is_actionable}  (must be False while pending)")

    print("\n  Simulating REJECT:")
    rejected = await stack.bot.handle_callback(
        user_id=next(iter(stack.guard.allowed_user_ids), 0),
        data=f"apv:{request.id}:reject",
        # No real callback query exists here, so there is nothing to acknowledge.
        callback_id=None,
    )
    if rejected is not None:
        print(f"  -> status={rejected.status.value} actionable={rejected.is_actionable}")
    else:
        print("  -> ignored (no whitelisted user configured)")

    rule("BROADCAST (queued, not necessarily sent)")
    stack.broadcast.trade_opened(
        symbol="SPUS", side="buy", quantity=Decimal("2"), entry_price=Decimal("59.83"),
        size_pct=0.24, stop_loss=Decimal("55.04"), take_profit=Decimal("65.21"),
        strategy="etf_momentum_rotation", probability=0.58,
        compliance_status="compliant", reason="DEMO ONLY",
    )
    stack.broadcast.risk_limit_hit("DEMO ONLY -- daily loss limit example")
    await stack.queue.drain()
    await asyncio.sleep(0.2)
    print(f"  queue stats: {stack.queue.stats()}")

    if send:
        rule("LIVE SEND")
        if not stack.broadcast.is_configured or stack.transport.name == "fake":
            print("  Cannot send: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID in .env")
        else:
            stack.broadcast.custom("Investment Box test message — Phase 2 smoke check.")
            await stack.queue.drain()
            await asyncio.sleep(1.0)
            print(f"  sent={stack.queue.sent_count} dropped={stack.queue.dropped_count}")

    await stack.queue.stop()

    rule("RESULT")
    print("  Telegram stack is wired. Nothing trades: there is no engine yet.")
    if stack.transport.name == "fake":
        print("  Everything above ran on the in-memory transport -- nothing was sent.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--send", action="store_true", help="actually post one test message to Telegram"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="read-only check that the bot can reach the configured channel",
    )
    args = parser.parse_args()
    if args.diagnose:
        raise SystemExit(asyncio.run(diagnose()))
    raise SystemExit(asyncio.run(main(args.send)))
