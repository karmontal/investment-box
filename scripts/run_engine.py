#!/usr/bin/env python3
"""Run the trading engine.

    uv run python scripts/run_engine.py --once     # one cycle, then exit
    uv run python scripts/run_engine.py            # scheduled, runs until stopped

Paper mode only. Live trading needs the Phase 7 pre-flight checklist, and the
engine refuses to start running without it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.core.logging import configure_logging
from investment_box.engine.runner import build_engine, run_forever
from investment_box.services.container import build_services
from investment_box.telegram.bot import build_telegram_stack


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


async def main(args: argparse.Namespace) -> int:
    configure_logging(args.log_level)
    services = build_services(configure_logs=False)

    telegram = build_telegram_stack(
        settings=services.settings,
        secrets=services.secrets,
        portfolio=services.portfolio,
        approvals=services.approvals,
        audit=services.audit,
        clock=services.clock,
        data_provider_name=services.market_data.provider.name,
        engine_state="starting",
    )

    runner = build_engine(services, strategy_name=args.strategy, telegram=telegram)
    # /funds and the dashboard read the same rankings the engine acts on.
    telegram.handlers.ctx.research = runner.research

    rule("STARTUP")
    for line in services.startup_banner():
        print(f"  {line}")

    if runner.blockers:
        rule("ENGINE WILL NOT TRADE")
        for blocker in runner.blockers:
            print(f"  ! {blocker}")
        print("\n  The engine starts PAUSED. It will observe and report, not trade.")
    else:
        rule("ENGINE READY")
        print(f"  Strategy:  {runner.research.strategy.name}")
        print(f"  Autonomy:  {services.settings.engine.autonomy_level.value}")

    if args.once:
        rule("SINGLE CYCLE")
        await telegram.bot.start()
        result = await runner.run_once()
        print(f"  Result: {result.summary()}")
        for label, items in (
            ("entries", result.entries),
            ("exits", result.exits),
            ("compliance exits", result.compliance_exits),
            ("awaiting approval", result.approvals_requested),
            ("errors", result.errors),
        ):
            for item in items:
                print(f"    {label}: {item}")
        if result.refusals:
            print(f"\n  Refusals ({len(result.refusals)}):")
            for refusal in result.refusals[:12]:
                print(f"    - {refusal}")
        await telegram.bot.stop()
        return 0

    rule("SCHEDULED")
    for job in runner.scheduler.jobs():
        print(f"  {job.name}: {job.description}")
    print("\n  Running. Ctrl-C to stop.")
    await run_forever(runner)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit")
    parser.add_argument("--strategy", default="etf_momentum_rotation")
    parser.add_argument("--log-level", default="INFO")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
