#!/usr/bin/env python3
"""Show the live-trading pre-flight checklist.

    uv run python scripts/preflight.py
    uv run python scripts/preflight.py --backtest-return 0.152

Read-only. It evaluates and prints; it cannot enable anything. Going live
requires typing the confirmation phrase in the dashboard, and even then only
when every check below passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.core.logging import configure_logging
from investment_box.engine.live_guard import LIVE_CONFIRMATION_PHRASE
from investment_box.engine.preflight import PreflightChecklist
from investment_box.services.container import build_services


def main(args: argparse.Namespace) -> int:
    configure_logging("ERROR")
    services = build_services(configure_logs=False)

    try:
        held = [str(p.symbol) for p in services.broker.get_positions()]
    except Exception:  # noqa: BLE001 - the checklist reports it itself
        held = []

    report = PreflightChecklist(
        services.settings, services.secrets, services.database, clock=services.clock
    ).evaluate(
        broker=services.broker,
        data_provider_name=services.market_data.provider.name,
        compliance_provider_name=services.settings.shariah.provider,
        held_symbols=held,
        backtest_return=args.backtest_return,
    )

    print(report.to_text())

    if report.passed:
        print()
        print("To enable live trading, open the dashboard's Controls tab and type:")
        print(f"    {LIVE_CONFIRMATION_PHRASE}")
        print()
        print("The checklist is re-evaluated at that moment, on every engine start,")
        print("and before every live order. Passing now is not a permanent state.")
        return 0

    print()
    print(f"{len(report.failures)} check(s) must be resolved first.")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-return",
        type=float,
        default=None,
        help="backtest total return over the paper period, e.g. 0.152 for 15.2%%",
    )
    raise SystemExit(main(parser.parse_args()))
