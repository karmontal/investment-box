#!/usr/bin/env python3
"""Phase 1 smoke check.

Runs with no credentials and no network: falls back to the mock broker and, if
yfinance cannot be reached, to synthetic prices -- and says so loudly in both
cases, because a number from a fallback is not the number you asked for.

    uv run python scripts/health_check.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.config.loader import load_universe_file
from investment_box.core.logging import configure_logging
from investment_box.services.container import build_services


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def main() -> int:
    configure_logging("WARNING")  # keep the report readable; warnings still show
    services = build_services(configure_logs=False)

    rule("STARTUP")
    for line in services.startup_banner():
        print(f"  {line}")

    rule("ACCOUNT")
    view = services.portfolio.get_account_view()
    print(f"  Mode:              {view.mode_tag}")
    print(f"  Equity:            ${view.equity}")
    print(f"  Settled cash:      ${view.cash_settled}")
    print(f"  Unsettled cash:    ${view.cash_unsettled}")
    print(f"  Allocated to bot:  ${view.capital.allocation}")
    print(f"  Available:         ${view.capital.available_settled}")
    print(f"  Open positions:    {view.capital.open_positions}/{view.capital.max_open_positions}")
    local = view.taken_at_local(services.settings.i18n.display_timezone)
    print(f"  Local time:        {local:%Y-%m-%d %H:%M %Z}")

    rule("TRADING CALENDAR")
    today = services.clock.now().date()
    calendar = services.calendar
    last_session = calendar.previous_trading_day(today, inclusive=True)
    print(f"  Today:                     {today} (trading day: {calendar.is_trading_day(today)})")
    print(f"  Last session:              {last_session}")
    print(f"  Next session:              {calendar.next_trading_day(today)}")
    print(f"  A sale today settles:      {calendar.settlement_date(last_session, 1)}")
    min_hold = services.settings.holding.min_holding_days
    earliest_exit = calendar.add_trading_days(last_session, min_hold)
    print(f"  Bought today, earliest exit ({min_hold}d): {earliest_exit}")

    rule("UNIVERSE (Mode A seed list)")
    universe = load_universe_file()
    unverified = [etf["symbol"] for etf in universe["etfs"] if not etf.get("verified")]
    print(f"  Configured ETFs: {len(universe['etfs'])}")
    for etf in universe["etfs"]:
        mark = "OK " if etf.get("verified") else "!! "
        name = etf.get("name") or "(name unconfirmed)"
        print(f"  {mark}{etf['symbol']:<6} {name}")
    if unverified:
        print(
            f"\n  {len(unverified)} symbol(s) are UNVERIFIED and will not be traded: "
            f"{', '.join(unverified)}"
        )
        print("  Confirm listing, certification and the certifying board from each")
        print("  fund's own documents, then set verified: true in config/universe_etf.yaml")

    rule("MARKET DATA")
    end = calendar.previous_trading_day(today, inclusive=True)
    start = end - dt.timedelta(days=120)
    for symbol in [etf["symbol"] for etf in universe["etfs"][:3]]:
        result = services.market_data.get_bars(symbol, start, end)
        if result.is_empty:
            print(f"  {symbol:<6} no data ({result.source})")
            continue
        last = result.frame.iloc[-1]
        flag = "  [SYNTHETIC -- NOT REAL PRICES]" if result.is_synthetic else ""
        print(
            f"  {symbol:<6} {len(result.frame):>4} bars  "
            f"last {result.frame.index[-1].date()} close ${last['close']:.2f}  "
            f"via {result.source}{flag}"
        )

    rule("COMPLIANCE GUARDS")
    from investment_box.shariah.constraints import CONSTRAINTS, is_forbidden_instrument

    print(f"  Margin allowed:            {CONSTRAINTS.margin_allowed}")
    print(f"  Short selling allowed:     {CONSTRAINTS.short_selling_allowed}")
    print(f"  Derivatives allowed:       {CONSTRAINTS.derivatives_allowed}")
    print(f"  Leveraged/inverse allowed: {CONSTRAINTS.leveraged_or_inverse_allowed}")
    print(f"  Crypto allowed:            {CONSTRAINTS.crypto_allowed}")
    for probe in ("TQQQ", "UVXY", "BITO", "SPUS"):
        reason = is_forbidden_instrument(probe)
        print(f"  {probe:<6} {'BLOCKED: ' + reason if reason else 'permitted'}")

    rule("RESULT")
    print("  Phase 1 components are wired and responding.")
    print("  Nothing trades yet: there is no engine, risk manager or strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
