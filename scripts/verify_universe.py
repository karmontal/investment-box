#!/usr/bin/env python3
"""Check the seed universe against live market data, and print what only you can confirm.

    uv run python scripts/verify_universe.py
    uv run python scripts/verify_universe.py --json

Read-only, and deliberately so. This script can tell you that a ticker trades,
how liquid it is and when its first bar appeared; it CANNOT tell you that a fund
is Shariah-certified. Certification is a human attestation read off the fund's
own prospectus or factsheet, so nothing here ever writes ``verified: true`` --
you do that by hand in ``config/universe_etf.yaml`` once you have checked the
documents. A script that could flip that flag would defeat the point of it.

Exit code is 0 when every machine-checkable test passes, 1 otherwise. A universe
that passes here is still untradable until the human column is filled in too.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.config.loader import load_settings, load_universe_file
from investment_box.core.logging import configure_logging
from investment_box.data.repository import MarketDataRepository
from investment_box.shariah.constraints import is_forbidden_instrument
from investment_box.universe.builder import MIN_HISTORY_BARS, Instrument

#: How far back to ask for bars. Earlier than any US ETF, so the first bar
#: returned is the fund's first trading day as the data vendor sees it.
_EPOCH = dt.date(1990, 1, 1)

#: A fund whose last bar is older than this has probably been delisted, merged
#: or renamed. Generous enough to survive a long holiday weekend.
_STALE_AFTER_DAYS = 7

#: Fields a human must fill in before an entry means anything.
_HUMAN_FIELDS = ("name", "issuer", "certifying_board", "inception")


@dataclass
class SymbolReport:
    """Everything the machine could establish about one seed entry."""

    symbol: str
    configured_name: str | None = None
    configured_issuer: str | None = None
    verified_in_config: bool = False
    trades: bool = False
    first_bar: str | None = None
    last_bar: str | None = None
    bars: int = 0
    last_price: float | None = None
    avg_dollar_volume_20d: float | None = None
    vendor_name: str | None = None
    problems: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.trades and not self.problems


def _normalise(name: str) -> str:
    """Compare fund names ignoring punctuation and case, but not words.

    "S&P World" vs "S&P World (ex-US)" must still differ: the parenthetical is
    the mandate.
    """
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", name.lower()).split())


def _vendor_name(symbol: str) -> str | None:
    """Best-effort long name from the data vendor.

    Unauthoritative -- vendors rename funds late and sometimes wrongly. It is
    here to help you find the right prospectus, never to stand in for one.
    """
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).get_info()
    except Exception:  # noqa: BLE001 - a missing nicety must not fail the run
        return None
    name = info.get("longName") or info.get("shortName")
    return str(name) if name else None


def _check(
    entry: dict[str, object], repo: MarketDataRepository, today: dt.date, *, vendor: bool
) -> SymbolReport:
    instrument = Instrument.from_config(entry)
    report = SymbolReport(
        symbol=instrument.symbol,
        configured_name=instrument.name,
        configured_issuer=instrument.issuer,
        verified_in_config=instrument.verified,
    )

    report.missing_fields = [f for f in _HUMAN_FIELDS if not entry.get(f)]

    if vendor:
        report.vendor_name = _vendor_name(instrument.symbol)

    # A configured name that disagrees with the vendor's usually means the fund
    # was renamed or the entry was written from memory. Either way the mandate
    # you believe you are buying is not the one on the prospectus.
    if report.vendor_name and instrument.name:
        if _normalise(report.vendor_name) != _normalise(instrument.name):
            report.problems.append(
                f"name drift: config says {instrument.name!r}, "
                f"vendor says {report.vendor_name!r} -- confirm which is current"
            )

    # A forbidden name or ticker is fatal regardless of what the data says.
    forbidden = is_forbidden_instrument(
        instrument.symbol,
        report.vendor_name or instrument.name,
        instrument.asset_class.value,
    )
    if forbidden:
        report.problems.append(f"FORBIDDEN: {forbidden}")

    try:
        result = repo.get_bars(instrument.symbol, _EPOCH, today, validate=False)
    except Exception as exc:  # noqa: BLE001 - reported per symbol, never fatal
        report.problems.append(f"no data: {exc}")
        return report

    frame = result.frame
    if frame.empty:
        report.problems.append("no bars returned -- ticker may not exist or may be delisted")
        return report

    report.trades = True
    report.bars = len(frame)
    first = frame.index[0].date()
    last = frame.index[-1].date()
    report.first_bar = first.isoformat()
    report.last_bar = last.isoformat()
    report.last_price = round(float(frame["close"].iloc[-1]), 2)

    window = frame.tail(20)
    report.avg_dollar_volume_20d = round(
        float((window["close"] * window["volume"]).mean()), 0
    )

    stale_days = (today - last).days
    if stale_days > _STALE_AFTER_DAYS:
        report.problems.append(
            f"stale: last bar {last} is {stale_days} days old -- delisted, merged or renamed?"
        )

    if report.bars < MIN_HISTORY_BARS:
        report.problems.append(
            f"only {report.bars} bars; strategies need {MIN_HISTORY_BARS} to rank it"
        )

    # Cross-check the configured inception against the first bar we can see.
    # A configured date EARLIER than the first bar is the dangerous direction:
    # it would let a backtest hold the fund before it existed.
    if instrument.inception is not None:
        drift = (first - instrument.inception).days
        if drift > 5:
            report.problems.append(
                f"inception {instrument.inception} precedes the first bar {first} "
                f"by {drift} days -- a backtest could hold it before it traded"
            )

    return report


def _money(value: float | None) -> str:
    return "-" if value is None else f"${value:,.0f}"


def _render(reports: list[SymbolReport], today: dt.date) -> str:
    lines: list[str] = []
    lines.append(f"UNIVERSE VERIFICATION -- {today.isoformat()}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"{'SYM':<6} {'TRADES':<7} {'FIRST BAR':<11} {'BARS':>6} "
                 f"{'PRICE':>9} {'20d $VOL':>14}  CFG")
    lines.append("-" * 78)
    for r in reports:
        lines.append(
            f"{r.symbol:<6} {('yes' if r.trades else 'NO'):<7} "
            f"{(r.first_bar or '-'):<11} {r.bars:>6} "
            f"{('-' if r.last_price is None else f'${r.last_price:,.2f}'):>9} "
            f"{_money(r.avg_dollar_volume_20d):>14}  "
            f"{'verified' if r.verified_in_config else 'unverified'}"
        )

    problems = [r for r in reports if r.problems]
    if problems:
        lines.append("")
        lines.append("PROBLEMS")
        lines.append("-" * 78)
        for r in problems:
            for p in r.problems:
                lines.append(f"  {r.symbol}: {p}")

    lines.append("")
    lines.append("WHAT THE MACHINE CANNOT CHECK")
    lines.append("-" * 78)
    lines.append("Read each fund's own prospectus or factsheet and confirm, in writing:")
    lines.append("  1. the fund is still listed and has not merged or been renamed,")
    lines.append("  2. it is CURRENTLY Shariah-certified (certification can lapse),")
    lines.append("  3. the name of the certifying board or advisor,")
    lines.append("  4. it is neither leveraged nor inverse,")
    lines.append("  5. the inception date printed in the prospectus.")
    lines.append("")
    for r in reports:
        if not r.missing_fields:
            continue
        vendor = f"  [vendor says: {r.vendor_name}]" if r.vendor_name else ""
        lines.append(f"  {r.symbol}: fill in {', '.join(r.missing_fields)}{vendor}")

    lines.append("")
    lines.append("Then set `verified: true` for that entry in config/universe_etf.yaml.")
    lines.append("This script will never set it for you: it is your attestation, not mine.")
    return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
    configure_logging("ERROR")
    settings = load_settings()
    repo = MarketDataRepository.from_settings(settings)
    today = repo.clock_date()

    config = load_universe_file()
    reports = [
        _check(entry, repo, today, vendor=not args.no_vendor)
        for entry in config.get("etfs", [])
    ]

    if args.json:
        print(json.dumps([asdict(r) for r in reports], indent=2))
    else:
        print(_render(reports, today))

    failures = [r for r in reports if not r.ok]
    if failures and not args.json:
        print()
        print(f"{len(failures)} of {len(reports)} symbol(s) failed a machine check.")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--no-vendor", action="store_true", help="skip the vendor metadata lookup (offline)"
    )
    raise SystemExit(main(parser.parse_args()))
