#!/usr/bin/env python3
"""Run the walk-forward backtest across every strategy and report honestly.

    uv run python scripts/run_backtest.py
    uv run python scripts/run_backtest.py --start 2021-01-01 --send

``--send`` posts the summary to the Telegram channel.

Research mode: this script backtests the configured universe even though the
symbols are marked unverified, because otherwise there is nothing to test. That
is fine for research and is NOT permission to trade them -- the engine still
refuses unverified symbols. The caveat appears at the top of every report.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.backtest import (
    BacktestReport,
    CostModel,
    WalkForwardBacktester,
    buy_and_hold,
)
from investment_box.config.loader import load_settings, load_universe_file
from investment_box.core.clock import TradingCalendar
from investment_box.core.logging import configure_logging
from investment_box.data.repository import MarketDataRepository
from investment_box.features.regime import RegimeDetector
from investment_box.strategies import STRATEGY_REGISTRY
from investment_box.universe import UniverseBuilder


def main(args: argparse.Namespace) -> int:
    configure_logging("WARNING")
    settings = load_settings()
    universe_config = load_universe_file()
    repository = MarketDataRepository.from_settings(settings)

    instruments = UniverseBuilder.load_instruments(universe_config)
    symbols = [i.symbol for i in instruments]
    benchmarks = UniverseBuilder.benchmark_symbols(universe_config)

    end = args.end or (dt.datetime.now(tz=dt.UTC).date() - dt.timedelta(days=1))
    start = args.start or (end - dt.timedelta(days=365 * 5))
    # Explicit range: the anchored default only spans +/- 800 days around today,
    # which would silently drop every test window before that.
    calendar = TradingCalendar(start=start, end=end)
    end = calendar.previous_trading_day(end, inclusive=True)

    print(f"Fetching {len(symbols)} symbols + {len(benchmarks)} benchmarks, {start} to {end} ...")
    fetched = repository.get_many([*symbols, *benchmarks], start, end)

    bars = {s: r.frame for s, r in fetched.items() if s in symbols and not r.frame.empty}
    missing = [s for s in symbols if s not in bars]
    synthetic = any(r.is_synthetic for r in fetched.values())

    coverage = {
        s: (f.index[0].date(), f.index[-1].date(), len(f)) for s, f in bars.items()
    }
    print("\nData coverage:")
    for symbol, (first, last, count) in sorted(coverage.items()):
        print(f"  {symbol:<6} {first} .. {last}  ({count} bars, {count / 252:.1f} years)")
    if missing:
        print(f"  no data: {', '.join(missing)}")

    report = BacktestReport(
        title="Shariah ETF strategies — walk-forward comparison",
        start=start,
        end=end,
        initial_capital=float(settings.capital.allocation_usd),
    )

    report.global_caveats.append(
        "RESEARCH ONLY. Every symbol in config/universe_etf.yaml is marked "
        "verified: false. The engine refuses to trade them; this backtest includes "
        "them so there is something to measure. Verify listing and Shariah "
        "certification before acting on any of this."
    )
    report.global_caveats.append(
        "There is NO point-in-time Shariah compliance history for this universe. "
        "The backtest assumes today's compliance status held throughout, which is "
        "look-ahead bias. A fund that was non-compliant in 2022 would have been "
        "untradeable then but is traded here."
    )
    shortest = min((c[2] for c in coverage.values()), default=0)
    if shortest < 504:
        report.global_caveats.append(
            f"The shortest history in the universe is {shortest} bars "
            f"({shortest / 252:.1f} years). Several of these funds launched recently, so "
            f"the usable sample is far shorter than the requested window."
        )
    if synthetic:
        report.global_caveats.append(
            "SYNTHETIC DATA WAS USED for at least one symbol. Those numbers are "
            "generated and mean nothing."
        )
    report.global_caveats.append(
        "Limit orders are assumed to fill at the next open. In reality some do not "
        "fill at all, which this backtest does not model."
    )

    if not bars:
        print("\nNo usable price data. Nothing to test.")
        return 1

    detector = RegimeDetector(repository)
    print("\nComputing market regime ...")
    sessions = [d for d in calendar.sessions if start <= d <= end]
    regimes = {}
    for day in sessions[::5]:  # every 5th session, forward-filled below
        regimes[day] = detector.detect(day)
    # Forward-fill so every session has the most recent known regime.
    filled = {}
    last = None
    for day in sessions:
        if day in regimes:
            last = regimes[day]
        if last is not None:
            filled[day] = last
    regimes = filled

    backtester = WalkForwardBacktester(settings, calendar=calendar)

    for name, strategy_cls in STRATEGY_REGISTRY.items():
        if args.only and name != args.only:
            continue
        print(f"Running {name} ...")
        result = backtester.run(
            strategy_cls(),
            bars,
            start=start,
            end=end,
            train_months=args.train_months,
            test_months=args.test_months,
            regimes=regimes,
        )
        report.add(result)

    # Benchmarks must cover the SAME period the strategies were evaluated on.
    # Strategies only trade inside their out-of-sample test windows, so
    # comparing their 2-year result against a 7-year buy-and-hold would be a
    # meaningless (and flattering-to-buy-and-hold) comparison.
    tested = [r for r in report.results if r.metrics and r.metrics.start is not None]
    if tested:
        oos_start = min(r.metrics.start for r in tested).date()  # type: ignore[union-attr]
        oos_end = max(r.metrics.end for r in tested).date()  # type: ignore[union-attr]
        report.global_caveats.append(
            f"Strategies are evaluated only on their out-of-sample windows "
            f"({oos_start} to {oos_end}); benchmarks are measured over the same "
            f"period so the comparison is like for like. The earlier years are "
            f"used for training only."
        )
    else:
        oos_start, oos_end = start, end

    costs = CostModel(settings.costs)
    for symbol in benchmarks:
        fetched_result = fetched.get(symbol)
        if fetched_result is None or fetched_result.frame.empty:
            continue
        frame = fetched_result.frame
        window = frame.loc[
            (frame.index >= str(oos_start)) & (frame.index <= str(oos_end))
        ]
        if window.empty:
            continue
        report.add_benchmark(
            buy_and_hold(
                window, float(settings.capital.allocation_usd), costs, f"buy & hold {symbol}"
            )
        )

    if args.save_track_record:
        _persist_track_records(report, oos_start, oos_end)

    text = report.to_text()
    print("\n" + text)

    out = Path(args.output) if args.output else Path("reports/backtest.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Saved to {out}")

    if args.send:
        asyncio.run(_send(settings, report))
    return 0


def _persist_track_records(report: BacktestReport, oos_start, oos_end) -> None:  # noqa: ANN001
    """Store what each strategy actually achieved on unseen data.

    Only walk-forward results are written as out-of-sample. That flag is what
    the engine checks before it will act on a forecast, so setting it for
    anything fitted in-sample would defeat the one guard that matters.
    """
    from investment_box.forecast.track_record_store import TrackRecordStore
    from investment_box.services.container import build_services

    services = build_services(configure_logs=False)
    store = TrackRecordStore(services.database)

    saved = 0
    for result in report.results:
        if result.metrics is None or result.metrics.start is None:
            print(f"  skipped {result.strategy}: produced no measurable window")
            continue
        store.save(
            result.metrics,
            source="walk_forward_backtest",
            out_of_sample=True,
            measured_at=services.clock.now(),
            notes="; ".join(result.caveats) or None,
        )
        saved += 1
        wr = result.metrics.win_rate
        print(
            f"  saved {result.strategy}: {result.metrics.num_trades} trades, "
            f"win rate {'n/a' if wr is None else f'{wr:.1%}'}, "
            f"{oos_start} to {oos_end}"
        )

    if saved:
        print(
            f"\nStored {saved} out-of-sample record(s). The engine reads them at "
            f"startup; restart it to pick them up."
        )


async def _send(settings, report: BacktestReport) -> None:  # noqa: ANN001
    from investment_box.services.container import build_services
    from investment_box.telegram.bot import build_telegram_stack

    services = build_services(configure_logs=False)
    stack = build_telegram_stack(
        settings=services.settings,
        secrets=services.secrets,
        portfolio=services.portfolio,
        approvals=services.approvals,
        audit=services.audit,
        clock=services.clock,
    )
    await stack.queue.start()
    stack.broadcast.custom(report.to_telegram(), priority=50)
    await stack.queue.drain()
    await asyncio.sleep(1.0)
    print(f"Telegram: sent={stack.queue.sent_count} dropped={stack.queue.dropped_count}")
    await stack.queue.stop()


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=_date, default=None)
    parser.add_argument("--end", type=_date, default=None)
    parser.add_argument("--train-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=3)
    parser.add_argument("--only", type=str, default=None, help="run one strategy by name")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--send", action="store_true", help="post the summary to Telegram")
    parser.add_argument(
        "--save-track-record",
        action="store_true",
        help=(
            "persist each strategy's out-of-sample result so the engine can act on "
            "it. Without a stored record every forecast reads as a coin flip and "
            "nothing trades."
        ),
    )
    raise SystemExit(main(parser.parse_args()))
