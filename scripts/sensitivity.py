#!/usr/bin/env python3
"""Sweep a strategy's parameters and show how much the result depends on them.

    uv run python scripts/sensitivity.py
    uv run python scripts/sensitivity.py --years 5

A backtest at one setting of the knobs tells you almost nothing. The question
that matters is whether the result survives moving them: a Sharpe that reads
1.35 at a 2% band and 0.4 at 1% and 3% is a curve fit that happened to land,
and trading it means trading the fit rather than the effect.

This exists because `defensive_core` came back with Sharpe 1.37 and a -9.1%
drawdown, and the backtest report flagged it as too good to be true -- rightly,
at 18 trades. The sweep is what distinguished "real but modest" from
"overfitted": Sharpe stayed in 1.12-1.35 and max drawdown in -8.8% to -9.8%
across a fivefold range of the band and both settings of the guard.

Currently sweeps `defensive_core`. Copy the loop for another strategy; the
shape is the point, not the coverage.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from investment_box.backtest import WalkForwardBacktester
from investment_box.config.loader import load_settings, load_universe_file
from investment_box.core.clock import TradingCalendar
from investment_box.core.logging import configure_logging
from investment_box.data.repository import MarketDataRepository
from investment_box.strategies.defensive_core import DefensiveCore, DefensiveCoreConfig
from investment_box.universe.builder import UniverseBuilder

#: Wide enough that a fitted result cannot survive it.
BANDS: tuple[float, ...] = (0.00, 0.01, 0.02, 0.03, 0.05)


def main(args: argparse.Namespace) -> int:
    configure_logging("ERROR")
    settings = load_settings()
    repository = MarketDataRepository.from_settings(settings)

    symbols = [
        i.symbol for i in UniverseBuilder.load_instruments(load_universe_file())
    ]
    end = repository.clock_date()
    start = end - dt.timedelta(days=365 * args.years)
    warmup = dt.timedelta(days=400)
    calendar = TradingCalendar(start=start - warmup, end=end)

    bars = {}
    for symbol in sorted(set(symbols)):
        try:
            fetched = repository.get_bars(symbol, start - warmup, end, validate=False)
        except Exception as exc:  # noqa: BLE001 - one missing symbol is not fatal
            print(f"  (skipping {symbol}: {exc})")
            continue
        if not fetched.frame.empty:
            bars[symbol] = fetched.frame

    if not bars:
        print("No data. Nothing to sweep.")
        return 1

    backtester = WalkForwardBacktester(settings, calendar=calendar)

    print(f"defensive_core parameter sensitivity -- {start} to {end}")
    print()
    header = f"{'band':>6} {'guard':>6} | {'return':>8} {'sharpe':>7} {'maxDD':>8} {'trades':>7}"
    print(header)
    print("-" * len(header))

    sharpes: list[float] = []
    drawdowns: list[float] = []

    for band in BANDS:
        for guard in (True, False):
            config = DefensiveCoreConfig(
                entry_band=band, exit_band=band, require_defensive_uptrend=guard
            )
            result = backtester.run(
                DefensiveCore(config), bars, start=start, end=end,
                train_months=12, test_months=3,
            )
            metrics = result.metrics
            if metrics is None:
                print(f"{band:>6.0%} {guard!s:>6} | no measurable window")
                continue
            sharpe = metrics.sharpe or 0.0
            sharpes.append(sharpe)
            drawdowns.append(metrics.max_drawdown)
            print(
                f"{band:>6.0%} {guard!s:>6} | {metrics.total_return:>8.1%} "
                f"{sharpe:>7.2f} {metrics.max_drawdown:>8.1%} {metrics.num_trades:>7}"
            )

    if not sharpes:
        return 1

    print("-" * len(header))
    print(f"Sharpe across every setting: {min(sharpes):.2f} .. {max(sharpes):.2f}")
    print(f"MaxDD  across every setting: {min(drawdowns):.1%} .. {max(drawdowns):.1%}")
    print()
    spread = max(sharpes) - min(sharpes)
    if spread > 0.5:
        print(
            f"Sharpe moves {spread:.2f} across the grid. That is a fit, not an effect -- "
            f"the result depends on the knob more than on the market."
        )
    else:
        print(
            f"Sharpe moves only {spread:.2f} across the grid, so the result does not "
            f"rest on a particular setting. That is necessary for trusting it, and "
            f"not sufficient: a small sample can be stable and still wrong."
        )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=int, default=5, help="lookback window")
    raise SystemExit(main(parser.parse_args()))
