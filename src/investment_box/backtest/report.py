"""Backtest comparison reporting.

The report is written to be read sceptically. Strategies that failed appear
alongside those that did well, caveats are printed before results rather than
in a footnote, and any result that looks too good triggers an explicit
overfitting warning.

The reasoning: the failure mode for a personal trading system is not "the
report was not pretty enough". It is deploying capital on a strategy whose
backtest looked good for reasons that will not repeat.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from investment_box.backtest.engine import BacktestResult
from investment_box.backtest.metrics import PerformanceMetrics

RULE = "=" * 78
THIN = "-" * 78


def _pct(value: float | None, places: int = 2) -> str:
    return "—" if value is None else f"{value * 100:.{places}f}%"


def _num(value: float | None, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}"


@dataclass
class BacktestReport:
    """A comparison across strategies and benchmarks."""

    title: str
    start: dt.date
    end: dt.date
    initial_capital: float
    results: list[BacktestResult] = field(default_factory=list)
    benchmarks: list[BacktestResult] = field(default_factory=list)
    #: Caveats about the whole run: data coverage, compliance history, etc.
    global_caveats: list[str] = field(default_factory=list)

    def add(self, result: BacktestResult) -> None:
        self.results.append(result)

    def add_benchmark(self, result: BacktestResult) -> None:
        self.benchmarks.append(result)

    @property
    def all_results(self) -> list[BacktestResult]:
        return [*self.results, *self.benchmarks]

    @property
    def any_reliable(self) -> bool:
        return any(
            r.metrics is not None and r.metrics.is_reliable for r in self.results
        )

    def _best_benchmark(self) -> BacktestResult | None:
        scored = [b for b in self.benchmarks if b.metrics is not None]
        if not scored:
            return None
        return max(scored, key=lambda b: b.metrics.total_return if b.metrics else 0.0)

    def best(self) -> BacktestResult | None:
        """Highest total return among strategies that actually traded.

        Explicitly *not* a recommendation -- with these sample sizes the
        ranking is mostly noise, and the text says so.
        """
        traded = [r for r in self.results if r.metrics and r.num_trades > 0]
        if not traded:
            return None
        return max(traded, key=lambda r: r.metrics.total_return if r.metrics else 0.0)

    # ---------------------------------------------------------------- render

    def to_text(self) -> str:
        lines: list[str] = [
            RULE,
            self.title,
            RULE,
            f"Period:          {self.start} to {self.end}",
            f"Capital:         ${self.initial_capital:,.2f}",
            f"Strategies:      {len(self.results)}",
            "",
        ]

        lines.extend(self._caveats_section())
        lines.extend(self._summary_table())
        lines.extend(self._detail_sections())
        lines.extend(self._verdict())
        return "\n".join(lines)

    def _caveats_section(self) -> list[str]:
        if not self.global_caveats:
            return []
        lines = ["READ THIS FIRST", THIN]
        lines += [f"  * {caveat}" for caveat in self.global_caveats]
        lines += [""]
        return lines

    def _summary_table(self) -> list[str]:
        header = (
            f"{'Strategy':<26}{'Return':>9}{'CAGR':>9}{'Sharpe':>8}"
            f"{'MaxDD':>9}{'Trades':>8}{'Win%':>7}{'Exp%':>7}"
        )
        lines = ["SUMMARY", THIN, header, THIN]

        for result in self.all_results:
            metrics = result.metrics
            if metrics is None:
                lines.append(f"{result.strategy:<26}{'no result — see caveats':>50}")
                continue
            flag = " !" if metrics.looks_too_good else ("  " if metrics.is_reliable else " ?")
            lines.append(
                f"{result.strategy:<26}"
                f"{_pct(metrics.total_return, 1):>9}"
                f"{_pct(metrics.cagr, 1):>9}"
                f"{_num(metrics.sharpe):>8}"
                f"{_pct(metrics.max_drawdown, 1):>9}"
                f"{metrics.num_trades:>8}"
                f"{_pct(metrics.win_rate, 0):>7}"
                f"{_pct(metrics.exposure, 0):>7}"
                f"{flag}"
            )

        lines += [
            THIN,
            "  ? = the sample is too small for these figures to be meaningful",
            "  ! = the result looks too good; suspect overfitting or a data leak",
            "",
        ]
        return lines

    def _detail_sections(self) -> list[str]:
        lines: list[str] = []
        for result in self.all_results:
            lines += [RULE, result.strategy, THIN]

            if result.metrics is None:
                lines += ["  Produced no result."]
                lines += [f"  * {c}" for c in result.caveats]
                lines += [""]
                continue

            lines += self._metric_block(result.metrics)

            if result.caveats:
                lines += ["", "  Run caveats:"]
                lines += [f"    * {c}" for c in result.caveats]

            if result.metrics.reliability_warnings:
                lines += ["", "  Why these numbers may not mean much:"]
                lines += [f"    * {w}" for w in result.metrics.reliability_warnings]

            if result.metrics.looks_too_good:
                lines += ["", "  !! THIS RESULT LOOKS TOO GOOD TO BE TRUE !!"]
                lines += [f"    * {w}" for w in result.metrics.overfitting_warnings()]
                lines += [
                    "    Before believing it, check: does the strategy read the bar it",
                    "    trades on? Were parameters chosen after seeing this data?",
                ]
            lines += [""]
        return lines

    @staticmethod
    def _metric_block(m: PerformanceMetrics) -> list[str]:
        cost_share = m.cost_share_of_gross
        return [
            f"  Total return      {_pct(m.total_return)}          "
            f"CAGR            {_pct(m.cagr)}",
            f"  Sharpe            {_num(m.sharpe):<12}      "
            f"Sortino         {_num(m.sortino)}",
            f"  Max drawdown      {_pct(m.max_drawdown)}          "
            f"Calmar          {_num(m.calmar)}",
            f"  Volatility (ann)  {_pct(m.annual_volatility)}          "
            f"Trading days    {m.trading_days}",
            "",
            f"  Trades            {m.num_trades:<12}      "
            f"Win rate        {_pct(m.win_rate, 0)}",
            f"  Avg trade         {_pct(m.avg_trade_pct)}          "
            f"Profit factor   {_num(m.profit_factor)}",
            f"  Avg winner        {_pct(m.avg_winner_pct)}          "
            f"Avg loser       {_pct(m.avg_loser_pct)}",
            f"  Avg holding       {_num(m.avg_holding_days, 1)} days     "
            f"Exposure        {_pct(m.exposure, 0)}",
            "",
            f"  Gross profit      ${m.gross_profit:,.2f}",
            f"  Costs paid        ${m.total_costs:,.2f}"
            + (f"  ({cost_share:.0%} of gross)" if cost_share is not None else ""),
            f"  Net profit        ${m.net_profit:,.2f}",
        ]

    def _verdict(self) -> list[str]:
        lines = [RULE, "VERDICT", THIN]

        traded = [r for r in self.results if r.num_trades > 0]
        if not traded:
            lines += [
                "  No strategy produced a single trade over this period.",
                "  That is a result, not a bug: with this universe, this sample length",
                "  and these filters, there was nothing to act on. Read the caveats.",
                "",
            ]
            return lines

        if not self.any_reliable:
            lines += [
                "  NOT ONE of these results rests on enough data to be trusted.",
                "  Every strategy is flagged for sample size, trade count or both.",
                "",
                "  The honest conclusion is that this backtest cannot tell you which",
                "  strategy is better. It can tell you which ones are broken, and it",
                "  can tell you roughly what trading costs. It cannot rank edges that",
                "  a two-to-three year sample is far too short to measure.",
                "",
            ]

        best = self.best()
        best_benchmark = self._best_benchmark()

        # The comparison that matters most, stated first and without hedging.
        # A strategy that loses to buying the index and sitting still has not
        # earned the complexity, the screen time or the execution risk.
        if best is not None and best.metrics is not None and best_benchmark is not None:
            strategy_return = best.metrics.total_return
            benchmark_return = best_benchmark.metrics.total_return  # type: ignore[union-attr]
            if benchmark_return > strategy_return:
                lines += [
                    "  NO STRATEGY BEAT BUYING AND HOLDING.",
                    "",
                    f"  Best strategy:  {best.strategy} at {_pct(strategy_return, 1)}",
                    f"  Best benchmark: {best_benchmark.strategy} at "
                    f"{_pct(benchmark_return, 1)}",
                    "",
                    "  On this sample, over this period, the active strategies destroyed",
                    "  value relative to simply holding a compliant ETF. Trading costs,",
                    "  time out of the market, and whipsaw all subtract from a buy-and-hold",
                    "  return that required no decisions at all.",
                    "",
                ]
            else:
                lines += [
                    f"  Best strategy:  {best.strategy} at {_pct(strategy_return, 1)}",
                    f"  Best benchmark: {best_benchmark.strategy} at "
                    f"{_pct(benchmark_return, 1)}",
                    "",
                    "  The strategy beat buy and hold on this sample. Treat that as a",
                    "  hypothesis, not a finding, until paper trading agrees with it.",
                    "",
                ]
        elif best is not None and best.metrics is not None:
            lines += [
                f"  Highest return: {best.strategy} at "
                f"{_pct(best.metrics.total_return, 1)} over {best.metrics.num_trades} trades.",
                "",
            ]

        if best is not None and best.metrics is not None and not best.metrics.is_reliable:
            lines += [
                "  The ranking between strategies is mostly noise at this sample size.",
                "",
            ]

        # Where the money actually went.
        expensive = [
            r for r in self.results
            if r.metrics and (r.metrics.cost_share_of_gross or 0) > 0.30
        ]
        if expensive:
            lines += ["  Costs are eating these strategies:"]
            for result in expensive:
                metrics = result.metrics
                if metrics is None:
                    continue
                share = metrics.cost_share_of_gross or 0.0
                lines += [
                    f"    {result.strategy}: ${metrics.total_costs:,.2f} of costs on "
                    f"${metrics.gross_profit:,.2f} gross ({share:.0%})"
                ]
            lines += [
                "    A strategy whose costs are a large share of gross profit needs a",
                "    bigger edge per trade or fewer trades, not better parameters.",
                "",
            ]

        benchmark_lines = [
            f"  {b.strategy}: {_pct(b.metrics.total_return, 1)}"
            for b in self.benchmarks
            if b.metrics is not None
        ]
        if benchmark_lines:
            lines += ["  All benchmarks:", *benchmark_lines]

        lines += [
            "",
            "  Before trading any of this: paper trade it first, and compare the paper",
            "  results against these numbers. A strategy whose live behaviour differs",
            "  materially from its backtest has a bug in one or the other.",
            "",
        ]
        return lines

    # ------------------------------------------------------------- telegram

    def to_telegram(self) -> str:
        """A compact summary for the broadcast channel.

        Deliberately leads with the caveat rather than the headline number.
        """
        lines = [f"<b>📊 Backtest: {self.title}</b>", f"<i>{self.start} → {self.end}</i>", ""]

        if not self.any_reliable:
            lines += [
                "⚠️ <b>No result here is statistically reliable.</b>",
                "The sample is too short to rank these strategies.",
                "",
            ]

        for result in self.all_results:
            m = result.metrics
            if m is None:
                lines.append(f"• <b>{result.strategy}</b> — no result")
                continue
            flag = " ‼️" if m.looks_too_good else (" ⚠️" if not m.is_reliable else "")
            lines.append(
                f"• <b>{result.strategy}</b>{flag}\n"
                f"   {_pct(m.total_return, 1)} | Sharpe {_num(m.sharpe)} | "
                f"DD {_pct(m.max_drawdown, 1)} | {m.num_trades} trades"
            )

        best = self.best()
        best_benchmark = self._best_benchmark()
        if (
            best is not None
            and best.metrics is not None
            and best_benchmark is not None
            and best_benchmark.metrics is not None
            and best_benchmark.metrics.total_return > best.metrics.total_return
        ):
            lines += [
                "",
                "<b>No strategy beat buying and holding.</b>",
                f"Best strategy {_pct(best.metrics.total_return, 1)} vs "
                f"{best_benchmark.strategy} {_pct(best_benchmark.metrics.total_return, 1)}.",
            ]

        if any(r.metrics and r.metrics.looks_too_good for r in self.results):
            lines += ["", "‼️ A result looks too good to be true — check for overfitting."]

        return "\n".join(lines)
