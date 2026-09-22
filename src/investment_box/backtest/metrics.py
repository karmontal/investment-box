"""Performance metrics.

Two principles:

* **A metric computed on too few observations is not a metric.** Every figure
  carries the sample size it came from, and :func:`evaluate` marks results as
  unreliable rather than quietly reporting a Sharpe ratio derived from eleven
  trades.
* **Report the uncomfortable numbers as prominently as the flattering ones.**
  Cost as a share of gross profit, exposure, and turnover all determine whether
  a strategy is real, and all tend to be omitted from backtest summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252

#: Below this many round trips, trade statistics are noise. Chosen as a round
#: number well under any threshold at which a win rate would be meaningful.
MIN_TRADES_FOR_CONFIDENCE = 30
#: Below this many observations, so is the Sharpe ratio.
MIN_DAYS_FOR_CONFIDENCE = 252


@dataclass
class PerformanceMetrics:
    """Everything the report shows for one strategy."""

    name: str
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    trading_days: int = 0

    total_return: float = 0.0
    cagr: float = 0.0
    annual_volatility: float = 0.0
    sharpe: float | None = None
    sortino: float | None = None
    max_drawdown: float = 0.0
    max_drawdown_days: int = 0
    calmar: float | None = None

    num_trades: int = 0
    win_rate: float | None = None
    profit_factor: float | None = None
    avg_trade_pct: float | None = None
    avg_winner_pct: float | None = None
    avg_loser_pct: float | None = None
    avg_holding_days: float | None = None

    exposure: float = 0.0
    turnover: float = 0.0
    total_costs: float = 0.0
    gross_profit: float = 0.0
    net_profit: float = 0.0

    #: Reasons the numbers above should not be trusted.
    reliability_warnings: list[str] = field(default_factory=list)

    @property
    def cost_share_of_gross(self) -> float | None:
        """Costs as a fraction of gross profit.

        Above ~30% the strategy is mostly paying the broker; above 100% it
        would have been profitable without costs and is not with them.
        """
        if self.gross_profit <= 0:
            return None
        return self.total_costs / self.gross_profit

    @property
    def is_reliable(self) -> bool:
        return not self.reliability_warnings

    @property
    def looks_too_good(self) -> bool:
        """Heuristics for results that usually indicate a bug or overfitting.

        Not proof of either -- but on a two-year sample of eight ETFs, a
        Sharpe above 2 or a win rate above 70% is far more likely to be a
        look-ahead leak or curve-fitting than a real edge.
        """
        suspicious = [
            self.sharpe is not None and self.sharpe > 2.0,
            self.win_rate is not None and self.win_rate > 0.70 and self.num_trades >= 10,
            self.max_drawdown > -0.03 and self.total_return > 0.20,
            self.profit_factor is not None and self.profit_factor > 4.0,
        ]
        return any(suspicious)

    def overfitting_warnings(self) -> list[str]:
        """Specific reasons to distrust an unusually good result."""
        out: list[str] = []
        if self.sharpe is not None and self.sharpe > 2.0:
            out.append(
                f"Sharpe {self.sharpe:.2f} is implausibly high for a simple strategy on "
                f"daily data. Suspect look-ahead or an overfitted parameter."
            )
        if self.win_rate is not None and self.win_rate > 0.70 and self.num_trades >= 10:
            out.append(
                f"Win rate {self.win_rate:.0%} over {self.num_trades} trades is very high. "
                f"Check that exits are not using same-bar information."
            )
        if self.max_drawdown > -0.03 and self.total_return > 0.20:
            out.append(
                f"A {self.total_return:.0%} return with only a {abs(self.max_drawdown):.1%} "
                f"drawdown is not a realistic risk/reward profile."
            )
        if self.profit_factor is not None and self.profit_factor > 4.0:
            out.append(f"Profit factor {self.profit_factor:.1f} is far outside the normal range.")
        return out


def _annualise_return(equity: pd.Series, trading_days: int) -> float:
    if equity.empty or trading_days <= 0 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    if total <= 0:
        return -1.0
    years = trading_days / TRADING_DAYS
    return float(total ** (1 / years) - 1) if years > 0 else 0.0


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Worst peak-to-trough decline, and how long it lasted in days."""
    if equity.empty:
        return 0.0, 0
    running_peak = equity.cummax()
    drawdown_series = equity / running_peak - 1.0
    worst = float(drawdown_series.min())

    # Longest stretch spent below a prior peak.
    below = drawdown_series < 0
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return worst, longest


def evaluate(
    name: str,
    equity: pd.Series,
    trades: list[dict[str, object]] | None = None,
    *,
    exposure: float = 0.0,
    turnover: float = 0.0,
    total_costs: float = 0.0,
    gross_profit: float = 0.0,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Compute every metric for one equity curve.

    ``risk_free_rate`` defaults to zero. For a Shariah-compliant account an
    interest-bearing risk-free rate is not an available alternative, so
    excess-return metrics are computed against zero rather than against a rate
    the account could not earn.
    """
    trades = trades or []
    metrics = PerformanceMetrics(name=name)

    if equity.empty or len(equity) < 2:
        metrics.reliability_warnings.append("no equity curve produced; nothing to evaluate")
        return metrics

    metrics.start = equity.index[0]
    metrics.end = equity.index[-1]
    metrics.trading_days = len(equity)
    metrics.total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)
    metrics.cagr = _annualise_return(equity, len(equity))

    daily = equity.pct_change().dropna()
    if not daily.empty:
        metrics.annual_volatility = float(daily.std() * np.sqrt(TRADING_DAYS))
        excess = daily - risk_free_rate / TRADING_DAYS
        if daily.std() > 0:
            metrics.sharpe = float(excess.mean() / daily.std() * np.sqrt(TRADING_DAYS))
        downside = daily[daily < 0]
        if len(downside) > 1 and downside.std() > 0:
            metrics.sortino = float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))

    metrics.max_drawdown, metrics.max_drawdown_days = max_drawdown(equity)
    if metrics.max_drawdown < 0:
        metrics.calmar = metrics.cagr / abs(metrics.max_drawdown)

    metrics.exposure = exposure
    metrics.turnover = turnover
    metrics.total_costs = total_costs
    metrics.gross_profit = gross_profit
    metrics.net_profit = gross_profit - total_costs

    _add_trade_stats(metrics, trades)
    _add_reliability_warnings(metrics, equity)
    return metrics


def _add_trade_stats(metrics: PerformanceMetrics, trades: list[dict[str, object]]) -> None:
    metrics.num_trades = len(trades)
    if not trades:
        return

    returns = np.array([t.get("return_pct", 0.0) for t in trades], dtype=float)
    profits = np.array([t.get("net_pnl", 0.0) for t in trades], dtype=float)
    holding = [
        float(value)
        for t in trades
        if (value := t.get("holding_days")) is not None and isinstance(value, (int, float))
    ]

    winners = returns[returns > 0]
    losers = returns[returns <= 0]

    metrics.win_rate = float(len(winners) / len(returns)) if len(returns) else None
    metrics.avg_trade_pct = float(returns.mean())
    metrics.avg_winner_pct = float(winners.mean()) if len(winners) else None
    metrics.avg_loser_pct = float(losers.mean()) if len(losers) else None
    metrics.avg_holding_days = float(np.mean(holding)) if holding else None

    gross_wins = profits[profits > 0].sum()
    gross_losses = abs(profits[profits < 0].sum())
    if gross_losses > 0:
        metrics.profit_factor = float(gross_wins / gross_losses)
    elif gross_wins > 0:
        # No losing trades at all: report it as undefined rather than infinite,
        # and let the reliability warnings explain why that is suspicious.
        metrics.profit_factor = None
        metrics.reliability_warnings.append(
            "no losing trades in the sample; profit factor is undefined, which on a "
            "short sample usually means too few trades rather than a good strategy"
        )


def _add_reliability_warnings(metrics: PerformanceMetrics, equity: pd.Series) -> None:
    if metrics.num_trades < MIN_TRADES_FOR_CONFIDENCE:
        metrics.reliability_warnings.append(
            f"only {metrics.num_trades} trades (want >= {MIN_TRADES_FOR_CONFIDENCE}); "
            f"win rate and profit factor are not meaningful at this sample size"
        )
    if metrics.trading_days < MIN_DAYS_FOR_CONFIDENCE:
        years = metrics.trading_days / TRADING_DAYS
        metrics.reliability_warnings.append(
            f"only {metrics.trading_days} trading days ({years:.1f} years); "
            f"Sharpe and drawdown estimates are unstable over such a short window"
        )
    if metrics.exposure < 0.10 and metrics.trading_days > 0:
        metrics.reliability_warnings.append(
            f"invested only {metrics.exposure:.0%} of the time; the strategy spent most "
            f"of the sample in cash, so returns say little about its edge"
        )
    cost_share = metrics.cost_share_of_gross
    if cost_share is not None and cost_share > 0.30:
        metrics.reliability_warnings.append(
            f"costs consumed {cost_share:.0%} of gross profit"
        )
