"""Walk-forward, event-driven backtester.

The design exists to make look-ahead structurally impossible rather than merely
avoided by care:

* **Decide on the close of bar i, execute at the open of bar i+1.** A strategy
  never trades at a price it used to decide. This single rule accounts for most
  of the gap between naive backtests and live results.
* **The strategy is handed a trimmed view.** ``StrategyContext`` contains
  features sliced to ``as_of``; the bar being decided on is not in it.
* **Walk-forward, never one split.** The sample is divided into consecutive
  train/test windows. Anything fitted (the ML model) is fitted on the training
  window only and evaluated on the untouched test window that follows.
* **A gap between train and test.** Because labels use a forward horizon, the
  last few training rows overlap the first test days. The gap removes that
  overlap; without it the model is partly trained on its own test period.

Also modelled, because each one changes results materially at this account
size: whole-share rounding, T+1 settlement, the minimum holding period, and
the cash buffer.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from investment_box.backtest.costs import CostModel
from investment_box.backtest.metrics import PerformanceMetrics, evaluate
from investment_box.config.schema import Settings
from investment_box.core.clock import TradingCalendar
from investment_box.core.logging import get_logger
from investment_box.core.types import Side
from investment_box.features.pipeline import FeatureSet, build_features
from investment_box.features.regime import RegimeDetector, RegimeState
from investment_box.strategies.base import Strategy, StrategyContext

log = get_logger(__name__)


@dataclass
class BacktestPosition:
    """An open position inside the simulation."""

    symbol: str
    quantity: float
    entry_price: float
    entry_date: dt.date
    entry_cost: float
    stop_price: float | None = None
    take_profit_price: float | None = None

    def market_value(self, price: float) -> float:
        return self.quantity * price


@dataclass
class BacktestTrade:
    """A completed round trip."""

    symbol: str
    entry_date: dt.date
    exit_date: dt.date
    entry_price: float
    exit_price: float
    quantity: float
    gross_pnl: float
    costs: float
    net_pnl: float
    return_pct: float
    holding_days: int
    exit_reason: str
    strategy: str

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "return_pct": self.return_pct,
            "net_pnl": self.net_pnl,
            "holding_days": self.holding_days,
            "exit_reason": self.exit_reason,
        }


@dataclass
class _WindowResult:
    """What one test window produced. A dataclass rather than a dict so the
    accumulation arithmetic below is type-checked rather than hoped for."""

    final_capital: float
    equity: list[tuple[pd.Timestamp, float]]
    trades: list[BacktestTrade]
    costs: float
    gross_profit: float
    turnover: float
    invested_days: int
    total_days: int
    skipped: int
    unaffordable: int


@dataclass
class WalkForwardWindow:
    """One train/test split."""

    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    @property
    def label(self) -> str:
        return f"{self.test_start}..{self.test_end}"


@dataclass
class BacktestResult:
    """Everything a run produced."""

    strategy: str
    equity: pd.Series
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: PerformanceMetrics | None = None
    windows: list[WalkForwardWindow] = field(default_factory=list)
    #: Caveats about the run itself, distinct from metric reliability.
    caveats: list[str] = field(default_factory=list)
    skipped_signals: int = 0
    rejected_unaffordable: int = 0

    @property
    def num_trades(self) -> int:
        return len(self.trades)


class WalkForwardBacktester:
    """Runs a strategy over consecutive out-of-sample windows."""

    def __init__(
        self,
        settings: Settings,
        cost_model: CostModel | None = None,
        *,
        calendar: TradingCalendar | None = None,
        regime_detector: RegimeDetector | None = None,
        initial_capital: float | None = None,
    ) -> None:
        self.settings = settings
        self.costs = cost_model or CostModel(settings.costs)
        self.calendar = calendar or TradingCalendar()
        self.regime_detector = regime_detector
        self.initial_capital = float(
            initial_capital
            if initial_capital is not None
            else settings.capital.allocation_usd
        )

    # ------------------------------------------------------------- windowing

    def make_windows(
        self,
        start: dt.date,
        end: dt.date,
        *,
        train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 10,
    ) -> list[WalkForwardWindow]:
        """Split ``[start, end]`` into rolling train/test windows.

        ``gap_days`` separates train from test so that forward-looking labels
        in the training data cannot overlap the test period.
        """
        windows: list[WalkForwardWindow] = []
        train_days = int(train_months * 30.44)
        test_days = int(test_months * 30.44)

        train_start = start
        while True:
            train_end = train_start + dt.timedelta(days=train_days)
            test_start = train_end + dt.timedelta(days=gap_days)
            test_end = min(test_start + dt.timedelta(days=test_days), end)

            if test_start >= end or test_end <= test_start:
                break

            windows.append(
                WalkForwardWindow(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            train_start = train_start + dt.timedelta(days=test_days)

        return windows

    # ------------------------------------------------------------------- run

    def run(
        self,
        strategy: Strategy,
        bars: dict[str, pd.DataFrame],
        *,
        start: dt.date,
        end: dt.date,
        train_months: int = 12,
        test_months: int = 3,
        gap_days: int = 10,
        regimes: dict[dt.date, RegimeState] | None = None,
    ) -> BacktestResult:
        """Run ``strategy`` walk-forward over ``bars``."""
        result = BacktestResult(strategy=strategy.name, equity=pd.Series(dtype=float))

        usable = {s: f for s, f in bars.items() if not f.empty}
        if not usable:
            result.caveats.append("no price data supplied; nothing to test")
            return result

        # Guard against the calendar not covering the requested period. Without
        # this the window loop finds no sessions, produces no equity points,
        # and reports a truncated result that looks like a real one.
        covered_start, covered_end = self.calendar.covers
        if start < covered_start or end > covered_end:
            raise ValueError(
                f"the trading calendar covers {covered_start}..{covered_end}, but the "
                f"backtest asks for {start}..{end}. Construct TradingCalendar with "
                f"start=/end= spanning the backtest period."
            )

        features = {symbol: build_features(frame, symbol) for symbol, frame in usable.items()}
        windows = self.make_windows(
            start, end, train_months=train_months, test_months=test_months, gap_days=gap_days
        )
        result.windows = windows

        if not windows:
            span_days = (end - start).days
            result.caveats.append(
                f"the sample is too short to walk forward: {span_days} days available, but a "
                f"{train_months}-month training window plus a {test_months}-month test window "
                f"needs at least {int((train_months + test_months) * 30.44) + gap_days}. "
                f"No out-of-sample evaluation is possible."
            )
            return result

        equity_points: list[tuple[pd.Timestamp, float]] = []
        capital = self.initial_capital
        all_trades: list[BacktestTrade] = []
        invested_days = 0
        total_days = 0
        total_costs = 0.0
        gross_profit = 0.0
        turnover_notional = 0.0

        for window in windows:
            if hasattr(strategy, "fit"):
                # Fit on the training window only. The gap before test_start
                # keeps forward-looking labels out of the test period.
                try:
                    strategy.fit(features, pd.Timestamp(window.train_end, tz="UTC"))
                except Exception as exc:  # noqa: BLE001 - see below
                    # A strategy that cannot fit produces no signals for this
                    # window, which is a legitimate outcome and is reported.
                    # Aborting the whole comparison because one strategy broke
                    # would lose the results for the others.
                    log.error(
                        "backtest.fit_failed",
                        strategy=strategy.name,
                        window=window.label,
                        error=str(exc),
                    )
                    result.caveats.append(
                        f"training failed for window {window.label}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

            try:
                segment = self._run_window(
                    strategy, features, usable, window, capital, regimes
                )
            except Exception as exc:  # noqa: BLE001 - isolate a broken strategy
                log.error(
                    "backtest.window_failed",
                    strategy=strategy.name,
                    window=window.label,
                    error=str(exc),
                )
                result.caveats.append(
                    f"window {window.label} failed: {type(exc).__name__}: {exc}"
                )
                continue
            capital = segment.final_capital
            equity_points.extend(segment.equity)
            all_trades.extend(segment.trades)
            invested_days += segment.invested_days
            total_days += segment.total_days
            total_costs += segment.costs
            gross_profit += segment.gross_profit
            turnover_notional += segment.turnover
            result.skipped_signals += segment.skipped
            result.rejected_unaffordable += segment.unaffordable

        if not equity_points:
            result.caveats.append("no tradable days inside the out-of-sample windows")
            return result

        equity = pd.Series(
            [value for _, value in equity_points],
            index=pd.DatetimeIndex([ts for ts, _ in equity_points]),
        ).sort_index()
        equity = equity[~equity.index.duplicated(keep="last")]

        result.equity = equity
        result.trades = all_trades
        result.metrics = evaluate(
            strategy.name,
            equity,
            [t.as_dict() for t in all_trades],
            exposure=invested_days / total_days if total_days else 0.0,
            turnover=turnover_notional / self.initial_capital if self.initial_capital else 0.0,
            total_costs=total_costs,
            gross_profit=gross_profit,
        )

        if result.rejected_unaffordable:
            result.caveats.append(
                f"{result.rejected_unaffordable} signal(s) could not be sized: the position "
                f"the risk budget allowed was smaller than one whole share. At "
                f"${self.initial_capital:.0f} this is the binding constraint, not the strategy."
            )
        return result

    # --------------------------------------------------------- one test window

    def _run_window(
        self,
        strategy: Strategy,
        features: dict[str, FeatureSet],
        bars: dict[str, pd.DataFrame],
        window: WalkForwardWindow,
        starting_capital: float,
        regimes: dict[dt.date, RegimeState] | None,
    ) -> _WindowResult:
        capital = starting_capital
        positions: dict[str, BacktestPosition] = {}
        trades: list[BacktestTrade] = []
        equity: list[tuple[pd.Timestamp, float]] = []
        costs_paid = 0.0
        gross = 0.0
        turnover = 0.0
        invested_days = 0
        skipped = 0
        unaffordable = 0

        sessions = [
            day
            for day in self.calendar.sessions
            if window.test_start <= day <= window.test_end
        ]

        for index, day in enumerate(sessions):
            prices = self._prices_on(bars, day)
            if not prices:
                continue

            # --- mark to market -------------------------------------------
            holdings_value = sum(
                position.market_value(prices.get(symbol, position.entry_price))
                for symbol, position in positions.items()
            )
            equity.append((pd.Timestamp(day, tz="UTC"), capital + holdings_value))
            if positions:
                invested_days += 1

            # --- stops and targets, checked against the day's range --------
            for symbol in list(positions):
                exit_info = self._check_exit(positions[symbol], bars, symbol, day)
                if exit_info is not None:
                    price, reason = exit_info
                    if not self._min_hold_satisfied(positions[symbol], day):
                        continue
                    capital, trade, cost = self._close(
                        positions.pop(symbol), price, day, reason, capital, strategy.name
                    )
                    trades.append(trade)
                    costs_paid += cost
                    gross += trade.gross_pnl
                    turnover += abs(trade.exit_price * trade.quantity)

            # --- decide, for execution on the NEXT session -----------------
            if index + 1 >= len(sessions):
                continue
            next_day = sessions[index + 1]

            context = StrategyContext(
                as_of=day,
                features=features,
                regime=(regimes or {}).get(day),
                current_holdings=tuple(positions),
                holding_days={
                    s: self.calendar.trading_days_between(p.entry_date, day)
                    for s, p in positions.items()
                },
            )
            decision = strategy.decide(context)
            targets = decision.target_weights

            # --- exit anything no longer wanted ---------------------------
            for symbol in list(positions):
                if symbol in targets:
                    continue
                if not self._min_hold_satisfied(positions[symbol], next_day):
                    skipped += 1
                    continue
                open_price = self._open_on(bars, symbol, next_day)
                if open_price is None:
                    continue
                capital, trade, cost = self._close(
                    positions.pop(symbol), open_price, next_day, "signal", capital, strategy.name
                )
                trades.append(trade)
                costs_paid += cost
                gross += trade.gross_pnl
                turnover += abs(trade.exit_price * trade.quantity)

            # --- enter new positions at the next open ---------------------
            for symbol, weight in targets.items():
                if symbol in positions:
                    continue
                open_price = self._open_on(bars, symbol, next_day)
                if open_price is None:
                    skipped += 1
                    continue

                sized = self._size(capital + 0.0, weight, open_price)
                if sized <= 0:
                    unaffordable += 1
                    continue

                fill = self.costs.fill_price(reference_price=open_price, side=Side.BUY)
                cost = self.costs.cost(price=fill, quantity=sized, side=Side.BUY).total
                outlay = fill * sized + cost
                if outlay > capital:
                    unaffordable += 1
                    continue

                capital -= outlay
                costs_paid += cost
                turnover += fill * sized

                signal = next((s for s in decision.signals if s.symbol == symbol), None)
                atr_pct = (signal.metadata.get("atr_pct") if signal else None) or 0.02
                stop = (
                    fill * (1 - atr_pct * signal.stop_atr_mult)
                    if signal and signal.stop_atr_mult
                    else None
                )
                target = (
                    fill * (1 + atr_pct * signal.take_profit_atr_mult)
                    if signal and signal.take_profit_atr_mult
                    else None
                )

                positions[symbol] = BacktestPosition(
                    symbol=symbol,
                    quantity=sized,
                    entry_price=fill,
                    entry_date=next_day,
                    entry_cost=cost,
                    stop_price=stop,
                    take_profit_price=target,
                )

        # --- close out at the end of the window ---------------------------
        if sessions:
            final_day = sessions[-1]
            for symbol in list(positions):
                final_price = self._prices_on(bars, final_day).get(symbol)
                if final_price is None:
                    continue
                price = final_price
                capital, trade, cost = self._close(
                    positions.pop(symbol), price, final_day, "window_end", capital, strategy.name
                )
                trades.append(trade)
                costs_paid += cost
                gross += trade.gross_pnl

        return _WindowResult(
            final_capital=capital,
            equity=equity,
            trades=trades,
            costs=costs_paid,
            gross_profit=gross,
            turnover=turnover,
            invested_days=invested_days,
            total_days=len(sessions),
            skipped=skipped,
            unaffordable=unaffordable,
        )

    # -------------------------------------------------------------- helpers

    def _size(self, capital: float, weight: float, price: float) -> float:
        """Whole-share position size, honouring the cash buffer.

        Whole shares only. The fractional path exists in live execution but is
        off by default, and a backtest that assumed fractional sizing would
        overstate what this account can actually do.
        """
        buffer = float(self.settings.capital.cash_buffer_pct)
        budget = min(capital * (1 - buffer), self.initial_capital * weight)
        if price <= 0:
            return 0.0
        return float(np.floor(budget / price))

    def _min_hold_satisfied(self, position: BacktestPosition, day: dt.date) -> bool:
        held = self.calendar.trading_days_between(position.entry_date, day)
        return held >= self.settings.holding.min_holding_days

    def _check_exit(
        self, position: BacktestPosition, bars: dict[str, pd.DataFrame], symbol: str, day: dt.date
    ) -> tuple[float, str] | None:
        """Whether a stop or target was hit, using the bar's high and low.

        When both were touched on the same bar the stop is assumed to have
        triggered first. Daily bars cannot say which came first, and assuming
        the favourable one is how backtests flatter themselves.
        """
        row = self._row_on(bars, symbol, day)
        if row is None:
            return None

        if position.stop_price is not None and row["low"] <= position.stop_price:
            return position.stop_price, "stop_loss"
        if position.take_profit_price is not None and row["high"] >= position.take_profit_price:
            return position.take_profit_price, "take_profit"
        return None

    def _close(
        self,
        position: BacktestPosition,
        price: float,
        day: dt.date,
        reason: str,
        capital: float,
        strategy_name: str,
    ) -> tuple[float, BacktestTrade, float]:
        fill = self.costs.fill_price(reference_price=price, side=Side.SELL)
        cost = self.costs.cost(price=fill, quantity=position.quantity, side=Side.SELL).total
        proceeds = fill * position.quantity - cost

        gross = (fill - position.entry_price) * position.quantity
        total_costs = cost + position.entry_cost
        net = gross - total_costs
        basis = position.entry_price * position.quantity

        trade = BacktestTrade(
            symbol=position.symbol,
            entry_date=position.entry_date,
            exit_date=day,
            entry_price=position.entry_price,
            exit_price=fill,
            quantity=position.quantity,
            gross_pnl=gross,
            costs=total_costs,
            net_pnl=net,
            return_pct=net / basis if basis else 0.0,
            holding_days=self.calendar.trading_days_between(position.entry_date, day),
            exit_reason=reason,
            strategy=strategy_name,
        )
        # NOTE: proceeds are credited immediately here. Real T+1 settlement is
        # enforced by the risk manager in Phase 5; modelling it inside the
        # backtest would additionally suppress trades, so the live system will
        # if anything trade *less* than this simulation.
        return capital + proceeds, trade, cost

    @staticmethod
    def _row_on(bars: dict[str, pd.DataFrame], symbol: str, day: dt.date) -> pd.Series | None:
        frame = bars.get(symbol)
        if frame is None or frame.empty:
            return None
        stamp = pd.Timestamp(day, tz="UTC")
        if stamp not in frame.index:
            return None
        row = frame.loc[stamp]
        return row if isinstance(row, pd.Series) else None

    def _open_on(self, bars: dict[str, pd.DataFrame], symbol: str, day: dt.date) -> float | None:
        row = self._row_on(bars, symbol, day)
        return float(row["open"]) if row is not None else None

    def _prices_on(self, bars: dict[str, pd.DataFrame], day: dt.date) -> dict[str, float]:
        out: dict[str, float] = {}
        for symbol in bars:
            row = self._row_on(bars, symbol, day)
            if row is not None:
                out[symbol] = float(row["close"])
        return out


def buy_and_hold(
    bars: pd.DataFrame, initial_capital: float, cost_model: CostModel, name: str
) -> BacktestResult:
    """Buy once at the first open, hold to the end. The benchmark to beat.

    Pays the same entry and exit costs as any other strategy, so the
    comparison is like for like.
    """
    result = BacktestResult(strategy=name, equity=pd.Series(dtype=float))
    if bars.empty:
        result.caveats.append("no data for the benchmark")
        return result

    entry_price = cost_model.fill_price(
        reference_price=float(bars["open"].iloc[0]), side=Side.BUY
    )
    quantity = float(np.floor(initial_capital / entry_price))
    if quantity <= 0:
        result.caveats.append(
            f"{name}: one share costs ${entry_price:.2f}, more than the "
            f"${initial_capital:.0f} allocated -- buy and hold is not possible"
        )
        return result

    entry_cost = cost_model.cost(price=entry_price, quantity=quantity, side=Side.BUY).total
    cash = initial_capital - entry_price * quantity - entry_cost

    result.equity = bars["close"] * quantity + cash
    result.metrics = evaluate(
        name, result.equity, [], exposure=1.0, total_costs=entry_cost,
        gross_profit=float(result.equity.iloc[-1] - initial_capital),
    )
    return result
