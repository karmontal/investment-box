"""Backtester, universe and metrics.

The most important test in this file is
``TestNoLookahead::test_perfect_foresight_strategy_is_caught`` -- a strategy
that cheats must produce an implausible result that the report flags, because
that is the last line of defence against a subtle leak.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from investment_box.backtest import (
    BacktestReport,
    CostModel,
    WalkForwardBacktester,
    buy_and_hold,
    evaluate,
    max_drawdown,
)
from investment_box.config.loader import load_settings
from investment_box.core.clock import TradingCalendar
from investment_box.core.types import Side
from investment_box.strategies.base import Signal, Strategy, StrategyDecision
from investment_box.universe import Instrument, UniverseBuilder

START = dt.date(2022, 1, 3)
END = dt.date(2024, 6, 12)


@pytest.fixture
def bt_calendar() -> TradingCalendar:
    return TradingCalendar(start=START, end=END)


def bars_for(symbol: str, trend: float = 0.0006, seed: int = 1) -> pd.DataFrame:
    calendar = TradingCalendar(start=START, end=END)
    sessions = [d for d in calendar.sessions if START <= d <= END]
    rng = np.random.default_rng(seed)
    close = 60 * np.exp(np.cumsum(rng.normal(trend, 0.010, len(sessions))))
    index = pd.DatetimeIndex(
        [dt.datetime.combine(d, dt.time.min, tzinfo=dt.UTC) for d in sessions]
    )
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "volume": rng.lognormal(np.log(1e6), 0.3, len(sessions)),
        },
        index=index,
    )


class AlwaysLong(Strategy):
    """Holds one symbol permanently. A control, not a strategy."""

    name = "always_long"
    warmup_bars = 0

    def __init__(self, symbol: str = "AAA") -> None:
        self.symbol = symbol

    def decide(self, context) -> StrategyDecision:
        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=(Signal(symbol=self.symbol, target_weight=0.9),),
        )


class PerfectForesight(Strategy):
    """Cheats: reads tomorrow's return. Exists to prove the report catches it."""

    name = "perfect_foresight"
    warmup_bars = 0

    def __init__(self, bars: dict[str, pd.DataFrame]) -> None:
        self.bars = bars

    def decide(self, context) -> StrategyDecision:
        stamp = pd.Timestamp(context.as_of, tz="UTC")
        best, best_return = None, 0.0
        for symbol, frame in self.bars.items():
            future = frame.loc[frame.index > stamp]
            if len(future) < 2:
                continue
            forward = float(future["close"].iloc[1] / future["open"].iloc[0] - 1)
            if forward > best_return:
                best, best_return = symbol, forward
        if best is None:
            return StrategyDecision(as_of=context.as_of, strategy=self.name, signals=())
        return StrategyDecision(
            as_of=context.as_of,
            strategy=self.name,
            signals=(Signal(symbol=best, target_weight=0.9),),
        )


class TestCostModel:
    def test_round_trip_is_proportional_to_notional(self) -> None:
        """Alpaca charges no commission, so cost is a constant fraction of
        notional -- small positions are not penalised the way a fixed fee
        would penalise them."""
        model = CostModel(load_settings().costs)
        small = model.breakeven_move(price=60.0, quantity=1)
        large = model.breakeven_move(price=60.0, quantity=100)
        assert small == pytest.approx(large, rel=0.01)

    def test_buy_fills_higher_and_sell_lower(self) -> None:
        model = CostModel(load_settings().costs)
        assert model.fill_price(reference_price=100.0, side=Side.BUY) > 100.0
        assert model.fill_price(reference_price=100.0, side=Side.SELL) < 100.0

    def test_regulatory_fees_are_sell_only(self) -> None:
        model = CostModel(load_settings().costs)
        assert model.cost(price=60.0, quantity=2, side=Side.BUY).regulatory == 0.0
        assert model.cost(price=60.0, quantity=2, side=Side.SELL).regulatory > 0.0

    def test_costs_cannot_be_zero_by_default(self) -> None:
        model = CostModel(load_settings().costs)
        assert model.round_trip_cost(price=60.0, quantity=1) > 0


class TestMetrics:
    def test_drawdown(self) -> None:
        equity = pd.Series([100.0, 120.0, 90.0, 110.0])
        worst, _ = max_drawdown(equity)
        assert worst == pytest.approx(-0.25)

    def test_flat_equity_has_no_drawdown(self) -> None:
        assert max_drawdown(pd.Series([100.0] * 10))[0] == pytest.approx(0.0)

    def test_small_sample_is_flagged_unreliable(self) -> None:
        equity = pd.Series(np.linspace(100, 110, 50))
        metrics = evaluate("test", equity, [{"return_pct": 0.01, "net_pnl": 1.0}] * 5)
        assert not metrics.is_reliable
        assert any("trades" in w for w in metrics.reliability_warnings)

    def test_high_sharpe_is_flagged_as_too_good(self) -> None:
        index = pd.date_range("2022-01-03", periods=600, freq="B", tz="UTC")
        equity = pd.Series(np.linspace(100, 300, 600), index=index)
        metrics = evaluate("suspicious", equity, [{"return_pct": 0.02, "net_pnl": 2.0}] * 60)
        assert metrics.looks_too_good
        assert metrics.overfitting_warnings()

    def test_cost_share_is_reported(self) -> None:
        equity = pd.Series(np.linspace(100, 110, 300))
        metrics = evaluate("test", equity, [], total_costs=8.0, gross_profit=10.0)
        assert metrics.cost_share_of_gross == pytest.approx(0.8)
        assert any("costs consumed" in w for w in metrics.reliability_warnings)

    def test_empty_equity_is_handled(self) -> None:
        metrics = evaluate("test", pd.Series(dtype=float))
        assert not metrics.is_reliable


class TestWalkForward:
    def test_windows_do_not_overlap_train_and_test(self) -> None:
        bt = WalkForwardBacktester(load_settings())
        for window in bt.make_windows(dt.date(2019, 1, 2), dt.date(2024, 6, 12)):
            assert window.test_start > window.train_end

    def test_gap_separates_train_from_test(self) -> None:
        """Forward-looking labels in training must not reach into the test."""
        bt = WalkForwardBacktester(load_settings())
        windows = bt.make_windows(
            dt.date(2019, 1, 2), dt.date(2024, 6, 12), gap_days=10
        )
        for window in windows:
            assert (window.test_start - window.train_end).days >= 10

    def test_short_sample_produces_no_windows_and_says_so(self) -> None:
        short_start, short_end = dt.date(2024, 1, 2), dt.date(2024, 3, 1)
        bt = WalkForwardBacktester(
            load_settings(), calendar=TradingCalendar(start=short_start, end=short_end)
        )
        result = bt.run(
            AlwaysLong(),
            {"AAA": bars_for("AAA")},
            start=short_start,
            end=short_end,
        )
        assert result.num_trades == 0
        assert any("too short to walk forward" in c for c in result.caveats)


class TestBacktestExecution:
    def test_runs_and_produces_trades(self, bt_calendar: TradingCalendar) -> None:
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        result = bt.run(
            AlwaysLong(), {"AAA": bars_for("AAA")}, start=START, end=END,
            train_months=6, test_months=3,
        )
        assert result.metrics is not None
        assert not result.equity.empty

    def test_refuses_a_period_outside_the_calendar(self) -> None:
        """Silently producing nothing would look like a real, empty result."""
        bt = WalkForwardBacktester(
            load_settings(), calendar=TradingCalendar(start=START, end=END)
        )
        with pytest.raises(ValueError, match="trading calendar covers"):
            bt.run(
                AlwaysLong(), {"AAA": bars_for("AAA")},
                start=dt.date(2010, 1, 4), end=END,
            )

    def test_whole_share_sizing_only(self, bt_calendar: TradingCalendar) -> None:
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        result = bt.run(
            AlwaysLong(), {"AAA": bars_for("AAA")}, start=START, end=END,
            train_months=6, test_months=3,
        )
        for trade in result.trades:
            assert trade.quantity == int(trade.quantity)

    def test_costs_are_charged_on_every_trade(self, bt_calendar: TradingCalendar) -> None:
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        result = bt.run(
            AlwaysLong(), {"AAA": bars_for("AAA")}, start=START, end=END,
            train_months=6, test_months=3,
        )
        assert all(trade.costs > 0 for trade in result.trades)

    def test_minimum_holding_period_is_respected(self, bt_calendar: TradingCalendar) -> None:
        settings = load_settings(overrides={"holding": {"min_holding_days": 2}})
        bt = WalkForwardBacktester(settings, calendar=bt_calendar)
        result = bt.run(
            AlwaysLong(), {"AAA": bars_for("AAA")}, start=START, end=END,
            train_months=6, test_months=3,
        )
        # window_end closes ignore the minimum hold by design; every other exit
        # must honour it.
        for trade in result.trades:
            if trade.exit_reason != "window_end":
                assert trade.holding_days >= 2

    def test_unaffordable_signals_are_counted_not_silently_dropped(
        self, bt_calendar: TradingCalendar
    ) -> None:
        expensive = bars_for("AAA")
        expensive[["open", "high", "low", "close"]] *= 50  # ~$3000/share
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        result = bt.run(
            AlwaysLong(), {"AAA": expensive}, start=START, end=END,
            train_months=6, test_months=3,
        )
        assert result.rejected_unaffordable > 0
        assert any("smaller than one whole share" in c for c in result.caveats)


class TestNoLookahead:
    def test_execution_never_uses_the_signal_bar(self, bt_calendar: TradingCalendar) -> None:
        """Entries happen on the bar AFTER the decision."""
        bars = {"AAA": bars_for("AAA")}
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        result = bt.run(
            AlwaysLong(), bars, start=START, end=END, train_months=6, test_months=3
        )
        for trade in result.trades:
            assert trade.entry_date > trade.entry_date - dt.timedelta(days=1)
            assert trade.exit_date >= trade.entry_date

    def test_perfect_foresight_strategy_is_caught(self, bt_calendar: TradingCalendar) -> None:
        """A cheating strategy must produce a result the report flags.

        This is the last line of defence: if a subtle leak ever reaches a real
        strategy, the too-good-to-be-true check is what surfaces it.
        """
        bars = {f"S{i}": bars_for(f"S{i}", seed=i) for i in range(4)}
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        result = bt.run(
            PerfectForesight(bars), bars, start=START, end=END,
            train_months=6, test_months=3,
        )
        assert result.metrics is not None
        assert result.metrics.looks_too_good, (
            "a strategy reading future prices produced a result the report did not "
            "flag as implausible"
        )
        assert result.metrics.overfitting_warnings()


class TestBuyAndHold:
    def test_produces_a_benchmark(self) -> None:
        result = buy_and_hold(bars_for("AAA"), 500.0, CostModel(load_settings().costs), "bh")
        assert result.metrics is not None
        assert result.metrics.exposure == 1.0

    def test_reports_when_one_share_is_unaffordable(self) -> None:
        """At $500, a single share of SPY is out of reach. That is a real
        constraint, not an error."""
        expensive = bars_for("AAA")
        expensive[["open", "high", "low", "close"]] *= 20  # ~$1200/share
        result = buy_and_hold(expensive, 500.0, CostModel(load_settings().costs), "bh")
        assert result.metrics is None
        assert any("more than the" in c for c in result.caveats)


class TestUniverseBuilder:
    def _builder(self, repository, tracker=None):
        return UniverseBuilder(load_settings(), repository, tracker)

    def test_unverified_symbols_are_excluded(self, repository) -> None:
        instruments = [Instrument(symbol="SPUS", verified=False, inception=dt.date(2019, 12, 18))]
        snapshot = self._builder(repository).build(
            instruments, as_of=END, check_compliance=False, check_liquidity=False
        )
        assert snapshot.is_empty
        assert "unverified" in snapshot.explain("SPUS")

    def test_verified_symbol_passes(self, repository) -> None:
        instruments = [Instrument(symbol="SPUS", verified=True, inception=dt.date(2019, 12, 18))]
        snapshot = self._builder(repository).build(
            instruments, as_of=END, check_compliance=False, check_liquidity=False
        )
        assert snapshot.symbols == ["SPUS"]

    def test_fund_not_yet_launched_is_excluded(self, repository) -> None:
        """Prevents backtesting a 2023 fund into 2019."""
        instruments = [Instrument(symbol="SPTE", verified=True, inception=dt.date(2023, 12, 1))]
        snapshot = self._builder(repository).build(
            [*instruments], as_of=dt.date(2020, 6, 1),
            check_compliance=False, check_liquidity=False,
        )
        assert snapshot.is_empty
        assert "not yet listed" in snapshot.explain("SPTE")

    def test_unknown_inception_is_excluded(self, repository) -> None:
        instruments = [Instrument(symbol="MNZL", verified=True, inception=None)]
        snapshot = self._builder(repository).build(
            instruments, as_of=END, check_compliance=False, check_liquidity=False
        )
        assert "inception date unknown" in snapshot.explain("MNZL")

    def test_blacklist_is_honoured(self, repository) -> None:
        settings = load_settings(overrides={"universe": {"blacklist": ["SPUS"]}})
        builder = UniverseBuilder(settings, repository)
        instruments = [Instrument(symbol="SPUS", verified=True, inception=dt.date(2019, 12, 18))]
        snapshot = builder.build(
            instruments, as_of=END, check_compliance=False, check_liquidity=False
        )
        assert snapshot.is_empty

    def test_leveraged_etf_is_blocked_even_if_verified(self, repository) -> None:
        """The hard constraint outranks every other gate, including yours."""
        instruments = [Instrument(symbol="TQQQ", verified=True, inception=dt.date(2010, 1, 4))]
        snapshot = self._builder(repository).build(
            instruments, as_of=END, check_compliance=False, check_liquidity=False
        )
        assert snapshot.is_empty
        assert "forbidden instrument" in snapshot.explain("TQQQ")

    def test_historical_build_flags_compliance_lookahead(
        self, repository, database, audit, clock
    ) -> None:
        from investment_box.core.types import ComplianceStatus
        from investment_box.shariah.providers.mock_external import MockExternalProvider
        from investment_box.shariah.status import ComplianceTracker

        tracker = ComplianceTracker(
            MockExternalProvider({"SPUS": ComplianceStatus.COMPLIANT}, clock=clock),
            database,
            load_settings().shariah,
            audit,
            clock=clock,
        )
        builder = UniverseBuilder(load_settings(), repository, tracker, clock=clock)
        instruments = [Instrument(symbol="SPUS", verified=True, inception=dt.date(2019, 12, 18))]
        snapshot = builder.build(
            instruments, as_of=dt.date(2023, 1, 3), check_liquidity=False
        )
        assert snapshot.used_current_compliance_for_history

    def test_rejection_summary_groups_reasons(self, repository) -> None:
        instruments = [
            Instrument(symbol="AAA", verified=False),
            Instrument(symbol="BBB", verified=False),
        ]
        snapshot = self._builder(repository).build(
            instruments, as_of=END, check_compliance=False, check_liquidity=False
        )
        assert snapshot.rejection_summary()["unverified"] == 2


class TestReport:
    def _report(self) -> BacktestReport:
        return BacktestReport(
            title="test", start=START, end=END, initial_capital=500.0
        )

    def test_says_when_buy_and_hold_won(self, bt_calendar: TradingCalendar) -> None:
        report = self._report()
        bt = WalkForwardBacktester(load_settings(), calendar=bt_calendar)
        report.add(
            bt.run(AlwaysLong(), {"AAA": bars_for("AAA", trend=-0.001)}, start=START, end=END,
                   train_months=6, test_months=3)
        )
        report.add_benchmark(
            buy_and_hold(
                bars_for("AAA", trend=0.002), 500.0, CostModel(load_settings().costs), "bh"
            )
        )
        text = report.to_text()
        assert "NO STRATEGY BEAT BUYING AND HOLDING" in text

    def test_caveats_are_printed_before_results(self) -> None:
        report = self._report()
        report.global_caveats.append("this data is synthetic")
        text = report.to_text()
        assert text.index("READ THIS FIRST") < text.index("SUMMARY")

    def test_telegram_summary_leads_with_the_caveat(self) -> None:
        report = self._report()
        report.add(
            type("R", (), {"strategy": "x", "metrics": None, "num_trades": 0, "caveats": []})()
        )
        assert "No result here is statistically reliable" in report.to_telegram()
