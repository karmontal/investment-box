"""Risk manager, sizing and the settled-cash ledger.

The property this file exists to prove: **the engine cannot spend unsettled
cash.** At ~$500 that is the binding constraint on the whole system, and a
good-faith violation is a real consequence with a real penalty.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from investment_box.config.loader import load_settings
from investment_box.core.clock import FrozenClock, TradingCalendar
from investment_box.core.errors import SettlementError
from investment_box.core.types import CashLedger
from investment_box.execution.base import AccountSnapshot, BrokerPosition
from investment_box.risk import (
    PositionSizer,
    RiskManager,
    RiskVerdict,
    SettlementLedger,
)

WEDNESDAY = dt.date(2024, 6, 12)


@pytest.fixture
def ledger(database, calendar: TradingCalendar, clock: FrozenClock) -> SettlementLedger:
    return SettlementLedger(database, calendar, settlement_days=1, clock=clock)


@pytest.fixture
def manager(settings, database, ledger, audit, clock, calendar) -> RiskManager:
    return RiskManager(settings, database, ledger, audit, clock=clock, calendar=calendar)


def account(
    equity: str = "500.00", settled: str = "500.00", unsettled: str = "0.00"
) -> AccountSnapshot:
    cash = CashLedger(settled=Decimal(settled), unsettled=Decimal(unsettled))
    return AccountSnapshot(
        equity=Decimal(equity),
        cash=cash,
        positions_value=Decimal("0.00"),
        buying_power=cash.available_for_trading,
        is_cash_account=True,
    )


def position(symbol: str, qty: str = "1", price: str = "60.00") -> BrokerPosition:
    return BrokerPosition(
        symbol=symbol,  # type: ignore[arg-type]
        quantity=Decimal(qty),
        avg_entry_price=Decimal(price),
        current_price=Decimal(price),
    )


class TestSettlementLedger:
    def test_sale_proceeds_start_unsettled(self, ledger: SettlementLedger) -> None:
        ledger.record_sale("SPUS", Decimal("120.00"), sold_on=WEDNESDAY)
        snapshot = ledger.snapshot(Decimal("500.00"), as_of=WEDNESDAY)
        assert snapshot.unsettled == Decimal("120.00")
        assert snapshot.available == Decimal("380.00")

    def test_proceeds_settle_on_the_next_trading_day(self, ledger: SettlementLedger) -> None:
        ledger.record_sale("SPUS", Decimal("120.00"), sold_on=WEDNESDAY)
        snapshot = ledger.snapshot(Decimal("500.00"), as_of=dt.date(2024, 6, 13))
        assert snapshot.unsettled == Decimal("0.00")
        assert snapshot.available == Decimal("500.00")

    def test_friday_sale_does_not_settle_over_the_weekend(
        self, ledger: SettlementLedger
    ) -> None:
        friday = dt.date(2024, 6, 14)
        ledger.record_sale("SPUS", Decimal("100.00"), sold_on=friday)
        assert ledger.snapshot(Decimal("500"), as_of=dt.date(2024, 6, 16)).unsettled == (
            Decimal("100.00")
        )
        assert ledger.snapshot(Decimal("500"), as_of=dt.date(2024, 6, 17)).unsettled == (
            Decimal("0.00")
        )

    def test_settlement_skips_a_holiday(self, database, clock: FrozenClock) -> None:
        calendar = TradingCalendar(start=dt.date(2024, 6, 1), end=dt.date(2024, 8, 1))
        ledger = SettlementLedger(database, calendar, 1, clock=clock)
        # Sold Wed Jul 3; Jul 4 is a holiday, so it settles Fri Jul 5.
        ledger.record_sale("SPUS", Decimal("100"), sold_on=dt.date(2024, 7, 3))
        assert ledger.snapshot(Decimal("500"), as_of=dt.date(2024, 7, 4)).unsettled == (
            Decimal("100.00")
        )
        assert ledger.snapshot(Decimal("500"), as_of=dt.date(2024, 7, 5)).unsettled == (
            Decimal("0.00")
        )

    def test_spending_unsettled_cash_raises(self, ledger: SettlementLedger) -> None:
        """The good-faith violation guard, as an exception rather than a bool."""
        ledger.record_sale("SPUS", Decimal("450.00"), sold_on=WEDNESDAY)
        snapshot = ledger.snapshot(Decimal("500.00"), as_of=WEDNESDAY)
        with pytest.raises(SettlementError, match="good-faith violation"):
            ledger.assert_affordable(Decimal("200.00"), snapshot)

    def test_affordable_within_settled_cash(self, ledger: SettlementLedger) -> None:
        snapshot = ledger.snapshot(Decimal("500.00"), as_of=WEDNESDAY)
        ledger.assert_affordable(Decimal("100.00"), snapshot)  # must not raise

    def test_ledger_survives_a_restart(
        self, database, calendar: TradingCalendar, clock: FrozenClock
    ) -> None:
        """An in-memory ledger would reset to 'all settled' after a crash."""
        first = SettlementLedger(database, calendar, 1, clock=clock)
        first.record_sale("SPUS", Decimal("200.00"), sold_on=WEDNESDAY)

        second = SettlementLedger(database, calendar, 1, clock=clock)
        assert second.snapshot(Decimal("500"), as_of=WEDNESDAY).unsettled == Decimal("200.00")

    def test_explain_names_the_next_settlement(self, ledger: SettlementLedger) -> None:
        ledger.record_sale("SPUS", Decimal("100.00"), sold_on=WEDNESDAY)
        text = ledger.snapshot(Decimal("500"), as_of=WEDNESDAY).explain()
        assert "unsettled" in text
        assert "2024-06-13" in text


class TestSizing:
    def test_risk_budget_is_a_fraction_of_allocation(self) -> None:
        sizer = PositionSizer(load_settings())
        assert sizer.risk_budget == Decimal("7.50")  # 1.5% of $500

    def test_size_respects_the_risk_budget(self) -> None:
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="SPUS", entry_price=Decimal("60"), available_cash=Decimal("450"),
            atr=Decimal("1.50"), stop_atr_mult=2.0,
        )
        assert result.is_tradable
        assert result.risk_amount <= sizer.risk_budget

    def test_size_never_rounds_up(self) -> None:
        """Rounding up spends more than the budget allows."""
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="SPUS", entry_price=Decimal("60"), available_cash=Decimal("450"),
            atr=Decimal("1.50"),
        )
        assert result.quantity == result.quantity.to_integral_value()
        assert result.notional <= Decimal("450")

    def test_unaffordable_share_is_refused_with_a_reason(self) -> None:
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="EXPENSIVE", entry_price=Decimal("5000"),
            available_cash=Decimal("450"), atr=Decimal("50"),
        )
        assert not result.is_tradable
        assert "one whole share costs" in result.reason

    def test_tiny_position_is_refused(self) -> None:
        """Below ~$20 the friction and rounding dominate any edge."""
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="CHEAP", entry_price=Decimal("3"), available_cash=Decimal("10"),
            atr=Decimal("0.05"),
        )
        assert not result.is_tradable

    def test_position_cap_binds_before_cash(self) -> None:
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="SPUS", entry_price=Decimal("20"), available_cash=Decimal("450"),
            atr=Decimal("0.10"),
        )
        assert result.notional <= sizer.max_position_notional
        assert "maximum position size" in result.reason

    def test_missing_atr_falls_back_rather_than_refusing(self) -> None:
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="SPUS", entry_price=Decimal("60"), available_cash=Decimal("450"), atr=None,
        )
        assert result.is_tradable
        assert result.stop_price is not None

    def test_fractional_disabled_by_default(self) -> None:
        sizer = PositionSizer(load_settings())
        result = sizer.size(
            symbol="SPUS", entry_price=Decimal("60"), available_cash=Decimal("450"),
            atr=Decimal("1.50"),
        )
        assert not result.is_fractional


class TestRiskLimits:
    def test_approves_a_normal_trade(self, manager: RiskManager) -> None:
        decision = manager.evaluate(
            symbol="SPUS", entry_price=Decimal("60"), account=account(),
            positions=[], atr=Decimal("1.50"),
        )
        assert decision.is_approved

    def test_refuses_when_paused(self, manager: RiskManager) -> None:
        manager.pause("testing")
        decision = manager.evaluate(
            symbol="SPUS", entry_price=Decimal("60"), account=account(), positions=[],
        )
        assert decision.verdict is RiskVerdict.PAUSED

    def test_refuses_a_symbol_already_held(self, manager: RiskManager) -> None:
        decision = manager.evaluate(
            symbol="SPUS", entry_price=Decimal("60"), account=account(),
            positions=[position("SPUS")],
        )
        assert decision.verdict is RiskVerdict.ALREADY_HELD

    def test_refuses_at_the_position_cap(self, manager: RiskManager) -> None:
        held = [position(f"S{i}") for i in range(5)]
        decision = manager.evaluate(
            symbol="SPUS", entry_price=Decimal("60"), account=account(), positions=held,
        )
        assert decision.verdict is RiskVerdict.MAX_POSITIONS

    def test_refuses_when_unsettled_cash_would_be_needed(
        self, manager: RiskManager, ledger: SettlementLedger
    ) -> None:
        """The central guarantee of this phase."""
        ledger.record_sale("HLAL", Decimal("480.00"), sold_on=WEDNESDAY)
        decision = manager.evaluate(
            symbol="SPUS", entry_price=Decimal("60"), account=account(),
            positions=[], atr=Decimal("1.50"),
        )
        assert decision.verdict in (
            RiskVerdict.UNSIZABLE, RiskVerdict.INSUFFICIENT_SETTLED_CASH
        )
        assert not decision.is_approved

    def test_drawdown_breach_pauses_the_engine(
        self, manager: RiskManager, database
    ) -> None:
        from investment_box.db.models import EquitySnapshot

        with database.session() as session:
            session.add(
                EquitySnapshot(
                    snapshot_date=dt.date(2024, 6, 10), equity=Decimal("500"),
                    cash_settled=Decimal("500"), cash_unsettled=Decimal("0"),
                    positions_value=Decimal("0"), trading_mode="paper",
                )
            )
        state = manager.state(account(equity="400.00"), [])
        assert state.is_paused
        assert manager.is_paused
        assert "drawdown" in state.pause_reason

    def test_exits_are_never_blocked_by_a_pause(self, manager: RiskManager) -> None:
        """Being unable to close a losing position would be the worst outcome."""
        manager.pause("drawdown")
        allowed, reason = manager.can_exit("SPUS", dt.date(2024, 6, 5))
        assert allowed, reason

    def test_minimum_hold_is_enforced(self, manager: RiskManager) -> None:
        allowed, reason = manager.can_exit("SPUS", dt.date(2024, 6, 12))
        assert not allowed
        assert "earliest exit" in reason

    def test_minimum_hold_counts_trading_days(self, manager: RiskManager) -> None:
        # Entered Friday, evaluated the following Wednesday: 3 trading days.
        allowed, reason = manager.can_exit("SPUS", dt.date(2024, 6, 7))
        assert allowed
        assert "3 trading days" in reason

    def test_resume_clears_the_pause(self, manager: RiskManager) -> None:
        manager.pause("testing")
        manager.resume()
        assert not manager.is_paused


class TestRiskState:
    def test_summary_reports_usage(self, manager: RiskManager) -> None:
        state = manager.state(account(), [position("SPUS")])
        assert "1/5 positions" in state.summary()

    def test_paused_summary_says_why(self, manager: RiskManager) -> None:
        manager.pause("drawdown breach")
        assert "PAUSED" in manager.state(account(), []).summary()

    def test_settlement_is_attached(self, manager: RiskManager) -> None:
        assert manager.state(account(), []).settlement is not None
