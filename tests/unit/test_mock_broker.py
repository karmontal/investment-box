"""Mock broker behaviour, focused on cash-account and settlement semantics."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from investment_box.core.clock import UTC, FrozenClock, TradingCalendar
from investment_box.core.types import OrderStatus, OrderType, Side, Symbol
from investment_box.execution.base import OrderRequest
from investment_box.execution.mock_broker import MockBroker


def buy(symbol: str, qty: str, key: str = "k1") -> OrderRequest:
    return OrderRequest(
        symbol=Symbol(symbol),
        side=Side.BUY,
        quantity=Decimal(qty),
        order_type=OrderType.MARKET,
        idempotency_key=key,
    )


def sell(symbol: str, qty: str, key: str = "k2") -> OrderRequest:
    return OrderRequest(
        symbol=Symbol(symbol),
        side=Side.SELL,
        quantity=Decimal(qty),
        order_type=OrderType.MARKET,
        idempotency_key=key,
    )


class TestAccount:
    def test_starts_with_all_cash_settled(self, broker: MockBroker) -> None:
        account = broker.get_account()
        assert account.equity == Decimal("500.00")
        assert account.cash.settled == Decimal("500.00")
        assert account.cash.unsettled == Decimal("0.00")

    def test_is_always_a_cash_account(self, broker: MockBroker) -> None:
        assert broker.get_account().is_cash_account

    def test_buying_power_is_settled_cash_not_equity(self, broker: MockBroker) -> None:
        broker.submit_order(buy("SPUS", "2"))
        account = broker.get_account()
        assert account.buying_power == account.cash.available_for_trading
        assert account.buying_power < account.equity


class TestBuying:
    def test_fill_creates_a_position_and_spends_cash(self, broker: MockBroker) -> None:
        result = broker.submit_order(buy("SPUS", "2"))
        assert result.status is OrderStatus.FILLED
        assert result.filled_avg_price == Decimal("45.00")

        position = broker.get_position("SPUS")
        assert position is not None
        assert position.quantity == Decimal("2")
        assert broker.get_account().cash.settled == Decimal("410.00")

    def test_buy_beyond_settled_cash_rejected(self, broker: MockBroker) -> None:
        result = broker.submit_order(buy("SPUS", "100"))
        assert result.status is OrderStatus.REJECTED
        assert "insufficient settled cash" in (result.rejection_reason or "")
        assert broker.get_position("SPUS") is None

    def test_adding_to_a_position_averages_the_basis(self, broker: MockBroker) -> None:
        broker.submit_order(buy("SPUS", "2", key="a"))
        broker.set_price("SPUS", Decimal("55.00"))
        broker.submit_order(buy("SPUS", "2", key="b"))
        position = broker.get_position("SPUS")
        assert position is not None
        assert position.quantity == Decimal("4")
        assert position.avg_entry_price == Decimal("50.00")  # (45+45+55+55)/4


class TestShortingIsImpossible:
    def test_selling_with_no_position_rejected(self, broker: MockBroker) -> None:
        result = broker.submit_order(sell("SPUS", "1"))
        assert result.status is OrderStatus.REJECTED
        assert "shorting is not permitted" in (result.rejection_reason or "")

    def test_overselling_a_position_rejected(self, broker: MockBroker) -> None:
        broker.submit_order(buy("SPUS", "2", key="a"))
        result = broker.submit_order(sell("SPUS", "5", key="b"))
        assert result.status is OrderStatus.REJECTED
        assert broker.get_position("SPUS") is not None


class TestSettlement:
    def test_sale_proceeds_start_unsettled(self, broker: MockBroker) -> None:
        broker.submit_order(buy("SPUS", "2", key="a"))
        broker.submit_order(sell("SPUS", "2", key="b"))
        account = broker.get_account()
        assert account.cash.unsettled == Decimal("90.00")
        assert account.cash.settled == Decimal("410.00")

    def test_unsettled_proceeds_cannot_fund_a_buy(self, broker: MockBroker) -> None:
        """The good-faith-violation path. This is the point of the whole ledger."""
        broker.submit_order(buy("SPUS", "10", key="a"))  # $450 of $500
        broker.submit_order(sell("SPUS", "10", key="b"))  # $450 back, unsettled

        account = broker.get_account()
        assert account.total_cash == Decimal("500.00")
        assert account.cash.settled == Decimal("50.00")

        # Total cash would cover this; settled cash does not.
        result = broker.submit_order(buy("HLAL", "5", key="c"))
        assert result.status is OrderStatus.REJECTED
        assert "unsettled proceeds cannot be used" in (result.rejection_reason or "")

    def test_proceeds_settle_on_the_next_trading_day(
        self, broker: MockBroker, clock: FrozenClock
    ) -> None:
        broker.submit_order(buy("SPUS", "10", key="a"))
        broker.submit_order(sell("SPUS", "10", key="b"))
        assert broker.get_account().cash.settled == Decimal("50.00")

        clock.advance(days=1)  # 2024-06-12 (Wed) -> 2024-06-13 (Thu)
        account = broker.get_account()
        assert account.cash.settled == Decimal("500.00")
        assert account.cash.unsettled == Decimal("0.00")

    def test_friday_sale_does_not_settle_over_the_weekend(
        self, calendar: TradingCalendar
    ) -> None:
        friday = FrozenClock(dt.datetime(2024, 6, 14, 20, 0, tzinfo=UTC))
        broker = MockBroker(
            starting_cash=Decimal("500.00"),
            clock=friday,
            calendar=calendar,
            prices={"SPUS": Decimal("45.00")},
            slippage_pct=0.0,
        )
        broker.submit_order(buy("SPUS", "10", key="a"))
        broker.submit_order(sell("SPUS", "10", key="b"))

        friday.advance(days=2)  # Sunday
        assert broker.get_account().cash.unsettled == Decimal("450.00")

        friday.advance(days=1)  # Monday
        assert broker.get_account().cash.settled == Decimal("500.00")


class TestIdempotency:
    def test_resubmitting_the_same_key_does_not_trade_twice(self, broker: MockBroker) -> None:
        first = broker.submit_order(buy("SPUS", "2", key="same"))
        second = broker.submit_order(buy("SPUS", "2", key="same"))

        assert first.broker_order_id == second.broker_order_id
        position = broker.get_position("SPUS")
        assert position is not None
        assert position.quantity == Decimal("2")
        assert broker.get_account().cash.settled == Decimal("410.00")

    def test_different_keys_do_trade_twice(self, broker: MockBroker) -> None:
        broker.submit_order(buy("SPUS", "2", key="one"))
        broker.submit_order(buy("SPUS", "2", key="two"))
        position = broker.get_position("SPUS")
        assert position is not None
        assert position.quantity == Decimal("4")


class TestOrderRequestValidation:
    def test_limit_order_needs_a_price(self) -> None:
        with pytest.raises(ValueError, match="needs a limit_price"):
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("1"),
                order_type=OrderType.LIMIT,
                idempotency_key="k",
            )

    def test_fractional_must_be_a_market_order(self) -> None:
        """A broker restriction, encoded so the engine cannot forget it."""
        with pytest.raises(ValueError, match="must be market orders"):
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("0.5"),
                order_type=OrderType.LIMIT,
                limit_price=Decimal("45"),
                idempotency_key="k",
                is_fractional=True,
            )

    def test_fractional_cannot_carry_bracket_legs(self) -> None:
        with pytest.raises(ValueError, match="cannot carry broker-side bracket"):
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("0.5"),
                order_type=OrderType.MARKET,
                idempotency_key="k",
                is_fractional=True,
                stop_loss_price=Decimal("40"),
            )

    def test_idempotency_key_is_required(self) -> None:
        with pytest.raises(ValueError, match="idempotency_key is required"):
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("1"),
                order_type=OrderType.MARKET,
                idempotency_key="",
            )


class TestFractionalSupport:
    def test_non_fractionable_symbol_rejected(
        self, clock: FrozenClock, calendar: TradingCalendar
    ) -> None:
        broker = MockBroker(
            starting_cash=Decimal("500"),
            clock=clock,
            calendar=calendar,
            prices={"XYZ": Decimal("40")},
            fractionable=set(),
        )
        result = broker.submit_order(
            OrderRequest(
                symbol=Symbol("XYZ"),
                side=Side.BUY,
                quantity=Decimal("0.5"),
                order_type=OrderType.MARKET,
                idempotency_key="k",
                is_fractional=True,
            )
        )
        assert result.status is OrderStatus.REJECTED
        assert "not fractionable" in (result.rejection_reason or "")
