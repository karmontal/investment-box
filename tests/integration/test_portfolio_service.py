"""The service layer -- the single read path shared by the UI and the bot."""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from investment_box.config.loader import load_settings
from investment_box.core.clock import FrozenClock
from investment_box.core.types import OrderType, Side, Symbol, TradingMode
from investment_box.db.models import Trade
from investment_box.db.session import Database
from investment_box.execution.base import OrderRequest
from investment_box.execution.mock_broker import MockBroker
from investment_box.services.portfolio import PortfolioService


def buy(symbol: str, qty: str, key: str) -> OrderRequest:
    return OrderRequest(
        symbol=Symbol(symbol),
        side=Side.BUY,
        quantity=Decimal(qty),
        order_type=OrderType.MARKET,
        idempotency_key=key,
    )


class TestAccountView:
    def test_flat_account(self, portfolio: PortfolioService) -> None:
        view = portfolio.get_account_view()
        assert view.equity == Decimal("500.00")
        assert view.positions == ()
        assert view.trading_mode is TradingMode.PAPER
        assert view.mode_tag == "[PAPER]"

    def test_mode_tag_is_live_when_live(
        self, broker: MockBroker, database: Database, clock: FrozenClock
    ) -> None:
        settings = load_settings(overrides={"trading_mode": "live"})
        view = PortfolioService(broker, database, settings, clock=clock).get_account_view()
        assert view.mode_tag == "[LIVE]"
        assert any("LIVE" in w for w in view.warnings)

    def test_position_appears_after_a_fill(
        self, portfolio: PortfolioService, broker: MockBroker
    ) -> None:
        broker.submit_order(buy("SPUS", "2", "k"))
        view = portfolio.get_account_view()
        assert len(view.positions) == 1

        position = view.positions[0]
        assert position.symbol == "SPUS"
        assert position.quantity == Decimal("2")
        assert position.market_value == Decimal("90.00")

    def test_unrealised_pnl_tracks_price(
        self, portfolio: PortfolioService, broker: MockBroker
    ) -> None:
        broker.submit_order(buy("SPUS", "2", "k"))
        broker.set_price("SPUS", Decimal("50.00"))
        position = portfolio.get_account_view().positions[0]
        assert position.unrealized_pnl == Decimal("10.00")
        assert position.unrealized_pnl_pct == pytest.approx(0.1111, abs=1e-3)

    def test_settled_and_unsettled_are_reported_separately(
        self, portfolio: PortfolioService, broker: MockBroker
    ) -> None:
        broker.submit_order(buy("SPUS", "10", "a"))
        broker.submit_order(
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.SELL,
                quantity=Decimal("10"),
                order_type=OrderType.MARKET,
                idempotency_key="b",
            )
        )
        view = portfolio.get_account_view()
        assert view.cash_unsettled == Decimal("450.00")
        assert view.cash_settled == Decimal("50.00")

    def test_non_cash_account_warns_first(
        self, portfolio: PortfolioService, broker: MockBroker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = broker.get_account

        def margin_account():
            snapshot = original()
            return type(snapshot)(
                equity=snapshot.equity,
                cash=snapshot.cash,
                positions_value=snapshot.positions_value,
                buying_power=snapshot.buying_power,
                is_cash_account=False,
                trading_blocked=False,
                taken_at=snapshot.taken_at,
            )

        monkeypatch.setattr(broker, "get_account", margin_account)
        view = portfolio.get_account_view()
        assert not view.is_cash_account
        assert "NOT A CASH ACCOUNT" in view.warnings[0]


class TestCapitalUsage:
    def test_allocation_not_account_equity_is_the_denominator(
        self, broker: MockBroker, database: Database, clock: FrozenClock
    ) -> None:
        """The bot may hold $500 of a larger account; percentages use the $500."""
        rich = MockBroker(
            starting_cash=Decimal("10000.00"),
            clock=clock,
            calendar=broker.calendar,
            prices={"SPUS": Decimal("45.00")},
            slippage_pct=0.0,
        )
        rich.submit_order(buy("SPUS", "2", "k"))
        settings = load_settings(overrides={"capital": {"allocation_usd": "500"}})
        view = PortfolioService(rich, database, settings, clock=clock).get_account_view()

        assert view.equity == Decimal("10000.00")
        assert view.capital.allocation == Decimal("500")
        assert view.positions[0].pct_of_allocation == pytest.approx(0.18)  # 90 / 500

    def test_available_cash_is_capped_by_the_allocation(
        self, broker: MockBroker, database: Database, clock: FrozenClock
    ) -> None:
        rich = MockBroker(
            starting_cash=Decimal("10000.00"),
            clock=clock,
            calendar=broker.calendar,
            slippage_pct=0.0,
        )
        settings = load_settings(overrides={"capital": {"allocation_usd": "500"}})
        usage = PortfolioService(rich, database, settings, clock=clock).get_account_view().capital
        assert usage.available_settled == Decimal("500.00")

    def test_slots_free_counts_down(
        self, portfolio: PortfolioService, broker: MockBroker
    ) -> None:
        usage = portfolio.get_account_view().capital
        assert usage.position_slots_free == 5

        broker.submit_order(buy("SPUS", "1", "a"))
        broker.submit_order(buy("HLAL", "1", "b"))
        assert portfolio.get_account_view().capital.position_slots_free == 3

    def test_deployed_pct(self, portfolio: PortfolioService, broker: MockBroker) -> None:
        broker.submit_order(buy("SPUS", "2", "k"))
        assert portfolio.get_account_view().capital.deployed_pct == pytest.approx(0.18)


class TestEquitySnapshots:
    def test_snapshot_is_idempotent_per_date(self, portfolio: PortfolioService) -> None:
        portfolio.record_equity_snapshot()
        portfolio.record_equity_snapshot()
        assert len(portfolio.equity_curve()) == 1

    def test_day_pnl_computed_against_the_previous_snapshot(
        self, portfolio: PortfolioService, broker: MockBroker, clock: FrozenClock
    ) -> None:
        portfolio.record_equity_snapshot()

        broker.submit_order(buy("SPUS", "2", "k"))
        broker.set_price("SPUS", Decimal("50.00"))
        clock.advance(days=1)

        row = portfolio.record_equity_snapshot()
        assert row.day_pnl == Decimal("10.00")

    def test_first_snapshot_has_no_day_pnl(self, portfolio: PortfolioService) -> None:
        """No prior point means no honest number. None, not zero."""
        assert portfolio.record_equity_snapshot().day_pnl is None

    def test_curve_is_ordered(self, portfolio: PortfolioService, clock: FrozenClock) -> None:
        for _ in range(3):
            portfolio.record_equity_snapshot()
            clock.advance(days=1)
        curve = portfolio.equity_curve()
        assert [row.snapshot_date for row in curve] == sorted(row.snapshot_date for row in curve)


class TestTradeHistory:
    def _closed_trade(self, net: str, symbol: str = "SPUS") -> Trade:
        return Trade(
            symbol=symbol,
            strategy="test",
            quantity=Decimal("2"),
            entry_price=Decimal("45"),
            entry_at=dt.datetime(2024, 6, 10, tzinfo=dt.UTC),
            entry_date=dt.date(2024, 6, 10),
            exit_price=Decimal("50"),
            exit_at=dt.datetime(2024, 6, 12, tzinfo=dt.UTC),
            exit_date=dt.date(2024, 6, 12),
            exit_reason="take_profit",
            net_pnl=Decimal(net),
            compliance_status_at_entry="compliant",
            compliance_source_at_entry="seed_universe",
            trading_mode="paper",
        )

    def test_realised_pnl_sums_closed_trades(
        self, portfolio: PortfolioService, database: Database
    ) -> None:
        with database.session() as session:
            session.add(self._closed_trade("10.00"))
            session.add(self._closed_trade("-4.00", "HLAL"))
        assert portfolio.realized_pnl_total() == Decimal("6.00")

    def test_paper_and_live_histories_do_not_mix(
        self, portfolio: PortfolioService, database: Database
    ) -> None:
        """A live P&L number must never be inflated by paper results."""
        live = self._closed_trade("1000.00")
        live.trading_mode = "live"
        with database.session() as session:
            session.add(self._closed_trade("10.00"))
            session.add(live)
        assert portfolio.realized_pnl_total() == Decimal("10.00")

    def test_closed_trades_newest_first(
        self, portfolio: PortfolioService, database: Database
    ) -> None:
        older = self._closed_trade("1.00")
        older.exit_at = dt.datetime(2024, 6, 11, tzinfo=dt.UTC)
        with database.session() as session:
            session.add(older)
            session.add(self._closed_trade("2.00", "HLAL"))
        trades = portfolio.closed_trades()
        assert trades[0].symbol == "HLAL"

    def test_open_trades_excluded_from_history(
        self, portfolio: PortfolioService, database: Database
    ) -> None:
        open_trade = self._closed_trade("5.00")
        open_trade.exit_at = None
        open_trade.exit_date = None
        with database.session() as session:
            session.add(open_trade)
        assert portfolio.closed_trades() == []
        assert len(portfolio.open_trades()) == 1

    def test_days_held_uses_trading_days(
        self, portfolio: PortfolioService, database: Database, broker: MockBroker
    ) -> None:
        """Entry on Friday, viewed on the following Wednesday, is 3 trading days."""
        trade = self._closed_trade("0")
        trade.exit_at = None
        trade.exit_date = None
        trade.entry_date = dt.date(2024, 6, 7)  # Friday
        with database.session() as session:
            session.add(trade)

        broker.submit_order(buy("SPUS", "2", "k"))
        # The clock fixture sits on Wednesday 2024-06-12.
        assert portfolio.get_account_view().positions[0].days_held == 3
