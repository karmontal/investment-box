"""Order management, the kill switch and the engine state machine.

Two properties dominate this file: an order can never be placed twice, and the
Shariah gate runs on the *final* order rather than the proposal.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from investment_box.core.clock import FrozenClock
from investment_box.core.types import ComplianceStatus, OrderStatus, Side
from investment_box.engine.kill_switch import KillSwitch
from investment_box.engine.state import EngineState, EngineStateMachine
from investment_box.execution.mock_broker import MockBroker
from investment_box.execution.order_manager import OrderManager, make_idempotency_key
from investment_box.risk.sizing import PositionSize


def size_for(symbol: str = "SPUS", qty: str = "2", price: str = "45.00") -> PositionSize:
    return PositionSize(
        symbol=symbol,
        quantity=Decimal(qty),
        entry_price=Decimal(price),
        stop_price=Decimal("40.00"),
        take_profit_price=Decimal("55.00"),
        notional=Decimal(qty) * Decimal(price),
        risk_amount=Decimal("10.00"),
        is_fractional=False,
        reason="test",
    )


@pytest.fixture
def orders(broker: MockBroker, database, settings, audit, clock) -> OrderManager:
    return OrderManager(broker, database, settings, audit, clock=clock)


class TestIdempotencyKeys:
    def test_deterministic(self) -> None:
        day = dt.date(2024, 6, 12)
        a = make_idempotency_key(symbol="SPUS", side=Side.BUY, intent="open", day=day)
        b = make_idempotency_key(symbol="SPUS", side=Side.BUY, intent="open", day=day)
        assert a == b

    def test_differs_by_side_symbol_intent_and_day(self) -> None:
        day = dt.date(2024, 6, 12)
        base = make_idempotency_key(symbol="SPUS", side=Side.BUY, intent="open", day=day)
        variants = [
            make_idempotency_key(symbol="HLAL", side=Side.BUY, intent="open", day=day),
            make_idempotency_key(symbol="SPUS", side=Side.SELL, intent="open", day=day),
            make_idempotency_key(symbol="SPUS", side=Side.BUY, intent="exit", day=day),
            make_idempotency_key(
                symbol="SPUS", side=Side.BUY, intent="open", day=dt.date(2024, 6, 13)
            ),
        ]
        assert len(set(variants) | {base}) == 5

    def test_fits_alpaca_client_order_id_limit(self) -> None:
        key = make_idempotency_key(
            symbol="VERYLONG", side=Side.BUY, intent="open", day=dt.date(2024, 6, 12)
        )
        assert len(key) <= 128

    def test_nonce_permits_a_deliberate_second_trade(self) -> None:
        day = dt.date(2024, 6, 12)
        a = make_idempotency_key(symbol="SPUS", side=Side.BUY, intent="open", day=day)
        b = make_idempotency_key(
            symbol="SPUS", side=Side.BUY, intent="open", day=day, nonce="2"
        )
        assert a != b


class TestOpenPosition:
    def test_places_a_limit_order_with_bracket_legs(
        self, orders: OrderManager, broker: MockBroker
    ) -> None:
        placed = orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        assert placed.accepted
        assert placed.request is not None
        assert placed.request.limit_price is not None
        assert placed.request.stop_loss_price == Decimal("40.00")
        assert not placed.synthetic_stop

    def test_duplicate_submission_is_suppressed(self, orders: OrderManager) -> None:
        """A retry after an ambiguous timeout must not buy twice."""
        first = orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        second = orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        assert first.accepted
        assert not second.accepted
        assert "already placed" in second.reason

    def test_compliance_gate_runs_on_the_final_order(self, orders: OrderManager) -> None:
        """Not on the proposal: the final quantity and price are what matter."""
        placed = orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.NON_COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        assert not placed.accepted
        assert "NON_COMPLIANT" in placed.reason

    @pytest.mark.parametrize(
        "status", [ComplianceStatus.DOUBTFUL, ComplianceStatus.UNKNOWN]
    )
    def test_doubtful_and_unknown_are_refused_without_approval(
        self, orders: OrderManager, status: ComplianceStatus
    ) -> None:
        placed = orders.open_position(
            size=size_for(), compliance_status=status, compliance_source="test",
            reference_price=Decimal("45.00"), available_cash=Decimal("450"),
        )
        assert not placed.accepted
        assert "human decision" in placed.reason

    def test_human_approval_unlocks_doubtful(self, orders: OrderManager) -> None:
        placed = orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.DOUBTFUL,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"), human_approved=True,
        )
        assert placed.accepted

    def test_buying_beyond_settled_cash_is_refused(self, orders: OrderManager) -> None:
        placed = orders.open_position(
            size=size_for(qty="10"), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("50"),
        )
        assert not placed.accepted
        assert "margin" in placed.reason

    def test_leveraged_etf_is_refused(self, orders: OrderManager) -> None:
        placed = orders.open_position(
            size=size_for(symbol="TQQQ"), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        assert not placed.accepted
        assert "forbidden instrument" in placed.reason

    def test_fractional_refused_when_disabled(self, orders: OrderManager) -> None:
        fractional = PositionSize(
            symbol="SPUS", quantity=Decimal("0.5"), entry_price=Decimal("45"),
            stop_price=Decimal("40"), take_profit_price=None,
            notional=Decimal("22.50"), risk_amount=Decimal("2.50"),
            is_fractional=True, reason="test",
        )
        placed = orders.open_position(
            size=fractional, compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        assert not placed.accepted
        assert "fractional trading is disabled" in placed.reason


class TestClosePosition:
    def test_closes_a_held_position(
        self, orders: OrderManager, broker: MockBroker
    ) -> None:
        orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        placed = orders.close_position(
            symbol="SPUS", quantity=Decimal("2"),
            compliance_status=ComplianceStatus.COMPLIANT, reason="signal",
        )
        assert placed.accepted
        assert placed.request is not None
        # Exits are market orders: a limit exit that does not fill leaves a
        # position the engine believes is closed.
        assert placed.request.order_type.value == "market"

    def test_selling_more_than_held_is_refused(self, orders: OrderManager) -> None:
        placed = orders.close_position(
            symbol="SPUS", quantity=Decimal("5"),
            compliance_status=ComplianceStatus.COMPLIANT, reason="signal",
        )
        assert not placed.accepted
        assert "short position" in placed.reason

    def test_non_compliant_holdings_can_still_be_sold(
        self, orders: OrderManager, broker: MockBroker
    ) -> None:
        """The exit policy requires it; blocking the sale would trap us."""
        orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        placed = orders.close_position(
            symbol="SPUS", quantity=Decimal("2"),
            compliance_status=ComplianceStatus.NON_COMPLIANT, reason="compliance_exit",
        )
        assert placed.accepted


class TestReconciliation:
    def test_reconcile_updates_local_state(
        self, orders: OrderManager, broker: MockBroker
    ) -> None:
        orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        # The mock fills synchronously, so nothing remains to reconcile.
        assert orders.reconcile() == []

    def test_reconcile_is_safe_with_no_orders(self, orders: OrderManager) -> None:
        assert orders.reconcile() == []


class TestEngineState:
    @pytest.fixture
    def machine(self, audit, clock: FrozenClock) -> EngineStateMachine:
        return EngineStateMachine(audit, clock=clock)

    def test_starts_idle(self, machine: EngineStateMachine) -> None:
        assert machine.state is EngineState.IDLE

    def test_start_and_pause(self, machine: EngineStateMachine) -> None:
        machine.start()
        assert machine.state.can_open_positions
        machine.pause("testing")
        assert not machine.state.can_open_positions

    def test_exits_allowed_while_paused(self, machine: EngineStateMachine) -> None:
        machine.pause("drawdown")
        assert machine.state.can_close_positions

    def test_kill_is_terminal(self, machine: EngineStateMachine) -> None:
        """A kill is not a pause; /resume must not undo it."""
        machine.start()
        machine.kill("emergency")
        assert machine.state is EngineState.KILLED
        assert not machine.resume()
        assert machine.state is EngineState.KILLED

    def test_kill_blocks_even_exits(self, machine: EngineStateMachine) -> None:
        machine.kill("emergency")
        assert not machine.state.can_close_positions

    def test_transitions_are_audited(self, machine: EngineStateMachine, audit) -> None:
        machine.start()
        machine.pause("testing")
        events = [e.event_type for e in audit.recent()]
        assert "engine.running" in events
        assert "engine.paused" in events


class TestKillSwitch:
    @pytest.fixture
    def switch(self, broker: MockBroker, orders: OrderManager, audit, clock) -> KillSwitch:
        return KillSwitch(broker, orders, EngineStateMachine(audit, clock=clock), audit)

    def test_kills_without_closing_by_default(
        self, switch: KillSwitch, orders: OrderManager
    ) -> None:
        orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        result = switch.activate("test")
        assert result.engine_killed
        assert result.positions_closed == []
        assert switch.broker.get_position("SPUS") is not None

    def test_closes_positions_when_asked(
        self, switch: KillSwitch, orders: OrderManager
    ) -> None:
        orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )
        result = switch.activate("emergency", close_positions=True)
        assert result.engine_killed
        assert "SPUS" in result.positions_closed
        assert switch.broker.get_position("SPUS") is None

    def test_reports_positions_it_could_not_close(
        self, switch: KillSwitch, monkeypatch: pytest.MonkeyPatch, orders: OrderManager
    ) -> None:
        """A partial kill must say exactly what is still open."""
        orders.open_position(
            size=size_for(), compliance_status=ComplianceStatus.COMPLIANT,
            compliance_source="test", reference_price=Decimal("45.00"),
            available_cash=Decimal("450"),
        )

        def explode(**_kwargs):
            raise RuntimeError("broker down")

        monkeypatch.setattr(switch.orders, "close_position", explode)
        result = switch.activate("emergency", close_positions=True)

        assert result.engine_killed  # still killed despite the failure
        assert "SPUS" in result.positions_failed
        assert "close these manually" in result.detail()

    def test_kills_even_if_cancelling_fails(
        self, switch: KillSwitch, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode() -> int:
            raise RuntimeError("broker down")

        monkeypatch.setattr(switch.broker, "cancel_all_orders", explode)
        result = switch.activate("emergency")
        assert result.engine_killed
        assert result.errors

    def test_is_audited(self, switch: KillSwitch, audit) -> None:
        switch.activate("test reason")
        events = [e.event_type for e in audit.recent()]
        assert "kill_switch.activated" in events
        assert "kill_switch.completed" in events


class TestMarginAccountIsRefused:
    """A margin account must block trading without blocking the application.

    Found against a real Alpaca paper account, which defaults to a margin
    multiplier of 4.0. The first version raised and took down the whole
    process; the operator could not even open the dashboard to see why.
    """

    def _broker(self, multiplier: float, shorting: bool = False):
        from investment_box.execution.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker.__new__(AlpacaBroker)
        broker.is_paper = True
        broker.base_url = "https://paper-api.alpaca.markets"
        broker._api_key = "k"
        broker._secret_key = "s"
        broker._verified_cash_account = False

        class _Account:
            def __init__(self) -> None:
                self.multiplier = multiplier
                self.shorting_enabled = shorting
                self.cash = "500"
                self.equity = "500"
                self.long_market_value = "0"
                self.trading_blocked = False

        class _Client:
            def get_account(self) -> _Account:
                return _Account()

        broker._client = _Client()
        return broker

    def test_margin_account_fails_verification(self) -> None:
        from investment_box.core.errors import BrokerError

        with pytest.raises(BrokerError, match="CASH account"):
            self._broker(4.0).verify_cash_account()

    def test_shorting_enabled_fails_verification(self) -> None:
        from investment_box.core.errors import BrokerError

        with pytest.raises(BrokerError, match="shorting"):
            self._broker(1.0, shorting=True).verify_cash_account()

    def test_cash_account_passes(self) -> None:
        broker = self._broker(1.0)
        broker.verify_cash_account()
        assert broker._verified_cash_account

    def test_account_snapshot_reports_it_is_not_cash(self) -> None:
        assert not self._broker(4.0).get_account().is_cash_account

    def test_orders_are_rejected_not_raised(self) -> None:
        """The operator must be able to see the problem, not just crash."""
        from investment_box.core.types import OrderType, Symbol
        from investment_box.execution.base import OrderRequest

        broker = self._broker(4.0)
        result = broker.submit_order(
            OrderRequest(
                symbol=Symbol("SPUS"),
                side=Side.BUY,
                quantity=Decimal("1"),
                order_type=OrderType.MARKET,
                idempotency_key="k1",
            )
        )
        assert result.status is OrderStatus.REJECTED
        assert "CASH account" in (result.rejection_reason or "")


class TestPaperLiveMismatch:
    def test_paper_flag_with_live_url_refused(self) -> None:
        from investment_box.core.errors import BrokerError
        from investment_box.execution.alpaca_broker import AlpacaBroker

        with pytest.raises(BrokerError, match="paper"):
            AlpacaBroker("k", "s", paper=True, base_url="https://api.alpaca.markets")

    def test_missing_credentials_refused(self) -> None:
        from investment_box.core.errors import BrokerError
        from investment_box.execution.alpaca_broker import AlpacaBroker

        with pytest.raises(BrokerError, match="API key"):
            AlpacaBroker("", "", paper=True)
