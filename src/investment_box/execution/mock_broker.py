"""In-memory broker for tests and for running with no credentials.

It models the parts that actually bite in a small cash account:

* **T+1 settlement.** Sale proceeds land in ``unsettled`` and move to
  ``settled`` only when the calendar says so. A buy may only spend settled
  cash, so the good-faith-violation path is testable without a real account.
* **Cash-account semantics.** No margin, no shorting. A sell larger than the
  held quantity is rejected, as is a buy exceeding settled cash.
* **Idempotency.** Re-submitting the same ``idempotency_key`` returns the
  original result rather than trading again.

Fills are immediate and at the last price plus a configurable slippage. That is
optimistic, and deliberately so: this broker exists to exercise control flow,
not to estimate returns. Realistic fills are the backtester's job.
"""

from __future__ import annotations

import datetime as dt
import itertools
from decimal import Decimal

from investment_box.core.clock import UTC, Clock, SystemClock, TradingCalendar
from investment_box.core.logging import get_logger
from investment_box.core.types import (
    CashLedger,
    OrderStatus,
    PendingSettlement,
    Side,
    Symbol,
    to_money,
    to_qty,
)
from investment_box.execution.base import (
    AccountSnapshot,
    BrokerPosition,
    OrderRequest,
    OrderResult,
)

log = get_logger(__name__)

DEFAULT_PRICE = Decimal("50.00")


class MockBroker:
    """A deterministic, in-memory cash-account broker."""

    name = "mock"
    is_paper = True

    def __init__(
        self,
        starting_cash: Decimal | float | str = Decimal("500.00"),
        *,
        clock: Clock | None = None,
        calendar: TradingCalendar | None = None,
        prices: dict[str, Decimal] | None = None,
        slippage_pct: float = 0.0005,
        settlement_days: int = 1,
        fractionable: set[str] | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        self.calendar = calendar or TradingCalendar()
        self.settlement_days = settlement_days
        self.slippage_pct = Decimal(str(slippage_pct))

        self._settled = to_money(starting_cash)
        self._pending: list[PendingSettlement] = []
        self._positions: dict[str, BrokerPosition] = {}
        self._orders: dict[str, OrderResult] = {}
        self._by_idempotency: dict[str, str] = {}
        self._prices: dict[str, Decimal] = {
            key.upper(): to_money(value) for key, value in (prices or {}).items()
        }
        self._ids = itertools.count(1)
        #: ``None`` means "everything is fractionable", which matches Alpaca for
        #: most liquid ETFs. Pass a set to model a narrower reality.
        self._fractionable = fractionable

    # ----------------------------------------------------------------- prices

    def set_price(self, symbol: Symbol | str, price: Decimal | float | str) -> None:
        """Set the mark for a symbol. Tests drive fills and P&L with this."""
        self._prices[str(symbol).upper()] = to_money(price)
        ticker = str(symbol).upper()
        if ticker in self._positions:
            held = self._positions[ticker]
            self._positions[ticker] = BrokerPosition(
                symbol=held.symbol,
                quantity=held.quantity,
                avg_entry_price=held.avg_entry_price,
                current_price=self._prices[ticker],
                opened_at=held.opened_at,
            )

    def get_last_price(self, symbol: Symbol | str) -> Decimal | None:
        return self._prices.get(str(symbol).upper(), DEFAULT_PRICE)

    def supports_fractional(self, symbol: Symbol | str) -> bool:
        if self._fractionable is None:
            return True
        return str(symbol).upper() in self._fractionable

    # ------------------------------------------------------------- settlement

    def settle_due(self, today: dt.date | None = None) -> Decimal:
        """Move matured proceeds into settled cash. Returns the amount moved.

        Called by :meth:`get_account`, so callers normally never invoke it
        directly; exposed for tests that want to assert on settlement timing.
        """
        today = today or self.clock.now().astimezone(UTC).date()
        matured = [entry for entry in self._pending if entry.settles_on <= today]
        if not matured:
            return Decimal("0.00")
        amount = to_money(sum((entry.amount for entry in matured), Decimal("0")))
        self._settled = to_money(self._settled + amount)
        self._pending = [entry for entry in self._pending if entry.settles_on > today]
        log.debug("mock_broker.settled", amount=str(amount), count=len(matured))
        return amount

    @property
    def _unsettled_total(self) -> Decimal:
        return to_money(sum((entry.amount for entry in self._pending), Decimal("0")))

    # ---------------------------------------------------------------- account

    def is_connected(self) -> bool:
        return True

    def get_account(self) -> AccountSnapshot:
        self.settle_due()
        positions_value = to_money(
            sum((position.market_value for position in self._positions.values()), Decimal("0"))
        )
        ledger = CashLedger(settled=self._settled, unsettled=self._unsettled_total)
        return AccountSnapshot(
            equity=to_money(ledger.total + positions_value),
            cash=ledger,
            positions_value=positions_value,
            # Buying power in a cash account is settled cash. Not equity, and
            # not cash including unsettled proceeds.
            buying_power=ledger.available_for_trading,
            is_cash_account=True,
            trading_blocked=False,
            taken_at=self.clock.now(),
        )

    def get_positions(self) -> list[BrokerPosition]:
        return sorted(self._positions.values(), key=lambda p: p.symbol)

    def get_position(self, symbol: Symbol | str) -> BrokerPosition | None:
        return self._positions.get(str(symbol).upper())

    # ----------------------------------------------------------------- orders

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Fill (or reject) immediately, respecting cash-account rules."""
        if request.idempotency_key in self._by_idempotency:
            existing_id = self._by_idempotency[request.idempotency_key]
            log.info("mock_broker.duplicate_suppressed", key=request.idempotency_key)
            return self._orders[existing_id]

        self.settle_due()
        ticker = str(request.symbol).upper()
        now = self.clock.now()
        price = self._fill_price(ticker, request.side)
        quantity = to_qty(request.quantity)

        rejection = self._validate(ticker, request.side, quantity, price)
        if rejection is not None:
            result = OrderResult(
                idempotency_key=request.idempotency_key,
                broker_order_id=None,
                symbol=Symbol(ticker),
                side=request.side,
                status=OrderStatus.REJECTED,
                requested_quantity=quantity,
                submitted_at=now,
                rejection_reason=rejection,
            )
            self._record(result, request.idempotency_key)
            log.warning("mock_broker.rejected", symbol=ticker, reason=rejection)
            return result

        if request.side is Side.BUY:
            self._apply_buy(ticker, quantity, price, now)
        else:
            self._apply_sell(ticker, quantity, price, now)

        result = OrderResult(
            idempotency_key=request.idempotency_key,
            broker_order_id=f"mock-{next(self._ids):06d}",
            symbol=Symbol(ticker),
            side=request.side,
            status=OrderStatus.FILLED,
            requested_quantity=quantity,
            filled_quantity=quantity,
            filled_avg_price=price,
            submitted_at=now,
            filled_at=now,
        )
        self._record(result, request.idempotency_key)
        return result

    def _validate(self, ticker: str, side: Side, quantity: Decimal, price: Decimal) -> str | None:
        if side is Side.BUY:
            cost = to_money(quantity * price)
            available = self.get_account().cash.available_for_trading
            if cost > available:
                return (
                    f"insufficient settled cash: need {cost}, have {available} "
                    f"(cash account -- unsettled proceeds cannot be used)"
                )
            if quantity != quantity.to_integral_value() and not self.supports_fractional(ticker):
                return f"{ticker} is not fractionable"
            return None

        held = self._positions.get(ticker)
        held_qty = held.quantity if held else Decimal("0")
        if quantity > held_qty:
            return (
                f"sell of {quantity} exceeds held {held_qty}; shorting is not "
                f"permitted in this account"
            )
        return None

    def _apply_buy(self, ticker: str, quantity: Decimal, price: Decimal, now: dt.datetime) -> None:
        cost = to_money(quantity * price)
        self._settled = to_money(self._settled - cost)
        existing = self._positions.get(ticker)
        if existing is None:
            self._positions[ticker] = BrokerPosition(
                symbol=Symbol(ticker),
                quantity=quantity,
                avg_entry_price=price,
                current_price=price,
                opened_at=now,
            )
        else:
            total_qty = existing.quantity + quantity
            new_basis = to_money(
                (existing.cost_basis + cost) / total_qty if total_qty else Decimal("0")
            )
            self._positions[ticker] = BrokerPosition(
                symbol=existing.symbol,
                quantity=to_qty(total_qty),
                avg_entry_price=new_basis,
                current_price=price,
                opened_at=existing.opened_at,
            )

    def _apply_sell(self, ticker: str, quantity: Decimal, price: Decimal, now: dt.datetime) -> None:
        proceeds = to_money(quantity * price)
        today = now.astimezone(UTC).date()
        self._pending.append(
            PendingSettlement(
                symbol=Symbol(ticker),
                amount=proceeds,
                sold_on=today,
                settles_on=self.calendar.settlement_date(today, self.settlement_days),
            )
        )
        held = self._positions[ticker]
        remaining = to_qty(held.quantity - quantity)
        if remaining <= 0:
            del self._positions[ticker]
        else:
            self._positions[ticker] = BrokerPosition(
                symbol=held.symbol,
                quantity=remaining,
                avg_entry_price=held.avg_entry_price,
                current_price=price,
                opened_at=held.opened_at,
            )

    def _fill_price(self, ticker: str, side: Side) -> Decimal:
        base = self._prices.get(ticker, DEFAULT_PRICE)
        drift = base * self.slippage_pct
        return to_money(base + drift if side is Side.BUY else base - drift)

    def _record(self, result: OrderResult, key: str) -> None:
        order_id = result.broker_order_id or f"rejected-{key}"
        self._orders[order_id] = result
        self._by_idempotency[key] = order_id

    def get_order(self, broker_order_id: str) -> OrderResult | None:
        return self._orders.get(broker_order_id)

    def list_open_orders(self) -> list[OrderResult]:
        """Always empty: this broker fills or rejects synchronously."""
        return [order for order in self._orders.values() if not order.is_terminal]

    def cancel_order(self, broker_order_id: str) -> bool:
        order = self._orders.get(broker_order_id)
        return order is not None and not order.is_terminal

    def cancel_all_orders(self) -> int:
        return len(self.list_open_orders())
