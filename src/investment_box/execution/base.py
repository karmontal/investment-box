"""The broker contract.

Deliberately narrow: everything the engine needs and nothing broker-specific,
so that adding Interactive Brokers later means writing one adapter rather than
touching the engine.

Note what is absent. There is no ``short``, no ``buy_to_cover``, no margin
parameter and no leverage setting -- not because Alpaca lacks them, but because
this application must never use them. A capability that does not exist in the
interface cannot be reached by a bug.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from investment_box.core.types import (
    CashLedger,
    OrderStatus,
    OrderType,
    Side,
    Symbol,
    TimeInForce,
    to_money,
)


@dataclass(frozen=True, slots=True)
class BrokerPosition:
    """An open position as the broker reports it."""

    symbol: Symbol
    quantity: Decimal
    avg_entry_price: Decimal
    current_price: Decimal
    opened_at: dt.datetime | None = None

    @property
    def market_value(self) -> Decimal:
        return to_money(self.quantity * self.current_price)

    @property
    def cost_basis(self) -> Decimal:
        return to_money(self.quantity * self.avg_entry_price)

    @property
    def unrealized_pnl(self) -> Decimal:
        return to_money(self.market_value - self.cost_basis)

    @property
    def unrealized_pnl_pct(self) -> float:
        basis = self.cost_basis
        if basis == 0:
            return 0.0
        return float(self.unrealized_pnl / basis)

    @property
    def is_fractional(self) -> bool:
        return self.quantity != self.quantity.to_integral_value()


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    """Account state at a point in time."""

    equity: Decimal
    cash: CashLedger
    positions_value: Decimal
    buying_power: Decimal
    #: True when the broker account is a cash (not margin) account. The engine
    #: refuses to trade a margin account.
    is_cash_account: bool
    #: Broker-side flag that the account is blocked from trading.
    trading_blocked: bool = False
    taken_at: dt.datetime | None = None

    @property
    def total_cash(self) -> Decimal:
        return self.cash.total


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """An order the engine wants placed.

    ``idempotency_key`` is required, not optional. It is the only thing that
    stops a retry after an ambiguous timeout from buying twice.
    """

    symbol: Symbol
    side: Side
    quantity: Decimal
    order_type: OrderType
    idempotency_key: str
    limit_price: Decimal | None = None
    stop_loss_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    #: Set when the quantity is fractional. Fractional orders cannot be limit
    #: orders and cannot carry bracket legs -- the order manager enforces this
    #: and manages the stop itself.
    is_fractional: bool = False

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"{self.symbol}: quantity must be positive, got {self.quantity}")
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            raise ValueError(f"{self.symbol}: a limit order needs a limit_price")
        if self.is_fractional and self.order_type is not OrderType.MARKET:
            raise ValueError(
                f"{self.symbol}: fractional orders must be market orders "
                f"(broker restriction), got {self.order_type}"
            )
        if self.is_fractional and (self.stop_loss_price or self.take_profit_price):
            raise ValueError(
                f"{self.symbol}: fractional orders cannot carry broker-side bracket "
                f"legs; the engine must manage the stop synthetically"
            )
        if not self.idempotency_key:
            raise ValueError(f"{self.symbol}: idempotency_key is required")


@dataclass(frozen=True, slots=True)
class OrderResult:
    """What the broker did with an order."""

    idempotency_key: str
    broker_order_id: str | None
    symbol: Symbol
    side: Side
    status: OrderStatus
    requested_quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    filled_avg_price: Decimal | None = None
    submitted_at: dt.datetime | None = None
    filled_at: dt.datetime | None = None
    rejection_reason: str | None = None
    raw: dict[str, object] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        }


@runtime_checkable
class Broker(Protocol):
    """What the engine needs from a brokerage."""

    name: str
    is_paper: bool

    def is_connected(self) -> bool:
        """Whether the broker is reachable and authenticated."""
        ...

    def get_account(self) -> AccountSnapshot:
        """Current equity, cash (split by settlement) and account flags."""
        ...

    def get_positions(self) -> list[BrokerPosition]:
        """Every open position."""
        ...

    def get_position(self, symbol: Symbol | str) -> BrokerPosition | None:
        """One position, or ``None`` if flat."""
        ...

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Place an order. Must be idempotent on ``request.idempotency_key``."""
        ...

    def get_order(self, broker_order_id: str) -> OrderResult | None:
        """Current state of a previously submitted order."""
        ...

    def list_open_orders(self) -> list[OrderResult]:
        """Every order not yet in a terminal state."""
        ...

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel one order. Returns whether it was cancelled."""
        ...

    def cancel_all_orders(self) -> int:
        """Cancel every open order. Returns how many. Used by the kill switch."""
        ...

    def get_last_price(self, symbol: Symbol | str) -> Decimal | None:
        """Latest trade price, for marking positions and sizing orders."""
        ...

    def supports_fractional(self, symbol: Symbol | str) -> bool:
        """Whether this symbol can be traded fractionally."""
        ...
