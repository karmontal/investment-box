"""Alpaca broker adapter.

Implements the same narrow :class:`~investment_box.execution.base.Broker`
protocol as the mock, so the engine cannot tell them apart. What it adds over
the mock is everything that makes a real broker different: asynchronous fills,
rejections, partial fills, and a buying-power figure that must not be trusted.

Three deliberate refusals:

* **It verifies the account is a cash account on connect and refuses to
  proceed otherwise.** A margin account would let a bug borrow.
* **It never reads ``buying_power`` for sizing.** Alpaca reports a figure that
  can include unsettled proceeds; the settled-cash ledger is authoritative.
* **It will not construct itself against a live endpoint unless the caller
  explicitly asks.** Paper is the default at every layer.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from investment_box.core.errors import BrokerError
from investment_box.core.logging import get_logger
from investment_box.core.types import (
    CashLedger,
    OrderStatus,
    OrderType,
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

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

#: Alpaca order states mapped to ours. Anything unrecognised becomes PENDING
#: rather than being guessed at -- an unknown state is not a terminal one.
_STATUS_MAP = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.EXPIRED,
    "canceled": OrderStatus.CANCELLED,
    "cancelled": OrderStatus.CANCELLED,
    "expired": OrderStatus.EXPIRED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.PENDING,
    "pending_cancel": OrderStatus.SUBMITTED,
    "pending_replace": OrderStatus.SUBMITTED,
    "stopped": OrderStatus.FILLED,
    "calculated": OrderStatus.SUBMITTED,
}


class AlpacaBroker:
    """Live adapter over ``alpaca-py``."""

    name = "alpaca"

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        *,
        paper: bool = True,
        base_url: str | None = None,
    ) -> None:
        if not api_key or not secret_key:
            raise BrokerError("Alpaca requires both an API key and a secret key")

        self.is_paper = paper
        self.base_url = base_url or (PAPER_URL if paper else LIVE_URL)

        if not paper and "paper" in self.base_url:
            raise BrokerError(
                "paper=False was requested but the base URL points at the paper "
                "endpoint. Refusing to start with contradictory settings."
            )
        if paper and "paper" not in self.base_url:
            raise BrokerError(
                "paper=True was requested but the base URL points at the live "
                "endpoint. Refusing to start with contradictory settings."
            )

        self._api_key = api_key
        self._secret_key = secret_key
        self._client: Any = None
        self._verified_cash_account = False

    # ------------------------------------------------------------- lifecycle

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as exc:
                raise BrokerError(
                    "alpaca-py is not installed. Run: uv sync --extra broker"
                ) from exc
            self._client = TradingClient(
                api_key=self._api_key, secret_key=self._secret_key, paper=self.is_paper
            )
        return self._client

    def is_connected(self) -> bool:
        try:
            self.client.get_account()
        except Exception as exc:  # noqa: BLE001 - connectivity probe
            log.warning("alpaca.not_connected", error=str(exc))
            return False
        return True

    def verify_cash_account(self) -> None:
        """Refuse to operate on a margin account.

        Raises:
            BrokerError: If the account can use margin. This is checked once on
                connect rather than per order, but it is checked before any
                order can be placed.
        """
        account = self.client.get_account()
        multiplier = float(getattr(account, "multiplier", 1) or 1)
        if multiplier > 1:
            raise BrokerError(
                f"This Alpaca account has a margin multiplier of {multiplier}. "
                f"Investment Box requires a CASH account: margin is never permitted. "
                f"Change the account type before running."
            )
        if getattr(account, "shorting_enabled", False):
            raise BrokerError(
                "This Alpaca account has shorting enabled. Investment Box never "
                "shorts; disable it before running."
            )
        self._verified_cash_account = True
        log.info("alpaca.cash_account_verified", multiplier=multiplier)

    # --------------------------------------------------------------- account

    def get_account(self) -> AccountSnapshot:
        try:
            account = self.client.get_account()
        except Exception as exc:
            raise BrokerError(f"could not fetch the Alpaca account: {exc}") from exc

        cash = to_money(account.cash or 0)
        # Alpaca does not expose a settled/unsettled split directly. The
        # settlement ledger owns that distinction; here everything is reported
        # as settled and the ledger subtracts what it knows is pending.
        ledger = CashLedger(settled=cash, unsettled=Decimal("0.00"))
        multiplier = float(getattr(account, "multiplier", 1) or 1)

        return AccountSnapshot(
            equity=to_money(account.equity or 0),
            cash=ledger,
            positions_value=to_money(
                Decimal(str(account.long_market_value or 0))
            ),
            # Deliberately the cash figure, NOT account.buying_power, which can
            # include unsettled proceeds.
            buying_power=cash,
            is_cash_account=multiplier <= 1,
            trading_blocked=bool(getattr(account, "trading_blocked", False)),
            taken_at=dt.datetime.now(tz=dt.UTC),
        )

    def get_positions(self) -> list[BrokerPosition]:
        try:
            raw = self.client.get_all_positions()
        except Exception as exc:
            raise BrokerError(f"could not fetch positions: {exc}") from exc
        return [self._to_position(p) for p in raw]

    def get_position(self, symbol: Symbol | str) -> BrokerPosition | None:
        ticker = str(symbol).upper()
        try:
            return self._to_position(self.client.get_open_position(ticker))
        except Exception:  # noqa: BLE001 - "no position" is reported as an error
            return None

    @staticmethod
    def _to_position(raw: Any) -> BrokerPosition:
        return BrokerPosition(
            symbol=Symbol(str(raw.symbol).upper()),
            quantity=to_qty(raw.qty),
            avg_entry_price=to_money(raw.avg_entry_price),
            current_price=to_money(raw.current_price or raw.avg_entry_price),
            opened_at=None,
        )

    # ---------------------------------------------------------------- orders

    def submit_order(self, request: OrderRequest) -> OrderResult:
        """Place an order.

        Idempotency uses Alpaca's ``client_order_id``: resubmitting the same
        key returns the existing order rather than trading twice. That is the
        only safe behaviour after an ambiguous network timeout.
        """
        if not self._verified_cash_account:
            try:
                self.verify_cash_account()
            except BrokerError as exc:
                # Refuse the order rather than raising. The engine surfaces the
                # reason; raising here would take down whatever called it.
                log.error("alpaca.order_refused_account_type", error=str(exc))
                return OrderResult(
                    idempotency_key=request.idempotency_key,
                    broker_order_id=None,
                    symbol=request.symbol,
                    side=request.side,
                    status=OrderStatus.REJECTED,
                    requested_quantity=request.quantity,
                    submitted_at=dt.datetime.now(tz=dt.UTC),
                    rejection_reason=str(exc),
                )

        existing = self._find_by_client_id(request.idempotency_key)
        if existing is not None:
            log.info("alpaca.duplicate_suppressed", key=request.idempotency_key)
            return existing

        try:
            order = self.client.submit_order(self._build_request(request))
        except Exception as exc:  # noqa: BLE001 - a rejection is a result, not a crash
            log.warning(
                "alpaca.submit_failed", symbol=str(request.symbol), error=str(exc)
            )
            return OrderResult(
                idempotency_key=request.idempotency_key,
                broker_order_id=None,
                symbol=request.symbol,
                side=request.side,
                status=OrderStatus.REJECTED,
                requested_quantity=request.quantity,
                submitted_at=dt.datetime.now(tz=dt.UTC),
                rejection_reason=str(exc),
            )
        return self._to_result(order, request.idempotency_key)

    def _build_request(self, request: OrderRequest) -> Any:
        from alpaca.trading.enums import OrderSide
        from alpaca.trading.enums import TimeInForce as AlpacaTIF
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLossRequest,
            TakeProfitRequest,
        )

        side = OrderSide.BUY if request.side is Side.BUY else OrderSide.SELL
        tif = AlpacaTIF.DAY if request.time_in_force.value == "day" else AlpacaTIF.GTC

        common: dict[str, Any] = {
            "symbol": str(request.symbol).upper(),
            "qty": float(request.quantity),
            "side": side,
            "time_in_force": tif,
            "client_order_id": request.idempotency_key,
        }

        # Bracket legs are only possible on whole-share limit orders. The
        # OrderRequest validator already rejects the fractional combination;
        # this mirrors it at the boundary.
        if not request.is_fractional and (
            request.stop_loss_price or request.take_profit_price
        ):
            common["order_class"] = "bracket"
            if request.take_profit_price:
                common["take_profit"] = TakeProfitRequest(
                    limit_price=float(request.take_profit_price)
                )
            if request.stop_loss_price:
                common["stop_loss"] = StopLossRequest(
                    stop_price=float(request.stop_loss_price)
                )

        if request.order_type is OrderType.LIMIT and request.limit_price is not None:
            return LimitOrderRequest(limit_price=float(request.limit_price), **common)
        return MarketOrderRequest(**common)

    def get_order(self, broker_order_id: str) -> OrderResult | None:
        try:
            order = self.client.get_order_by_id(broker_order_id)
        except Exception:  # noqa: BLE001
            return None
        return self._to_result(order, getattr(order, "client_order_id", "") or "")

    def _find_by_client_id(self, client_order_id: str) -> OrderResult | None:
        try:
            order = self.client.get_order_by_client_id(client_order_id)
        except Exception:  # noqa: BLE001 - not found is reported as an error
            return None
        return self._to_result(order, client_order_id)

    def list_open_orders(self) -> list[OrderResult]:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        try:
            orders = self.client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception as exc:
            raise BrokerError(f"could not list open orders: {exc}") from exc
        return [self._to_result(o, getattr(o, "client_order_id", "") or "") for o in orders]

    def cancel_order(self, broker_order_id: str) -> bool:
        try:
            self.client.cancel_order_by_id(broker_order_id)
        except Exception as exc:  # noqa: BLE001
            log.warning("alpaca.cancel_failed", order_id=broker_order_id, error=str(exc))
            return False
        return True

    def cancel_all_orders(self) -> int:
        """Cancel everything open. The kill switch's first action."""
        try:
            responses = self.client.cancel_orders()
        except Exception as exc:
            raise BrokerError(f"could not cancel orders: {exc}") from exc
        return len(responses or [])

    @staticmethod
    def _to_result(order: Any, idempotency_key: str) -> OrderResult:
        raw_status = str(getattr(order, "status", "")).lower().split(".")[-1]
        status = _STATUS_MAP.get(raw_status, OrderStatus.PENDING)
        if raw_status not in _STATUS_MAP:
            log.warning("alpaca.unknown_order_status", status=raw_status)

        filled_qty = to_qty(getattr(order, "filled_qty", 0) or 0)
        filled_price = getattr(order, "filled_avg_price", None)

        return OrderResult(
            idempotency_key=idempotency_key,
            broker_order_id=str(order.id),
            symbol=Symbol(str(order.symbol).upper()),
            side=Side.BUY if str(order.side).lower().endswith("buy") else Side.SELL,
            status=status,
            requested_quantity=to_qty(getattr(order, "qty", 0) or 0),
            filled_quantity=filled_qty,
            filled_avg_price=to_money(filled_price) if filled_price else None,
            submitted_at=getattr(order, "submitted_at", None),
            filled_at=getattr(order, "filled_at", None),
            rejection_reason=None,
            raw={"status": raw_status},
        )

    # ---------------------------------------------------------------- prices

    def get_last_price(self, symbol: Symbol | str) -> Decimal | None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestTradeRequest

            data = StockHistoricalDataClient(self._api_key, self._secret_key)
            response = data.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=str(symbol).upper())
            )
            trade = response[str(symbol).upper()]
            return to_money(trade.price)
        except Exception as exc:  # noqa: BLE001 - a missing price is not fatal
            log.warning("alpaca.price_failed", symbol=str(symbol), error=str(exc))
            return None

    def supports_fractional(self, symbol: Symbol | str) -> bool:
        try:
            asset = self.client.get_asset(str(symbol).upper())
        except Exception:  # noqa: BLE001
            return False
        return bool(getattr(asset, "fractionable", False))
