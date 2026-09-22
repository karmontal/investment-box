"""Order management: the only path from a decision to a broker.

Responsibilities, in the order they matter:

1. **Idempotency.** Every order carries a deterministic key derived from
   (symbol, side, intent, date). The key is written to the database under a
   unique constraint *before* the broker is called, so a retry after an
   ambiguous timeout cannot place a second order.
2. **The Shariah gate.** ``assert_order_permissible`` runs on the fully formed
   order -- final quantity, final price -- immediately before submission. Not
   on the proposal, which may differ.
3. **The hybrid execution mode.** Whole shares with a limit order and bracket
   legs where possible; a fractional market order with an engine-managed stop
   only where the sized position cannot afford one whole share, and only when
   that risk has been explicitly acknowledged.
4. **Reconciliation.** Broker state is authoritative. On restart, local orders
   are re-read from the broker rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from investment_box.config.schema import Settings
from investment_box.core.audit import AuditSink
from investment_box.core.clock import UTC, Clock, SystemClock
from investment_box.core.errors import ComplianceError
from investment_box.core.gates import OrderGate
from investment_box.core.logging import get_logger
from investment_box.core.types import (
    ComplianceStatus,
    OrderStatus,
    OrderType,
    Side,
    Symbol,
    TimeInForce,
    to_money,
)
from investment_box.db.models import Order
from investment_box.db.session import Database
from investment_box.execution.base import Broker, OrderRequest, OrderResult
from investment_box.risk.sizing import PositionSize
from investment_box.shariah.constraints import assert_order_permissible

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PlacedOrder:
    """The outcome of trying to place an order."""

    request: OrderRequest | None
    result: OrderResult | None
    accepted: bool
    reason: str
    #: True when the stop is held by the engine rather than the broker.
    synthetic_stop: bool = False

    @property
    def is_filled(self) -> bool:
        return self.result is not None and self.result.is_filled


def make_idempotency_key(
    *, symbol: str, side: Side, intent: str, day: dt.date, nonce: str = ""
) -> str:
    """A deterministic, collision-resistant order key.

    Deterministic so a retry produces the same key; hashed so it fits inside
    Alpaca's ``client_order_id`` limit while staying unique. ``nonce`` lets a
    deliberate second trade in the same symbol on the same day proceed.
    """
    raw = f"{symbol.upper()}|{side.value}|{intent}|{day.isoformat()}|{nonce}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return f"ib-{symbol.upper()}-{day:%Y%m%d}-{digest}"


class OrderManager:
    """Places, records and reconciles orders."""

    def __init__(
        self,
        broker: Broker,
        database: Database,
        settings: Settings,
        audit: AuditSink,
        *,
        clock: Clock | None = None,
        live_guard: OrderGate | None = None,
    ) -> None:
        self.broker = broker
        self.db = database
        self.settings = settings
        self.audit = audit
        self.clock = clock or SystemClock()
        # Consulted before every LIVE order. None in paper mode, where there is
        # nothing to guard.
        self.live_guard = live_guard

    # ------------------------------------------------------------------ open

    def open_position(
        self,
        *,
        size: PositionSize,
        compliance_status: ComplianceStatus,
        compliance_source: str,
        reference_price: Decimal,
        current_position_qty: Decimal = Decimal("0"),
        available_cash: Decimal,
        instrument_name: str | None = None,
        human_approved: bool = False,
        nonce: str = "",
    ) -> PlacedOrder:
        """Buy, honouring the hybrid execution mode and every hard constraint."""
        today = self.clock.now().astimezone(UTC).date()
        key = make_idempotency_key(
            symbol=size.symbol, side=Side.BUY, intent="open", day=today, nonce=nonce
        )

        if self._already_placed(key):
            log.info("order.duplicate_suppressed", key=key)
            return PlacedOrder(
                request=None, result=None, accepted=False,
                reason="an identical order was already placed today",
            )

        request = self._build_open_request(size, reference_price, key)
        if request is None:
            return PlacedOrder(
                request=None, result=None, accepted=False,
                reason=(
                    "position is fractional but fractional trading is disabled "
                    "(set execution.acknowledge_fractional_stop_risk to enable it)"
                ),
            )

        # The Shariah gate, on the final order. Not on the proposal.
        try:
            assert_order_permissible(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                position_quantity=current_position_qty,
                compliance_status=compliance_status,
                cash_available=available_cash,
                order_notional=size.notional,
                instrument_name=instrument_name,
                human_approved=human_approved,
            )
        except ComplianceError as exc:
            self.audit.record(
                "order.refused_compliance", str(exc), actor="order_manager",
                symbol=size.symbol,
            )
            log.error("order.compliance_refusal", symbol=size.symbol, error=str(exc))
            return PlacedOrder(
                request=request, result=None, accepted=False, reason=str(exc)
            )

        return self._submit(request, size, compliance_status, compliance_source)

    def _build_open_request(
        self, size: PositionSize, reference_price: Decimal, key: str
    ) -> OrderRequest | None:
        """Choose the order shape for this position.

        Whole share: marketable limit plus broker-side bracket legs.
        Fractional: market order, no legs, engine-managed stop -- and only if
        the risk of an unprotected position has been acknowledged.
        """
        if size.is_fractional:
            if not self.settings.execution.fractional_enabled:
                return None
            return OrderRequest(
                symbol=Symbol(size.symbol),
                side=Side.BUY,
                quantity=size.quantity,
                order_type=OrderType.MARKET,
                idempotency_key=key,
                is_fractional=True,
                time_in_force=TimeInForce.DAY,
            )

        offset = Decimal(str(self.settings.execution.limit_offset_pct))
        limit = to_money(reference_price * (Decimal("1") + offset))

        return OrderRequest(
            symbol=Symbol(size.symbol),
            side=Side.BUY,
            quantity=size.quantity,
            order_type=OrderType.LIMIT,
            limit_price=limit,
            stop_loss_price=size.stop_price,
            take_profit_price=size.take_profit_price,
            idempotency_key=key,
            time_in_force=TimeInForce.DAY,
        )

    # ------------------------------------------------------------------ close

    def close_position(
        self,
        *,
        symbol: str,
        quantity: Decimal,
        compliance_status: ComplianceStatus,
        reason: str,
        reference_price: Decimal | None = None,
        instrument_name: str | None = None,
        nonce: str = "",
    ) -> PlacedOrder:
        """Sell an existing position.

        Exits are never blocked by a paused engine or a loss limit. Being
        unable to close a losing position because a loss limit was hit would be
        the worst possible behaviour.
        """
        today = self.clock.now().astimezone(UTC).date()
        key = make_idempotency_key(
            symbol=symbol, side=Side.SELL, intent=reason, day=today, nonce=nonce
        )

        if self._already_placed(key):
            return PlacedOrder(
                request=None, result=None, accepted=False,
                reason="an identical exit was already placed today",
            )

        held = self.broker.get_position(symbol)
        held_qty = held.quantity if held else Decimal("0")
        if quantity > held_qty:
            return PlacedOrder(
                request=None, result=None, accepted=False,
                reason=(
                    f"cannot sell {quantity} of {symbol}: only {held_qty} is held. "
                    f"Selling more would open a short position."
                ),
            )

        is_fractional = quantity != quantity.to_integral_value()
        request = OrderRequest(
            symbol=Symbol(symbol),
            side=Side.SELL,
            quantity=quantity,
            # Exits are market orders: a limit exit that does not fill leaves a
            # position the engine believes is closed.
            order_type=OrderType.MARKET,
            idempotency_key=key,
            is_fractional=is_fractional,
            time_in_force=TimeInForce.DAY,
        )

        try:
            assert_order_permissible(
                symbol=request.symbol,
                side=Side.SELL,
                quantity=quantity,
                position_quantity=held_qty,
                compliance_status=compliance_status,
                cash_available=Decimal("0"),
                order_notional=Decimal("0"),
                instrument_name=instrument_name,
            )
        except ComplianceError as exc:
            self.audit.record(
                "order.refused_compliance", str(exc), actor="order_manager", symbol=symbol
            )
            return PlacedOrder(
                request=request, result=None, accepted=False, reason=str(exc)
            )

        return self._submit(request, None, compliance_status, "exit")

    # ------------------------------------------------------------- submission

    def _submit(
        self,
        request: OrderRequest,
        size: PositionSize | None,
        compliance_status: ComplianceStatus,
        compliance_source: str,
    ) -> PlacedOrder:
        """Record the order, then place it.

        Recording first is what makes idempotency real: the unique constraint
        on the key rejects a duplicate before the broker is ever called.
        """
        synthetic_stop = bool(request.is_fractional and size and size.stop_price)

        # Re-check the critical live conditions. Passing at activation does not
        # mean passing now: an account can switch to margin, a screen can go
        # stale, a kill flag can be raised from another process.
        if self.live_guard is not None:
            try:
                self.live_guard.assert_order_permitted(broker=self.broker)
            except Exception as exc:  # noqa: BLE001 - refuse, do not propagate
                log.error("order.live_guard_refused", error=str(exc))
                return PlacedOrder(
                    request=request, result=None, accepted=False, reason=str(exc)
                )

        try:
            self._record(request, size, synthetic_stop)
        except IntegrityError:
            log.info("order.duplicate_key", key=request.idempotency_key)
            return PlacedOrder(
                request=request, result=None, accepted=False,
                reason="duplicate order key: this order was already recorded",
            )

        result = self.broker.submit_order(request)
        self._update(result)

        if result.status is OrderStatus.REJECTED:
            self.audit.record(
                "order.rejected",
                f"{request.symbol} {request.side.value} {request.quantity}: "
                f"{result.rejection_reason}",
                actor="broker",
                symbol=str(request.symbol),
            )
            return PlacedOrder(
                request=request, result=result, accepted=False,
                reason=result.rejection_reason or "broker rejected the order",
                synthetic_stop=synthetic_stop,
            )

        self.audit.record(
            "order.submitted",
            f"{request.symbol} {request.side.value} {request.quantity} "
            f"({request.order_type.value})"
            + (" [engine-managed stop]" if synthetic_stop else ""),
            actor="order_manager",
            symbol=str(request.symbol),
            detail={
                "idempotency_key": request.idempotency_key,
                "broker_order_id": result.broker_order_id,
                "compliance_status": compliance_status.value,
                "compliance_source": compliance_source,
            },
        )
        return PlacedOrder(
            request=request, result=result, accepted=True,
            reason="submitted", synthetic_stop=synthetic_stop,
        )

    # ------------------------------------------------------------ persistence

    def _already_placed(self, key: str) -> bool:
        with self.db.session() as session:
            return session.scalar(
                select(Order.id).where(Order.idempotency_key == key).limit(1)
            ) is not None

    def _record(
        self, request: OrderRequest, size: PositionSize | None, synthetic_stop: bool
    ) -> None:
        with self.db.session() as session:
            session.add(
                Order(
                    idempotency_key=request.idempotency_key,
                    symbol=str(request.symbol),
                    side=request.side.value,
                    order_type=request.order_type.value,
                    quantity=request.quantity,
                    is_fractional=request.is_fractional,
                    limit_price=request.limit_price,
                    stop_price=size.stop_price if size else request.stop_loss_price,
                    take_profit_price=(
                        size.take_profit_price if size else request.take_profit_price
                    ),
                    stop_is_synthetic=synthetic_stop,
                    status=OrderStatus.PENDING.value,
                    submitted_at=self.clock.now(),
                    trading_mode=self.settings.trading_mode.value,
                )
            )

    def _update(self, result: OrderResult) -> None:
        with self.db.session() as session:
            row = session.scalar(
                select(Order).where(Order.idempotency_key == result.idempotency_key)
            )
            if row is None:
                return
            row.broker_order_id = result.broker_order_id
            row.status = result.status.value
            row.filled_quantity = result.filled_quantity
            row.filled_avg_price = result.filled_avg_price
            row.filled_at = result.filled_at
            row.rejection_reason = result.rejection_reason

    # --------------------------------------------------------- reconciliation

    def reconcile(self) -> list[OrderResult]:
        """Re-read non-terminal orders from the broker.

        Broker state wins. Local rows can be stale after a crash, a restart, or
        a fill that arrived while the process was down -- and acting on a stale
        belief about an open order is how a position gets doubled.
        """
        updated: list[OrderResult] = []
        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(Order).where(
                        Order.status.in_(
                            [
                                OrderStatus.PENDING.value,
                                OrderStatus.SUBMITTED.value,
                                OrderStatus.PARTIALLY_FILLED.value,
                            ]
                        ),
                        Order.trading_mode == self.settings.trading_mode.value,
                    )
                ).all()
            )
            pending = [(r.id, r.broker_order_id, r.idempotency_key) for r in rows]

        for _row_id, broker_id, key in pending:
            if not broker_id:
                continue
            result = self.broker.get_order(broker_id)
            if result is None:
                log.warning("order.vanished_at_broker", broker_order_id=broker_id, key=key)
                continue
            self._update(result)
            updated.append(result)

        if updated:
            log.info("order.reconciled", count=len(updated))
        return updated

    def cancel_stale_orders(self, *, older_than_minutes: int | None = None) -> int:
        """Cancel unfilled orders past the configured timeout."""
        timeout = older_than_minutes or self.settings.execution.order_timeout_minutes
        cutoff = self.clock.now() - dt.timedelta(minutes=timeout)
        cancelled = 0

        for order in self.broker.list_open_orders():
            submitted = order.submitted_at
            if submitted is None or submitted > cutoff:
                continue
            if order.broker_order_id and self.broker.cancel_order(order.broker_order_id):
                cancelled += 1
                self.audit.record(
                    "order.cancelled_stale",
                    f"{order.symbol}: unfilled after {timeout} minutes",
                    actor="order_manager",
                    symbol=str(order.symbol),
                )
        return cancelled
