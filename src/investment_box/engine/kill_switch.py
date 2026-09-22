"""The kill switch.

Three actions, in a fixed order that matters:

1. **Cancel open orders first.** Cancelling before closing prevents a pending
   buy from filling while positions are being liquidated, which would leave a
   new position behind after a "close everything" instruction.
2. **Optionally close all positions**, at market. Market orders, not limit:
   an emergency exit that does not fill is not an exit.
3. **Kill the engine**, which is terminal until restart.

The switch is written to keep going when a step fails. A partial kill that
reports exactly what it could not do is far more useful than one that aborts
halfway and leaves the operator guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from investment_box.core.audit import AuditSink
from investment_box.core.logging import get_logger
from investment_box.core.types import ComplianceStatus, ExitReason
from investment_box.engine.state import EngineStateMachine
from investment_box.execution.base import Broker
from investment_box.execution.order_manager import OrderManager

log = get_logger(__name__)


@dataclass
class KillResult:
    """What the kill switch managed to do."""

    orders_cancelled: int = 0
    positions_closed: list[str] = field(default_factory=list)
    positions_failed: list[str] = field(default_factory=list)
    engine_killed: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def fully_successful(self) -> bool:
        return self.engine_killed and not self.positions_failed and not self.errors

    def summary(self) -> str:
        parts = [f"{self.orders_cancelled} order(s) cancelled"]
        if self.positions_closed:
            parts.append(f"{len(self.positions_closed)} position(s) closed")
        if self.positions_failed:
            parts.append(f"{len(self.positions_failed)} FAILED TO CLOSE")
        parts.append("engine killed" if self.engine_killed else "ENGINE NOT KILLED")
        return ", ".join(parts)

    def detail(self) -> str:
        lines = [self.summary()]
        if self.positions_failed:
            lines.append(
                "Positions still open — close these manually with your broker: "
                + ", ".join(self.positions_failed)
            )
        lines.extend(f"error: {e}" for e in self.errors)
        return "\n".join(lines)


class KillSwitch:
    """Emergency stop."""

    def __init__(
        self,
        broker: Broker,
        orders: OrderManager,
        state: EngineStateMachine,
        audit: AuditSink,
    ) -> None:
        self.broker = broker
        self.orders = orders
        self.state = state
        self.audit = audit

    def activate(
        self, reason: str, *, close_positions: bool = False, actor: str = "user"
    ) -> KillResult:
        """Pull the switch.

        Args:
            reason: Recorded in the audit log and broadcast.
            close_positions: Liquidate everything at market. Off by default:
                closing positions realises losses and starts the settlement
                clock, which is not always what an operator wants when they
                only need the engine to stop deciding things.
            actor: Who pulled it.
        """
        log.error("kill_switch.activated", reason=reason, actor=actor,
                  close_positions=close_positions)
        self.audit.record(
            "kill_switch.activated",
            f"KILL SWITCH: {reason} (close_positions={close_positions})",
            actor=actor,
        )
        result = KillResult()

        # 1. Orders first, so nothing new fills mid-liquidation.
        try:
            result.orders_cancelled = self.broker.cancel_all_orders()
        except Exception as exc:  # noqa: BLE001 - keep going and report
            log.error("kill_switch.cancel_failed", error=str(exc))
            result.errors.append(f"could not cancel orders: {exc}")

        # 2. Positions, if asked.
        if close_positions:
            self._close_all(result)

        # 3. The engine itself. Done last so a failure above is still recorded
        #    against a killed engine rather than a running one.
        try:
            self.state.kill(reason, actor=actor)
            result.engine_killed = True
        except Exception as exc:  # noqa: BLE001
            log.error("kill_switch.state_failed", error=str(exc))
            result.errors.append(f"could not set the engine state: {exc}")

        self.audit.record(
            "kill_switch.completed", result.detail(), actor=actor,
            detail={"orders_cancelled": result.orders_cancelled,
                    "closed": result.positions_closed,
                    "failed": result.positions_failed},
        )
        return result

    def _close_all(self, result: KillResult) -> None:
        try:
            positions = self.broker.get_positions()
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"could not list positions: {exc}")
            return

        for position in positions:
            symbol = str(position.symbol)
            try:
                placed = self.orders.close_position(
                    symbol=symbol,
                    quantity=position.quantity,
                    # A kill switch must not be blocked by a compliance status.
                    # Selling is always permitted, whatever the status says.
                    compliance_status=ComplianceStatus.COMPLIANT,
                    reason=ExitReason.KILL_SWITCH.value,
                    nonce="kill",
                )
            except Exception as exc:  # noqa: BLE001
                result.positions_failed.append(symbol)
                result.errors.append(f"{symbol}: {exc}")
                continue

            if placed.accepted:
                result.positions_closed.append(symbol)
            else:
                result.positions_failed.append(symbol)
                result.errors.append(f"{symbol}: {placed.reason}")

    def estimate_exposure(self) -> Decimal:
        """Total market value at risk, for the confirmation prompt."""
        try:
            return Decimal(
                sum((p.market_value for p in self.broker.get_positions()), Decimal("0"))
            )
        except Exception:  # noqa: BLE001
            return Decimal("0")
