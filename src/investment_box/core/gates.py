"""Narrow protocols that lower layers depend on.

``execution`` must not import ``engine``: the kill switch lives in ``engine``
and imports the order manager, so a direct dependency the other way is a cycle.
The order manager depends on this protocol instead, and
:class:`investment_box.engine.live_guard.LiveGuard` satisfies it structurally.

Same pattern, and same reason, as :mod:`investment_box.core.audit`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class OrderGate(Protocol):
    """A final, stateful check run immediately before an order is submitted.

    Implementations must raise to refuse. Returning a boolean would invite a
    caller to ignore it.
    """

    def assert_order_permitted(
        self,
        *,
        broker: object | None = None,
        data_provider_name: str | None = None,
        compliance_provider_name: str | None = None,
        held_symbols: list[str] | None = None,
    ) -> None:
        """Raise if this order must not proceed.

        The parameters mirror what a gate may need to re-check: the broker's
        current account state, which data and screening sources are in use, and
        what is currently held. All optional, because a gate that cannot
        evaluate a condition must fail rather than demand the input.
        """
        ...
