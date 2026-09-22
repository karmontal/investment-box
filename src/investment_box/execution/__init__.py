"""Broker adapters and order management.

Phase 1 ships the :class:`Broker` protocol and an in-memory mock so that the
service layer, the dashboard and the Telegram bot can be built and tested with
no broker account. The Alpaca adapter and the order manager arrive in Phase 5.
"""

from investment_box.execution.base import (
    AccountSnapshot,
    Broker,
    BrokerPosition,
    OrderRequest,
    OrderResult,
)
from investment_box.execution.mock_broker import MockBroker

__all__ = [
    "AccountSnapshot",
    "Broker",
    "BrokerPosition",
    "MockBroker",
    "OrderRequest",
    "OrderResult",
]
