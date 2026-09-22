"""The service layer.

Every consumer -- the Streamlit dashboard, the Telegram bot, the scheduler,
scripts -- reads and writes state through here. Nothing else talks to the
broker or the database directly.

That is not tidiness for its own sake. It is what makes "the bot reads state
through the same service layer as the dashboard" an enforceable property: if
``/balance`` in Telegram and the equity card in the dashboard call the same
method, they cannot disagree.
"""

from investment_box.services.approvals import ApprovalRequest, ApprovalService
from investment_box.services.audit import AuditService
from investment_box.services.container import ServiceContainer, build_services

# ResearchService is imported lazily: it depends on forecast -> shariah, and
# shariah's tracker is constructed by callers that import services first. A
# module-level import here reintroduces a cycle.
from investment_box.services.portfolio import (
    AccountView,
    CapitalUsage,
    PortfolioService,
    PositionView,
)


def __getattr__(name: str) -> object:
    """Lazily expose ResearchService without creating an import cycle."""
    if name in ("ResearchService", "ResearchSnapshot"):
        from investment_box.services import research

        return getattr(research, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AccountView",
    "ApprovalRequest",
    "ApprovalService",
    "AuditService",
    "CapitalUsage",
    "PortfolioService",
    "PositionView",
    "ResearchService",
    "ResearchSnapshot",
    "ServiceContainer",
    "build_services",
]
