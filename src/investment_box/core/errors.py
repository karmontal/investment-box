"""Exception hierarchy.

Every error carries enough context to land in the audit log without the caller
having to reconstruct what went wrong.
"""

from __future__ import annotations


class InvestmentBoxError(Exception):
    """Base class for every error raised by this application."""


class ConfigError(InvestmentBoxError):
    """Configuration is missing, malformed, or internally inconsistent."""


class DataError(InvestmentBoxError):
    """Market data could not be fetched, or failed validation."""


class DataQualityError(DataError):
    """Data was fetched but is not fit to trade on."""

    def __init__(self, symbol: str, reason: str) -> None:
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"{symbol}: {reason}")


class BrokerError(InvestmentBoxError):
    """The broker rejected a request, or could not be reached."""


class ComplianceError(InvestmentBoxError):
    """A Shariah compliance rule was violated.

    Raised by the hard-constraint guards. These are never caught and retried --
    they mean the engine tried to do something it must never do, which is a bug.
    """


class RiskLimitError(InvestmentBoxError):
    """A risk limit would be breached by the proposed action."""


class SettlementError(RiskLimitError):
    """The action would spend unsettled cash (a good-faith violation)."""
