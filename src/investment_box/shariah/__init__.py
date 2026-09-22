"""Shariah compliance layer.

``constraints`` holds the hard, unconfigurable trading rules and is enforced at
the order boundary. The rest is screening: providers supply evidence,
:class:`ComplianceTracker` persists it append-only and answers status
questions, and purification and zakat compute what is owed.
"""

from investment_box.shariah.constraints import (
    HardConstraints,
    assert_order_permissible,
    is_forbidden_instrument,
)
from investment_box.shariah.providers.base import (
    FinancialRatios,
    ScreeningProvider,
    ScreenResult,
)
from investment_box.shariah.status import ComplianceRecord, ComplianceTracker

__all__ = [
    "ComplianceRecord",
    "ComplianceTracker",
    "FinancialRatios",
    "HardConstraints",
    "ScreenResult",
    "ScreeningProvider",
    "assert_order_permissible",
    "is_forbidden_instrument",
]
