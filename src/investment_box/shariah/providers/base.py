"""The screening provider contract.

A provider answers one question: is this symbol permissible to trade, and on
what evidence? The evidence matters as much as the answer -- a screen with no
recorded ratios, source or date cannot be audited later, and "we thought it was
compliant" is not an acceptable answer about a trade you have already made.

Providers never decide *policy*. Thresholds come from config; the provider
supplies the numbers and the business-activity findings, and
:class:`ScreenResult` records both what was measured and what it was measured
against.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from investment_box.core.types import ComplianceStatus

#: Business activities that make a company impermissible regardless of ratios.
#: The weapons category is configurable in some interpretations; it is included
#: by default and can be removed from ``ShariahConfig.excluded_activities``.
BUSINESS_ACTIVITIES: tuple[str, ...] = (
    "conventional_banking",
    "conventional_insurance",
    "interest_based_finance",
    "alcohol",
    "pork",
    "gambling",
    "adult_entertainment",
    "tobacco",
    "weapons",
)


@dataclass(frozen=True, slots=True)
class FinancialRatios:
    """The AAOIFI-style ratios, with the denominator they were computed against.

    A ratio without its denominator is meaningless: 25% of market cap and 25%
    of total assets are different claims, and different boards use different
    ones. ``denominator`` is therefore not optional.
    """

    denominator: str
    denominator_value: float | None = None
    debt_ratio: float | None = None
    interest_securities_ratio: float | None = None
    non_permissible_revenue_ratio: float | None = None

    @property
    def is_complete(self) -> bool:
        """Whether every ratio needed for a verdict is present.

        An incomplete set yields UNKNOWN, never a pass. Missing data is not
        evidence of compliance.
        """
        return all(
            value is not None
            for value in (
                self.debt_ratio,
                self.interest_securities_ratio,
                self.non_permissible_revenue_ratio,
            )
        )

    def breaches(self, thresholds: dict[str, float]) -> list[str]:
        """Which ratios meet or exceed their limit, described for a human.

        A missing ratio is not a breach -- it is an unknown, handled by the
        caller. The comparison is ``>=`` because AAOIFI thresholds are stated
        as strict upper bounds ("less than 30%").
        """
        checks = (
            ("debt", self.debt_ratio, thresholds.get("debt", 1.0)),
            (
                "interest-bearing securities",
                self.interest_securities_ratio,
                thresholds.get("interest_securities", 1.0),
            ),
            (
                "non-permissible revenue",
                self.non_permissible_revenue_ratio,
                thresholds.get("revenue", 1.0),
            ),
        )
        return [
            f"{name} {value:.1%} >= {limit:.0%}"
            for name, value, limit in checks
            if value is not None and value >= limit
        ]


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """One screening outcome, with everything needed to audit it later."""

    symbol: str
    status: ComplianceStatus
    source: str
    screened_at: dt.datetime
    ratios: FinancialRatios | None = None
    #: Impermissible business activities found, from :data:`BUSINESS_ACTIVITIES`.
    activity_flags: tuple[str, ...] = ()
    #: Why this verdict, in a sentence a human can read in an audit log.
    reason: str = ""
    #: Provider payload, kept verbatim for later dispute.
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_tradable(self) -> bool:
        """Only COMPLIANT is auto-tradable. DOUBTFUL and UNKNOWN are not."""
        return self.status.auto_tradable

    def failing_ratios(self, thresholds: dict[str, float]) -> list[str]:
        """Which ratios breach their thresholds, for the explanation."""
        return self.ratios.breaches(thresholds) if self.ratios else []


@runtime_checkable
class ScreeningProvider(Protocol):
    """Source of compliance verdicts."""

    name: str
    #: True when verdicts come from a recognised Shariah board rather than
    #: being computed locally. Recorded on every trade, because a computed
    #: screen is an estimate and a certified one is a ruling.
    is_certified_source: bool

    def is_available(self) -> bool:
        """Whether the provider can currently answer."""
        ...

    def screen(self, symbol: str, *, as_of: dt.date | None = None) -> ScreenResult:
        """Screen one symbol.

        Must return a result rather than raise on missing data: an
        ``UNKNOWN`` verdict is information, and the caller decides what to do
        with it. Raise only on transport failure.
        """
        ...

    def screen_many(
        self, symbols: list[str], *, as_of: dt.date | None = None
    ) -> dict[str, ScreenResult]:
        """Screen several symbols. One failure must not fail the batch."""
        ...


def unknown_result(symbol: str, source: str, reason: str, now: dt.datetime) -> ScreenResult:
    """Build an UNKNOWN verdict. The safe default whenever evidence is missing."""
    return ScreenResult(
        symbol=symbol.upper(),
        status=ComplianceStatus.UNKNOWN,
        source=source,
        screened_at=now,
        reason=reason,
    )
