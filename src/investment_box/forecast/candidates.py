"""Candidate ranking.

A candidate is a symbol the system could act on, with everything needed to
decide whether to: the forecast, the compliance status, the liquidity picture
and the strategy's record.

Ranking is intentionally conservative. A candidate is only ``ACTIONABLE`` when
every gate passes -- compliant, verified, forecast actionable, confidence above
none. Anything else is surfaced with the specific reason it fell short, because
"why isn't it trading X?" is a question the dashboard has to answer.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

from investment_box.core.types import ComplianceStatus
from investment_box.forecast.base import Confidence, Forecast
from investment_box.shariah.status import ComplianceRecord
from investment_box.universe.builder import UniverseEntry


class CandidateStatus(StrEnum):
    """Whether the system may act on this candidate."""

    ACTIONABLE = "actionable"
    #: Passes compliance but the forecast is too weak or unproven.
    WATCH = "watch"
    #: Blocked by compliance, liquidity or your rules.
    BLOCKED = "blocked"
    #: Needs a human decision before it could ever trade.
    NEEDS_APPROVAL = "needs_approval"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked candidate, ready to render."""

    symbol: str
    as_of: dt.date
    status: CandidateStatus
    reason: str
    forecast: Forecast | None = None
    compliance: ComplianceRecord | None = None
    universe_entry: UniverseEntry | None = None
    rank: int | None = None
    last_price: float | None = None

    @property
    def probability(self) -> float | None:
        return self.forecast.direction_probability if self.forecast else None

    @property
    def confidence(self) -> Confidence:
        return self.forecast.confidence if self.forecast else Confidence.NONE

    @property
    def score(self) -> float:
        """Ranking score: edge weighted by how much the evidence supports it.

        Multiplying by a confidence weight rather than sorting by probability
        alone stops an unproven strategy's 70% outranking a well-evidenced
        strategy's 58%.
        """
        if self.forecast is None:
            return 0.0
        weight = {
            Confidence.HIGH: 1.0,
            Confidence.MEDIUM: 0.6,
            Confidence.LOW: 0.3,
            Confidence.NONE: 0.0,
        }[self.forecast.confidence]
        return self.forecast.edge * weight

    @property
    def is_tradable(self) -> bool:
        return self.status is CandidateStatus.ACTIONABLE

    def explain(self) -> str:
        lines = [f"{self.symbol}: {self.status.value.upper()} — {self.reason}"]
        if self.forecast is not None:
            lines.append("  " + self.forecast.honest_summary().replace("\n", "\n  "))
        if self.compliance is not None:
            lines.append(
                f"  Compliance: {self.compliance.display_status.value} "
                f"({self.compliance.source}, {self.compliance.age_days}d old)"
            )
        return "\n".join(lines)


def rank_candidates(candidates: list[Candidate]) -> list[Candidate]:
    """Sort by status then score, and assign ranks.

    Actionable candidates always outrank watch-list ones regardless of score:
    a high score on something that cannot be traded is not a better candidate,
    it is a more frustrating one.
    """
    order = {
        CandidateStatus.ACTIONABLE: 0,
        CandidateStatus.NEEDS_APPROVAL: 1,
        CandidateStatus.WATCH: 2,
        CandidateStatus.BLOCKED: 3,
    }
    ranked = sorted(candidates, key=lambda c: (order[c.status], -c.score, c.symbol))
    return [
        Candidate(
            symbol=c.symbol,
            as_of=c.as_of,
            status=c.status,
            reason=c.reason,
            forecast=c.forecast,
            compliance=c.compliance,
            universe_entry=c.universe_entry,
            rank=index + 1,
            last_price=c.last_price,
        )
        for index, c in enumerate(ranked)
    ]


def classify(
    symbol: str,
    as_of: dt.date,
    *,
    forecast: Forecast | None,
    compliance: ComplianceRecord | None,
    universe_entry: UniverseEntry | None,
    last_price: float | None = None,
) -> Candidate:
    """Decide a candidate's status, and say specifically why.

    The order matters: hard blocks first, then compliance, then the forecast.
    A user asking why a symbol did not trade should get the *first* reason it
    failed, not the last.
    """

    def make(status: CandidateStatus, reason: str) -> Candidate:
        return Candidate(
            symbol=symbol,
            as_of=as_of,
            status=status,
            reason=reason,
            forecast=forecast,
            compliance=compliance,
            universe_entry=universe_entry,
            last_price=last_price,
        )

    if universe_entry is not None and not universe_entry.included:
        return make(CandidateStatus.BLOCKED, universe_entry.reason)

    if compliance is not None:
        status = compliance.display_status
        if status is ComplianceStatus.NON_COMPLIANT:
            return make(CandidateStatus.BLOCKED, f"not compliant: {compliance.reason}")
        if status in (ComplianceStatus.DOUBTFUL, ComplianceStatus.UNKNOWN):
            return make(
                CandidateStatus.NEEDS_APPROVAL,
                f"compliance is {status.value}: needs a human decision before trading",
            )

    if forecast is None:
        return make(CandidateStatus.WATCH, "no forecast: the strategy produced no signal")

    if forecast.confidence is Confidence.NONE:
        detail = forecast.caveats[0] if forecast.caveats else "insufficient evidence"
        return make(CandidateStatus.WATCH, f"forecast not actionable: {detail}")

    if not forecast.is_actionable:
        return make(
            CandidateStatus.WATCH,
            f"forecast is {forecast.direction_probability:.0%}, at or below a coin flip",
        )

    return make(
        CandidateStatus.ACTIONABLE,
        f"{forecast.direction_probability:.0%} over {forecast.horizon_days}d, "
        f"{forecast.confidence.value} confidence",
    )
