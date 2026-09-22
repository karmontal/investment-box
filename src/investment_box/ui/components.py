"""Shared rendering helpers.

The design rule for this dashboard: **a number never appears without the
context needed to judge it.** A probability is shown with its confidence and
sample size, a price with its timestamp, a backtest figure with its caveats.
Anything that would be misleading alone gets its qualifier rendered next to it,
not in a tooltip.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pandas as pd
import streamlit as st

from investment_box.core.clock import to_display
from investment_box.core.types import ComplianceStatus, TradingMode
from investment_box.forecast.base import Confidence
from investment_box.forecast.candidates import Candidate, CandidateStatus

STATUS_ICON = {
    CandidateStatus.ACTIONABLE: "🟢",
    CandidateStatus.NEEDS_APPROVAL: "🟡",
    CandidateStatus.WATCH: "⚪",
    CandidateStatus.BLOCKED: "🔴",
}

COMPLIANCE_ICON = {
    ComplianceStatus.COMPLIANT: "✅",
    ComplianceStatus.NON_COMPLIANT: "❌",
    ComplianceStatus.DOUBTFUL: "⚠️",
    ComplianceStatus.UNKNOWN: "❓",
}

CONFIDENCE_ICON = {
    Confidence.HIGH: "●●●",
    Confidence.MEDIUM: "●●○",
    Confidence.LOW: "●○○",
    Confidence.NONE: "○○○",
}


def money(value: Decimal | float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    amount = float(value)
    sign = "+" if signed and amount > 0 else ""
    return f"{sign}${amount:,.2f}"


def pct(value: float | None, places: int = 1, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{places}f}%"


def mode_banner(mode: TradingMode, broker_name: str, data_provider: str) -> None:
    """The banner that makes paper-vs-live unmissable.

    Live mode gets an error-styled banner, not an info one. Someone glancing at
    this page must never have to look twice to know whether real money is at
    risk.
    """
    if mode is TradingMode.LIVE:
        st.error(
            f"**LIVE TRADING** — real money is at risk. "
            f"Broker: {broker_name}. Data: {data_provider}.",
            icon="🔴",
        )
    else:
        st.info(
            f"**PAPER TRADING** — no real money. "
            f"Broker: {broker_name}. Data: {data_provider}.",
            icon="📄",
        )


def warning_list(warnings: list[str] | tuple[str, ...], title: str = "Read this first") -> None:
    """Render caveats before the data they qualify."""
    if not warnings:
        return
    with st.container(border=True):
        st.markdown(f"**{title}**")
        for warning in warnings:
            st.markdown(f"- {warning}")


def timestamp_caption(moment: dt.datetime, tz_name: str) -> None:
    from zoneinfo import ZoneInfo

    local = to_display(moment, ZoneInfo(tz_name))
    st.caption(f"As of {local:%Y-%m-%d %H:%M %Z}")


def candidate_table(candidates: list[Candidate]) -> pd.DataFrame:
    """Flatten candidates for display, keeping every qualifier visible."""
    rows = []
    for candidate in candidates:
        forecast = candidate.forecast
        rows.append(
            {
                "": STATUS_ICON[candidate.status],
                "Symbol": candidate.symbol,
                "Status": candidate.status.value,
                "P(up)": pct(candidate.probability, 0) if forecast else "—",
                "Confidence": CONFIDENCE_ICON[candidate.confidence],
                "Horizon": f"{forecast.horizon_days}d" if forecast else "—",
                "Range": (
                    f"{pct(forecast.expected_return_low)} … "
                    f"{pct(forecast.expected_return_high)}"
                    if forecast and forecast.volatility
                    else "—"
                ),
                "Record": forecast.track_record.summary() if forecast else "—",
                "Price": money(candidate.last_price),
                "Reason": candidate.reason,
            }
        )
    return pd.DataFrame(rows)


def compliance_badge(status: ComplianceStatus, age_days: int | None = None) -> str:
    icon = COMPLIANCE_ICON.get(status, "❓")
    suffix = f" ({age_days}d old)" if age_days is not None else ""
    return f"{icon} {status.value}{suffix}"


def empty_state(message: str, detail: str = "") -> None:
    """A deliberate empty state.

    Shown instead of a blank area, because a blank panel reads as "broken"
    when the honest answer is "there is nothing here, and here is why".
    """
    with st.container(border=True):
        st.markdown(f"**{message}**")
        if detail:
            st.caption(detail)
