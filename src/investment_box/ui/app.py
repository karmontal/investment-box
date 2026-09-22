"""Investment Box dashboard — read-only.

Phase 4 deliberately ships no controls. Every widget here reads; none writes.
The kill switch, autonomy selector and symbol rules arrive in Phase 6, once
there is an engine for them to affect. Shipping a button that appears to pause
a non-existent engine would be worse than shipping no button.

Run it:

    uv run streamlit run src/investment_box/ui/app.py
"""

from __future__ import annotations

from decimal import Decimal
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

from investment_box.core.clock import to_display
from investment_box.core.types import ComplianceStatus
from investment_box.services.container import ServiceContainer
from investment_box.strategies import STRATEGY_REGISTRY
from investment_box.ui import components as ui
from investment_box.ui.controls import render_controls
from investment_box.ui.state import (
    clear_caches,
    get_research,
    get_settings_service,
    get_state,
)

st.set_page_config(
    page_title="Investment Box",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    state = get_state()
    services = state.services
    settings = services.settings
    tz = settings.i18n.display_timezone

    st.title("Investment Box")
    ui.mode_banner(
        settings.trading_mode, services.broker.name, services.market_data.provider.name
    )

    with st.sidebar:
        st.header("Status")
        st.metric("Allocated capital", ui.money(settings.capital.allocation_usd))
        st.metric("Autonomy level", settings.engine.autonomy_level.value)
        st.metric("Engine", "running" if settings.engine.enabled else "idle (phase 4)")

        strategy_name = st.selectbox(
            "Strategy", list(STRATEGY_REGISTRY), index=0,
            help="Which strategy's candidates to show.",
        )

        if st.button("Refresh data", width='stretch'):
            clear_caches()
            st.rerun()

        st.divider()
        st.caption("Controls are on the Controls tab.")

    startup = services.settings.startup_warnings()
    if services.is_using_mock_broker:
        startup.insert(0, "Running against the in-memory mock broker; balances are simulated.")
    ui.warning_list(startup)

    (
        overview,
        positions_tab,
        candidates_tab,
        compliance_tab,
        universe_tab,
        purification_tab,
        controls_tab,
        audit_tab,
    ) = st.tabs(
        [
            "Overview",
            "Positions",
            "Candidates",
            "Compliance",
            "Universe",
            "Purification",
            "Controls",
            "Audit",
        ]
    )

    with overview:
        _overview(services, tz)
    with positions_tab:
        _positions(services, tz)
    with candidates_tab:
        _candidates(strategy_name)
    with compliance_tab:
        _compliance(strategy_name)
    with universe_tab:
        _universe(strategy_name)
    with purification_tab:
        _purification(services)
    with controls_tab:
        render_controls(get_settings_service(), services, engine=None)
    with audit_tab:
        _audit(services)


def _overview(services: ServiceContainer, tz: str) -> None:
    view = services.portfolio.get_account_view()

    columns = st.columns(4)
    columns[0].metric("Equity", ui.money(view.equity))
    columns[1].metric(
        "Day P&L",
        ui.money(view.day_pnl, signed=True),
        delta=ui.pct(view.day_pnl_pct, signed=True) if view.day_pnl_pct else None,
    )
    columns[2].metric("Settled cash", ui.money(view.cash_settled))
    columns[3].metric(
        "Unsettled",
        ui.money(view.cash_unsettled),
        help="Sale proceeds that cannot fund a purchase until they settle (T+1).",
    )

    ui.timestamp_caption(view.taken_at, tz)
    ui.warning_list(list(view.warnings), "Account warnings")

    st.subheader("Capital usage")
    usage = view.capital
    columns = st.columns(4)
    columns[0].metric("Deployed", ui.money(usage.deployed), delta=ui.pct(usage.deployed_pct))
    columns[1].metric("Available", ui.money(usage.available_settled))
    columns[2].metric("Positions", f"{usage.open_positions}/{usage.max_open_positions}")
    columns[3].metric("Cash buffer", ui.money(usage.cash_buffer))

    st.subheader("Equity curve")
    curve = services.portfolio.equity_curve(days=365)
    if not curve:
        ui.empty_state(
            "No equity history yet.",
            "A snapshot is recorded once per trading day. The engine that records "
            "them arrives in Phase 5.",
        )
        return

    frame = pd.DataFrame(
        {
            "date": [row.snapshot_date for row in curve],
            "equity": [float(row.equity) for row in curve],
        }
    )
    chart = (
        alt.Chart(frame)
        .mark_line(point=len(frame) < 30)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("equity:Q", title="Equity ($)", scale=alt.Scale(zero=False)),
            tooltip=["date:T", alt.Tooltip("equity:Q", format="$,.2f")],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width='stretch')


def _positions(services: ServiceContainer, tz: str) -> None:
    view = services.portfolio.get_account_view()
    if not view.positions:
        ui.empty_state("No open positions.", "Nothing is currently held.")
        return

    rows = []
    for position in view.positions:
        rows.append(
            {
                "Symbol": position.symbol,
                "Qty": f"{position.quantity:g}",
                "Entry": ui.money(position.avg_entry_price),
                "Current": ui.money(position.current_price),
                "Value": ui.money(position.market_value),
                "P&L": ui.money(position.unrealized_pnl, signed=True),
                "P&L %": ui.pct(position.unrealized_pnl_pct, signed=True),
                "Days held": position.days_held if position.days_held is not None else "—",
                "% of capital": ui.pct(position.pct_of_allocation),
                "Stop": (
                    ui.money(position.stop_loss_price)
                    + (" (engine-managed)" if position.stop_is_synthetic else "")
                    if position.stop_loss_price
                    else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    ui.timestamp_caption(view.taken_at, tz)

    unprotected = [p for p in view.positions if p.stop_loss_price and p.stop_is_synthetic]
    if unprotected:
        st.warning(
            f"{len(unprotected)} position(s) rely on an engine-managed stop. "
            f"If the engine stops, they are unprotected.",
            icon="⚠️",
        )


def _candidates(strategy_name: str) -> None:
    snapshot = get_research(strategy_name)

    ui.warning_list(snapshot.warnings, "Before reading these forecasts")

    if snapshot.regime is not None:
        icon = {"risk_on": "🟢", "cautious": "🟡", "risk_off": "🔴", "unknown": "⚪"}[
            snapshot.regime.regime.value
        ]
        st.info(
            f"{icon} **Market regime: {snapshot.regime.regime.value}** — "
            f"{snapshot.regime.reason}"
        )

    if snapshot.decision is not None and snapshot.decision.rationale:
        st.caption(f"Strategy says: {snapshot.decision.rationale}")

    if not snapshot.candidates:
        ui.empty_state("No candidates.", "The universe produced nothing to evaluate.")
        return

    actionable = snapshot.actionable
    if actionable:
        st.success(f"{len(actionable)} actionable candidate(s).")
    else:
        st.info(
            "Nothing is actionable. That is a normal outcome, not an error — "
            "see the reason column."
        )

    st.dataframe(
        ui.candidate_table(snapshot.candidates), width='stretch', hide_index=True
    )

    st.subheader("Detail")
    symbol = st.selectbox(
        "Symbol", [c.symbol for c in snapshot.candidates], key="candidate_detail"
    )
    chosen = next((c for c in snapshot.candidates if c.symbol == symbol), None)
    if chosen is None:
        return

    left, right = st.columns([2, 1])
    with left:
        st.markdown(f"**{chosen.symbol}** — {ui.STATUS_ICON[chosen.status]} {chosen.status.value}")
        st.markdown(chosen.reason)
        if chosen.forecast is not None:
            st.code(chosen.forecast.honest_summary(), language=None)
    with right:
        if chosen.forecast is not None:
            st.metric("P(up)", ui.pct(chosen.forecast.direction_probability, 0))
            st.metric("Confidence", chosen.forecast.confidence.value)
            st.metric("Edge over a coin flip", ui.pct(chosen.forecast.edge, 1, signed=True))
        if chosen.compliance is not None:
            st.markdown(
                "**Compliance**  \n"
                + ui.compliance_badge(
                    chosen.compliance.display_status, chosen.compliance.age_days
                )
            )


def _compliance(strategy_name: str) -> None:
    snapshot = get_research(strategy_name)

    st.subheader("Forecast calibration")
    report = snapshot.calibration
    if report is None or report.samples == 0:
        ui.empty_state(
            "No resolved predictions yet.",
            "Calibration compares predicted probabilities against what actually "
            "happened. It needs closed trades, which arrive in Phase 5.",
        )
    else:
        columns = st.columns(3)
        columns[0].metric("Resolved predictions", report.samples)
        columns[1].metric(
            "Brier score",
            f"{report.brier_score:.3f}" if report.brier_score is not None else "—",
            help="Mean squared error of the probabilities. Always guessing 50% scores 0.250.",
        )
        columns[2].metric("Bias", report.bias)
        ui.warning_list(report.warnings(), "Calibration warnings")

        if report.bins:
            frame = pd.DataFrame(
                {
                    "bucket": [b.label for b in report.bins],
                    "predicted": [b.mean_predicted for b in report.bins],
                    "observed": [b.observed_frequency for b in report.bins],
                    "count": [b.count for b in report.bins],
                }
            )
            melted = frame.melt(
                id_vars=["bucket", "count"], var_name="series", value_name="probability"
            )
            chart = (
                alt.Chart(melted)
                .mark_bar()
                .encode(
                    x=alt.X("bucket:N", title="Predicted probability"),
                    y=alt.Y("probability:Q", title=None, axis=alt.Axis(format="%")),
                    color="series:N",
                    xOffset="series:N",
                    tooltip=[
                        "bucket",
                        "series",
                        alt.Tooltip("probability:Q", format=".1%"),
                        "count",
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(chart, width='stretch')

    st.subheader("Compliance status")
    rows = []
    for candidate in snapshot.candidates:
        record = candidate.compliance
        rows.append(
            {
                "Symbol": candidate.symbol,
                "Status": (
                    ui.compliance_badge(record.display_status, record.age_days)
                    if record
                    else ui.compliance_badge(ComplianceStatus.UNKNOWN)
                ),
                "Source": record.source if record else "not screened",
                "Screened": (
                    record.screened_at.date().isoformat() if record else "—"
                ),
                "Reason": record.reason if record else "no screen on record",
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    st.caption(
        "Only COMPLIANT symbols are traded automatically. DOUBTFUL and UNKNOWN "
        "always require a human decision, and a screen older than the re-screen "
        "interval reads as UNKNOWN regardless of what it last said."
    )


def _universe(strategy_name: str) -> None:
    snapshot = get_research(strategy_name)
    universe = snapshot.universe
    if universe is None:
        ui.empty_state("Universe not built.")
        return

    columns = st.columns(3)
    columns[0].metric("Included", len(universe.included))
    columns[1].metric("Excluded", len(universe.excluded))
    columns[2].metric("Mode", universe.mode.value)

    if universe.used_current_compliance_for_history:
        st.warning(
            "This build used today's compliance status for a past date. There is no "
            "point-in-time Shariah compliance history for this universe, which is "
            "look-ahead bias.",
            icon="⚠️",
        )

    unverified = [e.symbol for e in universe.entries if not e.instrument.verified]
    if unverified:
        st.error(
            f"**{len(unverified)} symbol(s) are unverified and will never be "
            f"traded:** "
            f"{', '.join(unverified)}. Confirm listing and Shariah certification from "
            f"each fund's own documents, then set `verified: true` in "
            f"`config/universe_etf.yaml`.",
            icon="🚫",
        )

    rows = []
    for entry in universe.entries:
        rows.append(
            {
                "": "✅" if entry.included else "🚫",
                "Symbol": entry.symbol,
                "Name": entry.instrument.name or "—",
                "Verified": "yes" if entry.instrument.verified else "NO",
                "Inception": (
                    entry.instrument.inception.isoformat()
                    if entry.instrument.inception
                    else "unknown"
                ),
                "Price": ui.money(entry.last_price),
                "Avg $ volume": (
                    f"${entry.avg_dollar_volume:,.0f}" if entry.avg_dollar_volume else "—"
                ),
                "Bars": entry.bars_available or "—",
                "Verdict": entry.reason,
            }
        )
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    summary = universe.rejection_summary()
    if summary:
        st.caption("Exclusions by reason: " + ", ".join(f"{k} ({v})" for k, v in summary.items()))



def _purification(services: ServiceContainer) -> None:
    """Purification owed, and what could not be computed."""
    from investment_box.shariah.purification import PurificationTracker
    from investment_box.shariah.zakat import ZakatHolding, ZakatMethod, estimate_zakat

    tracker = PurificationTracker(services.database, services.audit, clock=services.clock)
    report = tracker.report()

    columns = st.columns(4)
    columns[0].metric("Dividends recorded", ui.money(report.total_dividends))
    columns[1].metric("Total to purify", ui.money(report.total_due))
    columns[2].metric("Outstanding", ui.money(report.outstanding))
    columns[3].metric("Already purified", ui.money(report.already_purified))

    ui.warning_list(report.warnings(), "Gaps in this figure")

    if not report.entries:
        ui.empty_state(
            "No dividends recorded yet.",
            "Dividends are recorded as they are received. Purification needs each "
            "fund's published non-permissible income ratio, which you enter.",
        )
    else:
        rows = [
            {
                "Symbol": e.symbol,
                "Pay date": e.pay_date.isoformat(),
                "Gross": ui.money(e.gross_amount),
                "Ratio": (
                    f"{e.non_permissible_ratio:.4%}"
                    if e.non_permissible_ratio is not None
                    else "UNKNOWN"
                ),
                "To purify": ui.money(e.amount_due),
                "Method": e.method.value,
                "Status": "purified" if e.purified_at else "outstanding",
            }
            for e in report.entries
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        st.download_button(
            "Download CSV",
            data=tracker.to_csv(report),
            file_name="purification.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Zakat estimate")
    st.caption(
        "An estimate produced by software, not a ruling. Scholars differ on how "
        "trading assets are treated."
    )

    view = services.portfolio.get_account_view()
    method = st.radio(
        "Method",
        list(ZakatMethod),
        format_func=lambda m: m.value.replace("_", " "),
        horizontal=True,
    ) or ZakatMethod.FULL_MARKET_VALUE
    nisab_raw = st.text_input(
        "Nisab threshold (USD, optional)",
        help="Tracks the current gold or silver price. Leave blank to skip the check.",
    )
    nisab = None
    if nisab_raw.strip():
        try:
            nisab = Decimal(nisab_raw)
        except Exception:  # noqa: BLE001
            st.warning("Not a valid amount; ignoring the threshold.")

    estimate = estimate_zakat(
        as_of=services.clock.now().date(),
        holdings=[
            ZakatHolding(symbol=p.symbol, market_value=p.market_value)
            for p in view.positions
        ],
        cash=view.cash_settled + view.cash_unsettled,
        method=method,
        nisab_threshold=nisab,
    )

    columns = st.columns(3)
    columns[0].metric("Zakatable base", ui.money(estimate.zakatable_base))
    columns[1].metric("Estimated zakat (2.5%)", ui.money(estimate.estimated_zakat))
    columns[2].metric(
        "Meets nisab",
        "—" if estimate.meets_nisab is None else ("yes" if estimate.meets_nisab else "no"),
    )
    ui.warning_list(estimate.caveats(), "Read before using this figure")


def _audit(services: ServiceContainer) -> None:
    """The full audit trail: every decision, action and refusal."""
    entries = services.audit.recent(limit=300)
    if not entries:
        ui.empty_state("Nothing recorded yet.")
        return

    event_types = sorted({e.event_type for e in entries})
    chosen = st.multiselect("Filter by event type", event_types, default=[])
    filtered = [e for e in entries if not chosen or e.event_type in chosen]

    rows = [
        {
            "When": to_display(
                e.created_at, ZoneInfo(services.settings.i18n.display_timezone)
            ).strftime("%Y-%m-%d %H:%M"),
            "Event": e.event_type,
            "Actor": e.actor,
            "Symbol": e.symbol or "—",
            "Summary": e.summary,
        }
        for e in filtered
    ]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=520)
    st.caption(
        f"{len(filtered)} of {len(entries)} recent entries. The audit log is "
        f"append-only: nothing in this application updates or deletes a row."
    )


main()
