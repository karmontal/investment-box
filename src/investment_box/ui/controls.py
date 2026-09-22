"""Dashboard controls.

The write side of the dashboard, kept in its own module so the read-only
guarantee for everything else stays easy to verify.

The rules encoded here:

* **Switching to live trading requires typing a phrase**, not ticking a box. A
  checkbox can be hit by accident; ``ENABLE LIVE TRADING`` cannot. And even
  then it is refused until the Phase 7 pre-flight checklist exists.
* **The kill switch asks twice**, and states the exposure it would affect
  before the second tap.
* **Hard Shariah constraints have no control at all.** Not a disabled one --
  none. They are stated as facts, because a greyed-out toggle implies it could
  be un-greyed.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import streamlit as st

from investment_box.core.types import AutonomyLevel, TradingMode, UniverseMode
from investment_box.engine.runner import EngineRunner
from investment_box.services.container import ServiceContainer
from investment_box.services.settings_service import SettingsService, SymbolRule
from investment_box.shariah.constraints import CONSTRAINTS
from investment_box.ui import components as ui

#: The exact phrase required to enable live trading.
LIVE_CONFIRMATION_PHRASE = "ENABLE LIVE TRADING"

AUTONOMY_LABELS = {
    AutonomyLevel.SUGGEST_ONLY: "1 — Suggest only (you approve every trade)",
    AutonomyLevel.AUTO_WITHIN_WHITELIST: "2 — Auto within whitelist, ask otherwise",
    AutonomyLevel.FULLY_AUTONOMOUS: "3 — Fully autonomous within all limits",
}

RULE_LABELS = {
    SymbolRule.ALLOWED: "Allowed",
    SymbolRule.NEEDS_APPROVAL: "Needs approval",
    SymbolRule.FORBIDDEN: "Forbidden",
}


def render_controls(
    settings_service: SettingsService,
    services: ServiceContainer,
    engine: EngineRunner | None = None,
) -> None:
    """The Controls tab."""
    st.subheader("Engine")
    _engine_controls(engine, settings_service)
    _kill_switch(services, settings_service, engine)

    st.divider()
    st.subheader("Capital and autonomy")
    _capital_and_autonomy(settings_service, services)

    st.divider()
    st.subheader("Symbol rules")
    _symbol_rules(settings_service, services)

    st.divider()
    st.subheader("Trading mode")
    _trading_mode(services)

    st.divider()
    st.subheader("Constraints that cannot be changed")
    _hard_constraints()


def _engine_controls(
    engine: EngineRunner | None, settings_service: SettingsService
) -> None:
    if settings_service.kill_requested:
        st.error(
            f"**KILL SWITCH IS ACTIVE** — {settings_service.kill_reason}\n\n"
            f"The engine will refuse to trade, in this process and any other, until "
            f"this is cleared.",
            icon="🚨",
        )
        if st.button("Clear the kill flag"):
            settings_service.clear_kill(actor="dashboard")
            st.rerun()

    if engine is None:
        st.caption(
            "No engine process is attached to this dashboard, so pause and resume "
            "are unavailable here. The kill switch below still works: it acts "
            "directly on the broker and sets a flag the engine honours."
        )
        return

    state = engine.state.status
    columns = st.columns([2, 1, 1])
    columns[0].metric("State", state.state.value)
    columns[1].metric("Cycles run", state.cycles_run)
    columns[2].metric("Last cycle", state.last_cycle_summary or "—")

    if engine.blockers:
        st.error(
            "**The engine will not trade:**\n\n"
            + "\n".join(f"- {b}" for b in engine.blockers),
            icon="🚫",
        )

    left, right = st.columns(2)
    if left.button("⏸ Pause", width='stretch', disabled=not state.is_running):
        engine.state.pause("paused from the dashboard", actor="dashboard")
        engine.risk.pause("paused from the dashboard", actor="dashboard")
        st.rerun()
    if right.button("▶️ Resume", width='stretch', disabled=state.is_running):
        if engine.state.resume(actor="dashboard"):
            engine.risk.resume(actor="dashboard")
            st.rerun()
        else:
            st.error("Cannot resume: the kill switch was used. Restart the process.")


def _kill_switch(
    services: ServiceContainer,
    settings_service: SettingsService,
    engine: EngineRunner | None = None,
) -> None:
    """Works with or without an engine process.

    Cancelling orders and closing positions act directly on the broker, so the
    money-at-risk part is immediate. The persisted flag then stops the engine
    wherever it is running.
    """
    from investment_box.engine.kill_switch import KillSwitch
    from investment_box.engine.state import EngineStateMachine
    from investment_box.execution.order_manager import OrderManager

    st.markdown("---")
    if engine is not None:
        switch = engine.kill_switch
    else:
        switch = KillSwitch(
            services.broker,
            OrderManager(
                services.broker, services.database, services.settings, services.audit,
                clock=services.clock,
            ),
            EngineStateMachine(services.audit, clock=services.clock),
            services.audit,
        )

    exposure = switch.estimate_exposure()

    with st.container(border=True):
        st.markdown("### 🚨 Kill switch")
        st.caption(
            "Cancels every open order and stops the engine. It cannot be resumed "
            "without restarting the process."
        )
        close = st.checkbox(
            f"Also close all positions at market ({ui.money(exposure)} exposure)",
            help=(
                "Realises losses immediately and starts the T+1 settlement clock. "
                "Leave unticked to stop the engine but keep positions."
            ),
        )

        if st.session_state.get("kill_armed"):
            st.error(
                "**Confirm: kill the engine"
                + (f" AND close {ui.money(exposure)} of positions" if close else "")
                + "?**"
            )
            left, right = st.columns(2)
            if left.button("Yes, do it", type="primary", width='stretch'):
                reason = "kill switch from the dashboard"
                result = switch.activate(reason, close_positions=close, actor="dashboard")
                # Persist it too, so an engine in another process stops as well.
                settings_service.request_kill(reason, actor="dashboard")
                st.session_state["kill_armed"] = False
                (st.success if result.fully_successful else st.error)(result.detail())
            if right.button("Cancel", width='stretch'):
                st.session_state["kill_armed"] = False
                st.rerun()
        elif st.button("🚨 KILL SWITCH", type="primary", width='stretch'):
            st.session_state["kill_armed"] = True
            st.rerun()


def _capital_and_autonomy(
    settings_service: SettingsService, services: ServiceContainer
) -> None:
    left, right = st.columns(2)

    with left:
        current = settings_service.capital_allocation
        st.markdown("**Capital the bot may use**")
        raw = st.text_input(
            "Allocated capital (USD)", value=str(current), label_visibility="collapsed"
        )
        if st.button("Update capital"):
            try:
                amount = Decimal(raw)
            except (InvalidOperation, ValueError):
                st.error("Not a valid amount.")
            else:
                try:
                    equity = services.portfolio.get_account_view().equity
                    settings_service.set_capital_allocation(
                        amount, account_equity=equity, actor="dashboard"
                    )
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.success(f"Allocated capital set to {ui.money(amount)}.")
                    st.rerun()

    with right:
        st.markdown("**Autonomy level**")
        current_level = settings_service.autonomy_level
        options = list(AUTONOMY_LABELS)
        chosen = st.selectbox(
            "Autonomy",
            options,
            index=options.index(current_level),
            format_func=lambda level: AUTONOMY_LABELS[level],
            label_visibility="collapsed",
        )
        if chosen is not current_level and st.button("Change autonomy"):
            settings_service.set_autonomy_level(chosen, actor="dashboard")
            st.success(f"Autonomy set to {AUTONOMY_LABELS[chosen]}.")
            st.rerun()

    st.markdown("**Universe mode**")
    modes = list(UniverseMode)
    mode = st.radio(
        "Universe mode",
        modes,
        index=modes.index(settings_service.universe_mode),
        format_func=lambda m: (
            "A — certified ETFs only" if m is UniverseMode.ETF_ONLY
            else "B — ETFs plus screened stocks"
        ),
        horizontal=True,
        label_visibility="collapsed",
    )
    if mode is not settings_service.universe_mode and st.button("Change universe mode"):
        if mode is UniverseMode.ETF_AND_SCREENED_STOCKS:
            st.warning(
                "Mode B screens individual stocks. The internal screener has no "
                "business-activity database, so most symbols will come back DOUBTFUL "
                "and need your decision. A certified provider is needed for this to "
                "be useful.",
                icon="⚠️",
            )
        settings_service.set_universe_mode(mode, actor="dashboard")
        st.rerun()


def _symbol_rules(
    settings_service: SettingsService, services: ServiceContainer
) -> None:
    st.caption(
        "A symbol you have never ruled on defaults to **needs approval**, not "
        "allowed. Forbidden always wins, whatever the strategy says."
    )

    from investment_box.config.loader import load_universe_file
    from investment_box.universe.builder import UniverseBuilder

    try:
        instruments = UniverseBuilder.load_instruments(load_universe_file())
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the universe: {exc}")
        return

    rules = settings_service.symbol_rules()
    for instrument in instruments:
        columns = st.columns([2, 3, 3])
        columns[0].markdown(f"**{instrument.symbol}**")
        columns[1].caption(instrument.name or "—")

        current = rules.get(instrument.symbol, SymbolRule.NEEDS_APPROVAL)
        options = list(RULE_LABELS)
        chosen = columns[2].selectbox(
            f"rule_{instrument.symbol}",
            options,
            index=options.index(current),
            format_func=lambda r: RULE_LABELS[r],
            label_visibility="collapsed",
            key=f"rule_select_{instrument.symbol}",
        )
        if chosen is not current:
            settings_service.set_symbol_rule(instrument.symbol, chosen, actor="dashboard")
            st.rerun()


def _trading_mode(services: ServiceContainer) -> None:
    mode = services.settings.trading_mode
    if mode is TradingMode.LIVE:
        st.error("**LIVE TRADING IS ACTIVE.** Real money is at risk.", icon="🔴")
        return

    st.info("Currently **PAPER**. No real money is at risk.", icon="📄")
    with st.expander("Switch to live trading"):
        st.warning(
            "Live trading is not available. It requires the Phase 7 pre-flight "
            "checklist — a minimum period of paper trading, paper results within "
            "tolerance of the backtest, and compliance screening verified for every "
            "held symbol. None of that exists yet.",
            icon="🚫",
        )
        typed = st.text_input(
            f"To enable it later you will type: {LIVE_CONFIRMATION_PHRASE}",
            placeholder=LIVE_CONFIRMATION_PHRASE,
        )
        if st.button("Enable live trading", disabled=True):
            st.error("Refused.")
        if typed == LIVE_CONFIRMATION_PHRASE:
            st.error(
                "Phrase correct, but live trading is still refused: the pre-flight "
                "checklist does not exist yet.",
                icon="🚫",
            )


def _hard_constraints() -> None:
    """Stated as facts. Deliberately not rendered as disabled toggles.

    A greyed-out switch implies it could be un-greyed; these cannot be changed
    from anywhere in the application.
    """
    st.caption(
        "These are frozen constants in `shariah/constraints.py`, asserted on every "
        "order. There is no setting for them — changing one requires editing source."
    )
    rows = [
        ("Margin / borrowing", CONSTRAINTS.margin_allowed),
        ("Short selling", CONSTRAINTS.short_selling_allowed),
        ("Options, futures, CFDs", CONSTRAINTS.derivatives_allowed),
        ("Leveraged or inverse funds", CONSTRAINTS.leveraged_or_inverse_allowed),
        ("Crypto and crypto derivatives", CONSTRAINTS.crypto_allowed),
    ]
    for label, allowed in rows:
        st.markdown(f"- **{label}:** {'allowed' if allowed else 'never permitted'}")
    st.markdown(
        "- **Unsettled cash:** never used to fund a purchase (T+1 settlement enforced)"
    )
