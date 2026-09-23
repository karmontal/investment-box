"""Shared state for the dashboard.

Streamlit re-runs the whole script on every interaction, so anything expensive
must be cached or it is rebuilt on every click. What is cached and for how long
matters here:

* **Services** are cached for the session. One database engine and one broker
  connection, not one per rerun.
* **Market data** is cached for minutes, not the session. A dashboard showing
  a price from an hour ago without saying so is worse than a slow dashboard.
* **Nothing that writes** is cached. Phase 4 is read-only, so this does not
  arise yet, but the rule is set now.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import streamlit as st

from investment_box.config.loader import load_universe_file
from investment_box.forecast.calibration import CalibrationTracker
from investment_box.forecast.track_record_store import TrackRecordStore
from investment_box.services.container import ServiceContainer, build_services
from investment_box.services.research import ResearchService, ResearchSnapshot
from investment_box.services.settings_service import SettingsService
from investment_box.shariah.providers.factory import build_screening_provider
from investment_box.shariah.status import ComplianceTracker
from investment_box.strategies import STRATEGY_REGISTRY
from investment_box.universe.builder import Instrument, UniverseBuilder

#: How long market-derived views stay cached. Short: a stale price shown
#: without a timestamp is misleading.
DATA_TTL_SECONDS = 120


@dataclass
class DashboardState:
    """Everything a page needs, built once per rerun."""

    services: ServiceContainer
    instruments: list[Instrument]
    benchmarks: list[str]


@st.cache_resource(show_spinner=False)
def get_services() -> ServiceContainer:
    """The service container, shared across reruns.

    ``cache_resource`` rather than ``cache_data``: this holds live connections,
    which must not be copied per rerun.
    """
    return build_services(configure_logs=True)


@st.cache_resource(show_spinner=False)
def get_compliance_tracker() -> ComplianceTracker:
    """Compliance tracker.

    Uses the same provider composition as the engine, so the dashboard can
    never show a verdict the engine would not act on. In Mode A that means
    certified funds answer from their own board and everything else is
    UNKNOWN, which is surfaced in the UI rather than hidden.
    """
    services = get_services()
    instruments = UniverseBuilder.load_instruments(load_universe_file())
    return ComplianceTracker(
        build_screening_provider(services.settings, instruments, clock=services.clock),
        services.database,
        services.settings.shariah,
        services.audit,
        clock=services.clock,
    )


@st.cache_resource(show_spinner=False)
def get_settings_service() -> SettingsService:
    """User-editable settings. A resource, not data: it writes."""
    from investment_box.services.settings_service import SettingsService

    services = get_services()
    return SettingsService(services.database, services.settings, services.audit)


@st.cache_data(show_spinner=False)
def get_universe() -> tuple[list[Instrument], list[str]]:
    config = load_universe_file()
    return UniverseBuilder.load_instruments(config), UniverseBuilder.benchmark_symbols(config)


def get_state() -> DashboardState:
    instruments, benchmarks = get_universe()
    return DashboardState(
        services=get_services(), instruments=instruments, benchmarks=benchmarks
    )


@st.cache_data(ttl=DATA_TTL_SECONDS, show_spinner="Building candidates…")
def get_research(strategy_name: str, _as_of: dt.date | None = None) -> ResearchSnapshot:
    """Ranked candidates for one strategy.

    The leading underscore on ``_as_of`` keeps Streamlit from hashing it while
    still letting callers pin the date in tests.
    """
    services = get_services()
    instruments, _ = get_universe()

    strategy_cls = STRATEGY_REGISTRY[strategy_name]
    research = ResearchService(
        services.settings,
        services.market_data,
        strategy_cls(),
        compliance=get_compliance_tracker(),
        calibration=CalibrationTracker(services.database),
        clock=services.clock,
    )
    # The same measured record the engine uses. If the dashboard loaded a
    # different one it would show forecasts the engine would not act on, and
    # the page exists to tell you what the engine is about to do.
    for record in TrackRecordStore(services.database).load(
        max_age_days=services.settings.engine.track_record_max_age_days,
        now=services.clock.now(),
    ).records:
        research.register_track_record(record)
    holdings = tuple(p.symbol for p in services.portfolio.get_positions())
    return research.build(instruments, as_of=_as_of, current_holdings=holdings)


def clear_caches() -> None:
    """Drop every cache. Bound to the Refresh button."""
    st.cache_data.clear()
    st.cache_resource.clear()
