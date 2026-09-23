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
import os
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


def _universe_fingerprint() -> tuple[float, int]:
    """Modification time and size of the universe file.

    Used as a cache key so editing the file invalidates the cache by itself.
    Size is included because a filesystem with coarse mtime resolution can
    report the same second for an edit made moments later.
    """
    from investment_box.config.loader import UNIVERSE_CONFIG_NAME, config_dir

    try:
        stat = (config_dir() / UNIVERSE_CONFIG_NAME).stat()
    except OSError:
        return (0.0, 0)
    return (stat.st_mtime, stat.st_size)


@st.cache_data(show_spinner=False)
def _load_universe(fingerprint: tuple[float, int]) -> tuple[list[Instrument], list[str]]:
    """Read the universe file. ``fingerprint`` exists solely to key the cache.

    Its name must NOT start with an underscore. Streamlit skips hashing any
    argument whose name begins with one -- the same rule ``get_research`` relies
    on deliberately for ``_as_of`` -- so calling it ``_fingerprint`` silently
    removed it from the cache key and left the cache exactly as unkeyed as
    before. The function looked fixed, the attribute existed, and the dashboard
    still served the list it read at startup.
    """
    del fingerprint  # used only as part of the cache key
    config = load_universe_file()
    return UniverseBuilder.load_instruments(config), UniverseBuilder.benchmark_symbols(config)


def get_universe() -> tuple[list[Instrument], list[str]]:
    """The configured universe, re-read whenever the file changes.

    `config/` is a live bind mount in the container precisely so the list can
    be edited without a rebuild. Caching it with no key defeated that: marking
    a fund `verified: true` changed nothing on screen until someone restarted
    the container, which looks exactly like the edit not having worked.
    """
    return _load_universe(_universe_fingerprint())


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

@st.cache_data(ttl=15, show_spinner=False)
def get_engine_status() -> tuple[str, dt.datetime | None]:
    """The engine's last reported state, and when it reported it.

    The engine runs in its own container and keeps its state machine in
    memory, so the dashboard cannot ask it directly. It does record every
    transition to the audit log, which both containers share, so that is the
    honest source. Previously this metric read `settings.engine.enabled` -- a
    config flag that defaults to false and that the engine never sets -- so a
    perfectly healthy engine was reported as "idle (phase 4)" forever.

    Returns ("unknown", None) when no engine has ever started against this
    database. A short TTL because this is a liveness indicator, not data.
    """
    from sqlalchemy import select

    from investment_box.db.models import AuditLog

    services = get_services()
    with services.database.session() as session:
        row = session.scalar(
            select(AuditLog)
            .where(AuditLog.event_type.like("engine.%"))
            .order_by(AuditLog.id.desc())
            .limit(1)
        )
        if row is None:
            return ("unknown", None)
        return (row.event_type.split(".", 1)[1], row.created_at)

#: When THIS process started. The signal that matters after a rebuild: a page
#: served by a process older than your last `docker compose up -d --build` is
#: a stale browser session, not a broken deployment. Distinguishing those two
#: by inspection cost an hour once.
PROCESS_STARTED_AT = dt.datetime.now(dt.UTC)

#: Commit the image was built from, stamped in by the Dockerfile's GIT_COMMIT
#: build arg. "unknown" when nobody passed one, which is fine -- the start time
#: alone answers the staleness question.
BUILD_COMMIT = os.environ.get("IB_BUILD_COMMIT", "unknown").strip() or "unknown"


def build_stamp(tz_name: str) -> str:
    """One line identifying exactly what is being served."""
    from zoneinfo import ZoneInfo

    from investment_box.core.clock import to_display

    started = to_display(PROCESS_STARTED_AT, ZoneInfo(tz_name))
    commit = BUILD_COMMIT if BUILD_COMMIT != "unknown" else "unstamped"
    return f"build {commit} · serving since {started:%Y-%m-%d %H:%M %Z}"
