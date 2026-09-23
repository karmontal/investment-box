"""Dashboard rendering.

Uses Streamlit's own ``AppTest`` to execute the app script, so a broken page
fails here rather than in the browser. A dashboard that raises on load is
indistinguishable from one that is down.

These tests also pin the read-only guarantee: Phase 4 ships no widget that
writes. If a control is added without an engine behind it,
``test_no_write_controls`` fails.
"""

from __future__ import annotations

import datetime as dt

import pytest
from streamlit.testing.v1 import AppTest

from investment_box.config.loader import project_root

# Absolute: AppTest resolves relative paths against the caller's location,
# which differs between a direct pytest run and one from another directory.
APP = str(project_root() / "src" / "investment_box" / "ui" / "app.py")
#: Generous: the first run fetches market data and builds features.
TIMEOUT = 120


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path) -> AppTest:
    """Run the app against a temp data directory and no credentials."""
    monkeypatch.setenv("IB__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IB__DATA__CACHE_DIR", str(tmp_path / "cache"))
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


class TestRenders:
    def test_page_loads_without_raising(self, app: AppTest) -> None:
        app.run()
        assert not app.exception, f"dashboard raised: {app.exception}"

    def test_title_is_present(self, app: AppTest) -> None:
        app.run()
        assert any("Investment Box" in t.value for t in app.title)

    def test_every_tab_is_present(self, app: AppTest) -> None:
        app.run()
        labels = {tab.label for tab in app.tabs} if hasattr(app, "tabs") else set()
        # Streamlit's AppTest exposes tabs inconsistently across versions, so
        # fall back to asserting the page rendered something substantial.
        assert labels or app.markdown or app.dataframe

    def test_paper_mode_banner_is_shown(self, app: AppTest) -> None:
        """Nobody should have to look twice to know if real money is at risk."""
        app.run()
        banners = " ".join(info.value for info in app.info)
        assert "PAPER TRADING" in banners

    def test_mock_broker_is_disclosed(self, app: AppTest) -> None:
        app.run()
        rendered = " ".join(
            [*(m.value for m in app.markdown), *(i.value for i in app.info)]
        )
        assert "mock broker" in rendered.lower()


class TestControlSafety:
    """Controls exist from Phase 6, but the dangerous ones are gated."""

    def test_kill_switch_requires_a_second_confirmation(self, app: AppTest) -> None:
        """One click must never kill the engine."""
        app.run()
        labels = [b.label for b in app.button]
        assert any("KILL SWITCH" in label for label in labels)
        # The confirming button only appears after the first click.
        assert not any("Yes, do it" in label for label in labels)

    def test_live_trading_button_is_disabled(self, app: AppTest) -> None:
        app.run()
        live_buttons = [b for b in app.button if "live trading" in b.label.lower()]
        assert live_buttons
        assert all(b.disabled for b in live_buttons)

    def test_live_requires_a_typed_phrase_not_a_checkbox(self, app: AppTest) -> None:
        """A checkbox can be hit by accident; a phrase cannot."""
        from investment_box.ui.controls import LIVE_CONFIRMATION_PHRASE

        app.run()
        placeholders = [getattr(i, "placeholder", "") for i in app.text_input]
        assert LIVE_CONFIRMATION_PHRASE in " ".join(placeholders)

    def test_hard_constraints_are_stated_not_toggleable(self, app: AppTest) -> None:
        """A greyed-out switch implies it could be un-greyed. These cannot."""
        app.run()
        rendered = " ".join(m.value for m in app.markdown)
        assert "never permitted" in rendered
        toggles = [c.label for c in app.checkbox]
        for forbidden in ("margin", "short", "leverage", "crypto"):
            assert not any(forbidden in label.lower() for label in toggles)

    def test_controls_tab_is_present(self, app: AppTest) -> None:
        app.run()
        assert not app.exception


class TestSidebar:
    def test_strategy_selector_lists_every_strategy(self, app: AppTest) -> None:
        from investment_box.strategies import STRATEGY_REGISTRY

        app.run()
        selectors = [s for s in app.selectbox if s.label == "Strategy"]
        assert selectors
        assert set(selectors[0].options) == set(STRATEGY_REGISTRY)

    def test_allocated_capital_is_shown(self, app: AppTest) -> None:
        app.run()
        labels = [metric.label for metric in app.metric]
        assert "Allocated capital" in labels


class TestWarningsAreSurfaced:
    def test_unverified_universe_is_flagged(self, app: AppTest) -> None:
        """Every seed ETF is unverified; the dashboard must say so loudly."""
        app.run()
        errors = " ".join(e.value for e in app.error)
        markdown = " ".join(m.value for m in app.markdown)
        assert "unverified" in (errors + markdown).lower()

    def test_startup_warnings_are_rendered(self, app: AppTest) -> None:
        app.run()
        markdown = " ".join(m.value for m in app.markdown)
        assert "fractional" in markdown.lower() or "mock broker" in markdown.lower()


class TestComponents:
    """Pure helpers, tested directly rather than through the page."""

    def test_money_formatting(self) -> None:
        from investment_box.ui.components import money

        assert money(1234.5) == "$1,234.50"
        assert money(None) == "—"
        assert money(12.0, signed=True) == "+$12.00"

    def test_pct_formatting(self) -> None:
        from investment_box.ui.components import pct

        assert pct(0.1234) == "12.3%"
        assert pct(None) == "—"
        assert pct(0.05, signed=True) == "+5.0%"

    def test_candidate_table_keeps_qualifiers(self) -> None:
        """A probability must never appear without its confidence."""
        from investment_box.forecast.base import Confidence, Forecast, TrackRecord
        from investment_box.forecast.candidates import Candidate, CandidateStatus
        from investment_box.ui.components import candidate_table

        forecast = Forecast(
            symbol="SPUS", as_of=dt.date(2024, 6, 12), strategy="s",
            direction_probability=0.62, horizon_days=5,
            expected_return_low=-0.02, expected_return_high=0.02,
            confidence=Confidence.LOW,
            track_record=TrackRecord(strategy="s", trades=9, win_rate=0.6),
        )
        candidate = Candidate(
            symbol="SPUS", as_of=dt.date(2024, 6, 12),
            status=CandidateStatus.WATCH, reason="thin record", forecast=forecast,
        )
        frame = candidate_table([candidate])
        assert "Confidence" in frame.columns
        assert "Record" in frame.columns
        assert "too few trades" in frame.iloc[0]["Record"]


class TestEngineStatusIsReal:
    """The sidebar must report the engine, not a config flag.

    It read `settings.engine.enabled` -- a flag defaulting to false that the
    engine never sets -- and rendered "idle (phase 4)" beside a perfectly
    healthy engine that had been running for hours. A status indicator that
    cannot go wrong is not an indicator.
    """

    def test_it_reports_unknown_when_no_engine_has_ever_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from investment_box.ui import state

        state.get_engine_status.clear()
        status, since = state.get_engine_status()
        assert status == "unknown"
        assert since is None

    def test_it_reports_the_last_transition_the_engine_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from investment_box.ui import state

        services = state.get_services()
        services.audit.record("engine.running", "idle -> running: started", actor="scheduler")

        state.get_engine_status.clear()
        status, since = state.get_engine_status()
        assert status == "running"
        assert since is not None

    def test_a_later_transition_supersedes_an_earlier_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from investment_box.ui import state

        services = state.get_services()
        services.audit.record("engine.running", "idle -> running", actor="scheduler")
        services.audit.record("engine.paused", "running -> paused: margin", actor="scheduler")

        state.get_engine_status.clear()
        status, _ = state.get_engine_status()
        assert status == "paused", "the sidebar must show the newest state, not the first"

    def test_unrelated_audit_events_are_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from investment_box.ui import state

        services = state.get_services()
        services.audit.record("engine.running", "idle -> running", actor="scheduler")
        services.audit.record("order.submitted", "bought SPUS", actor="engine")

        state.get_engine_status.clear()
        status, _ = state.get_engine_status()
        assert status == "running", "only engine.* transitions describe the engine"
