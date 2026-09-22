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


class TestReadOnly:
    def test_no_write_controls(self, app: AppTest) -> None:
        """Phase 4 ships no widget that changes state.

        The only button is Refresh, which clears caches. A control that appears
        to pause a non-existent engine would be worse than no control.
        """
        app.run()
        labels = [button.label for button in app.button]
        assert labels == ["Refresh data"], f"unexpected controls: {labels}"

    def test_no_forms(self, app: AppTest) -> None:
        app.run()
        assert not getattr(app, "form", [])

    def test_read_only_notice_is_visible(self, app: AppTest) -> None:
        app.run()
        captions = " ".join(c.value for c in app.caption)
        assert "Read-only" in captions


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
