"""Shared fixtures.

Every test runs offline and deterministically: a frozen clock, an in-memory
database, the mock broker and the synthetic data provider. No test touches the
network unless it is marked ``network``, and none are by default.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from investment_box.config.loader import clear_caches, load_settings
from investment_box.config.schema import Secrets, Settings
from investment_box.core.clock import UTC, FrozenClock, TradingCalendar
from investment_box.data.cache import ParquetCache
from investment_box.data.repository import MarketDataRepository
from investment_box.data.synthetic import SyntheticDataProvider
from investment_box.db.session import Database
from investment_box.execution.mock_broker import MockBroker
from investment_box.services.portfolio import PortfolioService

#: A Wednesday, well clear of holidays, with a known surrounding calendar.
REFERENCE_INSTANT = dt.datetime(2024, 6, 12, 20, 30, tzinfo=UTC)
REFERENCE_DATE = REFERENCE_INSTANT.date()


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop a developer's real .env or exported vars from reaching a test.

    Without this, running the suite on a machine that has ALPACA_API_KEY set
    would exercise a different code path than CI, which is exactly the kind of
    difference that hides a bug until it matters.
    """
    for name in (
        "ALPACA_API_KEY",
        "ALPACA_SECRET_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHANNEL_ID",
        "TELEGRAM_ALLOWED_USER_IDS",
        "SHARIAH_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in list(dict(**{k: v for k, v in __import__("os").environ.items()})):
        if name.startswith("IB__"):
            monkeypatch.delenv(name, raising=False)
    clear_caches()


@pytest.fixture
def clock() -> FrozenClock:
    return FrozenClock(REFERENCE_INSTANT)


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar(anchor=REFERENCE_DATE)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default config with data paths redirected into a temp directory."""
    return load_settings(
        overrides={
            "data_dir": str(tmp_path / "data"),
            "data": {"cache_dir": str(tmp_path / "cache")},
        }
    )


@pytest.fixture
def secrets() -> Secrets:
    return Secrets(_env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def database() -> Database:
    db = Database("sqlite:///:memory:")
    db.create_all()
    yield db
    db.dispose()


@pytest.fixture
def broker(clock: FrozenClock, calendar: TradingCalendar) -> MockBroker:
    return MockBroker(
        starting_cash=Decimal("500.00"),
        clock=clock,
        calendar=calendar,
        prices={"SPUS": Decimal("45.00"), "HLAL": Decimal("50.00"), "SPSK": Decimal("20.00")},
        slippage_pct=0.0,
        settlement_days=1,
    )


@pytest.fixture
def portfolio(
    broker: MockBroker, database: Database, settings: Settings, clock: FrozenClock
) -> PortfolioService:
    return PortfolioService(broker, database, settings, clock=clock)


@pytest.fixture
def synthetic_provider() -> SyntheticDataProvider:
    return SyntheticDataProvider()


@pytest.fixture
def repository(
    tmp_path: Path, synthetic_provider: SyntheticDataProvider, calendar: TradingCalendar
) -> MarketDataRepository:
    return MarketDataRepository(
        provider=synthetic_provider,
        cache=ParquetCache(tmp_path / "cache", ttl_hours=12),
        calendar=calendar,
    )
