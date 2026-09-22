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

    # Ignore any real .env on this machine. Clearing the environment is not
    # enough: pydantic-settings reads the file directly, so a developer with
    # live credentials configured would otherwise run a different test suite
    # than CI does -- and their real token would reach the tests.
    from investment_box.config import loader

    monkeypatch.setattr(loader, "DEFAULT_ENV_FILE", None)
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


# ----------------------------------------------------------------- Phase 2


@pytest.fixture
def audit(database: Database, settings: Settings):
    from investment_box.services.audit import AuditService

    return AuditService(database, settings.trading_mode)


@pytest.fixture
def approvals(database: Database, audit, clock: FrozenClock):
    from investment_box.services.approvals import ApprovalService

    return ApprovalService(database, audit, clock=clock, timeout_minutes=30)


@pytest.fixture
def transport():
    from investment_box.telegram.transport import FakeTransport

    return FakeTransport()


@pytest.fixture
def telegram_secrets() -> Secrets:
    """Secrets with a token, a channel and two whitelisted users."""
    return Secrets(  # type: ignore[call-arg]
        _env_file=None,
        telegram_bot_token="123456789:AAFakeTokenForTestsOnlyNotReal12345",
        telegram_channel_id="-1001234567890",
        telegram_allowed_user_ids="555000111,555000222",
    )


@pytest.fixture
def stack(
    settings: Settings,
    telegram_secrets: Secrets,
    portfolio,
    approvals,
    audit,
    clock: FrozenClock,
    transport,
):
    """A fully wired Telegram stack on the in-memory transport."""
    from investment_box.telegram.bot import build_telegram_stack

    return build_telegram_stack(
        settings=settings,
        secrets=telegram_secrets,
        portfolio=portfolio,
        approvals=approvals,
        audit=audit,
        clock=clock,
        transport=transport,
        data_provider_name="synthetic",
        engine_state="paused",
    )
