"""Wiring.

One place that decides which broker, which data provider and which database a
process uses, so that the dashboard, the bot and the scheduler are guaranteed
to be looking at the same thing.

The broker choice is deliberately conservative: without credentials you get the
mock broker, and switching to live requires *both* ``trading_mode: live`` in
config *and* a non-paper base URL. Either one alone keeps you on paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from investment_box.config.loader import get_secrets, load_settings
from investment_box.config.schema import Secrets, Settings
from investment_box.core.clock import Clock, SystemClock, TradingCalendar
from investment_box.core.logging import configure_logging, get_logger
from investment_box.core.types import TradingMode
from investment_box.data.repository import MarketDataRepository
from investment_box.db.session import Database
from investment_box.execution.base import Broker
from investment_box.execution.mock_broker import MockBroker
from investment_box.services.approvals import ApprovalService
from investment_box.services.audit import AuditService
from investment_box.services.portfolio import PortfolioService

log = get_logger(__name__)


@dataclass
class ServiceContainer:
    """Everything a process needs, constructed once."""

    settings: Settings
    secrets: Secrets
    database: Database
    broker: Broker
    market_data: MarketDataRepository
    portfolio: PortfolioService
    audit: AuditService
    approvals: ApprovalService
    calendar: TradingCalendar
    clock: Clock

    @property
    def is_using_mock_broker(self) -> bool:
        return self.broker.name == "mock"

    def startup_banner(self) -> list[str]:
        """Lines to print at startup and post to Telegram on boot.

        Anything that would make a number untrustworthy belongs here, stated
        plainly rather than buried in a log line.
        """
        lines = [
            f"Investment Box -- mode: {self.settings.trading_mode.value.upper()}",
            f"Broker: {self.broker.name} (paper={getattr(self.broker, 'is_paper', True)})",
            f"Data provider: {self.market_data.provider.name}",
            f"Allocated capital: ${self.settings.capital.allocation_usd}",
            f"Autonomy level: {self.settings.engine.autonomy_level.value}",
        ]
        if self.is_using_mock_broker:
            lines.append(
                "NOTE: no broker credentials found -- running against the in-memory mock "
                "broker. Balances and fills are simulated."
            )
        if getattr(self.market_data.provider, "is_synthetic", False):
            lines.append(
                "WARNING: market data is SYNTHETIC. Prices are generated, not real. "
                "Nothing computed from them means anything."
            )
        lines.extend(self.settings.startup_warnings())
        return lines


def build_services(
    *,
    settings: Settings | None = None,
    secrets: Secrets | None = None,
    broker: Broker | None = None,
    database: Database | None = None,
    clock: Clock | None = None,
    configure_logs: bool = True,
) -> ServiceContainer:
    """Construct the container.

    Every dependency is injectable so tests can substitute one piece without
    faking the rest.
    """
    settings = settings or load_settings()
    secrets = secrets or get_secrets()
    clock = clock or SystemClock()

    if configure_logs:
        configure_logging(settings.log_level)

    database = database or Database(settings.db_url)
    database.create_all()

    calendar = TradingCalendar()
    broker = broker or _select_broker(settings, secrets, clock, calendar)
    market_data = MarketDataRepository.from_settings(settings)

    portfolio = PortfolioService(broker, database, settings, clock=clock)
    audit = AuditService(database, settings.trading_mode)
    approvals = ApprovalService(database, audit, clock=clock)

    container = ServiceContainer(
        settings=settings,
        secrets=secrets,
        database=database,
        broker=broker,
        market_data=market_data,
        portfolio=portfolio,
        audit=audit,
        approvals=approvals,
        calendar=calendar,
        clock=clock,
    )
    for line in container.startup_banner():
        log.info("startup", message=line)
    return container


def _select_broker(
    settings: Settings, secrets: Secrets, clock: Clock, calendar: TradingCalendar
) -> Broker:
    """Pick a broker, refusing to guess when the signals conflict."""
    if not secrets.has_alpaca_credentials:
        log.warning(
            "broker.no_credentials",
            action="using mock broker",
            hint="set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env",
        )
        return MockBroker(
            starting_cash=Decimal(str(settings.capital.allocation_usd)),
            clock=clock,
            calendar=calendar,
            settlement_days=settings.settlement.settlement_days,
        )

    wants_live = settings.trading_mode is TradingMode.LIVE
    url_is_live = secrets.is_live_alpaca_url

    if wants_live != url_is_live:
        # A mismatch means someone changed one and forgot the other. Refusing is
        # the only safe answer: guessing "paper" hides a live intent, guessing
        # "live" trades real money by accident.
        raise ValueError(
            f"Refusing to start: trading_mode is '{settings.trading_mode.value}' but "
            f"ALPACA_BASE_URL points at a {'live' if url_is_live else 'paper'} endpoint. "
            f"Make both agree before starting."
        )

    # Phase 5 returns the real AlpacaBroker here. Until the execution layer and
    # its reconciliation logic exist, connecting to a real account would let a
    # half-built engine place orders.
    log.warning(
        "broker.alpaca_not_wired",
        action="using mock broker",
        note="the Alpaca adapter is wired in Phase 5",
    )
    return MockBroker(
        starting_cash=Decimal(str(settings.capital.allocation_usd)),
        clock=clock,
        calendar=calendar,
        settlement_days=settings.settlement.settlement_days,
    )
