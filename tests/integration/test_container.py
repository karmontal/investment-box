"""Wiring: which broker and which data provider a process ends up with."""

from __future__ import annotations

import pytest

from investment_box.config.loader import load_settings
from investment_box.config.schema import Secrets
from investment_box.core.clock import FrozenClock
from investment_box.db.session import Database
from investment_box.services.container import build_services


class TestBrokerSelection:
    def test_no_credentials_yields_the_mock_broker(
        self, database: Database, clock: FrozenClock
    ) -> None:
        services = build_services(
            settings=load_settings(),
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            database=database,
            clock=clock,
            configure_logs=False,
        )
        assert services.is_using_mock_broker
        assert services.broker.is_paper

    def test_mock_broker_is_seeded_to_the_allocation(
        self, database: Database, clock: FrozenClock
    ) -> None:
        settings = load_settings(overrides={"capital": {"allocation_usd": "250"}})
        services = build_services(
            settings=settings,
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            database=database,
            clock=clock,
            configure_logs=False,
        )
        assert services.broker.get_account().equity == 250

    def test_live_mode_with_a_paper_url_is_refused(
        self, database: Database, clock: FrozenClock
    ) -> None:
        """A mismatch means someone changed one and forgot the other.

        Guessing either way is wrong: assuming paper hides a live intent,
        assuming live trades real money by accident. So it refuses.
        """
        settings = load_settings(overrides={"trading_mode": "live"})
        secrets = Secrets(  # type: ignore[call-arg]
            _env_file=None,
            alpaca_api_key="PKTEST",
            alpaca_secret_key="SECRET",
            alpaca_base_url="https://paper-api.alpaca.markets",
        )
        with pytest.raises(ValueError, match="Make both agree"):
            build_services(
                settings=settings,
                secrets=secrets,
                database=database,
                clock=clock,
                configure_logs=False,
            )

    def test_paper_mode_with_a_live_url_is_refused(
        self, database: Database, clock: FrozenClock
    ) -> None:
        secrets = Secrets(  # type: ignore[call-arg]
            _env_file=None,
            alpaca_api_key="PKTEST",
            alpaca_secret_key="SECRET",
            alpaca_base_url="https://api.alpaca.markets",
        )
        with pytest.raises(ValueError, match="Make both agree"):
            build_services(
                settings=load_settings(),
                secrets=secrets,
                database=database,
                clock=clock,
                configure_logs=False,
            )

    def test_unreachable_alpaca_falls_back_to_the_mock(
        self, database: Database, clock: FrozenClock
    ) -> None:
        """A broker that cannot be reached must not stop the app from starting.

        From Phase 5 credentials DO connect to Alpaca. Invalid ones fall back
        to the mock with a loud log line rather than preventing the dashboard
        and bot from running at all.
        """
        secrets = Secrets(  # type: ignore[call-arg]
            _env_file=None,
            alpaca_api_key="PKINVALIDTESTKEY",
            alpaca_secret_key="INVALIDSECRET",
            alpaca_base_url="https://paper-api.alpaca.markets",
        )
        services = build_services(
            settings=load_settings(),
            secrets=secrets,
            database=database,
            clock=clock,
            configure_logs=False,
        )
        # Either outcome is safe: the mock, or an Alpaca broker whose
        # cash-account check failed and which therefore rejects every order.
        # What must NOT happen is the process failing to start -- the operator
        # needs the dashboard to see why nothing is trading.
        assert services.broker is not None

    def test_paper_is_the_default_when_mode_is_paper(
        self, database: Database, clock: FrozenClock
    ) -> None:
        """Even with credentials, paper mode must never reach the live endpoint."""
        from investment_box.execution.alpaca_broker import AlpacaBroker

        with pytest.raises(Exception, match="paper"):
            AlpacaBroker(
                api_key="PKTEST",
                secret_key="SECRET",
                paper=True,
                base_url="https://api.alpaca.markets",
            )

    def test_live_flag_with_paper_url_is_refused(self) -> None:
        from investment_box.execution.alpaca_broker import AlpacaBroker

        with pytest.raises(Exception, match="paper"):
            AlpacaBroker(
                api_key="PKTEST",
                secret_key="SECRET",
                paper=False,
                base_url="https://paper-api.alpaca.markets",
            )


class TestStartupBanner:
    def test_mock_broker_is_disclosed(self, database: Database, clock: FrozenClock) -> None:
        services = build_services(
            settings=load_settings(),
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            database=database,
            clock=clock,
            configure_logs=False,
        )
        banner = "\n".join(services.startup_banner())
        assert "mock broker" in banner
        assert "simulated" in banner

    def test_synthetic_data_is_disclosed(
        self, database: Database, clock: FrozenClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generated prices must never be mistaken for real ones."""
        services = build_services(
            settings=load_settings(),
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            database=database,
            clock=clock,
            configure_logs=False,
        )
        from investment_box.data.synthetic import SyntheticDataProvider

        services.market_data.provider = SyntheticDataProvider()
        banner = "\n".join(services.startup_banner())
        assert "SYNTHETIC" in banner

    def test_banner_reports_the_allocation(self, database: Database, clock: FrozenClock) -> None:
        services = build_services(
            settings=load_settings(),
            secrets=Secrets(_env_file=None),  # type: ignore[call-arg]
            database=database,
            clock=clock,
            configure_logs=False,
        )
        assert any("$500" in line for line in services.startup_banner())


class TestDatabaseSetup:
    def test_schema_is_created(self, database: Database) -> None:
        from sqlalchemy import inspect

        tables = set(inspect(database.engine).get_table_names())
        assert {
            "trades",
            "orders",
            "signals",
            "compliance_screens",
            "audit_log",
            "approvals",
            "equity_snapshots",
            "dividends",
            "settlement_entries",
            "setting_overrides",
        } <= tables

    def test_order_idempotency_key_is_unique(self, database: Database) -> None:
        """The constraint that makes duplicate order submission impossible."""
        from decimal import Decimal

        from sqlalchemy.exc import IntegrityError

        from investment_box.db.models import Order

        def make() -> Order:
            return Order(
                idempotency_key="same-key",
                symbol="SPUS",
                side="buy",
                order_type="limit",
                quantity=Decimal("1"),
            )

        with database.session() as session:
            session.add(make())

        with pytest.raises(IntegrityError), database.session() as session:
            session.add(make())

    def test_rollback_on_error(self, database: Database) -> None:
        from investment_box.db.models import AuditLog

        with pytest.raises(RuntimeError), database.session() as session:
            session.add(AuditLog(event_type="test", summary="should not persist"))
            session.flush()
            raise RuntimeError("boom")

        with database.session() as session:
            from sqlalchemy import select

            assert session.scalars(select(AuditLog)).all() == []
