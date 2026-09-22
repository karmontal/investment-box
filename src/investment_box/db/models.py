"""SQLAlchemy ORM models.

Two conventions hold throughout:

* Money is ``Numeric(18, 6)``, never ``Float``. SQLite has no native decimal
  type, so SQLAlchemy stores these as strings and returns ``Decimal`` -- which
  is what we want, and why the session sets a decimal-safe type on read.
* Every timestamp column is UTC, stored timezone-aware. Display conversion
  happens in the UI and Telegram layers, never in the database.

The audit tables (:class:`AuditLog`, :class:`ComplianceScreen`) are
append-only by convention: nothing in this application updates or deletes rows
in them.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Dialect,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class UTCDateTime(TypeDecorator[dt.datetime]):
    """A timestamp that is always timezone-aware UTC on the way in and out.

    SQLite has no timezone support: it stores whatever it is given and returns
    a naive datetime. That naive value then blows up -- or worse, silently
    compares wrong -- against the timezone-aware datetimes used everywhere else
    in this codebase.

    Fixing that at each call site leaves the same trap for the next column, so
    it is fixed here instead: writes must be aware (a naive write is a bug and
    raises), and reads are always returned as aware UTC.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(
        self, value: dt.datetime | None, dialect: Dialect
    ) -> dt.datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                f"Refusing to store a naive datetime: {value!r}. Every timestamp in "
                f"this application is timezone-aware UTC."
            )
        return value.astimezone(dt.UTC)

    def process_result_value(
        self, value: dt.datetime | None, dialect: Dialect
    ) -> dt.datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=dt.UTC) if value.tzinfo is None else value.astimezone(dt.UTC)


MONEY = Numeric(18, 6)
QTY = Numeric(18, 8)
#: Use this, never bare DateTime, for any column holding a moment in time.
TIMESTAMP = UTCDateTime()


class Base(DeclarativeBase):
    """Declarative base with a UTC-aware timestamp default."""


def _utcnow() -> dt.datetime:
    return dt.datetime.now(tz=dt.UTC)


class TimestampMixin:
    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, default=_utcnow, nullable=False
    )


class Signal(Base, TimestampMixin):
    """A strategy's opinion at a point in time, stored whether or not it traded.

    Keeping rejected signals is what makes the "why didn't it trade?" question
    answerable later.
    """

    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Bar date the signal was computed from -- NOT the date it may execute on.
    as_of_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    score: Mapped[float | None] = mapped_column(nullable=True)
    probability: Mapped[float | None] = mapped_column(nullable=True)
    expected_return_low: Mapped[float | None] = mapped_column(nullable=True)
    expected_return_high: Mapped[float | None] = mapped_column(nullable=True)
    suggested_holding_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(nullable=True)
    #: Why this signal did or did not become an order.
    disposition: Mapped[str] = mapped_column(String(32), nullable=False, default="generated")
    disposition_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_signals_symbol_date", "symbol", "as_of_date"),)


class Order(Base, TimestampMixin):
    """A broker order, including ones that were rejected or never filled."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    #: Deterministic key derived from (symbol, side, intent, date). The unique
    #: constraint is what makes order submission idempotent across retries and
    #: process restarts -- a duplicate insert fails rather than double-trading.
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(16), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(QTY, nullable=False)
    is_fractional: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    limit_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    stop_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    #: True when the stop is tracked by the engine rather than held at the
    #: broker -- the case for every fractional order. Unprotected if we crash.
    stop_is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    filled_quantity: Mapped[Decimal] = mapped_column(QTY, nullable=False, default=Decimal("0"))
    filled_avg_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    submitted_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    filled_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    signal_id: Mapped[int | None] = mapped_column(ForeignKey("signals.id"), nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    trading_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")

    signal: Mapped[Signal | None] = relationship()


class Trade(Base, TimestampMixin):
    """A completed or open round trip, with its compliance provenance.

    The compliance columns are snapshots taken *at entry*, not foreign keys to
    the current screen. The audit question is "what did we know when we
    traded?", which a live join would silently answer wrong after a re-screen.
    """

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False, default="buy")

    quantity: Mapped[Decimal] = mapped_column(QTY, nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    entry_at: Mapped[dt.datetime] = mapped_column(TIMESTAMP, nullable=False)
    entry_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)

    exit_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    exit_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    exit_date: Mapped[dt.date | None] = mapped_column(Date, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)

    stop_loss_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    take_profit_price: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)

    gross_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    costs_paid: Mapped[Decimal] = mapped_column(MONEY, nullable=False, default=Decimal("0"))
    net_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    holding_trading_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: Compliance snapshot at entry -- see the class docstring.
    compliance_status_at_entry: Mapped[str] = mapped_column(String(16), nullable=False)
    compliance_source_at_entry: Mapped[str] = mapped_column(String(64), nullable=False)
    compliance_screened_at: Mapped[dt.datetime | None] = mapped_column(
        TIMESTAMP, nullable=True
    )

    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    trading_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")

    @property
    def is_open(self) -> bool:
        return self.exit_at is None


class ComplianceScreen(Base, TimestampMixin):
    """One screening result. Append-only; never updated in place."""

    __tablename__ = "compliance_screens"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    screened_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, nullable=False, default=_utcnow, index=True
    )
    #: Which denominator the ratios used. Recorded because AAOIFI and other
    #: boards differ, and a ratio is meaningless without it.
    ratio_denominator: Mapped[str | None] = mapped_column(String(16), nullable=True)
    debt_ratio: Mapped[float | None] = mapped_column(nullable=True)
    interest_securities_ratio: Mapped[float | None] = mapped_column(nullable=True)
    non_permissible_revenue_ratio: Mapped[float | None] = mapped_column(nullable=True)
    business_activity_flags: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_screens_symbol_time", "symbol", "screened_at"),)


class Approval(Base, TimestampMixin):
    """A human decision requested by the engine (Phase 2 wires this to Telegram).

    ``expires_at`` plus a default of REJECT encodes the rule that a timeout is
    never an approval.
    """

    __tablename__ = "approvals"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    expires_at: Mapped[dt.datetime] = mapped_column(TIMESTAMP, nullable=False)
    responded_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP, nullable=True)
    responder_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuditLog(Base, TimestampMixin):
    """Append-only record of every decision, action and refusal."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(48), nullable=False, index=True)
    actor: Mapped[str] = mapped_column(String(32), nullable=False, default="engine")
    symbol: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    detail_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    trading_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")


class EquitySnapshot(Base):
    """Daily account snapshot, the source of the equity curve and drawdown checks."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    snapshot_date: Mapped[dt.date] = mapped_column(Date, nullable=False, unique=True)
    taken_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, nullable=False, default=_utcnow
    )
    equity: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    cash_settled: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    cash_unsettled: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    positions_value: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    day_pnl: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    open_positions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trading_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="paper")


class Dividend(Base, TimestampMixin):
    """Dividend received, with the ratio used for purification."""

    __tablename__ = "dividends"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    pay_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    gross_amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    non_permissible_ratio: Mapped[float | None] = mapped_column(nullable=True)
    purification_due: Mapped[Decimal | None] = mapped_column(MONEY, nullable=True)
    ratio_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    purified_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP, nullable=True)

    __table_args__ = (UniqueConstraint("symbol", "pay_date", name="uq_dividend_symbol_date"),)


class SettlementEntry(Base, TimestampMixin):
    """Sale proceeds pending settlement. Drives the settled-cash ledger."""

    __tablename__ = "settlement_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[Decimal] = mapped_column(MONEY, nullable=False)
    sold_on: Mapped[dt.date] = mapped_column(Date, nullable=False)
    settles_on: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    settled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trade_id: Mapped[int | None] = mapped_column(ForeignKey("trades.id"), nullable=True)


class SettingOverride(Base, TimestampMixin):
    """UI-editable settings that outlive a restart (symbol rules, autonomy, pause state)."""

    __tablename__ = "setting_overrides"

    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP, nullable=False, default=_utcnow, onupdate=_utcnow
    )
    updated_by: Mapped[str] = mapped_column(String(32), nullable=False, default="ui")
