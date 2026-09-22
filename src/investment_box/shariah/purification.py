"""Dividend purification.

Even a Shariah-screened fund holds companies with some impermissible income --
the screens permit up to a threshold rather than requiring zero. The portion of
any dividend attributable to that income is not yours to keep, and purification
is the practice of giving it away.

What this module does and does not claim:

* It **computes** the amount from a dividend and a non-permissible income
  ratio, and tracks what has been purified and what is outstanding.
* It does **not** invent the ratio. Fund issuers publish a purification figure
  per share or per dollar; where that is unavailable the entry is recorded as
  ``UNKNOWN`` with a zero amount and flagged, rather than estimated. An invented
  purification figure is worse than an absent one -- it produces a number you
  might act on that has no basis.
* It is **not a fatwa**. Methods differ between scholars; the method used is
  recorded on every entry so the basis of a figure is always visible.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import select

from investment_box.core.audit import AuditSink
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import to_money
from investment_box.db.models import Dividend
from investment_box.db.session import Database

log = get_logger(__name__)


class PurificationMethod(StrEnum):
    """How an amount was derived. Recorded on every entry."""

    #: The issuer published a purification rate. The only authoritative source.
    ISSUER_RATE = "issuer_rate"
    #: A ratio supplied by a screening provider.
    PROVIDER_RATIO = "provider_ratio"
    #: Entered by hand.
    MANUAL = "manual"
    #: No ratio available. Amount is zero and the entry is flagged.
    UNKNOWN = "unknown"

    @property
    def is_authoritative(self) -> bool:
        return self is PurificationMethod.ISSUER_RATE


@dataclass(frozen=True, slots=True)
class PurificationEntry:
    """One dividend and what it implies."""

    symbol: str
    pay_date: dt.date
    gross_amount: Decimal
    non_permissible_ratio: float | None
    amount_due: Decimal
    method: PurificationMethod
    purified_at: dt.datetime | None = None

    @property
    def is_outstanding(self) -> bool:
        return self.purified_at is None and self.amount_due > 0

    @property
    def needs_a_ratio(self) -> bool:
        """True when the amount could not be computed for lack of a ratio."""
        return self.non_permissible_ratio is None


@dataclass
class PurificationReport:
    """Totals over a period, with the gaps stated."""

    period_start: dt.date | None
    period_end: dt.date | None
    entries: list[PurificationEntry]

    @property
    def total_dividends(self) -> Decimal:
        return to_money(sum((e.gross_amount for e in self.entries), Decimal("0")))

    @property
    def total_due(self) -> Decimal:
        return to_money(sum((e.amount_due for e in self.entries), Decimal("0")))

    @property
    def outstanding(self) -> Decimal:
        return to_money(
            sum((e.amount_due for e in self.entries if e.is_outstanding), Decimal("0"))
        )

    @property
    def already_purified(self) -> Decimal:
        return to_money(self.total_due - self.outstanding)

    @property
    def entries_without_a_ratio(self) -> list[PurificationEntry]:
        return [e for e in self.entries if e.needs_a_ratio]

    def warnings(self) -> list[str]:
        out: list[str] = []
        missing = self.entries_without_a_ratio
        if missing:
            symbols = sorted({e.symbol for e in missing})
            out.append(
                f"{len(missing)} dividend(s) across {', '.join(symbols)} have no "
                f"non-permissible income ratio, so no amount could be computed. "
                f"These are NOT zero -- they are unknown. Find each fund's published "
                f"purification rate and enter it."
            )
        non_authoritative = [
            e for e in self.entries if e.amount_due > 0 and not e.method.is_authoritative
        ]
        if non_authoritative:
            out.append(
                f"{len(non_authoritative)} entry/entries use a ratio that did not come "
                f"from the fund issuer. Issuer-published rates are the authoritative "
                f"source; treat the rest as provisional."
            )
        return out

    def summary(self) -> str:
        if not self.entries:
            return "no dividends recorded"
        return (
            f"${self.total_due} due on ${self.total_dividends} of dividends; "
            f"${self.outstanding} still outstanding"
        )


class PurificationTracker:
    """Records dividends and tracks what is owed."""

    def __init__(
        self, database: Database, audit: AuditSink, *, clock: Clock | None = None
    ) -> None:
        self.db = database
        self.audit = audit
        self.clock = clock or SystemClock()

    def record_dividend(
        self,
        *,
        symbol: str,
        pay_date: dt.date,
        gross_amount: Decimal,
        non_permissible_ratio: float | None = None,
        method: PurificationMethod = PurificationMethod.UNKNOWN,
        source: str | None = None,
    ) -> PurificationEntry:
        """Record a dividend and compute what it implies.

        A missing ratio yields a zero amount *and* a flag -- never a guess. The
        report distinguishes "nothing owed" from "we do not know", because
        acting on the first when it is really the second means keeping money
        that is not yours.
        """
        ticker = symbol.upper()
        amount = (
            to_money(gross_amount * Decimal(str(non_permissible_ratio)))
            if non_permissible_ratio is not None
            else Decimal("0.00")
        )

        with self.db.session() as session:
            existing = session.scalar(
                select(Dividend).where(
                    Dividend.symbol == ticker, Dividend.pay_date == pay_date
                )
            )
            if existing is not None:
                existing.gross_amount = to_money(gross_amount)
                existing.non_permissible_ratio = non_permissible_ratio
                existing.purification_due = amount
                existing.ratio_source = source or method.value
            else:
                session.add(
                    Dividend(
                        symbol=ticker,
                        pay_date=pay_date,
                        gross_amount=to_money(gross_amount),
                        non_permissible_ratio=non_permissible_ratio,
                        purification_due=amount,
                        ratio_source=source or method.value,
                    )
                )

        self.audit.record(
            "purification.dividend_recorded",
            f"{ticker} {pay_date}: ${to_money(gross_amount)} gross, "
            f"${amount} to purify ({method.value})",
            actor="purification",
            symbol=ticker,
        )
        return PurificationEntry(
            symbol=ticker,
            pay_date=pay_date,
            gross_amount=to_money(gross_amount),
            non_permissible_ratio=non_permissible_ratio,
            amount_due=amount,
            method=method,
        )

    def mark_purified(self, symbol: str, pay_date: dt.date, *, actor: str = "user") -> bool:
        """Record that an amount has been given away."""
        with self.db.session() as session:
            row = session.scalar(
                select(Dividend).where(
                    Dividend.symbol == symbol.upper(), Dividend.pay_date == pay_date
                )
            )
            if row is None:
                return False
            row.purified_at = self.clock.now()
            amount = row.purification_due

        self.audit.record(
            "purification.paid",
            f"{symbol.upper()} {pay_date}: ${amount} purified",
            actor=actor,
            symbol=symbol.upper(),
        )
        return True

    def report(
        self, *, start: dt.date | None = None, end: dt.date | None = None
    ) -> PurificationReport:
        with self.db.session() as session:
            statement = select(Dividend).order_by(Dividend.pay_date.desc())
            if start is not None:
                statement = statement.where(Dividend.pay_date >= start)
            if end is not None:
                statement = statement.where(Dividend.pay_date <= end)
            rows = list(session.scalars(statement).all())
            for row in rows:
                session.expunge(row)

        entries = [
            PurificationEntry(
                symbol=row.symbol,
                pay_date=row.pay_date,
                gross_amount=row.gross_amount,
                non_permissible_ratio=row.non_permissible_ratio,
                amount_due=row.purification_due or Decimal("0.00"),
                method=_method_from(row.ratio_source),
                purified_at=row.purified_at,
            )
            for row in rows
        ]
        return PurificationReport(period_start=start, period_end=end, entries=entries)

    def outstanding_total(self) -> Decimal:
        return self.report().outstanding

    def to_csv(self, report: PurificationReport | None = None) -> str:
        """Export as CSV, with the method on every row.

        The method column is not optional: a figure whose basis is unknown
        should not be presented as equivalent to one from an issuer rate.
        """
        report = report or self.report()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "symbol",
                "pay_date",
                "gross_dividend_usd",
                "non_permissible_ratio",
                "purification_due_usd",
                "method",
                "purified_at",
                "status",
            ]
        )
        for entry in report.entries:
            writer.writerow(
                [
                    entry.symbol,
                    entry.pay_date.isoformat(),
                    f"{entry.gross_amount:.2f}",
                    (
                        f"{entry.non_permissible_ratio:.6f}"
                        if entry.non_permissible_ratio is not None
                        else "UNKNOWN"
                    ),
                    f"{entry.amount_due:.2f}",
                    entry.method.value,
                    entry.purified_at.isoformat() if entry.purified_at else "",
                    "purified" if entry.purified_at else "outstanding",
                ]
            )
        if report.entries_without_a_ratio:
            writer.writerow([])
            writer.writerow(
                [
                    "NOTE: rows with ratio UNKNOWN are not zero. No ratio was "
                    "available, so no amount could be computed."
                ]
            )
        return buffer.getvalue()


def _method_from(source: str | None) -> PurificationMethod:
    if not source:
        return PurificationMethod.UNKNOWN
    try:
        return PurificationMethod(source)
    except ValueError:
        return PurificationMethod.PROVIDER_RATIO
