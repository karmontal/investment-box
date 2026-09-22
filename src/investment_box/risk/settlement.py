"""The settled-cash ledger.

In a cash account, only *settled* cash may fund a purchase. Spending the
proceeds of a sale before they settle is a good-faith violation, and three of
them gets the account restricted to settled-cash-only trading for 90 days.

At ~$500 this is the binding constraint on the whole system, not a footnote.
With T+1 settlement and a two-trading-day minimum hold, each dollar realistically
cycles two or three times a month. The ledger below is what makes the engine
refuse trades it cannot actually fund, rather than discovering the problem from
a broker rejection.

The ledger is authoritative over the broker's own "buying power" figure, which
at some brokers includes unsettled proceeds.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import select

from investment_box.core.clock import UTC, Clock, SystemClock, TradingCalendar
from investment_box.core.errors import SettlementError
from investment_box.core.logging import get_logger
from investment_box.core.types import CashLedger, Symbol, to_money
from investment_box.db.models import SettlementEntry
from investment_box.db.session import Database

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PendingProceeds:
    """One sale awaiting settlement."""

    symbol: str
    amount: Decimal
    sold_on: dt.date
    settles_on: dt.date

    def is_settled_on(self, day: dt.date) -> bool:
        return day >= self.settles_on

    def days_until_settled(self, day: dt.date, calendar: TradingCalendar) -> int:
        if self.is_settled_on(day):
            return 0
        return calendar.trading_days_between(day, self.settles_on)


@dataclass
class SettlementSnapshot:
    """The cash picture, split by what may actually be spent."""

    as_of: dt.date
    settled: Decimal
    pending: list[PendingProceeds] = field(default_factory=list)
    #: Cash claimed by orders that are submitted but not yet filled.
    reserved: Decimal = Decimal("0.00")

    @property
    def unsettled(self) -> Decimal:
        return to_money(sum((p.amount for p in self.pending), Decimal("0")))

    @property
    def total(self) -> Decimal:
        return to_money(self.settled + self.unsettled)

    @property
    def available(self) -> Decimal:
        """What may fund a purchase right now. Never includes unsettled cash."""
        return to_money(max(Decimal("0"), self.settled - self.reserved))

    @property
    def next_settlement(self) -> PendingProceeds | None:
        return min(self.pending, key=lambda p: p.settles_on) if self.pending else None

    def as_ledger(self) -> CashLedger:
        return CashLedger(settled=self.settled, unsettled=self.unsettled, reserved=self.reserved)

    def explain(self) -> str:
        if not self.pending:
            return f"${self.available} available, all settled"
        soonest = self.next_settlement
        return (
            f"${self.available} available now; ${self.unsettled} unsettled"
            + (f", next settling {soonest.settles_on}" if soonest else "")
        )


class SettlementLedger:
    """Tracks sale proceeds through T+1 settlement.

    Persisted, because a restart must not lose track of money that is not yet
    spendable. An in-memory ledger would silently reset to "everything is
    settled" after a crash, which is precisely the wrong direction to fail.
    """

    def __init__(
        self,
        database: Database,
        calendar: TradingCalendar,
        settlement_days: int = 1,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.db = database
        self.calendar = calendar
        self.settlement_days = settlement_days
        self.clock = clock or SystemClock()

    def today(self) -> dt.date:
        return self.clock.now().astimezone(UTC).date()

    def record_sale(
        self, symbol: Symbol | str, proceeds: Decimal, *, sold_on: dt.date | None = None,
        trade_id: int | None = None,
    ) -> PendingProceeds:
        """Register sale proceeds as unsettled until their settlement date."""
        day = sold_on or self.today()
        settles_on = self.calendar.settlement_date(day, self.settlement_days)

        with self.db.session() as session:
            session.add(
                SettlementEntry(
                    symbol=str(symbol).upper(),
                    amount=to_money(proceeds),
                    sold_on=day,
                    settles_on=settles_on,
                    settled=False,
                    trade_id=trade_id,
                )
            )

        log.info(
            "settlement.recorded",
            symbol=str(symbol),
            amount=str(to_money(proceeds)),
            settles_on=str(settles_on),
        )
        return PendingProceeds(
            symbol=str(symbol).upper(),
            amount=to_money(proceeds),
            sold_on=day,
            settles_on=settles_on,
        )

    def settle_due(self, *, as_of: dt.date | None = None) -> Decimal:
        """Mark matured proceeds settled. Returns the amount released."""
        day = as_of or self.today()
        released = Decimal("0")

        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(SettlementEntry).where(
                        SettlementEntry.settled.is_(False),
                        SettlementEntry.settles_on <= day,
                    )
                ).all()
            )
            for row in rows:
                row.settled = True
                released += row.amount

        if released:
            log.info("settlement.released", amount=str(to_money(released)), count=len(rows))
        return to_money(released)

    def pending(self, *, as_of: dt.date | None = None) -> list[PendingProceeds]:
        """Proceeds that have not yet settled as of ``as_of``."""
        day = as_of or self.today()
        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(SettlementEntry)
                    .where(
                        SettlementEntry.settled.is_(False),
                        SettlementEntry.settles_on > day,
                    )
                    .order_by(SettlementEntry.settles_on)
                ).all()
            )
            for row in rows:
                session.expunge(row)

        return [
            PendingProceeds(
                symbol=row.symbol,
                amount=row.amount,
                sold_on=row.sold_on,
                settles_on=row.settles_on,
            )
            for row in rows
        ]

    def snapshot(
        self, settled_cash: Decimal, *, as_of: dt.date | None = None,
        reserved: Decimal = Decimal("0.00"),
    ) -> SettlementSnapshot:
        """The current cash picture.

        ``settled_cash`` comes from the broker. This subtracts what the ledger
        knows is still unsettled, because some brokers report a buying-power
        figure that includes it.
        """
        day = as_of or self.today()
        self.settle_due(as_of=day)
        pending = self.pending(as_of=day)
        unsettled = to_money(sum((p.amount for p in pending), Decimal("0")))

        return SettlementSnapshot(
            as_of=day,
            settled=to_money(max(Decimal("0"), settled_cash - unsettled)),
            pending=pending,
            reserved=reserved,
        )

    def assert_affordable(
        self, cost: Decimal, snapshot: SettlementSnapshot
    ) -> None:
        """Refuse a purchase that unsettled cash would be needed to fund.

        Raises:
            SettlementError: Always, rather than returning a boolean. A
                good-faith violation is not a condition to branch on; it is
                something the engine must never do.
        """
        if cost <= snapshot.available:
            return
        shortfall = to_money(cost - snapshot.available)
        soonest = snapshot.next_settlement
        when = f" ${snapshot.unsettled} settles {soonest.settles_on}" if soonest else ""
        raise SettlementError(
            f"needs ${to_money(cost)} but only ${snapshot.available} is settled "
            f"(short ${shortfall}).{when} Spending unsettled proceeds would be a "
            f"good-faith violation."
        )

    def can_afford(self, cost: Decimal, snapshot: SettlementSnapshot) -> bool:
        """Non-raising variant, for sizing decisions rather than order gates."""
        return cost <= snapshot.available
