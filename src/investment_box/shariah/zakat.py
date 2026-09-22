"""Zakat estimation.

**This produces an estimate, not a ruling.** It is labelled as such everywhere
it surfaces, and it refuses to present a single number without saying which
method produced it.

Scholars differ on how trading assets are treated. The two methods here bracket
the common positions:

* **Full market value** — the whole position is zakatable, on the reasoning
  that shares held for trading are merchandise. Simpler, and yields the larger
  figure.
* **Net zakatable assets** — only the underlying company's zakatable assets
  (cash, receivables, inventory) are counted, via a published ratio. Requires
  data this application does not have for most funds, so it is available only
  where you supply the ratio.

The nisab threshold is entered by you, because it tracks the current gold or
silver price and no hard-coded figure would stay correct.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from investment_box.core.logging import get_logger
from investment_box.core.types import to_money

log = get_logger(__name__)

#: The standard rate on monetary wealth held for a lunar year: 2.5%.
ZAKAT_RATE = Decimal("0.025")


class ZakatMethod(StrEnum):
    FULL_MARKET_VALUE = "full_market_value"
    NET_ZAKATABLE_ASSETS = "net_zakatable_assets"

    @property
    def description(self) -> str:
        return {
            ZakatMethod.FULL_MARKET_VALUE: (
                "The entire market value of each holding is treated as zakatable, on "
                "the reasoning that shares held for trading are merchandise. Yields "
                "the larger figure."
            ),
            ZakatMethod.NET_ZAKATABLE_ASSETS: (
                "Only the underlying companies' zakatable assets are counted, using a "
                "ratio you supply per holding. Requires data this application does not "
                "have by default."
            ),
        }[self]


@dataclass(frozen=True, slots=True)
class ZakatHolding:
    """One position as it enters the calculation."""

    symbol: str
    market_value: Decimal
    #: Fraction of the holding that is zakatable. Required for the net method.
    zakatable_ratio: float | None = None

    def zakatable_value(self, method: ZakatMethod) -> Decimal | None:
        if method is ZakatMethod.FULL_MARKET_VALUE:
            return self.market_value
        if self.zakatable_ratio is None:
            return None
        return to_money(self.market_value * Decimal(str(self.zakatable_ratio)))


@dataclass
class ZakatEstimate:
    """The result. Every field is an estimate and the type says so."""

    as_of: dt.date
    method: ZakatMethod
    holdings: list[ZakatHolding]
    cash: Decimal = Decimal("0.00")
    liabilities: Decimal = Decimal("0.00")
    nisab_threshold: Decimal | None = None
    #: Holdings excluded for lack of a ratio.
    excluded: list[str] = field(default_factory=list)

    @property
    def holdings_value(self) -> Decimal:
        total = Decimal("0")
        for holding in self.holdings:
            value = holding.zakatable_value(self.method)
            if value is not None:
                total += value
        return to_money(total)

    @property
    def zakatable_base(self) -> Decimal:
        return to_money(max(Decimal("0"), self.holdings_value + self.cash - self.liabilities))

    @property
    def meets_nisab(self) -> bool | None:
        """Whether the base reaches the threshold. ``None`` if none was given."""
        if self.nisab_threshold is None:
            return None
        return self.zakatable_base >= self.nisab_threshold

    @property
    def estimated_zakat(self) -> Decimal:
        if self.meets_nisab is False:
            return Decimal("0.00")
        return to_money(self.zakatable_base * ZAKAT_RATE)

    def caveats(self) -> list[str]:
        """Everything that qualifies the number above."""
        out = [
            "This is an ESTIMATE produced by software, not a ruling. Scholars differ "
            "on how trading assets are treated; consult someone qualified.",
            f"Method used: {self.method.value} — {self.method.description}",
        ]
        if self.nisab_threshold is None:
            out.append(
                "No nisab threshold was supplied, so the estimate assumes the "
                "threshold is met. Nisab tracks the current gold or silver price."
            )
        if self.excluded:
            out.append(
                f"Excluded for lack of a zakatable ratio: {', '.join(self.excluded)}. "
                f"Their value is NOT in the figure above."
            )
        out.append(
            "Zakat is due on wealth held for a full lunar year (hawl). This "
            "calculation looks only at a single date and does not check that."
        )
        return out

    def summary(self) -> str:
        return (
            f"Estimated zakat ${self.estimated_zakat} on a base of "
            f"${self.zakatable_base} ({self.method.value}) — an estimate, not a ruling"
        )


def estimate_zakat(
    *,
    as_of: dt.date,
    holdings: list[ZakatHolding],
    cash: Decimal = Decimal("0.00"),
    liabilities: Decimal = Decimal("0.00"),
    method: ZakatMethod = ZakatMethod.FULL_MARKET_VALUE,
    nisab_threshold: Decimal | None = None,
) -> ZakatEstimate:
    """Estimate zakat on a portfolio at a date.

    Holdings that cannot be valued under the chosen method are excluded and
    named, rather than being silently counted as zero.
    """
    excluded = [
        h.symbol for h in holdings if h.zakatable_value(method) is None
    ]
    if excluded:
        log.info("zakat.holdings_excluded", symbols=excluded, method=method.value)

    return ZakatEstimate(
        as_of=as_of,
        method=method,
        holdings=holdings,
        cash=to_money(cash),
        liabilities=to_money(liabilities),
        nisab_threshold=nisab_threshold,
        excluded=excluded,
    )
