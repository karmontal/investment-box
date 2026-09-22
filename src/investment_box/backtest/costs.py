"""Transaction cost model.

Every backtest pays these. They are not optional and cannot be zeroed from the
UI, because a costless backtest is not a backtest -- at ~$100 positions the
frictions are a large fraction of any plausible edge, and a model that ignores
them will recommend strategies that lose money in practice.

What is modelled:

* **Commission** -- zero at Alpaca for equities, but kept as a parameter so a
  different broker can be evaluated honestly.
* **Half-spread**, paid on entry *and* exit. This is the big one for thinly
  traded funds like UMMA.
* **Slippage** -- the gap between the decision price and the achieved fill.
* **SEC fee and FINRA TAF** -- sells only, small but real.

What is *not* modelled, and would make results worse if it were: market impact
(negligible at this size), partial fills, and the occasional day a limit order
simply does not fill. The last one matters and is noted in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from investment_box.config.schema import CostsConfig
from investment_box.core.types import Side

BPS = Decimal("10000")


@dataclass(frozen=True, slots=True)
class TradeCost:
    """Itemised cost of one side of a trade."""

    commission: float
    spread: float
    slippage: float
    regulatory: float

    @property
    def total(self) -> float:
        return self.commission + self.spread + self.slippage + self.regulatory

    def as_bps_of(self, notional: float) -> float:
        return (self.total / notional * 10_000) if notional > 0 else 0.0


class CostModel:
    """Computes costs and the effective fill price."""

    def __init__(self, config: CostsConfig) -> None:
        self.config = config

    def cost(self, *, price: float, quantity: float, side: Side) -> TradeCost:
        """Itemised cost for one execution."""
        notional = price * quantity

        commission = max(
            self.config.commission_per_share * quantity,
            self.config.commission_min if quantity > 0 else 0.0,
        )
        spread = notional * self.config.spread_bps / 10_000
        slippage = notional * self.config.slippage_bps / 10_000

        # SEC fee and FINRA TAF are charged on sales only.
        regulatory = 0.0
        if side is Side.SELL:
            regulatory = (
                notional * self.config.sec_fee_bps / 10_000
                + self.config.finra_taf_per_share * quantity
            )

        return TradeCost(
            commission=commission, spread=spread, slippage=slippage, regulatory=regulatory
        )

    def fill_price(self, *, reference_price: float, side: Side) -> float:
        """The price actually achieved, moving against us.

        Spread and slippage are expressed in the price rather than as a
        separate charge, because that is how they behave: you buy higher and
        sell lower, and the position is marked from there.
        """
        drift = (self.config.spread_bps / 2 + self.config.slippage_bps) / 10_000
        return reference_price * (1 + drift) if side is Side.BUY else reference_price * (1 - drift)

    def round_trip_cost(self, *, price: float, quantity: float) -> float:
        """Total cost of entering and exiting a position at the same price."""
        buy = self.cost(price=price, quantity=quantity, side=Side.BUY)
        sell = self.cost(price=price, quantity=quantity, side=Side.SELL)
        return buy.total + sell.total

    def breakeven_move(self, *, price: float, quantity: float) -> float:
        """The price move needed just to cover costs, as a fraction.

        The number that decides whether a strategy is viable at this account
        size. If a strategy's average winner is smaller than this, it cannot
        work no matter how often it is right.
        """
        notional = price * quantity
        if notional <= 0:
            return 0.0
        return self.round_trip_cost(price=price, quantity=quantity) / notional
