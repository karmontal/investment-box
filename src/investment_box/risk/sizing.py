"""Position sizing.

Two methods, both answering the same question: how many shares can we buy such
that hitting the stop costs no more than the per-trade risk budget?

* **ATR-based** (default) — stop distance comes from the instrument's own
  volatility, so a calm fund gets a tighter stop and more shares, and a
  volatile one a wider stop and fewer, for the same dollar risk.
* **Fixed-fractional** — a flat percentage stop. Simpler, and worse: it gives
  a $17 sukuk fund and a $75 equity fund the same stop distance in percentage
  terms regardless of how differently they move.

Everything here rounds **down**. Rounding a position up to reach a nicer number
spends more than the risk budget allows, which is the one direction that must
never happen.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from investment_box.config.schema import Settings
from investment_box.core.logging import get_logger
from investment_box.core.types import to_money, to_qty

log = get_logger(__name__)

#: Below this the position is not worth opening: the round-trip friction plus
#: whole-share rounding error dominate whatever edge the signal has.
MIN_POSITION_USD = Decimal("20.00")


@dataclass(frozen=True, slots=True)
class PositionSize:
    """A sizing decision, including the decision not to trade."""

    symbol: str
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal | None
    take_profit_price: Decimal | None
    notional: Decimal
    risk_amount: Decimal
    is_fractional: bool
    #: Why this size, or why zero.
    reason: str

    @property
    def is_tradable(self) -> bool:
        return self.quantity > 0

    @property
    def risk_pct_of(self) -> Decimal:
        return self.risk_amount

    def describe(self) -> str:
        if not self.is_tradable:
            return f"{self.symbol}: not sized — {self.reason}"
        return (
            f"{self.symbol}: {self.quantity} @ ${self.entry_price} "
            f"(${self.notional} notional, ${self.risk_amount} at risk) — {self.reason}"
        )


class PositionSizer:
    """Turns a signal plus a price into a share count."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def risk_budget(self) -> Decimal:
        """Dollars risked per trade: a fraction of *allocated* capital."""
        return to_money(
            self.settings.capital.allocation_usd
            * Decimal(str(self.settings.risk.risk_per_trade_pct))
        )

    @property
    def max_position_notional(self) -> Decimal:
        return to_money(
            self.settings.capital.allocation_usd
            * Decimal(str(self.settings.risk.max_position_pct))
        )

    def size(
        self,
        *,
        symbol: str,
        entry_price: Decimal,
        available_cash: Decimal,
        atr: Decimal | None = None,
        stop_atr_mult: float | None = None,
        take_profit_atr_mult: float | None = None,
        allow_fractional: bool | None = None,
    ) -> PositionSize:
        """Compute a position size, or explain why there isn't one."""
        fractional_ok = (
            self.settings.execution.fractional_enabled
            if allow_fractional is None
            else allow_fractional
        )

        if entry_price <= 0:
            return self._none(symbol, entry_price, "entry price is not positive")

        stop_price = self._stop_price(entry_price, atr, stop_atr_mult)
        take_profit = self._take_profit(entry_price, atr, take_profit_atr_mult)

        if stop_price is None:
            return self._none(
                symbol, entry_price,
                "no stop distance could be computed (ATR unavailable and no "
                "fixed-fractional fallback configured)",
            )

        stop_distance = entry_price - stop_price
        if stop_distance <= 0:
            return self._none(symbol, entry_price, "stop is at or above the entry price")

        # The core equation: shares such that (entry - stop) * shares <= budget.
        by_risk = self.risk_budget / stop_distance
        by_position_cap = self.max_position_notional / entry_price
        by_cash = available_cash / entry_price

        raw = min(by_risk, by_position_cap, by_cash)
        binding = self._binding_constraint(by_risk, by_position_cap, by_cash)

        if fractional_ok:
            quantity = to_qty(raw.quantize(Decimal("0.000001"), rounding=ROUND_DOWN))
            is_fractional = quantity != quantity.to_integral_value()
        else:
            quantity = raw.quantize(Decimal("1"), rounding=ROUND_DOWN)
            is_fractional = False

        if quantity <= 0:
            affordable = to_money(
                min(
                    self.risk_budget / stop_distance * entry_price,
                    self.max_position_notional,
                    available_cash,
                )
            )
            return self._none(
                symbol, entry_price,
                f"one whole share costs ${entry_price} but the budget allows "
                f"${affordable}"
                + ("" if fractional_ok else "; fractional trading is disabled"),
            )

        notional = to_money(quantity * entry_price)
        if notional < MIN_POSITION_USD:
            return self._none(
                symbol, entry_price,
                f"position would be ${notional}, below the ${MIN_POSITION_USD} minimum "
                f"where friction and rounding dominate any edge",
            )

        return PositionSize(
            symbol=symbol,
            quantity=quantity,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit,
            notional=notional,
            risk_amount=to_money(quantity * stop_distance),
            is_fractional=is_fractional,
            reason=f"{binding} was the binding constraint",
        )

    # -------------------------------------------------------------- internals

    def _stop_price(
        self, entry: Decimal, atr: Decimal | None, stop_atr_mult: float | None
    ) -> Decimal | None:
        method = self.settings.risk.sizing_method
        multiplier = Decimal(
            str(stop_atr_mult or self.settings.risk.default_stop_loss_atr_mult)
        )

        if method == "atr" and atr is not None and atr > 0:
            return to_money(entry - atr * multiplier)

        if method == "fixed_fractional" or atr is None or atr <= 0:
            # Fall back to a fixed percentage derived from the risk settings,
            # so a missing ATR degrades rather than blocking the trade.
            fraction = Decimal(str(self.settings.risk.max_position_pct)) / Decimal("2")
            return to_money(entry * (Decimal("1") - fraction))

        return None

    def _take_profit(
        self, entry: Decimal, atr: Decimal | None, mult: float | None
    ) -> Decimal | None:
        multiplier = mult or self.settings.risk.default_take_profit_atr_mult
        if atr is not None and atr > 0:
            return to_money(entry + atr * Decimal(str(multiplier)))
        return None

    @staticmethod
    def _binding_constraint(by_risk: Decimal, by_cap: Decimal, by_cash: Decimal) -> str:
        smallest = min(by_risk, by_cap, by_cash)
        if smallest == by_cash:
            return "available settled cash"
        if smallest == by_risk:
            return "the per-trade risk budget"
        return "the maximum position size"

    @staticmethod
    def _none(symbol: str, entry: Decimal, reason: str) -> PositionSize:
        return PositionSize(
            symbol=symbol,
            quantity=Decimal("0"),
            entry_price=entry,
            stop_price=None,
            take_profit_price=None,
            notional=Decimal("0"),
            risk_amount=Decimal("0"),
            is_fractional=False,
            reason=reason,
        )
