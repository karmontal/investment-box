"""Core value types and enums.

Money is `Decimal` everywhere that touches an order, a cash balance or a P&L
figure. Floats are used only for statistics and indicators, where the rounding
is irrelevant, never for anything the broker or the ledger sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import NewType

Symbol = NewType("Symbol", str)

#: Two-decimal quantiser for USD amounts.
CENT = Decimal("0.01")
#: Alpaca accepts up to 9 decimal places on fractional quantities; we use 6.
QTY_PRECISION = Decimal("0.000001")


def to_money(value: Decimal | float | int | str) -> Decimal:
    """Quantise a value to whole cents, rounding half up.

    Accepts floats for convenience at provider boundaries, but converts via
    ``str`` so that ``to_money(0.1 + 0.2)`` does not inherit binary noise.
    """
    dec = Decimal(str(value)) if not isinstance(value, Decimal) else value
    return dec.quantize(CENT, rounding=ROUND_HALF_UP)


def to_qty(value: Decimal | float | int | str) -> Decimal:
    """Quantise a share quantity to six decimal places."""
    dec = Decimal(str(value)) if not isinstance(value, Decimal) else value
    return dec.quantize(QTY_PRECISION, rounding=ROUND_HALF_UP)


class TradingMode(StrEnum):
    """Paper is the default and the only mode reachable without explicit opt-in."""

    PAPER = "paper"
    LIVE = "live"


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(StrEnum):
    DAY = "day"
    GTC = "gtc"


class OrderStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AssetClass(StrEnum):
    EQUITY_US = "equity_us"
    EQUITY_INTL = "equity_intl"
    EQUITY_SECTOR = "equity_sector"
    SUKUK = "sukuk"
    REIT = "reit"
    UNKNOWN = "unknown"


class ComplianceStatus(StrEnum):
    """Shariah screening outcome for a symbol.

    Only ``COMPLIANT`` is ever tradable automatically. ``DOUBTFUL`` and
    ``UNKNOWN`` require a human decision and are never auto-traded -- that rule
    is enforced in code, not configuration.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    DOUBTFUL = "doubtful"
    UNKNOWN = "unknown"

    @property
    def auto_tradable(self) -> bool:
        return self is ComplianceStatus.COMPLIANT


class UniverseMode(StrEnum):
    ETF_ONLY = "A"
    ETF_AND_SCREENED_STOCKS = "B"


class AutonomyLevel(StrEnum):
    SUGGEST_ONLY = "1"
    AUTO_WITHIN_WHITELIST = "2"
    FULLY_AUTONOMOUS = "3"


class ApprovalKind(StrEnum):
    """What a human is being asked to decide."""

    TRADE_PROPOSAL = "trade_proposal"
    COMPLIANCE_EXIT = "compliance_exit"
    RISK_OVERRIDE = "risk_override"
    QUESTION = "question"


class ApprovalStatus(StrEnum):
    """Lifecycle of an approval request.

    ``EXPIRED`` is kept distinct from ``REJECTED`` so the audit log records
    *why* a proposal did not trade -- but only ``APPROVED`` ever authorises an
    action. A timeout is never an approval.
    """

    PENDING = "pending"
    #: The user pressed Modify; the engine is waiting for a replacement size.
    AWAITING_MODIFICATION = "awaiting_modification"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            ApprovalStatus.APPROVED,
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
            ApprovalStatus.CANCELLED,
        }

    @property
    def authorises_action(self) -> bool:
        """The only status that permits anything to happen.

        Written as an explicit allow-list rather than ``!= REJECTED`` so that a
        status added later defaults to *not* authorising.
        """
        return self is ApprovalStatus.APPROVED


class ApprovalAction(StrEnum):
    """What the user pressed."""

    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    SNOOZE = "snooze"


class ExitReason(StrEnum):
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    SIGNAL = "signal"
    COMPLIANCE_EXIT = "compliance_exit"
    MANUAL = "manual"
    KILL_SWITCH = "kill_switch"
    MAX_HOLD = "max_hold"


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar. ``ts`` is the bar's *open* time, timezone-aware UTC."""

    symbol: Symbol
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        if self.ts.tzinfo is None:
            raise ValueError(f"Bar.ts must be timezone-aware, got naive {self.ts!r}")


@dataclass(frozen=True, slots=True)
class Quote:
    """A top-of-book snapshot, used for spread checks and marketable limits."""

    symbol: Symbol
    ts: datetime
    bid: Decimal
    ask: Decimal

    @property
    def mid(self) -> Decimal:
        return to_money((self.bid + self.ask) / 2)

    @property
    def spread_pct(self) -> float:
        """Spread as a fraction of the mid. Returns ``inf`` on a crossed book."""
        mid = self.mid
        if mid <= 0:
            return float("inf")
        return float((self.ask - self.bid) / mid)


@dataclass(frozen=True, slots=True)
class CashLedger:
    """Cash split by settlement state.

    In a cash account only ``settled`` may fund a new purchase. Spending
    ``unsettled`` proceeds is a good-faith violation, so the risk manager sizes
    against ``available_for_trading`` and never against total cash.
    """

    settled: Decimal
    unsettled: Decimal
    #: Cash reserved by orders that are submitted but not yet filled.
    reserved: Decimal = Decimal("0.00")

    @property
    def total(self) -> Decimal:
        return to_money(self.settled + self.unsettled)

    @property
    def available_for_trading(self) -> Decimal:
        """Settled cash net of what open orders have already claimed."""
        return to_money(max(Decimal("0.00"), self.settled - self.reserved))


@dataclass(frozen=True, slots=True)
class PendingSettlement:
    """Proceeds of a sale that become spendable on ``settles_on``."""

    symbol: Symbol
    amount: Decimal
    sold_on: date
    settles_on: date
