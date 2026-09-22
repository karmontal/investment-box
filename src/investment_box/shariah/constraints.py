"""Non-negotiable Shariah trading constraints.

These rules have no configuration key, no environment variable, and no UI
control. Changing them requires editing this file, which is the point: a
setting that can be toggled will eventually be toggled by accident.

:func:`assert_order_permissible` is called by the order manager on every single
order, immediately before submission, regardless of which strategy, autonomy
level or user action produced it. A violation raises
:class:`~investment_box.core.errors.ComplianceError`, which is never caught and
retried -- it means the engine attempted something it must never do.

Scope note: this module blocks *instrument and mechanism* violations, which can
be decided locally from an order. Whether a given company's business is
permissible is a screening question and lives in ``shariah/providers/``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from investment_box.core.errors import ComplianceError
from investment_box.core.types import ComplianceStatus, Side, Symbol


@dataclass(frozen=True, slots=True)
class HardConstraints:
    """The fixed rule set. Every field is ``True``/``False`` by definition, not by config."""

    #: Cash account only. No margin, no borrowing, no leverage of any kind.
    margin_allowed: Final[bool] = False
    #: No short selling. Selling what you do not own is impermissible.
    short_selling_allowed: Final[bool] = False
    #: No options, futures, CFDs, swaps, forwards.
    derivatives_allowed: Final[bool] = False
    #: No leveraged (2x/3x) or inverse (-1x) funds, even if the underlying is compliant.
    leveraged_or_inverse_allowed: Final[bool] = False
    #: No crypto or crypto derivatives.
    crypto_allowed: Final[bool] = False
    #: DOUBTFUL and UNKNOWN symbols are never traded without a human decision.
    auto_trade_requires_compliant_status: Final[bool] = True


CONSTRAINTS: Final = HardConstraints()

#: Substrings that mark a leveraged or inverse fund. Matched case-insensitively
#: against the instrument *name*, not the ticker, since tickers are arbitrary.
_LEVERAGED_NAME_MARKERS: Final[tuple[str, ...]] = (
    "2x", "3x", "-1x", "1.5x", "ultra", "ultrapro", "ultrashort",
    "leveraged", "inverse", "bear", "short ", "daily bull", "daily bear",
)

#: Ticker families that are structurally leveraged/inverse or derivative-based.
#: Not exhaustive -- a belt-and-braces check alongside the name match.
_FORBIDDEN_TICKER_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"^(TQQQ|SQQQ|SPXL|SPXS|SPXU|UPRO|SDOW|UDOW|TNA|TZA|SOXL|SOXS|LABU|LABD)$"),
    re.compile(r"^(UVXY|SVXY|VIXY|VXX|TVIX)$"),          # volatility derivatives
    re.compile(r"^(BITO|BITI|ETHU|BTF)$"),                # crypto futures funds
    re.compile(r"^[A-Z]{1,5}\d{6}[CP]\d+$"),              # OCC option symbols
    re.compile(r"^/[A-Z]{2,4}$"),                          # futures root symbols
    re.compile(r"(?i)USD[TC]$|^(BTC|ETH|DOGE|XRP|SOL)"),   # crypto pairs/tickers
)

#: Asset classes the broker may report that are categorically forbidden.
_FORBIDDEN_ASSET_CLASSES: Final[frozenset[str]] = frozenset(
    {"crypto", "option", "future", "forex", "cfd", "swap"}
)


def is_forbidden_instrument(
    symbol: str, name: str | None = None, asset_class: str | None = None
) -> str | None:
    """Return a reason string if the instrument is categorically forbidden, else ``None``.

    Checks are deliberately over-inclusive: a false positive costs one skipped
    trade, a false negative means trading something impermissible.

    Args:
        symbol: Ticker, case-insensitive.
        name: Full instrument name, if known. Catches leveraged funds whose
            tickers carry no hint.
        asset_class: Broker-reported class, if known.
    """
    ticker = symbol.strip().upper()

    if asset_class and asset_class.strip().lower() in _FORBIDDEN_ASSET_CLASSES:
        return f"asset class '{asset_class}' is not permissible"

    for pattern in _FORBIDDEN_TICKER_PATTERNS:
        if pattern.search(ticker):
            return f"ticker '{ticker}' matches a forbidden instrument pattern"

    if name:
        lowered = f" {name.strip().lower()} "
        for marker in _LEVERAGED_NAME_MARKERS:
            if marker in lowered:
                return f"instrument name contains '{marker.strip()}' (leveraged or inverse fund)"

    return None


def assert_order_permissible(
    *,
    symbol: Symbol | str,
    side: Side,
    quantity: Decimal,
    position_quantity: Decimal,
    compliance_status: ComplianceStatus,
    cash_available: Decimal,
    order_notional: Decimal,
    instrument_name: str | None = None,
    asset_class: str | None = None,
    human_approved: bool = False,
) -> None:
    """Final gate before an order reaches the broker.

    Args:
        symbol: Ticker being traded.
        side: Buy or sell.
        quantity: Order size, always positive.
        position_quantity: Currently held quantity of ``symbol``. A sell larger
            than this would be a short.
        compliance_status: The screened status at this moment.
        cash_available: Settled cash available. A buy exceeding it implies
            margin.
        order_notional: Estimated cash cost of the order.
        instrument_name: Full name, for the leveraged/inverse check.
        asset_class: Broker-reported class, for the derivative check.
        human_approved: Set when a human explicitly authorised this specific
            order. Relaxes *only* the auto-trade status rule -- never the
            margin, short, derivative or leverage rules.

    Raises:
        ComplianceError: On any violation. Not recoverable, not retryable.
    """
    ticker = str(symbol).strip().upper()

    if quantity <= 0:
        raise ComplianceError(f"{ticker}: order quantity must be positive, got {quantity}")

    reason = is_forbidden_instrument(ticker, instrument_name, asset_class)
    if reason is not None:
        raise ComplianceError(f"{ticker}: forbidden instrument -- {reason}")

    if side is Side.SELL and quantity > position_quantity:
        raise ComplianceError(
            f"{ticker}: sell of {quantity} exceeds held quantity {position_quantity}. "
            f"This would open a short position, which is never permitted."
        )

    if side is Side.BUY and order_notional > cash_available:
        raise ComplianceError(
            f"{ticker}: buy notional {order_notional} exceeds settled cash "
            f"{cash_available}. This would use margin, which is never permitted."
        )

    if compliance_status is ComplianceStatus.NON_COMPLIANT and side is Side.BUY:
        raise ComplianceError(
            f"{ticker}: cannot buy a NON_COMPLIANT symbol. "
            f"(Selling one is permitted, and required, under the exit policy.)"
        )

    if side is Side.BUY and not compliance_status.auto_tradable and not human_approved:
        raise ComplianceError(
            f"{ticker}: compliance status is {compliance_status.value.upper()}. "
            f"Only COMPLIANT symbols may be bought automatically; this one needs "
            f"an explicit human decision."
        )
