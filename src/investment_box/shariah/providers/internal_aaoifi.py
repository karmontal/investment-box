"""Locally-computed AAOIFI-style screening.

This provider computes the financial ratios itself from whatever fundamentals
the data layer can supply. It exists so Mode B is *possible* without a paid
vendor -- not because it is as good as one.

Three honest limitations, stated here because they determine how much weight
the output deserves:

1. **There is no business-activity database.** Deciding whether a company earns
   impermissible revenue requires segment-level revenue data and a judgement
   about each segment. yfinance supplies neither. This provider therefore
   relies on a coarse sector/industry mapping, which catches an obvious bank
   and misses a conglomerate with a financing arm.
2. **Fundamentals are stale and revised.** yfinance reports the most recent
   filing with no point-in-time history, so a screen "as of" a past date
   actually uses today's numbers. That is look-ahead, and it is flagged.
3. **A missing input yields UNKNOWN, never a pass.** Absence of evidence is
   not evidence of compliance.

The result: for individual stocks, treat this as a pre-filter that narrows a
list for human review, not as a ruling. For the certified ETFs of Mode A it is
not used at all -- those carry their own board's certification.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from investment_box.config.schema import ShariahConfig
from investment_box.core.clock import UTC, Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ComplianceStatus
from investment_box.shariah.providers.base import (
    FinancialRatios,
    ScreenResult,
    unknown_result,
)

log = get_logger(__name__)

#: Coarse sector/industry substrings that imply an impermissible core business.
#: Deliberately over-inclusive: a false positive costs one skipped symbol, a
#: false negative means trading something impermissible.
_ACTIVITY_MARKERS: dict[str, tuple[str, ...]] = {
    "conventional_banking": ("bank", "credit services", "mortgage finance", "savings"),
    "conventional_insurance": ("insurance", "reinsurance"),
    "interest_based_finance": (
        "capital markets",
        "asset management",
        "financial data",
        "lending",
        "consumer finance",
    ),
    "alcohol": ("beverages - brewers", "beverages - wineries", "distiller", "brewer"),
    "pork": ("pork", "swine"),
    "gambling": ("gambling", "casino", "resorts & casinos", "lottery", "betting"),
    "adult_entertainment": ("adult",),
    "tobacco": ("tobacco", "cigarette"),
    "weapons": ("aerospace & defense", "defense", "weapons", "firearms", "ammunition"),
}


class InternalAAOIFIProvider:
    """Computes the AAOIFI screen from available fundamentals."""

    name = "internal_aaoifi"
    is_certified_source = False

    def __init__(
        self,
        config: ShariahConfig,
        *,
        clock: Clock | None = None,
        fundamentals_fetcher: Any | None = None,
    ) -> None:
        self.config = config
        self.clock = clock or SystemClock()
        # Injected so tests do not need the network, and so a better
        # fundamentals source can be dropped in without touching the screen.
        self._fetch = fundamentals_fetcher or self._fetch_from_yfinance

    def is_available(self) -> bool:
        return True

    @property
    def _thresholds(self) -> dict[str, float]:
        return {
            "debt": self.config.debt_to_market_cap_max_pct,
            "interest_securities": self.config.interest_securities_to_market_cap_max_pct,
            "revenue": self.config.non_permissible_revenue_max_pct,
        }

    def screen(self, symbol: str, *, as_of: dt.date | None = None) -> ScreenResult:
        ticker = symbol.upper()
        now = self.clock.now()

        if as_of is not None and as_of < now.astimezone(UTC).date():
            # Worth saying out loud: the caller asked for a historical screen
            # and is about to get today's numbers.
            log.warning(
                "shariah.no_point_in_time_data",
                symbol=ticker,
                requested_as_of=str(as_of),
                note="today's fundamentals used; this is look-ahead bias",
            )

        try:
            data = self._fetch(ticker)
        except Exception as exc:  # noqa: BLE001 - a fetch failure is UNKNOWN, not a crash
            log.warning("shariah.fundamentals_failed", symbol=ticker, error=str(exc))
            return unknown_result(
                ticker, self.name, f"could not fetch fundamentals: {exc}", now
            )

        if not data:
            return unknown_result(ticker, self.name, "no fundamentals available", now)

        flags = self._activity_flags(data)
        ratios = self._compute_ratios(data)
        return self._verdict(ticker, ratios, flags, now, data)

    def screen_many(
        self, symbols: list[str], *, as_of: dt.date | None = None
    ) -> dict[str, ScreenResult]:
        results: dict[str, ScreenResult] = {}
        for symbol in symbols:
            try:
                results[symbol.upper()] = self.screen(symbol, as_of=as_of)
            except Exception as exc:  # noqa: BLE001 - isolate per-symbol failure
                log.warning("shariah.screen_failed", symbol=symbol, error=str(exc))
                results[symbol.upper()] = unknown_result(
                    symbol, self.name, f"screen raised: {exc}", self.clock.now()
                )
        return results

    # -------------------------------------------------------------- internals

    def _activity_flags(self, data: dict[str, Any]) -> tuple[str, ...]:
        """Match sector and industry text against the activity markers."""
        keys = ("sector", "industry", "long_business_summary")
        haystack = " ".join(str(data.get(key, "")).lower() for key in keys)
        excluded = set(self.config.excluded_activities)
        return tuple(
            activity
            for activity, markers in _ACTIVITY_MARKERS.items()
            if activity in excluded and any(marker in haystack for marker in markers)
        )

    def _compute_ratios(self, data: dict[str, Any]) -> FinancialRatios:
        """Compute the three ratios against the configured denominator."""
        denominator_key = (
            "market_cap" if self.config.ratio_denominator == "market_cap" else "total_assets"
        )
        denominator = _positive(data.get(denominator_key))

        debt = _positive(data.get("total_debt"), allow_zero=True)
        cash_and_securities = _positive(data.get("cash_and_securities"), allow_zero=True)

        def ratio(numerator: float | None) -> float | None:
            if numerator is None or denominator is None:
                return None
            return numerator / denominator

        # Without segment revenue there is no way to compute this properly. A
        # flagged activity is treated as 100% impermissible; an unflagged one
        # reports 0.0, which is an assumption, not a measurement -- and the
        # reason string says so.
        revenue_ratio: float | None = None
        if data.get("_activity_flagged") is not None:
            revenue_ratio = 1.0 if data["_activity_flagged"] else 0.0

        return FinancialRatios(
            denominator=self.config.ratio_denominator,
            denominator_value=denominator,
            debt_ratio=ratio(debt),
            interest_securities_ratio=ratio(cash_and_securities),
            non_permissible_revenue_ratio=revenue_ratio,
        )

    def _verdict(
        self,
        ticker: str,
        ratios: FinancialRatios,
        flags: tuple[str, ...],
        now: dt.datetime,
        data: dict[str, Any],
    ) -> ScreenResult:
        if flags:
            return ScreenResult(
                symbol=ticker,
                status=ComplianceStatus.NON_COMPLIANT,
                source=self.name,
                screened_at=now,
                ratios=ratios,
                activity_flags=flags,
                reason=f"impermissible business activity: {', '.join(flags)}",
                raw=data,
            )

        if ratios.debt_ratio is None or ratios.interest_securities_ratio is None:
            return ScreenResult(
                symbol=ticker,
                status=ComplianceStatus.UNKNOWN,
                source=self.name,
                screened_at=now,
                ratios=ratios,
                reason=(
                    "incomplete fundamentals: could not compute the financial ratios. "
                    "Missing data is not evidence of compliance."
                ),
                raw=data,
            )

        failing = ratios.breaches(self._thresholds)
        if failing:
            return ScreenResult(
                symbol=ticker,
                status=ComplianceStatus.NON_COMPLIANT,
                source=self.name,
                screened_at=now,
                ratios=ratios,
                reason="financial ratio breach: " + "; ".join(failing),
                raw=data,
            )

        # Passes the ratios, but revenue purity was assumed rather than
        # measured. DOUBTFUL, not COMPLIANT -- and DOUBTFUL is never
        # auto-traded, so this needs a human.
        return ScreenResult(
            symbol=ticker,
            status=ComplianceStatus.DOUBTFUL,
            source=self.name,
            screened_at=now,
            ratios=ratios,
            reason=(
                "financial ratios pass, but non-permissible revenue could not be "
                "measured (no segment-level data). Needs human review or a "
                "certified provider before trading."
            ),
            raw=data,
        )

    @staticmethod
    def _fetch_from_yfinance(symbol: str) -> dict[str, Any]:
        """Pull the fundamentals yfinance exposes. Research quality only."""
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        if not info or info.get("quoteType") is None:
            return {}

        cash = info.get("totalCash")
        return {
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "long_business_summary": info.get("longBusinessSummary"),
            "market_cap": info.get("marketCap"),
            "total_assets": info.get("totalAssets"),
            "total_debt": info.get("totalDebt"),
            "cash_and_securities": cash,
            "quote_type": info.get("quoteType"),
        }


def _positive(value: Any, *, allow_zero: bool = False) -> float | None:
    """Coerce to a usable float, rejecting None, NaN and negatives."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    if number < 0 or (number == 0 and not allow_zero):
        return None
    return number
