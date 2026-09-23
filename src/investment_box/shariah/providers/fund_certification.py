"""Compliance verdicts for funds that carry their own Shariah board's certification.

A certified ETF is not a company. Running one through a stock screener is a
category error: the screener looks for a balance sheet and business segments
that a fund does not have, and answers about whatever the data vendor happened
to map the ticker to. Applied to SPUS -- a fund whose entire mandate is Shariah
compliance, audited annually by Raqaba LLC -- the internal screener returns
``non_compliant: impermissible business activity: weapons``. That verdict is
noise, and acting on it would be worse than useless.

For these funds the ruling already exists. It was issued by the fund's own
Shariah board, it is named in ``config/universe_etf.yaml`` under
``certifying_board``, and a human confirmed it by setting ``verified: true``.
This provider does nothing more than report that ruling as what it is.

What it deliberately does not do:

* It does not certify anything itself. It reports an attestation made
  elsewhere, and names that source in every result.
* It never answers COMPLIANT for a symbol that is unverified, absent from the
  universe file, or verified but with no board recorded. Those are UNKNOWN,
  because a certification nobody can name is not a certification.
* It does not screen stocks. In Mode B a ``fallback`` provider handles anything
  that is not a certified fund, and the composite stops describing itself as a
  certified source, because a locally computed stock screen is an estimate.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import TYPE_CHECKING

from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ComplianceStatus
from investment_box.shariah.providers.base import (
    ScreeningProvider,
    ScreenResult,
    unknown_result,
)

if TYPE_CHECKING:
    from investment_box.universe.builder import Instrument

log = get_logger(__name__)


class CertifiedFundProvider:
    """Reports the certification recorded against each verified fund."""

    name = "fund_certification"

    def __init__(
        self,
        instruments: Iterable[Instrument],
        *,
        fallback: ScreeningProvider | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._funds = {i.symbol.upper(): i for i in instruments}
        self._fallback = fallback
        self.clock = clock or SystemClock()

        # A composite is only as certified as its weakest branch. With a
        # fallback that computes screens locally, some verdicts are estimates,
        # and every trade records which kind it got.
        self.is_certified_source = fallback is None or fallback.is_certified_source

    def is_available(self) -> bool:
        if self._fallback is not None and not self._fallback.is_available():
            return False
        return True

    def screen(self, symbol: str, *, as_of: dt.date | None = None) -> ScreenResult:
        ticker = symbol.upper()
        now = self.clock.now()
        fund = self._funds.get(ticker)

        if fund is None:
            return self._delegate(
                ticker,
                as_of,
                f"{ticker} is not a fund in config/universe_etf.yaml",
                now,
            )

        if not fund.verified:
            return unknown_result(
                ticker,
                self.name,
                f"{ticker} is in the universe file but unverified: its listing and "
                f"Shariah certification have not been confirmed by a human.",
                now,
            )

        board = (fund.certifying_board or "").strip()
        if not board:
            return unknown_result(
                ticker,
                self.name,
                f"{ticker} is marked verified but records no certifying board. A "
                f"certification nobody can name is not a certification.",
                now,
            )

        return ScreenResult(
            symbol=ticker,
            status=ComplianceStatus.COMPLIANT,
            source=self.name,
            screened_at=now,
            reason=f"certified by the fund's own Shariah board: {board}",
            raw={
                "certifying_board": board,
                "fund_name": fund.name,
                "issuer": fund.issuer,
                "verified_in_config": True,
            },
        )

    def screen_many(
        self, symbols: list[str], *, as_of: dt.date | None = None
    ) -> dict[str, ScreenResult]:
        return {s.upper(): self.screen(s, as_of=as_of) for s in symbols}

    def _delegate(
        self, ticker: str, as_of: dt.date | None, reason: str, now: dt.datetime
    ) -> ScreenResult:
        if self._fallback is None:
            return unknown_result(ticker, self.name, reason, now)
        try:
            return self._fallback.screen(ticker, as_of=as_of)
        except Exception as exc:  # noqa: BLE001 - a transport failure is UNKNOWN, not a pass
            log.warning("screening.fallback_failed", symbol=ticker, error=str(exc))
            return unknown_result(
                ticker, self._fallback.name, f"fallback screen failed: {exc}", now
            )
