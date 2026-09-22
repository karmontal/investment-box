"""A stand-in for a certified external screener (Zoya, Musaffa, ...).

Exists so the ``ScreeningProvider`` seam is exercised end to end before any
vendor contract is signed, and so tests have a provider whose verdicts they
control.

It reports ``is_certified_source = False`` despite standing in for a certified
one. That is deliberate: every trade records whether its verdict came from a
recognised board, and a mock must never be able to make that record say yes.
The startup warnings also refuse to stay quiet when Mode B is combined with
this provider.
"""

from __future__ import annotations

import datetime as dt

from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ComplianceStatus
from investment_box.shariah.providers.base import FinancialRatios, ScreenResult

log = get_logger(__name__)


class MockExternalProvider:
    """Returns verdicts from a caller-supplied table."""

    name = "mock_external"
    #: Never True. See the module docstring.
    is_certified_source = False

    def __init__(
        self,
        verdicts: dict[str, ComplianceStatus] | None = None,
        *,
        clock: Clock | None = None,
        default: ComplianceStatus = ComplianceStatus.UNKNOWN,
        available: bool = True,
    ) -> None:
        self.verdicts = {k.upper(): v for k, v in (verdicts or {}).items()}
        self.clock = clock or SystemClock()
        # An unlisted symbol is UNKNOWN, not compliant. A mock that answered
        # "compliant" for anything it had not heard of would make the tests
        # agree with a dangerous default.
        self.default = default
        self._available = available
        self.calls: list[str] = []

    def is_available(self) -> bool:
        return self._available

    def set_verdict(self, symbol: str, status: ComplianceStatus) -> None:
        self.verdicts[symbol.upper()] = status

    def screen(self, symbol: str, *, as_of: dt.date | None = None) -> ScreenResult:
        ticker = symbol.upper()
        self.calls.append(ticker)
        status = self.verdicts.get(ticker, self.default)

        ratios = (
            FinancialRatios(
                denominator="market_cap",
                denominator_value=1_000_000_000.0,
                debt_ratio=0.12,
                interest_securities_ratio=0.08,
                non_permissible_revenue_ratio=0.01,
            )
            if status is ComplianceStatus.COMPLIANT
            else None
        )

        return ScreenResult(
            symbol=ticker,
            status=status,
            source=self.name,
            screened_at=self.clock.now(),
            ratios=ratios,
            reason=f"MOCK verdict ({status.value}) -- not a real compliance ruling",
            raw={"mock": True},
        )

    def screen_many(
        self, symbols: list[str], *, as_of: dt.date | None = None
    ) -> dict[str, ScreenResult]:
        return {s.upper(): self.screen(s, as_of=as_of) for s in symbols}
