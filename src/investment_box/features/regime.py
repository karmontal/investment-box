"""Market regime detection.

The rotation strategy needs one binary-ish judgement: is the broad market in a
state where holding equity is sensible, or should capital sit in sukuk or cash?

Two inputs, both lagging on purpose. A leading regime indicator would be a
market-timing model, which is a much harder problem than this application is
trying to solve:

* **SPY trend** -- price relative to its 200-day moving average, the crudest
  and most robust trend filter there is.
* **VIX level** -- an elevated VIX means realised drawdowns are more likely to
  be large, which matters more than usual at a small account size where a
  single bad exit is a meaningful fraction of capital.

VIX is optional. When it is unavailable the regime falls back to trend alone
and says so, rather than silently pretending the second input agreed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from investment_box.core.logging import get_logger
from investment_box.data.repository import MarketDataRepository
from investment_box.features.indicators import moving_average_distance, realised_volatility

log = get_logger(__name__)

DEFAULT_TREND_WINDOW = 200
#: Above this, treat volatility as elevated. Roughly the 80th percentile of
#: VIX since 2010; not tuned on the backtest, to avoid fitting the filter to
#: the sample it is evaluated on.
VIX_ELEVATED = 25.0
VIX_EXTREME = 35.0


class MarketRegime(StrEnum):
    """What the filter permits."""

    RISK_ON = "risk_on"
    #: Trend intact but volatility elevated: hold, but do not add aggressively.
    CAUTIOUS = "cautious"
    #: Trend broken: rotate to sukuk or cash.
    RISK_OFF = "risk_off"
    UNKNOWN = "unknown"

    @property
    def allows_new_entries(self) -> bool:
        return self in (MarketRegime.RISK_ON, MarketRegime.CAUTIOUS)

    @property
    def is_defensive(self) -> bool:
        return self is MarketRegime.RISK_OFF


@dataclass(frozen=True, slots=True)
class RegimeState:
    """The regime on a date, with the evidence behind it."""

    as_of: dt.date
    regime: MarketRegime
    reason: str
    trend_distance: float | None = None
    vix_level: float | None = None
    benchmark_volatility: float | None = None
    #: True when VIX was unavailable and the verdict rests on trend alone.
    degraded: bool = False

    @property
    def allows_new_entries(self) -> bool:
        return self.regime.allows_new_entries


class RegimeDetector:
    """Computes the regime from SPY and VIX."""

    def __init__(
        self,
        repository: MarketDataRepository,
        *,
        trend_symbol: str = "SPY",
        vix_symbol: str = "^VIX",
        trend_window: int = DEFAULT_TREND_WINDOW,
    ) -> None:
        self.repository = repository
        self.trend_symbol = trend_symbol
        self.vix_symbol = vix_symbol
        self.trend_window = trend_window

    def detect(self, as_of: dt.date, *, lookback_days: int = 500) -> RegimeState:
        """The regime as of ``as_of``, using only data strictly before it.

        ``as_of`` is passed to the repository as the look-ahead cutoff, so the
        bar for ``as_of`` itself is not visible. The regime that decides
        Monday's trade is computed from data through Friday.
        """
        start = as_of - dt.timedelta(days=lookback_days)

        trend = self._series(self.trend_symbol, start, as_of)
        if trend is None or len(trend) < self.trend_window:
            return RegimeState(
                as_of=as_of,
                regime=MarketRegime.UNKNOWN,
                reason=(
                    f"insufficient {self.trend_symbol} history "
                    f"({0 if trend is None else len(trend)} bars, need {self.trend_window})"
                ),
                degraded=True,
            )

        distance = float(moving_average_distance(trend, self.trend_window).iloc[-1])
        benchmark_vol = realised_volatility(trend, 20).iloc[-1]
        benchmark_vol = float(benchmark_vol) if pd.notna(benchmark_vol) else None

        vix_series = self._series(self.vix_symbol, start, as_of)
        vix = None
        degraded = False
        if vix_series is None or vix_series.empty:
            degraded = True
            log.info("regime.vix_unavailable", as_of=str(as_of), note="using trend only")
        else:
            vix = float(vix_series.iloc[-1])

        return self._classify(as_of, distance, vix, benchmark_vol, degraded)

    @staticmethod
    def _classify(
        as_of: dt.date,
        distance: float,
        vix: float | None,
        benchmark_vol: float | None,
        degraded: bool,
    ) -> RegimeState:
        def state(regime: MarketRegime, reason: str) -> RegimeState:
            return RegimeState(
                as_of=as_of,
                regime=regime,
                reason=reason,
                trend_distance=distance,
                vix_level=vix,
                benchmark_volatility=benchmark_vol,
                degraded=degraded,
            )

        if distance < 0:
            return state(
                MarketRegime.RISK_OFF,
                f"SPY is {distance:.1%} below its 200-day average",
            )

        if vix is not None and vix >= VIX_EXTREME:
            return state(
                MarketRegime.RISK_OFF,
                f"VIX {vix:.1f} is extreme (>= {VIX_EXTREME}) despite an intact trend",
            )

        if vix is not None and vix >= VIX_ELEVATED:
            return state(
                MarketRegime.CAUTIOUS,
                f"trend intact ({distance:+.1%}) but VIX {vix:.1f} is elevated",
            )

        suffix = " (VIX unavailable; trend only)" if degraded else ""
        return state(
            MarketRegime.RISK_ON,
            f"SPY is {distance:+.1%} above its 200-day average{suffix}",
        )

    def _series(self, symbol: str, start: dt.date, as_of: dt.date) -> pd.Series | None:
        try:
            result = self.repository.get_bars(symbol, start, as_of, as_of=as_of, validate=False)
        except Exception as exc:  # noqa: BLE001 - a missing input degrades, never crashes
            log.warning("regime.fetch_failed", symbol=symbol, error=str(exc))
            return None
        return None if result.frame.empty else result.frame["close"]

    def history(self, dates: list[dt.date]) -> list[RegimeState]:
        """The regime on each of several dates. Used by the backtester."""
        return [self.detect(day) for day in dates]
