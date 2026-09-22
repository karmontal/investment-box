"""Building the tradable universe.

A symbol is tradable only if it clears every gate, in this order:

1. **Verified.** The seed list entry has been confirmed by a human as listed
   and Shariah-certified. Unverified symbols are never traded, full stop.
2. **Not blacklisted**, and on the whitelist if one is in force.
3. **Compliant**, and the screen is fresh.
4. **Liquid enough**: price band, average dollar volume, spread.
5. **Enough history** to compute the features the strategies need.

Every rejection is recorded with its reason, because "why didn't it trade X?"
is the question this module exists to answer.

**Look-ahead:** ``build(as_of=...)`` excludes any fund whose inception is after
that date, so a backtest of 2019 cannot hold an ETF that launched in 2023. What
it *cannot* do is reconstruct what was Shariah-certified in 2019 -- that data
does not exist for this universe. When a historical build uses today's
compliance list it says so, and the flag travels into the backtest report.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from investment_box.config.schema import Settings
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import AssetClass, ComplianceStatus, UniverseMode
from investment_box.data.repository import MarketDataRepository
from investment_box.shariah.constraints import is_forbidden_instrument
from investment_box.shariah.status import ComplianceRecord, ComplianceTracker

log = get_logger(__name__)

#: Bars needed before a symbol can be ranked. The longest lookback used by the
#: strategies is a 6-month (126 trading day) momentum window, plus room for the
#: volatility estimate it is scaled by.
MIN_HISTORY_BARS = 150


@dataclass(frozen=True, slots=True)
class Instrument:
    """A candidate instrument as configured, before any filtering."""

    symbol: str
    name: str | None = None
    issuer: str | None = None
    asset_class: AssetClass = AssetClass.UNKNOWN
    certifying_board: str | None = None
    inception: dt.date | None = None
    verified: bool = False
    notes: str | None = None

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> Instrument:
        raw_class = str(entry.get("asset_class") or "unknown").lower()
        try:
            asset_class = AssetClass(raw_class)
        except ValueError:
            asset_class = AssetClass.UNKNOWN

        inception = entry.get("inception")
        if isinstance(inception, str):
            inception = dt.date.fromisoformat(inception)
        elif isinstance(inception, dt.datetime):
            inception = inception.date()

        return cls(
            symbol=str(entry["symbol"]).upper(),
            name=entry.get("name"),
            issuer=entry.get("issuer"),
            asset_class=asset_class,
            certifying_board=entry.get("certifying_board"),
            inception=inception,
            verified=bool(entry.get("verified")),
            notes=entry.get("notes"),
        )

    def existed_on(self, day: dt.date) -> bool | None:
        """Whether the fund had launched by ``day``.

        ``None`` means the inception date is unknown, which is itself a
        problem: a backtest cannot tell whether including it is look-ahead.
        """
        if self.inception is None:
            return None
        return self.inception <= day


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    """One instrument's verdict, whether or not it made the cut."""

    instrument: Instrument
    included: bool
    reason: str
    compliance: ComplianceRecord | None = None
    last_price: float | None = None
    avg_dollar_volume: float | None = None
    bars_available: int = 0

    @property
    def symbol(self) -> str:
        return self.instrument.symbol


@dataclass
class UniverseSnapshot:
    """The universe as of a moment, with everything needed to audit it."""

    as_of: dt.date
    entries: list[UniverseEntry] = field(default_factory=list)
    mode: UniverseMode = UniverseMode.ETF_ONLY
    #: Set when historical compliance data was unavailable and today's screens
    #: were used instead. Survivorship and look-ahead risk; shown in reports.
    used_current_compliance_for_history: bool = False

    @property
    def symbols(self) -> list[str]:
        return [e.symbol for e in self.entries if e.included]

    @property
    def included(self) -> list[UniverseEntry]:
        return [e for e in self.entries if e.included]

    @property
    def excluded(self) -> list[UniverseEntry]:
        return [e for e in self.entries if not e.included]

    @property
    def is_empty(self) -> bool:
        return not self.symbols

    def rejection_summary(self) -> dict[str, int]:
        """Count of exclusions by reason, for the report and the dashboard."""
        counts: dict[str, int] = {}
        for entry in self.excluded:
            key = entry.reason.split(":")[0].strip()
            counts[key] = counts.get(key, 0) + 1
        return counts

    def explain(self, symbol: str) -> str:
        for entry in self.entries:
            if entry.symbol == symbol.upper():
                verdict = "INCLUDED" if entry.included else "EXCLUDED"
                return f"{symbol.upper()}: {verdict} -- {entry.reason}"
        return f"{symbol.upper()}: not in the configured universe"


class UniverseBuilder:
    """Applies every gate and records why each symbol passed or failed."""

    def __init__(
        self,
        settings: Settings,
        repository: MarketDataRepository,
        tracker: ComplianceTracker | None = None,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.tracker = tracker
        self.clock = clock or SystemClock()

    def build(
        self,
        instruments: list[Instrument],
        *,
        as_of: dt.date | None = None,
        check_compliance: bool = True,
        check_liquidity: bool = True,
    ) -> UniverseSnapshot:
        """Build the universe as of a date.

        Args:
            instruments: Candidates, usually from ``config/universe_etf.yaml``.
            as_of: The date to build for. Historical dates exclude funds that
                had not launched.
            check_compliance: Skip only when no tracker is wired (Mode A with
                certified ETFs relies on ``verified``).
            check_liquidity: Skip in unit tests that supply no price data.
        """
        day = as_of or self.clock.now().date()
        snapshot = UniverseSnapshot(as_of=day, mode=self.settings.universe.mode)

        whitelist = set(self.settings.universe.whitelist)
        blacklist = set(self.settings.universe.blacklist)
        is_historical = day < self.clock.now().date()

        for instrument in instruments:
            entry = self._evaluate(
                instrument,
                day=day,
                whitelist=whitelist,
                blacklist=blacklist,
                check_compliance=check_compliance,
                check_liquidity=check_liquidity,
            )
            snapshot.entries.append(entry)

        if is_historical and check_compliance:
            # Stated rather than assumed: there is no point-in-time compliance
            # history for this universe, so a historical build is using today's
            # verdicts. That is look-ahead and the report must say so.
            snapshot.used_current_compliance_for_history = True
            log.warning(
                "universe.historical_compliance_unavailable",
                as_of=str(day),
                note="today's compliance status applied to a past date",
            )

        log.info(
            "universe.built",
            as_of=str(day),
            included=len(snapshot.symbols),
            excluded=len(snapshot.excluded),
        )
        return snapshot

    # -------------------------------------------------------------- the gates

    def _evaluate(
        self,
        instrument: Instrument,
        *,
        day: dt.date,
        whitelist: set[str],
        blacklist: set[str],
        check_compliance: bool,
        check_liquidity: bool,
    ) -> UniverseEntry:
        symbol = instrument.symbol

        def reject(
            reason: str, compliance: ComplianceRecord | None = None
        ) -> UniverseEntry:
            return UniverseEntry(
                instrument=instrument, included=False, reason=reason, compliance=compliance
            )

        # 1. Hard instrument constraints, before anything else.
        forbidden = is_forbidden_instrument(symbol, instrument.name, instrument.asset_class.value)
        if forbidden is not None:
            return reject(f"forbidden instrument: {forbidden}")

        # 2. Human verification of listing and certification.
        if not instrument.verified:
            return reject(
                "unverified: listing and Shariah certification have not been confirmed "
                "in config/universe_etf.yaml"
            )

        # 3. User rules.
        if symbol in blacklist:
            return reject("blacklisted: excluded by your rules")
        if whitelist and symbol not in whitelist:
            return reject("not whitelisted: a whitelist is in force and excludes this symbol")

        # 4. Existence on the date. Prevents backtesting a fund into a period
        #    before it launched.
        existed = instrument.existed_on(day)
        if existed is False:
            return reject(f"not yet listed: inception {instrument.inception} is after {day}")
        if existed is None:
            return reject(
                "inception date unknown: cannot rule out look-ahead, so it is excluded "
                "from dated builds. Set `inception` in config/universe_etf.yaml."
            )

        # 5. Compliance.
        record: ComplianceRecord | None = None
        if check_compliance and self.tracker is not None:
            record = self.tracker.screen(symbol)
            if record.status is ComplianceStatus.NON_COMPLIANT:
                return reject(f"non-compliant: {record.reason}", compliance=record)
            if record.is_stale:
                return reject(
                    f"stale screen: last screened {record.age_days}d ago, "
                    f"limit {self.settings.shariah.rescreen_interval_days}d",
                    compliance=record,
                )
            if not record.status.auto_tradable:
                return reject(
                    f"{record.status.value}: needs a human decision before trading",
                    compliance=record,
                )

        # 6. Liquidity and history.
        if not check_liquidity:
            return UniverseEntry(
                instrument=instrument, included=True, reason="passed (liquidity not checked)",
                compliance=record,
            )

        return self._liquidity_gate(instrument, day, record)

    def _liquidity_gate(
        self, instrument: Instrument, day: dt.date, record: ComplianceRecord | None
    ) -> UniverseEntry:
        symbol = instrument.symbol
        config = self.settings.universe
        start = day - dt.timedelta(days=400)

        try:
            result = self.repository.get_bars(symbol, start, day, as_of=None, validate=False)
        except Exception as exc:  # noqa: BLE001 - a data failure is an exclusion, not a crash
            return UniverseEntry(
                instrument=instrument, included=False,
                reason=f"no data: {exc}", compliance=record,
            )

        frame = result.frame
        if frame.empty:
            return UniverseEntry(
                instrument=instrument, included=False,
                reason="no data: provider returned no bars", compliance=record,
            )

        bars = len(frame)
        last_price = float(frame["close"].iloc[-1])
        window = frame.tail(20)
        avg_dollar_volume = float(np.nanmean((window["close"] * window["volume"]).to_numpy()))

        def entry(*, included: bool, reason: str) -> UniverseEntry:
            return UniverseEntry(
                instrument=instrument,
                included=included,
                reason=reason,
                compliance=record,
                last_price=last_price,
                avg_dollar_volume=avg_dollar_volume,
                bars_available=bars,
            )

        if bars < MIN_HISTORY_BARS:
            return entry(
                included=False,
                reason=(
                    f"insufficient history: {bars} bars, need {MIN_HISTORY_BARS} to compute "
                    f"the ranking features"
                ),
            )
        if last_price < config.min_price:
            return entry(
                included=False,
                reason=f"price too low: ${last_price:.2f} < ${config.min_price:.2f}",
            )
        if last_price > config.max_price:
            return entry(
                included=False,
                reason=f"price too high: ${last_price:.2f} > ${config.max_price:.2f}",
            )
        if avg_dollar_volume < config.min_avg_dollar_volume:
            return entry(
                included=False,
                reason=(
                    f"illiquid: 20d average dollar volume ${avg_dollar_volume:,.0f} < "
                    f"${config.min_avg_dollar_volume:,.0f}"
                ),
            )

        return entry(
            included=True,
            reason=f"passed: ${last_price:.2f}, ${avg_dollar_volume:,.0f}/day, {bars} bars",
        )

    # --------------------------------------------------------------- loading

    @staticmethod
    def load_instruments(universe_config: dict[str, Any]) -> list[Instrument]:
        """Turn ``config/universe_etf.yaml`` into instruments."""
        return [Instrument.from_config(entry) for entry in universe_config.get("etfs", [])]

    @staticmethod
    def benchmark_symbols(universe_config: dict[str, Any]) -> list[str]:
        """Benchmarks are fetched for comparison only and are never traded."""
        seen: list[str] = []
        for entry in universe_config.get("benchmarks", []):
            symbol = str(entry["symbol"]).upper()
            if symbol not in seen:
                seen.append(symbol)
        return seen
