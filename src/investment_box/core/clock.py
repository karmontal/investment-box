"""Time, trading calendar and settlement arithmetic.

Three rules hold everywhere in this codebase:

1. Every datetime is timezone-aware. Naive datetimes are rejected, not coerced.
2. Everything is *stored* in UTC. Timezones are applied only when rendering for
   a human (``Asia/Jerusalem`` by default) or when reasoning about market hours
   (``America/New_York``).
3. "Days" in a trading context always mean NYSE trading days, never calendar
   days. A two-day minimum hold over a long weekend is four calendar days.

The ``Clock`` protocol exists so that tests can freeze time without patching
``datetime`` globally; every module takes a clock rather than calling
``datetime.now`` directly.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, runtime_checkable
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

UTC = dt.UTC
MARKET_TZ = ZoneInfo("America/New_York")
DEFAULT_DISPLAY_TZ = ZoneInfo("Asia/Jerusalem")

#: How far the cached calendar extends around "now". Regenerated on demand.
_CALENDAR_PAD_DAYS = 800


@runtime_checkable
class Clock(Protocol):
    """Source of the current time. Injected so tests can control it."""

    def now(self) -> dt.datetime:
        """Return the current time as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """The real wall clock."""

    def now(self) -> dt.datetime:
        return dt.datetime.now(tz=UTC)


@dataclass
class FrozenClock:
    """A clock pinned to a fixed instant, for tests and deterministic backtests."""

    instant: dt.datetime

    def __post_init__(self) -> None:
        if self.instant.tzinfo is None:
            raise ValueError("FrozenClock requires a timezone-aware instant")
        self.instant = self.instant.astimezone(UTC)

    def now(self) -> dt.datetime:
        return self.instant

    def advance(self, **timedelta_kwargs: float) -> None:
        """Move the clock forward, e.g. ``clock.advance(days=1, hours=2)``."""
        self.instant += dt.timedelta(**timedelta_kwargs)


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """Return ``value`` as UTC, rejecting naive datetimes.

    Naive datetimes are a silent source of look-ahead bugs -- a bar timestamped
    ``2024-01-02 09:30`` means something different in every timezone -- so they
    raise rather than being assumed to be anything.
    """
    if value.tzinfo is None:
        raise ValueError(f"Naive datetime not allowed: {value!r}")
    return value.astimezone(UTC)


def to_display(value: dt.datetime, tz: ZoneInfo = DEFAULT_DISPLAY_TZ) -> dt.datetime:
    """Convert a stored UTC datetime to the user's display timezone."""
    return ensure_utc(value).astimezone(tz)


def to_market(value: dt.datetime) -> dt.datetime:
    """Convert a stored UTC datetime to US Eastern market time."""
    return ensure_utc(value).astimezone(MARKET_TZ)


@lru_cache(maxsize=4)
def _schedule(start: dt.date, end: dt.date) -> pd.DatetimeIndex:
    """NYSE trading sessions between two dates, cached.

    ``pandas_market_calendars`` ships its holiday rules offline, so this never
    touches the network.
    """
    calendar = mcal.get_calendar("NYSE")
    sched = calendar.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(sched.index)


class TradingCalendar:
    """NYSE trading-day arithmetic.

    Instances are cheap; the underlying session index is cached process-wide.
    """

    def __init__(
        self,
        anchor: dt.date | None = None,
        *,
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> None:
        """Build a calendar covering a date range.

        Args:
            anchor: Centre of a +/- 800 day window. Fine for live trading.
            start: Explicit first date. Use this for backtests.
            end: Explicit last date.

        Pass ``start``/``end`` whenever the code will ask about dates far from
        today. The anchored default silently excludes anything outside its
        window -- a backtester iterating ``sessions`` over an uncovered period
        finds none and skips it without an obvious error.
        """
        if start is not None or end is not None:
            today = dt.datetime.now(tz=UTC).date()
            self._start = start or (today - dt.timedelta(days=_CALENDAR_PAD_DAYS))
            self._end = end or (today + dt.timedelta(days=_CALENDAR_PAD_DAYS))
            if self._end < self._start:
                raise ValueError(f"calendar end {self._end} precedes start {self._start}")
            # Pad both ends so settlement and holding-period arithmetic can step
            # past the requested boundaries without falling out of the window.
            self._start -= dt.timedelta(days=30)
            self._end += dt.timedelta(days=30)
        else:
            anchor = anchor or dt.datetime.now(tz=UTC).date()
            self._start = anchor - dt.timedelta(days=_CALENDAR_PAD_DAYS)
            self._end = anchor + dt.timedelta(days=_CALENDAR_PAD_DAYS)
        self._sessions = _schedule(self._start, self._end)

    @property
    def covers(self) -> tuple[dt.date, dt.date]:
        """The date range this calendar can answer questions about."""
        return self._start, self._end

    def _guard(self, day: dt.date) -> None:
        if not (self._start <= day <= self._end):
            raise ValueError(
                f"{day} falls outside the loaded calendar window "
                f"({self._start}..{self._end}). Construct TradingCalendar with an "
                f"anchor nearer that date."
            )

    @property
    def sessions(self) -> list[dt.date]:
        """Every trading day in the loaded window, ascending."""
        return [ts.date() for ts in self._sessions]

    def is_trading_day(self, day: dt.date) -> bool:
        self._guard(day)
        return pd.Timestamp(day) in self._sessions

    def next_trading_day(self, day: dt.date, *, inclusive: bool = False) -> dt.date:
        """The next trading day strictly after ``day`` (or including it)."""
        self._guard(day)
        if inclusive and self.is_trading_day(day):
            return day
        idx = int(self._sessions.searchsorted(pd.Timestamp(day), side="right"))
        if idx >= len(self._sessions):
            raise ValueError(f"No trading day after {day} within the calendar window")
        return self._sessions[idx].date()

    def previous_trading_day(self, day: dt.date, *, inclusive: bool = False) -> dt.date:
        """The last trading day strictly before ``day`` (or including it)."""
        self._guard(day)
        if inclusive and self.is_trading_day(day):
            return day
        idx = int(self._sessions.searchsorted(pd.Timestamp(day), side="left")) - 1
        if idx < 0:
            raise ValueError(f"No trading day before {day} within the calendar window")
        return self._sessions[idx].date()

    def add_trading_days(self, day: dt.date, n: int) -> dt.date:
        """Shift ``day`` by ``n`` trading days. ``n`` may be negative or zero."""
        self._guard(day)
        if n == 0:
            return day if self.is_trading_day(day) else self.next_trading_day(day)
        step = self.next_trading_day if n > 0 else self.previous_trading_day
        current = day
        for _ in range(abs(n)):
            current = step(current)
        return current

    def trading_days_between(self, start: dt.date, end: dt.date) -> int:
        """Count trading days in the half-open interval ``[start, end)``.

        Half-open so that a position opened and closed on the same day is held
        for zero trading days, which is what the minimum-hold check needs.
        """
        self._guard(start)
        self._guard(end)
        if end <= start:
            return 0
        lo = self._sessions.searchsorted(pd.Timestamp(start), side="left")
        hi = self._sessions.searchsorted(pd.Timestamp(end), side="left")
        return int(hi - lo)

    def settlement_date(self, trade_date: dt.date, t_plus: int = 1) -> dt.date:
        """When the proceeds of a trade on ``trade_date`` become settled cash.

        US equities are T+1 as of May 2024. Settlement counts *trading* days,
        so a Friday sale settles on Monday, and on Tuesday if Monday is a
        holiday.
        """
        if t_plus < 0:
            raise ValueError("t_plus must be non-negative")
        base = trade_date if self.is_trading_day(trade_date) else self.next_trading_day(trade_date)
        return self.add_trading_days(base, t_plus)

    def is_market_open(self, moment: dt.datetime) -> bool:
        """Whether the regular NYSE session is open at ``moment``.

        Uses the calendar's own per-day open/close, so half-days (1pm closes)
        are handled correctly rather than assumed to be 16:00.
        """
        moment = ensure_utc(moment)
        day = to_market(moment).date()
        self._guard(day)
        if not self.is_trading_day(day):
            return False
        calendar = mcal.get_calendar("NYSE")
        sched = calendar.schedule(start_date=day, end_date=day)
        if sched.empty:
            return False
        open_ts: dt.datetime = sched.iloc[0]["market_open"].to_pydatetime()
        close_ts: dt.datetime = sched.iloc[0]["market_close"].to_pydatetime()
        return bool(open_ts <= moment < close_ts)
