"""Scheduling.

APScheduler drives four jobs:

* **The trading cycle**, shortly after the close. Signals are computed from a
  completed bar and executed on the next session's open, matching exactly what
  the backtester assumed. Running mid-session would trade on a partial bar the
  backtest never saw.
* **Approval expiry**, every minute, so timed-out requests are visibly closed.
* **The daily summary**, after the cycle.
* **The weekly summary**, after Friday's close.

Every job is wrapped so a failure is logged and the schedule survives. A
scheduler that dies on one bad cycle is worse than one that reports the error
and tries again tomorrow.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from investment_box.config.schema import Settings
from investment_box.core.audit import AuditSink
from investment_box.core.clock import Clock, SystemClock, TradingCalendar
from investment_box.core.logging import get_logger

log = get_logger(__name__)

MARKET_TZ = ZoneInfo("America/New_York")


@dataclass
class ScheduledJob:
    """One job, for display in the dashboard and ``/status``."""

    name: str
    description: str
    next_run: dt.datetime | None = None


class EngineScheduler:
    """Wraps APScheduler with market-aware guards."""

    def __init__(
        self,
        settings: Settings,
        audit: AuditSink,
        *,
        clock: Clock | None = None,
        calendar: TradingCalendar | None = None,
    ) -> None:
        self.settings = settings
        self.audit = audit
        self.clock = clock or SystemClock()
        self.calendar = calendar or TradingCalendar()
        self._scheduler: object | None = None
        self._jobs: list[ScheduledJob] = []

    @property
    def is_running(self) -> bool:
        scheduler = self._scheduler
        return bool(scheduler and getattr(scheduler, "running", False))

    def is_trading_day(self, day: dt.date | None = None) -> bool:
        target = day or self.clock.now().astimezone(MARKET_TZ).date()
        try:
            return self.calendar.is_trading_day(target)
        except ValueError:
            # Outside the loaded calendar window: refuse rather than guess.
            log.warning("scheduler.date_outside_calendar", date=str(target))
            return False

    def build(
        self,
        *,
        on_cycle: Callable[[], Awaitable[None]],
        on_expire_approvals: Callable[[], Awaitable[None]] | None = None,
        on_daily_summary: Callable[[], Awaitable[None]] | None = None,
        on_weekly_summary: Callable[[], Awaitable[None]] | None = None,
    ) -> object:
        """Construct the scheduler and register every job."""
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        scheduler = AsyncIOScheduler(timezone=MARKET_TZ)
        hour, _, minute = self.settings.engine.signal_time.partition(":")

        scheduler.add_job(
            self._guard("trading cycle", on_cycle, trading_day_only=True),
            CronTrigger(day_of_week="mon-fri", hour=int(hour), minute=int(minute)),
            id="trading_cycle",
            max_instances=1,
            coalesce=True,  # a missed run does not pile up
            misfire_grace_time=1800,
        )
        self._jobs.append(
            ScheduledJob(
                "trading_cycle",
                f"weekdays at {self.settings.engine.signal_time} ET, trading days only",
            )
        )

        if on_expire_approvals is not None:
            scheduler.add_job(
                self._guard("approval expiry", on_expire_approvals),
                CronTrigger(minute="*"),
                id="expire_approvals",
                max_instances=1,
                coalesce=True,
            )
            self._jobs.append(ScheduledJob("expire_approvals", "every minute"))

        if on_daily_summary is not None:
            scheduler.add_job(
                self._guard("daily summary", on_daily_summary, trading_day_only=True),
                CronTrigger(day_of_week="mon-fri", hour=16, minute=45),
                id="daily_summary",
                max_instances=1,
                coalesce=True,
            )
            self._jobs.append(ScheduledJob("daily_summary", "weekdays 16:45 ET"))

        if on_weekly_summary is not None:
            scheduler.add_job(
                self._guard("weekly summary", on_weekly_summary),
                CronTrigger(day_of_week="fri", hour=17, minute=0),
                id="weekly_summary",
                max_instances=1,
                coalesce=True,
            )
            self._jobs.append(ScheduledJob("weekly_summary", "Fridays 17:00 ET"))

        self._scheduler = scheduler
        return scheduler

    def _guard(
        self,
        name: str,
        job: Callable[[], Awaitable[None]],
        *,
        trading_day_only: bool = False,
    ) -> Callable[[], Awaitable[None]]:
        """Wrap a job so a failure is recorded and the schedule survives."""

        async def wrapped() -> None:
            if trading_day_only and not self.is_trading_day():
                log.info("scheduler.skipped_non_trading_day", job=name)
                return
            log.info("scheduler.job_started", job=name)
            try:
                await job()
            except Exception as exc:  # noqa: BLE001 - see the module docstring
                log.error("scheduler.job_failed", job=name, error=str(exc))
                self.audit.record(
                    "scheduler.job_failed", f"{name} raised {type(exc).__name__}: {exc}",
                    actor="scheduler",
                )
            else:
                log.info("scheduler.job_finished", job=name)

        return wrapped

    def start(self) -> None:
        if self._scheduler is None:
            raise RuntimeError("call build() before start()")
        self._scheduler.start()  # type: ignore[attr-defined]
        self.audit.record("scheduler.started", "Scheduler started", actor="scheduler")
        log.info("scheduler.started", jobs=[j.name for j in self._jobs])

    def shutdown(self, *, wait: bool = True) -> None:
        if self._scheduler is not None and self.is_running:
            self._scheduler.shutdown(wait=wait)  # type: ignore[attr-defined]
            self.audit.record("scheduler.stopped", "Scheduler stopped", actor="scheduler")

    def jobs(self) -> list[ScheduledJob]:
        """Registered jobs with their next run times."""
        scheduler = self._scheduler
        if scheduler is None:
            return self._jobs
        for job in self._jobs:
            found = scheduler.get_job(job.name)  # type: ignore[attr-defined]
            job.next_run = getattr(found, "next_run_time", None) if found else None
        return self._jobs
