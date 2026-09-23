"""Parquet cache for OHLCV bars.

Daily bars for a decade are a few hundred kilobytes per symbol, so the cache is
one file per symbol per timeframe. Two properties matter:

* **Freshness is decided by the last bar, not the file mtime.** A file written
  five minutes ago whose last bar is from three weeks ago is stale.
* **Writes are atomic.** Written to a temp file then renamed, so a crash
  mid-write cannot leave a truncated parquet that poisons every later read.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import cast

import pandas as pd

from investment_box.core.clock import UTC
from investment_box.core.logging import get_logger
from investment_box.data.base import INDEX_NAME, OHLCVFrame, empty_frame

log = get_logger(__name__)

_SAFE_SYMBOL = re.compile(r"[^A-Za-z0-9._-]")


class ParquetCache:
    """On-disk cache of cleaned bars."""

    def __init__(self, directory: Path, ttl_hours: int = 12) -> None:
        self.directory = Path(directory)
        self.ttl = dt.timedelta(hours=ttl_hours)
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, symbol: str, timeframe: str = "1d") -> Path:
        """Cache path for a symbol.

        Tickers like ``^VIX`` and ``BRK.B`` contain characters that are awkward
        or unsafe in filenames, so they are sanitised.
        """
        safe = _SAFE_SYMBOL.sub("_", symbol.upper())
        return self.directory / f"{safe}__{timeframe}.parquet"

    def read(self, symbol: str, timeframe: str = "1d") -> OHLCVFrame:
        """Return cached bars, or an empty frame if absent or unreadable."""
        path = self.path_for(symbol, timeframe)
        if not path.exists():
            return empty_frame()
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - a corrupt cache must not be fatal
            log.warning("cache.read_failed", symbol=symbol, path=str(path), error=str(exc))
            return empty_frame()

        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index, utc=True)
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        frame.index.name = INDEX_NAME
        return frame

    def write(self, symbol: str, frame: OHLCVFrame, timeframe: str = "1d") -> None:
        """Atomically replace the cached frame for ``symbol``."""
        if frame.empty:
            return
        path = self.path_for(symbol, timeframe)
        tmp = path.with_suffix(f".parquet.tmp.{os.getpid()}")
        try:
            frame.to_parquet(tmp, engine="pyarrow", compression="snappy")
            os.replace(tmp, path)  # atomic within a filesystem
        finally:
            tmp.unlink(missing_ok=True)

    def merge_write(self, symbol: str, new_rows: OHLCVFrame, timeframe: str = "1d") -> OHLCVFrame:
        """Merge ``new_rows`` over the cached frame and persist the result.

        New rows win on conflict, so a re-fetch that carries a late split
        adjustment corrects the cache rather than being ignored.
        """
        existing = self.read(symbol, timeframe)
        if existing.empty:
            merged = new_rows
        elif new_rows.empty:
            merged = existing
        else:
            merged = pd.concat([existing, new_rows])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.write(symbol, merged, timeframe)
        return merged

    def last_bar_date(self, symbol: str, timeframe: str = "1d") -> dt.date | None:
        frame = self.read(symbol, timeframe)
        if frame.empty:
            return None
        return cast(pd.Timestamp, frame.index[-1]).date()

    def is_fresh(
        self, symbol: str, timeframe: str = "1d", *, now: dt.datetime | None = None
    ) -> bool:
        """Whether the cache is fresh enough to serve without re-fetching.

        Freshness is measured from the newest *bar*, not the file's mtime.
        """
        if self.ttl.total_seconds() == 0:
            return False
        last = self.last_bar_date(symbol, timeframe)
        if last is None:
            return False
        reference = (now or dt.datetime.now(tz=UTC)).astimezone(UTC)
        age = reference - dt.datetime.combine(last, dt.time(0, 0), tzinfo=UTC)
        return age <= self.ttl

    def invalidate(self, symbol: str, timeframe: str = "1d") -> None:
        self.path_for(symbol, timeframe).unlink(missing_ok=True)

    def clear(self) -> int:
        """Delete every cached file. Returns how many were removed."""
        removed = 0
        for path in self.directory.glob("*.parquet"):
            path.unlink(missing_ok=True)
            removed += 1
        return removed
