"""Cleaning and quality validation for OHLCV data.

The rule that governs this module: **never invent a price**. Missing bars are
reported and dropped, not forward-filled. A forward-filled close produces a
zero return, which a momentum strategy reads as "flat" and a volatility
estimate reads as "calm" -- both false, and both in the direction that makes
a backtest look better than reality.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from investment_box.core.errors import DataQualityError
from investment_box.core.logging import get_logger
from investment_box.data.base import INDEX_NAME, REQUIRED_COLUMNS, OHLCVFrame

log = get_logger(__name__)

#: A single-day move beyond this is treated as suspicious. Legitimate for a
#: biotech on trial results; for a broad ETF it usually means a split that the
#: provider did not adjust for.
EXTREME_RETURN_THRESHOLD = 0.35

#: Ratios close to these indicate an unadjusted split rather than a real move.
_COMMON_SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 20.0)
_SPLIT_RATIO_TOLERANCE = 0.03


@dataclass
class DataQualityReport:
    """What cleaning found and did. Attached to logs and surfaced in the UI."""

    symbol: str
    rows_in: int = 0
    rows_out: int = 0
    duplicates_dropped: int = 0
    nan_rows_dropped: int = 0
    invalid_ohlc_dropped: int = 0
    non_positive_dropped: int = 0
    zero_volume_days: int = 0
    extreme_moves: list[str] = field(default_factory=list)
    suspected_unadjusted_splits: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)

    @property
    def rows_dropped(self) -> int:
        return self.rows_in - self.rows_out

    @property
    def is_clean(self) -> bool:
        """No structural problems. Extreme moves alone do not make data unusable."""
        return (
            self.rows_dropped == 0
            and not self.suspected_unadjusted_splits
        )

    def summary(self) -> str:
        parts = [f"{self.symbol}: {self.rows_in}->{self.rows_out} rows"]
        if self.duplicates_dropped:
            parts.append(f"{self.duplicates_dropped} dup")
        if self.nan_rows_dropped:
            parts.append(f"{self.nan_rows_dropped} NaN")
        if self.invalid_ohlc_dropped:
            parts.append(f"{self.invalid_ohlc_dropped} bad OHLC")
        if self.non_positive_dropped:
            parts.append(f"{self.non_positive_dropped} non-positive")
        if self.suspected_unadjusted_splits:
            parts.append(f"{len(self.suspected_unadjusted_splits)} suspected unadjusted split(s)")
        return ", ".join(parts)


def clean_ohlcv(frame: OHLCVFrame, symbol: str, *, strict: bool = False) -> tuple[OHLCVFrame, DataQualityReport]:
    """Normalise and clean a raw provider frame.

    Steps, in order: normalise the index to UTC and sort it, drop duplicate
    timestamps (keeping the last), drop rows with missing prices, drop rows
    with non-positive prices, drop rows violating ``low <= open,close <= high``,
    then flag -- but keep -- extreme moves and suspected unadjusted splits.

    Args:
        frame: Raw provider output.
        symbol: For error messages and the report.
        strict: Raise :class:`DataQualityError` when a suspected unadjusted
            split is found, instead of only flagging it.

    Returns:
        The cleaned frame and a :class:`DataQualityReport`.
    """
    report = DataQualityReport(symbol=symbol, rows_in=len(frame))

    if frame.empty:
        return frame, report

    out = frame.copy()

    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise DataQualityError(symbol, f"missing columns: {sorted(missing)}")

    # --- index ---------------------------------------------------------------
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, utc=True)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    out.index.name = INDEX_NAME
    out = out.sort_index()

    before = len(out)
    out = out[~out.index.duplicated(keep="last")]
    report.duplicates_dropped = before - len(out)

    out = out[list(REQUIRED_COLUMNS)].astype("float64")

    # --- row-level validity --------------------------------------------------
    before = len(out)
    out = out.dropna(subset=list(REQUIRED_COLUMNS))
    report.nan_rows_dropped = before - len(out)

    before = len(out)
    price_cols = ["open", "high", "low", "close"]
    out = out[(out[price_cols] > 0).all(axis=1)]
    report.non_positive_dropped = before - len(out)

    before = len(out)
    valid = (
        (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
        & (out["volume"] >= 0)
    )
    out = out[valid]
    report.invalid_ohlc_dropped = before - len(out)

    report.rows_out = len(out)
    if out.empty:
        return out, report

    # --- flags (kept, not dropped) ------------------------------------------
    report.zero_volume_days = int((out["volume"] == 0).sum())

    returns = out["close"].pct_change()
    extreme = returns[returns.abs() > EXTREME_RETURN_THRESHOLD]
    for ts, value in extreme.items():
        stamp = cast(pd.Timestamp, ts).date().isoformat()
        report.extreme_moves.append(f"{stamp}: {value:+.1%}")
        ratio = _split_ratio_if_suspicious(float(value))
        if ratio is not None:
            report.suspected_unadjusted_splits.append(f"{stamp}: looks like a {ratio:g}:1 split")

    if report.suspected_unadjusted_splits:
        log.warning(
            "data.suspected_unadjusted_split",
            symbol=symbol,
            occurrences=report.suspected_unadjusted_splits,
        )
        if strict:
            raise DataQualityError(
                symbol,
                "suspected unadjusted split(s): "
                + "; ".join(report.suspected_unadjusted_splits)
                + ". Refusing to use this data -- signals computed on unadjusted "
                "prices are wrong in a way that flatters backtests.",
            )

    if report.rows_dropped:
        log.info("data.cleaned", symbol=symbol, summary=report.summary())

    return out, report


def _split_ratio_if_suspicious(daily_return: float) -> float | None:
    """Return the implied split ratio if ``daily_return`` looks like an unadjusted split."""
    if daily_return >= 0:
        # A forward split shows up as a large negative return in the price series.
        return None
    ratio = 1.0 / (1.0 + daily_return)
    for candidate in _COMMON_SPLIT_RATIOS:
        if abs(ratio - candidate) / candidate <= _SPLIT_RATIO_TOLERANCE:
            return candidate
    return None


def validate_ohlcv(
    frame: OHLCVFrame,
    symbol: str,
    expected_sessions: list[dt.date] | None = None,
    *,
    max_missing_ratio: float = 0.05,
) -> DataQualityReport:
    """Check a cleaned frame against the trading calendar.

    Args:
        frame: A frame already through :func:`clean_ohlcv`.
        symbol: For messages.
        expected_sessions: Trading days that should be present. Usually
            ``TradingCalendar.sessions`` sliced to the frame's range.
        max_missing_ratio: Above this fraction of missing sessions the data is
            treated as unusable rather than merely gappy.

    Raises:
        DataQualityError: When too many sessions are missing.
    """
    report = DataQualityReport(symbol=symbol, rows_in=len(frame), rows_out=len(frame))
    if frame.empty or not expected_sessions:
        return report

    present = {ts.date() for ts in frame.index}
    # Clip the START to the first bar we have: a fund that listed in 2023 has no
    # 2015 data, and calling that a gap would be noise.
    # Do NOT clip the end. A feed that stops early -- a stale cache, a truncated
    # response, a delisting -- is exactly the failure this check exists to catch,
    # and deriving the end from the data itself would make it invisible.
    lo = min(present)
    expected = [day for day in expected_sessions if day >= lo]
    if not expected:
        return report

    missing = sorted(set(expected) - present)
    report.gaps = [day.isoformat() for day in missing]

    ratio = len(missing) / len(expected)
    if ratio > max_missing_ratio:
        raise DataQualityError(
            symbol,
            f"{len(missing)} of {len(expected)} expected sessions missing "
            f"({ratio:.1%} > {max_missing_ratio:.1%}); first missing {missing[0]}",
        )
    if missing:
        log.info("data.gaps", symbol=symbol, missing=len(missing), first=str(missing[0]))
    return report


def assert_no_lookahead(frame: OHLCVFrame, as_of: pd.Timestamp | None) -> None:
    """Assert that ``frame`` contains no bar at or after ``as_of``.

    Called by the backtester before handing data to a strategy. A strategy that
    can see the bar it is deciding on will beat any benchmark and will lose
    money live, so this is a hard check rather than a warning.
    """
    if as_of is None or frame.empty:
        return
    cutoff = as_of if as_of.tzinfo else as_of.tz_localize("UTC")
    leaked = frame.index[frame.index >= cutoff]
    if len(leaked) > 0:
        raise AssertionError(
            f"Look-ahead: {len(leaked)} bar(s) at or after the as-of time {cutoff} "
            f"were visible to the strategy (first leaked: {leaked[0]})"
        )


def summarise_coverage(frame: OHLCVFrame) -> dict[str, object]:
    """Compact description of what a frame covers, for logs and the UI."""
    if frame.empty:
        return {"rows": 0, "start": None, "end": None}
    return {
        "rows": len(frame),
        "start": frame.index[0].date().isoformat(),
        "end": frame.index[-1].date().isoformat(),
        "median_dollar_volume": float(
            np.nanmedian((frame["close"] * frame["volume"]).to_numpy())
        ),
    }
