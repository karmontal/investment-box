"""Calendar and settlement arithmetic.

These tests use real dates with known quirks -- holidays, long weekends, the
July 4 half-day -- rather than synthetic ones, because the bugs here come from
exactly those cases.
"""

from __future__ import annotations

import datetime as dt

import pytest

from investment_box.core.clock import (
    UTC,
    FrozenClock,
    SystemClock,
    TradingCalendar,
    ensure_utc,
    to_display,
    to_market,
)


class TestClocks:
    def test_system_clock_is_utc_aware(self) -> None:
        assert SystemClock().now().tzinfo is not None

    def test_frozen_clock_does_not_move(self) -> None:
        clock = FrozenClock(dt.datetime(2024, 6, 12, 12, 0, tzinfo=UTC))
        assert clock.now() == clock.now()

    def test_frozen_clock_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            FrozenClock(dt.datetime(2024, 6, 12, 12, 0))  # noqa: DTZ001

    def test_advance(self) -> None:
        clock = FrozenClock(dt.datetime(2024, 6, 12, 12, 0, tzinfo=UTC))
        clock.advance(days=2)
        assert clock.now().date() == dt.date(2024, 6, 14)


class TestTimezoneHelpers:
    def test_ensure_utc_rejects_naive(self) -> None:
        with pytest.raises(ValueError, match="Naive datetime"):
            ensure_utc(dt.datetime(2024, 6, 12, 12, 0))  # noqa: DTZ001

    def test_display_conversion(self) -> None:
        moment = dt.datetime(2024, 6, 12, 20, 15, tzinfo=UTC)
        local = to_display(moment)
        assert local.hour == 23  # Asia/Jerusalem is UTC+3 in June
        assert local.tzinfo is not None

    def test_market_conversion(self) -> None:
        # 20:15 UTC is 16:15 New York in June -- just after the close.
        assert to_market(dt.datetime(2024, 6, 12, 20, 15, tzinfo=UTC)).hour == 16


class TestTradingDays:
    @pytest.fixture
    def cal(self) -> TradingCalendar:
        return TradingCalendar(anchor=dt.date(2024, 6, 1))

    def test_weekend_is_not_a_trading_day(self, cal: TradingCalendar) -> None:
        assert not cal.is_trading_day(dt.date(2024, 6, 15))  # Saturday
        assert not cal.is_trading_day(dt.date(2024, 6, 16))  # Sunday
        assert cal.is_trading_day(dt.date(2024, 6, 14))  # Friday

    def test_holiday_is_not_a_trading_day(self, cal: TradingCalendar) -> None:
        assert not cal.is_trading_day(dt.date(2024, 7, 4))  # Independence Day
        assert not cal.is_trading_day(dt.date(2024, 6, 19))  # Juneteenth

    def test_next_trading_day_skips_the_weekend(self, cal: TradingCalendar) -> None:
        assert cal.next_trading_day(dt.date(2024, 6, 14)) == dt.date(2024, 6, 17)

    def test_next_trading_day_inclusive(self, cal: TradingCalendar) -> None:
        friday = dt.date(2024, 6, 14)
        assert cal.next_trading_day(friday, inclusive=True) == friday
        assert cal.next_trading_day(friday, inclusive=False) == dt.date(2024, 6, 17)

    def test_previous_trading_day(self, cal: TradingCalendar) -> None:
        assert cal.previous_trading_day(dt.date(2024, 6, 17)) == dt.date(2024, 6, 14)

    def test_add_trading_days_over_a_holiday(self, cal: TradingCalendar) -> None:
        # Jul 3 (half day) -> Jul 4 is a holiday -> Jul 5 -> Jul 8 (Monday)
        assert cal.add_trading_days(dt.date(2024, 7, 3), 2) == dt.date(2024, 7, 8)

    def test_add_negative_trading_days(self, cal: TradingCalendar) -> None:
        assert cal.add_trading_days(dt.date(2024, 6, 17), -1) == dt.date(2024, 6, 14)

    def test_trading_days_between_is_half_open(self, cal: TradingCalendar) -> None:
        same = dt.date(2024, 6, 12)
        assert cal.trading_days_between(same, same) == 0

    def test_trading_days_between_excludes_weekend(self, cal: TradingCalendar) -> None:
        # Fri 14th -> Mon 17th is one trading day, not three calendar days.
        assert cal.trading_days_between(dt.date(2024, 6, 14), dt.date(2024, 6, 17)) == 1

    def test_minimum_hold_over_a_long_weekend(self, cal: TradingCalendar) -> None:
        """A 2-trading-day hold opened on Friday cannot exit before Tuesday.

        Counting calendar days would allow a Sunday exit, which is the bug this
        test exists to catch.
        """
        entry = dt.date(2024, 6, 14)  # Friday
        earliest_exit = cal.add_trading_days(entry, 2)
        assert earliest_exit == dt.date(2024, 6, 18)  # Tuesday
        assert cal.trading_days_between(entry, earliest_exit) == 2

    def test_out_of_window_date_raises(self, cal: TradingCalendar) -> None:
        with pytest.raises(ValueError, match="outside the loaded calendar"):
            cal.is_trading_day(dt.date(1990, 1, 2))


class TestSettlement:
    @pytest.fixture
    def cal(self) -> TradingCalendar:
        return TradingCalendar(anchor=dt.date(2024, 6, 1))

    def test_t_plus_one_on_a_normal_day(self, cal: TradingCalendar) -> None:
        assert cal.settlement_date(dt.date(2024, 6, 12), 1) == dt.date(2024, 6, 13)

    def test_friday_sale_settles_monday(self, cal: TradingCalendar) -> None:
        assert cal.settlement_date(dt.date(2024, 6, 14), 1) == dt.date(2024, 6, 17)

    def test_settlement_skips_a_holiday(self, cal: TradingCalendar) -> None:
        # Sold Wed Jul 3; Jul 4 is a holiday, so proceeds settle Fri Jul 5.
        assert cal.settlement_date(dt.date(2024, 7, 3), 1) == dt.date(2024, 7, 5)

    def test_sale_on_a_non_trading_day_rolls_forward(self, cal: TradingCalendar) -> None:
        # A timestamp landing on a Saturday is treated as the next session.
        assert cal.settlement_date(dt.date(2024, 6, 15), 1) == dt.date(2024, 6, 18)

    def test_t_plus_zero_is_the_trade_date(self, cal: TradingCalendar) -> None:
        assert cal.settlement_date(dt.date(2024, 6, 12), 0) == dt.date(2024, 6, 12)

    def test_negative_t_plus_rejected(self, cal: TradingCalendar) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            cal.settlement_date(dt.date(2024, 6, 12), -1)


class TestMarketHours:
    def test_closed_on_a_holiday(self) -> None:
        cal = TradingCalendar(anchor=dt.date(2024, 7, 1))
        assert not cal.is_market_open(dt.datetime(2024, 7, 4, 15, 0, tzinfo=UTC))

    def test_open_mid_session(self) -> None:
        cal = TradingCalendar(anchor=dt.date(2024, 6, 1))
        # 15:00 UTC == 11:00 ET, mid-session.
        assert cal.is_market_open(dt.datetime(2024, 6, 12, 15, 0, tzinfo=UTC))

    def test_closed_after_the_bell(self) -> None:
        cal = TradingCalendar(anchor=dt.date(2024, 6, 1))
        # 21:00 UTC == 17:00 ET.
        assert not cal.is_market_open(dt.datetime(2024, 6, 12, 21, 0, tzinfo=UTC))

    def test_half_day_close(self) -> None:
        """July 3 2024 closed at 13:00 ET, not 16:00."""
        cal = TradingCalendar(anchor=dt.date(2024, 7, 1))
        # 17:00 UTC == 13:00 ET -- already closed on a half day.
        assert not cal.is_market_open(dt.datetime(2024, 7, 3, 17, 30, tzinfo=UTC))
        assert cal.is_market_open(dt.datetime(2024, 7, 3, 15, 0, tzinfo=UTC))
