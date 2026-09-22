"""The one-way broadcast channel.

Posts trade events, summaries and alerts. Every method is fire-and-forget: it
hands the message to the queue and returns immediately, so nothing here can
delay or fail an order.

Priorities matter during an outage, because the queue sheds its lowest-priority
messages when full. An alert must survive a backlog that a routine daily
summary does not.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from investment_box.core.logging import get_logger
from investment_box.core.types import TradingMode
from investment_box.i18n.translator import Translator
from investment_box.telegram.formatting import (
    SEPARATOR,
    bilingual,
    esc,
    local_time,
    mode_tag,
    money,
    pct,
    qty,
)
from investment_box.telegram.queue import MessageQueue

log = get_logger(__name__)

PRIORITY_ALERT = 100
PRIORITY_TRADE = 50
PRIORITY_SUMMARY = 10


class BroadcastChannel:
    """Posts to the private Telegram channel."""

    def __init__(
        self,
        queue: MessageQueue,
        channel_id: str | None,
        trading_mode: TradingMode,
        translator: Translator,
    ) -> None:
        self.queue = queue
        self.channel_id = channel_id
        self.trading_mode = trading_mode
        self.translator = translator

    @property
    def is_configured(self) -> bool:
        """Whether a channel is set. Unset means broadcasts go nowhere."""
        return bool(self.channel_id)

    def _post(self, text: str, *, priority: int = PRIORITY_SUMMARY) -> None:
        if not self.channel_id:
            log.debug("broadcast.no_channel_configured", preview=text[:60])
            return
        self.queue.enqueue(self.channel_id, text, priority=priority)

    def _wrap(self, body: str) -> str:
        return f"{mode_tag(self.trading_mode)}\n{body}"

    # ----------------------------------------------------------------- trades

    def trade_opened(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        entry_price: Decimal,
        size_pct: float,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
        strategy: str,
        probability: float | None,
        compliance_status: str,
        reason: str,
    ) -> None:
        def build(t: Translator) -> str:
            lines = [
                f"🟢 <b>{esc(t.t('trade.opened'))}</b>",
                "",
                f"<b>{esc(symbol)}</b> {esc(side.upper())} {qty(quantity)} @ {money(entry_price)}",
                f"<b>{esc(t.t('trade.size_pct'))}:</b> {pct(size_pct)} "
                f"{esc(t.t('balance.allocated'))}",
            ]
            if stop_loss is not None:
                lines.append(f"<b>{esc(t.t('trade.stop_loss'))}:</b> {money(stop_loss)}")
            if take_profit is not None:
                lines.append(f"<b>{esc(t.t('trade.take_profit'))}:</b> {money(take_profit)}")
            lines.append(f"<b>{esc(t.t('trade.strategy'))}:</b> {esc(strategy)}")
            if probability is not None:
                lines.append(
                    f"<b>{esc(t.t('trade.probability'))}:</b> {pct(probability, places=1)}"
                )
            lines.append(f"<b>{esc(t.t('compliance.status'))}:</b> {esc(compliance_status)}")
            lines.append(f"<b>{esc(t.t('trade.reason'))}:</b> {esc(reason)}")
            return "\n".join(lines)

        self._post(self._wrap(bilingual(self.translator, build)), priority=PRIORITY_TRADE)

    def trade_closed(
        self,
        *,
        symbol: str,
        exit_price: Decimal,
        holding_days: int,
        pnl: Decimal,
        pnl_pct: float,
        exit_reason: str,
    ) -> None:
        icon = "🔵" if pnl >= 0 else "🔴"

        def build(t: Translator) -> str:
            return "\n".join(
                [
                    f"{icon} <b>{esc(t.t('trade.closed'))}</b>",
                    "",
                    f"<b>{esc(symbol)}</b> @ {money(exit_price)}",
                    f"<b>{esc(t.t('trade.pnl'))}:</b> {money(pnl, signed=True)} "
                    f"({pct(pnl_pct, signed=True)})",
                    f"<b>{esc(t.t('trade.holding_period'))}:</b> {holding_days}d",
                    f"<b>{esc(t.t('trade.exit_reason'))}:</b> {esc(exit_reason)}",
                ]
            )

        self._post(self._wrap(bilingual(self.translator, build)), priority=PRIORITY_TRADE)

    # -------------------------------------------------------------- summaries

    def daily_summary(
        self,
        *,
        equity: Decimal,
        day_pnl: Decimal | None,
        day_pnl_pct: float | None,
        open_positions: int,
        trades_today: int,
        cash_settled: Decimal,
        cash_unsettled: Decimal,
        benchmarks: dict[str, float] | None = None,
        taken_at: dt.datetime | None = None,
    ) -> None:
        def build(t: Translator) -> str:
            lines = [
                f"📊 <b>{esc(t.t('summary.daily_title'))}</b>",
                "",
                f"<b>{esc(t.t('balance.equity'))}:</b> {money(equity)}",
                f"<b>{esc(t.t('balance.day_pnl'))}:</b> {money(day_pnl, signed=True)} "
                f"({pct(day_pnl_pct, signed=True)})",
                f"<b>{esc(t.t('positions.title'))}:</b> {open_positions}",
                f"<b>{esc(t.t('summary.trades_today'))}:</b> {trades_today}",
                f"<b>{esc(t.t('balance.cash_settled'))}:</b> {money(cash_settled)}   "
                f"<b>{esc(t.t('balance.cash_unsettled'))}:</b> {money(cash_unsettled)}",
            ]
            if benchmarks:
                parts = [
                    f"{esc(name)} {pct(value, signed=True)}"
                    for name, value in benchmarks.items()
                ]
                lines.append(f"<b>{esc(t.t('summary.vs_benchmark'))}:</b> {'   '.join(parts)}")
            return "\n".join(lines)

        body = bilingual(self.translator, build)
        stamp = f"\n\n<i>{esc(local_time(taken_at))}</i>" if taken_at else ""
        self._post(self._wrap(body) + stamp, priority=PRIORITY_SUMMARY)

    def weekly_summary(
        self,
        *,
        equity: Decimal,
        week_pnl: Decimal,
        week_pnl_pct: float,
        win_rate: float | None,
        max_drawdown: float | None,
        trades: int,
        purification_total: Decimal | None = None,
    ) -> None:
        def build(t: Translator) -> str:
            lines = [
                f"📈 <b>{esc(t.t('summary.weekly_title'))}</b>",
                "",
                f"<b>{esc(t.t('balance.equity'))}:</b> {money(equity)}",
                f"<b>{esc(t.t('balance.all_time_pnl'))}:</b> {money(week_pnl, signed=True)} "
                f"({pct(week_pnl_pct, signed=True)})",
                f"<b>{esc(t.t('summary.win_rate'))}:</b> "
                f"{pct(win_rate) if win_rate is not None else '—'}   ({trades})",
                f"<b>{esc(t.t('summary.drawdown'))}:</b> "
                f"{pct(max_drawdown) if max_drawdown is not None else '—'}",
            ]
            if purification_total is not None:
                lines.append(
                    f"<b>{esc(t.t('summary.purification'))}:</b> {money(purification_total)}"
                )
            return "\n".join(lines)

        self._post(self._wrap(bilingual(self.translator, build)), priority=PRIORITY_SUMMARY)

    # ----------------------------------------------------------------- alerts

    def alert(self, kind_key: str, detail: str, *, symbol: str | None = None) -> None:
        """Post an alert. Highest priority -- survives a queue backlog."""

        def build(t: Translator) -> str:
            lines = [
                f"🚨 <b>{esc(t.t('alert.title'))}: {esc(t.t(kind_key))}</b>",
                "",
            ]
            if symbol:
                lines.append(f"<b>{esc(symbol)}</b>")
            lines.append(esc(detail))
            return "\n".join(lines)

        self._post(self._wrap(bilingual(self.translator, build)), priority=PRIORITY_ALERT)

    def risk_limit_hit(self, detail: str) -> None:
        self.alert("alert.risk_limit", detail)

    def compliance_changed(self, symbol: str, old: str, new: str, action: str) -> None:
        self.alert(
            "alert.compliance_change",
            f"{old.upper()} → {new.upper()}. {action}",
            symbol=symbol,
        )

    def broker_disconnected(self, detail: str) -> None:
        self.alert("alert.broker_disconnect", detail)

    def kill_switch(self, detail: str) -> None:
        self.alert("alert.kill_switch", detail)

    def error(self, detail: str) -> None:
        self.alert("alert.error", detail)

    def startup(self, lines: list[str]) -> None:
        """Post the startup banner so a restart is visible in the channel."""
        body = "\n".join(esc(line) for line in lines)
        self._post(
            f"{mode_tag(self.trading_mode)}\n<b>Investment Box</b>\n{SEPARATOR}\n{body}",
            priority=PRIORITY_ALERT,
        )

    def custom(self, text: str, *, priority: int = PRIORITY_SUMMARY) -> None:
        """Escape hatch for a pre-built message, e.g. the Phase 3 backtest report."""
        self._post(self._wrap(text), priority=priority)

    def stats(self) -> dict[str, Any]:
        return {"configured": self.is_configured, **self.queue.stats()}
