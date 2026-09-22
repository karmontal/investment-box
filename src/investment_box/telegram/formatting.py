"""Message construction.

Every outgoing message starts with ``[PAPER]`` or ``[LIVE]``. That tag is
prepended here, in one place, rather than by each call site -- a message whose
mode is ambiguous is worse than no message.

Bilingual rendering: when the language is ``both``, the message body is built
once per language and the two blocks are joined by a rule. Interleaving them
line by line produces something neither language reads well, and mixing RTL and
LTR inside a line mangles tickers and numbers.

Numbers, tickers, prices and timestamps are never localised.
"""

from __future__ import annotations

import datetime as dt
import html
from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from investment_box.core.types import ApprovalKind, TradingMode
from investment_box.i18n.translator import RLM, Translator
from investment_box.services.approvals import ApprovalRequest
from investment_box.services.portfolio import AccountView, PositionView

SEPARATOR = "─" * 24
DEFAULT_TZ = ZoneInfo("Asia/Jerusalem")


def mode_tag(mode: TradingMode) -> str:
    """The ``[PAPER]`` / ``[LIVE]`` prefix carried by every message."""
    return f"[{mode.value.upper()}]"


def esc(value: object) -> str:
    """Escape for Telegram's HTML parse mode.

    Applied to anything that could contain user- or vendor-supplied text --
    fund names, strategy names, exception messages. An unescaped ``&`` in
    "S&P 500 Sharia" is enough for Telegram to reject the whole message.
    """
    return html.escape(str(value), quote=False)


def money(value: Decimal | float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    amount = Decimal(str(value))
    sign = "+" if signed and amount > 0 else ""
    return f"{sign}${amount:,.2f}"


def pct(value: float | None, *, signed: bool = False, places: int = 2) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.{places}f}%"


def qty(value: Decimal | float | None) -> str:
    """Render a share quantity, trimming trailing zeros on fractional sizes."""
    if value is None:
        return "—"
    amount = Decimal(str(value))
    if amount == amount.to_integral_value():
        return str(int(amount))
    return f"{amount.normalize():f}"


def local_time(moment: dt.datetime, tz: ZoneInfo = DEFAULT_TZ) -> str:
    return moment.astimezone(tz).strftime("%Y-%m-%d %H:%M %Z")


def bilingual(
    translator: Translator, build: Callable[[Translator], str]
) -> str:
    """Render ``build`` once per configured language.

    ``build`` receives a single-language translator, so formatters never have
    to think about the bilingual case themselves.
    """
    if translator.language != "both":
        return build(translator)
    english = build(Translator("en"))
    arabic = build(Translator("ar"))
    return f"{english}\n{SEPARATOR}\n{RLM}{arabic}"


def _line(label: str, value: object) -> str:
    return f"<b>{esc(label)}:</b> {esc(value)}"


# --------------------------------------------------------------------- account


def format_balance(view: AccountView, translator: Translator, tz: ZoneInfo = DEFAULT_TZ) -> str:
    """The ``/balance`` reply."""

    def build(t: Translator) -> str:
        lines = [
            f"<b>{esc(t.t('balance.title'))}</b>",
            "",
            _line(t.t("balance.equity"), money(view.equity)),
            _line(t.t("balance.cash_settled"), money(view.cash_settled)),
            _line(t.t("balance.cash_unsettled"), money(view.cash_unsettled)),
            _line(t.t("balance.allocated"), money(view.capital.allocation)),
            _line(
                t.t("balance.deployed"),
                f"{money(view.capital.deployed)} ({pct(view.capital.deployed_pct)})",
            ),
            _line(t.t("balance.available"), money(view.capital.available_settled)),
            "",
            _line(
                t.t("balance.day_pnl"),
                f"{money(view.day_pnl, signed=True)} ({pct(view.day_pnl_pct, signed=True)})"
                if view.day_pnl is not None
                else "—",
            ),
            _line(t.t("balance.all_time_pnl"), money(view.all_time_pnl, signed=True)),
        ]
        return "\n".join(lines)

    body = bilingual(translator, build)
    return f"{mode_tag(view.trading_mode)}\n{body}\n\n<i>{esc(local_time(view.taken_at, tz))}</i>"


def format_positions(view: AccountView, translator: Translator) -> str:
    """The ``/positions`` reply."""

    def build(t: Translator) -> str:
        if not view.positions:
            return f"<b>{esc(t.t('positions.title'))}</b>\n\n{esc(t.t('positions.none'))}"

        blocks = [f"<b>{esc(t.t('positions.title'))}</b> ({len(view.positions)})"]
        for position in view.positions:
            blocks.append(_position_block(position, t))
        return "\n\n".join(blocks)

    body = bilingual(translator, build)
    warnings = _warning_block(view.warnings)
    return f"{mode_tag(view.trading_mode)}\n{body}{warnings}"


def _position_block(position: PositionView, t: Translator) -> str:
    lines = [
        f"<b>{esc(position.symbol)}</b>  "
        f"{qty(position.quantity)} @ {money(position.avg_entry_price)}",
        f"  {esc(t.t('positions.current'))}: {money(position.current_price)}  "
        f"{money(position.unrealized_pnl, signed=True)} "
        f"({pct(position.unrealized_pnl_pct, signed=True)})",
        f"  {esc(t.t('positions.days_held'))}: "
        f"{position.days_held if position.days_held is not None else '—'}"
        f"   {pct(position.pct_of_allocation)} {esc(t.t('balance.allocated'))}",
    ]
    if position.stop_loss_price is not None:
        marker = f" ({esc(t.t('positions.synthetic_stop'))})" if position.stop_is_synthetic else ""
        lines.append(f"  {esc(t.t('positions.stop'))}: {money(position.stop_loss_price)}{marker}")
    if position.take_profit_price is not None:
        lines.append(f"  {esc(t.t('positions.target'))}: {money(position.take_profit_price)}")
    if position.compliance_status:
        lines.append(f"  {esc(t.t('compliance.status'))}: {esc(position.compliance_status)}")
    return "\n".join(lines)


def format_status(
    *,
    view: AccountView,
    engine_state: str,
    autonomy_level: str,
    data_provider: str,
    queue_stats: dict[str, Any],
    translator: Translator,
) -> str:
    """The ``/status`` reply."""

    def build(t: Translator) -> str:
        return "\n".join(
            [
                f"<b>{esc(t.t('status.title'))}</b>",
                "",
                _line(t.t("telegram.engine_state"), engine_state),
                _line(t.t("status.mode"), view.trading_mode.value.upper()),
                _line(t.t("status.autonomy"), autonomy_level),
                _line(t.t("status.broker"), view.broker_name),
                _line(t.t("status.data_provider"), data_provider),
                _line(
                    t.t("telegram.queue"),
                    f"{queue_stats.get('queued', 0)} queued, "
                    f"{queue_stats.get('sent', 0)} sent, "
                    f"{queue_stats.get('dropped', 0)} dropped",
                ),
            ]
        )

    body = bilingual(translator, build)
    return f"{mode_tag(view.trading_mode)}\n{body}{_warning_block(view.warnings)}"


def format_funds(
    funds: Sequence[dict[str, Any]], mode: TradingMode, translator: Translator
) -> str:
    """The ``/funds`` reply.

    Phase 2 shows the configured universe and its verification state. Live
    rankings arrive in Phase 4 -- until then this deliberately shows no scores
    rather than placeholder ones that could be mistaken for signals.
    """

    def build(t: Translator) -> str:
        lines = [f"<b>{esc(t.t('funds.title'))}</b>"]
        for fund in funds:
            mark = "✅" if fund.get("verified") else "⚠️"
            name = fund.get("name") or "—"
            lines.append(f"{mark} <b>{esc(fund['symbol'])}</b> — {esc(name)}")
            status = fund.get("compliance_status")
            if status:
                lines.append(f"    {esc(t.t('compliance.status'))}: {esc(status)}")
        unverified = [f["symbol"] for f in funds if not f.get("verified")]
        if unverified:
            lines.append("")
            lines.append(esc(t.t("funds.unverified_note", count=len(unverified))))
        return "\n".join(lines)

    return f"{mode_tag(mode)}\n{bilingual(translator, build)}"


def format_history(trades: Sequence[Any], mode: TradingMode, translator: Translator) -> str:
    """The ``/history`` reply."""

    def build(t: Translator) -> str:
        if not trades:
            return esc(t.t("telegram.no_history"))
        lines = [f"<b>{esc(t.t('history.title'))}</b> ({len(trades)})"]
        for trade in trades:
            pnl = money(trade.net_pnl, signed=True)
            pnl_pct = ""
            if trade.entry_price and trade.exit_price and trade.quantity:
                basis = Decimal(str(trade.entry_price)) * Decimal(str(trade.quantity))
                if basis:
                    ratio = float(Decimal(str(trade.net_pnl or 0)) / basis)
                    pnl_pct = f" ({pct(ratio, signed=True)})"
            lines.append(
                f"<b>{esc(trade.symbol)}</b> {money(trade.entry_price)} → "
                f"{money(trade.exit_price)}  {pnl}{pnl_pct}"
            )
            lines.append(
                f"    {esc(trade.exit_date)}  "
                f"{esc(t.t('trade.exit_reason'))}: {esc(trade.exit_reason or '—')}  "
                f"{trade.holding_trading_days if trade.holding_trading_days is not None else '—'}d"
            )
        return "\n".join(lines)

    return f"{mode_tag(mode)}\n{bilingual(translator, build)}"


def format_help(translator: Translator, mode: TradingMode) -> str:
    """The ``/help`` reply."""

    def build(t: Translator) -> str:
        commands = [
            ("/status", t.t("telegram.help_status")),
            ("/balance", t.t("telegram.help_balance")),
            ("/positions", t.t("telegram.help_positions")),
            ("/funds", t.t("telegram.help_funds")),
            ("/history [n]", t.t("telegram.help_history")),
            ("/pending", t.t("telegram.help_pending")),
            ("/help", t.t("telegram.help_help")),
        ]
        lines = [f"<b>{esc(t.t('telegram.help_title'))}</b>", ""]
        lines += [f"<code>{esc(cmd)}</code> — {esc(desc)}" for cmd, desc in commands]
        lines += ["", f"<i>{esc(t.t('telegram.help_readonly_note'))}</i>"]
        lines += [f"<i>{esc(t.t('telegram.help_live_note'))}</i>"]
        return "\n".join(lines)

    return f"{mode_tag(mode)}\n{bilingual(translator, build)}"


# -------------------------------------------------------------------- approval

_KIND_KEYS = {
    ApprovalKind.TRADE_PROPOSAL: "approval.trade_proposal",
    ApprovalKind.COMPLIANCE_EXIT: "approval.compliance_exit",
    ApprovalKind.RISK_OVERRIDE: "approval.risk_override",
    ApprovalKind.QUESTION: "approval.question",
}


def format_approval_request(
    request: ApprovalRequest,
    mode: TradingMode,
    translator: Translator,
    *,
    now: dt.datetime,
) -> str:
    """The body of an approval message.

    The deadline and the "no answer means rejection" rule are stated in the
    message itself. A user should not have to remember the policy to read the
    message correctly.
    """
    payload = request.payload

    def build(t: Translator) -> str:
        lines = [
            f"<b>⚠️ {esc(t.t('approval.title'))} — {esc(t.t(_KIND_KEYS[request.kind]))}</b>",
            "",
        ]
        if payload.get("symbol"):
            headline = f"<b>{esc(payload['symbol'])}</b>"
            if payload.get("side"):
                headline += f"  {esc(str(payload['side']).upper())}"
            if payload.get("quantity"):
                headline += f"  {esc(qty(Decimal(str(payload['quantity']))))}"
            if payload.get("entry_price"):
                headline += f" @ {money(Decimal(str(payload['entry_price'])))}"
            lines.append(headline)

        optional: list[tuple[str, str, Callable[[Any], str]]] = [
            ("trade.size_pct", "size_pct", lambda v: pct(float(v))),
            ("trade.stop_loss", "stop_loss", lambda v: money(Decimal(str(v)))),
            ("trade.take_profit", "take_profit", lambda v: money(Decimal(str(v)))),
            ("trade.strategy", "strategy", str),
            ("trade.probability", "probability", lambda v: pct(float(v), places=1)),
            ("compliance.status", "compliance_status", str),
            ("trade.reason", "reason", str),
        ]
        for key, field, render in optional:
            if payload.get(field) is not None:
                lines.append(_line(t.t(key), render(payload[field])))

        if payload.get("question"):
            lines.append(esc(payload["question"]))

        if payload.get("backtest_summary"):
            lines.append("")
            lines.append(f"<i>{esc(payload['backtest_summary'])}</i>")

        minutes = max(0, request.seconds_remaining(now) // 60)
        lines += [
            "",
            f"<b>{esc(t.t('approval.expires_in'))}:</b> {minutes} {esc(t.t('approval.minutes'))}",
            f"<i>{esc(t.t('approval.timeout_note'))}</i>",
            f"<i>{esc(t.t('approval.recheck_note'))}</i>",
        ]
        return "\n".join(lines)

    return f"{mode_tag(mode)}\n{bilingual(translator, build)}"


def format_approval_resolution(
    request: ApprovalRequest, mode: TradingMode, translator: Translator
) -> str:
    """The text an approval message is edited to once it is decided."""
    from investment_box.core.types import ApprovalStatus

    key = {
        ApprovalStatus.APPROVED: "approval.approved_by",
        ApprovalStatus.REJECTED: "approval.rejected_by",
        ApprovalStatus.EXPIRED: "approval.expired",
        ApprovalStatus.CANCELLED: "approval.cancelled",
        ApprovalStatus.AWAITING_MODIFICATION: "approval.send_new_size",
        ApprovalStatus.PENDING: "approval.title",
    }[request.status]

    def build(t: Translator) -> str:
        icon = {
            ApprovalStatus.APPROVED: "✅",
            ApprovalStatus.REJECTED: "❌",
            ApprovalStatus.EXPIRED: "⏱",
            ApprovalStatus.CANCELLED: "🚫",
        }.get(request.status, "•")
        header = f"{icon} <b>{esc(t.t(key))}</b>"
        detail = []
        if request.symbol:
            detail.append(esc(request.symbol))
        if request.responder_id and request.status.is_terminal:
            detail.append(esc(request.responder_id))
        if request.response_note:
            detail.append(esc(request.response_note))
        return header + (f"\n{' · '.join(detail)}" if detail else "")

    return f"{mode_tag(mode)}\n{bilingual(translator, build)}"


def format_pending(
    requests: Sequence[ApprovalRequest],
    mode: TradingMode,
    translator: Translator,
    *,
    now: dt.datetime,
) -> str:
    """The ``/pending`` reply."""

    def build(t: Translator) -> str:
        if not requests:
            return esc(t.t("telegram.no_pending"))
        lines = [f"<b>{esc(t.t('approval.title'))}</b> ({len(requests)})"]
        for request in requests:
            minutes = max(0, request.seconds_remaining(now) // 60)
            label = request.symbol or t.t(_KIND_KEYS[request.kind])
            lines.append(
                f"#{request.id} <b>{esc(label)}</b> — "
                f"{esc(t.t('approval.expires_in'))} {minutes} {esc(t.t('approval.minutes'))}"
            )
        return "\n".join(lines)

    return f"{mode_tag(mode)}\n{bilingual(translator, build)}"


def approval_buttons(approval_id: int, translator: Translator) -> list[list[Any]]:
    """The four inline buttons, labelled in the configured language(s)."""
    from investment_box.telegram.transport import InlineButton

    def label(key: str) -> str:
        if translator.language == "both":
            return f"{Translator('en').t(key)} / {Translator('ar').t(key)}"
        return translator.t(key)

    return [
        [
            InlineButton(f"✅ {label('approval.approve')}", f"apv:{approval_id}:approve"),
            InlineButton(f"❌ {label('approval.reject')}", f"apv:{approval_id}:reject"),
        ],
        [
            InlineButton(f"✏️ {label('approval.modify')}", f"apv:{approval_id}:modify"),
            InlineButton(f"⏸ {label('approval.snooze')}", f"apv:{approval_id}:snooze"),
        ],
    ]


# --------------------------------------------------------------------- helpers


def _warning_block(warnings: Sequence[str]) -> str:
    if not warnings:
        return ""
    body = "\n".join(f"⚠️ {esc(warning)}" for warning in warnings)
    return f"\n\n{body}"
