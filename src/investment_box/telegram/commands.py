"""Read-only command handlers.

Every handler goes through the service layer -- the same one the dashboard
uses -- and none of them touch the broker or the database directly. That is
what guarantees ``/balance`` and the dashboard's equity card can never
disagree.

Handlers are written against plain arguments rather than
``python-telegram-bot`` update objects, so they are testable without
constructing a fake Update. :mod:`investment_box.telegram.bot` does the
adapting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from investment_box.config.loader import load_universe_file
from investment_box.config.schema import Settings
from investment_box.core.clock import Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.i18n.translator import Translator
from investment_box.services.approvals import ApprovalService
from investment_box.services.portfolio import PortfolioService
from investment_box.services.research import ResearchService, ResearchSnapshot
from investment_box.telegram import formatting
from investment_box.telegram.queue import MessageQueue

log = get_logger(__name__)

DEFAULT_HISTORY_LIMIT = 10
MAX_HISTORY_LIMIT = 50


@dataclass
class CommandContext:
    """Everything the handlers need, injected rather than imported."""

    settings: Settings
    portfolio: PortfolioService
    approvals: ApprovalService
    queue: MessageQueue
    translator: Translator
    clock: Clock = field(default_factory=SystemClock)
    engine_state: str = "idle"
    data_provider_name: str = "unknown"
    #: Supplies live candidate rankings. When absent, /funds falls back to the
    #: configured list without scores -- never to placeholder numbers, which
    #: would be read as signals.
    research: ResearchService | None = None


class CommandHandlers:
    """The read-only commands available in Phase 2."""

    def __init__(self, context: CommandContext) -> None:
        self.ctx = context

    @property
    def tz(self) -> Any:
        return self.ctx.settings.i18n.tzinfo

    # ------------------------------------------------------------- commands

    def status(self) -> str:
        view = self.ctx.portfolio.get_account_view()
        return formatting.format_status(
            view=view,
            engine_state=self.ctx.engine_state,
            autonomy_level=self.ctx.settings.engine.autonomy_level.value,
            data_provider=self.ctx.data_provider_name,
            queue_stats=self.ctx.queue.stats(),
            translator=self.ctx.translator,
        )

    def balance(self) -> str:
        view = self.ctx.portfolio.get_account_view()
        return formatting.format_balance(view, self.ctx.translator, self.tz)

    def positions(self) -> str:
        view = self.ctx.portfolio.get_account_view()
        return formatting.format_positions(view, self.ctx.translator)

    def funds(self) -> str:
        """The universe with live rankings, compliance status and forecasts.

        Goes through the same :class:`ResearchService` the dashboard uses, so
        the two can never disagree. Without a research service it degrades to
        the configured list with no scores -- never to placeholder numbers,
        which would be read as signals.
        """
        try:
            universe = load_universe_file()
        except Exception as exc:  # noqa: BLE001 - a bad config file must not kill the bot
            log.error("commands.universe_load_failed", error=str(exc))
            tag = formatting.mode_tag(self.ctx.settings.trading_mode)
            return f"{tag}\nCould not read the universe file."

        configured = [
            {
                "symbol": str(entry["symbol"]).upper(),
                "name": entry.get("name"),
                "verified": bool(entry.get("verified")),
                "compliance_status": None,
            }
            for entry in universe.get("etfs", [])
        ]

        if self.ctx.research is None:
            return formatting.format_funds(
                configured, self.ctx.settings.trading_mode, self.ctx.translator
            )

        try:
            snapshot = self._research_snapshot(universe)
        except Exception as exc:  # noqa: BLE001 - fall back rather than fail the command
            log.error("commands.research_failed", error=str(exc))
            return formatting.format_funds(
                configured, self.ctx.settings.trading_mode, self.ctx.translator
            )

        by_symbol = {c.symbol: c for c in snapshot.candidates}
        funds = []
        for entry in configured:
            candidate = by_symbol.get(entry["symbol"])
            if candidate is None:
                funds.append(entry)
                continue
            compliance = candidate.compliance
            funds.append(
                {
                    **entry,
                    "compliance_status": (
                        compliance.display_status.value if compliance else None
                    ),
                    "rank": candidate.rank,
                    "candidate_status": candidate.status.value,
                    "probability": candidate.probability,
                    "confidence": candidate.confidence.value,
                    "reason": candidate.reason,
                }
            )
        # Preserve the ranking order the service produced.
        funds.sort(key=lambda f: f.get("rank") or 999)

        return formatting.format_funds(
            funds,
            self.ctx.settings.trading_mode,
            self.ctx.translator,
            warnings=snapshot.warnings,
            regime=snapshot.regime.regime.value if snapshot.regime else None,
        )

    def _research_snapshot(self, universe: dict[str, Any]) -> ResearchSnapshot:
        from investment_box.universe.builder import UniverseBuilder

        assert self.ctx.research is not None
        instruments = UniverseBuilder.load_instruments(universe)
        return self.ctx.research.build(instruments)

    def history(self, limit_arg: str | None = None) -> str:
        limit = self._parse_limit(limit_arg)
        trades = self.ctx.portfolio.closed_trades(limit)
        return formatting.format_history(
            trades, self.ctx.settings.trading_mode, self.ctx.translator
        )

    def pending(self) -> str:
        requests = self.ctx.approvals.pending()
        return formatting.format_pending(
            requests,
            self.ctx.settings.trading_mode,
            self.ctx.translator,
            now=self.ctx.clock.now(),
        )

    def help(self) -> str:
        return formatting.format_help(self.ctx.translator, self.ctx.settings.trading_mode)

    def unknown(self) -> str:
        tag = formatting.mode_tag(self.ctx.settings.trading_mode)
        return f"{tag}\n{self.ctx.translator.t('telegram.unknown_command')}"

    # -------------------------------------------------------------- internals

    @staticmethod
    def _parse_limit(raw: str | None) -> int:
        """Parse ``/history 25``, clamped and tolerant of junk.

        A bad argument falls back to the default rather than erroring: the user
        asked for their history, and refusing over a typo is unhelpful.
        """
        if not raw:
            return DEFAULT_HISTORY_LIMIT
        token = raw.strip().split()[0] if raw.strip() else ""
        if not token.isdigit():
            return DEFAULT_HISTORY_LIMIT
        return max(1, min(MAX_HISTORY_LIMIT, int(token)))

    def dispatch(self, command: str, argument: str | None = None) -> str:
        """Route a command name to its handler. Unknown commands get the help hint."""
        name = command.lstrip("/").split("@")[0].lower()
        match name:
            case "status":
                return self.status()
            case "balance":
                return self.balance()
            case "positions":
                return self.positions()
            case "funds":
                return self.funds()
            case "history":
                return self.history(argument)
            case "pending":
                return self.pending()
            case "help" | "start":
                return self.help()
        return self.unknown()
