"""The live-trading pre-flight checklist.

Live trading is not a setting. It is a state you may only enter by producing
evidence, and this module is what asks for the evidence.

Design decisions that matter more than the individual checks:

* **Every check must pass. There is no override.** A checklist with a bypass is
  a suggestion. If a check is wrong, fix the check in source and explain why in
  the commit -- do not add a flag.
* **Checks are evidence-based, not acknowledgement-based.** "Have you paper
  traded for eight weeks?" is a question you can lie to. "The database contains
  eight weeks of paper trades" is not.
* **It is re-evaluated continuously**, not once at activation. Passing in
  January does not mean passing in March, so the engine re-runs it on every
  start and the order manager re-runs the critical subset before every live
  order.
* **A check that cannot be evaluated FAILS.** Missing evidence is not passing
  evidence. This is the same rule the compliance layer uses, for the same
  reason.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import func, select

from investment_box.config.schema import Secrets, Settings
from investment_box.core.clock import UTC, Clock, SystemClock
from investment_box.core.logging import get_logger
from investment_box.core.types import ComplianceStatus, TradingMode
from investment_box.db.models import EquitySnapshot, Trade
from investment_box.db.session import Database

log = get_logger(__name__)

#: Minimum paper-trading history before live is even considered. Eight weeks
#: is not long enough to prove a strategy works -- nothing is -- but it is long
#: enough to surface the operational failures that matter: a broker outage, a
#: settlement surprise, a stop that did not fire.
MIN_PAPER_WEEKS = 8
#: Minimum closed paper trades. Below this the paper record says nothing about
#: whether execution behaves.
MIN_PAPER_TRADES = 20
#: How far paper results may diverge from the backtest before the gap itself is
#: the finding. A large divergence means one of the two has a bug.
MAX_BACKTEST_DIVERGENCE = 0.50
#: Days within which every held symbol must have been screened.
MAX_SCREEN_AGE_DAYS = 7


class CheckStatus(StrEnum):
    PASS = "pass"  # noqa: S105 - a check verdict, not a credential
    FAIL = "fail"
    #: Could not be evaluated. Treated as a failure -- missing evidence is not
    #: passing evidence.
    UNKNOWN = "unknown"

    @property
    def permits_live(self) -> bool:
        return self is CheckStatus.PASS


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One check, its verdict, and what to do about it."""

    name: str
    status: CheckStatus
    detail: str
    #: What the user must do to make it pass. Empty when it already passes.
    remedy: str = ""
    #: Critical checks are re-run before every live order, not only at startup.
    critical: bool = False

    @property
    def passed(self) -> bool:
        return self.status.permits_live

    def describe(self) -> str:
        icon = {"pass": "PASS", "fail": "FAIL", "unknown": "????"}[self.status.value]
        line = f"[{icon}] {self.name}: {self.detail}"
        return line + (f"\n         -> {self.remedy}" if self.remedy else "")


@dataclass
class PreflightReport:
    """The whole checklist."""

    evaluated_at: dt.datetime
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Every check must pass. No exceptions, no overrides."""
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.failures if c.critical]

    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        if self.passed:
            return f"All {len(self.checks)} pre-flight checks pass."
        return (
            f"{passed}/{len(self.checks)} checks pass. "
            f"Live trading is refused until all of them do."
        )

    def to_text(self) -> str:
        lines = [
            "=" * 74,
            "LIVE TRADING PRE-FLIGHT CHECKLIST",
            "=" * 74,
            f"Evaluated: {self.evaluated_at:%Y-%m-%d %H:%M %Z}",
            "",
        ]
        lines += [check.describe() for check in self.checks]
        lines += ["", "-" * 74, self.summary()]
        if not self.passed:
            lines += [
                "",
                "There is no override. If a check is wrong, fix the check in source",
                "and say why -- do not add a flag to skip it.",
            ]
        return "\n".join(lines)


class PreflightChecklist:
    """Evaluates whether live trading may be enabled."""

    def __init__(
        self,
        settings: Settings,
        secrets: Secrets,
        database: Database,
        *,
        clock: Clock | None = None,
    ) -> None:
        self.settings = settings
        self.secrets = secrets
        self.db = database
        self.clock = clock or SystemClock()

    def evaluate(
        self,
        *,
        broker: object | None = None,
        data_provider_name: str | None = None,
        compliance_provider_name: str | None = None,
        held_symbols: list[str] | None = None,
        backtest_return: float | None = None,
    ) -> PreflightReport:
        """Run every check."""
        report = PreflightReport(evaluated_at=self.clock.now())
        report.checks = [
            self._paper_duration(),
            self._paper_trade_count(),
            self._paper_vs_backtest(backtest_return),
            self._no_synthetic_data(data_provider_name),
            self._certified_compliance_provider(compliance_provider_name),
            self._universe_verified(),
            self._holdings_screened(held_symbols or []),
            self._cash_account(broker),
            self._kill_flag_clear(),
            self._alerting_configured(),
            self._risk_limits_sane(),
            self._credentials_match_mode(),
        ]
        if not report.passed:
            log.info(
                "preflight.failed",
                failures=[c.name for c in report.failures],
            )
        return report

    def critical_only(self, **kwargs: object) -> PreflightReport:
        """The subset re-run before every live order.

        Deliberately narrow: re-running the full checklist on every order would
        be slow enough that someone would eventually cache it, and a cached
        safety check is not a safety check.
        """
        full = self.evaluate(**kwargs)  # type: ignore[arg-type]
        report = PreflightReport(evaluated_at=full.evaluated_at)
        report.checks = [c for c in full.checks if c.critical]
        return report

    # ---------------------------------------------------------------- checks

    def _paper_duration(self) -> CheckResult:
        name = f"At least {MIN_PAPER_WEEKS} weeks of paper trading"
        with self.db.session() as session:
            first = session.scalar(
                select(func.min(EquitySnapshot.snapshot_date)).where(
                    EquitySnapshot.trading_mode == TradingMode.PAPER.value
                )
            )
        if first is None:
            return CheckResult(
                name, CheckStatus.FAIL, "no paper-trading history at all",
                remedy="Run the engine on paper and let it record daily snapshots.",
            )

        today = self.clock.now().astimezone(UTC).date()
        weeks = (today - first).days / 7
        if weeks >= MIN_PAPER_WEEKS:
            return CheckResult(
                name, CheckStatus.PASS, f"{weeks:.1f} weeks since {first}"
            )
        return CheckResult(
            name, CheckStatus.FAIL,
            f"only {weeks:.1f} weeks since {first}",
            remedy=f"Keep paper trading for another {MIN_PAPER_WEEKS - weeks:.1f} weeks.",
        )

    def _paper_trade_count(self) -> CheckResult:
        name = f"At least {MIN_PAPER_TRADES} closed paper trades"
        count = self._closed_paper_trades()
        if count >= MIN_PAPER_TRADES:
            return CheckResult(name, CheckStatus.PASS, f"{count} closed trades")
        return CheckResult(
            name, CheckStatus.FAIL, f"only {count} closed trades",
            remedy=(
                "Below this the paper record says nothing about whether execution "
                "behaves -- fills, stops, settlement."
            ),
        )

    def _paper_vs_backtest(self, backtest_return: float | None) -> CheckResult:
        name = "Paper results within tolerance of the backtest"
        if backtest_return is None:
            return CheckResult(
                name, CheckStatus.UNKNOWN,
                "no backtest return supplied for comparison",
                remedy=(
                    "Run scripts/run_backtest.py over the paper period and pass its "
                    "return in. A divergence is the point of this check."
                ),
            )

        paper = self._paper_return()
        if paper is None:
            return CheckResult(
                name, CheckStatus.UNKNOWN, "no paper equity curve to compare",
                remedy="Let the engine record daily equity snapshots.",
            )

        divergence = abs(paper - backtest_return)
        detail = f"paper {paper:+.1%} vs backtest {backtest_return:+.1%}"
        if divergence <= MAX_BACKTEST_DIVERGENCE:
            return CheckResult(name, CheckStatus.PASS, detail)
        return CheckResult(
            name, CheckStatus.FAIL,
            f"{detail} -- diverged by {divergence:.1%}",
            remedy=(
                "A large gap means one of the two has a bug. Find out which before "
                "risking real money."
            ),
        )

    def _no_synthetic_data(self, provider: str | None) -> CheckResult:
        name = "Market data is real"
        if provider is None:
            return CheckResult(
                name, CheckStatus.UNKNOWN, "data provider unknown",
                remedy="Pass the provider name in.", critical=True,
            )
        if provider == "synthetic":
            return CheckResult(
                name, CheckStatus.FAIL, "the synthetic generator is in use",
                remedy="Configure a real data provider.", critical=True,
            )
        return CheckResult(name, CheckStatus.PASS, f"using {provider}", critical=True)

    def _certified_compliance_provider(self, provider: str | None) -> CheckResult:
        name = "Compliance screening comes from a certified source"
        if provider is None:
            return CheckResult(
                name, CheckStatus.UNKNOWN, "screening provider unknown",
                remedy="Pass the provider name in.", critical=True,
            )
        if provider in ("mock_external", "internal_aaoifi"):
            return CheckResult(
                name, CheckStatus.FAIL,
                f"'{provider}' is not a certified source",
                remedy=(
                    "A mock is not a ruling, and the internal screener is an estimate "
                    "with no business-activity data. Configure a certified provider "
                    "before trading real money on its verdicts."
                ),
                critical=True,
            )
        return CheckResult(name, CheckStatus.PASS, f"using {provider}", critical=True)

    def _universe_verified(self) -> CheckResult:
        name = "Every tradable symbol is verified"
        from investment_box.config.loader import load_universe_file
        from investment_box.universe.builder import UniverseBuilder

        try:
            instruments = UniverseBuilder.load_instruments(load_universe_file())
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name, CheckStatus.UNKNOWN, f"could not read the universe: {exc}",
                remedy="Fix config/universe_etf.yaml.", critical=True,
            )

        unverified = [i.symbol for i in instruments if not i.verified]
        if not unverified:
            return CheckResult(
                name, CheckStatus.PASS, f"all {len(instruments)} verified", critical=True,
            )
        return CheckResult(
            name, CheckStatus.FAIL,
            f"{len(unverified)} unverified: {', '.join(unverified)}",
            remedy=(
                "Confirm listing, Shariah certification and the certifying board from "
                "each fund's own documents, then set verified: true."
            ),
            critical=True,
        )

    def _holdings_screened(self, held: list[str]) -> CheckResult:
        name = "Every held symbol has a fresh compliance screen"
        if not held:
            return CheckResult(name, CheckStatus.PASS, "no open positions")

        from investment_box.db.models import ComplianceScreen

        cutoff = self.clock.now() - dt.timedelta(days=MAX_SCREEN_AGE_DAYS)
        stale: list[str] = []
        with self.db.session() as session:
            for symbol in held:
                row = session.scalar(
                    select(ComplianceScreen)
                    .where(ComplianceScreen.symbol == symbol.upper())
                    .order_by(ComplianceScreen.screened_at.desc())
                    .limit(1)
                )
                if row is None or row.screened_at < cutoff:
                    stale.append(symbol.upper())
                elif ComplianceStatus(row.status) is not ComplianceStatus.COMPLIANT:
                    stale.append(f"{symbol.upper()} ({row.status})")

        if not stale:
            return CheckResult(
                name, CheckStatus.PASS, f"{len(held)} holding(s) screened", critical=True
            )
        return CheckResult(
            name, CheckStatus.FAIL,
            f"unscreened, stale or non-compliant: {', '.join(stale)}",
            remedy="Re-screen before going live.", critical=True,
        )

    def _cash_account(self, broker: object | None) -> CheckResult:
        name = "Broker account is a cash account"
        if broker is None:
            return CheckResult(
                name, CheckStatus.UNKNOWN, "no broker supplied",
                remedy="Pass the broker in.", critical=True,
            )
        try:
            account = broker.get_account()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            return CheckResult(
                name, CheckStatus.UNKNOWN, f"broker unreachable: {exc}",
                remedy="Fix broker connectivity.", critical=True,
            )

        if not account.is_cash_account:
            return CheckResult(
                name, CheckStatus.FAIL, "the account has margin enabled",
                remedy="Margin is never permitted. Use a cash account.", critical=True,
            )
        if account.trading_blocked:
            return CheckResult(
                name, CheckStatus.FAIL, "the broker reports trading is blocked",
                remedy="Resolve it with the broker.", critical=True,
            )
        return CheckResult(name, CheckStatus.PASS, "cash account confirmed", critical=True)

    def _kill_flag_clear(self) -> CheckResult:
        name = "No kill switch is active"
        from investment_box.services.audit import AuditService
        from investment_box.services.settings_service import SettingsService

        service = SettingsService(
            self.db, self.settings, AuditService(self.db, self.settings.trading_mode)
        )
        if service.kill_requested:
            return CheckResult(
                name, CheckStatus.FAIL, f"kill active: {service.kill_reason}",
                remedy="Clear it in the dashboard if the cause is resolved.",
                critical=True,
            )
        return CheckResult(name, CheckStatus.PASS, "clear", critical=True)

    def _alerting_configured(self) -> CheckResult:
        name = "Alerting is configured"
        if not self.secrets.telegram_bot_token:
            return CheckResult(
                name, CheckStatus.FAIL, "no Telegram bot token",
                remedy=(
                    "Real money without alerts means you find out about a problem "
                    "when you next happen to look."
                ),
            )
        if not self.secrets.telegram_channel_id:
            return CheckResult(
                name, CheckStatus.FAIL, "no broadcast channel configured",
                remedy="Set TELEGRAM_CHANNEL_ID.",
            )
        problem = self.secrets.telegram_channel_problem
        if problem:
            return CheckResult(name, CheckStatus.FAIL, problem, remedy="Correct the id.")
        if not self.secrets.allowed_telegram_ids:
            return CheckResult(
                name, CheckStatus.FAIL, "no whitelisted Telegram users",
                remedy="Set TELEGRAM_ALLOWED_USER_IDS so you can reach the bot.",
            )
        return CheckResult(name, CheckStatus.PASS, "bot, channel and whitelist set")

    def _risk_limits_sane(self) -> CheckResult:
        name = "Risk limits are within sane bounds"
        risk = self.settings.risk
        problems: list[str] = []
        if risk.risk_per_trade_pct > 0.05:
            problems.append(f"risk per trade {risk.risk_per_trade_pct:.0%} exceeds 5%")
        if risk.max_drawdown_pct > 0.35:
            problems.append(f"drawdown limit {risk.max_drawdown_pct:.0%} exceeds 35%")
        if risk.max_position_pct > 0.50:
            problems.append(f"max position {risk.max_position_pct:.0%} exceeds 50%")
        if self.settings.capital.cash_buffer_pct <= 0:
            problems.append("no cash buffer")

        if problems:
            return CheckResult(
                name, CheckStatus.FAIL, "; ".join(problems),
                remedy="These bounds are not opinions about strategy; they bound blast radius.",
            )
        return CheckResult(
            name, CheckStatus.PASS,
            f"{risk.risk_per_trade_pct:.1%}/trade, {risk.max_drawdown_pct:.0%} max drawdown",
        )

    def _credentials_match_mode(self) -> CheckResult:
        name = "Broker endpoint matches the intended mode"
        if not self.secrets.has_alpaca_credentials:
            return CheckResult(
                name, CheckStatus.FAIL, "no broker credentials",
                remedy="Set ALPACA_API_KEY and ALPACA_SECRET_KEY.", critical=True,
            )
        if not self.secrets.is_live_alpaca_url:
            return CheckResult(
                name, CheckStatus.FAIL,
                "ALPACA_BASE_URL still points at the paper endpoint",
                remedy=(
                    "Going live requires the live endpoint. Change it deliberately, "
                    "and understand that it is the last reversible step."
                ),
                critical=True,
            )
        return CheckResult(name, CheckStatus.PASS, "live endpoint configured", critical=True)

    # -------------------------------------------------------------- internals

    def _closed_paper_trades(self) -> int:
        with self.db.session() as session:
            count = session.scalar(
                select(func.count(Trade.id)).where(
                    Trade.exit_at.is_not(None),
                    Trade.trading_mode == TradingMode.PAPER.value,
                )
            )
        return int(count or 0)

    def _paper_return(self) -> float | None:
        with self.db.session() as session:
            rows = list(
                session.scalars(
                    select(EquitySnapshot.equity)
                    .where(EquitySnapshot.trading_mode == TradingMode.PAPER.value)
                    .order_by(EquitySnapshot.snapshot_date)
                ).all()
            )
        if len(rows) < 2 or rows[0] == 0:
            return None
        return float(Decimal(str(rows[-1])) / Decimal(str(rows[0])) - 1)
