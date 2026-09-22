"""User-editable settings that outlive a restart.

Config files hold defaults. This holds *your* decisions -- which symbols are
allowed, how much autonomy the engine has, how much capital it may use -- so
they survive a restart and are auditable.

Two rules the whole design rests on:

* **The hard Shariah constraints are not here, and cannot be.** There is no key
  for margin, shorting, derivatives, leveraged funds or crypto. They live as
  frozen constants in ``shariah/constraints.py``. A setting that can be toggled
  will eventually be toggled by accident.
* **Every change is audited** with who, when, and from what to what. "Why did
  it start trading X?" must be answerable months later.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any

from sqlalchemy import select

from investment_box.config.schema import Settings
from investment_box.core.audit import AuditSink
from investment_box.core.logging import get_logger
from investment_box.core.types import AutonomyLevel, UniverseMode
from investment_box.db.models import SettingOverride
from investment_box.db.session import Database

log = get_logger(__name__)


class SymbolRule(StrEnum):
    """What you have decided about one symbol."""

    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    NEEDS_APPROVAL = "needs_approval"

    @property
    def auto_tradable(self) -> bool:
        return self is SymbolRule.ALLOWED


#: Keys this service will accept. An unknown key is rejected rather than
#: stored, so a typo cannot silently create a setting nothing reads.
_KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "autonomy_level",
        "capital_allocation_usd",
        "universe_mode",
        "engine_enabled",
        "max_trades_per_day",
        "max_trades_per_week",
        "min_holding_days",
        "non_compliant_exit_days",
        "symbol_rules",
        "excluded_sectors",
        "max_price",
        "min_avg_dollar_volume",
        "kill_requested",
        "kill_reason",
    }
)

#: Settings that exist only in config and must never be overridable here.
#: Attempting one raises rather than being ignored, so the refusal is visible.
_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "trading_mode",  # live requires a typed confirmation, not a stored value
        "margin_allowed",
        "short_selling_allowed",
        "derivatives_allowed",
        "leveraged_or_inverse_allowed",
        "crypto_allowed",
        "block_unsettled_usage",
    }
)


@dataclass(frozen=True, slots=True)
class SettingChange:
    """One recorded change, for the audit view."""

    key: str
    previous: Any
    current: Any
    changed_at: dt.datetime
    changed_by: str


class SettingsService:
    """Reads and writes the user's persisted decisions."""

    def __init__(
        self, database: Database, settings: Settings, audit: AuditSink
    ) -> None:
        self.db = database
        self.settings = settings
        self.audit = audit

    # ------------------------------------------------------------- primitives

    def get(self, key: str, default: Any = None) -> Any:
        with self.db.session() as session:
            row = session.scalar(
                select(SettingOverride).where(SettingOverride.key == key)
            )
            if row is None:
                return default
            try:
                return json.loads(row.value_json)
            except json.JSONDecodeError:
                log.warning("settings.corrupt_value", key=key)
                return default

    def set(self, key: str, value: Any, *, actor: str = "ui") -> None:
        """Persist a setting.

        Raises:
            ValueError: For an unknown key, or one that must never be
                overridable. Refusing loudly beats storing something nothing
                reads, or something that looks like it relaxes a hard rule.
        """
        if key in _FORBIDDEN_KEYS:
            raise ValueError(
                f"'{key}' cannot be set here. Hard constraints live in "
                f"shariah/constraints.py, and switching to live trading requires a "
                f"typed confirmation in the dashboard."
            )
        if key not in _KNOWN_KEYS:
            raise ValueError(f"unknown setting '{key}'. Known: {sorted(_KNOWN_KEYS)}")

        previous = self.get(key)
        payload = json.dumps(value, default=str)

        with self.db.session() as session:
            row = session.scalar(
                select(SettingOverride).where(SettingOverride.key == key)
            )
            if row is None:
                session.add(
                    SettingOverride(key=key, value_json=payload, updated_by=actor)
                )
            else:
                row.value_json = payload
                row.updated_by = actor

        log.info("settings.changed", key=key, actor=actor)
        self.audit.record(
            "settings.changed",
            f"{key}: {previous!r} -> {value!r}",
            actor=actor,
            detail={"key": key, "previous": previous, "current": value},
        )

    def all(self) -> dict[str, Any]:
        with self.db.session() as session:
            rows = list(session.scalars(select(SettingOverride)).all())
            for row in rows:
                session.expunge(row)
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row.key] = json.loads(row.value_json)
            except json.JSONDecodeError:
                continue
        return out

    def history(self, limit: int = 50) -> list[SettingChange]:
        """Recent changes, newest first."""
        entries = self.audit.recent(limit=limit, event_type="settings.changed")  # type: ignore[attr-defined]
        out: list[SettingChange] = []
        for entry in entries:
            detail = json.loads(entry.detail_json) if entry.detail_json else {}
            out.append(
                SettingChange(
                    key=detail.get("key", "?"),
                    previous=detail.get("previous"),
                    current=detail.get("current"),
                    changed_at=entry.created_at,
                    changed_by=entry.actor,
                )
            )
        return out

    # ----------------------------------------------------------- symbol rules

    def symbol_rules(self) -> dict[str, SymbolRule]:
        raw = self.get("symbol_rules", {}) or {}
        out: dict[str, SymbolRule] = {}
        for symbol, value in raw.items():
            try:
                out[symbol.upper()] = SymbolRule(value)
            except ValueError:
                log.warning("settings.unknown_symbol_rule", symbol=symbol, value=value)
        return out

    def rule_for(self, symbol: str) -> SymbolRule:
        """Your decision about one symbol.

        The default is ``NEEDS_APPROVAL``, not ``ALLOWED``. A symbol you have
        never ruled on is not one you have approved.
        """
        return self.symbol_rules().get(symbol.upper(), SymbolRule.NEEDS_APPROVAL)

    def set_symbol_rule(self, symbol: str, rule: SymbolRule, *, actor: str = "ui") -> None:
        rules = {k: v.value for k, v in self.symbol_rules().items()}
        rules[symbol.upper()] = rule.value
        self.set("symbol_rules", rules, actor=actor)

    def clear_symbol_rule(self, symbol: str, *, actor: str = "ui") -> None:
        rules = {k: v.value for k, v in self.symbol_rules().items()}
        rules.pop(symbol.upper(), None)
        self.set("symbol_rules", rules, actor=actor)

    def allowed_symbols(self) -> list[str]:
        return sorted(s for s, r in self.symbol_rules().items() if r is SymbolRule.ALLOWED)

    def forbidden_symbols(self) -> list[str]:
        return sorted(s for s, r in self.symbol_rules().items() if r is SymbolRule.FORBIDDEN)

    # -------------------------------------------------------- effective values

    @property
    def autonomy_level(self) -> AutonomyLevel:
        raw = self.get("autonomy_level")
        if raw is None:
            return self.settings.engine.autonomy_level
        try:
            return AutonomyLevel(str(raw))
        except ValueError:
            return self.settings.engine.autonomy_level

    def set_autonomy_level(self, level: AutonomyLevel, *, actor: str = "ui") -> None:
        self.set("autonomy_level", level.value, actor=actor)

    @property
    def capital_allocation(self) -> Decimal:
        raw = self.get("capital_allocation_usd")
        if raw is None:
            return self.settings.capital.allocation_usd
        try:
            value = Decimal(str(raw))
        except (InvalidOperation, ValueError):
            return self.settings.capital.allocation_usd
        return value if value > 0 else self.settings.capital.allocation_usd

    def set_capital_allocation(
        self, amount: Decimal, *, account_equity: Decimal | None = None, actor: str = "ui"
    ) -> None:
        """Set how much the bot may use.

        Raises:
            ValueError: If the amount is not positive, or exceeds the account.
                Allocating more than the account holds would make every
                percentage limit meaningless.
        """
        if amount <= 0:
            raise ValueError("allocated capital must be positive")
        if account_equity is not None and amount > account_equity:
            raise ValueError(
                f"cannot allocate ${amount} of an account worth ${account_equity}"
            )
        self.set("capital_allocation_usd", str(amount), actor=actor)

    @property
    def universe_mode(self) -> UniverseMode:
        raw = self.get("universe_mode")
        if raw is None:
            return self.settings.universe.mode
        try:
            return UniverseMode(str(raw))
        except ValueError:
            return self.settings.universe.mode

    def set_universe_mode(self, mode: UniverseMode, *, actor: str = "ui") -> None:
        self.set("universe_mode", mode.value, actor=actor)

    @property
    def engine_enabled(self) -> bool:
        raw = self.get("engine_enabled")
        return self.settings.engine.enabled if raw is None else bool(raw)

    def set_engine_enabled(self, enabled: bool, *, actor: str = "ui") -> None:
        self.set("engine_enabled", enabled, actor=actor)

    # ------------------------------------------------------------ kill flag

    @property
    def kill_requested(self) -> bool:
        """Whether a kill has been requested from anywhere.

        Persisted because the dashboard and the engine are separate processes.
        A kill pulled in one must stop the other, and must survive a restart --
        otherwise restarting the engine would quietly undo the kill.
        """
        return bool(self.get("kill_requested", False))

    @property
    def kill_reason(self) -> str:
        return str(self.get("kill_reason", "") or "")

    def request_kill(self, reason: str, *, actor: str = "ui") -> None:
        self.set("kill_requested", True, actor=actor)
        self.set("kill_reason", reason, actor=actor)

    def clear_kill(self, *, actor: str = "user") -> None:
        """Clear the flag. Deliberately manual: restarting must not undo a kill."""
        self.set("kill_requested", False, actor=actor)
        self.set("kill_reason", "", actor=actor)

    def effective_blacklist(self) -> list[str]:
        """Config blacklist plus anything you marked FORBIDDEN."""
        return sorted(set(self.settings.universe.blacklist) | set(self.forbidden_symbols()))

    def effective_whitelist(self) -> list[str]:
        """Config whitelist plus anything you marked ALLOWED."""
        return sorted(set(self.settings.universe.whitelist) | set(self.allowed_symbols()))
