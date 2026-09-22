"""Typed configuration models.

Validation lives here rather than at the point of use, so an inconsistent
config fails at startup with a clear message instead of halfway through a
trading session. Cross-field invariants (e.g. five positions at 20% each must
not exceed 100% of capital) are checked explicitly.

Secrets are deliberately kept in a separate model (:class:`Secrets`) that is
never serialised alongside the rest of the config.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from investment_box.core.types import (
    AutonomyLevel,
    ComplianceStatus,
    OrderType,
    TradingMode,
    UniverseMode,
)

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CapitalConfig(_Frozen):
    """How much of the account the bot is permitted to touch."""

    allocation_usd: Decimal = Field(default=Decimal("500.00"), gt=0)
    cash_buffer_pct: Fraction = 0.10

    @property
    def deployable_usd(self) -> Decimal:
        """Allocation net of the permanently-held cash buffer."""
        return self.allocation_usd * (Decimal("1") - Decimal(str(self.cash_buffer_pct)))


class RiskConfig(_Frozen):
    risk_per_trade_pct: Fraction = 0.015
    max_position_pct: Fraction = 0.20
    max_open_positions: int = Field(default=5, ge=1, le=50)
    max_daily_loss_pct: Fraction = 0.03
    max_drawdown_pct: Fraction = 0.15
    max_sector_exposure_pct: Fraction = 0.50
    default_stop_loss_atr_mult: PositiveFloat = 2.0
    default_take_profit_atr_mult: PositiveFloat = 3.0
    sizing_method: Literal["atr", "fixed_fractional"] = "atr"
    max_trades_per_day: int = Field(default=2, ge=0)
    max_trades_per_week: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def _check_coherent(self) -> RiskConfig:
        if self.risk_per_trade_pct > self.max_position_pct:
            raise ValueError(
                "risk_per_trade_pct cannot exceed max_position_pct: risking more than "
                "the position is worth is not expressible"
            )
        if self.max_daily_loss_pct > self.max_drawdown_pct:
            raise ValueError(
                "max_daily_loss_pct exceeds max_drawdown_pct, so the drawdown "
                "auto-pause could never trigger before the daily halt"
            )
        if self.max_trades_per_week < self.max_trades_per_day:
            raise ValueError("max_trades_per_week must be >= max_trades_per_day")
        return self

    @property
    def theoretical_max_exposure_pct(self) -> float:
        """Worst-case fraction of capital deployable at once.

        Informational: the risk manager also enforces the cash buffer, so
        actual exposure is lower. Values above 1.0 are legal here -- the
        position-count limit and available cash bind first -- but are surfaced
        in the startup log because they mean position sizing, not the position
        cap, is what keeps you diversified.
        """
        return self.max_open_positions * self.max_position_pct


class HoldingConfig(_Frozen):
    """Holding-period limits, in NYSE trading days."""

    min_holding_days: int = Field(default=2, ge=1)
    typical_max_holding_days: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _check_order(self) -> HoldingConfig:
        if self.min_holding_days > self.typical_max_holding_days:
            raise ValueError("min_holding_days cannot exceed typical_max_holding_days")
        return self


class SettlementConfig(_Frozen):
    settlement_days: int = Field(default=1, ge=0, le=5)
    block_unsettled_usage: bool = True

    @field_validator("block_unsettled_usage")
    @classmethod
    def _must_block(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "block_unsettled_usage cannot be disabled: trading on unsettled "
                "proceeds in a cash account is a good-faith violation and is "
                "outside what this application permits."
            )
        return value


class ExecutionConfig(_Frozen):
    mode: Literal["hybrid", "whole_share_only", "fractional_always"] = "hybrid"
    allow_fractional_fallback: bool = True
    default_order_type: OrderType = OrderType.LIMIT
    limit_offset_pct: Fraction = 0.001
    order_timeout_minutes: int = Field(default=30, ge=1)
    acknowledge_fractional_stop_risk: bool = False

    #: Set by the validator when fractional was requested but then disabled for
    #: lack of an explicit risk acknowledgement. Kept so startup can explain why
    #: the setting the user wrote is not the setting in effect.
    fractional_suppressed: bool = False

    @model_validator(mode="after")
    def _check_fractional(self) -> ExecutionConfig:
        requested = self.mode == "fractional_always" or (
            self.mode == "hybrid" and self.allow_fractional_fallback
        )
        if requested and not self.acknowledge_fractional_stop_risk:
            # Not fatal -- execution degrades to whole-share-only and says so at
            # startup. Fractional orders carry no broker-side stop, so enabling
            # them is an explicit opt-in rather than a default.
            object.__setattr__(self, "allow_fractional_fallback", False)
            object.__setattr__(self, "fractional_suppressed", True)
        return self

    @property
    def fractional_enabled(self) -> bool:
        return self.allow_fractional_fallback and self.acknowledge_fractional_stop_risk


class UniverseConfig(_Frozen):
    mode: UniverseMode = UniverseMode.ETF_ONLY
    min_price: PositiveFloat = 5.0
    max_price: PositiveFloat = 10_000.0
    min_avg_dollar_volume: PositiveFloat = 250_000.0
    max_spread_pct: Fraction = 0.005
    whitelist: list[str] = Field(default_factory=list)
    blacklist: list[str] = Field(default_factory=list)

    @field_validator("whitelist", "blacklist")
    @classmethod
    def _upper(cls, value: list[str]) -> list[str]:
        return [symbol.strip().upper() for symbol in value if symbol.strip()]

    @model_validator(mode="after")
    def _check(self) -> UniverseConfig:
        if self.min_price >= self.max_price:
            raise ValueError("min_price must be below max_price")
        overlap = set(self.whitelist) & set(self.blacklist)
        if overlap:
            raise ValueError(f"symbols on both whitelist and blacklist: {sorted(overlap)}")
        return self


class ShariahConfig(_Frozen):
    """Screening thresholds.

    These are the *configurable* part of compliance. The non-negotiable part --
    no margin, no shorting, no derivatives, no leveraged or inverse funds --
    is in ``shariah/constraints.py`` and has no representation here on purpose.
    """

    provider: Literal["internal_aaoifi", "mock_external", "zoya", "musaffa"] = "internal_aaoifi"
    non_permissible_revenue_max_pct: Fraction = 0.05
    debt_to_market_cap_max_pct: Fraction = 0.30
    interest_securities_to_market_cap_max_pct: Fraction = 0.30
    ratio_denominator: Literal["market_cap", "total_assets"] = "market_cap"
    rescreen_interval_days: int = Field(default=7, ge=1)
    non_compliant_exit_days: int = Field(default=3, ge=0)
    idle_cash_park_symbol: str | None = "SPSK"

    @property
    def auto_tradable_statuses(self) -> frozenset[ComplianceStatus]:
        """Statuses the engine may act on without a human.

        Hard-coded, not configurable: DOUBTFUL and UNKNOWN always require a
        human decision.
        """
        return frozenset({ComplianceStatus.COMPLIANT})


class CostsConfig(_Frozen):
    """Frictions applied to every backtest. Never zero them out."""

    commission_per_share: float = Field(default=0.0, ge=0)
    commission_min: float = Field(default=0.0, ge=0)
    slippage_bps: float = Field(default=5.0, ge=0)
    spread_bps: float = Field(default=4.0, ge=0)
    sec_fee_bps: float = Field(default=0.278, ge=0)
    finra_taf_per_share: float = Field(default=0.000166, ge=0)

    @property
    def round_trip_bps(self) -> float:
        """Rough all-in cost of a round trip, in basis points of notional.

        Spread and slippage are paid on entry and exit; SEC fees on the sell
        only. At a ~$100 position this is the number that decides whether a
        strategy is viable at all.
        """
        return 2 * (self.slippage_bps + self.spread_bps) + self.sec_fee_bps


class DataConfig(_Frozen):
    provider: Literal["yfinance", "alpaca"] = "yfinance"
    cache_dir: Path = Path("./data/cache")
    cache_ttl_hours: int = Field(default=12, ge=0)
    history_start: str = "2015-01-01"
    bar_timeframe: Literal["1d"] = "1d"


class EngineConfig(_Frozen):
    autonomy_level: AutonomyLevel = AutonomyLevel.SUGGEST_ONLY
    market_timezone: str = "America/New_York"
    signal_time: str = "16:15"
    enabled: bool = False

    @field_validator("autonomy_level", mode="before")
    @classmethod
    def _coerce_autonomy(cls, value: object) -> object:
        # YAML parses a bare `autonomy_level: 1` as an int, which is the natural
        # thing to write. Accept it rather than making the user quote it.
        return str(value) if isinstance(value, int) else value

    @field_validator("market_timezone")
    @classmethod
    def _valid_tz(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @field_validator("signal_time")
    @classmethod
    def _valid_time(cls, value: str) -> str:
        hours, _, minutes = value.partition(":")
        if not (hours.isdigit() and minutes.isdigit()):
            raise ValueError(f"signal_time must be HH:MM, got {value!r}")
        if not (0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59):
            raise ValueError(f"signal_time out of range: {value!r}")
        return value


class I18nConfig(_Frozen):
    language: Literal["en", "ar", "both"] = "both"
    display_timezone: str = "Asia/Jerusalem"

    @field_validator("display_timezone")
    @classmethod
    def _valid_tz(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @property
    def tzinfo(self) -> ZoneInfo:
        return ZoneInfo(self.display_timezone)


class Secrets(BaseSettings):
    """Credentials, read only from the environment or ``.env``.

    Wrapped in ``SecretStr`` so that an accidental ``repr`` or model dump prints
    ``**********`` rather than the value. Kept out of :class:`Settings` so that
    dumping the config for the UI or the audit log cannot leak them.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    alpaca_api_key: SecretStr | None = None
    alpaca_secret_key: SecretStr | None = None
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    shariah_api_key: SecretStr | None = None
    shariah_api_base_url: str | None = None

    telegram_bot_token: SecretStr | None = None
    telegram_channel_id: str | None = None
    telegram_allowed_user_ids: str = ""

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def allowed_telegram_ids(self) -> frozenset[int]:
        """Parse the comma-separated whitelist, ignoring malformed entries.

        An empty set disables the interactive bot entirely -- fail closed.
        """
        ids: set[int] = set()
        for chunk in self.telegram_allowed_user_ids.split(","):
            chunk = chunk.strip()
            if chunk.lstrip("-").isdigit():
                ids.add(int(chunk))
        return frozenset(ids)

    @property
    def is_live_alpaca_url(self) -> bool:
        return "paper" not in self.alpaca_base_url.lower()


class Settings(BaseSettings):
    """The complete non-secret configuration."""

    model_config = SettingsConfigDict(
        env_prefix="IB__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Put the environment ahead of init kwargs.

        pydantic-settings defaults to init > env, but the YAML layers arrive as
        init kwargs, which would make a committed default silently beat an
        IB__* environment variable -- the opposite of the documented precedence.
        Sources are deep-merged, so a partial nested override such as
        ``IB__RISK__MAX_OPEN_POSITIONS`` replaces just that key.
        """
        return (env_settings, dotenv_settings, init_settings, file_secret_settings)

    trading_mode: TradingMode = TradingMode.PAPER
    log_level: str = "INFO"
    data_dir: Path = Path("./data")

    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    holding: HoldingConfig = Field(default_factory=HoldingConfig)
    settlement: SettlementConfig = Field(default_factory=SettlementConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    shariah: ShariahConfig = Field(default_factory=ShariahConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    i18n: I18nConfig = Field(default_factory=I18nConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "investment_box.db"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.db_path.resolve()}"

    @property
    def is_paper(self) -> bool:
        return self.trading_mode is TradingMode.PAPER

    #: Reference price used only to illustrate sizing at startup. Roughly where
    #: the main ETFs in the seed universe trade (SPUS ~$60, HLAL ~$75 as of
    #: Sept 2026); not used for any decision.
    _REFERENCE_SHARE_PRICE = Decimal("60")
    #: Reference stop distance for the same illustration.
    _REFERENCE_STOP_PCT = Decimal("0.08")

    def _sizing_warnings(self) -> list[str]:
        """Warn when risk-based sizing cannot buy a whole share.

        The binding number at a small balance is not the position *cap* but the
        position the risk budget actually buys: risk_per_trade / stop_distance.
        At $500 with a 1.5% risk budget and an 8% stop that is well under one
        share of a $50 ETF, which is the whole reason the fractional question
        exists.
        """
        out: list[str] = []
        risk_budget = self.capital.deployable_usd * Decimal(str(self.risk.risk_per_trade_pct))
        sized = risk_budget / self._REFERENCE_STOP_PCT
        cap = self.capital.deployable_usd * Decimal(str(self.risk.max_position_pct))
        effective = min(sized, cap)
        shares = effective / self._REFERENCE_SHARE_PRICE

        if shares < 1:
            out.append(
                f"Risk-based sizing yields ~${effective:.2f} per trade "
                f"({shares:.2f} shares of a ${self._REFERENCE_SHARE_PRICE} ETF). Below one "
                f"whole share, so entries will need the fractional path or be skipped."
            )
        if effective < Decimal("100"):
            out.append(
                f"At ~${effective:.2f} per position, round-trip frictions of "
                f"~{self.costs.round_trip_bps:.1f} bps (~"
                f"${effective * Decimal(str(self.costs.round_trip_bps)) / Decimal('10000'):.3f}) "
                f"are a meaningful share of any realistic edge."
            )
        return out

    def startup_warnings(self) -> list[str]:
        """Config smells worth surfacing at startup but not worth refusing to run.

        Returned rather than logged so the dashboard and the Telegram startup
        message can show the same list.
        """
        warnings: list[str] = []

        if self.trading_mode is TradingMode.LIVE:
            warnings.append(
                "TRADING MODE IS LIVE. Real money is at risk. Live mode additionally "
                "requires a typed confirmation in the dashboard before any order is sent."
            )
        if self.risk.theoretical_max_exposure_pct > 1.0:
            warnings.append(
                f"max_open_positions x max_position_pct = "
                f"{self.risk.theoretical_max_exposure_pct:.0%} of capital. Available cash "
                f"and the cash buffer will bind before the position cap does."
            )
        if self.execution.fractional_suppressed:
            warnings.append(
                "Fractional fallback was requested but is DISABLED because "
                "execution.acknowledge_fractional_stop_risk is false. Candidates that "
                "cannot afford one whole share will be skipped rather than traded."
            )

        warnings.extend(self._sizing_warnings())
        if self.universe.mode is UniverseMode.ETF_AND_SCREENED_STOCKS and (
            self.shariah.provider == "mock_external"
        ):
            warnings.append(
                "Universe Mode B is on while the screening provider is 'mock_external'. "
                "Mock screens are not real compliance decisions and must not be traded on."
            )
        return warnings
