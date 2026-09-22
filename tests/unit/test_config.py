"""Config layering, validation and the invariants that must not be configurable."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from investment_box.config.loader import deep_merge, load_settings
from investment_box.config.schema import (
    ExecutionConfig,
    RiskConfig,
    Secrets,
    SettlementConfig,
    UniverseConfig,
)
from investment_box.core.types import TradingMode


class TestDeepMerge:
    def test_nested_maps_merge_key_by_key(self) -> None:
        base = {"risk": {"a": 1, "b": 2}, "top": 1}
        override = {"risk": {"b": 3}}
        assert deep_merge(base, override) == {"risk": {"a": 1, "b": 3}, "top": 1}

    def test_lists_are_replaced_not_appended(self) -> None:
        """A local blacklist must override the default, not extend it."""
        base = {"universe": {"blacklist": ["AAA", "BBB"]}}
        override = {"universe": {"blacklist": ["CCC"]}}
        assert deep_merge(base, override)["universe"]["blacklist"] == ["CCC"]

    def test_does_not_mutate_the_base(self) -> None:
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestLayering:
    def test_defaults_load(self) -> None:
        settings = load_settings()
        assert settings.trading_mode is TradingMode.PAPER
        assert settings.capital.allocation_usd == Decimal("500.0")

    def test_local_overrides_default(self, tmp_path: Path) -> None:
        local = tmp_path / "local.yaml"
        local.write_text(yaml.safe_dump({"risk": {"max_open_positions": 2}}))
        settings = load_settings(local_path=local)
        assert settings.risk.max_open_positions == 2
        # An untouched sibling keeps its default.
        assert settings.risk.risk_per_trade_pct == 0.015

    def test_environment_beats_yaml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        local = tmp_path / "local.yaml"
        local.write_text(yaml.safe_dump({"risk": {"max_open_positions": 2}}))
        monkeypatch.setenv("IB__RISK__MAX_OPEN_POSITIONS", "7")
        settings = load_settings(local_path=local)
        assert settings.risk.max_open_positions == 7

    def test_paper_is_the_default_mode(self) -> None:
        assert load_settings().is_paper


class TestRiskValidation:
    def test_risk_above_position_cap_rejected(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed max_position_pct"):
            RiskConfig(risk_per_trade_pct=0.5, max_position_pct=0.2)

    def test_daily_loss_above_drawdown_rejected(self) -> None:
        with pytest.raises(ValueError, match="could never trigger"):
            RiskConfig(max_daily_loss_pct=0.2, max_drawdown_pct=0.1)

    def test_weekly_below_daily_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >="):
            RiskConfig(max_trades_per_day=5, max_trades_per_week=2)

    def test_theoretical_exposure(self) -> None:
        config = RiskConfig(max_open_positions=5, max_position_pct=0.2)
        assert config.theoretical_max_exposure_pct == pytest.approx(1.0)


class TestSettlementCannotBeDisabled:
    def test_blocking_unsettled_cash_is_not_optional(self) -> None:
        """Trading on unsettled proceeds is outside what this app permits."""
        with pytest.raises(ValueError, match="good-faith violation"):
            SettlementConfig(block_unsettled_usage=False)

    def test_default_is_t_plus_one(self) -> None:
        assert SettlementConfig().settlement_days == 1


class TestExecutionFractionalGate:
    def test_fractional_disabled_without_acknowledgement(self) -> None:
        config = ExecutionConfig(mode="hybrid", allow_fractional_fallback=True)
        assert not config.fractional_enabled
        assert config.fractional_suppressed

    def test_fractional_enabled_with_acknowledgement(self) -> None:
        config = ExecutionConfig(
            mode="hybrid",
            allow_fractional_fallback=True,
            acknowledge_fractional_stop_risk=True,
        )
        assert config.fractional_enabled
        assert not config.fractional_suppressed

    def test_whole_share_mode_does_not_flag_suppression(self) -> None:
        config = ExecutionConfig(mode="whole_share_only", allow_fractional_fallback=False)
        assert not config.fractional_suppressed


class TestUniverseValidation:
    def test_symbols_are_upper_cased(self) -> None:
        assert UniverseConfig(whitelist=["spus", " hlal "]).whitelist == ["SPUS", "HLAL"]

    def test_symbol_on_both_lists_rejected(self) -> None:
        with pytest.raises(ValueError, match="both whitelist and blacklist"):
            UniverseConfig(whitelist=["SPUS"], blacklist=["spus"])

    def test_inverted_price_band_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_price must be below"):
            UniverseConfig(min_price=100.0, max_price=10.0)


class TestSecrets:
    def test_absent_credentials_detected(self) -> None:
        assert not Secrets(_env_file=None).has_alpaca_credentials  # type: ignore[call-arg]

    def test_secret_value_not_in_repr(self) -> None:
        secrets = Secrets(_env_file=None, alpaca_api_key="PKSUPERSECRET123")  # type: ignore[call-arg]
        assert "PKSUPERSECRET123" not in repr(secrets)

    def test_telegram_whitelist_parsing(self) -> None:
        secrets = Secrets(_env_file=None, telegram_allowed_user_ids=" 123, 456 ,bad, ")  # type: ignore[call-arg]
        assert secrets.allowed_telegram_ids == frozenset({123, 456})

    def test_empty_whitelist_is_empty_not_permissive(self) -> None:
        """Fail closed: no configured ids means the bot talks to nobody."""
        assert Secrets(_env_file=None).allowed_telegram_ids == frozenset()  # type: ignore[call-arg]

    def test_paper_url_detected(self) -> None:
        assert not Secrets(_env_file=None).is_live_alpaca_url  # type: ignore[call-arg]

    def test_live_url_detected(self) -> None:
        secrets = Secrets(_env_file=None, alpaca_base_url="https://api.alpaca.markets")  # type: ignore[call-arg]
        assert secrets.is_live_alpaca_url


class TestStartupWarnings:
    def test_live_mode_warns_loudly(self) -> None:
        settings = load_settings(overrides={"trading_mode": "live"})
        assert any("LIVE" in w for w in settings.startup_warnings())

    def test_suppressed_fractional_is_explained(self) -> None:
        warnings = load_settings().startup_warnings()
        assert any("DISABLED" in w and "fractional" in w.lower() for w in warnings)

    def test_mode_b_with_mock_screener_warns(self) -> None:
        settings = load_settings(
            overrides={"universe": {"mode": "B"}, "shariah": {"provider": "mock_external"}}
        )
        assert any("Mock screens" in w for w in settings.startup_warnings())


class TestTelegramChannelValidation:
    """Catch a malformed channel id at startup, not after four failed retries."""

    def _secrets(self, channel_id: str) -> Secrets:
        return Secrets(_env_file=None, telegram_channel_id=channel_id)  # type: ignore[call-arg]

    def test_valid_channel_id(self) -> None:
        assert self._secrets("-1001234567890").telegram_channel_problem is None

    def test_public_username_is_valid(self) -> None:
        assert self._secrets("@my_channel").telegram_channel_problem is None

    def test_unset_is_not_a_problem(self) -> None:
        assert self._secrets("").telegram_channel_problem is None

    def test_missing_leading_minus_is_caught(self) -> None:
        """The real failure: the sign gets dropped when copied from a t.me URL."""
        problem = self._secrets("1001234567890").telegram_channel_problem
        assert problem is not None
        assert "-100" in problem

    def test_positive_id_is_caught(self) -> None:
        problem = self._secrets("123456789").telegram_channel_problem
        assert problem is not None
        assert "negative" in problem

    def test_garbage_is_caught(self) -> None:
        assert self._secrets("not-an-id").telegram_channel_problem is not None


def _all_modules() -> list[str]:
    """Every importable module in the package.

    Walking modules rather than listing packages: the Phase 4 version checked
    only top-level packages and therefore missed
    execution.order_manager -> engine -> execution.order_manager, which is the
    same class of bug one level down.
    """
    import pkgutil

    import investment_box

    names: list[str] = ["investment_box"]
    for info in pkgutil.walk_packages(
        investment_box.__path__, prefix="investment_box."
    ):
        # The Streamlit app executes on import (it calls main() at module
        # scope), so importing it here would try to render a page.
        if info.name.startswith("investment_box.ui.app"):
            continue
        if ".migrations" in info.name:
            continue
        names.append(info.name)
    return names


class TestNoImportCycles:
    """Every module must import cleanly when imported FIRST.

    A cycle only shows up when the wrong module is imported first, so the test
    suite's own import order hides it. Two real cycles shipped this way:
    forecast -> shariah -> services -> forecast, and
    execution.order_manager -> engine -> execution.order_manager.
    """

    def test_every_module_imports_first(self) -> None:
        import subprocess
        import sys

        failures: list[str] = []
        for module in _all_modules():
            result = subprocess.run(  # noqa: S603
                [sys.executable, "-c", f"import {module}"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                tail = result.stderr.strip().splitlines()[-1:] or ["<no stderr>"]
                failures.append(f"{module}: {tail[0]}")

        assert not failures, "modules that fail when imported first:\n" + "\n".join(
            failures
        )

    @pytest.mark.parametrize(
        "module",
        [
            "investment_box.backtest",
            "investment_box.config",
            "investment_box.core",
            "investment_box.data",
            "investment_box.engine",
            "investment_box.execution",
            "investment_box.features",
            "investment_box.forecast",
            "investment_box.risk",
            "investment_box.services",
            "investment_box.shariah",
            "investment_box.strategies",
            "investment_box.telegram",
            "investment_box.universe",
        ],
    )
    def test_package_imports_first(self, module: str) -> None:
        import subprocess
        import sys

        result = subprocess.run(  # noqa: S603
            [sys.executable, "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"importing {module} first failed:\n{result.stderr[-1500:]}"
        )
