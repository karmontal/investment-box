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
from investment_box.core.types import AutonomyLevel, TradingMode


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


class TestDataDirRelocatesEverything:
    """One setting must move ALL state.

    Found by actually running the container: cache_dir defaulted to a relative
    './data/cache' independent of data_dir, so IB__DATA_DIR=/data did not move
    it and the non-root container tried to write inside /app.
    """

    def test_cache_dir_follows_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IB__DATA_DIR", "/somewhere/else")
        settings = load_settings()
        assert settings.resolved_cache_dir == Path("/somewhere/else/cache")

    def test_db_path_follows_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IB__DATA_DIR", "/somewhere/else")
        assert load_settings().db_path == Path("/somewhere/else/investment_box.db")

    def test_every_state_path_is_under_data_dir(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing may escape data_dir, or a container cannot relocate it."""
        monkeypatch.setenv("IB__DATA_DIR", "/data")
        settings = load_settings()
        root = Path("/data")
        for path in (settings.db_path, settings.resolved_cache_dir):
            assert root in path.parents or path == root, f"{path} escapes {root}"

    def test_explicit_cache_dir_still_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IB__DATA_DIR", "/data")
        monkeypatch.setenv("IB__DATA__CACHE_DIR", "/fast-disk/cache")
        assert load_settings().resolved_cache_dir == Path("/fast-disk/cache")

    def test_repository_uses_the_resolved_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from investment_box.data.repository import MarketDataRepository

        monkeypatch.setenv("IB__DATA_DIR", str(tmp_path / "state"))
        repository = MarketDataRepository.from_settings(load_settings())
        assert repository.cache.directory == tmp_path / "state" / "cache"


class TestSeedUniverseProvenance:
    """The shipped `config/universe_etf.yaml` must stay honest about its sources.

    These guard a specific future mistake: flipping `verified: true` on an entry
    whose provenance was never filled in. The flag is meant to assert "I read the
    documents"; an entry with a null certifying board proves nobody did.
    """

    @staticmethod
    def _entries() -> list[dict[str, object]]:
        from investment_box.config.loader import load_universe_file

        return list(load_universe_file()["etfs"])

    def test_every_entry_has_a_symbol_and_is_unique(self) -> None:
        symbols = [str(e["symbol"]).upper() for e in self._entries()]
        assert symbols, "the seed universe is empty"
        assert len(symbols) == len(set(symbols)), f"duplicate symbols: {symbols}"

    @pytest.mark.parametrize("field", ["name", "issuer", "certifying_board", "inception"])
    def test_a_verified_entry_has_its_provenance_filled_in(self, field: str) -> None:
        missing = [
            str(e["symbol"])
            for e in self._entries()
            if e.get("verified") and not e.get(field)
        ]
        assert not missing, (
            f"{missing} are marked verified but have no {field}. Verification means "
            f"you read it off the fund's own documents -- fill it in or unverify."
        )

    def test_no_entry_is_a_forbidden_instrument(self) -> None:
        from investment_box.shariah.constraints import is_forbidden_instrument

        for entry in self._entries():
            reason = is_forbidden_instrument(
                str(entry["symbol"]),
                entry.get("name") and str(entry["name"]),
                entry.get("asset_class") and str(entry["asset_class"]),
            )
            assert reason is None, f"{entry['symbol']}: {reason}"

    def test_inception_dates_are_dates_not_strings_or_future(self) -> None:
        import datetime as dt

        today = dt.date.today()  # noqa: DTZ011 - a fund cannot launch tomorrow in any zone
        for entry in self._entries():
            inception = entry.get("inception")
            if inception is None:
                continue
            if isinstance(inception, dt.datetime):
                inception = inception.date()
            assert isinstance(inception, dt.date), (
                f"{entry['symbol']}: inception {inception!r} did not parse as a date; "
                f"write it unquoted as YYYY-MM-DD"
            )
            assert inception <= today, f"{entry['symbol']}: inception {inception} is in the future"


class TestDeploymentConfigDoesNotReachTests:
    """A machine's own `config/local.yaml` must not change what the suite asserts.

    Found the hard way: verifying funds and raising `autonomy_level` to 2 in a
    real deployment broke five unrelated tests on that machine and nowhere
    else. A test suite whose result depends on the operator's tuning cannot
    tell you whether the code is correct.
    """

    def test_the_suite_sees_shipped_defaults_not_local_overrides(self) -> None:
        from investment_box.config.loader import config_dir, load_settings

        local = config_dir() / "local.yaml"
        settings = load_settings()

        if local.exists():
            raw = local.read_text()
            if "autonomy_level: 2" in raw:
                assert settings.engine.autonomy_level is AutonomyLevel.SUGGEST_ONLY, (
                    "config/local.yaml leaked into the test suite"
                )
            if "whitelist: [" in raw:
                assert settings.universe.whitelist == [], (
                    "config/local.yaml's whitelist leaked into the test suite"
                )

        # True regardless of whether this machine happens to have a local.yaml.
        assert settings.engine.autonomy_level is AutonomyLevel.SUGGEST_ONLY
        assert settings.universe.whitelist == []


class TestEverySourceFileIsInTheRepository:
    """A clone must contain the whole application.

    This exists because it did not. `.gitignore` carried an unanchored `data/`
    to keep the runtime state directory out of the repo; git applies such a
    pattern at every depth, so it also excluded `src/investment_box/data/` --
    the entire data layer. Every local test still passed, because the files
    were present on disk and merely untracked. The failure only surfaced on a
    fresh clone, as `ModuleNotFoundError: No module named 'investment_box.data'`.

    Note this is NOT how `.dockerignore` reads the same line: its patterns are
    anchored to the build context root, so the image was fine and the
    repository was not. Identical text, different semantics.
    """

    @staticmethod
    def _tracked() -> set[str]:
        import subprocess

        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(
            ["git", "ls-files", "src", "scripts", "tests"],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return {line.strip() for line in out.stdout.splitlines() if line.strip()}

    def test_no_python_source_file_is_untracked(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tracked = self._tracked()

        missing: list[str] = []
        for directory in ("src", "scripts", "tests"):
            for path in (root / directory).rglob("*.py"):
                if any(part in {"__pycache__", ".venv"} for part in path.parts):
                    continue
                rel = path.relative_to(root).as_posix()
                if rel not in tracked:
                    missing.append(rel)

        assert not missing, (
            "these source files exist on disk but are not in the repository, so a "
            f"clone would not build: {sorted(missing)}. Check .gitignore for an "
            f"unanchored directory pattern -- git applies those at every depth."
        )

    def test_every_package_directory_has_a_tracked_init(self) -> None:
        root = Path(__file__).resolve().parents[2]
        tracked = self._tracked()

        missing = [
            (pkg.relative_to(root) / "__init__.py").as_posix()
            for pkg in (root / "src" / "investment_box").rglob("*")
            if pkg.is_dir()
            and pkg.name != "__pycache__"
            and (pkg / "__init__.py").exists()
            and (pkg.relative_to(root) / "__init__.py").as_posix() not in tracked
        ]
        assert not missing, f"untracked package initialisers: {sorted(missing)}"


class TestUniverseEditsAreSeenWithoutARestart:
    """`config/` is a live bind mount so the list can be edited in place.

    It was cached with no key, so marking a fund `verified: true` changed
    nothing on screen until the container restarted -- indistinguishable from
    the edit not having worked.
    """

    def test_the_fingerprint_changes_when_the_file_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from investment_box.config import loader

        target = tmp_path / "universe_etf.yaml"
        target.write_text("etfs:\n  - symbol: SPUS\n    verified: false\n")
        monkeypatch.setattr(loader, "config_dir", lambda: tmp_path)

        from investment_box.ui.state import _universe_fingerprint

        before = _universe_fingerprint()
        target.write_text("etfs:\n  - symbol: SPUS\n    verified: true\n    name: x\n")
        after = _universe_fingerprint()

        assert before != after, (
            "editing the universe file must change its fingerprint, or the dashboard "
            "keeps serving the version it read at startup"
        )

    def test_the_cache_key_argument_is_actually_hashed_by_streamlit(self) -> None:
        """Streamlit ignores any cached argument whose name starts with `_`.

        That rule is used deliberately elsewhere in this module (`_as_of` on
        `get_research`), which is what made the mistake so easy: naming the key
        `_fingerprint` removed it from the cache key, so the cache stayed as
        unkeyed as it had been. The attribute existed, the code looked fixed,
        and the dashboard kept serving the list it read at startup.
        """
        import inspect

        from investment_box.ui.state import _load_universe

        # Streamlit wraps the function; unwrap to reach the real signature.
        target = getattr(_load_universe, "__wrapped__", _load_universe)
        params = list(inspect.signature(target).parameters)

        assert params, "_load_universe must take a cache-key argument"
        leading_underscore = [p for p in params if p.startswith("_")]
        assert not leading_underscore, (
            f"{leading_underscore} start with an underscore, so Streamlit will not "
            f"hash them and the cache will never invalidate. Drop the underscore."
        )

    def test_a_missing_file_yields_a_stable_fingerprint_rather_than_raising(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from investment_box.config import loader
        from investment_box.ui.state import _universe_fingerprint

        monkeypatch.setattr(loader, "config_dir", lambda: tmp_path / "absent")
        assert _universe_fingerprint() == (0.0, 0)
