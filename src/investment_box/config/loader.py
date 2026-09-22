"""Configuration loading and layering.

Precedence, lowest to highest::

    config/default.yaml  <  config/local.yaml  <  environment (IB__*)

``local.yaml`` is gitignored, so your tuning never shows up in a diff, and
secrets only ever live in the environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from investment_box.config.schema import Secrets, Settings
from investment_box.core.errors import ConfigError

DEFAULT_CONFIG_NAME = "default.yaml"
LOCAL_CONFIG_NAME = "local.yaml"
UNIVERSE_CONFIG_NAME = "universe_etf.yaml"

#: The dotenv file consulted for settings and secrets. Tests set this to
#: ``None`` so that a developer's real .env -- with their real broker and
#: Telegram credentials in it -- can never influence a test run or leak into
#: a test's view of the world. Without this, the suite behaves differently on a
#: configured machine than in CI, which defeats the point of having it.
DEFAULT_ENV_FILE: str | None = ".env"


def project_root() -> Path:
    """The repository root, i.e. the directory containing ``config/``.

    Resolved from this file's location rather than the working directory, so
    that the scheduler, the dashboard and pytest all agree regardless of where
    they were started from.
    """
    return Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    """The config directory, overridable with ``IB_CONFIG_DIR`` for tests."""
    override = os.environ.get("IB_CONFIG_DIR")
    return Path(override) if override else project_root() / "config"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict.

    Nested mappings merge key by key; every other type (including lists) is
    replaced wholesale. Replacing lists is deliberate -- a local ``blacklist``
    should override the default rather than silently append to it.
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_settings(
    *,
    config_path: Path | None = None,
    local_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a :class:`Settings` from the YAML layers plus the environment.

    Args:
        config_path: Defaults to ``config/default.yaml``.
        local_path: Defaults to ``config/local.yaml`` if it exists.
        overrides: Final in-process layer, used by tests.

    Raises:
        ConfigError: If a file is unreadable or the merged result is invalid.
    """
    directory = config_dir()
    base = _read_yaml(config_path or directory / DEFAULT_CONFIG_NAME)
    local = _read_yaml(local_path or directory / LOCAL_CONFIG_NAME)

    merged = deep_merge(base, local)
    if overrides:
        merged = deep_merge(merged, overrides)

    try:
        # Environment variables are applied by pydantic-settings on top of the
        # values passed here, so IB__* always wins over both YAML layers.
        return Settings(_env_file=DEFAULT_ENV_FILE, **merged)  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - re-raised with context
        raise ConfigError(f"Invalid configuration: {exc}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings.

    Use :func:`load_settings` directly in tests to avoid the cache.
    """
    return load_settings()


@lru_cache(maxsize=1)
def get_secrets() -> Secrets:
    """Process-wide cached secrets."""
    return Secrets(_env_file=DEFAULT_ENV_FILE)  # type: ignore[call-arg]


def load_universe_file(path: Path | None = None) -> dict[str, Any]:
    """Load ``config/universe_etf.yaml``.

    Returns the raw mapping; :mod:`investment_box.universe` turns it into
    validated instrument records in Phase 3.
    """
    target = path or config_dir() / UNIVERSE_CONFIG_NAME
    data = _read_yaml(target)
    if "etfs" not in data:
        raise ConfigError(f"{target} has no 'etfs' key")
    return data


def clear_caches() -> None:
    """Drop cached settings. Called by tests and after a UI config change."""
    get_settings.cache_clear()
    get_secrets.cache_clear()
