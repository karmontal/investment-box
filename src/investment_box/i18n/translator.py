"""Message lookup with an English fallback.

Two decisions worth stating:

* A missing Arabic key falls back to English rather than rendering an empty
  string or the raw key. A partially-translated alert is still readable; a
  blank one is not.
* ``language: both`` renders English and Arabic together, separated by a rule.
  Telegram has no per-message language switch, so bilingual means both in one
  message rather than picking one and hoping.

Numbers, tickers and timestamps are never translated -- Arabic-Indic digits in
a price field would be actively harmful.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from investment_box.core.logging import get_logger

log = get_logger(__name__)

CATALOGUE_DIR = Path(__file__).parent
FALLBACK_LANGUAGE = "en"
SUPPORTED = ("en", "ar")

#: Right-to-left mark, used to stop Arabic text mangling adjacent Latin tickers.
RLM = "‏"


@lru_cache(maxsize=4)
def _load_catalogue(language: str) -> dict[str, Any]:
    path = CATALOGUE_DIR / f"{language}.yaml"
    if not path.exists():
        log.warning("i18n.catalogue_missing", language=language)
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _lookup(catalogue: dict[str, Any], key: str) -> str | None:
    """Resolve a dotted key such as ``balance.equity``."""
    node: Any = catalogue
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return str(node) if isinstance(node, (str, int, float)) else None


class Translator:
    """Renders message keys in one language, or both."""

    def __init__(self, language: str = "both") -> None:
        if language not in (*SUPPORTED, "both"):
            raise ValueError(f"unsupported language {language!r}; expected one of en, ar, both")
        self.language = language

    @property
    def is_rtl(self) -> bool:
        return self.language == "ar"

    def t(self, key: str, **params: object) -> str:
        """Translate ``key``, formatting any ``{placeholders}`` from ``params``."""
        if self.language == "both":
            english = self._one("en", key, **params)
            arabic = self._one("ar", key, **params)
            if english == arabic:
                return english
            return f"{english}\n{RLM}{arabic}"
        return self._one(self.language, key, **params)

    def _one(self, language: str, key: str, **params: object) -> str:
        value = _lookup(_load_catalogue(language), key)
        if value is None and language != FALLBACK_LANGUAGE:
            value = _lookup(_load_catalogue(FALLBACK_LANGUAGE), key)
        if value is None:
            log.warning("i18n.missing_key", key=key, language=language)
            return key  # surfacing the key beats rendering nothing
        if params:
            try:
                return value.format(**params)
            except (KeyError, IndexError):
                log.warning("i18n.format_failed", key=key, params=sorted(params))
                return value
        return value

    def bilingual_label(self, key: str) -> str:
        """``English / عربي`` on one line, for table headers and short labels."""
        english = self._one("en", key)
        arabic = self._one("ar", key)
        if self.language == "en" or english == arabic:
            return english
        if self.language == "ar":
            return arabic
        return f"{english} / {RLM}{arabic}"


@lru_cache(maxsize=4)
def get_translator(language: str = "both") -> Translator:
    return Translator(language)


def clear_catalogue_cache() -> None:
    """Drop cached catalogues. Tests, and the UI after editing a catalogue."""
    _load_catalogue.cache_clear()
    get_translator.cache_clear()
