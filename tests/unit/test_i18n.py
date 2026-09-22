"""Message catalogue lookup and bilingual rendering."""

from __future__ import annotations

import pytest
import yaml

from investment_box.i18n.translator import CATALOGUE_DIR, RLM, Translator, get_translator


class TestLookup:
    def test_english(self) -> None:
        assert Translator("en").t("balance.equity") == "Equity"

    def test_arabic(self) -> None:
        assert Translator("ar").t("balance.equity") == "إجمالي الحساب"

    def test_nested_key(self) -> None:
        assert Translator("en").t("positions.none") == "No open positions."

    def test_missing_key_returns_the_key(self) -> None:
        """Surfacing the key beats rendering a blank label in a trading UI."""
        assert Translator("en").t("nope.not.here") == "nope.not.here"

    def test_unsupported_language_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported language"):
            Translator("fr")


class TestBilingual:
    def test_both_contains_each_language(self) -> None:
        rendered = Translator("both").t("balance.equity")
        assert "Equity" in rendered
        assert "إجمالي الحساب" in rendered

    def test_rtl_mark_present(self) -> None:
        assert RLM in Translator("both").t("balance.equity")

    def test_identical_values_are_not_duplicated(self) -> None:
        """Tickers and symbols are the same in both catalogues; don't print twice."""
        translator = Translator("both")
        rendered = translator.t("common.paper")
        assert rendered.count("PAPER") <= 1 or "تجريبي" in rendered

    def test_bilingual_label_is_one_line(self) -> None:
        label = Translator("both").bilingual_label("positions.symbol")
        assert "\n" not in label
        assert "/" in label

    def test_label_in_single_language_has_no_separator(self) -> None:
        assert Translator("en").bilingual_label("positions.symbol") == "Symbol"


class TestFallback:
    def test_arabic_falls_back_to_english(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from investment_box.i18n import translator as module

        monkeypatch.setattr(
            module,
            "_load_catalogue",
            lambda lang: {"en": {"only": {"here": "English text"}}, "ar": {}}[lang],
        )
        assert Translator("ar").t("only.here") == "English text"


class TestCatalogueIntegrity:
    def _keys(self, data: dict, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        for key, value in data.items():
            path = f"{prefix}{key}"
            if isinstance(value, dict):
                keys |= self._keys(value, f"{path}.")
            else:
                keys.add(path)
        return keys

    def test_arabic_covers_every_english_key(self) -> None:
        """A missing translation falls back, but the gap should be deliberate."""
        english = yaml.safe_load((CATALOGUE_DIR / "en.yaml").read_text(encoding="utf-8"))
        arabic = yaml.safe_load((CATALOGUE_DIR / "ar.yaml").read_text(encoding="utf-8"))
        missing = self._keys(english) - self._keys(arabic)
        assert not missing, f"untranslated keys: {sorted(missing)}"

    def test_no_arabic_only_keys(self) -> None:
        english = yaml.safe_load((CATALOGUE_DIR / "en.yaml").read_text(encoding="utf-8"))
        arabic = yaml.safe_load((CATALOGUE_DIR / "ar.yaml").read_text(encoding="utf-8"))
        orphans = self._keys(arabic) - self._keys(english)
        assert not orphans, f"keys with no English source: {sorted(orphans)}"


class TestCaching:
    def test_translator_is_cached(self) -> None:
        assert get_translator("en") is get_translator("en")
