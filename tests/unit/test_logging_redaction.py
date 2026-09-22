"""Secret redaction.

Principle 5 says secrets never reach a log. These tests treat that as a
property of the pipeline rather than a habit of the call site.
"""

from __future__ import annotations

from investment_box.core.logging import REDACTED, redact_secrets


def scrub(event: dict) -> dict:
    return redact_secrets(None, "info", event)


class TestKeyRedaction:
    def test_api_key_field_redacted(self) -> None:
        assert scrub({"api_key": "PKLIVE123456789ABC"})["api_key"] == REDACTED

    def test_prefixed_and_suffixed_names_redacted(self) -> None:
        out = scrub(
            {
                "alpaca_secret_key": "abc",
                "telegram_bot_token": "def",
                "db_password": "ghi",
                "authorization": "jkl",
            }
        )
        assert all(value == REDACTED for value in out.values())

    def test_case_insensitive(self) -> None:
        assert scrub({"API_KEY": "abc"})["API_KEY"] == REDACTED

    def test_innocent_fields_untouched(self) -> None:
        out = scrub({"symbol": "SPUS", "quantity": 2, "event": "order.filled"})
        assert out == {"symbol": "SPUS", "quantity": 2, "event": "order.filled"}


class TestValueRedaction:
    def test_telegram_token_in_free_text(self) -> None:
        out = scrub({"event": "failed calling https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/sendMessage"})
        assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in out["event"]
        assert REDACTED in out["event"]

    def test_alpaca_style_key_in_free_text(self) -> None:
        out = scrub({"error": "auth failed for PKTEST1234567890ABCD"})
        assert "PKTEST1234567890ABCD" not in out["error"]

    def test_query_string_secret(self) -> None:
        out = scrub({"url": "https://example.com/v1?api_key=supersecretvalue&symbol=SPUS"})
        assert "supersecretvalue" not in out["url"]
        assert "SPUS" in out["url"]

    def test_bearer_token(self) -> None:
        out = scrub({"headers": "Authorization: Bearer abcdef1234567890xyz"})
        assert "abcdef1234567890xyz" not in out["headers"]


class TestNestedStructures:
    def test_nested_dict(self) -> None:
        out = scrub({"config": {"broker": "alpaca", "secret_key": "shh"}})
        assert out["config"]["secret_key"] == REDACTED
        assert out["config"]["broker"] == "alpaca"

    def test_list_of_strings(self) -> None:
        out = scrub({"messages": ["ok", "token=abcdefghijklmnop"]})
        assert "abcdefghijklmnop" not in out["messages"][1]

    def test_list_of_dicts(self) -> None:
        out = scrub({"items": [{"api_key": "x", "symbol": "SPUS"}]})
        assert out["items"][0]["api_key"] == REDACTED
        assert out["items"][0]["symbol"] == "SPUS"

    def test_tuple_preserved_as_tuple(self) -> None:
        out = scrub({"pair": ("a", "b")})
        assert isinstance(out["pair"], tuple)


class TestExceptionText:
    def test_secret_inside_an_exception_message_is_scrubbed(self) -> None:
        """The realistic leak: nobody logs a key on purpose, a traceback does it."""
        message = (
            "ConnectionError: GET https://paper-api.alpaca.markets/v2/account "
            "failed (key=PKABCDEFGHIJ1234567)"
        )
        out = scrub({"exc_info": message})
        assert "PKABCDEFGHIJ1234567" not in out["exc_info"]
