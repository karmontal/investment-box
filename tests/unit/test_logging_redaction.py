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


class TestStdlibRedaction:
    """Regression tests for a real leak.

    Every credential below is fabricated. Never paste a live token into a test,
    even a revoked one: tests are committed, and a committed secret outlives
    the incident that produced it.

    The Telegram Bot API puts the token in the URL path, and httpx logs the
    full URL through the stdlib logger -- bypassing structlog's processors
    entirely. This reached a live terminal before it was caught.
    """

    @staticmethod
    def _capture(logger_name: str, msg: str, *args: object) -> str:
        import io
        import logging

        from investment_box.core.logging import SecretRedactingFilter

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.addFilter(SecretRedactingFilter())
        handler.setFormatter(logging.Formatter("%(message)s"))

        logger = logging.getLogger(logger_name)
        logger.handlers = [handler]
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.warning(msg, *args)
        return stream.getvalue()

    def test_telegram_token_in_a_url_argument(self) -> None:
        """The exact shape httpx emits: token in the URL, URL passed as an arg."""
        output = self._capture(
            "httpx",
            'HTTP Request: %s %s "%s"',
            "POST",
            "https://api.telegram.org/bot1234567890:AAFfakeTokenForTestsOnly_NotReal12345/sendMessage",
            "HTTP/1.1 400 Bad Request",
        )
        assert "AAFfakeTokenForTestsOnly_NotReal12345" not in output
        assert REDACTED in output
        # The useful part of the message must survive.
        assert "api.telegram.org" in output
        assert "400 Bad Request" in output

    def test_token_embedded_in_the_message_itself(self) -> None:
        output = self._capture(
            "some.library",
            "calling https://api.telegram.org/bot123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw/getMe",
        )
        assert "AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw" not in output

    def test_alpaca_key_in_an_argument(self) -> None:
        output = self._capture("alpaca", "auth header: %s", "PKTEST1234567890ABCDEF")
        assert "PKTEST1234567890ABCDEF" not in output

    def test_dict_style_args(self) -> None:
        output = self._capture(
            "some.library", "request %(url)s", {"url": "https://x/?api_key=supersecretvalue"}
        )
        assert "supersecretvalue" not in output

    def test_innocent_messages_pass_through_intact(self) -> None:
        output = self._capture("investment_box", "placed order for %s", "SPUS")
        assert "SPUS" in output
        assert REDACTED not in output

    def test_noisy_http_loggers_are_pinned_to_warning(self) -> None:
        """Second line of defence: don't even emit the URL at INFO."""
        import logging

        from investment_box.core.logging import configure_logging

        configure_logging("INFO")
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("httpcore").level >= logging.WARNING

    def test_debug_level_still_redacts(self) -> None:
        """Turning logging up for troubleshooting must not turn redaction off."""
        import logging

        from investment_box.core.logging import configure_logging

        configure_logging("DEBUG")
        root_handlers = logging.getLogger().handlers
        assert root_handlers
        assert any(
            any(f.__class__.__name__ == "SecretRedactingFilter" for f in h.filters)
            for h in root_handlers
        )
