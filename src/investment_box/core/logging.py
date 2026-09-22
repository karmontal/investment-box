"""Structured logging with mandatory secret redaction.

Principle 5 of the build spec says secrets must never reach a log. Relying on
every call site to remember that is not a control, so redaction is a processor
in the structlog pipeline: it runs on every event, and it scrubs both known
secret key names and anything that pattern-matches an API key or token.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

import structlog

#: Field names whose values are replaced wholesale, matched case-insensitively
#: on substrings so that ``alpaca_secret_key`` and ``bot_token`` both hit.
_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "authorization",
    "auth_header",
    "private_key",
)

#: Value-level patterns, for secrets that arrive inside a message string rather
#: than as their own field -- e.g. an exception text echoing a request URL.
_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Telegram bot token: <digits>:<35 base64-ish chars>. No leading \b -- in
    # practice the token arrives inside a URL as ".../bot<token>/sendMessage",
    # where "t" to "1" is not a word boundary and \b would never match.
    re.compile(r"\d{6,12}:[A-Za-z0-9_-]{30,}"),
    # Alpaca-style keys: long uppercase alphanumeric runs
    re.compile(r"\b(?:PK|AK)[A-Z0-9]{16,}\b"),
    # Anything in a query string that looks like a key
    re.compile(r"(?i)\b(api[_-]?key|token|secret)=([^&\s]+)"),
    # Bearer tokens
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}"),
)

REDACTED = "***REDACTED***"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _SECRET_KEY_FRAGMENTS)


def _scrub_text(text: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def _scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {
            k: (REDACTED if _is_secret_key(str(k)) else _scrub_value(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        scrubbed = [_scrub_value(v) for v in value]
        return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


def redact_secrets(
    _logger: Any, _method: str, event_dict: structlog.types.EventDict
) -> structlog.types.EventDict:
    """structlog processor: strip secrets from keys and values alike."""
    return {
        key: (REDACTED if _is_secret_key(str(key)) else _scrub_value(value))
        for key, value in event_dict.items()
    }


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    log_file: Path | None = None,
) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent -- safe to call from tests and from each entry point.

    Args:
        level: Standard logging level name.
        json_output: Emit JSON lines instead of a human-readable console
            rendering. Used in Docker where logs are shipped, not read directly.
        log_file: Optional file to tee logs into, in addition to stderr.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(format="%(message)s", level=numeric_level, handlers=handlers, force=True)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            redact_secrets,  # always last before rendering
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
