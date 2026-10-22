"""Logging configuration with a filter that redacts API keys from every record."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

# Key shapes we never want in logs even if they are not the configured secret.
_KEY_PATTERNS = (
    re.compile(r"nvapi-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{10,}"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(x-api-key:\s*)[A-Za-z0-9._\-]+"),
)


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Replace configured secrets and known key patterns in ``text``."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    for pattern in _KEY_PATTERNS:
        text = pattern.sub(lambda m: (m.group(1) if m.groups() else "") + REDACTED, text)
    return text


class SecretRedactingFilter(logging.Filter):
    """Rewrites each record's final message with secrets removed."""

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        cleaned = redact(message, self._secrets)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


def configure_logging(level: str, secrets: Iterable[str]) -> None:
    """Install a root handler whose filter redacts ``secrets``."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(SecretRedactingFilter(secrets))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    # httpx logs full request lines at INFO; keep it quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
