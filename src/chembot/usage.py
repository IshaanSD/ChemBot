"""Process-wide daily usage meter with optional per-key caps.

Every outbound cost — the OpenAI call plus each free-but-rate-limited provider
call (openFDA, PubMed/NCBI, RxNorm) — is recorded here under a key. Keys can
carry a daily limit; recording past it raises `DailyCapExceeded`, which the API
turns into a 429.

By default no key has a limit, so the CLI and eval paths (which import the
instrumented clients) just increment counters that nobody reads — zero behavior
change. The API sets limits at startup.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Canonical usage keys.
QUERY = "query"      # one live (non-cached) answered question
OPENAI = "openai"    # one OpenAI chat completion
OPENFDA = "openfda"  # one openFDA tool call (label or event)
PUBMED = "pubmed"    # one PubMed/NCBI tool call
RXNORM = "rxnorm"    # one RxNorm tool call


class DailyCapExceeded(Exception):
    def __init__(self, key: str, limit: int):
        self.key = key
        self.limit = limit
        super().__init__(f"Daily cap for '{key}' ({limit}) reached")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._date: Optional[str] = None
        self._counts: Dict[str, int] = {}
        self._limits: Dict[str, Optional[int]] = {}

    def set_limits(self, limits: Dict[str, Optional[int]]) -> None:
        with self._lock:
            self._limits.update(limits)

    def _roll_locked(self, today: str) -> None:
        if self._date != today:
            self._date = today
            self._counts = {}

    def record(self, key: str, n: int = 1) -> None:
        """Increment `key` by `n`. Raises DailyCapExceeded if it would exceed the
        key's limit (checked before incrementing, so the cap is never overshot)."""
        today = _today()
        with self._lock:
            self._roll_locked(today)
            limit = self._limits.get(key)
            current = self._counts.get(key, 0)
            if limit is not None and current + n > limit:
                raise DailyCapExceeded(key, limit)
            self._counts[key] = current + n
            if limit is not None:
                logger.info("usage %s=%d/%d", key, self._counts[key], limit)

    def snapshot(self) -> dict:
        today = _today()
        with self._lock:
            self._roll_locked(today)
            return {
                "date": self._date,
                "counts": dict(self._counts),
                "limits": {k: v for k, v in self._limits.items()},
            }


# Process-wide singleton.
METER = UsageMeter()


def record_provider_call(key: str, n: int = 1) -> None:
    """Record an outbound provider call. Called from the client choke-points."""
    METER.record(key, n)


def _int_or_none(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


def configure_limits_from_env() -> Dict[str, Optional[int]]:
    """Set daily caps from env vars, returning the applied limits.

    DAILY_QUERY_LIMIT gates live answered questions (the master cost ceiling);
    the per-provider caps are backstops. Empty/unset means unlimited.
    """
    limits: Dict[str, Optional[int]] = {
        QUERY: _int_or_none(os.getenv("DAILY_QUERY_LIMIT", "50")),
        OPENAI: _int_or_none(os.getenv("DAILY_OPENAI_LIMIT", "50")),
        OPENFDA: _int_or_none(os.getenv("DAILY_OPENFDA_LIMIT", "150")),
        PUBMED: _int_or_none(os.getenv("DAILY_PUBMED_LIMIT", "150")),
        RXNORM: _int_or_none(os.getenv("DAILY_RXNORM_LIMIT", "150")),
    }
    METER.set_limits(limits)
    return limits
