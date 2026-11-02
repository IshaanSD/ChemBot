"""Answer cache: repeated questions are served without re-running the pipeline.

A cache hit costs nothing — no OpenAI call, no provider calls — so hits are free
and never counted against the daily cap. This is the main cost lever: it also
lets pre-warmed example questions stay answerable even once the live cap is hit.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

_MAX_ENTRIES = 512
_lock = threading.Lock()
# normalized question -> (answer, sources)
_cache: "dict[str, Tuple[str, list]]" = {}


def normalize_question(question: str) -> str:
    return " ".join(question.strip().lower().split())


def peek(question: str) -> Optional[Tuple[str, list]]:
    """Return (answer, sources) if cached, without ever running the pipeline."""
    key = normalize_question(question)
    with _lock:
        return _cache.get(key)


def store(question: str, answer: str, sources: list) -> None:
    key = normalize_question(question)
    with _lock:
        if key not in _cache and len(_cache) >= _MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))  # evict oldest
        _cache[key] = (answer, sources)


def size() -> int:
    with _lock:
        return len(_cache)
