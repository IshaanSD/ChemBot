"""Example questions shown in the UI.

These double as the pre-warm set: at startup the server can run them once to
populate the answer cache (see PREWARM_EXAMPLES), so clicking an example is a
free cache hit that works even after the daily live cap is reached.
"""

from __future__ import annotations

from typing import List

EXAMPLE_QUESTIONS: List[str] = [
    "What are the contraindications for ibuprofen?",
    "What adverse events have been reported for Ozempic?",
    "What recent studies exist on semaglutide and Alzheimer's disease?",
    "Is ibuprofen safe during pregnancy?",
    "Is semaglutide associated with pancreatitis?",
]


def get_examples() -> List[str]:
    return list(EXAMPLE_QUESTIONS)
