"""
MONO, CS, AVG configs as specified in the paper.
Initialize the model using ``ILP4LID.from_config("mono")``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

CONFIGS: Dict[str, Dict[str, Any]] = {
    # best for code-switched sentences, likely to overpredict labels on monolingual data
    "cs": {
        "constraints": set(),
        "penalty_for_another_lang": 0.0,
        "k_weights": [1, 1],
    },
    # best for maximizing code-switching precision over recall
    "mono": {
        "constraints": {"top_alpha_m2", "min_len"},
        "penalty_for_another_lang": 20.0,
        "min_chars": 10,
        "k_weights": [1, 0.7],
    },
    # recommended config
    "avg": {
        "constraints": {"top_alpha_m2", "min_len"},
        "penalty_for_another_lang": 15.0,
        "k_weights": [1, 0.75],
    },
}


def resolve_config(name: Optional[str]) -> Dict[str, Any]:
    """The preset's `ILP4LID(...)` keyword arguments; falls back to "avg" if `name` is unknown."""
    if name not in CONFIGS:
        print(f"unknown config {name!r}; defaulting to 'avg'")
        name = "avg"
    return dict(CONFIGS[name])
