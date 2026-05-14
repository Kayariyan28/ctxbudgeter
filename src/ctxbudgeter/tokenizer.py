"""Token counting with optional tiktoken backend and a deterministic heuristic fallback.

The fallback never raises and is good enough for budgeting (within ~10-15% of true token
counts for English/code text). For accurate counts on OpenAI models, install with the
[tiktoken] extra. For Anthropic models, tiktoken's cl100k_base is used as a proxy — the
official Anthropic tokenizer requires an API call, which we deliberately avoid in core.
"""

from __future__ import annotations

import math
from typing import Optional

# Model-name → tiktoken encoding mapping. Prefix match is used as fallback.
_TIKTOKEN_ENCODINGS: dict[str, str] = {
    # OpenAI
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-4.1-mini": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "o1": "o200k_base",
    "o1-mini": "o200k_base",
    "o3": "o200k_base",
    "o3-mini": "o200k_base",
    "o4": "o200k_base",
    "o4-mini": "o200k_base",
    # Anthropic — no official tiktoken support; cl100k_base is a reasonable proxy.
    "claude-": "cl100k_base",
    # Generic fallback prefixes
    "text-embedding-3": "cl100k_base",
}


def _resolve_encoding(model: str) -> str:
    model_lc = model.lower()
    if model_lc in _TIKTOKEN_ENCODINGS:
        return _TIKTOKEN_ENCODINGS[model_lc]
    for prefix, encoding in _TIKTOKEN_ENCODINGS.items():
        if model_lc.startswith(prefix):
            return encoding
    if model_lc.startswith(("gpt-4o", "gpt-4.1", "o1", "o3", "o4")):
        return "o200k_base"
    return "cl100k_base"


class TokenCounter:
    """Count tokens for a model, preferring tiktoken when installed.

    Usage:
        counter = TokenCounter("claude-sonnet-4.6")
        n = counter.count("hello world")
    """

    __slots__ = ("model", "_encoder", "_using_tiktoken")

    def __init__(self, model: str = "claude-sonnet-4.6") -> None:
        self.model = model
        self._encoder: Optional[object] = None
        self._using_tiktoken = False
        try:
            import tiktoken  # type: ignore[import-not-found]

            self._encoder = tiktoken.get_encoding(_resolve_encoding(model))
            self._using_tiktoken = True
        except Exception:
            self._encoder = None
            self._using_tiktoken = False

    @property
    def backend(self) -> str:
        return "tiktoken" if self._using_tiktoken else "heuristic"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))  # type: ignore[attr-defined]
            except Exception:
                pass
        return self._heuristic_count(text)

    @staticmethod
    def _heuristic_count(text: str) -> int:
        """Conservative deterministic estimator.

        Uses ceil((chars + words/2) / 4). Slightly overestimates vs. true BPE for
        English/code (good — we'd rather reserve a bit more headroom than blow the
        context window).
        """
        chars = len(text)
        words = len(text.split())
        approx = (chars + words / 2) / 4.0
        return max(1, math.ceil(approx))
