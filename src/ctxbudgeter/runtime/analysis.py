"""Deterministic text analysis used by the runtime, governor, summarizer and evaluator.

No model calls. The core idea is *salient terms*: the tokens in a piece of context
that carry hard-to-reconstruct information — numbers, identifiers, file paths, URLs,
code symbols, quoted strings and capitalized names. A summary that drops them has
lost information an agent cannot infer back, so salient-term retention is the
working proxy for "critical information retention" throughout this package.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_NUMBER = r"\b\d[\d,]*(?:\.\d+)?%?"
_IDENT = r"\b[A-Z][A-Z0-9]*[-_][A-Z0-9][A-Z0-9_-]*\b"  # ALPHA-7, HTTP_404, FY-2019
_PATH = r"(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+\.\w+|\b[\w-]+\.(?:py|ts|js|md|json|yaml|yml|toml|csv|txt|go|rs|java|sql)\b"
_URL = r"https?://[^\s)>\]]+"
_SYMBOL = r"\b[a-z_][a-z0-9]*(?:_[a-z0-9]+)+\b|\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"  # snake_case, camelCase
_QUOTED = r"`[^`\n]{2,60}`|\"[^\"\n]{2,60}\""
_CAPNAME = r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+\b"  # Multi Word Names

_SALIENT_RE = re.compile("|".join(f"(?:{p})" for p in (_URL, _PATH, _IDENT, _QUOTED, _SYMBOL, _CAPNAME, _NUMBER)))
_WORD_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\(`\"])|\n+")

_STOP_NUMBERS = {"0", "1", "2"}


def salient_terms(text: str) -> set[str]:
    """Return the set of salient terms in ``text`` (normalized, deduplicated)."""
    out: set[str] = set()
    for m in _SALIENT_RE.finditer(text):
        tok = m.group(0).strip("`\"'.,;:()[]")
        if not tok or tok in _STOP_NUMBERS:
            continue
        out.add(tok.replace(",", "") if tok[0].isdigit() else tok)
    return out


def retention(original_terms: set[str], summary: str) -> float:
    """Fraction of ``original_terms`` still present in ``summary`` (1.0 if none)."""
    if not original_terms:
        return 1.0
    kept = salient_terms(summary)
    flat = summary.replace(",", "")
    hit = sum(1 for t in original_terms if t in kept or t in flat)
    return hit / len(original_terms)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences / lines, dropping blanks."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()]


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, k: int = 5) -> set[tuple[str, ...]]:
    w = words(text)
    if len(w) < k:
        return {tuple(w)} if w else set()
    return {tuple(w[i : i + k]) for i in range(len(w) - k + 1)}


def jaccard(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def unique_line_ratio(text: str) -> float:
    """Unique non-blank lines / total non-blank lines. Low values mean repetitive logs."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return 1.0
    return len(set(lines)) / len(lines)


def query_terms(query: str) -> set[str]:
    """Search terms for archive lookup: salient terms plus content words >= 4 chars."""
    terms = {t.lower() for t in salient_terms(query)}
    terms |= {w for w in words(query) if len(w) >= 4}
    return terms


def match_score(terms: Iterable[str], text: str) -> float:
    """Simple deterministic relevance: fraction of query terms found in text."""
    terms = list(terms)
    if not terms:
        return 0.0
    low = text.lower()
    return sum(1 for t in terms if t in low) / len(terms)
