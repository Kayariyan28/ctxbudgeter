"""Local-first PII & secret scanner.

Detects emails, phone-like values, credit-card-like numbers (Luhn-validated),
JWTs, common API-key shapes (OpenAI, Anthropic, GitHub, AWS, Google, Slack,
Stripe), private-key blocks, .env-style assignments, secret-ish variable
assignments, and URLs with embedded credentials.

SECURITY INVARIANT
------------------
The scanner NEVER emits a full secret. Every ``matched_preview`` is masked:
only a short, non-reconstructable fingerprint of the match is shown
(e.g. ``sk-...Ab3``). ``redact()`` replaces matches in text with safe tokens.

This is a deterministic regex/heuristic engine. It runs offline, makes no
network calls, and is NOT a compliance certification. Treat it as a useful
guardrail, not proof of safety.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Severity = Literal["low", "medium", "high", "critical"]

_SEVERITY_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2, "critical": 3}
RiskLevel = Literal["none", "low", "medium", "high", "critical"]


@dataclass(frozen=True)
class Finding:
    """A single scanner hit. ``matched_preview`` is always masked."""

    category: str
    severity: Severity
    matched_preview: str
    recommendation: str
    start: int | None = None
    end: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    """Result of scanning a piece of text."""

    findings: list[Finding] = field(default_factory=list)
    text_length: int = 0

    @property
    def has_pii(self) -> bool:
        return any(f.category in _PII_CATEGORIES for f in self.findings)

    @property
    def has_secrets(self) -> bool:
        return any(f.category in _SECRET_CATEGORIES for f in self.findings)

    @property
    def risk_level(self) -> RiskLevel:
        if not self.findings:
            return "none"
        worst = max(_SEVERITY_ORDER[f.severity] for f in self.findings)
        return ("low", "medium", "high", "critical")[worst]  # type: ignore[return-value]

    @property
    def risk_score(self) -> int:
        """0-100 deterministic risk score from finding severities."""
        if not self.findings:
            return 0
        weights = {"low": 5, "medium": 15, "high": 30, "critical": 50}
        raw = sum(weights[f.severity] for f in self.findings)
        return min(100, raw)

    masked_text: str = ""

    def to_dict(self) -> dict:
        return {
            "has_pii": self.has_pii,
            "has_secrets": self.has_secrets,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "text_length": self.text_length,
            "findings": [f.to_dict() for f in self.findings],
        }


# ----- masking ----------------------------------------------------------------


def _mask(value: str, *, keep_prefix: int = 0, keep_suffix: int = 3) -> str:
    """Mask a secret/PII value so it can never be reconstructed.

    Shows up to ``keep_prefix`` leading chars and ``keep_suffix`` trailing chars,
    with the middle replaced by ``...``. For very short values, shows nothing but
    a fixed marker. This is the single choke-point that guarantees no full secret
    ever leaves the scanner.
    """
    value = value.strip()
    n = len(value)
    if n == 0:
        return "[empty]"
    if n <= 4:
        return "****"
    prefix = value[:keep_prefix] if keep_prefix and n > keep_prefix + keep_suffix else ""
    suffix = value[-keep_suffix:] if keep_suffix and n > keep_suffix + 2 else ""
    return f"{prefix}...{suffix}" if (prefix or suffix) else "****"


def _mask_keep_scheme(value: str, scheme_len: int, keep_suffix: int = 3) -> str:
    """Mask but keep a known key-scheme prefix (e.g. 'sk-', 'ghp_', 'AKIA')."""
    value = value.strip()
    scheme = value[:scheme_len]
    tail = value[-keep_suffix:] if len(value) > scheme_len + keep_suffix + 1 else ""
    return f"{scheme}...{tail}" if tail else f"{scheme}..."


# ----- detectors ---------------------------------------------------------------

_PII_CATEGORIES = {"email", "phone", "credit_card"}
_SECRET_CATEGORIES = {
    "jwt",
    "api_key_generic",
    "openai_key",
    "anthropic_key",
    "github_token",
    "aws_access_key_id",
    "google_api_key",
    "slack_token",
    "stripe_key",
    "private_key",
    "env_assignment",
    "secret_variable",
    "url_with_credentials",
}


@dataclass(frozen=True)
class _Detector:
    category: str
    severity: Severity
    pattern: re.Pattern[str]
    recommendation: str
    masker: str = "default"  # "default" | "scheme:N" | "block"
    luhn: bool = False


def _build_detectors() -> list[_Detector]:
    d: list[_Detector] = []

    d.append(_Detector(
        "private_key", "critical",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
        "Remove private key material from context. Never send keys to a model.",
        masker="block",
    ))
    # Anthropic is listed BEFORE OpenAI so 'sk-ant-...' is labeled anthropic
    # (both are critical; ordered detection is stable, so the first to claim the
    # span wins and the OpenAI pattern's overlapping match is deduped away).
    d.append(_Detector(
        "anthropic_key", "critical",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b"),
        "Revoke and rotate this Anthropic key; exclude it from context.",
        masker="scheme:7",
    ))
    d.append(_Detector(
        "openai_key", "critical",
        re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}\b"),
        "Revoke and rotate this OpenAI-style key; exclude it from context.",
        masker="scheme:3",
    ))
    d.append(_Detector(
        "github_token", "critical",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{16,}\b"),
        "Revoke this GitHub token immediately; exclude it from context.",
        masker="scheme:4",
    ))
    d.append(_Detector(
        "aws_access_key_id", "critical",
        re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{16}\b"),
        "Rotate this AWS access key; exclude it from context.",
        masker="scheme:4",
    ))
    d.append(_Detector(
        "google_api_key", "high",
        re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
        "Rotate this Google API key; exclude it from context.",
        masker="scheme:4",
    ))
    d.append(_Detector(
        "slack_token", "high",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}\b"),
        "Revoke this Slack token; exclude it from context.",
        masker="scheme:5",
    ))
    d.append(_Detector(
        "stripe_key", "critical",
        re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"),
        "Rotate this Stripe key; exclude it from context.",
        masker="scheme:8",
    ))
    d.append(_Detector(
        "jwt", "high",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        "JWTs can carry credentials/claims. Strip or redact before sending.",
        masker="scheme:6",
    ))
    d.append(_Detector(
        "url_with_credentials", "high",
        re.compile(r"\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@]+:[^\s/:@]+@[^\s]+"),
        "URL contains inline credentials. Move secrets to a secret store.",
        masker="default",
    ))
    d.append(_Detector(
        "private_key_pem_body", "critical",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]{0,4096}?-----END [A-Z ]*PRIVATE KEY-----"),
        "Remove the full private key block from context.",
        masker="block",
    ))
    # Generic credential-looking assignments: API_KEY=..., password: "...", token = '...'
    d.append(_Detector(
        "env_assignment", "high",
        re.compile(
            r"(?im)^\s*[A-Z0-9_]*(?:SECRET|TOKEN|API[_-]?KEY|PASSWORD|PASSWD|PRIVATE[_-]?KEY|ACCESS[_-]?KEY)[A-Z0-9_]*\s*=\s*['\"]?([^\s'\"]{6,})['\"]?"
        ),
        "Looks like a secret assignment. Exclude .env-style values from context.",
        masker="default",
    ))
    d.append(_Detector(
        "secret_variable", "medium",
        re.compile(
            r"(?i)\b(?:secret|token|api[_-]?key|password|passwd|client[_-]?secret)\b\s*[:=]\s*['\"]([^'\"]{6,})['\"]"
        ),
        "A secret-looking variable was found. Confirm it is not sensitive.",
        masker="default",
    ))
    d.append(_Detector(
        "api_key_generic", "medium",
        re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),
        "Long high-entropy token. Verify this is not a credential before sending.",
        masker="default",
    ))
    d.append(_Detector(
        "email", "low",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "Email address (PII). Mask or remove if not required.",
        masker="default",
    ))
    d.append(_Detector(
        "phone", "low",
        re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?![\w.])"),
        "Phone-like value (PII). Mask or remove if not required.",
        masker="default",
    ))
    d.append(_Detector(
        "credit_card", "high",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "Credit-card-like number (PCI). Remove from context.",
        masker="default", luhn=True,
    ))
    return d


_DETECTORS = _build_detectors()


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not (13 <= len(nums) <= 19):
        return False
    total = 0
    parity = len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _preview_for(det: _Detector, matched: str) -> str:
    if det.masker == "block":
        return "-----BEGIN PRIVATE KEY----- [masked]"
    if det.masker.startswith("scheme:"):
        return _mask_keep_scheme(matched, int(det.masker.split(":", 1)[1]))
    return _mask(matched)


class ContextScanner:
    """Deterministic, local PII/secret scanner.

    Usage::

        scanner = ContextScanner()
        result = scanner.scan(text)
        if result.has_secrets:
            safe = scanner.redact(text)
    """

    def __init__(self, *, detectors: list[_Detector] | None = None) -> None:
        self._detectors = detectors if detectors is not None else _DETECTORS

    def scan(self, text: str) -> ScanResult:
        """Scan text and return a ScanResult. Never returns unmasked secrets."""
        if not isinstance(text, str):
            text = str(text)
        findings: list[Finding] = []
        claimed: list[tuple[int, int]] = []  # (start, end) of higher-severity hits

        # Order detectors so specific/high-severity ones claim spans first; the
        # generic api_key_generic detector then skips already-claimed regions.
        ordered = sorted(
            self._detectors, key=lambda d: -_SEVERITY_ORDER[d.severity]
        )
        for det in ordered:
            for m in det.pattern.finditer(text):
                full = m.group(0)
                # the capturing group (if any) is the sensitive value
                value = m.group(1) if m.groups() else full
                start, end = m.start(), m.end()
                if det.luhn and not _luhn_ok(full):
                    continue
                if det.category == "api_key_generic" and _overlaps(start, end, claimed):
                    continue
                if det.category == "phone" and _looks_like_year_range(full):
                    continue
                findings.append(Finding(
                    category=det.category,
                    severity=det.severity,
                    matched_preview=_preview_for(det, value),
                    recommendation=det.recommendation,
                    start=start,
                    end=end,
                ))
                if _SEVERITY_ORDER[det.severity] >= 1:
                    claimed.append((start, end))

        findings = _dedupe(findings)
        result = ScanResult(findings=findings, text_length=len(text))
        result.masked_text = self._redact_with(text, findings)
        return result

    def redact(self, text: str) -> str:
        """Return ``text`` with detected secrets/PII replaced by safe tokens."""
        return self.scan(text).masked_text

    def scan_file(self, path: str | Path, *, encoding: str = "utf-8") -> ScanResult:
        """Scan a text file. Reads with errors='replace' so binary-ish files
        don't crash the scan."""
        p = Path(path)
        text = p.read_text(encoding=encoding, errors="replace")
        return self.scan(text)

    # ----- internal -----

    def _redact_with(self, text: str, findings: list[Finding]) -> str:
        # Replace from the end so offsets stay valid.
        spans = sorted(
            [(f.start, f.end, f.category) for f in findings if f.start is not None and f.end is not None],
            key=lambda s: s[0],
            reverse=True,
        )
        out = text
        for start, end, category in spans:
            token = f"[REDACTED:{category}]"
            out = out[:start] + token + out[end:]
        return out


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(not (end <= s or start >= e) for s, e in spans)


def _looks_like_year_range(value: str) -> bool:
    stripped = value.strip()
    return bool(re.fullmatch("\\d{4}\\s*[-\u2013]\\s*\\d{4}", stripped))


def _dedupe(findings: list[Finding]) -> list[Finding]:
    """Drop exact-span duplicates, keeping the highest-severity category."""
    best: dict[tuple[int | None, int | None], Finding] = {}
    extra: list[Finding] = []
    for f in findings:
        key = (f.start, f.end)
        if f.start is None:
            extra.append(f)
            continue
        cur = best.get(key)
        if cur is None or _SEVERITY_ORDER[f.severity] > _SEVERITY_ORDER[cur.severity]:
            best[key] = f
    merged = [*best.values(), *extra]
    merged.sort(key=lambda f: (f.start if f.start is not None else 1 << 30, f.category))
    return merged
