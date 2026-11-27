"""Versioned storage for adaptive policies.

Layout: ``<root>/<model-slug>/v<version>-<policy_id>.json``. Versions increase per
model; nothing is overwritten except a policy's own status/validation record. Only a
``validated`` policy can become *active* for a model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .tuner import AdaptivePolicy


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", model.lower()).strip("-") or "model"


class PolicyStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _dir(self, model: str) -> Path:
        return self.root / _slug(model)

    def _path(self, p: AdaptivePolicy) -> Path:
        return self._dir(p.model) / f"v{p.version}-{p.policy_id}.json"

    def history(self, model: str) -> list[AdaptivePolicy]:
        d = self._dir(model)
        if not d.exists():
            return []
        out = [AdaptivePolicy.from_dict(json.loads(f.read_text(encoding="utf-8"))) for f in d.glob("v*.json")]
        return sorted(out, key=lambda p: (p.version, p.created_at))

    def save(self, policy: AdaptivePolicy) -> AdaptivePolicy:
        """Persist ``policy``, assigning the next version number for its model."""
        hist = self.history(policy.model)
        latest = max((p.version for p in hist), default=0)
        if policy.version <= latest or any(p.policy_id == policy.policy_id for p in hist):
            policy.version = latest + 1
        path = self._path(policy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(policy.to_dict(), indent=2, default=str), encoding="utf-8")
        return policy

    def get(self, model: str, policy_id: str) -> AdaptivePolicy | None:
        return next((p for p in self.history(model) if p.policy_id == policy_id), None)

    def latest(self, model: str) -> AdaptivePolicy | None:
        h = self.history(model)
        return h[-1] if h else None

    def active(self, model: str) -> AdaptivePolicy | None:
        """The newest *validated* policy for ``model``."""
        v = [p for p in self.history(model) if p.status == "validated"]
        return v[-1] if v else None

    def record_validation(self, policy: AdaptivePolicy, result: dict[str, Any]) -> AdaptivePolicy:
        """Append a validation result and set status validated/rejected accordingly."""
        old = self._path(policy)
        policy.validation.append(result)
        policy.status = "validated" if result.get("passed") else "rejected"
        old.parent.mkdir(parents=True, exist_ok=True)
        old.write_text(json.dumps(policy.to_dict(), indent=2, default=str), encoding="utf-8")
        return policy
