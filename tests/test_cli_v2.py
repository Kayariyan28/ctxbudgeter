"""Tests for the new v0.2 CLI commands: pack, validate, scan --emit-pack."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ctxbudgeter.cli import app

runner = CliRunner()


YAML_SPEC = """
model: claude-sonnet-4.6
token_budget: 24000
reserved_output_tokens: 4000

items:
  - name: system_rules
    content: |
      You are a careful coding agent.
    kind: system
    priority: 100
    required: true
    cache_policy: stable

  - name: task
    content: "Test"
    kind: task
    priority: 95
    required: true
"""


def test_pack_command_text(tmp_path: Path) -> None:
    spec = tmp_path / "pack.yaml"
    spec.write_text(YAML_SPEC)
    result = runner.invoke(app, ["pack", str(spec)])
    assert result.exit_code == 0, result.stdout
    assert "Included:" in result.stdout
    assert "system_rules" in result.stdout
    assert "task" in result.stdout


def test_pack_command_json(tmp_path: Path) -> None:
    spec = tmp_path / "pack.yaml"
    spec.write_text(YAML_SPEC)
    result = runner.invoke(app, ["pack", str(spec), "--format", "json"])
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["model"] == "claude-sonnet-4.6"


def test_pack_fail_below_threshold(tmp_path: Path) -> None:
    spec = tmp_path / "pack.yaml"
    spec.write_text(YAML_SPEC)
    # health is below 200, should fail
    result = runner.invoke(app, ["pack", str(spec), "--fail-below", "200"])
    assert result.exit_code == 1
    assert "below threshold" in result.stdout


def test_pack_fail_below_passing(tmp_path: Path) -> None:
    spec = tmp_path / "pack.yaml"
    spec.write_text(YAML_SPEC)
    result = runner.invoke(app, ["pack", str(spec), "--fail-below", "0"])
    assert result.exit_code == 0


def test_validate_good_spec(tmp_path: Path) -> None:
    spec = tmp_path / "pack.yaml"
    spec.write_text(YAML_SPEC)
    result = runner.invoke(app, ["validate", str(spec)])
    assert result.exit_code == 0
    assert "valid" in result.stdout


def test_validate_bad_spec(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("model: x\nweird_root_key: 1\n")
    result = runner.invoke(app, ["validate", str(bad)])
    assert result.exit_code == 1


def test_scan_emit_pack(tmp_path: Path) -> None:
    # Set up a small repo
    (tmp_path / "README.md").write_text("# Demo")
    (tmp_path / "main.py").write_text("print('hi')\n")
    out_pack = tmp_path / "pack.yaml"
    result = runner.invoke(
        app,
        ["scan", str(tmp_path), "--emit-pack", str(out_pack), "--task", "Fix the bug"],
    )
    assert result.exit_code == 0
    assert out_pack.exists()
    text = out_pack.read_text(encoding="utf-8")
    assert "model:" in text
    assert "items:" in text
    assert "README.md" in text
    assert "Fix the bug" in text


def test_scan_emit_pack_then_load(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo")
    out_pack = tmp_path / "pack.yaml"
    result = runner.invoke(
        app,
        ["scan", str(tmp_path), "--emit-pack", str(out_pack), "--task", "fix"],
    )
    assert result.exit_code == 0
    # Load the emitted spec and compile
    result2 = runner.invoke(app, ["pack", str(out_pack)])
    assert result2.exit_code == 0, result2.stdout
    assert "task" in result2.stdout


def test_compile_secret_policy_refuse(tmp_path: Path) -> None:
    """Secret policy passthrough from CLI is not strictly tested via files,
    but we can verify the flag is parsed."""
    (tmp_path / "README.md").write_text("# Demo")
    result = runner.invoke(
        app,
        ["compile", str(tmp_path), "--task", "X", "--budget", "5000", "--secret-policy", "redact"],
    )
    assert result.exit_code == 0
