"""Tests for the `bom` CLI command.

Regression coverage for GitHub issue #8: `ctxbudgeter bom` crashed with a TypeError on
compiled-pack input because the advisory note was printed via `Console.print(stderr=True)`,
which rich has never accepted — the stream is chosen when the Console is constructed.
"""

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
    content: "Fix the auth bug."
    kind: task
    priority: 95
    required: true
"""


def _compiled_pack(tmp_path: Path) -> Path:
    """Produce a compiled-pack snapshot the way the issue reporter did."""
    spec = tmp_path / "pack.yaml"
    spec.write_text(YAML_SPEC, encoding="utf-8")
    saved = tmp_path / "pack-out.json"
    result = runner.invoke(app, ["pack", str(spec), "--save-pack", str(saved)])
    assert result.exit_code == 0, result.stdout
    assert "decisions" in json.loads(saved.read_text(encoding="utf-8"))
    return saved


def _bom_file(tmp_path: Path) -> Path:
    """Produce a full-fidelity BOM via `compile --bom`."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')\n", encoding="utf-8")
    out = tmp_path / "context_bom.json"
    result = runner.invoke(
        app, ["compile", str(src), "--task", "Fix the auth bug.", "--bom", str(out)]
    )
    assert result.exit_code == 0, result.stdout
    return out


def test_bom_from_compiled_pack_markdown(tmp_path: Path) -> None:
    """The exact reproduction from issue #8 — must not raise TypeError."""
    saved = _compiled_pack(tmp_path)
    result = runner.invoke(app, ["bom", str(saved), "-f", "markdown"])
    assert result.exit_code == 0, result.stdout
    assert result.exception is None
    assert "# Context Bill of Materials" in result.stdout


def test_bom_from_compiled_pack_json_stdout_is_parseable(tmp_path: Path) -> None:
    """The advisory note must not contaminate stdout, or `-f json` is unpipeable."""
    saved = _compiled_pack(tmp_path)
    result = runner.invoke(app, ["bom", str(saved), "-f", "json"])
    assert result.exit_code == 0, result.stdout
    parsed = json.loads(result.stdout)
    assert parsed["schema_version"]
    assert "note:" not in result.stdout


def test_bom_compiled_pack_note_goes_to_stderr(tmp_path: Path) -> None:
    saved = _compiled_pack(tmp_path)
    result = runner.invoke(app, ["bom", str(saved), "-f", "json"])
    assert result.exit_code == 0
    assert "compiled-pack snapshot" in result.stderr


def test_bom_from_bom_json_emits_no_note(tmp_path: Path) -> None:
    """Full-fidelity BOM input is not degraded, so it gets no advisory note."""
    bom_path = _bom_file(tmp_path)
    result = runner.invoke(app, ["bom", str(bom_path), "-f", "json"])
    assert result.exit_code == 0, result.stdout
    json.loads(result.stdout)
    assert "compiled-pack snapshot" not in result.stderr


def test_bom_writes_to_out_file(tmp_path: Path) -> None:
    saved = _compiled_pack(tmp_path)
    out = tmp_path / "bom.md"
    result = runner.invoke(app, ["bom", str(saved), "-f", "markdown", "-o", str(out)])
    assert result.exit_code == 0, result.stdout
    assert "# Context Bill of Materials" in out.read_text(encoding="utf-8")


def test_bom_missing_file_exits_cleanly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["bom", str(tmp_path / "nope.json")])
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_bom_unrecognized_input_exits_cleanly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    result = runner.invoke(app, ["bom", str(bad)])
    assert result.exit_code == 2
