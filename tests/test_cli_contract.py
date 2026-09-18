"""Contract tests for the CLI surface itself.

These cover failure modes the rest of the suite structurally cannot see:

* Help rendering. Every other CLI test invokes a command with real arguments, which
  never exercises click's usage formatter. typer 0.15.0-0.15.3 against click >= 8.2
  crash on *every* `--help` with
  `TypeError: Parameter.make_metavar() missing 1 required positional argument`, while
  a full green suite reported 256 passed — so the declared floor admitted a build
  whose CLI could not print help.
* Rich markup in error text. Messages naming an extra (`ctxbudgeter[yaml]`) are passed
  through `Console.print`, which parses `[yaml]` as a style tag and drops it, telling
  the user to install the package they already have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ctxbudgeter.cli import app

runner = CliRunner()

COMMANDS = [
    "scan", "compile", "pack", "validate", "report", "scan-risk", "bom", "diff",
    "eval", "cache-plan", "viz", "viz-diff", "mcp-audit", "mcp-select", "mcp-viz",
]


def test_root_help_renders() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "ctxbudgeter" in result.stdout


@pytest.mark.parametrize("command", COMMANDS)
def test_subcommand_help_renders(command: str) -> None:
    """Guards the dependency floor: help formatting is where typer/click break."""
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
    assert result.exception is None
    assert "Usage" in result.stdout or "Options" in result.stdout


def test_every_command_has_a_help_test() -> None:
    """A newly added command must not silently escape help coverage."""
    import typer.main

    click_names = set(typer.main.get_command(app).commands)  # authoritative CLI names
    assert click_names, "no commands registered"
    missing = click_names - set(COMMANDS)
    assert not missing, f"commands with no --help test: {sorted(missing)}"


def test_spec_error_preserves_extras_name(tmp_path: Path, monkeypatch) -> None:
    """`ctxbudgeter[yaml]` must survive rich markup parsing."""
    import ctxbudgeter.spec as spec_mod

    def _boom(path):
        raise spec_mod.SpecError(
            "YAML pack specs require PyYAML. Install with `pip install ctxbudgeter[yaml]`."
        )

    monkeypatch.setattr(spec_mod, "load_pack", _boom)
    f = tmp_path / "pack.yaml"
    f.write_text("model: x\n", encoding="utf-8")
    result = runner.invoke(app, ["pack", str(f)])
    assert result.exit_code == 2
    flat = " ".join(result.output.split())
    assert "ctxbudgeter[yaml]" in flat, f"extras name was swallowed: {flat!r}"


def _snapshot(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "README.md").write_text("# Docs\n\nStable reference content.\n", encoding="utf-8")
    (src / "auth.py").write_text("def login(u, p):\n    return u == p\n", encoding="utf-8")
    out = tmp_path / "compiled.json"
    r = runner.invoke(app, ["compile", str(src), "--task", "fix auth", "--save-pack", str(out)])
    assert r.exit_code == 0, r.output
    return out


def test_bom_from_snapshot_preserves_assembly_order(tmp_path: Path) -> None:
    """Item order is the prompt order, not the scoring order it is stored in."""
    snap = _snapshot(tmp_path)
    raw = json.loads(snap.read_text(encoding="utf-8"))
    expected = raw["included_order"]
    assert [d["name"] for d in raw["decisions"] if d["status"] != "excluded"] != expected, (
        "fixture no longer distinguishes scoring order from assembly order"
    )
    result = runner.invoke(app, ["bom", str(snap), "-f", "json"])
    assert result.exit_code == 0, result.output
    assert [i["name"] for i in json.loads(result.stdout)["included_items"]] == expected


def test_cache_plan_on_snapshot_is_not_all_zeros(tmp_path: Path) -> None:
    """A snapshot has no re-materialized items; the plan must not silently read empty."""
    snap = _snapshot(tmp_path)
    recorded = json.loads(snap.read_text(encoding="utf-8"))["cacheable_prefix_tokens"]
    assert recorded > 0, "fixture no longer produces a cacheable prefix"
    result = runner.invoke(app, ["cache-plan", str(snap), "-f", "json"])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["ordered_segments"], "cache plan listed no segments"
    assert plan["stable_prefix_tokens"] == recorded
    assert plan["cacheable_token_estimate"] == recorded
