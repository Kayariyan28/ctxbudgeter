"""Smoke tests for the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ctxbudgeter.cli import app

runner = CliRunner()


def _write_sample_repo(root: Path) -> None:
    (root / "README.md").write_text("# Demo project\nHello.")
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("def main():\n    print('hello')\n")
    (root / "src" / "huge.py").write_text("# stub\n" + ("x = 1\n" * 5000))
    (root / "debug.log").write_text("garbage" * 5000)
    (root / "node_modules").mkdir()
    (root / "node_modules" / "skip.js").write_text("should not be scanned")


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "ctxbudgeter" in result.stdout


def test_scan_basic(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "README.md" in result.stdout
    assert "main.py" in result.stdout
    # ignore patterns honored
    assert "skip.js" not in result.stdout


def test_scan_writes_inventory(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    out = tmp_path / "inv.json"
    result = runner.invoke(app, ["scan", str(tmp_path), "--output", str(out)])
    assert result.exit_code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["total_tokens"] > 0
    assert any(f["relative_path"].endswith("README.md") for f in data["files"])


def test_compile_text_format(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    result = runner.invoke(
        app,
        ["compile", str(tmp_path), "--task", "Fix the bug", "--budget", "5000"],
    )
    assert result.exit_code == 0, result.stdout
    assert "Included:" in result.stdout
    assert "task" in result.stdout
    assert "health score" in result.stdout.lower()


def test_compile_json_format(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    result = runner.invoke(
        app,
        ["compile", str(tmp_path), "--task", "X", "--budget", "5000", "--format", "json"],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed["health_score"] >= 0


def test_compile_markdown_format(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    result = runner.invoke(
        app,
        ["compile", str(tmp_path), "--task", "X", "--budget", "5000", "--format", "markdown"],
    )
    assert result.exit_code == 0
    assert "# Context Compilation Report" in result.stdout


def test_compile_with_save_pack_and_report_roundtrip(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    saved = tmp_path / "pack.json"
    result = runner.invoke(
        app,
        [
            "compile",
            str(tmp_path),
            "--task",
            "X",
            "--budget",
            "5000",
            "--save-pack",
            str(saved),
            "--format",
            "text",
        ],
    )
    assert result.exit_code == 0
    assert saved.exists()
    result2 = runner.invoke(app, ["report", str(saved)])
    assert result2.exit_code == 0
    assert "Included:" in result2.stdout

    result3 = runner.invoke(app, ["report", str(saved), "--format", "markdown"])
    assert result3.exit_code == 0
    assert "# Context Compilation Report" in result3.stdout


def test_compile_invalid_format(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    result = runner.invoke(
        app,
        ["compile", str(tmp_path), "--task", "X", "--budget", "5000", "--format", "xml"],
    )
    assert result.exit_code == 2


def test_compile_with_system_file(tmp_path: Path) -> None:
    _write_sample_repo(tmp_path)
    sys_file = tmp_path / "system.txt"
    sys_file.write_text("You are a careful agent.")
    result = runner.invoke(
        app,
        [
            "compile",
            str(tmp_path),
            "--task",
            "Y",
            "--budget",
            "5000",
            "--system",
            str(sys_file),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    names = [d["name"] for d in parsed["decisions"]]
    assert "system_rules" in names


def test_scan_nonexistent_path() -> None:
    result = runner.invoke(app, ["scan", "/definitely/does/not/exist/__"])
    assert result.exit_code == 2
