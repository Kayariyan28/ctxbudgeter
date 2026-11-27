"""Typer CLI: `scan`, `compile`, `report`, `pack`, `validate`.

The CLI is a thin layer over ContextPack — useful for Claude Code, ad-hoc audits,
CI checks, and YAML-driven workflows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .item import CachePolicy, ContextKind
from .pack import ContextPack
from .report import to_json, to_markdown, to_text
from .tokenizer import TokenCounter


def _harden_stream_encoding() -> None:
    """Let stdout/stderr survive non-ASCII on legacy consoles (e.g. Windows cp1252).

    The CLI emits ✓/✗/•/— in human-facing output, and Markdown payloads carry them too.
    On a console whose encoding cannot represent those characters — cp1252 is still the
    default on many Windows setups — an unhardened stream raises UnicodeEncodeError
    mid-render. Prefer UTF-8; if the stream refuses to be reconfigured, at least degrade
    unmappable characters instead of crashing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # pragma: no cover — not a TextIOWrapper (e.g. captured)
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover — stream refuses reconfiguration
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


_harden_stream_encoding()

app = typer.Typer(
    add_completion=False,
    help="ctxbudgeter — compile clean, cheap, auditable context for AI agents.",
)
console = Console()
# Advisory notes go to stderr so stdout stays a clean, pipeable payload for `-f json`
# and `-f markdown`. rich takes the stream at construction — Console.print() has no
# `stderr=`/`file=` keyword.
err_console = Console(stderr=True)

DEFAULT_IGNORE = [
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".idea", ".vscode", ".DS_Store",
]

BINARY_SUFFIXES = {
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".webp", ".bmp",
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z",
    ".mp3", ".mp4", ".mov", ".wav", ".webm",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
}


@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"ctxbudgeter {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def scan(
    path: Path = typer.Argument(Path("."), help="Directory to scan."),
    model: str = typer.Option("claude-sonnet-4.6", "--model", "-m"),
    max_files: int = typer.Option(200, "--max-files"),
    ignore: list[str] = typer.Option(DEFAULT_IGNORE, "--ignore"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write JSON inventory."),
    emit_pack: Path | None = typer.Option(
        None, "--emit-pack",
        help="Write a starter pack.yaml spec for the scanned files.",
    ),
    task: str | None = typer.Option(
        None, "--task", "-t",
        help="If --emit-pack is set, include this as the required task item.",
    ),
    budget: int = typer.Option(24_000, "--budget", "-b", help="Budget to write into emitted pack.yaml."),
) -> None:
    """Walk a directory, count tokens, optionally emit a starter pack.yaml."""
    if not path.exists():
        console.print(f"[red]error:[/red] path does not exist: {path}")
        raise typer.Exit(code=2)

    counter = TokenCounter(model)
    files = _walk(path, set(ignore), max_files)

    table = Table(title=f"scan: {path}  •  tokenizer: {counter.backend}")
    table.add_column("File", overflow="fold")
    table.add_column("Kind")
    table.add_column("Tokens", justify="right")
    table.add_column("Priority", justify="right")
    table.add_column("Cache")

    inventory: list[dict[str, Any]] = []
    total = 0
    for fpath in files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        toks = counter.count(text)
        total += toks
        kind, prio, cache = _classify(fpath)
        rel = _safe_rel(fpath, path)
        table.add_row(str(rel), kind, f"{toks:,}", str(prio), cache)
        inventory.append({
            "path": str(fpath),
            "relative_path": str(rel),
            "kind": kind,
            "tokens": toks,
            "suggested_priority": prio,
            "suggested_cache_policy": cache,
        })

    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {total:,} tokens across {len(files)} files  "
        f"•  Tokenizer: {counter.backend}"
    )
    if output is not None:
        output.write_text(json.dumps({
            "path": str(path),
            "model": model,
            "tokenizer_backend": counter.backend,
            "total_tokens": total,
            "files": inventory,
        }, indent=2), encoding="utf-8")
        console.print(f"[green]Inventory written:[/green] {output}")
    if emit_pack is not None:
        _emit_pack_yaml(emit_pack, inventory, path, model, budget, task)
        console.print(f"[green]Pack spec written:[/green] {emit_pack}")


@app.command("compile")
def compile_cmd(
    path: Path = typer.Argument(Path("."), help="Directory to gather context from."),
    task: str = typer.Option(..., "--task", "-t"),
    budget: int = typer.Option(24_000, "--budget", "-b"),
    reserved_output: int = typer.Option(4_000, "--reserved-output"),
    model: str = typer.Option("claude-sonnet-4.6", "--model", "-m"),
    system: Path | None = typer.Option(None, "--system", "-s"),
    ignore: list[str] = typer.Option(DEFAULT_IGNORE, "--ignore"),
    max_files: int = typer.Option(200, "--max-files"),
    format: str = typer.Option("text", "--format", "-f", help="text | markdown | json."),
    output: Path | None = typer.Option(None, "--output", "-o"),
    prompt_output: Path | None = typer.Option(None, "--prompt-output"),
    save_pack: Path | None = typer.Option(None, "--save-pack"),
    secret_policy: str = typer.Option(
        "warn", "--secret-policy",
        help="How to handle sensitivity='secret' items: allow | warn | refuse | redact.",
    ),
    policy_file: Path | None = typer.Option(
        None, "--policy", help="ContextPolicy YAML/JSON (ctxbudgeter.yaml). Governs the compile."
    ),
    bom_out: Path | None = typer.Option(
        None, "--bom", help="Write a Context Bill of Materials JSON to this path."
    ),
    report_out: Path | None = typer.Option(
        None, "--report", help="Write a Markdown context report to this path."
    ),
) -> None:
    """Compile context from a directory + task, print a report."""
    if not path.exists():
        console.print(f"[red]error:[/red] path does not exist: {path}")
        raise typer.Exit(code=2)
    if format not in ("text", "markdown", "json"):
        console.print(f"[red]error:[/red] format must be text | markdown | json (got {format!r})")
        raise typer.Exit(code=2)

    policy = None
    if policy_file is not None:
        from .policy import ContextPolicy

        if not policy_file.exists():
            console.print(f"[red]error:[/red] policy file not found: {policy_file}")
            raise typer.Exit(code=2)
        policy = ContextPolicy.from_yaml(str(policy_file))

    if policy is not None:
        pack = ContextPack(model=model, policy=policy)
    else:
        pack = ContextPack(model=model, token_budget=budget, reserved_output_tokens=reserved_output)
    pack.set_secret_policy(secret_policy)
    if system is not None:
        pack.add(
            name="system_rules", content=system.read_text(encoding="utf-8"),
            kind="system", priority=100, required=True, cache_policy="stable",
        )
    pack.add(name="task", content=task, kind="task", priority=95, required=True, cache_policy="dynamic")

    for f in _walk(path, set(ignore), max_files):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if not text.strip():
            continue
        kind, prio, cache = _classify(f)
        item_name = str(_safe_rel(f, path))
        if item_name in pack:
            continue
        try:
            pack.add(
                name=item_name, content=text, kind=kind, priority=prio,
                cache_policy=cache, source=str(f), compressible=True,
            )
        except ValueError:
            continue

    compiled = pack.compile(task=task)
    rendered = (
        to_markdown(compiled) if format == "markdown"
        else to_json(compiled) if format == "json"
        else to_text(compiled)
    )
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
        console.print(f"[green]Report written:[/green] {output}")
    else:
        sys.stdout.write(rendered + "\n")

    if prompt_output is not None:
        prompt_output.write_text(compiled.as_text(), encoding="utf-8")
        console.print(f"[green]Prompt written:[/green] {prompt_output}")
    if save_pack is not None:
        save_pack.write_text(to_json(compiled), encoding="utf-8")
        console.print(f"[green]Pack saved:[/green] {save_pack}")
    if bom_out is not None:
        compiled.bom.to_json(str(bom_out))
        console.print(f"[green]BOM written:[/green] {bom_out}")
    if report_out is not None:
        compiled.bom.to_markdown(str(report_out))
        console.print(f"[green]Report (markdown) written:[/green] {report_out}")


@app.command()
def pack(
    spec: Path = typer.Argument(..., help="Path to a pack.yaml / pack.toml / pack.json."),
    format: str = typer.Option("text", "--format", "-f"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    save_pack: Path | None = typer.Option(None, "--save-pack"),
    fail_below: int | None = typer.Option(
        None, "--fail-below",
        help="Exit non-zero if health_score is below this threshold (for CI).",
    ),
) -> None:
    """Compile a pack from a declarative YAML/TOML/JSON spec file."""
    from .spec import SpecError, load_pack

    if not spec.exists():
        console.print(f"[red]error:[/red] pack spec does not exist: {spec}")
        raise typer.Exit(code=2)
    try:
        p = load_pack(spec)
    except SpecError as e:
        # escape(): the message names extras like `ctxbudgeter[yaml]`, and rich would
        # parse `[yaml]` as a style tag and silently drop it — telling the user to
        # install the package they already have.
        console.print(f"[red]spec error:[/red] {escape(str(e))}")
        raise typer.Exit(code=2) from e

    compiled = p.compile()
    rendered = (
        to_markdown(compiled) if format == "markdown"
        else to_json(compiled) if format == "json"
        else to_text(compiled)
    )
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
        console.print(f"[green]Report written:[/green] {output}")
    else:
        sys.stdout.write(rendered + "\n")
    if save_pack is not None:
        save_pack.write_text(to_json(compiled), encoding="utf-8")
        console.print(f"[green]Pack saved:[/green] {save_pack}")
    if fail_below is not None and compiled.health_score < fail_below:
        console.print(
            f"[red]health_score {compiled.health_score} is below threshold {fail_below}[/red]"
        )
        raise typer.Exit(code=1)


@app.command()
def validate(
    spec: Path = typer.Argument(..., help="Pack spec file to validate."),
) -> None:
    """Validate a pack spec file without compiling."""
    from .spec import validate as _validate

    issues = _validate(spec)
    if issues:
        for i in issues:
            console.print(f"[red]✗[/red] {escape(str(i))}")
        raise typer.Exit(code=1)
    console.print(f"[green]✓[/green] {spec} is valid")


@app.command()
def report(
    pack_json: Path = typer.Argument(..., help="Saved compiled-pack JSON."),
    format: str = typer.Option("text", "--format", "-f"),
) -> None:
    """Render a saved compiled-pack JSON as text or markdown."""
    if not pack_json.exists():
        console.print(f"[red]error:[/red] pack file does not exist: {pack_json}")
        raise typer.Exit(code=2)
    data = json.loads(pack_json.read_text(encoding="utf-8"))
    if format == "json":
        sys.stdout.write(json.dumps(data, indent=2) + "\n")
        return
    rendered = _render_from_dict(data, markdown=(format == "markdown"))
    sys.stdout.write(rendered + "\n")


# ----- helpers ----------------------------------------------------------------


def _safe_rel(p: Path, root: Path) -> Path:
    try:
        return p.relative_to(root)
    except ValueError:
        return p


def _walk(root: Path, ignore: set[str], max_files: int) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = set(p.parts)
        if parts & ignore:
            continue
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size > 2_000_000:
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out


def _classify(path: Path) -> tuple[ContextKind, int, CachePolicy]:
    name = path.name.lower()
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    if name in {"readme.md", "readme.rst", "readme.txt", "readme"}:
        return ("project_doc", 80, "stable")
    if name in {"claude.md", "agents.md", "agent.md"}:
        return ("system", 90, "stable")
    if name in {"contributing.md", "architecture.md", "design.md"}:
        return ("project_doc", 70, "stable")
    if suffix == ".md":
        return ("project_doc", 60, "stable")
    if "test" in stem or "test" in path.parent.name.lower():
        return ("code", 30, "stable")
    if suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".kt",
                  ".cpp", ".cc", ".c", ".h", ".hpp", ".rb", ".php", ".swift", ".m", ".scala"}:
        return ("code", 50, "stable")
    if suffix in {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg"}:
        return ("data", 40, "stable")
    if suffix in {".html", ".css", ".scss", ".sass"}:
        return ("code", 35, "stable")
    if suffix in {".log", ".tmp"}:
        return ("data", 10, "ephemeral")
    return ("other", 30, "dynamic")


def _emit_pack_yaml(
    out_path: Path,
    inventory: list[dict[str, Any]],
    root: Path,
    model: str,
    budget: int,
    task: str | None,
) -> None:
    """Write a starter pack.yaml from a scan inventory."""
    lines: list[str] = []
    lines.append(f"# Generated by ctxbudgeter scan {root}")
    lines.append("# Adjust priorities / cache_policy / required fields before use.")
    lines.append(f"model: {model}")
    lines.append(f"token_budget: {budget}")
    lines.append("reserved_output_tokens: 4000")
    lines.append("secret_policy: warn")
    lines.append("")
    lines.append("items:")
    if task:
        lines.append("  - name: task")
        lines.append(f"    content: {json.dumps(task)}")
        lines.append("    kind: task")
        lines.append("    priority: 95")
        lines.append("    required: true")
        lines.append("    cache_policy: dynamic")
        lines.append("")
    for f in inventory:
        name = f["relative_path"]
        # YAML-safe quote
        safe_name = json.dumps(name)
        lines.append(f"  - name: {safe_name}")
        lines.append(f"    from_file: {safe_name}")
        lines.append(f"    kind: {f['kind']}")
        lines.append(f"    priority: {f['suggested_priority']}")
        lines.append(f"    cache_policy: {f['suggested_cache_policy']}")
        lines.append("    compressible: true")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _render_from_dict(data: dict[str, Any], *, markdown: bool) -> str:
    decisions = data.get("decisions", [])
    included = [d for d in decisions if d["status"] in ("included", "compressed", "truncated", "redacted")]
    excluded = [d for d in decisions if d["status"] == "excluded"]
    lines: list[str] = []
    if markdown:
        lines.append("# Context Compilation Report")
        lines.append("")
        lines.append(f"- **Model:** `{data.get('model')}`")
        lines.append(f"- **Health score:** **{data.get('health_score')}/100**")
        if data.get("health_breakdown"):
            bits = ", ".join(f"`{k}: {v:+d}`" for k, v in sorted(data["health_breakdown"].items()))
            lines.append(f"  - Breakdown: {bits}")
        lines.append(
            f"- **Tokens:** {data.get('used_tokens', 0):,} used / "
            f"{data.get('available_tokens', 0):,} available "
            f"(budget {data.get('token_budget', 0):,})"
        )
        lines.append(f"- **Cacheable prefix:** {data.get('cacheable_prefix_tokens', 0):,} tokens")
        lines.append("")
        if included:
            lines.append("## Included")
            lines.append("")
            lines.append("| Name | Kind | Tokens | Cache | Priority | Sensitivity | Status |")
            lines.append("|------|------|-------:|-------|---------:|-------------|--------|")
            for d in included:
                sens = d.get("sensitivity", "internal")
                lines.append(
                    f"| `{d['name']}` | {d['kind']} | {d['tokens']:,} "
                    f"| {d['cache_policy']} | {d['priority']} | {sens} | {d['status']} |"
                )
            lines.append("")
        if excluded:
            lines.append("## Excluded")
            lines.append("")
            lines.append("| Name | Kind | Tokens | Priority | Reason |")
            lines.append("|------|------|-------:|---------:|--------|")
            for d in excluded:
                lines.append(
                    f"| `{d['name']}` | {d['kind']} | {d['original_tokens']:,} "
                    f"| {d['priority']} | {d['reason']} |"
                )
            lines.append("")
        return "\n".join(lines)
    lines.append("Included:")
    if not included:
        lines.append("  (none)")
    for d in included:
        notes = []
        if d.get("required"):
            notes.append("required")
        if d.get("cache_policy") == "stable":
            notes.append("stable cache prefix")
        notes.append(d.get("kind", "other"))
        sens_tag = " [!secret]" if d.get("sensitivity") == "secret" else ""
        lines.append(f"  - {d['name']}: {d['tokens']:,} tokens, {', '.join(notes)}{sens_tag}")
    if excluded:
        lines.append("")
        lines.append("Excluded:")
        for d in excluded:
            lines.append(f"  - {d['name']}: {d['reason']}")
    lines.append("")
    lines.append(f"Estimated input tokens: {data.get('used_tokens', 0):,}")
    lines.append(f"Reserved output tokens: {data.get('reserved_output_tokens', 0):,}")
    lines.append(f"Cacheable prefix: {data.get('cacheable_prefix_tokens', 0):,} tokens")
    lines.append(f"Token budget: {data.get('token_budget', 0):,}")
    lines.append(f"Context health score: {data.get('health_score', 0)}/100")
    lines.append(f"Tokenizer: {data.get('tokenizer_backend', 'unknown')}")
    return "\n".join(lines)


# ============================================================================
# ContextOps commands (v0.3): scan-risk, bom, diff, eval, cache-plan, viz,
# viz-diff, mcp-audit, mcp-select, mcp-viz
# ============================================================================


@app.command("scan-risk")
def scan_risk_cmd(
    path: Path = typer.Argument(..., help="File or directory to scan for PII/secrets."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
    markdown: bool = typer.Option(False, "--markdown", help="Emit Markdown."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write output to a file."),
    ignore: list[str] = typer.Option(DEFAULT_IGNORE, "--ignore"),
    max_files: int = typer.Option(200, "--max-files"),
) -> None:
    """Scan files for PII and secrets (local-first, masked previews only)."""
    from .scanner import ContextScanner

    if not path.exists():
        console.print(f"[red]error:[/red] path does not exist: {path}")
        raise typer.Exit(code=2)
    scanner = ContextScanner()
    targets = [path] if path.is_file() else _walk(path, set(ignore), max_files)

    file_results: list[dict[str, Any]] = []
    for f in targets:
        try:
            res = scanner.scan_file(f)
        except OSError:
            continue
        if res.findings:
            file_results.append({"path": str(f), **res.to_dict()})

    if json_out:
        text = json.dumps({"scanned": len(targets), "files_with_findings": file_results}, indent=2)
        _emit(text, out)
        return
    if markdown:
        lines = ["# Risk Scan", "", f"Scanned {len(targets)} file(s).", ""]
        for fr in file_results:
            lines.append(f"## `{fr['path']}` — risk: **{fr['risk_level']}**")
            for fnd in fr["findings"]:
                lines.append(f"- [{fnd['severity']}] {fnd['category']}: `{fnd['matched_preview']}`")
            lines.append("")
        _emit("\n".join(lines), out)
        return

    if not file_results:
        console.print(f"[green]✓[/green] No PII/secret findings across {len(targets)} file(s).")
        return
    table = Table(title=f"risk scan: {path}")
    table.add_column("File", overflow="fold")
    table.add_column("Risk")
    table.add_column("Category")
    table.add_column("Severity")
    table.add_column("Preview (masked)")
    for fr in file_results:
        for fnd in fr["findings"]:
            table.add_row(fr["path"], fr["risk_level"], fnd["category"], fnd["severity"], fnd["matched_preview"])
    console.print(table)
    console.print(f"\n[bold]{len(file_results)}[/bold] file(s) with findings out of {len(targets)} scanned.")


@app.command()
def bom(
    input: Path = typer.Argument(..., help="A saved compiled-pack JSON (from `compile --save-pack`) or a BOM JSON."),
    format: str = typer.Option("markdown", "--format", "-f", help="json | markdown."),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    """Render a Context Bill of Materials from a compiled-pack or BOM JSON file."""
    from .bom import ContextBOM
    from .compiler import compiled_pack_from_dict

    if not input.exists():
        console.print(f"[red]error:[/red] file does not exist: {input}")
        raise typer.Exit(code=2)
    data = json.loads(input.read_text(encoding="utf-8"))
    if "included_items" in data and "schema_version" in data:
        bom_obj = ContextBOM.from_dict(data)
    elif "decisions" in data:
        bom_obj = ContextBOM.from_compiled(compiled_pack_from_dict(data))
        err_console.print(
            "[yellow]note:[/yellow] input is a compiled-pack snapshot; item details are "
            "limited. For full fidelity, produce a BOM with `compile --bom out.json`."
        )
    else:
        console.print("[red]error:[/red] unrecognized input; expected a BOM or compiled-pack JSON.")
        raise typer.Exit(code=2)
    text = bom_obj.to_json() if format == "json" else bom_obj.to_markdown()
    _emit(text, out)


@app.command()
def diff(
    old_bom: Path = typer.Argument(..., help="Old BOM JSON."),
    new_bom: Path = typer.Argument(..., help="New BOM JSON."),
    format: str = typer.Option("text", "--format", "-f", help="text | json | markdown."),
    out: Path | None = typer.Option(None, "--out", "-o"),
    fail_on_risk_increase: bool = typer.Option(
        False, "--fail-on-risk-increase", help="Exit non-zero if risk score increased."
    ),
) -> None:
    """Diff two Context Bills of Materials."""
    from .diff import ContextDiff

    for p in (old_bom, new_bom):
        if not p.exists():
            console.print(f"[red]error:[/red] file does not exist: {p}")
            raise typer.Exit(code=2)
    d = ContextDiff.compare(str(old_bom), str(new_bom))
    text = (
        d.to_json() if format == "json"
        else d.to_markdown() if format == "markdown"
        else d.to_text()
    )
    _emit(text, out)
    if fail_on_risk_increase and d.risk_increased:
        console.print(f"[red]Risk increased by {d.risk_change}[/red]")
        raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd(
    suite: Path = typer.Argument(..., help="Eval suite YAML/JSON file."),
    bom: Path | None = typer.Option(None, "--bom", help="BOM JSON to evaluate."),
    format: str = typer.Option("text", "--format", "-f", help="text | json."),
) -> None:
    """Run a context-eval suite against a BOM (CI gate)."""
    from .bom import ContextBOM
    from .evals import EvalSuite

    if not suite.exists():
        console.print(f"[red]error:[/red] eval suite not found: {suite}")
        raise typer.Exit(code=2)
    if bom is None or not bom.exists():
        console.print("[red]error:[/red] --bom <context_bom.json> is required")
        raise typer.Exit(code=2)
    suite_obj = EvalSuite.from_file(str(suite))
    bom_obj = ContextBOM.from_json(str(bom))
    results = suite_obj.run(bom_obj)
    all_passed = all(r.passed for r in results)
    if format == "json":
        sys.stdout.write(json.dumps([r.to_dict() for r in results], indent=2) + "\n")
    else:
        for r in results:
            mark = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
            console.print(f"{mark}  {r.name}")
            for c in r.checks:
                cm = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
                console.print(f"    {cm} {c.name}: {c.detail}")
    raise typer.Exit(code=0 if all_passed else 1)


@app.command("cache-plan")
def cache_plan_cmd(
    bom: Path = typer.Argument(..., help="BOM JSON (or compiled-pack JSON) to analyze."),
    format: str = typer.Option("text", "--format", "-f", help="text | json."),
) -> None:
    """Analyze cache layout of a compiled context."""
    from .compiler import compiled_pack_from_dict

    if not bom.exists():
        console.print(f"[red]error:[/red] file does not exist: {bom}")
        raise typer.Exit(code=2)
    from .bom import ContextBOM
    from .cache import CachePlanner

    data = json.loads(bom.read_text(encoding="utf-8"))
    if "included_items" in data and "schema_version" in data:
        plan = CachePlanner().analyze_bom(ContextBOM.from_dict(data))
    elif "decisions" in data:
        # A snapshot re-materializes no ContextItems, so `cache_plan()` would walk an
        # empty list and return an all-zero plan with exit 0 — a silently wrong answer.
        # Build the BOM (whose items come from the recorded decisions) and use the
        # content-free analyzer that exists for exactly this case.
        plan = CachePlanner().analyze_bom(ContextBOM.from_compiled(compiled_pack_from_dict(data)))
    else:
        console.print("[red]error:[/red] expected a BOM JSON (from `compile --bom`) or compiled-pack JSON.")
        raise typer.Exit(code=2)
    if format == "json":
        sys.stdout.write(json.dumps(plan.to_dict(), indent=2) + "\n")
        return
    console.print(f"Cacheable estimate: [bold]{plan.cacheable_token_estimate:,}[/bold] tokens "
                  f"({plan.cache_efficiency_score}/100 efficiency)")
    console.print(f"Stable prefix: {plan.stable_prefix_length} items / {plan.stable_prefix_tokens:,} tokens")
    console.print(f"Dynamic section: {plan.dynamic_section_length} items / {plan.dynamic_section_tokens:,} tokens")
    if plan.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for w in plan.warnings:
            console.print(f"  ! {w}")
    if plan.recommendations:
        console.print("\n[cyan]Recommendations:[/cyan]")
        for r in plan.recommendations:
            console.print(f"  → {r}")


@app.command()
def viz(
    bom: Path = typer.Argument(..., help="BOM JSON (or compiled-pack JSON)."),
    out: Path = typer.Option(Path("context_mri.html"), "--out", "-o", help="Output HTML path."),
) -> None:
    """Generate a Context MRI HTML report (requires no extra deps to render)."""
    from .bom import ContextBOM
    from .compiler import compiled_pack_from_dict
    from .viz import ContextMRI

    if not bom.exists():
        console.print(f"[red]error:[/red] file does not exist: {bom}")
        raise typer.Exit(code=2)
    data = json.loads(bom.read_text(encoding="utf-8"))
    if "schema_version" in data and "included_items" in data:
        bom_obj = ContextBOM.from_dict(data)
    else:
        bom_obj = ContextBOM.from_compiled(compiled_pack_from_dict(data))
    ContextMRI(bom=bom_obj).export_html(str(out))
    console.print(f"[green]Context MRI written:[/green] {out}")


@app.command("viz-diff")
def viz_diff_cmd(
    old_bom: Path = typer.Argument(..., help="Old BOM JSON."),
    new_bom: Path = typer.Argument(..., help="New BOM JSON."),
    out: Path = typer.Option(Path("context_diff.html"), "--out", "-o"),
) -> None:
    """Generate a visual before/after context diff HTML report."""
    from .viz import ContextDiffViz

    for p in (old_bom, new_bom):
        if not p.exists():
            console.print(f"[red]error:[/red] file does not exist: {p}")
            raise typer.Exit(code=2)
    ContextDiffViz.from_files(str(old_bom), str(new_bom)).export_html(str(out))
    console.print(f"[green]Context diff written:[/green] {out}")


@app.command("mcp-audit")
def mcp_audit_cmd(
    tools: Path = typer.Argument(..., help="MCP tool schemas JSON file."),
    format: str = typer.Option("text", "--format", "-f", help="text | json."),
    out: Path | None = typer.Option(None, "--out", "-o"),
) -> None:
    """Audit MCP tool schemas: token cost, complexity, risk, overlap."""
    from .mcp import MCPToolBudgeter

    if not tools.exists():
        console.print(f"[red]error:[/red] file does not exist: {tools}")
        raise typer.Exit(code=2)
    result = MCPToolBudgeter().audit(str(tools))
    if format == "json":
        _emit(result.to_json(), out)
        return
    table = Table(title=f"mcp-audit: {tools}")
    table.add_column("Tool", overflow="fold")
    table.add_column("Tokens", justify="right")
    table.add_column("Complexity", justify="right")
    table.add_column("Risky")
    for a in sorted(result.assessments, key=lambda a: -a.tokens):
        table.add_row(a.name, f"{a.tokens:,}", str(a.schema_complexity), ", ".join(a.risk_terms) or "—")
    console.print(table)
    console.print(f"\nTotal: {result.total_tokens:,} tokens across {len(result.assessments)} tools")
    for w in result.warnings:
        console.print(f"  [yellow]![/yellow] {w}")


@app.command("mcp-select")
def mcp_select_cmd(
    tools: Path = typer.Argument(..., help="MCP tool schemas JSON file."),
    task: str = typer.Option(..., "--task", "-t", help="Task to select tools for."),
    budget: int = typer.Option(6_000, "--budget", "-b", help="Token budget for tools."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write selected tool names/result JSON."),
) -> None:
    """Select the most relevant MCP tools for a task under a token budget."""
    from .mcp import MCPToolBudgeter

    if not tools.exists():
        console.print(f"[red]error:[/red] file does not exist: {tools}")
        raise typer.Exit(code=2)
    result = MCPToolBudgeter(token_budget=budget).select_tools(task=task, tools=str(tools))
    if out is not None:
        out.write_text(result.to_json(), encoding="utf-8")
        console.print(f"[green]Selection written:[/green] {out}")
    console.print(f"[bold]Selected[/bold] ({result.selected_tokens:,}/{budget:,} tokens): {result.selected_tools}")
    console.print(f"[dim]Excluded:[/dim] {result.excluded_tools}")
    if result.risky_tools:
        console.print(f"[yellow]Risky:[/yellow] {result.risky_tools}")


@app.command("mcp-viz")
def mcp_viz_cmd(
    tools: Path = typer.Argument(..., help="MCP tool schemas JSON file."),
    task: str | None = typer.Option(None, "--task", "-t", help="Task to select tools for."),
    budget: int = typer.Option(6_000, "--budget", "-b"),
    out: Path = typer.Option(Path("mcp_map.html"), "--out", "-o"),
) -> None:
    """Generate an MCP tool map HTML report."""
    from .viz import MCPToolViz

    if not tools.exists():
        console.print(f"[red]error:[/red] file does not exist: {tools}")
        raise typer.Exit(code=2)
    MCPToolViz.from_file(str(tools), task=task, budget=budget).export_html(str(out))
    console.print(f"[green]MCP map written:[/green] {out}")


# ============================================================ 0.4: Adaptive Context Engine

skill_app = typer.Typer(help="Portable context-management skill (SKILL.md).", add_completion=False)
autonomy_app = typer.Typer(help="Graduated context autonomy: certify, show, set.", add_completion=False)
app.add_typer(skill_app, name="skill")
app.add_typer(autonomy_app, name="autonomy")


@skill_app.command("export")
def skill_export_cmd(
    dest: Path = typer.Argument(Path(".claude/skills"), help="Skills directory to export into."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing copy."),
) -> None:
    """Copy the context-management skill (SKILL.md + references/) into a skills directory."""
    from .skills import export_skill

    try:
        target = export_skill(dest, overwrite=overwrite)
    except FileExistsError as e:
        err_console.print(f"[red]error:[/red] {escape(str(e))}")
        raise typer.Exit(code=1) from None
    console.print(f"[green]Skill exported:[/green] {target}")


@skill_app.command("show")
def skill_show_cmd(
    brief: bool = typer.Option(False, "--brief", help="Compact (~300 token) operating summary."),
    model: str | None = typer.Option(None, "--model", "-m",
                                     help="Specialise to a model's published behaviour prior."),
    autonomy: int | None = typer.Option(None, "--autonomy", help="Autonomy level 0-4 to describe."),
) -> None:
    """Print the skill as a system-prompt section (policy thresholds, autonomy, adaptations)."""
    from .skills import render_guidance

    if model:
        from .adaptive import PolicyTuner, prior_for

        prior = prior_for(model)
        if prior is None:
            err_console.print(f"[yellow]no published prior for {escape(model)}; showing base guidance[/yellow]")
            text = render_guidance(autonomy=autonomy, brief=brief)
        else:
            text = PolicyTuner().tune(prior).render_guidance(autonomy=autonomy, brief=brief)
    else:
        text = render_guidance(autonomy=autonomy, brief=brief)
    sys.stdout.write(text)


@app.command("bench")
def bench_cmd(
    agent: str = typer.Option("competent", "--agent", "-a",
                              help="Reference agent: competent | anxious | avoidant | reckless."),
    modes: str = typer.Option("harness,unrestricted,guided,adaptive", "--modes"),
    families: str = typer.Option("research,coding,analytics,workflow,multi_agent", "--families"),
    budgets: str = typer.Option("4000,8000", "--budgets", help="Comma-separated context budgets (tokens)."),
    tasks_per_family: int = typer.Option(2, "--tasks", "-n"),
    stream_tokens: int = typer.Option(12_000, "--stream-tokens", help="Approx. tokens of material per task."),
    seed: int = typer.Option(7, "--seed"),
    model: str = typer.Option("bench-model", "--model", "-m",
                              help="Model id recorded in traces (selects a prior if no calibration data)."),
    no_optimize: bool = typer.Option(False, "--no-optimize", help="Use the rule-tuned policy without search."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write the full JSON report."),
    traces: Path | None = typer.Option(None, "--traces", help="Write decision traces (JSONL)."),
) -> None:
    """Run CTXBench: harness vs unrestricted vs guided vs adaptive context management."""
    from .bench import BenchRunner, TaskGenerator, make_agent
    from .runtime import TraceLog

    try:
        make_agent(agent)
        fams = [f.strip() for f in families.split(",") if f.strip()]
        tasks = TaskGenerator(seed=seed).suite(fams, n=tasks_per_family, stream_tokens=stream_tokens)
        runner = BenchRunner(tasks, lambda: make_agent(agent), modes=[m.strip() for m in modes.split(",")],
                             budgets=[int(b) for b in budgets.split(",")], model=model,
                             trace=TraceLog(traces) if traces else None, optimize=not no_optimize)
    except ValueError as e:
        err_console.print(f"[red]error:[/red] {escape(str(e))}")
        raise typer.Exit(code=2) from None
    report = runner.run()
    if out is not None:
        report.to_json(out)
        err_console.print(f"[green]Report written:[/green] {out}")
    sys.stdout.write(report.to_markdown())


@app.command("profile")
def profile_cmd(
    traces: list[Path] = typer.Argument(..., help="Decision-trace JSONL file(s)."),
    store: Path | None = typer.Option(None, "--store", help="Save tuned policies to this PolicyStore directory."),
    out: Path | None = typer.Option(None, "--out", "-o", help="Write profiles + policies JSON."),
) -> None:
    """Profile how each model manages context and propose an adaptive policy."""
    from .adaptive import BehaviorProfiler, PolicyStore, PolicyTuner
    from .runtime import load_traces

    missing = [p for p in traces if not p.exists()]
    if missing:
        err_console.print(f"[red]error:[/red] missing: {', '.join(map(str, missing))}")
        raise typer.Exit(code=2)
    profiles = BehaviorProfiler().profile(load_traces(*traces))
    result: dict[str, dict[str, object]] = {}
    for model, prof in profiles.items():
        pol = PolicyTuner().tune(prof)
        if store is not None:
            PolicyStore(store).save(pol)
        result[model] = {"profile": prof.to_dict(), "policy": pol.to_dict()}
        console.print(f"[bold]{escape(model)}[/bold]: {prof.sessions} sessions, behaviours: "
                      f"{', '.join(prof.behaviors) or 'none'}")
        for r in pol.rules_applied:
            console.print(f"  • {escape(r)}")
    if out is not None:
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        console.print(f"[green]Written:[/green] {out}")


@app.command("train-export")
def train_export_cmd(
    traces: list[Path] = typer.Argument(..., help="Decision-trace JSONL file(s)."),
    out_dir: Path = typer.Option(Path("ctx_datasets"), "--out", "-o"),
) -> None:
    """Build SFT, preference (DPO) and reward datasets from decision traces."""
    from .runtime import load_traces
    from .train import build_datasets

    missing = [p for p in traces if not p.exists()]
    if missing:
        err_console.print(f"[red]error:[/red] missing: {', '.join(map(str, missing))}")
        raise typer.Exit(code=2)
    manifest = build_datasets(load_traces(*traces), out_dir)
    console.print_json(json.dumps(manifest))


@app.command("serve-mcp")
def serve_mcp_cmd(
    budget: int = typer.Option(128_000, "--budget", "-b"),
    model: str = typer.Option("claude-sonnet-4.6", "--model", "-m"),
    reserved: int = typer.Option(4_000, "--reserved-output"),
    state: Path | None = typer.Option(None, "--state", help="Persist session state here (JSON)."),
    autonomy: str = typer.Option("GOVERNED", "--autonomy", help="OBSERVE | RECOMMEND | SUPERVISED | GOVERNED | ADAPTIVE"),
    allow_add: bool = typer.Option(False, "--allow-add", help="Expose context_add for harness-side event pushes."),
) -> None:
    """Serve the governed context API as an MCP server over stdio."""
    from .adapters.mcp_server import ContextMCPServer
    from .governance import AutonomyLevel
    from .runtime import ContextRuntime

    try:
        level = AutonomyLevel.parse(autonomy)
        if state is not None:
            server = ContextMCPServer.from_state(state, budget=budget, model=model, allow_add=allow_add,
                                                 reserved_output_tokens=reserved, autonomy=level)
        else:
            server = ContextMCPServer(ContextRuntime(budget=budget, model=model, reserved_output_tokens=reserved,
                                                     autonomy=level), allow_add=allow_add)
    except (ValueError, KeyError) as e:
        err_console.print(f"[red]error:[/red] {escape(str(e))}")
        raise typer.Exit(code=2) from None
    server.serve()


@app.command("context-tools")
def context_tools_cmd(
    provider: str = typer.Option("openai", "--provider", "-p", help="openai | anthropic | mcp"),
) -> None:
    """Print the context-API tool definitions for a provider."""
    from .adapters.context_tools import anthropic_tools, mcp_tools, openai_tools

    fn = {"openai": openai_tools, "anthropic": anthropic_tools, "mcp": mcp_tools}.get(provider)
    if fn is None:
        err_console.print("[red]error:[/red] provider must be openai, anthropic or mcp")
        raise typer.Exit(code=2)
    sys.stdout.write(json.dumps(fn(), indent=2) + "\n")


@autonomy_app.command("certify")
def autonomy_certify_cmd(
    report: Path = typer.Argument(..., help="CTXBench JSON report (from `bench --out`)."),
    registry: Path = typer.Option(Path(".ctxbudgeter/autonomy.json"), "--registry"),
    agent_id: str | None = typer.Option(None, "--agent-id", help="Defaults to the report's agent."),
    model: str | None = typer.Option(None, "--model", help="Defaults to the report's model."),
    mode: str = typer.Option("adaptive", "--mode", help="Which mode's episodes count as evidence."),
) -> None:
    """Promote, hold or demote an agent's context autonomy from benchmark evidence."""
    from .governance import AutonomyRegistry, Evidence

    if not report.exists():
        err_console.print(f"[red]error:[/red] file does not exist: {report}")
        raise typer.Exit(code=2)
    data = json.loads(report.read_text(encoding="utf-8"))
    eps = [e for e in data.get("episodes", []) if e.get("mode") == mode]
    if not eps:
        err_console.print(f"[red]error:[/red] no '{escape(mode)}' episodes in the report")
        raise typer.Exit(code=2)
    ev = Evidence(episodes=len(eps), mean_cis=sum(e["cis"] for e in eps) / len(eps),
                  critical_violations=sum(e["critical_violations"] for e in eps),
                  mean_retention=sum(e["retention"] for e in eps) / len(eps), source=str(report))
    reg = AutonomyRegistry(registry)
    res = reg.certify(agent_id or data["agent"], model or data["model"], ev)
    console.print(f"[bold]{res.action}[/bold]: L{int(res.current)} → L{int(res.new)} ({res.new.name})")
    for r in res.reasons:
        console.print(f"  • {escape(r)}")


@autonomy_app.command("show")
def autonomy_show_cmd(
    agent_id: str = typer.Argument(...),
    model: str = typer.Argument(...),
    registry: Path = typer.Option(Path(".ctxbudgeter/autonomy.json"), "--registry"),
) -> None:
    """Show an agent's autonomy level and certification history."""
    from .governance import AutonomyRegistry

    reg = AutonomyRegistry(registry)
    lvl = reg.level(agent_id, model)
    console.print(f"{escape(agent_id)}@{escape(model)}: L{int(lvl)} ({lvl.name})")
    for h in reg.history(agent_id, model):
        console.print(f"  {h['action']}: L{h['from']} → L{h['to']} — {escape('; '.join(h['reasons']))}")


def _emit(text: str, out: Path | None) -> None:
    """Write text to a file or stdout."""
    if out is not None:
        out.write_text(text, encoding="utf-8")
        console.print(f"[green]Written:[/green] {out}")
    else:
        sys.stdout.write(text + "\n")


if __name__ == "__main__":  # pragma: no cover
    app()
