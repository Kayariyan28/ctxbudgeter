"""Typer CLI: `scan`, `compile`, `report`, `pack`, `validate`.

The CLI is a thin layer over ContextPack — useful for Claude Code, ad-hoc audits,
CI checks, and YAML-driven workflows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .pack import ContextPack
from .report import to_json, to_markdown, to_text
from .tokenizer import TokenCounter

app = typer.Typer(
    add_completion=False,
    help="ctxbudgeter — compile clean, cheap, auditable context for AI agents.",
)
console = Console()

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

    inventory: list[dict] = []
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
        }, indent=2))
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
) -> None:
    """Compile context from a directory + task, print a report."""
    if not path.exists():
        console.print(f"[red]error:[/red] path does not exist: {path}")
        raise typer.Exit(code=2)
    if format not in ("text", "markdown", "json"):
        console.print(f"[red]error:[/red] format must be text | markdown | json (got {format!r})")
        raise typer.Exit(code=2)

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

    compiled = pack.compile()
    rendered = (
        to_markdown(compiled) if format == "markdown"
        else to_json(compiled) if format == "json"
        else to_text(compiled)
    )
    if output is not None:
        output.write_text(rendered)
        console.print(f"[green]Report written:[/green] {output}")
    else:
        sys.stdout.write(rendered + "\n")

    if prompt_output is not None:
        prompt_output.write_text(compiled.as_text())
        console.print(f"[green]Prompt written:[/green] {prompt_output}")
    if save_pack is not None:
        save_pack.write_text(to_json(compiled))
        console.print(f"[green]Pack saved:[/green] {save_pack}")


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
        console.print(f"[red]spec error:[/red] {e}")
        raise typer.Exit(code=2) from e

    compiled = p.compile()
    rendered = (
        to_markdown(compiled) if format == "markdown"
        else to_json(compiled) if format == "json"
        else to_text(compiled)
    )
    if output is not None:
        output.write_text(rendered)
        console.print(f"[green]Report written:[/green] {output}")
    else:
        sys.stdout.write(rendered + "\n")
    if save_pack is not None:
        save_pack.write_text(to_json(compiled))
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
            console.print(f"[red]✗[/red] {i}")
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
    data = json.loads(pack_json.read_text())
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


def _classify(path: Path) -> tuple[str, int, str]:
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
    inventory: list[dict],
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
    out_path.write_text("\n".join(lines))


def _render_from_dict(data: dict, *, markdown: bool) -> str:
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


if __name__ == "__main__":  # pragma: no cover
    app()
