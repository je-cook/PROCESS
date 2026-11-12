#!/usr/bin/env python3
"""Calculate a transparent PROCESS Health Score (PHS).

Requires these commands on PATH:
    ruff, lizard, radon, vulture, lint-imports, git

Typical installation:
    python -m pip install ruff lizard radon vulture import-linter

Usage:
    python process_health.py process \
        --json process-health.json \
        --markdown process-health.md \
        --csv process-health.csv

The script does not modify source code. It runs each analyser directly because their
machine-readable outputs are more reliable than parsing `pre-commit run` output.
Pre-commit can still invoke this script as a local hook.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXCLUDED_NAMES = {"_version.py"}
EXCLUDED_PARTS = {".git", ".venv", "venv", "build", "dist", "__pycache__"}


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def run(command: list[str], cwd: Path, *, allowed: set[int] = {0}) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in allowed:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(command)}\n{detail}")
    return completed


def require_commands(commands: Iterable[str]) -> None:
    missing = [command for command in commands if shutil.which(command) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required commands: {names}. Install with:\n"
            "python -m pip install ruff lizard radon vulture import-linter"
        )


def normalise_path(raw: str, root: Path) -> str:
    path = Path(raw)
    if path.is_absolute():
        try:
            path = path.resolve().relative_to(root.resolve())
        except ValueError:
            return path.as_posix()
    return path.as_posix().lstrip("./")


def discover_python_files(root: Path, target: Path) -> list[str]:
    base = target if target.is_absolute() else root / target
    files: list[str] = []
    for path in base.rglob("*.py"):
        relative = normalise_path(str(path), root)
        parts = set(Path(relative).parts)
        if path.name in EXCLUDED_NAMES or parts & EXCLUDED_PARTS:
            continue
        files.append(relative)
    return sorted(files)


@dataclass
class FunctionMetric:
    name: str
    nloc: int
    ccn: int
    parameters: int


@dataclass
class FileMetrics:
    path: str
    nloc: int = 0
    functions: list[FunctionMetric] = field(default_factory=list)
    maintainability_index: float = 100.0
    ruff_points: float = 0.0
    ruff_findings: int = 0
    dead_weighted_nloc: float = 0.0
    dead_findings: int = 0
    architecture_violations: int = 0
    commits_12_months: int = 0
    penalties: dict[str, float] = field(default_factory=dict)
    health: float = 10.0
    classification: str = "excellent"
    hotspot_risk: float = 0.0


def collect_lizard(root: Path, files: list[str], metrics: dict[str, FileMetrics]) -> None:
    # Lizard CSV columns are stable enough in recent releases, but locate fields by
    # header name where possible and fall back to the documented order.
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        report = Path(handle.name)
    try:
        completed = run(["lizard", "--csv", *files], root, allowed={0, 1})
        report.write_text(completed.stdout, encoding="utf-8")
        rows = list(csv.reader(report.read_text(encoding="utf-8", errors="replace").splitlines()))
        if not rows:
            return
        header = [cell.strip().lower() for cell in rows[0]]

        def column(*names: str, fallback: int) -> int:
            for name in names:
                if name in header:
                    return header.index(name)
            return fallback

        nloc_i = column("nloc", fallback=0)
        ccn_i = column("ccn", "cyclomatic_complexity", fallback=1)
        params_i = column("parameter_count", "params", fallback=3)
        name_i = column("function_name", "name", fallback=7)
        file_i = column("file", "filename", fallback=6)
        for row in rows[1:]:
            if len(row) <= max(nloc_i, ccn_i, params_i, name_i, file_i):
                continue
            path = normalise_path(row[file_i], root)
            if path not in metrics:
                continue
            try:
                function = FunctionMetric(
                    name=row[name_i],
                    nloc=int(float(row[nloc_i])),
                    ccn=int(float(row[ccn_i])),
                    parameters=int(float(row[params_i])),
                )
            except ValueError:
                continue
            metrics[path].functions.append(function)
    finally:
        report.unlink(missing_ok=True)


def collect_radon(root: Path, target: str, metrics: dict[str, FileMetrics]) -> None:
    raw = json.loads(run(["radon", "raw", "-j", target], root).stdout or "{}")
    for filename, values in raw.items():
        path = normalise_path(filename, root)
        if path in metrics:
            metrics[path].nloc = int(values.get("sloc", values.get("loc", 0)))

    mi = json.loads(run(["radon", "mi", "-j", target], root).stdout or "{}")
    for filename, values in mi.items():
        path = normalise_path(filename, root)
        if path in metrics:
            metrics[path].maintainability_index = float(values.get("mi", 100.0))


def ruff_weight(code: str) -> float:
    # Higher weight for correctness and security; moderate for complexity and
    # refactoring; low for documentation/style.
    if code.startswith(("F", "E9", "S", "B", "DTZ", "RUF0")):
        return 3.0
    if code.startswith(("C90", "PLR", "PERF", "SIM", "RET", "TRY")):
        return 2.0
    return 0.5


def collect_ruff(root: Path, target: str, metrics: dict[str, FileMetrics]) -> None:
    completed = run(
        ["ruff", "check", target, "--output-format", "json", "--no-fix"],
        root,
        allowed={0, 1},
    )
    findings = json.loads(completed.stdout or "[]")
    for finding in findings:
        path = normalise_path(finding.get("filename", ""), root)
        if path not in metrics:
            continue
        code = finding.get("code") or ""
        metrics[path].ruff_findings += 1
        metrics[path].ruff_points += ruff_weight(code)


def collect_vulture(root: Path, target: str, metrics: dict[str, FileMetrics]) -> None:
    completed = run(
        ["vulture", target, "--min-confidence", "90", "--sort-by-size"],
        root,
        allowed={0, 1, 2, 3},
    )
    # Common output:
    # file.py:12: unused function 'x' (60% confidence, 8 lines)
    pattern = re.compile(
        r"^(?P<file>.+?):(?P<line>\d+): .*?\((?P<confidence>\d+)% confidence"
        r"(?:, (?P<lines>\d+) lines?)?\)$"
    )
    for line in completed.stdout.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        path = normalise_path(match.group("file"), root)
        if path not in metrics:
            continue
        confidence = int(match.group("confidence"))
        lines = int(match.group("lines") or 1)
        metrics[path].dead_findings += 1
        metrics[path].dead_weighted_nloc += lines * ((confidence - 50) / 50)


def collect_git(root: Path, files: list[str], metrics: dict[str, FileMetrics], since: str) -> None:
    completed = run(
        ["git", "log", f"--since={since}", "--name-only", "--pretty=format:"],
        root,
    )
    counts = Counter(line.strip() for line in completed.stdout.splitlines() if line.strip())
    for path in files:
        metrics[path].commits_12_months = counts[path]


def collect_import_linter(root: Path) -> tuple[int, str]:
    config = root / ".importlinter"
    if not config.exists():
        return 0, "not configured"
    completed = run(["lint-imports"], root, allowed={0, 1})
    output = f"{completed.stdout}\n{completed.stderr}"
    # Import Linter is contract-oriented and does not expose a stable JSON report.
    # Count broken contracts as repository-level violations. The total penalty is
    # then assigned to the repository, not guessed onto source files.
    patterns = [
        r"Broken contracts:\s*(\d+)",
        r"Contracts broken:\s*(\d+)",
    ]
    for pattern in patterns:
        if match := re.search(pattern, output, re.IGNORECASE):
            return int(match.group(1)), "measured"
    if completed.returncode == 0:
        return 0, "measured"
    # Conservative fallback: a failing invocation means at least one contract broke.
    return 1, "estimated from non-zero lint-imports exit"


def calculate_scores(metrics: dict[str, FileMetrics]) -> None:
    commits = [item.commits_12_months for item in metrics.values()]
    p95_commits = max(1.0, percentile([float(value) for value in commits], 0.95))

    for item in metrics.values():
        ccns = [float(function.ccn) for function in item.functions]
        function_nlocs = [float(function.nloc) for function in item.functions]
        parameters = [float(function.parameters) for function in item.functions]

        p95_ccn = percentile(ccns, 0.95)
        max_ccn = max(ccns, default=0.0)
        p95_nloc = percentile(function_nlocs, 0.95)
        max_nloc = max(function_nlocs, default=0.0)
        max_parameters = max(parameters, default=0.0)

        complexity = 0.60 * clamp((p95_ccn - 5) / 15) + 0.40 * clamp((max_ccn - 10) / 30)
        size = (
            0.45 * clamp((p95_nloc - 30) / 70)
            + 0.30 * clamp((max_nloc - 50) / 150)
            + 0.25 * clamp((max_parameters - 6) / 12)
        )
        maintainability = clamp((80 - item.maintainability_index) / 60)
        kloc = max(item.nloc / 1000, 0.1)
        ruff = clamp((item.ruff_points / kloc) / 20)
        dead_code = clamp((item.dead_weighted_nloc / kloc) / 50)
        architecture = 1 - math.exp(-item.architecture_violations / 2)

        item.penalties = {
            "complexity": round(complexity, 4),
            "size": round(size, 4),
            "maintainability": round(maintainability, 4),
            "ruff": round(ruff, 4),
            "dead_code": round(dead_code, 4),
            "architecture": round(architecture, 4),
        }
        total = (
            0.25 * complexity
            + 0.15 * size
            + 0.20 * maintainability
            + 0.15 * ruff
            + 0.10 * dead_code
            + 0.15 * architecture
        )
        item.health = round(10 - 9 * total, 2)
        item.classification = classification(item.health)
        item.hotspot_risk = round(
            (10 - item.health)
            * math.log2(2 + item.commits_12_months)
            * math.sqrt(max(item.nloc, 1) / 100),
            2,
        )
        change_factor = 1 + 2 * math.log(1 + item.commits_12_months) / math.log(1 + p95_commits)
        item._exposure_weight = math.sqrt(max(item.nloc, 1)) * min(3.0, change_factor)  # type: ignore[attr-defined]


def classification(score: float) -> str:
    if score >= 9.0:
        return "excellent"
    if score >= 8.0:
        return "healthy"
    if score >= 6.5:
        return "watch"
    if score >= 4.0:
        return "unhealthy"
    return "critical"


def repository_summary(metrics: dict[str, FileMetrics], architecture_violations: int, architecture_status: str) -> dict[str, Any]:
    items = list(metrics.values())
    weights = [getattr(item, "_exposure_weight") for item in items]
    repository_health = sum(weights) / sum(
        weight / max(item.health, 0.1) for weight, item in zip(weights, items)
    ) if items else 10.0
    median = statistics.median([item.health for item in items]) if items else 10.0
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_health": round(repository_health, 2),
        "median_file_health": round(median, 2),
        "classification": classification(repository_health),
        "critical_file_count": sum(item.health < 4.0 for item in items),
        "architecture_violations": architecture_violations,
        "architecture_status": architecture_status,
        "files": [
            {
                key: value
                for key, value in asdict(item).items()
                if key != "functions"
            }
            | {
                "function_count": len(item.functions),
                "max_ccn": max((function.ccn for function in item.functions), default=0),
                "max_function_nloc": max((function.nloc for function in item.functions), default=0),
                "max_parameters": max((function.parameters for function in item.functions), default=0),
            }
            for item in sorted(items, key=lambda value: (-value.hotspot_risk, value.path))
        ],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# PROCESS Code Health",
        "",
        f"- **Repository health:** {summary['repository_health']:.2f}/10 ({summary['classification']})",
        f"- **Median file health:** {summary['median_file_health']:.2f}/10",
        f"- **Critical files:** {summary['critical_file_count']}",
        f"- **Architecture violations:** {summary['architecture_violations']} ({summary['architecture_status']})",
        "",
        "## Highest-risk hotspots",
        "",
        "| Health | Risk | NLOC | Commits | File |",
        "|---:|---:|---:|---:|---|",
    ]
    for item in summary["files"][:20]:
        lines.append(
            f"| {item['health']:.2f} | {item['hotspot_risk']:.2f} | {item['nloc']} | "
            f"{item['commits_12_months']} | `{item['path']}` |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- 9.0–10.0: excellent",
        "- 8.0–8.9: healthy",
        "- 6.5–7.9: watch",
        "- 4.0–6.4: unhealthy",
        "- 1.0–3.9: critical",
        "",
        "The repository score is an exposure-weighted harmonic mean. Frequently changed, larger, unhealthy files therefore influence it more strongly.",
    ])
    return "\n".join(lines) + "\n"


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "path", "health", "classification", "hotspot_risk", "nloc",
        "commits_12_months", "function_count", "max_ccn",
        "max_function_nloc", "max_parameters", "maintainability_index",
        "ruff_findings", "dead_findings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in summary["files"]:
            writer.writerow({field: item.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the PROCESS Health Score.")
    parser.add_argument("target", nargs="?", default="process", help="Package directory to analyse")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--since", default="12 months ago", help="Git history window")
    parser.add_argument("--json", dest="json_path", type=Path, default=Path("process-health.json"))
    parser.add_argument("--markdown", type=Path, default=Path("process-health.md"))
    parser.add_argument("--csv", dest="csv_path", type=Path, default=Path("process-health.csv"))
    parser.add_argument("--minimum-health", type=float, default=None, help="Fail below this repository score")
    parser.add_argument("--maximum-critical-files", type=int, default=None, help="Fail above this count")
    parser.add_argument("--skip-import-linter", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    target = Path(args.target)
    required = ["ruff", "lizard", "radon", "vulture", "git"]
    if not args.skip_import_linter:
        required.append("lint-imports")
    require_commands(required)

    files = discover_python_files(root, target)
    if not files:
        raise RuntimeError(f"No Python files found beneath {target}")
    metrics = {path: FileMetrics(path=path) for path in files}

    collect_lizard(root, files, metrics)
    collect_radon(root, args.target, metrics)
    collect_ruff(root, args.target, metrics)
    collect_vulture(root, args.target, metrics)
    collect_git(root, files, metrics, args.since)

    architecture_violations, architecture_status = (0, "skipped")
    if not args.skip_import_linter:
        architecture_violations, architecture_status = collect_import_linter(root)

    calculate_scores(metrics)
    summary = repository_summary(metrics, architecture_violations, architecture_status)

    json_path = args.json_path if args.json_path.is_absolute() else root / args.json_path
    markdown_path = args.markdown if args.markdown.is_absolute() else root / args.markdown
    csv_path = args.csv_path if args.csv_path.is_absolute() else root / args.csv_path
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(summary), encoding="utf-8")
    write_csv(csv_path, summary)

    print(markdown_report(summary))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"CSV: {csv_path}")

    failed = False
    if args.minimum_health is not None and summary["repository_health"] < args.minimum_health:
        print(
            f"FAIL: repository health {summary['repository_health']:.2f} "
            f"is below {args.minimum_health:.2f}",
            file=sys.stderr,
        )
        failed = True
    if (
        args.maximum_critical_files is not None
        and summary["critical_file_count"] > args.maximum_critical_files
    ):
        print(
            f"FAIL: {summary['critical_file_count']} critical files exceeds "
            f"{args.maximum_critical_files}",
            file=sys.stderr,
        )
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
