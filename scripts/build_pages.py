"""Publish a curated, read-only GitHub Pages snapshot without exposing local runs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "cua_jev" / "ui" / "static"
WEBSITE = ROOT / "website"
SNAPSHOT = WEBSITE / "snapshot.json"
MEDIA = WEBSITE / "media"
TASKS = ("edge", "excel", "vscode", "explorer")
MODES = ("hybrid", "gui-only")
PUBLIC_ROW_FIELDS = (
    "agent",
    "action_space",
    "representative_run_id",
    "median_wall_time_ms",
    "median_model_cost_usd",
    "median_reference_cost_usd",
)
PRIVATE_TEXT = re.compile(r"[A-Za-z]:\\|/Users/|/home/|apikey_|TYPESAFE_API_KEY|sk-[A-Za-z0-9]", re.I)


def _safe_text(value: Any) -> str:
    if not isinstance(value, str) or PRIVATE_TEXT.search(value):
        raise ValueError("Snapshot contains non-public or invalid text")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_snapshot() -> None:
    """Explicitly export allowlisted display fields from measured local records."""
    from cua_jev.ui.manager import TASK_CATALOG, RunManager

    manager = RunManager(ROOT)
    benchmarks: dict[str, Any] = {}
    steps: dict[str, Any] = {}
    for task in TASKS:
        rows = []
        for row in manager.benchmarks(task)["rows"]:
            if (row["agent"], row["action_space"]) not in {
                ("jev", "hybrid"),
                ("jev", "gui_only"),
                ("codex_computer_use", "hybrid"),
            }:
                continue
            run_id = row["representative_run_id"]
            if not run_id:
                raise ValueError(f"Missing successful representative run for {task}")
            public_row = {key: row.get(key) for key in PUBLIC_ROW_FIELDS}
            rows.append(public_row)
            record = manager.steps(run_id)
            public_steps = []
            for step in record["steps"]:
                public_steps.append(
                    {
                        "number": step["number"],
                        "title": _safe_text(step["title"]),
                        "channel": step.get("channel"),
                        "capability": _safe_text(step["capability"]) if step.get("capability") else None,
                        "verified": step.get("verified"),
                    }
                )
            steps[run_id] = {
                "run_id": run_id,
                "source": record["source"],
                "terminal_verified": record["terminal_verified"],
                "steps": public_steps,
            }
        if len(rows) != 3:
            raise ValueError(f"Expected Jev Hybrid, Jev GUI Only and Codex Hybrid for {task}")
        benchmarks[task] = {"rows": rows}
    tasks = {
        task: {key: TASK_CATALOG[task][key] for key in ("title", "description", "steps", "hybrid_routes")}
        for task in TASKS
    }
    snapshot = {
        "schema_version": 1,
        "captured_on": date.today().isoformat(),
        "tasks": tasks,
        "benchmarks": benchmarks,
        "steps": steps,
    }
    _write_json(SNAPSHOT, snapshot)
    print(f"Wrote curated snapshot: {SNAPSHOT}")


def build(output: Path = ROOT / "dist-pages") -> None:
    """Build from tracked, reviewed inputs only; CI never reads ignored run data."""
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1 or set(snapshot.get("tasks", {})) != set(TASKS):
        raise ValueError("Invalid website snapshot")
    target = output.resolve()
    default = (ROOT / "dist-pages").resolve()
    if target.exists():
        if target != default:
            raise ValueError("Refusing to overwrite a non-default output directory")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(SOURCE / "icons", target / "static" / "icons")
    for name in ("app.js", "style.css", "logo-mark.svg"):
        shutil.copy2(SOURCE / name, target / "static" / name)
    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    marker = '<html lang="en">'
    if marker not in html:
        raise ValueError("Static site mode marker cannot be installed")
    (target / "index.html").write_text(
        html.replace(marker, '<html lang="en" data-site-mode="static">', 1), encoding="utf-8"
    )
    (target / ".nojekyll").touch()
    demos: dict[str, dict[str, str]] = {}
    (target / "media").mkdir()
    for task in TASKS:
        demos[task] = {}
        for mode in MODES:
            name = f"{task}-{mode}.mp4"
            source = MEDIA / name
            if not source.is_file():
                raise ValueError(f"Missing curated demo: {source}")
            shutil.copy2(source, target / "media" / name)
            demos[task]["gui_only" if mode == "gui-only" else mode] = f"media/{name}"
    for name in ("open-workspace-python.mp4", "open-workspace-python-poster.jpg"):
        source = MEDIA / name
        if not source.is_file():
            raise ValueError(f"Missing curated open-task media: {source}")
        shutil.copy2(source, target / "media" / name)
    _write_json(
        target / "data" / "bootstrap.json",
        {"tasks": snapshot["tasks"], "demos": demos, "captured_on": snapshot["captured_on"]},
    )
    for task in TASKS:
        _write_json(target / "data" / "benchmarks" / f"{task}.json", snapshot["benchmarks"][task])
    for run_id, record in snapshot["steps"].items():
        if not re.fullmatch(r"[a-zA-Z0-9-]+", run_id):
            raise ValueError("Invalid public run id")
        _write_json(target / "data" / "steps" / f"{run_id}.json", record)
    print(f"Built GitHub Pages site: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-snapshot", action="store_true", help="Export local run summaries for review"
    )
    parser.add_argument(
        "--build", action="store_true", help="Build the static site from the tracked snapshot"
    )
    args = parser.parse_args()
    if args.refresh_snapshot:
        refresh_snapshot()
    if args.build or not args.refresh_snapshot:
        build()
