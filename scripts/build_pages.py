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
WINDOWS_DEMOS = WEBSITE / "windows_demos.json"
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


def _windows_cases() -> tuple[list[dict[str, Any]], bool]:
    data = json.loads(WINDOWS_DEMOS.read_text(encoding="utf-8"))
    if data.get("schema_version") != 2 or not isinstance(data.get("cases"), list):
        raise ValueError("Invalid Windows case catalog")
    cases = data["cases"]
    if len(cases) > 4 or len({item.get("id") for item in cases}) != len(cases):
        raise ValueError("Windows case catalog must contain at most four unique cases")
    for item in cases:
        if item.get("verified") is not True or not 17 <= item.get("actions", -1) <= 23:
            raise ValueError("Only verified, roughly twenty-action cases may be published")
        if item.get("jev_calls") != item["actions"] or not 1 <= item.get("model_calls", 0):
            raise ValueError("Windows case decision or model counts are inconsistent")
        if (
            len(item.get("steps", [])) != item["actions"]
            or sum(item.get("channels", {}).values()) != item["actions"]
        ):
            raise ValueError("Windows case trace does not match its action mix")
        for key in ("id", "title", "summary"):
            _safe_text(item[key])
        for app in item.get("apps", []):
            _safe_text(app)
        for step in item["steps"]:
            _safe_text(step["title"])
            _safe_text(step["channel"])
            _safe_text(step.get("app", ""))
        media_path = item.get("video", "")
        if not re.fullmatch(r"media/windows-[a-z0-9-]+\.mp4", media_path):
            raise ValueError("Windows case video must be a curated relative media path")
        if not (WEBSITE / media_path).is_file():
            raise ValueError(f"Missing reviewed Windows recording: {media_path}")
    return cases, len(cases) == 4


def refresh_snapshot() -> None:
    """Explicitly export allowlisted display fields from measured local records."""
    from cua_jev.cost import (
        CODEX_ENTERPRISE_USD_PER_MILLION,
        CODEX_MODEL,
        JEV_INPUT_USD_PER_MILLION,
        JEV_MODEL,
    )
    from cua_jev.ui.manager import TASK_CATALOG, RunManager

    manager = RunManager(ROOT)
    benchmarks: dict[str, Any] = {}
    steps: dict[str, Any] = {}
    jev_tokens: dict[str, float] = {}
    codex_tokens: dict[str, dict[str, int]] = {}
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
            if row["agent"] == "jev" and row["action_space"] == "hybrid":
                jev_tokens[task] = row["median_input_tokens"]
            if row["agent"] == "codex_computer_use":
                if row["successful_samples"] != 1:
                    raise ValueError("Review Codex cost methodology before publishing multiple pilots")
                codex_tokens[task] = {
                    "input": row["median_input_tokens"],
                    "cached_input": row["median_cached_input_tokens"],
                    "output": row["median_output_tokens"],
                }
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
        if len(rows) != 3 or jev_tokens.get(task) is None or any(
            value is None for value in codex_tokens.get(task, {}).values()
        ) or task not in codex_tokens:
            raise ValueError(f"Expected Jev Hybrid, Jev GUI Only and Codex Hybrid for {task}")
        benchmarks[task] = {"rows": rows}
    tasks = {
        task: {key: TASK_CATALOG[task][key] for key in ("title", "description", "steps", "hybrid_routes")}
        for task in TASKS
    }
    snapshot = {
        "schema_version": 1,
        "captured_on": date.today().isoformat(),
        "cost_evidence": {
            "note": (
                "Model-cost estimates from published rates, not observed bills. Jev values "
                "use successful Hybrid medians; Codex uses one pilot per task in an existing conversation."
            ),
            "jev": {
                "model": JEV_MODEL,
                "input_usd_per_million": JEV_INPUT_USD_PER_MILLION,
                "rate_source": "https://docs.typesafe.ai/models",
                "median_hybrid_input_tokens": jev_tokens,
            },
            "codex": {
                "model": CODEX_MODEL,
                "usd_per_million": dict(zip(
                    ("uncached_input", "cached_input", "output"),
                    CODEX_ENTERPRISE_USD_PER_MILLION,
                    strict=True,
                )),
                "rate_source": "https://help.openai.com/en/articles/20001415-chatgpt-rate-card-enterprise-token-based-pricing",
                "pilot_tokens": codex_tokens,
            },
            "limits": (
                "Codex token events were allocated to task windows in a long conversation; "
                "estimates omit tooling, infrastructure, and subscription costs. "
                "This is not a fresh-start per-task bill."
            ),
        },
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
    windows_cases, windows_ready = _windows_cases()
    target = output.resolve()
    default = (ROOT / "dist-pages").resolve()
    if target.exists():
        if target != default:
            raise ValueError("Refusing to overwrite a non-default output directory")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copytree(SOURCE / "icons", target / "static" / "icons")
    for name in ("app.js", "style.css", "home.js", "home.css", "logo-mark.svg", "open-task-loop.svg"):
        shutil.copy2(SOURCE / name, target / "static" / name)
    html = (SOURCE / "index.html").read_text(encoding="utf-8")
    marker = '<html lang="en">'
    if marker not in html:
        raise ValueError("Static site mode marker cannot be installed")
    early = html.replace(marker, '<html lang="en" data-site-mode="static">', 1)
    (target / "early-work.html").write_text(early, encoding="utf-8")
    home = (SOURCE / "home.html").read_text(encoding="utf-8")
    (target / "preview.html").write_text(home, encoding="utf-8")
    (target / "index.html").write_text(home if windows_ready else early, encoding="utf-8")
    (target / ".nojekyll").touch()
    _write_json(target / "data" / "windows_demos.json", {
        "schema_version": 2, "cases": windows_cases,
    })
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
    for name in ("open-workspace-research.mp4", "open-workspace-research-poster.jpg"):
        source = MEDIA / name
        if not source.is_file():
            raise ValueError(f"Missing curated open-task media: {source}")
        shutil.copy2(source, target / "media" / name)
    for item in windows_cases:
        media_path = item["video"]
        shutil.copy2(WEBSITE / media_path, target / media_path)
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
