"""Publish a curated, read-only GitHub Pages snapshot without exposing local runs."""

from __future__ import annotations

import argparse
import json
import math
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
MAC_DEMOS = WEBSITE / "mac_demos.json"
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
        poster_path = item.get("poster")
        if poster_path is not None:
            if not isinstance(poster_path, str) or not re.fullmatch(
                r"media/windows-[a-z0-9-]+\.jpg", poster_path
            ):
                raise ValueError("Windows case poster must be a curated relative media path")
            if not (WEBSITE / poster_path).is_file():
                raise ValueError(f"Missing reviewed Windows poster: {poster_path}")
    return cases, len(cases) == 4


def _mac_cases() -> list[dict[str, Any]]:
    """Accept one explicitly reviewed Mac recording, never raw run metadata."""
    if not MAC_DEMOS.exists():
        return []
    data = json.loads(MAC_DEMOS.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict) or set(data) != {"schema_version", "cases"}
        or type(data["schema_version"]) is not int or data["schema_version"] != 1
        or not isinstance(data["cases"], list) or len(data["cases"]) > 1
    ):
        raise ValueError("Invalid macOS case catalog")
    fields = {
        "id", "verified", "title", "summary", "apps", "video", "poster", "actions",
        "jev_calls", "model_calls", "vlm_calls", "wall_time_s", "channels", "routes",
        "planner_model", "jev_model", "recording_scope", "timing_note", "steps",
    }
    channels = {"script", "api", "gui", "cli", "mcp"}

    def public_text(value: Any, limit: int = 2000) -> None:
        text = _safe_text(value)
        if not text.strip() or len(text) > limit:
            raise ValueError("Invalid macOS public text")

    def count(value: Any) -> bool:
        return type(value) is int and 0 <= value <= 1_000_000

    for item in data["cases"]:
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError("macOS case must contain only the reviewed public fields")
        if item["id"] != "python-onboarding" or item["verified"] is not True:
            raise ValueError("Only the verified macOS onboarding case may be published")
        for key in ("title", "summary", "planner_model", "jev_model", "recording_scope", "timing_note"):
            public_text(item[key])
        if not item["jev_model"].startswith("jev-") or "fallback" in item["jev_model"].casefold():
            raise ValueError("macOS showcase requires a Jev-selected run")
        if not isinstance(item["apps"], list) or not 1 <= len(item["apps"]) <= 10:
            raise ValueError("Invalid macOS application list")
        for app in item["apps"]:
            public_text(app, 100)
        if (
            not all(count(item[key]) for key in ("actions", "jev_calls", "model_calls", "vlm_calls"))
            or not 1 <= item["actions"] <= 100 or item["jev_calls"] < item["actions"]
            or item["model_calls"] < 1
            or type(item["wall_time_s"]) not in (int, float)
            or not math.isfinite(item["wall_time_s"]) or item["wall_time_s"] <= 0
        ):
            raise ValueError("Invalid macOS action, request, or timing metrics")
        for key, allowed in (("channels", channels), ("routes", {"ax", "gui"})):
            values = item[key]
            if not isinstance(values, dict) or not set(values) <= allowed or not all(
                count(value) for value in values.values()
            ):
                raise ValueError("Invalid macOS channel or native-route counts")
        if sum(item["channels"].values()) != item["actions"]:
            raise ValueError("macOS channel counts do not match the actions")
        steps = item["steps"]
        if not isinstance(steps, list) or len(steps) != item["actions"]:
            raise ValueError("macOS steps do not match the actions")
        actual_channels = dict.fromkeys(channels, 0)
        actual_routes = {"ax": 0, "gui": 0}
        for step in steps:
            if (
                not isinstance(step, dict) or not {"title", "channel"} <= set(step)
                or not set(step) <= {"title", "channel", "app", "route"}
                or not isinstance(step["channel"], str) or step["channel"] not in channels
            ):
                raise ValueError("Invalid macOS public action step")
            public_text(step["title"])
            if "app" in step:
                public_text(step["app"], 100)
            route = step.get("route")
            if "route" in step and (
                not isinstance(route, str) or route not in actual_routes
                or step["channel"] != {"ax": "api", "gui": "gui"}[route]
            ):
                raise ValueError("Invalid macOS native route for action channel")
            if step["channel"] == "gui" and route != "gui":
                raise ValueError("macOS GUI steps must identify their native route")
            actual_channels[step["channel"]] += 1
            if route:
                actual_routes[route] += 1
        if any(actual_channels[key] != item["channels"].get(key, 0) for key in channels) or any(
            actual_routes[key] != item["routes"].get(key, 0) for key in actual_routes
        ):
            raise ValueError("macOS steps disagree with channel or native-route counts")
        for key, extension in (("video", "mp4"), ("poster", "jpg")):
            if item[key] != f"media/macos-python-onboarding.{extension}":
                raise ValueError("macOS media must use its curated relative path")
            source = WEBSITE / item[key]
            if source.is_symlink() or not source.resolve().is_relative_to(WEBSITE.resolve()):
                raise ValueError("macOS media must remain inside the reviewed website directory")
            if not source.is_file():
                raise ValueError(f"Missing reviewed macOS {key}")
    return data["cases"]


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
    mac_cases = _mac_cases()
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
    (target / "index.html").write_text(home if windows_ready or mac_cases else early, encoding="utf-8")
    (target / ".nojekyll").touch()
    _write_json(target / "data" / "windows_demos.json", {
        "schema_version": 2, "cases": windows_cases,
    })
    _write_json(target / "data" / "mac_demos.json", {"schema_version": 1, "cases": mac_cases})
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
    for item in windows_cases + mac_cases:
        media_path = item["video"]
        shutil.copy2(WEBSITE / media_path, target / media_path)
        if item.get("poster"):
            shutil.copy2(WEBSITE / item["poster"], target / item["poster"])
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
