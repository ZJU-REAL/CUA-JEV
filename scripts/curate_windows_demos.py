"""Promote a reviewed successful long run to the public Windows case catalog.

Only concise allowlisted metrics, candidate descriptions, and the selected video
are exported. Raw observations, policy requests, model responses, and traces stay
in ignored local directories. Run once per case after human video/content review.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "website" / "windows_demos.json"
MEDIA = ROOT / "website" / "media"
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
PRIVATE = re.compile(r"[A-Za-z]:\\|/Users/|/home/|apikey_|TYPESAFE_API_KEY|sk-[A-Za-z0-9]", re.I)
VIDEO_DURATION = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


def _public(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or PRIVATE.search(value):
        raise ValueError("Case metadata contains empty or non-public text")
    return value.strip()


def _video_duration(path: Path) -> float:
    import imageio_ffmpeg

    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True, text=True, check=False,
    )
    match = VIDEO_DURATION.search(result.stderr)
    if not match:
        raise ValueError("Reviewed video has no readable duration")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration < 5:
        raise ValueError("Reviewed video is unexpectedly short")
    return duration


def _trace_steps(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len({event.get("run_id") for event in events}) != 1:
        raise ValueError("A public case must come from exactly one episode")
    episode = [event["payload"] for event in events if event.get("kind") == "episode"]
    if len(episode) != 1 or episode[0].get("status") != "success":
        raise ValueError("A public case requires one successful terminal episode")
    choices: dict[str, str] = {}
    steps: list[dict[str, str]] = []
    verification_count = 0
    jev_calls = 0
    planner_model = ""
    jev_model = ""
    for event in sorted(events, key=lambda item: item.get("sequence", 0)):
        kind, payload = event.get("kind"), event.get("payload", {})
        if kind == "candidates":
            choices = {
                item["id"]: item.get("description", "")
                for item in payload.get("items", []) if isinstance(item, dict)
            }
        elif kind == "decision":
            if not str(payload.get("model", "")).startswith("jev-"):
                raise ValueError("A public Jev case contains a non-Jev decision")
            jev_calls += 1
            jev_model = payload["model"]
        elif kind == "commitment":
            if payload.get("channel") == "control":
                raise ValueError("Control sentinels must not be presented as external actions")
            description = _public(choices.get(payload.get("candidate_id"), ""))
            steps.append({
                "title": description,
                "channel": _public(payload.get("channel")),
                "capability": _public(payload.get("capability")),
            })
        elif kind == "verification":
            if payload.get("passed") is not True:
                raise ValueError("Every showcased action must have passed verification")
            verification_count += 1
        elif kind == "model_usage" and payload.get("role") == "planner":
            planner_model = _public(payload.get("model"))
    if not 17 <= len(steps) <= 23:
        raise ValueError("A Windows long-form demo requires 17–23 executed actions")
    if len(steps) != episode[0].get("steps") or verification_count != len(steps):
        raise ValueError("Trace action and verification counts disagree")
    if jev_calls != len(steps) or not planner_model:
        raise ValueError("Jev/model evidence is incomplete")
    return steps, {
        "jev_calls": jev_calls,
        "jev_model": jev_model,
        "planner_model": planner_model,
        "channels": dict(Counter(step["channel"] for step in steps)),
        "wall_time_s": round(episode[0]["duration_ms"] / 1000, 1),
    }


def curate(args: argparse.Namespace) -> dict[str, Any]:
    if not SLUG.fullmatch(args.id):
        raise ValueError("Case id must be a lowercase hyphenated slug")
    if not args.trace.is_file() or not args.metrics.is_file() or not args.video.is_file():
        raise ValueError("Trace, private metrics, and reviewed video must all exist")
    steps, trace = _trace_steps(args.trace)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("status") != "success" or any(
        metrics.get(key) != expected for key, expected in (
            ("actions", len(steps)), ("jev_calls", trace["jev_calls"]),
            ("channels", trace["channels"]),
        )
    ):
        raise ValueError("Private recording metrics do not match the trace")
    if not isinstance(metrics.get("model_calls"), int) or metrics["model_calls"] < 1:
        raise ValueError("Actual model call count is unavailable")
    if not isinstance(metrics.get("vlm_calls"), int) or metrics["vlm_calls"] < 0:
        raise ValueError("Actual VLM call count is unavailable")
    if abs(metrics.get("wall_time_ms", -1) / 1000 - trace["wall_time_s"]) > 0.11:
        raise ValueError("Recording and trace wall time differ")
    video_duration = _video_duration(args.video)
    speed = trace["wall_time_s"] / video_duration
    if not 0.85 <= speed <= 2:
        raise ValueError("Reviewed video duration is inconsistent with wall time")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    cases = catalog.get("cases", [])
    if catalog.get("schema_version") != 2 or len(cases) >= 4:
        raise ValueError("Windows catalog is invalid or already complete")
    if any(item["id"] == args.id for item in cases):
        raise ValueError("Case id is already published")
    apps = [_public(app) for app in args.app]
    if len(apps) < 2 or len(set(apps)) != len(apps):
        raise ValueError("A cross-app case needs at least two distinct app labels")
    app_by_channel = {
        "script": args.browser_app, "gui": args.browser_app, "mcp": "MCP reader",
        "cli": "Terminal", "api": "Files",
    }
    for step in steps:
        if step["capability"] == "artifact.open_vscode":
            step["app"] = "VS Code"
        elif step["capability"] == "artifact.open_notepad":
            step["app"] = "Notepad"
        else:
            step["app"] = app_by_channel.get(step["channel"], "Runtime")
        del step["capability"]

    media_name = f"windows-{args.id}.mp4"
    media_path = MEDIA / media_name
    if media_path.exists():
        raise ValueError("Refusing to overwrite curated public video")
    case = {
        "id": args.id, "verified": True,
        "title": _public(args.title), "summary": _public(args.summary),
        "apps": apps, "video": f"media/{media_name}",
        "actions": len(steps), "jev_calls": trace["jev_calls"],
        "model_calls": metrics["model_calls"], "vlm_calls": metrics["vlm_calls"],
        "jev_model": trace["jev_model"], "planner_model": trace["planner_model"],
        "wall_time_s": trace["wall_time_s"], "channels": trace["channels"],
        "recording_speed": "real-time playback" if speed < 1.05
        else f"{speed:.2f}× playback",
        "steps": steps,
    }
    # The source video has already been reviewed. Only this media file and the
    # allowlisted case fields leave ignored local storage.
    shutil.copy2(args.video, media_path)
    cases.append(case)
    CATALOG.write_text(
        json.dumps({"schema_version": 2, "cases": cases}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--app", action="append", required=True)
    parser.add_argument("--browser-app", default="Edge")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    case = curate(args)
    print(f"Curated {case['id']}: {case['actions']} verified actions, "
          f"{case['jev_calls']} Jev calls, {case['model_calls']} planner calls")


if __name__ == "__main__":
    main()
