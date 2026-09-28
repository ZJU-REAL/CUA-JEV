"""Record a real Mac model + Jev task as separate, explicitly scoped window clips.

Usage: python scripts/record_macos_computer.py --output artifacts/demos/mac-take \
    <open-computer options, including a native window and TextEdit handoff>

The output is a NEW DIRECTORY containing raw MP4s, a coverage manifest, metrics,
and a private JSONL trace. It is not a continuous cross-application movie and is
not automatically published. Failed takes stay under runs/recording-work/.
The runner owns --trace and --record-desktop; do not supply those options.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from cua_jev.artifact_surface import ArtifactSurface
from cua_jev.cli import _chat_planner, _open_computer_runner, _parser
from cua_jev.config import load_local_env
from cua_jev.macos_bridge import MacBridge
from cua_jev.macos_surface import MacAccessibilitySurface
from cua_jev.mcp_surface import McpToolSurface
from cua_jev.open_browser import OpenBrowserTask
from cua_jev.open_computer import OpenComputerTask
from cua_jev.trace_analysis import analyze_traces

BROWSER_BUNDLES = {
    "chrome": "com.google.Chrome", "msedge": "com.microsoft.edgemac",
}


def _browser_bundle(browser: Any, channel: str) -> str:
    if channel != "chromium":
        return BROWSER_BUNDLES[channel]
    # Recent Playwright bundles use Chrome for Testing; older ones use Chromium.
    # Read the launched runtime's plist instead of matching every Chrome window.
    executable = Path(browser._playwright.chromium.executable_path)
    for parent in executable.parents:
        if parent.suffix == ".app":
            with (parent / "Contents/Info.plist").open("rb") as handle:
                bundle = plistlib.load(handle).get("CFBundleIdentifier")
            if bundle in {"org.chromium.Chromium", "com.google.chrome.for.testing"}:
                return bundle
            break
    raise RuntimeError("cannot identify the bundled browser's macOS application")


def _close_all(operations: tuple[Callable[[], None], ...]) -> None:
    failure: BaseException | None = None
    for operation in operations:
        try:
            operation()
        except BaseException as exc:
            failure = failure or exc
    if failure is not None:
        raise failure


class WindowSegments:
    """One helper records one fixed window at a time, never a desktop region."""

    def __init__(self, directory: Path, *, bridge: Any | None = None) -> None:
        self.directory = directory
        self.bridge = bridge or MacBridge()
        self.segments: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.closed = False

    def start(self, role: str, *, title_pattern: str, bundle_id: str | None = None,
              expected_window_id: int | None = None) -> None:
        if self.active is not None or self.closed:
            raise RuntimeError("finish the active recording before changing its window")
        selected = self.bridge.call("select", title_pattern=title_pattern, bundle_id=bundle_id)
        if expected_window_id is not None and selected["window_id"] != expected_window_id:
            raise RuntimeError("recording window does not match the independently verified window ID")
        path = self.directory / f"{role}.mp4"
        if path.exists():
            raise FileExistsError("refusing to overwrite a recording segment")
        segment = {
            "role": role, "file": path.name, "window_id": selected["window_id"],
            "bundle_id": selected["bundle_id"], "initial_title": selected["window_title"],
            "coverage": "one pinned native window; no desktop, audio, or cursor",
            "start_requested_at": time.time(), "started_at": None,
            "stop_requested_at": None, "stopped_at": None, "finalized": False,
        }
        self.segments.append(segment)
        try:
            result = self.bridge.call("record_start", path=str(path.resolve()))
            self.active = segment
            segment["started_at"] = time.time()
            if result.get("recording") is not True or result.get("window_id") != selected["window_id"]:
                raise RuntimeError("native recording did not confirm the selected window")
            segment["pixel_size"] = [result.get("width"), result.get("height")]
        except BaseException as exc:
            segment["error"] = f"{type(exc).__name__}: {exc}"
            raise

    def stop(self) -> None:
        if self.active is None:
            return
        segment, self.active = self.active, None
        segment["stop_requested_at"] = time.time()
        try:
            result = self.bridge.call("record_stop")
            if result.get("finalized") is not True:
                raise RuntimeError("native recording did not confirm finalization")
            path = self.directory / segment["file"]
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("finalized recording segment is missing or empty")
            segment["finalized"] = True
        except BaseException as exc:
            segment["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            segment["stopped_at"] = time.time()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            _close_all((self.stop, self.bridge.close))


class MacRecording:
    def __init__(self, task: OpenComputerTask, args: argparse.Namespace, directory: Path,
                 *, final_hold: float = 2, bridge_factory: Callable = MacBridge) -> None:
        self.task, self.args = task, args
        self.browser = WindowSegments(directory, bridge=bridge_factory())
        self.desktop = WindowSegments(directory, bridge=bridge_factory())
        self.final_hold = final_hold
        self.transitions: list[dict[str, Any]] = []
        self.editor_started = False
        self.editor_handoff: dict[str, Any] | None = None

    @property
    def segments(self) -> list[dict[str, Any]]:
        return [*self.browser.segments, *self.desktop.segments]

    def start(self) -> None:
        # All initial selection/focus staging precedes the first model observation.
        native_id = self.task.providers["d"].capture().native["window_id"]
        title = self.task.browser.page.title()
        if not title.strip():
            raise RuntimeError("the browser must have an identifiable title before recording")
        self.browser.start(
            "browser", title_pattern=re.escape(title),
            bundle_id=_browser_bundle(self.task.browser, self.args.browser_channel),
        )
        self.desktop.start(
            "native", title_pattern=self.args.window_title, bundle_id=self.args.app_bundle_id,
            expected_window_id=native_id,
        )

    def check_permissions(self) -> None:
        health = self.browser.bridge.call("health")
        if health.get("accessibility") is not True or health.get("screen_recording") is not True:
            raise RuntimeError("Accessibility and Screen Recording permissions are required")

    def after_evaluate(self, candidate: Any, receipt: Any, verification: Any, outcome: Any) -> None:
        if candidate.capability == "artifact.open_textedit" and verification.passed:
            expected = self.task.artifact_surface.opened_window_handle
            title = receipt.output.get("window_title")
            if type(expected) is not int or not isinstance(title, str) or not title:
                raise RuntimeError("verified TextEdit handoff has no exact window identity")
            self.editor_handoff = {
                "window_id": expected, "title": title, "decision_id": receipt.decision_id,
            }
        # The model may open the artifact before finishing another requirement.
        # Keep the browser track until independent terminal acceptance passes.
        if not outcome.success or self.editor_handoff is None or self.editor_started:
            return
        expected, title = self.editor_handoff["window_id"], self.editor_handoff["title"]
        self.browser.stop()
        self.browser.start(
            "textedit", title_pattern=f"^{re.escape(title)}$",
            bundle_id="com.apple.TextEdit", expected_window_id=expected,
        )
        self.editor_started = True
        self.transitions.append({
            "kind": "verified_textedit_handoff", "at": time.time(),
            "decision_id": self.editor_handoff["decision_id"], "window_id": expected,
            "note": "TextEdit capture starts after exact-document verification and task acceptance; "
                    "opening is not captured.",
        })
        time.sleep(self.final_hold)

    def close(self) -> None:
        _close_all((self.browser.close, self.desktop.close))


def _attach_recording(task: OpenComputerTask, recording: MacRecording) -> None:
    original_evaluate, original_close = task.evaluate, task.close
    closed = False

    def evaluate(observation, candidate, receipt, verification):
        outcome = original_evaluate(observation, candidate, receipt, verification)
        recording.after_evaluate(candidate, receipt, verification, outcome)
        return outcome

    def close():
        nonlocal closed
        if not closed:
            closed = True
            # EpisodeRunner closes the task before returning. Finalize captures
            # while their browser/native windows still exist, also on failures.
            _close_all((recording.close, original_close))

    task.evaluate, task.close = evaluate, close


def _build_task(args: argparse.Namespace, planner: Any) -> OpenComputerTask:
    return OpenComputerTask(
        goal=args.goal, planner=planner,
        browser=OpenBrowserTask(
            goal=args.goal, start_url=args.url, planner=planner, headed=True,
            browser_channel=args.browser_channel, allow_form_input=args.allow_form_input,
            allow_external_actions=args.allow_browser_actions,
        ),
        desktop_surface=MacAccessibilitySurface(
            window_title_re=args.window_title, bundle_id=args.app_bundle_id,
            allow_text_input=args.allow_window_text_input, allow_button_actions=args.allow_window_actions,
        ),
        local_tool_root=args.local_tool_root,
        mcp_surface=McpToolSurface.from_profile(args.mcp_profile) if args.mcp_profile else None,
        artifact_surface=ArtifactSurface(args.artifact_path, allow_open_textedit=True),
        mcp_current_page_only=args.mcp_current_page_only, mcp_read_for_visits=args.mcp_read_for_visits,
        min_verified_sources=args.min_verified_sources,
        required_source_titles=args.require_source_title_contains,
        required_visits=args.require_visit_url_contains,
        required_artifact_contains=args.artifact_contains, required_url_contains=args.require_url_contains,
        required_window_title_contains=args.require_window_title_contains,
        required_window_text_contains=args.require_window_text_contains,
        required_capabilities=args.require_capability,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _promote(staging: Path, destination: Path) -> None:
    # mkdir is create-only, including a destination created during the episode.
    destination.mkdir(parents=True, exist_ok=False)
    try:
        # Keep the private source intact if a cross-device copy fails.
        import shutil

        for source in staging.iterdir():
            with source.open("rb") as reader, (destination / source.name).open("xb") as writer:
                shutil.copyfileobj(reader, writer)
    except Exception:
        raise RuntimeError(f"bundle copy incomplete; original evidence remains in {staging}") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--output", type=Path, help="new directory for a locally reviewable raw bundle")
    parser.add_argument("--final-hold", type=float, default=2, help="TextEdit terminal hold, 0.5–5 seconds")
    parser.add_argument("--help", action="store_true")
    options, task_argv = parser.parse_known_args(argv)
    if options.help:
        parser.print_help()
        print("Task arguments are the same as cua-jev open-computer --help.")
        return 0
    if sys.platform != "darwin":
        parser.error("native recording requires macOS")
    if options.output is None or options.output.exists() or options.output.is_symlink():
        parser.error("--output must name a new directory")
    if not 0.5 <= options.final_hold <= 5:
        parser.error("--final-hold must be 0.5–5 seconds")
    if any(item.split("=", 1)[0] in {"--trace", "--record-desktop"} for item in task_argv):
        parser.error("the bundle owns trace and recording paths; omit --trace and --record-desktop")
    args = _parser().parse_args(["open-computer", *task_argv])
    if args.policy != "jev":
        parser.error("this recorder requires --policy jev")
    if args.browser_vision_model or args.desktop_vision_model or args.allow_screenshot_upload:
        parser.error("this text-model recorder does not upload screenshots")
    if args.allow_browser_visual_clicks or args.allow_desktop_visual_clicks:
        parser.error("visual grounding is outside this recording path")
    if not (args.window_title and args.require_window_text_contains and args.allow_window_text_input
            and args.allow_window_actions and args.local_tool_root):
        parser.error("provide a native window/text gate, native action opt-ins, and --local-tool-root")
    if not args.artifact_path or not args.open_artifact_textedit:
        parser.error("provide --artifact-path and --open-artifact-textedit")
    if args.open_artifact_notepad or args.open_artifact_vscode:
        parser.error("this recorder supports only the exact-document TextEdit handoff")
    if bool(args.mcp_profile) != args.allow_mcp_actions or args.max_steps < 1:
        parser.error("MCP needs its matching opt-in; --max-steps must be positive")
    for capability in ("cli.python_version", "desktop.fill", "desktop.click", "artifact.open_textedit"):
        if capability not in args.require_capability:
            args.require_capability.append(capability)
    load_local_env()
    if not os.getenv("CUA_JEV_MODEL_API_KEY") or not os.getenv("TYPESAFE_API_KEY"):
        parser.error("configure model and Jev keys locally; never pass them on the command line")
    staging = Path("runs/recording-work") / f"macos-{uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    args.trace = str(staging / "trace.jsonl")
    planner = task = recording = result = None
    failure = None
    started = time.time()
    try:
        planner = _chat_planner(args)
        task = _build_task(args, planner)
        recording = MacRecording(task, args, staging, final_hold=options.final_hold)
        _attach_recording(task, recording)
        recording.check_permissions()
        task.reset()
        recording.start()
        result = _open_computer_runner(args).run(task, reset=False)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            _close_all(tuple(resource.close for resource in (task, planner) if resource is not None))
        except BaseException as exc:
            failure = failure or exc
    segments = recording.segments if recording else []
    summary = None
    jev_requests = 0
    try:
        if Path(args.trace).is_file():
            summary = analyze_traces([args.trace])
            for line in Path(args.trace).read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if event.get("kind") == "policy_exchange":
                    attempts = event.get("payload", {}).get("attempts")
                    if type(attempts) is int and attempts >= 0:
                        jev_requests += attempts
    except Exception as exc:
        failure = failure or exc
    ready = bool(result and result.success and recording.editor_started and len(segments) == 3
                 and all(item["finalized"] for item in segments) and summary is not None and failure is None)
    steps = result.steps if result else ()
    metrics = {
        "episode": result.to_dict() if result else None,
        "recording_complete": ready, "actions": len(steps),
        "verified_actions": sum(step.verification.passed for step in steps),
        "jev_decisions": sum(step.decision.model.startswith("jev-") for step in steps),
        "jev_reported_requests": jev_requests,
        "planner_calls": getattr(task, "planner_calls", 0),
        "planner_requests": getattr(planner, "planning_requests", 0),
        "planner_usage": getattr(planner, "usage_totals", {}),
        "planner_wall_ms": getattr(planner, "planning_wall_ms", None),
        "vlm_requests": 0, "channels": dict(Counter(str(step.receipt.channel) for step in steps)),
        "native_routes": dict(Counter(
            step.receipt.output.get("route", "unknown") for step in steps
            if step.receipt.capability.startswith("desktop.") and step.receipt.success
        )),
        "usage_note": "Jev request attempts count recorded policy exchanges only; an interrupted/failed "
                      "request may have no exchange. Provider usage and costs can remain unknown.",
        "trace_summary": summary, "cost_usd": None,
    }
    _write_json(staging / "metrics.json", metrics)
    _write_json(staging / "manifest.json", {
        "schema_version": 1, "kind": "separate_native_window_recordings", "private": True,
        "recording_complete": ready, "started_at": started, "ended_at": time.time(),
        "clock": "Unix wall-clock seconds; request/ack bounds, not frame-accurate synchronization",
        "coverage_note": "Browser and native clips overlap in time. TextEdit begins only after verified "
                         "handoff. CLI/MCP actions have trace receipts, not terminal-window footage. "
                         "This is not one continuous cross-application recording.",
        "segments": segments, "transitions": recording.transitions if recording else [],
        "trace": "trace.jsonl", "metrics": "metrics.json",
        "artifact": str(args.artifact_path.resolve()),
        "failure": f"{type(failure).__name__}: {failure}" if failure else None,
    })
    if failure is not None and not isinstance(failure, Exception):
        print(f"Interrupted; private evidence remains in {staging.resolve()}")
        raise failure
    if not ready:
        print(f"Take not promoted; inspect private evidence in {staging.resolve()}")
        return 1
    _promote(staging, options.output.resolve())
    print(f"Verified episode and three raw window clips: {options.output.resolve()}; "
          "review before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
