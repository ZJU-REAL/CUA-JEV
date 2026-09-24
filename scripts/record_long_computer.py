"""Record a genuine model + Jev open-computer episode in a visible Edge window.

Only a successful, verified episode is promoted to the requested demo path.
Failed takes and private traces remain in ignored local ``runs/`` paths.
This recorder captures the Edge window plus a route/status HUD, not the desktop.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from record_open_workspace import WindowRecorder, stage_window

from cua_jev.artifact_surface import ArtifactSurface
from cua_jev.cli import _open_computer_runner, _parser
from cua_jev.config import load_local_env
from cua_jev.mcp_surface import McpToolSurface
from cua_jev.model_planner import ChatModelPlanner
from cua_jev.open_browser import OpenBrowserTask
from cua_jev.open_computer import OpenComputerTask


def main(argv: list[str] | None = None) -> int:
    recorder_parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    recorder_parser.add_argument("--output", type=Path)
    recorder_parser.add_argument("--metrics-output", type=Path)
    recorder_parser.add_argument("--fps", type=int, default=12)
    recorder_parser.add_argument("--help", action="store_true")
    recording, task_argv = recorder_parser.parse_known_args(argv)
    if recording.help:
        print("Usage: python scripts/record_long_computer.py --output DEMO.mp4 ")
        print("       [--fps 12] <the same options as cua-jev open-computer>")
        return 0
    if recording.output is None:
        recorder_parser.error("--output is required")
    if not 1 <= recording.fps <= 30:
        recorder_parser.error("--fps must be 1-30")
    if recording.output.exists():
        recorder_parser.error("refusing to overwrite an existing demo video")
    if recording.metrics_output and recording.metrics_output.exists():
        recorder_parser.error("refusing to overwrite existing run metrics")

    load_local_env()
    args = _parser().parse_args(["open-computer", *task_argv])
    if args.window_title:
        recorder_parser.error("this Edge-window recorder does not capture a desktop window")
    if args.browser_vision_model or args.desktop_vision_model:
        recorder_parser.error("vision recording needs a separate privacy-reviewed path")
    if args.policy != "jev":
        recorder_parser.error("publishable demo recording requires --policy jev")
    if not os.getenv("CUA_JEV_MODEL_API_KEY") or not os.getenv("TYPESAFE_API_KEY"):
        recorder_parser.error("set the model and Jev keys locally, never in the command line")
    if args.mcp_profile and not args.allow_mcp_actions:
        recorder_parser.error("MCP profile needs --allow-mcp-actions")
    if (args.open_artifact_vscode or args.open_artifact_notepad) and not args.artifact_path:
        recorder_parser.error("artifact editor handoff needs --artifact-path")
    if args.open_artifact_vscode and args.open_artifact_notepad:
        recorder_parser.error("select only one artifact editor")
    args.headed_browser = True
    planner = ChatModelPlanner(
        args.model_base_url, args.model,
        api_key=os.environ["CUA_JEV_MODEL_API_KEY"],
        allow_insecure_http=args.allow_insecure_model_http,
        use_env_proxy=args.use_env_proxy,
    )
    browser = OpenBrowserTask(
        goal=args.goal, start_url=args.url, planner=planner,
        headed=True, browser_channel=args.browser_channel,
        allow_form_input=args.allow_form_input,
        allow_external_actions=args.allow_browser_actions,
    )
    task = OpenComputerTask(
        goal=args.goal, browser=browser, planner=planner,
        local_tool_root=args.local_tool_root,
        mcp_surface=McpToolSurface.from_profile(args.mcp_profile)
        if args.mcp_profile else None,
        artifact_surface=ArtifactSurface(
            args.artifact_path, allow_open_vscode=args.open_artifact_vscode,
            allow_open_notepad=args.open_artifact_notepad,
        )
        if args.artifact_path else None,
        mcp_current_page_only=args.mcp_current_page_only,
        mcp_read_for_visits=args.mcp_read_for_visits,
        min_verified_sources=args.min_verified_sources,
        required_source_titles=args.require_source_title_contains,
        required_visits=args.require_visit_url_contains,
        required_artifact_contains=args.artifact_contains,
        required_url_contains=args.require_url_contains,
        required_capabilities=args.require_capability,
    )
    state = {
        "task": "BROWSER + TOOLS", "profile_label": "MODEL + JEV",
        "step": 0, "total": 0, "channel": "—", "status": "running",
        "window_handle": 0,
    }
    original_evaluate = task.evaluate

    def evaluate_with_hud(observation, candidate, receipt, verification):
        outcome = original_evaluate(observation, candidate, receipt, verification)
        state["step"] += 1
        state["channel"] = str(candidate.channel)
        state["status"] = "verified" if verification.passed else "replan"
        if candidate.capability in {"artifact.open_vscode", "artifact.open_notepad"} and verification.passed:
            handle = task.artifact_surface.opened_window_handle
            if handle is None:
                raise RuntimeError("verified editor has no new window handle")
            state["window_handle"] = stage_window(
                browser, "", handle=handle,
            )
        return outcome

    task.evaluate = evaluate_with_hud  # type: ignore[method-assign]
    staging = Path("runs/recording-work") / f"open-computer-{uuid4().hex}.mp4"
    recorder = WindowRecorder(staging, state, fps=recording.fps, inset_px=12)
    started = False
    result = None
    try:
        task.reset()
        title = task.browser.page.title()
        state["window_handle"] = stage_window(browser, rf".*{re.escape(title)}.*")
        # Edge may surface a first-run translation flyout above the real page.
        # Dismiss it before recording; this is staging, not a task action.
        browser.screen.press("esc")
        time.sleep(0.4)
        recorder.start()
        started = True
        result = _open_computer_runner(args).run(task, reset=False)
        state["status"] = "success" if result.success else "failed"
        time.sleep(1.2)
    finally:
        if started:
            recorder.stop()
        task.close()
        planner.close()

    if result is None or not result.success:
        print(f"Episode not published: {result.status if result else 'error'}; raw={staging.resolve()}")
        return 1
    recording.output.parent.mkdir(parents=True, exist_ok=True)
    if recording.output.exists():
        raise FileExistsError("demo output appeared during the run; refusing to overwrite")
    staging.replace(recording.output)
    if recording.metrics_output is not None:
        recording.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        recording.metrics_output.write_text(json.dumps({
            "status": "success", "actions": len(result.steps),
            "jev_calls": sum(
                step.decision.model.startswith("jev-") for step in result.steps
            ),
            "model_calls": task.planner_calls,
            "vlm_calls": 0,
            "wall_time_ms": result.duration_ms,
            "channels": result.channel_counts,
        }, indent=2) + "\n", encoding="utf-8")
    print(
        f"status=success steps={len(result.steps)} wall_ms={result.duration_ms:.0f} "
        f"planner_calls={task.planner_calls} model_tokens="
        f"{planner.usage_totals.get('total_tokens', 'n/a')} "
        f"video={recording.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
