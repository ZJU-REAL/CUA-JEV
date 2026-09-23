"""Record one genuine model + Jev browser-to-VS-Code episode on Windows.

The runner starts the capture only after the real Edge window is ready, so
initial desktop frames are not published. Review the raw video before copying
it into website/media/.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from record_v2_demos import DesktopRecorder

from cua_jev.cli import _open_workspace_runner
from cua_jev.config import load_local_env
from cua_jev.model_planner import ChatModelPlanner
from cua_jev.open_workspace import OpenWorkspaceTask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--required-source-text", action="append", default=[])
    parser.add_argument("--trace", default="runs/open-workspace-recorded.jsonl")
    parser.add_argument("--output", default="artifacts/demos/open-workspace.mp4")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    args.policy = "jev"
    args.use_env_proxy = False
    load_local_env()
    key = os.getenv("CUA_JEV_MODEL_API_KEY")
    if not key or not os.getenv("TYPESAFE_API_KEY"):
        parser.error("set local CUA_JEV_MODEL_API_KEY and TYPESAFE_API_KEY")

    planner = ChatModelPlanner(
        args.model_base_url, args.model, api_key=key,
        allow_insecure_http=args.allow_insecure_model_http,
    )
    task = OpenWorkspaceTask(
        goal=args.goal, start_url=args.url, note_path=Path(args.note),
        planner=planner, required_source_texts=args.required_source_text, maximize_editor=True,
    )
    state = {
        "task": "EDGE → VS CODE", "profile_label": "MODEL + JEV · OPEN TASK",
        "step": 0, "total": 0, "channel": "—", "status": "running",
    }
    original_evaluate = task.evaluate

    def evaluate_with_hud(observation, candidate, receipt, verification):
        evaluation = original_evaluate(observation, candidate, receipt, verification)
        state["step"] += 1
        state["channel"] = str(candidate.channel)
        state["status"] = "verified" if verification.passed else "replan"
        return evaluation

    task.evaluate = evaluate_with_hud  # type: ignore[method-assign]
    recorder = DesktopRecorder(Path(args.output), state, fps=args.fps)
    recording = False
    try:
        task.reset()
        task.screen.focus(rf".*{re.escape(task.page.title())}.*", maximize=True)
        recorder.start()
        recording = True
        previous_hold = os.environ.get("CUA_JEV_COMPLETION_HOLD_SECONDS")
        os.environ["CUA_JEV_COMPLETION_HOLD_SECONDS"] = "2"
        try:
            result = _open_workspace_runner(args).run(task, reset=False)
        finally:
            if previous_hold is None:
                os.environ.pop("CUA_JEV_COMPLETION_HOLD_SECONDS", None)
            else:
                os.environ["CUA_JEV_COMPLETION_HOLD_SECONDS"] = previous_hold
        state["status"] = "success" if result.success else "failed"
        time.sleep(1.2)
        print(
            f"status={result.status} steps={len(result.steps)} wall_ms={result.duration_ms:.0f} "
            f"planner_calls={task.planner_calls} video={Path(args.output).resolve()}",
            flush=True,
        )
        return 0 if result.success else 1
    finally:
        if recording:
            recorder.stop()
        task.close()
        planner.close()


if __name__ == "__main__":
    raise SystemExit(main())
