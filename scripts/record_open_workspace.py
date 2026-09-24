"""Record one genuine model + Jev browser-to-VS-Code episode on Windows.

The runner starts the capture only after the real Edge window is ready, so
initial desktop frames are not published. Review the raw video before copying
it into website/media/.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from record_v2_demos import _dependencies, _draw_hud

from cua_jev.cli import _open_workspace_runner
from cua_jev.config import load_local_env
from cua_jev.model_planner import ChatModelPlanner
from cua_jev.open_workspace import OpenWorkspaceTask


class WindowRecorder:
    """Record the task window, optionally hiding the rounded desktop edge."""

    def __init__(
        self, output: Path, state: dict[str, Any], *, fps: int = 10, inset_px: int = 0
    ) -> None:
        if not 0 <= inset_px <= 32:
            raise ValueError("window inset must be between 0 and 32 pixels")
        self.output = output
        self.state = state
        self.fps = fps
        self.inset_px = inset_px
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._capture, name="cua-jev-window-recorder", daemon=True)

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("window recorder did not stop")
        if self.error:
            raise RuntimeError(f"window recording failed: {self.error}") from self.error

    def _capture(self) -> None:
        import ctypes
        from ctypes import wintypes

        ffmpeg, numpy, image, image_draw, image_font, image_grab = _dependencies()
        writer = None
        last_frame = None
        try:
            writer = ffmpeg.write_frames(
                str(self.output), (1440, 810), fps=self.fps, codec="libx264",
                pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7,
                macro_block_size=2, output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            recording_started = time.perf_counter()
            sent_frames = 0
            while not self.stop_event.is_set():
                handle = int(self.state.get("window_handle") or 0)
                rect = wintypes.RECT()
                if handle and ctypes.windll.user32.GetWindowRect(handle, ctypes.byref(rect)):
                    inset = self.inset_px
                    bounds = (
                        rect.left + inset, rect.top + inset,
                        rect.right - inset, rect.bottom - inset,
                    )
                    if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                        raw = image_grab.grab(bbox=bounds, all_screens=True).convert("RGB")
                        scale = min(1440 / raw.width, 810 / raw.height)
                        size = (max(2, round(raw.width * scale)), max(2, round(raw.height * scale)))
                        fitted = raw.resize(size, image.Resampling.BILINEAR)
                        frame = image.new("RGB", (1440, 810), "#f4f8fb")
                        frame.paste(fitted, ((1440 - size[0]) // 2, (810 - size[1]) // 2))
                        last_frame = frame
                if last_frame is not None:
                    shown = _draw_hud(last_frame.copy(), dict(self.state), image, image_draw, image_font)
                    encoded = numpy.asarray(shown).tobytes()
                    target_frames = max(1, round((time.perf_counter() - recording_started) * self.fps))
                    while sent_frames < target_frames:
                        writer.send(encoded)
                        sent_frames += 1
                next_frame = recording_started + (sent_frames + 1) / self.fps
                self.stop_event.wait(max(0.0, next_frame - time.perf_counter()))
        except BaseException as exc:
            self.error = exc
        finally:
            if writer is not None:
                writer.close()


def stage_window(
    task: OpenWorkspaceTask, title_re: str, *, handle: int | None = None,
) -> int:
    import ctypes

    window = (
        task.screen.focus_handle(handle, maximize=False) if handle is not None
        else task.screen.focus(title_re, maximize=False)
    )
    window.restore()
    if not ctypes.windll.user32.MoveWindow(int(window.handle), 35, 35, 1900, 1070, True):
        raise RuntimeError("could not size the visible task window for recording")
    window.set_focus()
    time.sleep(0.5)
    return int(window.handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--required-source-text", action="append", default=[])
    parser.add_argument("--research-topic", action="append", default=[])
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
        planner=planner, required_source_texts=args.required_source_text, maximize_editor=False,
        research_topics=args.research_topic,
    )
    state = {
        "task": "EDGE → VS CODE", "profile_label": "MODEL + JEV · OPEN TASK",
        "step": 0, "total": len(args.research_topic) * 2 + 2 if args.research_topic else 0,
        "channel": "—", "status": "running", "window_handle": 0,
    }
    original_evaluate = task.evaluate

    def evaluate_with_hud(observation, candidate, receipt, verification):
        evaluation = original_evaluate(observation, candidate, receipt, verification)
        state["step"] += 1
        state["channel"] = str(candidate.channel)
        state["status"] = "verified" if verification.passed else "replan"
        if candidate.capability == "workspace.open_note" and verification.passed:
            state["window_handle"] = stage_window(
                task, rf".*{re.escape(task.note_path.name)}.*Visual Studio Code.*"
            )
        return evaluation

    task.evaluate = evaluate_with_hud  # type: ignore[method-assign]
    recorder = WindowRecorder(Path(args.output), state, fps=args.fps)
    recording = False
    try:
        task.reset()
        state["window_handle"] = stage_window(
            task, rf".*{re.escape(task.page.title())}.*"
        )
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
            f"planner_calls={task.planner_calls} planner_wall_ms={planner.planning_wall_ms:.0f} "
            f"model_tokens={planner.usage_totals.get('total_tokens', 'n/a')} "
            f"video={Path(args.output).resolve()}",
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
