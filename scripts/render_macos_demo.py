"""Render a reviewable, time-aligned composite from a Mac recording bundle.

Raw clips are read-only. This is an explicitly labeled multi-window composite,
not a continuous cross-application recording. Start acknowledgements provide
approximate wall-clock alignment; the manifest cannot establish frame accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

WIDTH, HEIGHT = 1920, 1080
BROWSER_BOX = (36, 180, 1228, 730)
NATIVE_BOX = (1292, 180, 592, 354)
EDITOR_BOX = (36, 180, 1848, 730)
CHANNELS = {"script", "gui", "api", "cli", "mcp", "control"}


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _local_file(bundle: Path, value: Any, suffix: str) -> Path:
    if not isinstance(value, str) or Path(value).suffix.lower() != suffix:
        raise ValueError(f"expected a bundle-local {suffix} file")
    path = (bundle / value).resolve()
    if not path.is_relative_to(bundle) or not path.is_file():
        raise ValueError("bundle evidence must be an existing file inside the bundle")
    return path


def load_bundle(directory: Path) -> dict[str, Any]:
    bundle = directory.resolve(strict=True)
    manifest_path = _local_file(bundle, "manifest.json", ".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1
            or manifest.get("kind") != "separate_native_window_recordings"
            or manifest.get("recording_complete") is not True):
        raise ValueError("rendering requires a complete Mac recording bundle")
    metrics_path = _local_file(bundle, manifest.get("metrics"), ".json")
    trace_path = _local_file(bundle, manifest.get("trace"), ".jsonl")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("recording_complete") is not True or metrics.get("episode", {}).get("status") != "success":
        raise ValueError("the recorded episode and finalization must have succeeded")
    segments = {}
    for item in manifest.get("segments", []):
        role = item.get("role")
        if role not in {"browser", "native", "textedit"} or role in segments:
            raise ValueError("expected exactly one clip for each recorded window role")
        if item.get("finalized") is not True:
            raise ValueError("recording segment was not finalized")
        times = [_number(item.get(key), key) for key in (
            "start_requested_at", "started_at", "stop_requested_at", "stopped_at",
        )]
        if times != sorted(times) or times[1] >= times[2]:
            raise ValueError("recording timestamps are inconsistent")
        segments[role] = {**item, "path": _local_file(bundle, item.get("file"), ".mp4")}
    if set(segments) != {"browser", "native", "textedit"}:
        raise ValueError("browser, native, and TextEdit clips are required")
    if segments["textedit"]["started_at"] < segments["browser"]["stop_requested_at"]:
        raise ValueError("TextEdit must be a separate terminal recording after the browser clip")

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not events or len({event.get("run_id") for event in events}) != 1:
        raise ValueError("expected one nonempty episode trace")
    sequences = [event.get("sequence") for event in events]
    if any(type(value) is not int for value in sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("trace event sequences must be unique integers")
    actions = []
    decisions = {}
    episodes = []
    for event in sorted(events, key=lambda item: item["sequence"]):
        payload = event.get("payload", {})
        if event.get("kind") == "decision":
            decisions[payload.get("decision_id")] = payload.get("model", "")
        elif event.get("kind") == "episode":
            episodes.append(payload)
        elif event.get("kind") == "receipt":
            start = _number(payload.get("started_at"), "receipt.started_at")
            end = _number(payload.get("ended_at"), "receipt.ended_at")
            if (end < start or payload.get("channel") not in CHANNELS
                    or (actions and start < actions[-1]["started_at"])):
                raise ValueError("invalid action time or channel")
            capability = payload.get("capability")
            if not isinstance(capability, str) or not 1 <= len(capability) <= 120:
                raise ValueError("invalid action capability")
            actions.append({
                "index": len(actions) + 1, "channel": payload["channel"], "capability": capability,
                "started_at": start, "ended_at": end, "decision_id": payload.get("decision_id"),
                "success": payload.get("success") is True, "verified": None, "verified_at": None,
            })
        elif event.get("kind") == "verification" and actions:
            if actions[-1]["verified"] is not None:
                raise ValueError("verification has no unique preceding receipt")
            verified_at = _number(event.get("timestamp"), "verification.timestamp")
            if verified_at < actions[-1]["ended_at"]:
                raise ValueError("verification cannot precede completion of its action")
            actions[-1]["verified"] = payload.get("passed") is True
            actions[-1]["verified_at"] = verified_at
    counts = dict(Counter(item["channel"] for item in actions))
    if not actions or len(actions) != metrics.get("actions") or counts != metrics.get("channels"):
        raise ValueError("trace action counts disagree with metrics")
    if any(item["verified"] is None for item in actions):
        raise ValueError("a recorded action has no verification event")
    if sum(item["verified"] for item in actions) != metrics.get("verified_actions"):
        raise ValueError("trace verification counts disagree with metrics")
    if (len(episodes) != 1 or episodes[0].get("status") != "success"
            or episodes[0].get("steps") != len(actions)):
        raise ValueError("trace must contain one matching successful episode")
    models = [decisions.get(action["decision_id"], "") for action in actions]
    if any(not model.startswith("jev-") or "fallback" in model for model in models):
        raise ValueError("a recorded action has no genuine Jev decision")
    return {"directory": bundle, "manifest": manifest, "metrics": metrics,
            "segments": segments, "actions": actions, "channels": counts,
            "evidence": [manifest_path, metrics_path, trace_path]}


def _font_path(requested: Path | None) -> Path:
    candidates = [requested] if requested else [
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path and path.is_file():
            return path.resolve()
    raise ValueError("no suitable font found; provide --font PATH_TO_TTF_OR_TTC")


def _probe(path: Path) -> dict[str, Any]:
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    duration = _number(metadata.get("duration"), "video duration")
    if duration <= 0:
        raise ValueError("the encoded clip must have a positive readable duration")
    return {"duration_s": duration, "size": list(metadata["source_size"]), "fps": metadata["fps"]}


def _background(path: Path, font_path: Path, bundle: dict, *, speed: float, terminal: bool) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (WIDTH, HEIGHT), "#f2f3ee")
    draw = ImageDraw.Draw(image)
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in (20, 22, 24, 28, 38, 48)}

    def text(position, value, size=24, fill="#173244"):
        draw.text(position, value, font=fonts[size], fill=fill)

    draw.rectangle((0, 0, WIDTH, 112), fill="#152f40")
    text((36, 23), "CUA-JEV", 48, "#ffffff")
    text((318, 43), "MODEL + JEV  /  macOS", 24, "#afe0cb")
    text((1464, 41), f"{speed:g}x playback", 28, "#ffffff")
    if terminal:
        text((36, 130), "03 / TEXTEDIT TERMINAL WINDOW — SEPARATE VERIFIED CLIP", 24)
        boxes = [EDITOR_BOX]
    else:
        text((36, 130), "01 / BROWSER WINDOW", 24)
        text((1292, 130), "02 / NATIVE HANDOFF WINDOW", 24)
        boxes = [BROWSER_BOX, NATIVE_BOX]
    for x, y, width, height in boxes:
        draw.rectangle((x - 1, y - 1, x + width + 1, y + height + 1), fill="#d7dfdc")
        draw.rectangle((x, y, x + width, y + height), fill="#e5eae5")
        text((x + 20, y + height // 2), "No clip coverage at this instant", 22, "#526573")
    count = len(bundle["actions"])
    verified = sum(action["verified"] for action in bundle["actions"])
    channels = "  /  ".join(f"{name.upper()} {value}" for name, value in sorted(bundle["channels"].items()))
    if not terminal:
        draw.rounded_rectangle((1292, 558, 1884, 910), radius=16, fill="#152f40")
        text((1316, 580), "TRACE RECEIPTS", 22, "#afe0cb")
        text((1316, 624), "Awaiting first recorded action", 24, "#ffffff")
        text((1316, 856), f"{count} actions / {verified} independently verified", 20, "#afe0cb")
    text((36, 940), f"{count} recorded actions  /  {verified} verified   ·   {channels}", 24)
    text((36, 986), "SYNCHRONIZED WINDOW COMPOSITE — NOT ONE CONTINUOUS CROSS-APP RECORDING", 22)
    text((36, 1022), "Approximate start-ack alignment. CLI / MCP are trace receipts; no terminal footage."
         if not terminal else "TextEdit begins after exact-document verification and task acceptance.",
         22, "#526573")
    image.save(path)


def _clip_filter(index: int, box: tuple, *, speed: float, offset: float, fps: int, label: str) -> str:
    _, _, width, height = box
    return (f"[{index}:v]setpts=(PTS-STARTPTS)/{speed:g}+{offset:.6f}/TB,fps={fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white,setsar=1[{label}]")


def _action_filters(bundle: dict, temporary: Path, base: float, speed: float,
                    terminal_start: float) -> list[str]:
    filters = []
    actions = bundle["actions"]
    for position, action in enumerate(actions):
        start = max(0, (action["started_at"] - base) / speed)
        end = min(terminal_start, (actions[position + 1]["started_at"] - base) / speed
                  if position + 1 < len(actions) else terminal_start)
        if end <= start:
            continue
        verified_at = max(start, min(end, (action["verified_at"] - base) / speed))
        for state, left, right in (
            ("EXECUTING", start, verified_at),
            ("VERIFIED" if action["verified"] else "NOT VERIFIED", verified_at, end),
        ):
            if right <= left:
                continue
            text_file = f"action-{position}-{state.replace(' ', '-')}.txt"
            (temporary / text_file).write_text(
                f"ACTION {position + 1:02d} / {len(actions):02d}\n\n"
                f"{action['channel'].upper()}  /  {state}\n{action['capability']}\n\n"
                f"Source time +{action['started_at'] - base:.1f}s", encoding="utf-8",
            )
            enable = f"gte(t,{left:.6f})*lt(t,{right:.6f})"
            filters.append(
                f"drawbox=x=1308:y=620:w=560:h=224:color=0x152f40:t=fill:enable='{enable}',"
                f"drawtext=fontfile=font.ttf:textfile={text_file}:expansion=none:fontcolor=white:"
                f"fontsize=25:line_spacing=7:x=1316:y=628:enable='{enable}'"
            )
    return filters


def _run(command: list[str], *, cwd: Path, timeout: float) -> None:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-3500:]}")


def render(bundle_path: Path, output: Path, poster: Path, report: Path, *, speed: float = 4,
           terminal_speed: float = 1, fps: int = 24, font: Path | None = None,
           poster_at: float | None = None) -> dict[str, Any]:
    import imageio_ffmpeg

    for value in (speed, terminal_speed):
        if not math.isfinite(value) or not 0.25 <= value <= 16:
            raise ValueError("playback speeds must be finite values between 0.25 and 16")
    if type(fps) is not int or not 1 <= fps <= 60:
        raise ValueError("fps must be an integer between 1 and 60")
    destinations = [path.resolve() for path in (output, poster, report)]
    if len(set(destinations)) != 3 or any(path.exists() or path.is_symlink()
                                        for path in (output, poster, report)):
        raise FileExistsError("video, poster, and report must be distinct new output paths")
    if (output.suffix.lower() != ".mp4" or poster.suffix.lower() not in {".jpg", ".jpeg"}
            or report.suffix.lower() != ".json"):
        raise ValueError("output must be MP4, poster JPEG, and report JSON")
    bundle = load_bundle(bundle_path)
    font_path = _font_path(font)
    segments = bundle["segments"]
    metadata = {role: _probe(item["path"]) for role, item in segments.items()}
    base = min(segments[role]["started_at"] for role in ("browser", "native"))
    terminal_start = (segments["textedit"]["started_at"] - base) / speed
    duration = terminal_start + metadata["textedit"]["duration_s"] / terminal_speed
    if terminal_start <= 0:
        raise ValueError("terminal clip must follow the main recording timeline")
    chosen_poster = min(7.0, terminal_start / 2) if poster_at is None else poster_at
    if not math.isfinite(chosen_poster) or not 0 <= chosen_poster < duration:
        raise ValueError("poster time must be inside the rendered video")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Rendering {duration:.1f}s: main {speed:g}x, separate TextEdit {terminal_speed:g}x", flush=True)
    with tempfile.TemporaryDirectory(prefix="cua-jev-mac-render-") as temporary_name:
        temporary = Path(temporary_name)
        shutil.copyfile(font_path, temporary / "font.ttf")
        _background(temporary / "main.png", font_path, bundle, speed=speed, terminal=False)
        _background(temporary / "terminal.png", font_path, bundle, speed=terminal_speed, terminal=True)
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-n"]
        for role in ("browser", "native", "textedit"):
            command.extend(["-i", str(segments[role]["path"])])
        for name in ("main.png", "terminal.png"):
            command.extend(["-loop", "1", "-framerate", str(fps), "-i", name])
        graph = [
            "[3:v]format=yuv420p[base]",
            f"[base][4:v]overlay=0:0:enable='gte(t,{terminal_start:.6f})'[layout]",
            _clip_filter(0, BROWSER_BOX, speed=speed,
                         offset=(segments["browser"]["started_at"] - base) / speed,
                         fps=fps, label="browser"),
            _clip_filter(1, NATIVE_BOX, speed=speed,
                         offset=(segments["native"]["started_at"] - base) / speed,
                         fps=fps, label="native"),
            _clip_filter(2, EDITOR_BOX, speed=terminal_speed,
                         offset=terminal_start, fps=fps, label="editor"),
        ]
        previous = "layout"
        for role, box in (("browser", BROWSER_BOX), ("native", NATIVE_BOX), ("editor", EDITOR_BOX)):
            enable = f"{'gte' if role == 'editor' else 'lt'}(t,{terminal_start:.6f})"
            graph.append(f"[{previous}][{role}]overlay={box[0]}:{box[1]}:eof_action=pass:repeatlast=0:"
                         f"enable='{enable}'[{role}out]")
            previous = f"{role}out"
        captions = _action_filters(bundle, temporary, base, speed, terminal_start)
        graph.append(f"[{previous}]" + (",".join(captions) + "," if captions else "") + "format=yuv420p[out]")
        (temporary / "filters.txt").write_text(";\n".join(graph), encoding="utf-8")
        command.extend([
            "-filter_complex_threads", "2", "-filter_complex_script", "filters.txt", "-map", "[out]",
            "-t", f"{duration:.6f}", "-an", "-r", str(fps), "-c:v", "libx264", "-threads", "2",
            "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "demo.mp4",
        ])
        _run(command, cwd=temporary, timeout=max(120, duration * 10))
        _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-ss", str(chosen_poster),
              "-i", "demo.mp4", "-frames:v", "1", "-q:v", "2", "poster.jpg"], cwd=temporary, timeout=60)
        encoded = _probe(temporary / "demo.mp4")
        details = {
            "schema_version": 1, "kind": "timestamp_aligned_window_composite", "private": True,
            "source_bundle": str(bundle["directory"]), "source_start_ack_at": base,
            "alignment": "Each first encoded frame is anchored to record_start acknowledgement. "
                         "Request/ack intervals bound uncertainty; synchronization is not frame-accurate.",
            "main_speed": speed, "terminal_speed": terminal_speed, "terminal_starts_at_s": terminal_start,
            "planned_duration_s": duration, "output_duration_s": encoded["duration_s"],
            "fps": encoded["fps"], "size": encoded["size"], "poster_at_s": chosen_poster,
            "action_count": len(bundle["actions"]), "channels": bundle["channels"],
            "actions": bundle["actions"], "raw_clips": {
                role: {"file": item["file"], "window_id": item.get("window_id"), **metadata[role],
                       "start_ack_at": item["started_at"],
                       "start_ack_uncertainty_s": item["started_at"] - item["start_requested_at"],
                       "sha256": _sha256(item["path"])}
                for role, item in segments.items()
            },
            "coverage": "Browser/native clips overlap; missing coverage is shown without frozen frames. "
                        "TextEdit is a separately captioned terminal clip. CLI/MCP have receipts only.",
            "evidence_sha256": {path.name: _sha256(path) for path in bundle["evidence"]},
        }
        (temporary / "report.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        for source, destination in zip(("demo.mp4", "poster.jpg", "report.json"), destinations, strict=True):
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (temporary / source).open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)
    return details


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new MP4 output")
    parser.add_argument("--poster", type=Path, help="new JPEG; defaults to output basename.jpg")
    parser.add_argument("--report", type=Path, help="new JSON; defaults to output basename.render.json")
    parser.add_argument("--speed", type=float, default=4)
    parser.add_argument("--terminal-speed", type=float, default=1)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--poster-at", type=float, help="timestamp in rendered video, in seconds")
    args = parser.parse_args(argv)
    render(args.bundle, args.output, args.poster or args.output.with_suffix(".jpg"),
           args.report or args.output.with_suffix(".render.json"), speed=args.speed,
           terminal_speed=args.terminal_speed, fps=args.fps, font=args.font, poster_at=args.poster_at)
    print(f"Review composite and poster: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
