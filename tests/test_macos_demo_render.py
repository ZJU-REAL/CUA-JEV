"""Offline alignment/trace validation, including a tiny synthetic FFmpeg render."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def renderer():
    path = Path(__file__).resolve().parents[1] / "scripts/render_macos_demo.py"
    spec = importlib.util.spec_from_file_location("render_macos_demo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence(tmp_path):
    bundle = tmp_path / "raw"
    bundle.mkdir()
    segments = []
    for role, start, stop in (("browser", 100, 104), ("native", 101, 105), ("textedit", 105.3, 106.3)):
        (bundle / f"{role}.mp4").write_bytes(b"placeholder; encoded only by the integration test")
        segments.append({
            "role": role, "file": f"{role}.mp4", "window_id": 10 + len(segments),
            "start_requested_at": start - 0.1, "started_at": start,
            "stop_requested_at": stop, "stopped_at": stop + 0.1, "finalized": True,
        })
    manifest = {
        "schema_version": 1, "kind": "separate_native_window_recordings", "recording_complete": True,
        "segments": segments, "metrics": "metrics.json", "trace": "trace.jsonl",
    }
    metrics = {
        "recording_complete": True, "episode": {"status": "success"},
        "actions": 2, "verified_actions": 2, "channels": {"script": 1, "cli": 1},
    }
    events = []

    def event(kind, payload, timestamp):
        events.append({"sequence": len(events), "run_id": "synthetic-unit-test", "kind": kind,
                       "payload": payload, "timestamp": timestamp})

    for index, channel in enumerate(("script", "cli")):
        started = 101.2 + index * 2
        event("decision", {"decision_id": str(index), "model": "jev-test"}, started - 0.1)
        event("receipt", {"started_at": started, "ended_at": started + 0.1,
                          "channel": channel, "capability": "browser.click" if index == 0
                          else "artifact.open_textedit", "decision_id": str(index), "success": True},
              started + 0.1)
        event("verification", {"passed": True}, started + 0.2)
    event("episode", {"status": "success", "steps": 2}, 106.5)
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    (bundle / "metrics.json").write_text(json.dumps(metrics))
    (bundle / "trace.jsonl").write_text("\n".join(json.dumps(item) for item in events) + "\n")
    return bundle


def test_loader_derives_action_channels_and_times_from_trace(renderer, tmp_path):
    bundle = renderer.load_bundle(evidence(tmp_path))
    assert bundle["channels"] == {"script": 1, "cli": 1}
    assert [item["started_at"] for item in bundle["actions"]] == [101.2, 103.2]
    assert all(item["verified"] for item in bundle["actions"])


@pytest.mark.parametrize("change", ["outside_path", "mismatched_counts", "unfinished", "fake_jev"])
def test_loader_refuses_unsafe_or_inconsistent_evidence(renderer, tmp_path, change):
    bundle = evidence(tmp_path)
    if change == "outside_path":
        (tmp_path / "outside.jsonl").write_text("must not be read")
        path = bundle / "manifest.json"
        data = json.loads(path.read_text())
        data["trace"] = "../outside.jsonl"
        path.write_text(json.dumps(data))
    elif change == "mismatched_counts":
        path = bundle / "metrics.json"
        data = json.loads(path.read_text())
        data["actions"] = 20
        path.write_text(json.dumps(data))
    elif change == "unfinished":
        path = bundle / "manifest.json"
        data = json.loads(path.read_text())
        data["segments"][0]["finalized"] = False
        path.write_text(json.dumps(data))
    else:
        path = bundle / "trace.jsonl"
        path.write_text(path.read_text().replace("jev-test", "rule"))
    with pytest.raises(ValueError):
        renderer.load_bundle(bundle)


def test_caption_intervals_are_compressed_from_real_receipt_and_verification_times(renderer, tmp_path):
    bundle = renderer.load_bundle(evidence(tmp_path))
    captions = renderer._action_filters(bundle, tmp_path, 100, 2, 2.65)
    assert "gte(t,0.600000)*lt(t,0.700000)" in captions[0]
    assert "gte(t,0.700000)*lt(t,1.600000)" in captions[1]
    assert "SCRIPT  /  VERIFIED" in (tmp_path / "action-0-VERIFIED.txt").read_text()


@pytest.mark.parametrize("verification_time", [100.0, 101.25])
def test_loader_rejects_verification_before_action_completes(renderer, tmp_path, verification_time):
    bundle = evidence(tmp_path)
    trace = bundle / "trace.jsonl"
    events = [json.loads(line) for line in trace.read_text().splitlines()]
    next(event for event in events if event["kind"] == "verification")["timestamp"] = verification_time
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    with pytest.raises(ValueError, match="verification cannot precede completion"):
        renderer.load_bundle(bundle)


def test_outputs_are_create_only_before_attempting_any_video_probe(renderer, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        renderer.render(tmp_path / "absent", target, tmp_path / "poster.jpg", tmp_path / "report.json")
    assert target.read_bytes() == b"keep"


def test_synthetic_render_aligns_tracks_labels_terminal_and_keeps_raw_unchanged(renderer, tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg")
    from PIL import Image

    try:
        renderer._font_path(None)
    except ValueError:
        pytest.skip("no supported local test font")
    bundle = evidence(tmp_path)
    for role, color, duration in (("browser", "red", 4), ("native", "blue", 4), ("textedit", "lime", 1)):
        subprocess.run([
            ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
            "-i", f"color=c={color}:s=320x180:r=8:d={duration}", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(bundle / f"{role}.mp4"),
        ], check=True, timeout=30)
    before = {path.name: renderer._sha256(path) for path in bundle.glob("*.mp4")}
    output, poster, report = (tmp_path / name for name in ("demo.mp4", "poster.jpg", "report.json"))
    detail = renderer.render(bundle, output, poster, report, speed=2, terminal_speed=1, fps=8)
    assert detail["terminal_starts_at_s"] == pytest.approx(2.65)
    assert detail["planned_duration_s"] == pytest.approx(3.65)
    assert detail["output_duration_s"] == pytest.approx(3.65, abs=1 / 8)
    assert detail["action_count"] == 2
    assert Image.open(poster).size == (1920, 1080)
    assert before == {path.name: renderer._sha256(path) for path in bundle.glob("*.mp4")}
    frames = ffmpeg.read_frames(str(output))
    metadata = next(frames)
    width, height = metadata["size"]
    samples = {}
    try:
        for index, data in enumerate(frames):
            if index in {1, 9, 17, 21, 24}:
                image = Image.frombytes("RGB", (width, height), data)
                samples[index] = (image.getpixel((200, 260)), image.getpixel((1400, 300)))
    finally:
        frames.close()
    assert samples[1][0][0] > 220  # browser is visible immediately
    assert max(samples[1][1]) - min(samples[1][1]) < 20  # native clip has not started
    assert samples[9][1][2] > 220 and samples[9][1][0] < 30  # native blue after its delayed start
    assert max(samples[17][0]) - min(samples[17][0]) < 20  # browser ends; no frozen red frame
    assert samples[17][1][2] > 220 and samples[17][1][0] < 30  # native remains covered
    assert all(max(pixel) - min(pixel) < 20 for pixel in samples[21])  # both clips have ended
    assert samples[24][0][1] > 220 and samples[24][1][1] > 220  # full TextEdit terminal scene
