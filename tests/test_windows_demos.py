from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _curator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "curate_windows_demos.py"
    spec = importlib.util.spec_from_file_location("curate_windows_demos", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case_files(tmp_path: Path, count: int = 18):
    trace = tmp_path / "trace.jsonl"
    events = []
    sequence = 0
    for index in range(count):
        for kind, payload in (
            ("candidates", {"items": [{
                "id": "choice", "description": f"Read public chapter {index + 1}",
            }]}),
            ("decision", {"model": "jev-test", "candidate_id": "choice"}),
            ("commitment", {
                "candidate_id": "choice", "channel": "script", "capability": "browser.click",
            }),
            ("verification", {"passed": True}),
        ):
            events.append({"run_id": "one-run", "sequence": sequence, "kind": kind, "payload": payload})
            sequence += 1
    events.extend([
        {"run_id": "one-run", "sequence": sequence, "kind": "episode", "payload": {
            "status": "success", "steps": count, "duration_ms": 110_000,
        }},
        {"run_id": "one-run", "sequence": sequence + 1, "kind": "model_usage", "payload": {
            "role": "planner", "model": "text-planner", "usage": {"total_tokens": 123},
        }},
    ])
    trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps({
        "status": "success", "actions": count, "jev_calls": count,
        "model_calls": count + 1, "vlm_calls": 0,
        "wall_time_ms": 110_000, "channels": {"script": count},
    }), encoding="utf-8")
    video = tmp_path / "take.mp4"
    video.write_bytes(b"reviewed video")
    return trace, metrics, video


def test_curator_derives_counts_and_exports_only_safe_case_fields(tmp_path, monkeypatch):
    curator = _curator()
    trace, metrics, video = _case_files(tmp_path)
    catalog = tmp_path / "windows_demos.json"
    catalog.write_text('{"schema_version":2,"cases":[]}', encoding="utf-8")
    media = tmp_path / "media"
    media.mkdir()
    monkeypatch.setattr(curator, "CATALOG", catalog)
    monkeypatch.setattr(curator, "MEDIA", media)
    monkeypatch.setattr(curator, "_video_duration", lambda _path: 100.0)
    case = curator.curate(SimpleNamespace(
        id="public-research", title="Public research", summary="A documented task",
        app=["Edge", "VS Code"], browser_app="Edge", trace=trace,
        metrics=metrics, video=video,
    ))
    assert case["actions"] == case["jev_calls"] == 18
    assert case["model_calls"] == 19
    assert case["channels"] == {"script": 18}
    assert case["recording_speed"] == "1.10× playback"
    assert len(case["steps"]) == 18
    assert (media / "windows-public-research.mp4").is_file()
    assert "C:\\" not in catalog.read_text(encoding="utf-8")


def test_curator_rejects_short_or_failed_run(tmp_path):
    curator = _curator()
    trace, _, _ = _case_files(tmp_path, count=12)
    with pytest.raises(ValueError, match="17–23"):
        curator._trace_steps(trace)
    text = trace.read_text(encoding="utf-8").replace('"status": "success"', '"status": "failed"')
    trace.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="successful"):
        curator._trace_steps(trace)
