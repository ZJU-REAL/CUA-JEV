from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _builder():
    source = Path(__file__).resolve().parents[1] / "scripts" / "build_pages.py"
    spec = importlib.util.spec_from_file_location("build_pages", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracked_snapshot_is_complete_and_display_safe():
    builder = _builder()
    snapshot = json.loads(builder.SNAPSHOT.read_text(encoding="utf-8"))
    assert snapshot["schema_version"] == 1
    assert set(snapshot["tasks"]) == set(builder.TASKS)
    for task in builder.TASKS:
        rows = snapshot["benchmarks"][task]["rows"]
        assert {(row["agent"], row["action_space"]) for row in rows} == {
            ("jev", "hybrid"),
            ("jev", "gui_only"),
            ("codex_computer_use", "hybrid"),
        }
        for row in rows:
            record = snapshot["steps"][row["representative_run_id"]]
            assert record["terminal_verified"]
            assert record["steps"]
            for step in record["steps"]:
                builder._safe_text(step["title"])
    assert "TYPESAFE_API_KEY" not in json.dumps(snapshot)


def test_pages_build_uses_relative_assets_and_snapshot(tmp_path):
    builder = _builder()
    output = tmp_path / "pages"
    builder.build(output)
    html = (output / "index.html").read_text(encoding="utf-8")
    early = (output / "early-work.html").read_text(encoding="utf-8")
    assert 'data-site-mode="static"' in early
    assert 'href="static/style.css"' in early
    assert 'src="static/app.js"' in early
    assert (output / ".nojekyll").is_file()
    bootstrap = json.loads((output / "data" / "bootstrap.json").read_text(encoding="utf-8"))
    for task in builder.TASKS:
        assert (output / "static" / "icons" / f"{task}.svg").is_file()
        assert (output / "data" / "benchmarks" / f"{task}.json").is_file()
        for path in bootstrap["demos"][task].values():
            assert (output / path).is_file()
    assert len(list((output / "data" / "steps").glob("*.json"))) == 12
    assert (output / "static" / "open-task-loop.svg").is_file()
    assert (output / "media" / "open-workspace-research.mp4").is_file()
    assert "Four desktop workflow cases" in early
    assert "One computer-use loop" in html
    assert "One computer-use loop" in (output / "preview.html").read_text(encoding="utf-8")
    catalog = json.loads((output / "data" / "windows_demos.json").read_text(encoding="utf-8"))
    assert len(catalog["cases"]) == 4
    for case in catalog["cases"]:
        assert (output / case["video"]).is_file()


def test_four_verified_windows_cases_activate_new_home(tmp_path, monkeypatch):
    builder = _builder()
    media = tmp_path / "website" / "media"
    media.mkdir(parents=True)
    cases = []
    for index in range(4):
        name = f"windows-case-{index}.mp4"
        (media / name).write_bytes(b"reviewed video")
        cases.append({
            "id": f"case-{index}", "verified": True,
            "title": f"Case {index}", "summary": "A verified Windows task",
            "apps": ["Edge", "VS Code"], "video": f"media/{name}",
            "actions": 17, "jev_calls": 17, "model_calls": 17, "vlm_calls": 0,
            "wall_time_s": 110.0, "channels": {"script": 17},
            "steps": [{"title": "Verified click", "channel": "script", "app": "Edge"}] * 17,
        })
    catalog = tmp_path / "website" / "windows_demos.json"
    catalog.write_text(json.dumps({"schema_version": 2, "cases": cases}), encoding="utf-8")
    monkeypatch.setattr(builder, "WEBSITE", tmp_path / "website")
    monkeypatch.setattr(builder, "WINDOWS_DEMOS", catalog)
    output = tmp_path / "pages"
    builder.build(output)
    assert "One computer-use loop" in (output / "index.html").read_text(encoding="utf-8")
    assert "Four desktop workflow cases" in (output / "early-work.html").read_text(encoding="utf-8")
    assert (output / "media" / "windows-case-0.mp4").is_file()
