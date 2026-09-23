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
    assert 'data-site-mode="static"' in html
    assert 'href="static/style.css"' in html
    assert 'src="static/app.js"' in html
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
