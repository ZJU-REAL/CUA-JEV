from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _builder():
    source = Path(__file__).resolve().parents[1] / "scripts" / "build_pages.py"
    spec = importlib.util.spec_from_file_location("mac_pages_builder", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case():
    """Synthetic test data only; the published catalog starts empty."""
    return {
        "id": "python-onboarding", "verified": True,
        "title": "Reviewed test fixture", "summary": "Synthetic validation fixture.",
        "apps": ["Chromium", "Native fixture"],
        "video": "media/macos-python-onboarding.mp4",
        "poster": "media/macos-python-onboarding.jpg",
        "actions": 2, "jev_calls": 2, "model_calls": 2, "vlm_calls": 0,
        "wall_time_s": 12.5, "channels": {"script": 1, "api": 1}, "routes": {"ax": 1},
        "planner_model": "test-planner", "jev_model": "jev-test",
        "recording_scope": "Selected application windows only.",
        "timing_note": "Idle intervals were shortened; measured wall time is unchanged.",
        "steps": [
            {"title": "Open a local page", "channel": "script", "app": "Chromium"},
            {"title": "Save a local note", "channel": "api", "route": "ax"},
        ],
    }


@pytest.fixture
def mac_catalog(tmp_path, monkeypatch):
    builder = _builder()
    website = tmp_path / "website"
    (website / "media").mkdir(parents=True)
    case = _case()
    for key in ("video", "poster"):
        (website / case[key]).write_bytes(f"reviewed test {key}".encode())
    path = website / "mac_demos.json"
    monkeypatch.setattr(builder, "WEBSITE", website)
    monkeypatch.setattr(builder, "MAC_DEMOS", path)

    def write(cases):
        path.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")

    write([case])
    return builder, case, write


def test_mac_catalog_may_be_absent_or_empty(mac_catalog):
    builder, _, write = mac_catalog
    write([])
    assert builder._mac_cases() == []
    builder.MAC_DEMOS.unlink()
    assert builder._mac_cases() == []


def test_mac_catalog_builds_separately_and_copies_only_reviewed_media(mac_catalog, tmp_path, monkeypatch):
    builder, case, _ = mac_catalog
    monkeypatch.setattr(builder, "_windows_cases", lambda: ([], False))
    (builder.WEBSITE / "media" / "private-recording.mp4").write_bytes(b"must stay local")
    output = tmp_path / "published"
    builder.build(output)
    assert json.loads((output / "data/mac_demos.json").read_text()) == {
        "schema_version": 1, "cases": [case],
    }
    assert json.loads((output / "data/windows_demos.json").read_text())["cases"] == []
    assert (output / case["video"]).read_bytes() == b"reviewed test video"
    assert (output / case["poster"]).read_bytes() == b"reviewed test poster"
    assert not (output / "media/private-recording.mp4").exists()
    assert 'id="macos-cases"' in (output / "index.html").read_text()


@pytest.mark.parametrize(("field", "value"), [
    ("verified", False), ("id", "unreviewed-case"), ("actions", True),
    ("actions", 0), ("actions", 101), ("jev_calls", 1), ("model_calls", 0),
    ("vlm_calls", -1), ("wall_time_s", float("nan")), ("wall_time_s", True),
    ("channels", {"script": 2}), ("channels", {"script": 1, "api": True}),
    ("routes", {"ax": 2}), ("routes", {"uia": 1}), ("apps", []),
    ("planner_model", ""), ("jev_model", "rule"), ("recording_scope", " "),
    ("jev_model", "jev-fallback"), ("jev_model", "jev-unavailable/rule-fallback"),
    ("jev_model", "jev-unavailable/rule-FALLBACK"),
    ("timing_note", "/Users/private/local-trace.jsonl"),
    ("video", "media/windows-python-guide.mp4"),
    ("poster", "../runs/private.jpg"), ("unexpected_private_field", "secret"),
])
def test_mac_catalog_rejects_unreviewed_or_inconsistent_fields(mac_catalog, field, value):
    builder, case, write = mac_catalog
    case[field] = value
    write([case])
    with pytest.raises(ValueError):
        builder._mac_cases()


@pytest.mark.parametrize("steps", [
    [],
    [{"title": "One", "channel": "script"}, {"title": "Two", "channel": "cli"}],
    [{"title": "One", "channel": "script"}, {"title": "Two", "channel": "api", "route": "gui"}],
    [{"title": "One", "channel": "script"}, {"title": "Two", "channel": "api", "route": None}],
    [{"title": "One", "channel": "script"}, {"title": "Two", "channel": "api", "raw": "private"}],
])
def test_mac_catalog_requires_a_consistent_public_action_account(mac_catalog, steps):
    builder, case, write = mac_catalog
    case["steps"] = steps
    write([case])
    with pytest.raises(ValueError):
        builder._mac_cases()


def test_mac_catalog_rejects_duplicates_missing_media_and_symlink_escape(mac_catalog, tmp_path):
    builder, case, write = mac_catalog
    write([case, copy.deepcopy(case)])
    with pytest.raises(ValueError, match="catalog"):
        builder._mac_cases()
    write([case])
    video = builder.WEBSITE / case["video"]
    video.unlink()
    with pytest.raises(ValueError, match="Missing reviewed"):
        builder._mac_cases()
    private = tmp_path / "private.mp4"
    private.write_bytes(b"private")
    video.symlink_to(private)
    with pytest.raises(ValueError, match="reviewed website directory"):
        builder._mac_cases()


def test_mac_frontend_validates_and_escapes_reviewed_case_data():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable for static-site JavaScript validation")
    source = Path(__file__).resolve().parents[1] / "src/cua_jev/ui/static/home.js"
    case = _case()
    case["summary"] = '<img src=x onerror="alert(1)">'
    program = r"""
const fs = require('fs');
const vm = require('vm');
const assert = require('node:assert/strict');
const sandbox = {document: {querySelector: () => null}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const item = JSON.parse(fs.readFileSync(0, 'utf8'));
const catalog = {schema_version: 1, cases: [item]};
assert.equal(sandbox.validatedMacCases(catalog).length, 1);
assert.equal(sandbox.validatedMacCases({schema_version: 1, cases: []}).length, 0);
const html = sandbox.macCaseCard(item);
assert.match(html, /preload="none"/);
assert.match(html, /CAPTURE SCOPE/);
assert.match(html, /PLAYBACK &amp; TIMING/);
assert.match(html, /&lt;img/);
assert.ok(!html.includes('<img src=x'));
for (const change of [
  {actions: true}, {jev_calls: 0}, {channels: {script: 2}}, {routes: {ax: 2}},
  {video: 'https://example.org/private.mp4'}, {timing_note: '/Users/private/test'},
  {unexpected: 'private'}, {jev_model: 'rule'}, {jev_model: 'jev-fallback'},
  {jev_model: 'jev-unavailable/rule-fallback'}, {jev_model: 'jev-unavailable/rule-FALLBACK'},
  {steps: [{title:'a', channel:'script'}, {title:'b', channel:'api', route:'gui'}]}
]) {
  assert.throws(() => sandbox.validatedMacCases({schema_version: 1, cases: [{...item, ...change}]}));
}
assert.throws(() => sandbox.validatedMacCases({schema_version: 1, cases: [item, item]}));
"""
    subprocess.run([node, "-e", program, str(source)], input=json.dumps(case),
                   text=True, capture_output=True, check=True, timeout=15)
