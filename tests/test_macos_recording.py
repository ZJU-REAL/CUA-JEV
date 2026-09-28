"""Permission-free tests of the Mac demo recorder's scope and cleanup rules."""

from __future__ import annotations

import importlib.util
import json
import plistlib
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def recorder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "record_macos_computer.py"
    spec = importlib.util.spec_from_file_location("record_macos_computer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Bridge:
    def __init__(self, events, *, fail_stop=False, fail_start=False, editor_id=30, screen_locked=None):
        self.events, self.fail_stop, self.editor_id = events, fail_stop, editor_id
        self.fail_start = fail_start
        self.screen_locked = screen_locked
        self.selected = None
        self.path = None
        self.last_recording_diagnostics = None
        self.recording_events = [{"name": "recording_finished", "at": 1234.5}]

    def call(self, command, **args):
        self.last_recording_diagnostics = None
        self.events.append((command, args))
        if command == "health":
            return {"accessibility": True, "screen_recording": True, "screen_locked": self.screen_locked}
        if command == "select":
            window_id = {"com.google.Chrome": 10, "com.apple.TextEdit": self.editor_id}.get(
                args["bundle_id"], 20,
            )
            self.selected = {
                "window_id": window_id, "bundle_id": args["bundle_id"] or "fixture",
                "window_title": args["title_pattern"],
            }
            return self.selected
        if command == "record_start":
            if self.fail_start:
                self.last_recording_diagnostics = {"events": self.recording_events}
                raise RuntimeError("recording failed to start")
            self.path = Path(args["path"])
            self.path.write_bytes(b"test recording data")
            return {"recording": True, "window_id": self.selected["window_id"], "width": 800, "height": 600}
        if command == "record_stop":
            if self.fail_stop:
                self.last_recording_diagnostics = {"events": self.recording_events}
                raise RuntimeError("recording failed to finalize")
            return {"finalized": True, "events": self.recording_events}
        raise AssertionError(command)

    def close(self):
        self.events.append(("bridge.close", {}))


def task_fixture(events):
    return SimpleNamespace(
        providers={"d": SimpleNamespace(capture=lambda: SimpleNamespace(native={"window_id": 20}))},
        browser=SimpleNamespace(page=SimpleNamespace(title=lambda: "Documentation")),
        artifact_surface=SimpleNamespace(opened_window_handle=30),
        evaluate=lambda *args: SimpleNamespace(success=True, terminal=True),
        close=lambda: events.append(("task.close", {})),
    )


def args_fixture():
    return SimpleNamespace(browser_channel="chrome", window_title="^Fixture$", app_bundle_id=None)


@pytest.mark.parametrize("locked", [True, False, None])
def test_recording_preflight_preserves_unknown_lock_status(recorder, tmp_path, locked):
    recording = recorder.MacRecording(task_fixture([]), args_fixture(), tmp_path,
                                     bridge_factory=lambda: Bridge([], screen_locked=locked))
    if locked is True:
        with pytest.raises(RuntimeError, match="screen is locked"):
            recording.check_permissions()
    else:
        recording.check_permissions()
    assert recording.segments == []


def test_locked_session_records_failure_without_gui_or_api_calls(recorder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recorder.sys, "platform", "darwin")
    monkeypatch.setattr(recorder, "load_local_env", lambda: None)
    monkeypatch.setattr(recorder.os, "getenv", lambda _: "offline-fixture-key")
    events = []
    original_recording = recorder.MacRecording
    monkeypatch.setattr(recorder, "MacRecording", lambda *a, **k: original_recording(
        *a, **k, bridge_factory=lambda: Bridge(events, screen_locked=True),
    ))
    monkeypatch.setattr(recorder, "_chat_planner", lambda _: SimpleNamespace(close=lambda: None))
    task = task_fixture(events)
    task.reset = lambda: pytest.fail("must not launch GUI while locked")
    monkeypatch.setattr(recorder, "_build_task", lambda *a: task)
    monkeypatch.setattr(recorder, "_open_computer_runner", lambda *a: pytest.fail("must not call APIs"))
    destination = tmp_path / "bundle"
    assert recorder.main([
        "--output", str(destination), "--goal", "Offline lock preflight", "--url", "https://docs.test/",
        "--window-title", "^Fixture$", "--require-window-text-contains", "Reviewed: yes",
        "--allow-window-text-input", "--allow-window-actions", "--local-tool-root", str(tmp_path),
        "--artifact-path", str(tmp_path / "guide.txt"), "--open-artifact-textedit",
        "--model-base-url", "https://model.invalid/v1", "--model", "test-model", "--policy", "jev",
    ]) == 1
    stage = next((tmp_path / "runs/recording-work").iterdir())
    manifest = json.loads((stage / "manifest.json").read_text())
    metrics = json.loads((stage / "metrics.json").read_text())
    assert "screen is locked" in manifest["failure"]
    assert not manifest["recording_complete"] and not manifest["segments"]
    assert metrics["planner_requests"] == metrics["jev_reported_requests"] == 0
    assert not destination.exists()
    assert [command for command, _ in events] == ["health", "bridge.close", "bridge.close", "task.close"]


def test_recordings_pin_each_window_and_mark_overlapping_coverage(recorder, tmp_path):
    events = []
    task = task_fixture(events)
    recording = recorder.MacRecording(
        task, args_fixture(), tmp_path, final_hold=0, bridge_factory=lambda: Bridge(events),
    )
    recorder._attach_recording(task, recording)
    recording.check_permissions()
    recording.start()
    outcome = task.evaluate(
        None, SimpleNamespace(capability="artifact.open_textedit"),
        SimpleNamespace(output={"window_title": "guide.txt"}, decision_id="verified-handoff"),
        SimpleNamespace(passed=True),
    )
    task.close()
    task.close()
    assert outcome.success
    assert [item["role"] for item in recording.segments] == ["browser", "textedit", "native"]
    assert [item["window_id"] for item in recording.segments] == [10, 30, 20]
    assert all(item["finalized"] for item in recording.segments)
    assert all(item["started_at"] <= item["stop_requested_at"] <= item["stopped_at"]
               for item in recording.segments)
    assert recording.transitions[0]["decision_id"] == "verified-handoff"
    assert events[-1][0] == "task.close"
    assert sum(command == "record_stop" for command, _ in events) == 3
    assert sum(command == "task.close" for command, _ in events) == 1


def test_wrong_textedit_window_cannot_be_recorded(recorder, tmp_path):
    events = []
    recording = recorder.MacRecording(
        task_fixture(events), args_fixture(), tmp_path,
        bridge_factory=lambda: Bridge(events, editor_id=999),
    )
    recording.start()
    with pytest.raises(RuntimeError, match="window ID"):
        recording.after_evaluate(
            SimpleNamespace(capability="artifact.open_textedit"),
            SimpleNamespace(output={"window_title": "same-title.txt"}, decision_id="decision"),
            SimpleNamespace(passed=True), SimpleNamespace(success=True),
        )
    recording.close()
    assert not (tmp_path / "textedit.mp4").exists()
    assert not recording.editor_started


def test_unverified_handoff_never_switches_recording(recorder, tmp_path):
    recording = recorder.MacRecording(task_fixture([]), args_fixture(), tmp_path,
                                     bridge_factory=lambda: Bridge([]))
    recording.start()
    recording.after_evaluate(SimpleNamespace(capability="artifact.open_textedit"), None,
                             SimpleNamespace(passed=False), SimpleNamespace(success=False))
    recording.close()
    assert [item["role"] for item in recording.segments] == ["browser", "native"]


def test_early_editor_open_keeps_browser_recording_until_terminal_acceptance(recorder, tmp_path):
    recording = recorder.MacRecording(task_fixture([]), args_fixture(), tmp_path, final_hold=0,
                                     bridge_factory=lambda: Bridge([]))
    recording.start()
    recording.after_evaluate(
        SimpleNamespace(capability="artifact.open_textedit"),
        SimpleNamespace(output={"window_title": "guide.txt"}, decision_id="early-open"),
        SimpleNamespace(passed=True), SimpleNamespace(success=False),
    )
    assert recording.browser.active["role"] == "browser"
    assert not recording.editor_started
    recording.after_evaluate(SimpleNamespace(capability="desktop.click"), None,
                             SimpleNamespace(passed=True), SimpleNamespace(success=True))
    assert recording.browser.active["role"] == "textedit"
    assert recording.transitions[0]["decision_id"] == "early-open"
    recording.close()


def test_failed_finalization_still_closes_other_capture_and_task(recorder, tmp_path):
    events = []
    bridges = iter((Bridge(events, fail_stop=True), Bridge(events)))
    task = task_fixture(events)
    recording = recorder.MacRecording(task, args_fixture(), tmp_path, bridge_factory=lambda: next(bridges))
    recorder._attach_recording(task, recording)
    recording.start()
    with pytest.raises(RuntimeError, match="finalize"):
        task.close()
    assert events[-1][0] == "task.close"
    assert sum(command == "bridge.close" for command, _ in events) == 2
    assert sum(command == "record_stop" for command, _ in events) == 2
    assert not recording.browser.segments[0]["finalized"]
    assert recording.desktop.segments[0]["finalized"]


def test_window_cannot_change_while_recording_and_segments_are_create_only(recorder, tmp_path):
    bridge = Bridge([])
    windows = recorder.WindowSegments(tmp_path, bridge=bridge)
    windows.start("native", title_pattern="Fixture")
    with pytest.raises(RuntimeError, match="active recording"):
        windows.start("another", title_pattern="Other")
    windows.stop()
    with pytest.raises(FileExistsError):
        windows.start("native", title_pattern="Fixture")
    windows.close()


@pytest.mark.parametrize("operation", ["start", "stop"])
def test_segment_preserves_failed_recording_diagnostics_before_bridge_reuse(recorder, tmp_path, operation):
    bridge = Bridge([], fail_start=operation == "start", fail_stop=operation == "stop")
    windows = recorder.WindowSegments(tmp_path, bridge=bridge)
    with pytest.raises(RuntimeError, match="recording failed"):
        windows.start("native", title_pattern="Fixture")
        windows.stop()
    segment = windows.segments[0]
    assert not segment["finalized"]
    assert segment["recording_diagnostics"] == {
        "events": [{"name": "recording_finished", "at": 1234.5}],
    }
    bridge.call("health")
    bridge.recording_events[0]["at"] = 9999
    assert bridge.last_recording_diagnostics is None
    assert segment["recording_diagnostics"]["events"][0]["at"] == 1234.5
    windows.close()


def test_finalized_segment_keeps_a_snapshot_of_native_recording_events(recorder, tmp_path):
    bridge = Bridge([])
    windows = recorder.WindowSegments(tmp_path, bridge=bridge)
    windows.start("native", title_pattern="Fixture")
    windows.stop()
    segment = windows.segments[0]
    assert segment["finalized"]
    assert segment["recording_events"] == [{"name": "recording_finished", "at": 1234.5}]
    bridge.recording_events[0]["at"] = 9999
    assert segment["recording_events"][0]["at"] == 1234.5
    assert "recording_diagnostics" not in segment
    windows.close()


def test_promotion_never_overwrites_a_directory_created_during_the_run(recorder, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "manifest.json").write_text("{}")
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "keep.txt").write_text("user content")
    with pytest.raises(FileExistsError):
        recorder._promote(stage, destination)
    assert (destination / "keep.txt").read_text() == "user content"
    assert not (destination / "manifest.json").exists()
    assert (stage / "manifest.json").is_file()


def test_bundle_promotion_keeps_private_source_evidence(recorder, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    recorder._write_json(stage / "manifest.json", {"private": True})
    recorder._promote(stage, tmp_path / "new-bundle")
    assert json.loads((tmp_path / "new-bundle" / "manifest.json").read_text())["private"]
    assert (stage / "manifest.json").is_file()


@pytest.mark.parametrize("flag", ["--trace", "--record-desktop"])
def test_runner_rejects_external_trace_and_recording_paths(recorder, tmp_path, monkeypatch, flag):
    monkeypatch.setattr(recorder.sys, "platform", "darwin")
    with pytest.raises(SystemExit, match="2"):
        recorder.main(["--output", str(tmp_path / "bundle"), flag, "unmanaged"])


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_cleanup_interruption_still_releases_remaining_resources(recorder, exception):
    closed = []

    def cancel():
        closed.append("first")
        raise exception()

    with pytest.raises(exception):
        recorder._close_all((cancel, lambda: closed.append("second")))
    assert closed == ["first", "second"]


@pytest.mark.parametrize("finalization_fails", [False, True])
def test_main_keeps_complete_evidence_and_promotes_only_success(
    recorder, tmp_path, monkeypatch, finalization_fails,
):
    from cua_jev.episode import EpisodeResult, EpisodeStatus
    from cua_jev.models import ActionReceipt, Channel, Decision, Verification
    from cua_jev.runtime import StepResult
    from cua_jev.trace import JsonlTrace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(recorder.sys, "platform", "darwin")
    monkeypatch.setattr(recorder, "load_local_env", lambda: None)
    monkeypatch.setattr(recorder.os, "getenv", lambda key: "test-key-not-a-real-credential")
    monkeypatch.setattr(recorder.time, "sleep", lambda _: None)
    events = []
    original_recording = recorder.MacRecording
    monkeypatch.setattr(recorder, "MacRecording", lambda *args, **kwargs: original_recording(
        *args, **kwargs, bridge_factory=lambda: Bridge(events, fail_stop=finalization_fails),
    ))
    planner = SimpleNamespace(
        planning_requests=1, usage_totals={"total_tokens": 123}, planning_wall_ms=10,
        close=lambda: events.append(("planner.close", {})),
    )
    monkeypatch.setattr(recorder, "_chat_planner", lambda args: planner)
    task = task_fixture(events)
    task.reset = lambda: events.append(("task.reset", {}))
    task.planner_calls = 1
    monkeypatch.setattr(recorder, "_build_task", lambda args, model: task)

    def factory(args):
        def run(environment, *, reset):
            assert reset is False
            receipt = ActionReceipt(
                "obs", "decision", "editor", Channel.CLI, "artifact.open_textedit", True, 1, 2,
                output={"window_title": "guide.txt"},
            )
            verification = Verification(True, "artifact.editor_open")
            step = StepResult(Decision("obs", "editor", {"editor": 1}, 1, "jev-test", 5),
                              receipt, verification)
            status = EpisodeStatus.SUCCESS
            try:
                environment.evaluate(None, SimpleNamespace(capability="artifact.open_textedit"),
                                     receipt, verification)
                environment.close()
            except RuntimeError:
                status = EpisodeStatus.ENVIRONMENT_ERROR
            result = EpisodeResult("fixture", status, "test result", (step,), 1, 2, {"cli": 1})
            trace = JsonlTrace(args.trace)
            trace.append("policy_exchange", {"attempts": 2, "usage": {"total_tokens": 7}})
            trace.append("episode", result.to_dict())
            return result

        return SimpleNamespace(run=run)

    monkeypatch.setattr(recorder, "_open_computer_runner", factory)
    destination = tmp_path / "bundle"
    status = recorder.main([
        "--output", str(destination), "--goal", "Test recorded handoff", "--url", "https://docs.test/",
        "--window-title", "^Fixture$", "--require-window-text-contains", "Reviewed: yes",
        "--allow-window-text-input", "--allow-window-actions", "--local-tool-root", str(tmp_path),
        "--artifact-path", str(tmp_path / "guide.txt"), "--open-artifact-textedit",
        "--model-base-url", "https://model.invalid/v1", "--model", "test-model", "--policy", "jev",
        "--browser-channel", "chrome",
    ])
    stage = next((tmp_path / "runs/recording-work").iterdir())
    manifest = json.loads((stage / "manifest.json").read_text())
    metrics = json.loads((stage / "metrics.json").read_text())
    assert (stage / "trace.jsonl").is_file()
    assert "not one continuous" in manifest["coverage_note"]
    assert metrics["planner_usage"] == {"total_tokens": 123}
    assert metrics["jev_reported_requests"] == 2
    assert "test-key-not-a-real-credential" not in (stage / "manifest.json").read_text()
    assert status == int(finalization_fails)
    assert manifest["recording_complete"] is (not finalization_fails)
    if finalization_fails:
        assert any(segment.get("recording_diagnostics", {}).get("events")
                   for segment in manifest["segments"])
    else:
        assert all(segment["recording_events"] == [{"name": "recording_finished", "at": 1234.5}]
                   for segment in manifest["segments"])
    assert destination.exists() is (not finalization_fails)
    assert events[-1][0] == "planner.close"


@pytest.mark.parametrize("bundle", ["org.chromium.Chromium", "com.google.chrome.for.testing"])
def test_bundled_browser_identity_comes_from_actual_runtime_plist(recorder, tmp_path, bundle):
    contents = tmp_path / "Bundled browser.app/Contents"
    contents.mkdir(parents=True)
    (contents / "Info.plist").write_bytes(plistlib.dumps({"CFBundleIdentifier": bundle}))
    browser = SimpleNamespace(_playwright=SimpleNamespace(chromium=SimpleNamespace(
        executable_path=str(contents / "MacOS/browser"),
    )))
    assert recorder._browser_bundle(browser, "chromium") == bundle
