"""Offline checks for the bounded Mac showcase launcher; never start native apps."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setattr(sys, "path", sys.path.copy())
    source = Path(__file__).resolve().parents[1] / "scripts/macos_python_demo.py"
    spec = importlib.util.spec_from_file_location("macos_python_demo", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path):
    return ["--output", str(tmp_path / "output"), "--model-base-url", "https://model.invalid/v1",
            "--model", "offline-test-model"]


class Probe:
    def __init__(self, *, locked=False):
        self.locked = locked
        self.calls = []
        self.closed = 0

    def call(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command == "health":
            return {"screen_locked": self.locked, "accessibility": True, "screen_recording": True}
        assert command == "select"
        return {"window_id": 42}

    def close(self):
        self.closed += 1


class Child:
    def __init__(self, *, wait_times_out=False):
        self.calls = []
        self.wait_times_out = wait_times_out

    def poll(self):
        return None

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout=None):
        self.calls.append(("wait", timeout))
        if self.wait_times_out:
            from subprocess import TimeoutExpired

            self.wait_times_out = False
            raise TimeoutExpired("owned-child", timeout)
        return 0

    def kill(self):
        self.calls.append("kill")


def patch_environment(demo, monkeypatch, tmp_path, probe):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo.sys, "platform", "darwin")
    monkeypatch.setattr(demo, "load_local_env", lambda: None)
    monkeypatch.setattr(demo, "MacBridge", lambda: probe)
    monkeypatch.setattr(demo.time, "sleep", lambda _: None)


def test_locked_session_stops_before_fixture_caffeinate_or_model(demo, monkeypatch, tmp_path):
    probe = Probe(locked=True)
    patch_environment(demo, monkeypatch, tmp_path, probe)
    monkeypatch.setattr(demo.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch a child"))
    monkeypatch.setattr(demo, "native_binary", lambda: pytest.fail("must not prepare the fixture"))
    monkeypatch.setattr(demo, "record_main", lambda _: pytest.fail("must not call GUI or model APIs"))
    with pytest.raises(RuntimeError, match="Unlock this Mac"):
        demo.main(arguments(tmp_path))
    assert [command for command, _ in probe.calls] == ["health"]
    assert probe.closed == 1
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("outcome", [0, 1, RuntimeError, KeyboardInterrupt])
def test_launcher_reaps_only_its_two_children_on_return_or_failure(
    demo, monkeypatch, tmp_path, outcome,
):
    probe = Probe()
    patch_environment(demo, monkeypatch, tmp_path, probe)
    fixture, awake, unrelated = Child(), Child(), Child()
    launches, calls = [], []
    native_path = tmp_path / "native-fixture"
    monkeypatch.setattr(demo, "native_binary", lambda: native_path)

    def launch(command, **kwargs):
        launches.append(command)
        if command[0] == "/usr/bin/caffeinate":
            assert command[1:] == ["-di", "-w", str(demo.os.getpid())]
            return awake
        assert command == [str(native_path), "--fixture"]
        return fixture

    def record(args):
        calls.append(args)
        assert len(launches) == 2 and probe.closed == 1
        if isinstance(outcome, type):
            raise outcome("recording stopped")
        return outcome

    monkeypatch.setattr(demo.subprocess, "Popen", launch)
    monkeypatch.setattr(demo, "record_main", record)
    if isinstance(outcome, type):
        with pytest.raises(outcome, match="recording stopped"):
            demo.main(arguments(tmp_path))
    else:
        assert demo.main(arguments(tmp_path)) == outcome
    assert len(calls) == 1
    assert [command for command, _ in probe.calls] == ["health", "select"]
    assert probe.closed == 2
    assert fixture.calls == awake.calls == ["terminate", ("wait", 5)]
    assert unrelated.calls == []


def test_launcher_kills_timed_out_owned_child_and_still_reaps_other_child(demo, monkeypatch, tmp_path):
    probe = Probe()
    patch_environment(demo, monkeypatch, tmp_path, probe)
    fixture, awake = Child(wait_times_out=True), Child()
    processes = iter((awake, fixture))
    monkeypatch.setattr(demo, "native_binary", lambda: tmp_path / "native-fixture")
    monkeypatch.setattr(demo.subprocess, "Popen", lambda *a, **k: next(processes))
    monkeypatch.setattr(demo, "record_main", lambda _: 1)
    assert demo.main(arguments(tmp_path)) == 1
    assert fixture.calls == ["terminate", ("wait", 5), "kill", ("wait", None)]
    assert awake.calls == ["terminate", ("wait", 5)]


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_probe_cleanup_failure_still_reaps_both_owned_children(demo, monkeypatch, tmp_path, failure):
    probe = Probe()
    patch_environment(demo, monkeypatch, tmp_path, probe)
    fixture, awake = Child(), Child()
    processes = iter((awake, fixture))
    monkeypatch.setattr(demo, "native_binary", lambda: tmp_path / "native-fixture")
    monkeypatch.setattr(demo.subprocess, "Popen", lambda *a, **k: next(processes))
    monkeypatch.setattr(demo, "record_main", lambda _: 0)

    def fail_final_close():
        probe.closed += 1
        if probe.closed == 2:
            raise failure("probe cleanup failed")

    monkeypatch.setattr(probe, "close", fail_final_close)
    with pytest.raises(failure, match="probe cleanup failed"):
        demo.main(arguments(tmp_path))
    assert fixture.calls == awake.calls == ["terminate", ("wait", 5)]


def test_prepare_task_creates_seven_topic_current_page_scope_without_starting_services(demo, tmp_path):
    root = tmp_path / "new-take"
    output = tmp_path / "raw-output"
    args = demo.prepare_task(root, output, "https://model.invalid/v1", "offline-test-model")

    def option(name):
        return args[args.index(name) + 1]

    topics = [args[index + 1] for index, value in enumerate(args)
              if value == "--require-source-title-contains"]
    assert topics == [
        "More Control Flow Tools", "Data Structures", "Modules", "Input and Output",
        "Errors and Exceptions", "10. Brief Tour of the Standard Library",
        "Virtual Environments and Packages",
    ]
    assert option("--min-verified-sources") == "7"
    assert "--mcp-current-page-only" in args
    assert option("--url") == "https://docs.python.org/3/tutorial/index.html"
    profile = json.loads(Path(option("--mcp-profile")).read_text())
    assert profile["server"]["args"] == [
        "-m", "cua_jev.mcp_web_reader", "--origin", "https://docs.python.org",
    ]
    assert Path(profile["server"]["command"]).is_absolute()
    assert len(profile["tools"]) == 1
    reader = profile["tools"][0]
    assert reader["name"] == "fetch_page"
    assert reader["parameters"] == {"href": {"type": "string", "maxLength": 500}}
    assert reader["required"] == ["href"] and reader["expected_from_arguments"] == {"url": "href"}
    workspace = Path(option("--local-tool-root"))
    artifact = Path(option("--artifact-path"))
    assert workspace == root / "workspace" and list(workspace.iterdir()) == []
    assert artifact.parent == workspace and not artifact.exists()
    assert option("--require-window-text-contains") == f"Saved: {artifact.name}\nReviewed: yes"
    assert option("--policy") == "jev" and "--route" not in args
    assert json.loads((root / "task.json").read_text())["args"] == args
    assert not output.exists()


def test_prepare_task_never_overwrites_existing_scope_or_symlink(demo, tmp_path):
    root = tmp_path / "existing-take"
    root.mkdir()
    original = root / "reader.json"
    original.write_text("keep existing profile")
    alias = tmp_path / "linked-take"
    alias.symlink_to(root, target_is_directory=True)
    for scope in (root, alias):
        with pytest.raises(FileExistsError):
            demo.prepare_task(scope, tmp_path / "output", "https://model.invalid/v1", "offline-test-model")
    assert original.read_text() == "keep existing profile"
    assert not (root / "workspace").exists()


def test_existing_output_is_rejected_before_loading_credentials_or_services(demo, monkeypatch, tmp_path):
    monkeypatch.setattr(demo.sys, "platform", "darwin")
    (tmp_path / "output").mkdir()
    monkeypatch.setattr(demo, "load_local_env", lambda: pytest.fail("must reject before loading credentials"))
    monkeypatch.setattr(demo, "MacBridge", lambda: pytest.fail("must not launch native helper"))
    with pytest.raises(SystemExit, match="2"):
        demo.main(arguments(tmp_path))
