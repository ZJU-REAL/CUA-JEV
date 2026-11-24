"""JSON RPC public surfaces, typed-input restrictions and evaluation lifecycle."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("remote_worker", _SCRIPTS / "osworld_remote_worker.py")
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


def observation():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80)).save(buffer, format="PNG")
    return {
        "screenshot": buffer.getvalue(),
        "accessibility_tree": "<desktop />",
        "evaluator": "SECRET_EVALUATOR",
        "instruction": "SECRET_DUPLICATE",
    }


def test_public_observation_excludes_all_private_fields():
    serialized = worker.public_observation(observation())
    assert set(serialized) == {"screenshot", "accessibility_tree"}
    assert serialized["screenshot"].startswith("iVBOR")
    assert "SECRET" not in json.dumps(serialized)


def test_ready_timing_uses_stderr_and_preserves_exact_rpc_frame(capsys):
    timeline = {"phases_ms": {}}
    output = io.StringIO()
    frame = {"event": "ready", "protocol": 1, "instruction": "PRIVATE public task"}
    worker.emit_frame(output, frame, startup_timeline=timeline)
    assert output.getvalue() == json.dumps(frame) + "\n"
    phases = timeline["phases_ms"]
    assert list(phases) == ["ready_serialization_started", "ready_write_started",
                            "ready_write_completed", "ready_flush_completed"]
    assert list(phases.values()) == sorted(phases.values())
    captured = capsys.readouterr()
    assert captured.out == "" and "PRIVATE" not in captured.err
    assert [json.loads(line)["phase"] for line in captured.err.splitlines()] == list(phases)


@pytest.mark.parametrize("failure", ["write", "flush"])
def test_ready_timing_does_not_claim_failed_write_or_flush(failure):
    class BrokenStream(io.StringIO):
        def write(self, value):
            if failure == "write":
                raise BrokenPipeError("private pipe details")
            return super().write(value)

        def flush(self):
            raise BrokenPipeError("private pipe details")

    timeline = {"phases_ms": {}}
    with pytest.raises(BrokenPipeError):
        worker.emit_frame(BrokenStream(), {"event": "ready"}, startup_timeline=timeline)
    assert "ready_flush_completed" not in timeline["phases_ms"]
    assert ("ready_write_completed" in timeline["phases_ms"]) is (failure == "flush")


def test_closed_diagnostic_stream_does_not_interrupt_cleanup_timing(monkeypatch):
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(sys, "stderr", closed)
    timeline = {"phases_ms": {}}
    worker.record_startup_phase(timeline, "environment_close_started")
    assert timeline["phases_ms"]["environment_close_started"] >= 0


@pytest.mark.parametrize("readiness_fails", [False, True])
@pytest.mark.parametrize("settling_enabled", [False, True])
def test_worker_main_persists_startup_timeline_and_cleanup_without_extra_rpc_events(
    monkeypatch, tmp_path, readiness_fails, settling_enabled,
):
    # Exercise main with in-memory pipes and a fake harness, never an OSWorld/VM import.
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    osworld = project / "OSWorld"
    task_file = osworld / "evaluation_examples/examples/os/task.json"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("{}")
    output = project / "runs/new"
    owner_uid = project.stat().st_uid
    events = []
    initial = observation()
    png = io.BytesIO()
    Image.new("RGB", (100, 80), "white").save(png, format="PNG")
    initial["screenshot"] = png.getvalue()

    class Clock:
        now = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    smoke = sys.modules[worker.wait_for_desktop_settled.__module__]
    monkeypatch.setattr(smoke, "time", Clock())
    monkeypatch.setattr(smoke, "_readiness_observe", lambda env, _remaining: env._get_obs())

    class Pipe(io.StringIO):
        def fileno(self):
            return 123

        def close(self):
            self.was_closed = True

    rpc, stderr = Pipe(), Pipe()

    class Environment:
        def __init__(self, **_):
            events.append("environment_created")

        def close(self):
            events.append("environment_closed")

        def _get_obs(self):
            events.append("observe")
            return initial

    @contextmanager
    def context(_args, _owner):
        runtime = {"cleanup_confirmed": False}
        try:
            yield runtime
        finally:
            runtime["cleanup_confirmed"] = True
            events.append("context_closed")

    def readiness(_env, observed, *, output, report):
        if readiness_fails:
            raise RuntimeError("PRIVATE startup exception body")
        report["desktop_readiness"] = {"status": "ready"}
        return observed

    monkeypatch.setattr(worker, "__file__", str(project / "scripts/worker.py"))
    monkeypatch.setattr(worker.os, "getuid", lambda: owner_uid, raising=False)
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(
        getpwuid=lambda _uid: SimpleNamespace(pw_name="experimenter"),
    ))
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", SimpleNamespace(DesktopEnv=Environment))
    monkeypatch.setattr(worker, "validate_vm_path", lambda path: path)
    monkeypatch.setattr(worker, "load_task", lambda *_: {"instruction": "Open settings"})
    monkeypatch.setattr(worker, "source_metadata", lambda *_: {"source": "fake"})
    monkeypatch.setattr(worker, "owned_vm_context", context)
    monkeypatch.setattr(worker, "reset_checked", lambda *_: initial)
    monkeypatch.setattr(worker, "wait_for_desktop_ready", readiness)
    real_session = worker.WorkerSession
    monkeypatch.setattr(worker, "WorkerSession", lambda *args: real_session(
        *args, guard_factory=lambda _env: SimpleNamespace(close=lambda: events.append("guard_closed")),
    ))
    monkeypatch.setattr(worker.os, "dup", lambda _fd: 123)
    monkeypatch.setattr(worker.os, "dup2", lambda *_: None)
    monkeypatch.setattr(worker.os, "fdopen", lambda *_a, **_k: rpc)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "stdout", Pipe())
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))  # EOF closes the owned environment.
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    code = worker.main([
        "--expected-user", "experimenter", "--osworld-root", str(osworld),
        "--vm-path", str(project / "base.qcow2"), "--task", "os/task", "--output", str(output),
    ] + ([] if settling_enabled else ["--no-passive-settling"]))
    assert code == int(readiness_fails)
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["vm_runtime"]["cleanup_confirmed"] is True
    timeline = metadata["startup_timeline"]
    assert timeline["origin"] == "before_project_imports"
    phases = timeline["phases_ms"]
    assert list(phases.values()) == sorted(phases.values())
    assert phases["project_imports_completed"] <= phases["osworld_import_started"]
    assert phases["environment_close_started"] <= phases["environment_close_completed"]
    assert phases["environment_close_completed"] <= phases["worker_finished"]
    assert events[-2:] == ["environment_closed", "context_closed"]
    frames = [json.loads(line) for line in rpc.getvalue().splitlines()]
    assert len(frames) == 1 and frames[0]["event"] == ("fatal" if readiness_fails else "ready")
    assert "PRIVATE" not in stderr.getvalue() + json.dumps(metadata) + rpc.getvalue()
    if readiness_fails:
        assert "ready_write_started" not in phases
        assert not (output / "official-result.json").exists()
    else:
        settling = frames[0]["passive_settling"]
        assert settling["enabled"] is settling_enabled
        assert settling["status"] == ("budget_exhausted" if settling_enabled else "disabled")
        assert settling["application_ready_proven"] is False
        assert settling["elapsed_ms"] == (9000 if settling_enabled else 0)
        assert events.count("observe") == (9 if settling_enabled else 0)
        assert phases["passive_settling_completed"] <= phases["ready_serialization_started"]
        assert metadata["passive_settling"]["samples_observed"] == (10 if settling_enabled else 0)
        assert phases["ready_flush_completed"] <= phases["environment_close_started"]
        assert json.loads((output / "official-result.json").read_text())["closed"] is True


def test_expected_user_is_required_and_not_hardcoded(capsys):
    arguments = [
        "--osworld-root",
        "/repo",
        "--vm-path",
        "/base.qcow2",
        "--task",
        "os/task",
        "--output",
        "/repo/runs/new",
    ]
    with pytest.raises(SystemExit):
        worker.parse_args(arguments)
    assert "--expected-user" in capsys.readouterr().err
    args = worker.parse_args([*arguments, "--expected-user", "experimenter"])
    assert args.vm_provider == "docker"
    assert worker.parse_args([*arguments, "--expected-user", "experimenter",
                              "--vm-provider", "qemu"]).vm_provider == "qemu"
    assert worker.parse_args([*arguments, "--expected-user", "experimenter",
                              "--vm-provider", "qemu", "--guest-mcp-port"]).guest_mcp_port
    with pytest.raises(SystemExit):
        worker.parse_args([*arguments, "--expected-user", "experimenter", "--guest-mcp-port"])
    assert worker.require_expected_user(args.expected_user, "experimenter") == "experimenter"
    with pytest.raises(PermissionError):
        worker.require_expected_user("experimenter", "someone-else")


@pytest.mark.parametrize("startup_fails", [False, True])
@pytest.mark.parametrize("boot_timeout", [None, 300.0])
def test_qemu_worker_context_scopes_owned_ports_and_never_falls_back_to_docker(
    monkeypatch, tmp_path, startup_fails, boot_timeout,
):
    events = []
    ports = set()
    metadata = {"actual_provider": "qemu", "network_mode": "user_restricted_loopback"}

    @contextmanager
    def qemu(base, output, owner):
        assert base == tmp_path / "base.qcow2" and output == tmp_path / "run" and owner == "experimenter"
        events.append("qemu_enter")
        try:
            if startup_fails:
                raise RuntimeError("startup failed")
            runtime = SimpleNamespace(metadata=metadata, owned_ports=ports, startup_timeout_s=180)
            yield runtime
            assert runtime.startup_timeout_s == (boot_timeout or 180)
        finally:
            events.append("qemu_close")

    @contextmanager
    def bounded(actual):
        assert actual is ports
        events.append("http_enter")
        try:
            yield
        finally:
            events.append("http_close")

    monkeypatch.setitem(sys.modules, "cua_jev.osworld_qemu", SimpleNamespace(isolated_qemu=qemu))
    monkeypatch.setattr(worker, "bounded_vm_http", bounded)
    monkeypatch.setattr(
        worker, "isolated_docker", lambda *_a, **_k: pytest.fail("unexpected Docker fallback")
    )
    args = SimpleNamespace(vm_provider="qemu", vm_path=tmp_path / "base.qcow2", output=tmp_path / "run",
                           qemu_boot_timeout=boot_timeout)
    if startup_fails:
        with pytest.raises(RuntimeError, match="startup failed"):
            with worker.owned_vm_context(args, "experimenter"):
                pytest.fail("failed startup must not yield")
        assert events == ["qemu_enter", "qemu_close"]
    else:
        with worker.owned_vm_context(args, "experimenter") as actual:
            assert actual is metadata
            ports.add(43210)
        assert events == ["qemu_enter", "http_enter", "http_close", "qemu_close"]


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf", "601"])
def test_qemu_boot_budget_rejects_unbounded_or_invalid_values(budget):
    with pytest.raises(SystemExit):
        worker.parse_args(["--expected-user", "experimenter", "--osworld-root", "/repo",
                           "--vm-path", "/base.qcow2", "--task", "os/task", "--output", "/repo/run",
                           "--vm-provider", "qemu", "--qemu-boot-timeout", budget])


def cached_task():
    return {
        "id": "task-id",
        "instruction": "Open the profile page",
        "config": [
            {
                "type": "download",
                "parameters": {
                    "files": [
                        {
                            "url": "https://assets.example.test/original-profile.tar.gz",
                            "path": "/home/user/profile.tar.gz",
                        }
                    ]
                },
            }
        ],
        "evaluator": {"secret": "PRIVATE_EVALUATION"},
    }


def cache_name(task):
    item = task["config"][0]["parameters"]["files"][0]
    return f"{uuid.uuid5(uuid.NAMESPACE_URL, item['url'])}_profile.tar.gz"


@pytest.mark.parametrize("per_task_directory", [False, True])
def test_original_asset_cache_copies_into_new_run_without_changing_source(tmp_path, per_task_directory):
    task = cached_task()
    unchanged_config = json.dumps(task, sort_keys=True)
    cache, output = tmp_path / "source", tmp_path / "run"
    cache.mkdir()
    location = cache / task["id"] if per_task_directory else cache
    location.mkdir(exist_ok=True)
    source = location / cache_name(task)
    source.write_bytes(b"original-profile-bytes")
    source_mtime = source.stat().st_mtime_ns
    output.mkdir()
    records = worker.seed_asset_cache(task, cache, output, owner_uid=cache.stat().st_uid)
    destination = output / "cache" / task["id"] / cache_name(task)
    assert destination.read_bytes() == source.read_bytes() == b"original-profile-bytes"
    assert source.stat().st_mtime_ns == source_mtime
    assert json.dumps(task, sort_keys=True) == unchanged_config
    assert records == [
        {
            "source": str(source.resolve()),
            "destination": str(destination),
            "sha256": hashlib.sha256(b"original-profile-bytes").hexdigest(),
            "size_bytes": 22,
        }
    ]
    public = worker.public_observation({**observation(), "metadata": records, "task": task})
    assert "PRIVATE_EVALUATION" not in json.dumps(public)
    assert str(source) not in json.dumps(public)


def test_asset_cache_miss_keeps_original_download_configuration(tmp_path):
    cache, output = tmp_path / "source", tmp_path / "run"
    cache.mkdir()
    output.mkdir()
    task = cached_task()
    unchanged = json.dumps(task, sort_keys=True)
    assert worker.seed_asset_cache(task, None, output, owner_uid=cache.stat().st_uid) == []
    assert worker.seed_asset_cache(task, cache, output, owner_uid=cache.stat().st_uid) == []
    assert not (output / "cache").exists()
    assert json.dumps(task, sort_keys=True) == unchanged


def test_asset_cache_requires_owned_directory_regular_owned_file_and_no_links(tmp_path, monkeypatch):
    cache, output = tmp_path / "source", tmp_path / "run"
    cache.mkdir()
    output.mkdir()
    task = cached_task()
    owner = cache.stat().st_uid
    with pytest.raises(PermissionError, match="directory"):
        worker.seed_asset_cache(task, cache, output, owner_uid=owner + 1)
    source = cache / cache_name(task)
    source.mkdir()
    with pytest.raises(PermissionError, match="regular"):
        worker.seed_asset_cache(task, cache, output, owner_uid=owner)
    source.rmdir()
    source.write_bytes(b"untouched")
    original_stat = Path.stat

    def fake_stat(path, *args, **kwargs):
        value = original_stat(path, *args, **kwargs)
        if path == source:
            return SimpleNamespace(st_uid=owner + 1, st_mode=value.st_mode)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", fake_stat)
        with pytest.raises(PermissionError, match="regular"):
            worker.seed_asset_cache(task, cache, output, owner_uid=owner)
    original_link_check = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == source or original_link_check(path))
    with pytest.raises(PermissionError, match="symlink"):
        worker.seed_asset_cache(task, cache, output, owner_uid=owner)
    assert source.read_bytes() == b"untouched" and not (output / "cache").exists()


def test_asset_cache_rejects_task_path_traversal(tmp_path):
    task = {**cached_task(), "id": "../../another-user"}
    with pytest.raises(ValueError, match="task id"):
        worker.seed_asset_cache(task, tmp_path, tmp_path / "run", owner_uid=tmp_path.stat().st_uid)


@pytest.mark.parametrize(
    "action",
    [
        "pyautogui.click(1, 2)",
        {"action_type": "EXECUTE", "parameters": {"command": "code"}},
        {"action_type": "CLICK", "parameters": {"x": 101, "y": 1, "button": "left", "num_clicks": 1}},
        {"action_type": "TYPING", "parameters": {"text": "密码"}},
        {"action_type": "PRESS", "parameters": {"key": "');import os;#"}},
        {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "'"]}},
        {"action_type": "SCROLL", "parameters": {"dx": 1, "dy": 1}},
        {"action_type": "TYPING", "parameters": {"text": "okay"}, "command": "hidden"},
    ],
)
def test_worker_rejects_arbitrary_commands_and_invalid_input(action):
    with pytest.raises(worker.ProtocolError):
        worker.validate_action(action, observation())


def test_rpc_evaluate_once_and_close_cleanup(tmp_path):
    calls = []

    class Env:
        controller = SimpleNamespace()

        def _get_obs(self):
            return observation()

        def step(self, action, *, pause):
            calls.append("step")
            assert action["action_type"] == "PRESS" and pause == 0.5
            return observation(), 999, True, {"private": "SECRET"}

        def evaluate(self):
            calls.append("evaluate")
            return 1.0

        def close(self):
            calls.append("close")

    session = worker.WorkerSession(Env(), observation(), tmp_path)
    requests = [
        {"id": 1, "method": "observe"},
        {"id": 2, "method": "step", "action": {"action_type": "PRESS", "parameters": {"key": "enter"}}},
        {"id": 3, "method": "evaluate"},
        {"id": 4, "method": "evaluate"},
        {"id": 5, "method": "observe"},
        {"id": 6, "method": "close"},
    ]
    output = io.StringIO()
    worker.serve(session, io.StringIO("".join(json.dumps(r) + "\n" for r in requests)), output)
    frames = [json.loads(line) for line in output.getvalue().splitlines()]
    assert frames[1]["result"]["reward"] == 0 and not frames[1]["result"]["done"]
    assert frames[1]["result"]["info"] == {}
    assert frames[2]["result"] == 1.0
    assert not frames[3]["ok"] and not frames[4]["ok"]
    assert frames[5]["result"] is None
    assert calls == ["step", "evaluate", "close"]
    assert "SECRET" not in output.getvalue()
    session.close()
    assert calls.count("close") == 1
    saved = json.loads((tmp_path / "official-result.json").read_text())
    assert saved["official_score"] == 1.0 and saved["closed"]


def test_controller_none_is_an_explicit_failed_rpc_without_leaking_exception(tmp_path):
    class Env:
        controller = SimpleNamespace(execute_python_command=lambda *_: None)

        def step(self, *_a, **_k):
            self.controller.execute_python_command("typed command")
            pytest.fail("controller failure must be surfaced")

        def close(self):
            pass

    session = worker.WorkerSession(Env(), observation(), tmp_path)
    output = io.StringIO()
    request = {"id": 1, "method": "step", "action": {"action_type": "TYPING", "parameters": {"text": "safe"}}}
    worker.serve(session, io.StringIO(json.dumps(request) + "\n"), output)
    frame = json.loads(output.getvalue())
    assert not frame["ok"] and frame["error"] == "RuntimeError"
    assert "typed command" not in output.getvalue()
    session.close()


def test_eof_owner_closes_without_evaluation_and_malformed_request_is_safe(tmp_path):
    calls = []
    env = SimpleNamespace(
        controller=SimpleNamespace(),
        close=lambda: calls.append("close"),
        evaluate=lambda: pytest.fail("EOF must not evaluate"),
    )
    session = worker.WorkerSession(env, observation(), tmp_path)
    output = io.StringIO()
    try:
        worker.serve(session, io.StringIO("malformed-secret\n"), output)
    finally:
        session.close()
    assert calls == ["close"]
    assert "malformed-secret" not in output.getvalue()
    assert json.loads(output.getvalue())["error"] == "JSONDecodeError"
