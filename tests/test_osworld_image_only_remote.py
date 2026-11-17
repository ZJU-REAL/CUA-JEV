"""Negotiated partial step evidence over local pipes and an in-memory worker."""

from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from cua_jev.osworld_remote import RemoteOSWorldEnv, RemoteOSWorldError
from cua_jev.osworld_step_observation import IMAGE_ONLY_INFO_KEY, IMAGE_ONLY_VERSION

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "image_only_remote_worker", _SCRIPTS / "osworld_remote_worker.py",
)
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)

CLICK = {"action_type": "CLICK", "parameters": {"x": 10, "y": 12, "button": "left", "num_clicks": 1}}
SELECT = {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "a"]}}
TYPING = {"action_type": "TYPING", "parameters": {"text": "final value"}}
MARKER = {IMAGE_ONLY_INFO_KEY: IMAGE_ONLY_VERSION}


def observation(*, tree="<desktop />", shade=0):
    buffer = io.BytesIO()
    Image.new("RGB", (100, 80), (shade, shade, shade)).save(buffer, format="PNG")
    return {"screenshot": buffer.getvalue(), "accessibility_tree": tree}


class Environment:
    require_a11y_tree = True

    def __init__(self, failure=None):
        self.controller = SimpleNamespace()
        self.calls = []
        self.failure = failure
        self.evaluations = 0
        self.closed = False

    def _get_obs(self):
        self.calls.append(("observe", self.require_a11y_tree))
        return observation(tree="<fresh />" if self.require_a11y_tree else None, shade=len(self.calls))

    def step(self, action, *, pause):
        self.calls.append(("step", action, pause, self.require_a11y_tree))
        if self.failure == "raise":
            raise RuntimeError("PRIVATE controller failure")
        if self.failure == "tuple":
            return (None,)
        observed = self._get_obs()
        if self.failure == "tree":
            observed["accessibility_tree"] = "<wrong-old-tree />"
        if self.failure == "image":
            observed["screenshot"] = None
        return observed, 999, True, ({"error": "PRIVATE"} if self.failure == "info" else {})

    def evaluate(self):
        self.evaluations += 1
        return 1.0

    def close(self):
        self.closed = True


def request(action, *, mode=IMAGE_ONLY_VERSION, identifier=1):
    result = {"id": identifier, "method": "step", "action": action, "pause": 0.25}
    if mode is not None:
        result["observation_mode"] = mode
    return result


def test_worker_keeps_intermediate_screenshots_counters_and_fresh_reads(tmp_path):
    env = Environment()
    session = worker.WorkerSession(env, observation(), tmp_path, image_only_fill_intermediates=True)
    first = session.handle(request(CLICK))
    assert first["info"] == MARKER and first["observation"]["accessibility_tree"] is None
    first_image = (tmp_path / "step-001.png").read_bytes()
    assert first_image == session.observation["screenshot"]
    assert session.handle({"id": 2, "method": "observe"})["accessibility_tree"] == "<fresh />"
    second = session.handle(request(SELECT, identifier=3))
    assert second["info"] == MARKER and second["observation"]["accessibility_tree"] is None
    assert (tmp_path / "step-002.png").read_bytes() != first_image
    assert session.handle({"id": 4, "method": "observe"})["accessibility_tree"] == "<fresh />"
    final = session.handle(request(TYPING, mode=None, identifier=5))
    assert final["info"] == {} and final["observation"]["accessibility_tree"] == "<fresh />"
    assert [(call[1], call[2], call[3]) for call in env.calls if call[0] == "step"] == [
        (CLICK, 0.25, False), (SELECT, 0.25, False), (TYPING, 0.25, True),
    ]
    assert env.require_a11y_tree is True and session.steps == 3
    assert session.handle({"id": 6, "method": "evaluate"}) == 1.0
    assert env.evaluations == 1
    assert (tmp_path / "final.png").read_bytes() == (tmp_path / "step-003.png").read_bytes()
    with pytest.raises(worker.ProtocolError):
        session.handle({"id": 7, "method": "evaluate"})
    session.close()
    result = json.loads((tmp_path / "official-result.json").read_text())
    assert result["environment_steps"] == 3 and result["official_score"] == 1 and result["closed"]


@pytest.mark.parametrize("failure", ["raise", "tuple", "tree", "image", "info"])
def test_worker_restores_flag_and_never_retries_failed_intermediate_input(tmp_path, failure):
    env = Environment(failure)
    session = worker.WorkerSession(env, observation(), tmp_path, image_only_fill_intermediates=True)
    with pytest.raises((RuntimeError, worker.ActionError)):
        session.handle(request(CLICK))
    assert env.require_a11y_tree is True
    assert len([call for call in env.calls if call[0] == "step"]) == 1
    assert session.steps == 0
    assert not (tmp_path / "step-001.png").exists()
    session.close()


@pytest.mark.parametrize("flag", [None, False, 1, "true"])
def test_worker_enabled_option_requires_exact_native_true_before_input(tmp_path, flag):
    env = Environment()
    env.require_a11y_tree = flag
    with pytest.raises(ValueError):
        worker.WorkerSession(env, observation(), tmp_path, image_only_fill_intermediates=True)
    assert env.calls == [] and not env.closed


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("mode", [True, "full", "image_only_v2", {"mode": "image_only_v1"}])
def test_worker_rejects_unsupported_or_malformed_modes_before_input(tmp_path, enabled, mode):
    env = Environment()
    session = worker.WorkerSession(env, observation(), tmp_path, image_only_fill_intermediates=enabled)
    with pytest.raises(worker.ProtocolError):
        session.handle(request(CLICK, mode=mode))
    assert env.calls == [] and session.steps == 0
    session.close()


@pytest.mark.parametrize("action", [
    TYPING,
    {"action_type": "CLICK", "parameters": {**CLICK["parameters"], "button": "right"}},
    {"action_type": "CLICK", "parameters": {**CLICK["parameters"], "num_clicks": 2}},
    {"action_type": "CLICK", "parameters": {**CLICK["parameters"], "x": 100}},
    {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "s"]}},
    {"action_type": "PRESS", "parameters": {"key": "enter"}},
    {"action_type": "SCROLL", "parameters": {"dx": 0, "dy": -3}},
])
def test_worker_image_only_mode_does_not_accept_other_inputs(tmp_path, action):
    env = Environment()
    session = worker.WorkerSession(env, observation(), tmp_path, image_only_fill_intermediates=True)
    with pytest.raises(ValueError):
        session.handle(request(action))
    assert env.calls == [] and session.steps == 0 and env.require_a11y_tree is True
    session.close()


def test_worker_default_has_no_partial_mode_and_normal_step_stays_full(tmp_path):
    env = Environment()
    session = worker.WorkerSession(env, observation(), tmp_path)
    with pytest.raises(worker.ProtocolError):
        session.handle(request(CLICK))
    assert env.calls == []
    result = session.handle(request(CLICK, mode=None))
    assert result["info"] == {} and result["observation"]["accessibility_tree"] == "<fresh />"
    assert session.steps == 1 and env.require_a11y_tree is True
    session.close()


def test_worker_flag_is_opt_in_and_conflicting_no_a11y_rejected(capsys):
    args = ["--expected-user", "user", "--osworld-root", "/owned", "--vm-path", "/owned/base.qcow2",
            "--task", "os/task", "--output", "/owned/run"]
    assert worker.parse_args(args).image_only_fill_intermediates is False
    assert worker.parse_args([*args, "--image-only-fill-intermediates"]).image_only_fill_intermediates is True
    with pytest.raises(SystemExit):
        worker.parse_args([*args, "--image-only-fill-intermediates", "--no-a11y"])
    assert "requires accessibility" in capsys.readouterr().err


@pytest.mark.parametrize("enabled,native_flag", [(False, None), (True, True), (True, 1), (True, False)])
def test_worker_main_advertises_only_explicit_supported_option(monkeypatch, tmp_path, enabled, native_flag):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    osworld = project / "OSWorld"
    task_file = osworld / "evaluation_examples/examples/os/task.json"
    task_file.parent.mkdir(parents=True)
    task_file.write_text("{}")
    output = project / "runs/new"
    owner_uid = project.stat().st_uid
    events = []

    class Pipe(io.StringIO):
        def fileno(self):
            return 123

        def close(self):
            pass

    rpc = Pipe()

    class EnvironmentStub(Environment):
        def __init__(self, **_kwargs):
            super().__init__()
            self.require_a11y_tree = native_flag

        def close(self):
            events.append("closed")

    @contextmanager
    def owned_context(_args, _owner):
        runtime = {"cleanup_confirmed": False}
        try:
            yield runtime
        finally:
            runtime["cleanup_confirmed"] = True
            events.append("context_closed")

    def settle(_env, observed, *, output, report, enabled):
        report["passive_settling"] = {}
        return observed

    monkeypatch.setattr(worker, "__file__", str(project / "scripts/worker.py"))
    monkeypatch.setattr(worker.os, "getuid", lambda: owner_uid, raising=False)
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(
        getpwuid=lambda _uid: SimpleNamespace(pw_name="experimenter"),
    ))
    monkeypatch.setitem(sys.modules, "desktop_env.desktop_env", SimpleNamespace(DesktopEnv=EnvironmentStub))
    monkeypatch.setattr(worker, "validate_vm_path", lambda path: path)
    monkeypatch.setattr(worker, "load_task", lambda *_: {"instruction": "Change one setting"})
    monkeypatch.setattr(worker, "source_metadata", lambda *_: {"source": "fake"})
    monkeypatch.setattr(worker, "owned_vm_context", owned_context)
    monkeypatch.setattr(worker, "reset_checked", lambda *_: observation())
    monkeypatch.setattr(worker, "wait_for_desktop_ready", lambda _env, obs, **_kwargs: obs)
    monkeypatch.setattr(worker, "wait_for_desktop_settled", settle)
    monkeypatch.setattr(worker, "settling_summary", lambda _value: {})
    monkeypatch.setattr(worker, "save_observation_artifacts", lambda *_: None)
    monkeypatch.setattr(worker.os, "dup", lambda _fd: 123)
    monkeypatch.setattr(worker.os, "dup2", lambda *_: None)
    monkeypatch.setattr(worker.os, "fdopen", lambda *_a, **_k: rpc)
    monkeypatch.setattr(worker.signal, "signal", lambda *_: None)
    monkeypatch.setattr(sys, "stdout", Pipe())
    monkeypatch.setattr(sys, "stderr", Pipe())
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", sys.dont_write_bytecode)
    args = ["--expected-user", "experimenter", "--osworld-root", str(osworld),
            "--vm-path", str(project / "base.qcow2"), "--task", "os/task", "--output", str(output)]
    code = worker.main([*args, *(["--image-only-fill-intermediates"] if enabled else [])])
    supported = enabled and native_flag is True
    assert code == int(enabled and not supported)
    frames = [json.loads(line) for line in rpc.getvalue().splitlines()]
    assert len(frames) == 1
    if code:
        assert frames[0]["event"] == "fatal" and "capabilities" not in frames[0]
    elif supported:
        assert frames[0]["capabilities"] == {
            "intermediate_fill_observation": "image_only_v1", "screenshot_only": "screenshot_v1",
        }
    else:
        assert frames[0]["event"] == "ready"
        assert frames[0]["capabilities"] == {"screenshot_only": "screenshot_v1"}
    assert events == ["closed", "context_closed"]
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["vm_runtime"]["cleanup_confirmed"] is True
    assert (metadata.get("intermediate_fill_observation") == "image_only_v1") is supported


PIPE_WORKER = r'''
import base64,json,sys,time
mode,path=sys.argv[1:]
obs={"screenshot":base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode(),"accessibility_tree":"<full />"}
ready={"event":"ready","protocol":1,"instruction":"Change one setting","observation":obs}
if mode != "legacy":
 ready["capabilities"]={"intermediate_fill_observation":"image_only_v1"}
if mode == "bad_capability": ready["capabilities"]={"intermediate_fill_observation":True}
def emit(obj): print(json.dumps(obj),flush=True)
emit(ready)
for line in sys.stdin:
 req=json.loads(line)
 with open(path,"a") as out: out.write(json.dumps(req)+"\n")
 if req["method"] == "close":
  emit({"id":req["id"],"ok":True,"result":None});break
 if mode == "lost": break
 if mode == "timeout": time.sleep(0.15)
 if mode == "rejected":
  emit({"id":req["id"],"ok":False,"error":"PRIVATE"});continue
 result=obs
 if req["method"] == "step":
  partial="observation_mode" in req
  returned={**obs,"accessibility_tree":None} if partial else obs
  info={"cua_jev_intermediate_observation":"image_only_v1"} if partial else {}
  if mode == "missing_marker": info={}
  if mode == "wrong_marker": info={"cua_jev_intermediate_observation":"image_only_v2"}
  if mode == "extra_info": info["private"]="PRIVATE"
  if mode == "full_tree": returned=obs
  if mode == "unexpected_partial":
   info={"cua_jev_intermediate_observation":"image_only_v1"}
   returned={**obs,"accessibility_tree":None}
  result={"observation":returned,"reward":0,"done":False,"info":info}
 emit({"id":req["id"],"ok":True,"result":result})
'''


def remote(tmp_path, mode):
    requests = tmp_path / "requests.jsonl"

    def launch(_argv, **kwargs):
        return subprocess.Popen(
            [sys.executable, "-B", "-u", "-c", PIPE_WORKER, mode, str(requests)], **kwargs,
        )

    env = RemoteOSWorldEnv("owned-host", ["owned-worker"], log_path=tmp_path / "worker.log",
                           timeout_s=0.05 if mode == "timeout" else 1,
                           startup_timeout_s=2, close_timeout_s=1, popen_factory=launch)
    return env, requests


def test_negotiated_pipe_mode_is_explicit_and_regular_requests_stay_unchanged(tmp_path):
    env, requests = remote(tmp_path, "enabled")
    assert env.supports_image_only_steps is True
    assert env._get_obs()["accessibility_tree"] == "<full />"
    result = env.step_image_only(CLICK, pause=0.25)
    assert result[0]["accessibility_tree"] is None and result[3] == MARKER
    assert env.step(TYPING)[0]["accessibility_tree"] == "<full />"
    env.close()
    sent = [json.loads(line) for line in requests.read_text().splitlines()]
    assert sent == [
        {"id": 1, "method": "observe"},
        {"id": 2, "method": "step", "action": CLICK, "pause": 0.25, "observation_mode": "image_only_v1"},
        {"id": 3, "method": "step", "action": TYPING, "pause": 0.5},
        {"id": 4, "method": "close"},
    ]
    assert env.cleanup_status == "confirmed"


def test_legacy_pipe_cannot_request_image_only_and_default_wire_is_unchanged(tmp_path):
    env, requests = remote(tmp_path, "legacy")
    assert env.supports_image_only_steps is False
    with pytest.raises(ValueError):
        env.step_image_only(CLICK)
    assert env.step(CLICK)[0]["accessibility_tree"] == "<full />"
    env.close()
    sent = [json.loads(line) for line in requests.read_text().splitlines()]
    assert sent == [{"id": 1, "method": "step", "action": CLICK, "pause": 0.5},
                    {"id": 2, "method": "close"}]


@pytest.mark.parametrize("mode", [
    "missing_marker", "wrong_marker", "extra_info", "full_tree", "rejected", "lost", "timeout",
])
def test_pipe_invalid_or_lost_partial_response_is_static_failure_without_input_retry(tmp_path, mode):
    env, requests = remote(tmp_path, mode)
    try:
        with pytest.raises(RemoteOSWorldError) as caught:
            env.step_image_only(SELECT)
        assert "PRIVATE" not in str(caught.value)
    finally:
        if mode in {"lost", "timeout"}:
            with pytest.raises(RemoteOSWorldError, match="cleanup was not confirmed"):
                env.close()
            assert env.cleanup_status == "unconfirmed_worker_exit"
        else:
            env.close()
    sent = [json.loads(line) for line in requests.read_text().splitlines()]
    assert len([row for row in sent if row["method"] == "step"]) == 1


def test_normal_step_rejects_partial_evidence_even_from_capable_worker(tmp_path):
    env, requests = remote(tmp_path, "unexpected_partial")
    try:
        with pytest.raises(RemoteOSWorldError):
            env.step(TYPING)
    finally:
        env.close()
    assert sum(json.loads(line)["method"] == "step" for line in requests.read_text().splitlines()) == 1


def test_pipe_invalid_capability_fails_startup(tmp_path):
    with pytest.raises(RemoteOSWorldError) as caught:
        remote(tmp_path, "bad_capability")
    assert caught.value.code == "invalid_handshake"


@pytest.mark.parametrize("action", [TYPING, {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "s"]}}])
def test_remote_rejects_nonintermediate_input_without_sending_rpc(tmp_path, action):
    env, requests = remote(tmp_path, "enabled")
    try:
        with pytest.raises(ValueError):
            env.step_image_only(action)
    finally:
        env.close()
    assert [json.loads(line)["method"] for line in requests.read_text().splitlines()] == ["close"]
