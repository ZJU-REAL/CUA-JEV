"""Real local pipe protocol tests; no SSH, models or virtual machines."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

import pytest

from cua_jev.osworld_remote import RemoteOSWorldEnv, RemoteOSWorldError

WORKER = r"""
import base64,hashlib,json,sys,time
mode=sys.argv[1]
obs={"screenshot":base64.b64encode(b"\x89PNG\r\n\x1a\nfake-test-image").decode(),
     "accessibility_tree":"<desktop/>"}
def emit(value): print(json.dumps(value),flush=True)
if mode == "fatal":
 emit({"event":"fatal","error":"PRIVATE REMOTE DETAILS"});sys.exit(0)
if mode == "startup_eof": sys.exit(7)
if mode == "startup_json": print('PRIVATE NOT JSON',flush=True);sys.exit(0)
if mode == "startup_framing": print('PRIVATE PARTIAL',end='',flush=True);sys.exit(0)
if mode == "startup_object": emit(['PRIVATE']);sys.exit(0)
if mode == "startup_handshake": emit({'event':'PRIVATE'});sys.exit(0)
if mode == "startup_observation": obs['screenshot']='PRIVATE'
if mode == "slow_start": time.sleep(0.25)
if mode == "stuck_start": time.sleep(10)
ready={"event":"ready","protocol":1,"instruction":"Open display settings", "observation":obs}
if mode in ('settling', 'bad_settling'):
 ready['passive_settling']={
  'version':'public_structure_v1','enabled':False,'timeout_s':30.0,'max_samples':10,
  'min_stable_samples':3,'min_stable_span_s':4.0,'interval_s':1.0,
  'status':'disabled','reason':'disabled','samples_observed':0,'elapsed_ms':0.0,
  'stable_samples':0,'stable_span_ms':0.0,'surface_stability':'unproven',
  'application_ready_proven':False,'future_loading_excluded':False,
  'last_screenshot_sha256':hashlib.sha256(base64.b64decode(obs['screenshot'])).hexdigest(),
  'last_accessibility_sha256':hashlib.sha256(obs['accessibility_tree'].encode()).hexdigest()}
 if mode == 'bad_settling': ready['passive_settling']['reason']='PRIVATE unknown diagnostic'
emit(ready)
for line in sys.stdin:
 req=json.loads(line);method=req['method']
 if method == "close":
  if mode == "hang_close": time.sleep(10)
  emit({"id":req['id'],"ok":True,"result":None});break
 if mode == "eof": break
 if mode == "malformed":
  print('NOT JSON PRIVATE DATA',flush=True);continue
 if mode == "mismatch":
  emit({"id":req['id']+1,"ok":True,"result":obs});continue
 if mode == "remote_error" or (mode == "evaluation_error" and method == "evaluate"):
  emit({"id":req['id'],"ok":False,"error":"PRIVATE REMOTE DETAILS"});continue
 result=obs
 if method == "step":
  print('typed step reached fake worker',file=sys.stderr,flush=True)
  result={"observation":obs,"reward":0,"done":False,"info":{}}
 if method == "evaluate": result=0.75
 if mode == "bad_score" and method == "evaluate": result="0.75"
 if mode == "bad_image": result={**obs,"screenshot":"not base64"}
 emit({"id":req['id'],"ok":True,"result":result})
"""


def remote(tmp_path, mode="normal", **overrides):
    calls = []

    def launch(argv, **kwargs):
        calls.append((argv, kwargs.copy()))
        return subprocess.Popen([sys.executable, "-u", "-c", WORKER, mode], **kwargs)

    options = dict(timeout_s=1, startup_timeout_s=2, close_timeout_s=0.5)
    options.update(overrides)
    env = RemoteOSWorldEnv(
        "user@zju-18",
        ["/owned folder/python", "worker.py", "--task", "a'b;c$(x)"],
        log_path=tmp_path / "worker.log",
        popen_factory=launch,
        **options,
    )
    return env, calls


def test_remote_protocol_and_stderr_remain_local(tmp_path):
    env, calls = remote(tmp_path)
    assert env.instruction == "Open display settings"
    assert env.passive_settling is None
    assert env.initial_observation["screenshot"].startswith(b"\x89PNG")
    assert env._get_obs() == env.initial_observation
    assert env.controller.get_screenshot() == env.initial_observation["screenshot"]
    obs, reward, done, info = env.step({"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "p"]}})
    assert obs == env.initial_observation and reward == 0 and done is False and info == {}
    assert env.evaluate() == 0.75
    env.close()
    assert env.cleanup_status == "confirmed"
    assert "typed step reached" in (tmp_path / "worker.log").read_text()
    argv, kwargs = calls[0]
    assert argv[-2] == "user@zju-18"
    assert shlex.split(argv[-1]) == ["/owned folder/python", "worker.py", "--task", "a'b;c$(x)"]
    assert kwargs["shell"] is False


def test_optional_settling_metadata_survives_real_pipe_handshake_without_new_events(tmp_path):
    env, _ = remote(tmp_path, "settling")
    assert env.passive_settling["status"] == "disabled"
    assert env.passive_settling["application_ready_proven"] is False
    assert env._get_obs() == env.initial_observation
    env.close()
    assert env.cleanup_status == "confirmed"


def test_invalid_optional_settling_is_static_handshake_failure_and_unknown_cleanup(tmp_path):
    with pytest.raises(RemoteOSWorldError) as caught:
        remote(tmp_path, "bad_settling")
    assert caught.value.code == "invalid_handshake" and "PRIVATE" not in str(caught.value)
    assert caught.value.startup_diagnostics["cleanup_status"] == "unconfirmed_startup"


def test_evaluation_once_and_no_post_evaluation_observation(tmp_path):
    env, _ = remote(tmp_path)
    assert env.evaluate() == 0.75
    with pytest.raises(RemoteOSWorldError, match="already been evaluated"):
        env.evaluate()
    with pytest.raises(RemoteOSWorldError, match="already been evaluated"):
        env._get_obs()
    env.close()
    env.close()
    with pytest.raises(RemoteOSWorldError, match="closed"):
        env._get_obs()


@pytest.mark.parametrize("mode", ["evaluation_error", "bad_score"])
def test_evaluation_errors_are_not_zero_or_retried(tmp_path, mode):
    env, _ = remote(tmp_path, mode)
    with pytest.raises(RemoteOSWorldError) as caught:
        env.evaluate()
    assert "PRIVATE" not in str(caught.value)
    with pytest.raises(RemoteOSWorldError, match="already been evaluated"):
        env.evaluate()
    env.close()


@pytest.mark.parametrize("mode", ["malformed", "mismatch", "remote_error", "bad_image"])
def test_bad_responses_do_not_leak_worker_output(tmp_path, mode):
    env, _ = remote(tmp_path, mode)
    with pytest.raises(RemoteOSWorldError) as caught:
        env._get_obs()
    assert "PRIVATE" not in str(caught.value)
    env.close()


def test_eof_is_failure_and_cleanup_is_not_assumed(tmp_path):
    env, _ = remote(tmp_path, "eof")
    with pytest.raises(RemoteOSWorldError, match="closed its output"):
        env._get_obs()
    with pytest.raises(RemoteOSWorldError, match="cleanup was not confirmed"):
        env.close()
    assert env.cleanup_status == "unconfirmed_worker_exit"


def test_bounded_cleanup_terminates_only_own_process_and_reports_uncertainty(tmp_path):
    env, _ = remote(tmp_path, "hang_close", close_timeout_s=0.03)
    with pytest.raises(RemoteOSWorldError, match="cleanup was not confirmed"):
        env.close()
    assert env.cleanup_status == "unconfirmed_forced_termination"
    assert env._process.poll() is not None


@pytest.mark.parametrize("mode,pattern", [("fatal", "initialization failed"), ("slow_start", "timed out")])
def test_startup_failure_is_bounded_and_static(tmp_path, mode, pattern):
    # Only slow_start tests the 80 ms deadline. The fatal case must allow the
    # real Python child to launch before asserting its explicit error frame.
    with pytest.raises(RemoteOSWorldError, match=pattern) as caught:
        remote(tmp_path, mode, startup_timeout_s=0.08 if mode == "slow_start" else 2)
    assert caught.value.code == ("response_timeout" if mode == "slow_start" else "worker_fatal")
    assert "PRIVATE" not in str(caught.value)


@pytest.mark.parametrize("mode,code", [
    ("fatal", "worker_fatal"), ("slow_start", "response_timeout"),
    ("startup_eof", "unexpected_eof"), ("startup_json", "invalid_json"),
    ("startup_framing", "invalid_jsonl"), ("startup_object", "invalid_response"),
    ("startup_handshake", "invalid_handshake"), ("startup_observation", "invalid_observation"),
])
def test_startup_failure_has_safe_diagnostics_and_never_confirms_vm_cleanup(tmp_path, mode, code):
    with pytest.raises(RemoteOSWorldError) as caught:
        remote(tmp_path, mode, startup_timeout_s=0.08 if mode == "slow_start" else 2)
    diagnostic = caught.value.startup_diagnostics
    assert diagnostic["error_code"] == caught.value.code == code
    assert diagnostic["startup_elapsed_ms"] >= diagnostic["wait_elapsed_ms"] >= 0
    assert diagnostic["cleanup_elapsed_ms"] >= 0
    assert diagnostic["handshake_accepted_elapsed_ms"] is None
    assert diagnostic["ssh_exit_confirmed"] is True
    assert diagnostic["ssh_returncode"] == (7 if mode == "startup_eof" else 0)
    assert diagnostic["ssh_exit_success"] is (mode != "startup_eof")
    assert diagnostic["cleanup_status"] == "unconfirmed_startup"
    assert diagnostic["resource_cleanup_confirmed"] is False
    assert "PRIVATE" not in json.dumps(diagnostic)
    if mode == "slow_start":
        assert diagnostic["phase"] == "ready_wait"
        assert diagnostic["wait_elapsed_ms"] >= 70


def test_launch_exception_body_is_not_in_startup_diagnostics(tmp_path):
    def fail_launch(*_args, **_kwargs):
        raise OSError("PRIVATE command and authentication details")

    with pytest.raises(RemoteOSWorldError) as caught:
        RemoteOSWorldEnv("host", ["worker"], log_path=tmp_path / "worker.log", popen_factory=fail_launch)
    diagnostic = caught.value.startup_diagnostics
    assert diagnostic["error_code"] == "startup_process_error"
    assert diagnostic["phase"] == "process_launch" and diagnostic["wait_elapsed_ms"] is None
    assert diagnostic["ssh_returncode"] is None and diagnostic["ssh_exit_confirmed"] is False
    assert diagnostic["resource_cleanup_confirmed"] is False
    assert "PRIVATE" not in str(caught.value) + json.dumps(diagnostic)


def test_startup_timeout_preserves_forced_process_cleanup_uncertainty(tmp_path):
    with pytest.raises(RemoteOSWorldError) as caught:
        remote(tmp_path, "stuck_start", startup_timeout_s=0.08, close_timeout_s=0.03)
    diagnostic = caught.value.startup_diagnostics
    assert diagnostic["error_code"] == "response_timeout"
    assert diagnostic["ssh_exit_confirmed"] is True and diagnostic["ssh_exit_success"] is False
    assert diagnostic["cleanup_status"] == "unconfirmed_forced_termination"
    assert diagnostic["resource_cleanup_confirmed"] is False


def test_successful_handshake_records_received_and_accepted_times(tmp_path):
    env, _ = remote(tmp_path)
    diagnostic = env.startup_diagnostics
    assert diagnostic["phase"] == "ready" and diagnostic["error_code"] is None
    assert 0 <= diagnostic["wait_elapsed_ms"] <= diagnostic["handshake_received_elapsed_ms"]
    assert diagnostic["handshake_received_elapsed_ms"] <= diagnostic["handshake_accepted_elapsed_ms"]
    assert diagnostic["handshake_accepted_elapsed_ms"] == diagnostic["startup_elapsed_ms"]
    env.close()


@pytest.mark.parametrize("value", [True, False, 0, -1, 601, float("nan"), float("inf")])
def test_startup_timeout_rejects_invalid_values_before_process_launch(tmp_path, value):
    with pytest.raises(ValueError, match="timeouts"):
        RemoteOSWorldEnv("host", ["worker"], log_path=tmp_path / "worker.log", startup_timeout_s=value,
                         popen_factory=lambda *_a, **_k: pytest.fail("unexpected launch"))
    assert not (tmp_path / "worker.log").exists()


@pytest.mark.parametrize("host", ["-oProxyCommand=x", "x;touch", "host name", "host\n", "@x", ""])
def test_bad_host_rejected_before_launch(tmp_path, host):
    with pytest.raises(ValueError, match="SSH host"):
        RemoteOSWorldEnv(host, ["python"], log_path=tmp_path / "worker.log")
    assert not (tmp_path / "worker.log").exists()


def test_untyped_input_is_rejected_before_rpc(tmp_path):
    env, _ = remote(tmp_path)
    with pytest.raises(ValueError, match="typed"):
        env.step("arbitrary command")
    with pytest.raises(ValueError, match="pause"):
        env.step({"action_type": "CLICK", "parameters": {}}, pause=float("nan"))
    env.close()
