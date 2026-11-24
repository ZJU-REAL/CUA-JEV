"""The explicit diagnostic wrapper preserves real worker behavior using no VM."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("profile_test_worker", _SCRIPTS / "osworld_remote_worker.py")
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)
_SPEC = importlib.util.spec_from_file_location(
    "fill_profile_test",
    _SCRIPTS / "osworld_fill_observation_profiler.py",
)
profiler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiler)

CLICK = {"action_type": "CLICK", "parameters": {"x": 10, "y": 12, "button": "left", "num_clicks": 1}}
SELECT = {"action_type": "HOTKEY", "parameters": {"keys": ["ctrl", "a"]}}
TYPING = {"action_type": "TYPING", "parameters": {"text": "PRIVATE_TYPED_VALUE"}}


class Controller:
    def __init__(self):
        self.screenshot_calls = self.ax_calls = 0
        self.fail_ax = False

    def get_screenshot(self):
        self.screenshot_calls += 1
        output = io.BytesIO()
        Image.new("RGB", (100, 80), (self.screenshot_calls, 0, 0)).save(output, format="PNG")
        return output.getvalue()

    def get_accessibility_tree(self):
        self.ax_calls += 1
        if self.fail_ax:
            raise ValueError("PRIVATE_AX_FAILURE")
        return "<desktop>PRIVATE_AX_LITERAL</desktop>"

    def get_terminal_output(self):
        pytest.fail("terminal reads must remain disabled")


class Environment:
    def __init__(self):
        self.controller = Controller()
        self.require_a11y_tree = True
        self.inputs = []
        self.fail_step = False
        self.evaluations = 0
        self.closed = False

    def _get_obs(self):
        return {
            "screenshot": self.controller.get_screenshot(),
            "accessibility_tree": self.controller.get_accessibility_tree()
            if self.require_a11y_tree
            else None,
        }

    def step(self, action, *, pause):
        self.inputs.append((action, pause))
        if self.fail_step:
            raise TimeoutError("PRIVATE_UNCERTAIN_SUBMISSION")
        return self._get_obs(), 999, True, {}

    def evaluate(self):
        self.evaluations += 1
        return 1.0

    def close(self):
        self.closed = True


def requests(enabled):
    result = [{"id": 1, "method": "observe"}]
    for identifier, action in [(2, CLICK), (4, SELECT), (6, TYPING)]:
        request = {"id": identifier, "method": "step", "action": action, "pause": 0.25}
        if enabled and identifier != 6:
            request["observation_mode"] = "image_only_v1"
        result.append(request)
        if identifier != 6:
            result.append({"id": identifier + 1, "method": "observe"})
    return result


def records(output):
    return [json.loads(line) for line in (output / profiler.LOG_NAME).read_text().splitlines()]


@pytest.mark.parametrize("enabled", [False, True])
def test_full_worker_rpc_sequence_matches_production_and_shows_actual_controller_reads(tmp_path, enabled):
    plain, measured = tmp_path / "plain", tmp_path / "measured"
    plain.mkdir()
    measured.mkdir()
    before = worker.WorkerSession
    profiled = profiler.profiling_session(worker)
    all_results, environments = [], []
    for session_type, output in [(before, plain), (profiled, measured)]:
        env = Environment()
        session = session_type(env, env._get_obs(), output, image_only_fill_intermediates=enabled)
        all_results.append([session.handle(request) for request in requests(enabled)])
        assert session.handle({"id": 7, "method": "evaluate"}) == 1.0
        session.handle({"id": 8, "method": "close"})
        assert session.steps == 3 and env.evaluations == 1 and env.closed
        assert env.require_a11y_tree is True
        assert (
            not {"get_screenshot", "get_accessibility_tree", "get_terminal_output"}
            & vars(env.controller).keys()
        )
        environments.append(env)
    assert worker.WorkerSession is before
    assert all_results[0] == all_results[1]
    assert environments[0].inputs == environments[1].inputs == [(CLICK, 0.25), (SELECT, 0.25), (TYPING, 0.25)]
    assert not (plain / profiler.LOG_NAME).exists()
    for name in ["step-001.png", "step-002.png", "step-003.png", "final.png"]:
        assert (plain / name).read_bytes() == (measured / name).read_bytes()
    outcomes = [json.loads((output / "official-result.json").read_text()) for output in (plain, measured)]
    for outcome in outcomes:
        outcome.pop("evaluation_wall_ms")
    assert outcomes[0] == outcomes[1]
    logged = records(measured)
    assert logged[0]["event"] == "profile_header" and logged[0]["diagnostic_only"] is True
    assert logged[-1]["event"] == "profile_footer" and logged[-1]["worker_closed"] is True
    rows = [row for row in logged if row["event"] == "rpc_observation"]
    assert [row["request_id"] for row in rows] == [1, 2, 3, 4, 5, 6]
    assert [row["environment_steps_before"] for row in rows] == [0, 0, 1, 1, 2, 2]
    assert [row["environment_steps_after"] for row in rows] == [0, 1, 1, 2, 2, 3]
    assert [row["controller_reads"]["accessibility_tree"]["calls"] for row in rows] == (
        [1, 0, 1, 0, 1, 1] if enabled else [1] * 6
    )
    assert all(row["controller_reads"]["screenshot"]["calls"] == 1 for row in rows)
    assert all(row["controller_reads"]["terminal"]["calls"] == 0 for row in rows)
    assert all(row["a11y_flag_before"] is True and row["a11y_flag_after"] is True for row in rows)
    assert all(row["instrumentation"] == "active" and row["outcome"] == "ok" for row in rows)
    assert sum(row["capture_status"] == "image_only_returned" for row in rows) == (2 if enabled else 0)
    assert rows[-1]["capture_status"] == "full_returned"
    assert "PRIVATE" not in (measured / profiler.LOG_NAME).read_text()
    assert all(row["semantic_accessibility_completeness"] == "not_assessed" for row in rows)


def test_uncertain_input_error_is_not_retried_and_flag_and_controller_are_restored(tmp_path):
    env = Environment()
    session = profiler.profiling_session(worker)(
        env,
        env._get_obs(),
        tmp_path,
        image_only_fill_intermediates=True,
    )
    env.fail_step = True
    with pytest.raises(TimeoutError, match="PRIVATE_UNCERTAIN_SUBMISSION"):
        session.handle(requests(True)[1])
    session.close()
    assert env.inputs == [(CLICK, 0.25)] and env.require_a11y_tree is True
    row = records(tmp_path)[1]
    assert row["environment_steps_before"] == row["environment_steps_after"] == 0
    assert row["outcome"] == "error" and row["error_type"] == "TimeoutError"
    assert row["controller_reads"]["screenshot"]["calls"] == 0
    assert row["capture_status"] == "unavailable"
    assert (
        not {"get_screenshot", "get_accessibility_tree", "get_terminal_output"} & vars(env.controller).keys()
    )
    assert "PRIVATE" not in (tmp_path / profiler.LOG_NAME).read_text()


def test_failed_read_is_counted_without_replacing_original_error(tmp_path):
    env = Environment()
    session = profiler.profiling_session(worker)(env, env._get_obs(), tmp_path)
    env.controller.fail_ax = True
    with pytest.raises(ValueError, match="PRIVATE_AX_FAILURE"):
        session.handle({"id": 1, "method": "observe"})
    session.close()
    row = records(tmp_path)[1]
    assert row["outcome"] == "error" and row["error_type"] == "ValueError"
    assert row["controller_reads"]["accessibility_tree"]["calls"] == 1
    assert row["controller_reads"]["screenshot"]["calls"] == 1
    assert env.inputs == []


def test_caller_stopping_before_typing_does_not_trigger_any_input_from_profiler(tmp_path):
    env = Environment()
    session = profiler.profiling_session(worker)(
        env,
        env._get_obs(),
        tmp_path,
        image_only_fill_intermediates=True,
    )
    # The caller can stop after its deadline; the diagnostic wrapper never advances a macro itself.
    for request in requests(True)[:4]:
        session.handle(request)
    session.close()
    assert env.inputs == [(CLICK, 0.25), (SELECT, 0.25)]
    assert [row["action_type"] for row in records(tmp_path)[1:] if row.get("method") == "step"] == [
        "CLICK",
        "HOTKEY",
    ]


def test_record_limit_is_explicit_and_does_not_block_later_inputs(tmp_path):
    env = Environment()
    session = profiler.profiling_session(worker, max_records=2)(env, env._get_obs(), tmp_path)
    for request in requests(False):
        session.handle(request)
    session.close()
    logged = records(tmp_path)
    assert [row["event"] for row in logged] == [
        "profile_header",
        "rpc_observation",
        "rpc_observation",
        "profile_truncated",
        "profile_footer",
    ]
    assert session.steps == 3 and len(env.inputs) == 3 and env.controller.ax_calls == 7


def test_diagnostic_write_failure_does_not_swallow_input_response(tmp_path, capsys):
    class BrokenLog:
        def write(self, _value):
            raise OSError("PRIVATE_DISK_FAILURE")

        def close(self):
            pass

    env = Environment()
    session = profiler.profiling_session(worker)(env, env._get_obs(), tmp_path)
    session._profile_stream.close()
    session._profile_stream = BrokenLog()
    for request in requests(False)[1:2]:
        result = session.handle(request)
        assert result["observation"]["accessibility_tree"] == "<desktop>PRIVATE_AX_LITERAL</desktop>"
    result = session.handle({"id": 3, "method": "observe"})
    assert result["accessibility_tree"] == "<desktop>PRIVATE_AX_LITERAL</desktop>"
    session.close()
    assert env.inputs == [(CLICK, 0.25)] and session.steps == 1
    stderr = capsys.readouterr().err
    assert "PRIVATE" not in stderr
    assert len(stderr.splitlines()) == 1 and json.loads(stderr)["code"] == "log_write_failed"


def test_instrumentation_setup_failure_executes_original_request_once(tmp_path, capsys):
    env = Environment()
    env.controller.get_terminal_output = None
    session = profiler.profiling_session(worker)(env, env._get_obs(), tmp_path)
    result = session.handle(requests(False)[1])
    session.close()
    assert result["observation"]["accessibility_tree"] == "<desktop>PRIVATE_AX_LITERAL</desktop>"
    assert env.inputs == [(CLICK, 0.25)] and session.steps == 1
    assert "get_screenshot" not in vars(env.controller) and "get_accessibility_tree" not in vars(
        env.controller
    )
    assert records(tmp_path)[1]["instrumentation"] == "unavailable"
    assert json.loads(capsys.readouterr().err)["code"] == "instrumentation_unavailable"


@pytest.mark.parametrize("raises", [False, True])
def test_explicit_main_restores_original_worker_class(monkeypatch, raises):
    original = type("OriginalSession", (), {})
    fake = SimpleNamespace(__file__=str(_SCRIPTS / "osworld_remote_worker.py"), WorkerSession=original)

    def main(argv):
        assert argv == ["--owned-options"] and fake.WorkerSession is not original
        if raises:
            raise RuntimeError("original main failure")
        return 7

    fake.main = main
    monkeypatch.setitem(sys.modules, "osworld_remote_worker", fake)
    monkeypatch.setattr(sys, "path", list(sys.path))
    if raises:
        with pytest.raises(RuntimeError, match="original main failure"):
            profiler.main(["--owned-options"])
    else:
        assert profiler.main(["--owned-options"]) == 7
    assert fake.WorkerSession is original


@pytest.mark.parametrize("limit", [0, -1, 513, True, 2.0])
def test_invalid_record_limits_rejected_before_any_session(limit):
    with pytest.raises(ValueError):
        profiler.profiling_session(worker, max_records=limit)
