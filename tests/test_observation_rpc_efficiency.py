"""Exercise negotiated screenshot transport and budget/evaluator isolation."""

import base64
import json
import time
from types import SimpleNamespace

import pytest
import test_osworld_remote as remote_tests
from test_osworld_remote_worker import observation, worker

from cua_jev.osworld_remote import RemoteOSWorldError


def test_negotiated_screenshot_avoids_full_observe_over_real_pipe(tmp_path, monkeypatch):
    script = remote_tests.WORKER.replace(
        "emit(ready)",
        "ready['capabilities']={'screenshot_only':'screenshot_v1'}\nemit(ready)",
    ).replace(
        "result=obs\n",
        "result=obs\n if method == 'observe': raise RuntimeError('full read forbidden')\n"
        " if method == 'screenshot': result=obs['screenshot']\n",
    )
    monkeypatch.setattr(remote_tests, "WORKER", script)
    env, _ = remote_tests.remote(tmp_path)
    try:
        assert env.supports_screenshot_only
        assert env.controller.get_screenshot() == env.initial_observation["screenshot"]
        assert env.evaluate() == 0.75
        with pytest.raises(RemoteOSWorldError, match="already been evaluated"):
            env.controller.get_screenshot()
    finally:
        env.close()


@pytest.mark.parametrize("capabilities", [
    {"screenshot_only": "unknown"}, {"screenshot_only": "screenshot_v1", "unknown": True}, {},
])
def test_unknown_screenshot_capability_fails_negotiation(tmp_path, monkeypatch, capabilities):
    script = remote_tests.WORKER.replace(
        "emit(ready)", f"ready['capabilities']={capabilities!r}\nemit(ready)",
    )
    monkeypatch.setattr(remote_tests, "WORKER", script)
    with pytest.raises(RemoteOSWorldError) as caught:
        remote_tests.remote(tmp_path)
    assert caught.value.code == "invalid_handshake"


def test_worker_screenshot_does_not_attach_stale_tree_or_change_complete_snapshot(tmp_path):
    before = observation()
    env = SimpleNamespace(
        controller=SimpleNamespace(get_screenshot=lambda: before["screenshot"]),
        _get_obs=lambda: pytest.fail("screenshot RPC acquired full AX"), close=lambda: None,
    )
    session = worker.WorkerSession(env, before, tmp_path)
    timing = worker.ObservationCaptureTiming(env)
    session.capture_timing = timing
    try:
        value = session.handle({"id": 1, "method": "screenshot"})
        assert base64.b64decode(value) == before["screenshot"]
        assert session.observation is before
        assert timing.report["groups"] == {}
        assert timing.report["standalone_screenshots"]["calls"] == 1
        assert timing.report["standalone_screenshots"]["wall_ms"] >= 0
        session.evaluated = True
        with pytest.raises(worker.ProtocolError, match="terminal"):
            session.handle({"id": 2, "method": "screenshot"})
    finally:
        session.close()
        timing.restore()


def test_ax_timeout_returns_missing_tree_and_restores_before_evaluation(tmp_path):
    calls = []

    def read_tree():
        calls.append("ax")
        if calls.count("ax") == 1:
            time.sleep(2)
        return "<desktop/>"

    class Env:
        controller = SimpleNamespace(get_accessibility_tree=read_tree)

        def _get_obs(self):
            return {**observation(), "accessibility_tree": self.controller.get_accessibility_tree()}

        def evaluate(self):
            assert self.controller.get_accessibility_tree is read_tree
            assert self._get_obs()["accessibility_tree"] == "<desktop/>"
            return 1.0

        def close(self):
            assert self.controller.get_accessibility_tree is read_tree

    env = Env()
    session = worker.WorkerSession(env, observation(), tmp_path, ax_timeout_s=0.03)
    timing = worker.ObservationCaptureTiming(env)
    session.capture_timing = timing
    try:
        for request_id in (1, 2):
            assert session.handle({"id": request_id, "method": "observe"})["accessibility_tree"] is None
        assert calls == ["ax"]
        assert session.handle({"id": 3, "method": "evaluate"}) == 1.0
        report = json.loads((tmp_path / "observation-budget.json").read_text())
        assert report["timeouts"] == 1 and report["skipped"] == 1 and report["restored"]
    finally:
        session.close()
        timing.restore()
