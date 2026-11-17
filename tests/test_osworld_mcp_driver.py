import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def driver():
    spec = importlib.util.spec_from_file_location(
        "mcp_pilot_test", Path(__file__).parents[1] / "scripts/run_osworld_mcp_pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_evaluator_runs_once_after_agent_and_before_cleanup(driver, monkeypatch, tmp_path):
    events = []

    class Env:
        instruction = "public goal"
        initial_observation = {"screenshot": b"test PNG"}
        cleanup_status = "not_requested"

        def evaluate(self):
            events.append("final_evaluate")
            return 1.0

        def close(self):
            events.append("environment_close")
            self.cleanup_status = "confirmed"

    class Resource:
        def close(self):
            pass

    class Backend(Resource):
        def __init__(self, *args):
            self.catalog = SimpleNamespace(tools={"test": {"name": "test"}})
            self.metrics = {"mcp_calls": 1}

        @property
        def observation_phase_metrics(self):
            return {}

    class Planner(Resource):
        def __init__(self, *args, **kwargs):
            self.client = httpx.Client()

        def close(self):
            self.client.close()

    class Runner:
        def __init__(self, backend, planner, *, policy, trace, config):
            self.trace = trace
            assert not hasattr(backend, "evaluate")

        def run(self, goal):
            assert goal == "public goal"
            events.append("agent_finished")
            self.trace.append("test", {})
            return SimpleNamespace(to_dict=lambda: {"status": "unverified", "jev_calls": 0})

    args = SimpleNamespace(worker_python=Path("/owned/python"), expected_user="test",
                           osworld_root=tmp_path, vm_path=tmp_path / "base.qcow2",
                           mcp_bundle=tmp_path / "bundle", mcp_installer=tmp_path / "installer",
                           max_steps=3, timeout=60)
    for key in ("CUA_JEV_MODEL_BASE_URL", "CUA_JEV_VLM_MODEL", "CUA_JEV_MODEL_API_KEY"):
        monkeypatch.setenv(key, "test")
    monkeypatch.setattr(driver, "RemoteOSWorldEnv", lambda **kw: Env())
    monkeypatch.setattr(driver, "install_guest_service", lambda *args: "http://127.0.0.1:1234/mcp")
    monkeypatch.setattr(driver, "PersistentMcpSession", lambda **kw: SimpleNamespace(metrics={}))
    monkeypatch.setattr(driver, "OSWorldBackend", lambda env: object())
    monkeypatch.setattr(driver, "JointOSWorldBackend", Backend)
    monkeypatch.setattr(driver, "JointOSWorldPlanner", Planner)
    monkeypatch.setattr(driver, "AgentRunner", Runner)
    row = driver.run_one(args, "os/test", "vlm", tmp_path / "episode")
    assert events == ["agent_finished", "final_evaluate", "environment_close"]
    assert row["official_score"] == 1.0 and row["cleanup_status"] == "confirmed"
    assert row["evaluation_status"] == "valid"
    assert "error_type" not in row
    assert row["observation_phases"] == {}
    assert row["mcp_backend_metrics"] == {"mcp_calls": 1}
    assert row["total_wall_ms"] >= row["agent_wall_ms"]


def test_unreturned_environment_cannot_claim_confirmed_cleanup(driver, monkeypatch, tmp_path):
    args = SimpleNamespace(worker_python=Path("/owned/python"), expected_user="test",
                           osworld_root=tmp_path, vm_path=tmp_path / "base.qcow2")

    def failed(**kwargs):
        raise RuntimeError("worker startup did not return an owned handle")

    monkeypatch.setattr(driver, "RemoteOSWorldEnv", failed)
    row = driver.run_one(args, "os/test", "vlm", tmp_path / "episode")
    assert row["cleanup_status"] == "unconfirmed"
    assert row["official_score"] is None
    assert row["error_type"] == "RuntimeError"


def test_owned_virtualenv_launcher_keeps_venv_path_after_ownership_check(driver, tmp_path):
    real = tmp_path / "base-python"
    real.touch()
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(real)
    assert driver.owned_path(launcher, tmp_path, preserve_launcher=True) == launcher
    assert driver.owned_path(launcher, tmp_path) == real
    outside = tmp_path / "bad-python"
    outside.symlink_to("/usr/bin/python3")
    with pytest.raises(ValueError):
        driver.owned_path(outside, tmp_path, preserve_launcher=True)
