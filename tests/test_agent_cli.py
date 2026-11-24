"""CLI/configuration checks with local test doubles; no browser or model requests."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from cua_jev import agent_browser, agent_cli, agent_planner
from cua_jev.agent_types import AgentControl, AgentSnapshot


def _args(*extra):
    parser = argparse.ArgumentParser()
    agent_cli.configure_parser(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args([
        "agent-browser", "--goal", "Save the local ticket", "--url", "https://local.test/start", *extra,
    ])


def _snapshot():
    return AgentSnapshot(
        "https://local.test/saved", "Saved ticket", "Ticket saved. Review complete.",
        (
            AgentControl("e0", "Subject", "input", value="A=B"),
            AgentControl("e1", "Follow up", "input", input_type="checkbox", checked=True),
        ),
    )


def test_acceptance_requires_every_explicit_condition_and_exact_unique_fields():
    args = _args(
        "--require-url-contains", "/saved", "--require-text", "Ticket saved.",
        "--require-text", "Review complete.", "--expect-field", "Subject=A=B",
        "--expect-checked", "Follow up",
    )
    check = agent_cli.acceptance(args)
    snapshot = _snapshot()
    assert check(snapshot)
    assert not check(replace(snapshot, url="https://local.test/start"))
    assert not check(replace(snapshot, text="Ticket saved."))
    assert not check(replace(snapshot, controls=(replace(snapshot.controls[0], value="A"),
                                                snapshot.controls[1])))
    assert not check(replace(snapshot, controls=(snapshot.controls[0],
                                                replace(snapshot.controls[1], checked=False))))
    assert not check(replace(snapshot, controls=(*snapshot.controls, snapshot.controls[0])))
    assert not check(replace(snapshot, controls=(*snapshot.controls, snapshot.controls[1])))


def test_no_explicit_acceptance_is_unverified_instead_of_defaulting_to_true():
    assert agent_cli.acceptance(_args()) is None


@pytest.mark.parametrize("field", ["Subject", "=A"])
def test_malformed_field_acceptance_is_rejected(field):
    with pytest.raises(ValueError, match="LABEL=VALUE"):
        agent_cli.acceptance(_args("--expect-field", field))


def test_empty_field_is_a_valid_explicit_acceptance_requirement():
    check = agent_cli.acceptance(_args("--expect-field", "Subject="))
    snapshot = _snapshot()
    assert not check(snapshot)
    assert check(replace(snapshot, controls=(replace(snapshot.controls[0], value=""),)))


def test_planner_configuration_requires_model_endpoint_and_upload_opt_in(monkeypatch):
    monkeypatch.delenv("CUA_JEV_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("CUA_JEV_VLM_MODEL", raising=False)

    def unexpected_client(*args, **kwargs):
        pytest.fail("A model client must not be constructed without required configuration")

    monkeypatch.setattr(agent_planner, "MultimodalPlanner", unexpected_client)
    for flags in ((), ("--model", "test-vlm"), ("--model-base-url", "https://model.test/v1")):
        with pytest.raises(ValueError, match="configure CUA_JEV_MODEL_BASE_URL"):
            agent_cli.make_planner(_args(*flags))
    with pytest.raises(ValueError, match="allow-screenshot-upload"):
        agent_cli.make_planner(_args("--model", "test-vlm", "--model-base-url", "https://model.test/v1"))


@pytest.mark.parametrize("primary_key", [None, "test-primary-key"])
def test_planner_uses_explicit_credentials_and_transport_opt_ins(monkeypatch, primary_key):
    monkeypatch.setenv("CUA_JEV_PLANNER_API_KEY", "test-fallback-key")
    monkeypatch.setenv("CUA_JEV_MODEL_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("CUA_JEV_VLM_MODEL", "test-vlm")
    if primary_key:
        monkeypatch.setenv("CUA_JEV_MODEL_API_KEY", primary_key)
    else:
        monkeypatch.delenv("CUA_JEV_MODEL_API_KEY", raising=False)
    captured = {}

    def create(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return captured

    monkeypatch.setattr(agent_planner, "MultimodalPlanner", create)
    agent_cli.make_planner(_args("--allow-screenshot-upload"))
    assert captured == {
        "args": ("https://model.test/v1", "test-vlm"),
        "kwargs": {"api_key": primary_key or "test-fallback-key", "allow_insecure_http": False,
                   "use_env_proxy": False},
    }
    agent_cli.make_planner(_args(
        "--allow-screenshot-upload", "--allow-insecure-model-http", "--use-env-proxy",
    ))
    assert captured["kwargs"]["allow_insecure_http"] is True
    assert captured["kwargs"]["use_env_proxy"] is True


@pytest.fixture
def fake_runtime(monkeypatch):
    """Record CLI wiring and resource lifecycle without any browser/model I/O."""
    events = []
    state = SimpleNamespace(events=events, success=False, failure=None)

    class Resource:
        def __init__(self, name):
            self.name = name

        def close(self):
            events.append(("close", self.name))

    class Page:
        url = "about:blank"

        def goto(self, url, **kwargs):
            events.append(("goto", url))
            if state.failure == "navigation":
                raise RuntimeError("test navigation failure")
            self.url = url

    page = Page()

    class Browser(Resource):
        def new_context(self, **kwargs):
            state.context_options = kwargs
            return SimpleNamespace(new_page=lambda: page)

    def launch(**kwargs):
        events.append(("launch", kwargs))
        if state.failure == "launch":
            raise RuntimeError("test launch failure")
        return Browser("browser")

    @contextmanager
    def playwright():
        yield SimpleNamespace(chromium=SimpleNamespace(launch=launch))

    api = ModuleType("playwright.sync_api")
    api.sync_playwright = playwright
    monkeypatch.setitem(sys.modules, "playwright.sync_api", api)

    class Backend(Resource):
        def __init__(self, page, *, allowed_origins, efficient_observations=False):
            super().__init__("backend")
            assert page.url == "about:blank"
            events.append(("backend", list(allowed_origins)))
            state.efficient_observations = efficient_observations
            self.observation_metrics = {"full": 0, "structured_only": 0, "wall_ms": 0.0}

    class Runner:
        def __init__(self, backend, planner, **kwargs):
            state.runner_options = kwargs

        def run(self, goal, *, success):
            state.goal = goal
            state.acceptance = success
            return SimpleNamespace(
                success=state.success,
                to_dict=lambda: {
                    "success": state.success, "status": "success" if state.success else "unverified",
                },
            )

    def policy(**kwargs):
        events.append(("policy", kwargs))
        if state.failure == "policy":
            raise RuntimeError("test policy configuration failure")
        return Resource("policy")

    monkeypatch.setattr(agent_cli, "make_planner", lambda args: Resource("planner"))
    monkeypatch.setattr(agent_cli, "JevPolicy", policy)
    monkeypatch.setattr(agent_cli, "AgentRunner", Runner)
    monkeypatch.setattr(agent_browser, "BrowserAgentBackend", Backend)
    return state


@pytest.mark.parametrize("mode,verified", [("vlm", False), ("hybrid", True)])
@pytest.mark.parametrize("efficient", [False, True])
def test_cli_writes_honest_result_scopes_before_navigation_and_closes_resources(
    tmp_path, fake_runtime, mode, verified, efficient,
):
    fake_runtime.success = verified
    output = tmp_path / "results" / "summary.json"
    args = _args(
        "--mode", mode, "--model", "test-vlm", "--output", str(output),
        "--trace", str(tmp_path / "trace.jsonl"), "--allow-origin", "https://other.test",
        "--headed", "--browser-channel", "msedge", "--max-steps", "7", "--max-plans", "8",
        "--plan-horizon", "4", "--timeout", "25", "--allow-actions",
        *(["--efficient-observations"] if efficient else []),
    )
    assert agent_cli.run(args) == (0 if verified else 1)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {"mode": mode, "vlm_model": "test-vlm", "success": verified,
                       "status": "success" if verified else "unverified",
                       "observation_metrics": {"full": 0, "structured_only": 0, "wall_ms": 0.0}}
    assert fake_runtime.acceptance is None
    assert fake_runtime.goal == "Save the local ticket"
    config = fake_runtime.runner_options["config"]
    assert config.defer_immediate_plan_refresh is efficient
    assert fake_runtime.efficient_observations is efficient
    assert (config.mode, config.max_steps, config.max_plans, config.plan_horizon,
            config.timeout_s, config.allow_actions) == (mode, 7, 8, 4, 25, True)
    assert fake_runtime.context_options == {
        "viewport": {"width": 1100, "height": 760}, "service_workers": "block",
    }
    events = fake_runtime.events
    assert ("launch", {"headless": False, "channel": "msedge"}) in events
    assert events.index(("backend", [args.url, "https://other.test"])) < events.index(("goto", args.url))
    assert ("close", "backend") in events and ("close", "browser") in events
    assert ("close", "planner") in events
    assert (("policy", {"retries": 2}) in events) is (mode == "hybrid")
    assert (("close", "policy") in events) is (mode == "hybrid")


@pytest.mark.parametrize("failure", ["policy", "launch", "navigation"])
def test_cli_closes_created_resources_when_setup_fails(tmp_path, fake_runtime, failure):
    fake_runtime.failure = failure
    args = _args("--trace", str(tmp_path / "trace.jsonl"), "--output", str(tmp_path / "summary.json"))
    with pytest.raises(RuntimeError, match="test"):
        agent_cli.run(args)
    closed = {name for event, name in fake_runtime.events if event == "close"}
    assert "planner" in closed
    if failure != "policy":
        assert "policy" in closed
    if failure == "navigation":
        assert {"backend", "browser"} <= closed
    assert not (tmp_path / "summary.json").exists()


def test_cli_has_no_implicit_action_authorization(tmp_path, fake_runtime):
    args = _args("--mode", "vlm", "--trace", str(tmp_path / "trace.jsonl"),
                 "--output", str(tmp_path / "summary.json"))
    agent_cli.run(args)
    assert fake_runtime.runner_options["config"].allow_actions is False
