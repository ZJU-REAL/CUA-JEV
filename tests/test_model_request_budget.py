"""Request timeouts are explicit, shared across modes and bounded by agent time."""

import argparse
from dataclasses import replace

import httpx
import pytest
from test_current_ref_root_admission import snapshot as snapshot
from test_property_preparation import Backend, Environment

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner


def reply(request, observed):
    observed.append(dict(request.extensions["timeout"]))
    return httpx.Response(200, json={"choices": [{"message": {
        "content": '{"summary":"Offline complete","steps":[],"done":true}',
    }}]})


@pytest.mark.parametrize("value", [True, False, 0, -1, 301, float("inf"), float("nan"), "90"])
def test_request_timeout_rejects_nonfinite_or_unbounded_configuration(value):
    with pytest.raises(ValueError, match="request_timeout_s"):
        MultimodalPlanner("https://offline.invalid", "mock", request_timeout_s=value)


def test_explicit_request_wait_is_not_limited_by_the_base_clients_old_default(snapshot):
    observed = []
    transport = httpx.MockTransport(lambda request: reply(request, observed))
    with httpx.Client(timeout=40, transport=transport) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", client=client,
                                    request_timeout_s=90)
        planner.plan("Offline state", snapshot, [], max_actions=1)
        assert observed == [{"connect": 8, "read": 90, "write": 90, "pool": 90}]
        assert client.timeout.read == 40


def test_external_client_timeout_remains_unchanged_without_an_override_or_episode(snapshot):
    observed = []
    timeout = httpx.Timeout(73, connect=7, write=17, pool=19)
    transport = httpx.MockTransport(lambda request: reply(request, observed))
    with httpx.Client(timeout=timeout, transport=transport) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", client=client)
        planner.plan("Offline state", snapshot, [], max_actions=1)
        assert observed == [{"connect": 7, "read": 73, "write": 17, "pool": 19}]
        assert client.timeout == timeout


def test_remaining_episode_caps_network_phases_without_mutating_client(snapshot, monkeypatch):
    observed = []
    monkeypatch.setattr("cua_jev.agent_planner.time.perf_counter", lambda: 1000.0)
    transport = httpx.MockTransport(lambda request: reply(request, observed))
    with httpx.Client(timeout=40, transport=transport) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", client=client,
                                    request_timeout_s=90)
        planner.set_episode_deadline(1002.5)
        planner.plan("Offline state", snapshot, [], max_actions=1)
        assert observed == [{"connect": 2.5, "read": 2.5, "write": 2.5, "pool": 2.5}]
        planner.set_episode_deadline(None)
        planner.plan("Offline state", snapshot, [], max_actions=1)
        assert observed[-1]["read"] == 90 and client.timeout.read == 40


def test_expired_episode_submits_no_model_request(snapshot, monkeypatch):
    observed = []
    monkeypatch.setattr("cua_jev.agent_planner.time.perf_counter", lambda: 1000.0)
    with httpx.Client(transport=httpx.MockTransport(lambda request: reply(request, observed))) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", client=client,
                                    request_timeout_s=90)
        planner.set_episode_deadline(999)
        with pytest.raises(TimeoutError, match="before model request"):
            planner.plan("Offline state", snapshot, [], max_actions=1)
        assert observed == [] and planner.calls == 0


def test_runner_installs_and_clears_the_request_deadline(snapshot):
    observed = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: reply(request, observed))) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", client=client,
                                    request_timeout_s=90)
        env = Environment(snapshot)
        result = AgentRunner(Backend(env, pause=0), planner, config=AgentConfig(
            mode="vlm", allow_actions=True, max_plans=1, timeout_s=20,
        )).run("Offline complete")
        assert result.planner_calls == 1 and result.status == "unverified"
        assert 0 < observed[0]["read"] <= 20 and planner._episode_deadline is None
        assert not env.actions


def test_observation_exhausting_agent_time_cannot_start_another_planning_call(snapshot, monkeypatch):
    clock, requests = [1000.0], []
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])

    class SlowBackend(Backend):
        def observe(self):
            clock[0] += 21
            return replace(snapshot, text="Newly observed state")

    with httpx.Client(transport=httpx.MockTransport(lambda request: reply(request, requests))) as client:
        planner = MultimodalPlanner("https://offline.invalid", "mock", profile="desktop", client=client)
        env = Environment(snapshot)
        result = AgentRunner(SlowBackend(env, pause=0), planner, config=AgentConfig(
            mode="vlm", allow_actions=True, max_plans=1, timeout_s=20,
        )).run("Offline state")
        assert result.status == "timeout" and result.planner_calls == planner.calls == 0
        assert not requests and not env.actions


@pytest.mark.parametrize("value", ["0", "-1", "301", "nan", "inf", "true"])
def test_osworld_cli_rejects_invalid_request_wait(value):
    from test_osworld_driver import driver

    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-request-timeout", value])
