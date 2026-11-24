"""A slow observation cannot authorize a new desktop input after its deadline."""

from __future__ import annotations

import io
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image
from cua_jev.trace import JsonlTrace


class Clock:
    now = 1000.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    value = Clock()
    monkeypatch.setattr("time.time", value.read)
    monkeypatch.setattr("time.perf_counter", value.read)
    monkeypatch.setattr("time.sleep", value.advance)
    return value


class Environment:
    def __init__(self, clock):
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buffer, format="PNG")
        self.snapshot = AgentSnapshot("osworld://desktop", "Editor", "Editor", (
            AgentControl("entry", "Document", "input", role="textbox", app="Editor", window="Editor",
                         editable=True, focused=True, box=(100, 100, 400, 400)),
        ), image=screenshot_image(buffer.getvalue()))
        self.clock = clock
        self.reads = 0
        self.read_delays = {}
        self.step_delay = 0
        self.actions = []

    def _get_obs(self):
        self.reads += 1
        self.clock.advance(self.read_delays.get(self.reads, 0))
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        self.clock.advance(self.step_delay)
        if action["action_type"] == "TYPING":
            self.snapshot = replace(self.snapshot, controls=(
                replace(self.snapshot.controls[0], value=action["parameters"]["text"]),
            ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def fill_step():
    target = AgentTarget(name="Document", role="textbox", app="Editor")
    return AgentStep("fill", "fill", value="requested text", target=target, recovery="observe",
                     postconditions=(AgentCondition("value_equals", target, "requested text"),))


def compile_action(backend, env, step=None):
    return backend.compile(step or fill_step(), env.snapshot, env.snapshot)[0]


@pytest.mark.parametrize("late_read,expected_actions", [
    (1, []),
    (2, ["CLICK"]),
    (3, ["CLICK", "HOTKEY"]),
])
def test_deadline_after_preflight_or_focus_read_blocks_next_input(clock, late_read, expected_actions):
    env, backend = Environment(clock), None
    backend = Backend(env, pause=0)
    env.read_delays[late_read] = 2
    action = compile_action(backend, env)
    receipt = backend.execute_with_deadline(action, "observation", "decision", deadline=clock.now + 1)
    assert not receipt.success and receipt.output["failure_code"] == "input_deadline_exceeded"
    assert [action["action_type"] for action in env.actions] == expected_actions
    assert receipt.output["environment_steps"] == backend.environment_steps == len(expected_actions)
    assert not backend.verify(action, receipt).passed
    assert not backend.verify_pending(action, receipt).passed
    assert backend._execution_deadline is None and not backend._issued
    assert not backend._pending and backend._action_observation is None


def test_late_step_response_does_not_send_remaining_visual_fill_inputs(clock):
    env, backend = Environment(clock), None
    backend = Backend(env, pause=0)
    env.step_delay = 2
    step = AgentStep("visual", "visual_fill", value="text", box=(100, 100, 400, 400), label="Document")
    action = compile_action(backend, env, step)
    receipt = backend.execute_with_deadline(action, "observation", "decision", deadline=clock.now + 1)
    assert not receipt.success and receipt.output["environment_steps"] == 1
    assert [action["action_type"] for action in env.actions] == ["CLICK"]
    assert backend._execution_deadline is None


def test_expired_execution_does_not_poison_later_ordinary_execution(clock):
    env = Environment(clock)
    backend = Backend(env, pause=0)
    first = compile_action(backend, env)
    failed = backend.execute_with_deadline(first, "first", "decision", deadline=clock.now)
    assert not failed.success and not env.actions and failed.output["environment_steps"] == 0
    clock.advance(3)
    second = compile_action(backend, env)
    receipt = backend.execute(second, "second", "decision")
    assert receipt.success and backend.verify(second, receipt).passed
    assert [action["action_type"] for action in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert backend._execution_deadline is None


def test_new_deadline_execution_can_follow_expired_context(clock):
    env = Environment(clock)
    backend = Backend(env, pause=0)
    first = compile_action(backend, env)
    assert not backend.execute_with_deadline(first, "first", "decision", deadline=clock.now).success
    second = compile_action(backend, env)
    receipt = backend.execute_with_deadline(second, "second", "decision", deadline=clock.now + 5)
    assert receipt.success and backend.verify(second, receipt).passed
    assert backend._execution_deadline is None


@pytest.mark.parametrize("deadline", [float("inf"), float("-inf"), float("nan"), None, "1001", True])
def test_invalid_deadline_is_rejected_without_input_or_context_change(clock, deadline):
    env = Environment(clock)
    backend = Backend(env, pause=0)
    action = compile_action(backend, env)
    with pytest.raises(ValueError, match="deadline must be finite"):
        backend.execute_with_deadline(action, "observation", "decision", deadline=deadline)
    assert not env.actions and env.reads == 0 and backend._execution_deadline is None


def test_unexpected_execute_error_still_restores_deadline_context(clock):
    env = Environment(clock)

    class CrashedBackend(Backend):
        def execute(self, *_args):
            raise RuntimeError("unexpected test failure")

    backend = CrashedBackend(env, pause=0)
    action = compile_action(backend, env)
    with pytest.raises(RuntimeError):
        backend.execute_with_deadline(action, "observation", "decision", deadline=clock.now + 5)
    assert backend._execution_deadline is None and not env.actions


@pytest.mark.parametrize("constraint", ["episode", "reusable_plan"])
def test_runner_passes_earliest_deadline_and_records_failed_receipt(clock, constraint):
    env = Environment(clock)
    # Runner initial / after-planning observations complete; execute preflight is late.
    env.read_delays[3] = 2
    backend, trace = Backend(env, pause=0), JsonlTrace()

    class Planner:
        def plan(self, *_args, **_kwargs):
            return AgentPlan("Fill document", (fill_step(),), reusable=True)

    config = AgentConfig(allow_actions=True, max_steps=1,
                         timeout_s=1 if constraint == "episode" else 30,
                         plan_ttl_s=1 if constraint == "reusable_plan" else 30)
    result = AgentRunner(backend, Planner(), policy=DirectPlanPolicy(), config=config, trace=trace).run(
        "Fill requested text",
    )
    receipts = [event["payload"] for event in trace.events if event["kind"] == "receipt"]
    assert result.steps == 1 and len(receipts) == 1 and not receipts[0]["success"]
    assert receipts[0]["output"]["failure_code"] == "input_deadline_exceeded"
    assert receipts[0]["output"]["environment_steps"] == 0 and not env.actions
    assert backend._execution_deadline is None and not backend._issued


def test_ordinary_nonreusable_plan_passes_episode_deadline_only(clock):
    env = Environment(clock)
    env.read_delays[3] = 2
    backend = Backend(env, pause=0)

    class Planner:
        def plan(self, *_args, **_kwargs):
            return AgentPlan("Click document", (AgentStep(
                "click", "visual_click", box=(100, 100, 400, 400), label="Document",
            ),))

    result = AgentRunner(backend, Planner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True, max_steps=1,
                                            timeout_s=30, plan_ttl_s=1)).run("Click document")
    assert result.steps == 1 and len(env.actions) == 1
    assert backend._execution_deadline is None
