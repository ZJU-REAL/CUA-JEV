"""Real desktop backend with fake observations; focus refusal never grants typing."""

import copy
import json
from dataclasses import replace

import pytest
from test_osworld_backend import Environment, semantic_control, semantic_ui
from test_semantic_tools_v57 import current
from test_semantic_tools_v57 import observed as observed
from test_stage_program_planning_preference_v61 import requested

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy, _text_focus_guard_feedback
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_flat_current_prompt import native_current_ref_guidance
from cua_jev.osworld_backend import OSWorldBackend, _TextFocusGuardRejected
from cua_jev.trace import JsonlTrace

SECRET = "PRIVATE_TYPED_VALUE"
TARGET = AgentTarget(name="Search", role="textbox", app="Demo", window="Preferences")


class FocusEnvironment(Environment):
    def __init__(self, fail_after="CLICK", change="focus"):
        super().__init__()
        self.fail_after, self.change = fail_after, change
        self.xml = self.tree(True)
        self.change_after = False

    def tree(self, focused):
        state = 's:editable="true"' + (' s:focused="true"' if focused else "")
        return semantic_ui(semantic_control("entry", "Search", states=state, value="old"))

    def step(self, action, pause):
        result = super().step(action, pause)
        if action["action_type"] == self.fail_after:
            if self.change == "owner":
                self.xml = self.tree(True).replace('name="Preferences"', 'name="Other owner"')
            elif self.change == "missing":
                self.xml = semantic_ui("")
            elif self.change == "box":
                self.xml = self.tree(True).replace('(80, 120)', '(240, 120)')
            else:
                self.xml = self.tree(False)
        return self._get_obs(), *result[1:]


class CountedBackend(OSWorldBackend):
    def __init__(self, env):
        super().__init__(env, pause=0)
        self.observe_calls = 0

    def observe(self):
        self.observe_calls += 1
        return super().observe()


def offered(backend):
    snapshot = backend.observe()
    step = AgentStep("fill", "fill", value=SECRET, target=TARGET,
                     postconditions=(AgentCondition("value_equals", TARGET, SECRET),))
    return backend.compile(step, snapshot, snapshot)[0]


def failure(stage="CLICK", change="focus"):
    env = FocusEnvironment(stage, change)
    backend = CountedBackend(env)
    candidate = offered(backend)
    backend.offered_candidate = candidate
    receipt = backend.execute(candidate, "observation", "decision")
    return env, backend, receipt


@pytest.mark.parametrize("stage,count,phase", [("CLICK", 1, "after_focus_click"),
                                             ("HOTKEY", 2, "after_select_all")])
@pytest.mark.parametrize("change", ["focus", "owner", "missing", "box"])
def test_focus_guard_retains_each_refusal_and_exact_input_observe_count(stage, count, phase, change):
    env, backend, receipt = failure(stage, change)
    assert receipt.success is False and not backend.verify(backend.offered_candidate, receipt).passed
    assert receipt.error == "OSWorld action failed (ValueError)"
    assert receipt.output["environment_steps"] == count
    assert len(env.calls) == count and backend.observe_calls == count + 2
    assert [item[0]["action_type"] for item in env.calls] == ["CLICK", "HOTKEY"][:count]
    assert all(item[0]["action_type"] != "TYPING" for item in env.calls)
    assert receipt.output["failure_code"] == "text_focus_guard_rejected"
    item = receipt.output["text_focus_guard"]
    assert item["stage"] == phase and item["typing_not_sent"] is True
    assert item["fresh_observe_returned"] is True and item["a11y_complete"] is True
    assert item["image_present"] is True
    assert item["target_matched"] is (change == "focus")
    if change == "focus":
        assert item["observed_focused_true"] is False  # Normalized observation, not real-world focus proof.
    else:
        assert item["observed_focused_true"] is None
    assert SECRET not in json.dumps(receipt.output)
    feedback = _text_focus_guard_feedback(receipt)
    assert "not proven" in feedback["feedback"] and "typing was not sent" in feedback["feedback"]


def test_preflight_zero_input_refusal_is_not_mislabeled_as_focus_click_or_typing_failure():
    env = FocusEnvironment()
    backend = CountedBackend(env)
    candidate = offered(backend)
    env.xml = env.tree(True).replace('name="Preferences"', 'name="Other owner"')
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and receipt.output["environment_steps"] == 0
    assert env.calls == [] and backend.observe_calls == 2
    assert "text_focus_guard" not in receipt.output and _text_focus_guard_feedback(receipt) is None


@pytest.mark.parametrize("typed", [False, True])
def test_generic_or_forged_private_exception_from_environment_has_no_local_focus_authority(typed):
    env = FocusEnvironment()
    backend = CountedBackend(env)
    candidate = offered(backend)
    error = (_TextFocusGuardRejected("after_focus_click", backend._last_observed_snapshot, None)
             if typed else ValueError("PRIVATE_HTTP_URL_AND_BODY"))
    error.diagnostic = {"PRIVATE": "PRIVATE_IDENTITY_AND_VALUE"}
    env.step = lambda *args, **kwargs: (_ for _ in ()).throw(error)
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and receipt.output["environment_steps"] == 1
    assert "text_focus_guard" not in receipt.output
    assert "PRIVATE" not in json.dumps(receipt.to_dict()) and _text_focus_guard_feedback(receipt) is None


def test_invalid_local_diagnostic_marker_is_omitted_without_changing_guard_failure():
    env = FocusEnvironment()
    backend = CountedBackend(env)
    original = backend._focused_target

    def corrupt(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except _TextFocusGuardRejected as error:
            error.diagnostic["PRIVATE_NAME"] = "PRIVATE_VALUE"
            raise

    backend._focused_target = corrupt
    receipt = backend.execute(offered(backend), "observation", "decision")
    assert not receipt.success and len(env.calls) == 1
    assert "text_focus_guard" not in receipt.output and "PRIVATE" not in json.dumps(receipt.to_dict())


@pytest.mark.parametrize("change", ["extra", "version", "stage", "unhashable_stage", "missing", "string_fact",
    "contradiction", "typing", "observe", "zero_inputs", "wrong_phase_count", "bool_inputs",
    "success", "capability"])
def test_invalid_or_inconsistent_receipt_data_cannot_produce_new_feedback(change):
    _env, _backend, receipt = failure()
    output = copy.deepcopy(receipt.output)
    item = output["text_focus_guard"]
    if change == "extra":
        item["name"] = "PRIVATE_VALUE"
    elif change == "version":
        item["version"] = "PRIVATE_VALUE"
    elif change == "stage":
        item["stage"] = "PRIVATE_VALUE"
    elif change == "unhashable_stage":
        item["stage"] = []
    elif change == "missing":
        del item["a11y_complete"]
    elif change == "string_fact":
        item["observed_focused_true"] = "false"
    elif change == "contradiction":
        item["observed_focused_true"] = True
    elif change == "typing":
        item["typing_not_sent"] = False
    elif change == "observe":
        item["fresh_observe_returned"] = False
    elif change in {"zero_inputs", "wrong_phase_count", "bool_inputs"}:
        output["environment_steps"] = {"zero_inputs": 0, "wrong_phase_count": 2, "bool_inputs": True}[change]
    changed = replace(receipt, output=output, success=change == "success",
                      capability="other.capability" if change == "capability" else receipt.capability)
    assert _text_focus_guard_feedback(changed) is None


def test_runner_keeps_original_failure_horizon_and_H1_system_instructions(observed):
    env = FocusEnvironment()
    env.xml = env.tree(False)
    backend = CountedBackend(env)

    class Planner:
        profile = "desktop"
        desktop_contract = "property_intent_v52"
        desktop_output_format = "semantic_tools_v64"

        def __init__(self):
            self.calls = []

        def plan(self, goal, snapshot, history, *, max_actions=6, **kwargs):
            self.calls.append((max_actions, copy.deepcopy(history)))
            return AgentPlan("Fill observed field", (AgentStep("fill", "fill", value=SECRET, target=TARGET,
                postconditions=(AgentCondition("value_equals", TARGET, SECRET),)),), reusable=True)

    planner = Planner()
    trace = JsonlTrace()
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, max_failures=3, max_plans=10,
                           plan_horizon=6)).run("Fill field")
    assert result.status == "stuck" and result.steps == result.planner_calls == len(planner.calls) == 3
    assert [call[0] for call in planner.calls] == [6, 1, 1]
    assert len(env.calls) == backend.environment_steps == 3 and result.plan_reuses == 0
    for _limit, history in planner.calls[1:]:
        feedback = history[-1]
        assert feedback["reason"] == "input_failed" and feedback["recovery_required"] is True
        assert "not proven" in feedback["feedback"]
        assert feedback["text_focus_guard"]["typing_not_sent"] is True
    raw = current(1, "declare_native_current_action")
    expected, original, _ = requested(observed, raw, "declare_native_current_action",
                                      fmt="semantic_tools_v58", limit=1)
    actual, request, _ = requested(observed, raw, "declare_native_current_action",
                                   fmt="semantic_tools_v64", limit=1)
    expected_body = json.loads(original.content)
    expected_body["messages"][0]["content"] = native_current_ref_guidance(
        expected_body["messages"][0]["content"])
    expected_content = json.dumps(expected_body, ensure_ascii=False, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8")
    assert actual == expected and request.content == expected_content
