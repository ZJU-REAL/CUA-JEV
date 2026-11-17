"""Satisfied stages explain their zero-input result without granting success."""

import io

from PIL import Image

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace

FIELD = AgentTarget(name="Level", role="textbox", app="Editor")
COMMITTED = AgentCondition("value_equals", FIELD, "100")
SAVED = AgentCondition("exists", AgentTarget(name="Saved", role="label", app="Editor"))


class Desktop:
    def __init__(self):
        self.inputs = []
        self.saved = False
        image = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(image, "PNG")
        self.image = image.getvalue()

    def _get_obs(self):
        contents = (
            '<label name="Saved"/>' if self.saved else
            '<text name="Level" editable="true" value="100" focused="true" '
            'screencoord="(20, 20)" size="(100, 30)"/>'
            '<button name="Apply" enabled="true" screencoord="(200, 20)" size="(100, 30)"/>'
        )
        return {"screenshot": self.image, "accessibility_tree":
                '<desktop><application name="Editor"><frame name="Properties" active="true">'
                + contents + '</frame></application></desktop>'}

    def step(self, action, pause=0):
        self.inputs.append(action)
        assert action["action_type"] == "CLICK" and action["parameters"]["x"] > 190
        self.saved = True
        return self._get_obs(), None, False, {}


def edit():
    return AgentStep("edit", "fill", target=FIELD, value="100", postconditions=(COMMITTED,))


def commit(*, depends_on=()):
    return AgentStep("commit", "click", target=AgentTarget(name="Apply", role="button", app="Editor"),
                     depends_on=depends_on, preconditions=(COMMITTED,), postconditions=(SAVED,))


class Planner:
    def __init__(self, *, repeat=False, same_plan=False):
        self.repeat, self.same_plan = repeat, same_plan
        self.histories = []
        self.limits = []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.histories.append(list(history))
        self.limits.append(max_actions)
        if self.same_plan:
            return AgentPlan("Edit and apply", (edit(), commit(depends_on=("edit",))), reusable=True)
        if self.repeat or len(self.histories) == 1:
            return AgentPlan("Set the requested level", (edit(),), reusable=True)
        assert history[-1]["reason"] == "stage_already_satisfied"
        return AgentPlan("Apply the entered level", (commit(),), reusable=True,
                         completion_conditions=(SAVED,))


def run(planner, **config):
    desktop, trace = Desktop(), JsonlTrace()
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(allow_actions=True, **config)).run(
                             "Set level 100 and apply", success=lambda _snapshot: desktop.saved)
    return result, desktop, trace


def test_already_satisfied_edit_explains_next_stage_and_only_commit_sends_input():
    planner = Planner()
    result, desktop, trace = run(planner)
    assert result.success and result.planner_calls == 2 and result.steps == len(desktop.inputs) == 1
    assert result.condition_skips == 1 and result.plan_reuses == 0
    feedback = planner.histories[1][-1]
    assert feedback["reason"] == "stage_already_satisfied" and feedback["input_sent"] is False
    assert feedback["task_success"] is False and "100" not in str(feedback)
    assert feedback["recovery_required"] is True and planner.limits == [6, 1]
    assert feedback["satisfied_steps"][0]["operation"] == "fill"
    assert len([e for e in trace.events if e["kind"] == "agent_stage_satisfied"]) == 1


def test_repeated_satisfied_stages_cannot_extend_original_plan_budget_or_claim_success():
    planner = Planner(repeat=True)
    result, desktop, _trace = run(planner, max_plans=3)
    assert result.status == "max_plans" and not result.success
    assert result.planner_calls == result.condition_skips == 3
    assert result.steps == result.plan_completion_claims == 0 and desktop.inputs == []
    assert all(h[-1]["reason"] == "stage_already_satisfied" for h in planner.histories[1:])


def test_satisfied_edit_with_unmet_commit_in_same_plan_keeps_execution_and_reuse_semantics():
    planner = Planner(same_plan=True)
    result, desktop, trace = run(planner)
    assert result.success and result.planner_calls == 1 and len(desktop.inputs) == result.steps == 1
    assert result.condition_skips == 1
    assert not any(e["kind"] == "agent_stage_satisfied" for e in trace.events)
