"""Planning validation cannot consume or erase the execution recovery budget."""

from __future__ import annotations

from dataclasses import replace

import pytest

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import matching_controls
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Risk, Verification
from cua_jev.trace import JsonlTrace


class Backend:
    def __init__(self, outcomes=(), *, available=True):
        self.outcomes = outcomes
        self.available = available
        self.inputs = []
        self.pending_reads = 0
        self.completed = []

    def observe(self):
        controls = (AgentControl("edit", "Edit", "button", role="button", app="Editor"),)
        controls += tuple(AgentControl(label, label, "text", role="label") for label in self.completed)
        return AgentSnapshot("osworld://desktop", "Editor", "Editor", controls)

    def compile(self, step, planned, current):
        if not self.available:
            return ()
        return (ActionCandidate(
            step.id, Channel.GUI, "agent.desktop.click", "Click Edit",
            arguments={"route": "desktop", "result_label": step.postconditions[0].target.name},
            verifier="agent.effect", intent=step.id, risk=Risk.LOCAL_WRITE,
        ),)

    def execute(self, candidate, observation_id, decision_id):
        index = len(self.inputs)
        self.inputs.append(candidate.id)
        passed = self.outcomes[index] if index < len(self.outcomes) else False
        if passed:
            self.completed.append(candidate.arguments["result_label"])
        return ActionReceipt(
            observation_id, decision_id, candidate.id, candidate.channel, candidate.capability, True, 0, 0,
            output={"environment_steps": 1, "effect_observed": passed},
        )

    def verify(self, candidate, receipt):
        return Verification(receipt.output["effect_observed"], "agent.effect")

    def verify_pending(self, candidate, receipt):
        self.pending_reads += 1
        return Verification(False, "agent.effect")


class Planner:
    def __init__(self, script):
        self.script = script
        self.calls = []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.calls.append({"limit": max_actions, "history": list(history)})
        index = len(self.calls) - 1
        item = self.script[min(index, len(self.script) - 1)]
        if item == "invalid":
            raise ValueError("private invalid model output")
        if item == "runtime_error":
            raise RuntimeError("private model transport detail")
        if item == "unconfirmed_completion":
            return AgentPlan("Completion must still be observed", (), done=True,
                             completion_conditions=(AgentCondition("exists", AgentTarget(name="Missing")),))
        navigation = item == "navigation"
        target = AgentTarget(app="Editor") if navigation else AgentTarget(name="Edit", app="Editor")
        return AgentPlan("Open the next interface", (AgentStep(
            f"step-{index}", "click", target=target, label="Reach the next interface",
            navigation=navigation, recovery="observe",
            postconditions=(AgentCondition("exists", AgentTarget(name=f"Ready-{index}")),),
        ),), reusable=True)


@pytest.fixture(autouse=True)
def no_recovery_sleep(monkeypatch):
    monkeypatch.setattr("cua_jev.agent_runner.time.sleep", lambda _seconds: None)


def run(backend, planner, *, success=None, **config):
    trace = JsonlTrace()
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, **config),
    ).run("Complete the public task", success=success)
    return result, trace


def test_two_invalid_plans_do_not_remove_first_unconfirmed_input_recovery():
    backend = Backend((False, True))
    planner = Planner(("invalid", "invalid", "valid", "valid"))
    result, trace = run(backend, planner, success=lambda _snapshot: bool(backend.completed))
    assert result.success and result.steps == 2 and result.planner_calls == 4
    assert [call["limit"] for call in planner.calls] == [6, 6, 6, 1]
    assert backend.inputs == ["step-2", "step-3"]
    assert backend.pending_reads == result.recovery_observations == 2
    feedback = planner.calls[3]["history"][-1]
    assert feedback["reason"] == "postcondition_unconfirmed"
    assert feedback["recovery_required"] and not feedback["verified"]
    assert feedback["feedback"].startswith("Input was sent")
    assert "private invalid model output" not in str(trace.events)


@pytest.mark.parametrize("failure", ["invalid", "runtime_error"])
def test_consecutive_planning_failures_still_stop_before_any_input(failure):
    backend, planner = Backend(), Planner((failure,))
    result, _trace = run(backend, planner)
    assert result.status == "planner_error" and result.reason == "consecutive planning failures"
    assert result.planner_calls == 3 and result.steps == 0 and backend.inputs == []


def test_runner_rejection_does_not_reset_planning_streak_before_validation():
    backend, planner = Backend(), Planner(("navigation",))
    result, _trace = run(backend, planner, mode="vlm")
    assert result.status == "planner_error" and result.planner_calls == 3
    assert [call["limit"] for call in planner.calls] == [1, 1, 1]
    assert result.steps == 0 and backend.inputs == []


def test_valid_plans_cannot_reset_repeated_unconfirmed_action_streak():
    backend, planner = Backend(), Planner(("valid",))
    result, _trace = run(backend, planner)
    assert result.status == "stuck" and result.reason == "consecutive action failures"
    assert result.planner_calls == result.steps == len(backend.inputs) == 3
    assert backend.pending_reads == result.recovery_observations == 6


def test_invalid_plans_between_inputs_neither_consume_nor_reset_action_streak():
    backend = Backend()
    planner = Planner(("invalid", "invalid", "valid", "invalid", "invalid", "valid", "valid"))
    result, _trace = run(backend, planner)
    assert result.status == "stuck" and result.reason == "consecutive action failures"
    assert result.planner_calls == 7 and result.steps == 3
    assert backend.inputs == ["step-2", "step-5", "step-6"]


def test_verified_action_clears_only_execution_failure_streak():
    backend = Backend((False, False, True, False, False, False))
    planner = Planner(("valid",))
    result, _trace = run(backend, planner)
    assert result.status == "stuck" and result.reason == "consecutive action failures"
    assert result.planner_calls == result.steps == 6
    assert backend.completed == ["Ready-2"] and backend.pending_reads == 10


@pytest.mark.parametrize("kind,expected_reason", [
    ("missing_target", "no freshly grounded action"),
    ("unconfirmed_completion", "goal completion conditions remain unconfirmed"),
])
def test_non_input_no_progress_keeps_bounded_action_budget(kind, expected_reason):
    backend = Backend(available=False)
    planner = Planner(("invalid", "invalid", kind))
    result, _trace = run(backend, planner)
    assert result.status == "stuck" and result.reason == expected_reason
    assert result.planner_calls == 5 and result.steps == 0 and backend.inputs == []


@pytest.mark.parametrize("config,status,plans,steps", [
    ({"max_plans": 4}, "max_plans", 4, 2),
    ({"max_steps": 1}, "max_steps", 2, 1),
])
def test_total_budgets_remain_bounds_with_independent_failure_streaks(config, status, plans, steps):
    backend = Backend()
    planner = Planner(("invalid", "valid", "invalid", "valid"))
    result, _trace = run(backend, planner, **config)
    assert result.status == status and result.planner_calls == plans and result.steps == steps


class RebindingBackend(Backend):
    def __init__(self, *, late=True, duplicate_routes=False, capability="agent.osworld.click",
                 difference="window"):
        super().__init__((True,))
        self.observations = 0
        self.late, self.duplicate_routes, self.capability = late, duplicate_routes, capability
        first = AgentControl("first", "Edit", "button", role="button", app="Editor",
                             window="First window", box=(10, 10, 30, 30))
        changes = {"ref": "second", "box": (60, 10, 80, 30), "window": "Second window"}
        second = replace(first, **(changes if difference == "window" else {difference: changes[difference]}))
        self.controls = (first, second)

    def observe(self):
        self.observations += 1
        controls = self.controls[:1] if self.late and self.observations == 1 else self.controls
        return AgentSnapshot("osworld://desktop", "Editor", "Editor", controls)

    def compile(self, step, planned, current):
        matches = matching_controls(current, step.target)
        routes = [(control, "desktop") for control in matches]
        if self.duplicate_routes:
            routes = [(matches[0], "primary"), (matches[0], "fallback")]
        template = super().compile(step, planned, current)[0]
        return tuple(replace(
            template, id=f"{step.id}:{index}:{control.ref}", capability=self.capability,
            arguments={**template.arguments, "route": route, "bound_control": control.to_dict()},
        ) for index, (control, route) in enumerate(routes))


def test_vlm_replans_when_another_physical_target_appears_after_planning():
    class NarrowPlanner(Planner):
        def __init__(self):
            super().__init__(("valid",))
            self.seen_controls = []

        def plan(self, goal, snapshot, history, *, max_actions):
            self.seen_controls.append(tuple(control.ref for control in snapshot.controls))
            plan = super().plan(goal, snapshot, history, max_actions=max_actions)
            if len(self.calls) > 1:
                step = replace(plan.steps[0], target=AgentTarget(name="Edit", window="Second window"))
                plan = replace(plan, steps=(step,))
            return plan

    backend, planner = RebindingBackend(), NarrowPlanner()
    result, trace = run(backend, planner, mode="vlm", success=lambda _: bool(backend.completed))
    assert result.success and result.planner_calls == 2 and result.steps == 1
    assert planner.seen_controls == [("first",), ("first", "second")]
    assert backend.inputs == ["step-1:0:second"]
    assert result.candidate_counts == (1,) and result.recovery_observations == 0
    feedback = planner.calls[1]["history"][-1]
    assert feedback["reason"] == "ambiguous_target"
    assert feedback["ready_steps"][0]["candidate_count"] == 2
    assert feedback["ready_steps"][0]["match_count"] == 2
    assert result.replan_reasons["ambiguous_target"] == 1
    assert len([event for event in trace.events if event["kind"] == "decision"]) == 1


@pytest.mark.parametrize("difference", ["window", "ref", "box"])
def test_vlm_ambiguity_of_distinct_physical_bindings_exhausts_no_input_budget(difference):
    backend, planner = RebindingBackend(late=False, difference=difference), Planner(("valid",))
    result, trace = run(backend, planner, mode="vlm")
    assert result.status == "stuck" and result.reason == "no freshly grounded action"
    assert result.planner_calls == 3 and result.steps == 0 and backend.inputs == []
    assert result.recovery_observations == 0
    assert all(event["kind"] != "decision" for event in trace.events)
    assert len([event for event in trace.events if event["kind"] == "agent_recovery"
                and event["payload"].get("reason") == "ambiguous_target"]) == 3


def test_hybrid_keeps_multiple_real_desktop_choices():
    backend, planner = RebindingBackend(late=False), Planner(("valid",))
    result, _trace = run(backend, planner, success=lambda _: bool(backend.completed))
    assert result.success and result.steps == 1 and result.planner_calls == 1
    assert result.candidate_counts == (2,) and result.jev_calls == 1
    assert backend.inputs == ["step-0:0:first"]


@pytest.mark.parametrize("capability", ["agent.browser.click", "agent.osworld.click"])
def test_vlm_keeps_multiple_routes_to_the_same_physical_binding(capability):
    backend = RebindingBackend(late=False, duplicate_routes=True, capability=capability)
    result, _trace = run(backend, Planner(("valid",)), mode="vlm", success=lambda _: bool(backend.completed))
    assert result.success and result.steps == 1 and result.planner_calls == 1
    assert result.candidate_counts == (2,) and result.jev_calls == 0
    assert backend.inputs == ["step-0:0:first"]


class ContextBackend(Backend):
    """A contract stub; real receipt/structure checks have backend tests."""

    def __init__(self, outcomes=(), *, context_current=True, input_count=1, receipt_success=True):
        super().__init__(outcomes)
        self.reads = 0
        self.takes = []
        self.confirms = []
        self.context_current = context_current
        self.input_count = input_count
        self.receipt_success = receipt_success

    def observe(self):
        self.reads += 1
        return super().observe()

    def execute(self, *args):
        receipt = super().execute(*args)
        return replace(receipt, success=self.receipt_success,
                       output={**receipt.output, "environment_steps": self.input_count})

    def take_unverified_recovery_context(self, candidate, receipt):
        self.takes.append((candidate.id, self.reads))
        self.ticket = {"context_fingerprint": "a" * 64, "new_control_count": 1,
                       "reason": "new_actionable_context_after_unconfirmed_input"}
        return self.ticket

    def confirm_unverified_recovery_context(self, context, snapshot):
        self.confirms.append(self.reads)
        assert context is self.ticket and snapshot.url == "osworld://desktop"
        return self.context_current


@pytest.mark.parametrize("mode", ["hybrid", "vlm"])
def test_context_recovery_grants_one_fresh_single_action_without_verifying_prior_failures(mode):
    backend = ContextBackend((False, False, False, True))
    planner = Planner(("valid",))
    result, trace = run(backend, planner, mode=mode, success=lambda _: bool(backend.completed))
    assert result.success and result.steps == result.planner_calls == 4
    assert result.context_recoveries == 1 and result.post_snapshot_reuses == 0
    assert len(backend.takes) == len(backend.confirms) == 1
    assert backend.confirms[0] == backend.takes[0][1] + 1
    assert planner.calls[-1]["limit"] == 1
    feedback = planner.calls[-1]["history"][-1]
    assert feedback["action_failures"] == 3 and feedback["episode_recovery_limit"] == 1
    assert feedback["recovery_required"] and not feedback["verified"] and not feedback["task_success"]
    assert [e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step"] == [
        False, False, False, True,
    ]
    assert result.plan_reuses == 0 and result.to_dict()["context_recoveries"] == 1


def test_context_recovery_keeps_failure_streak_and_cannot_repeat_while_effects_stay_unknown():
    backend = ContextBackend()
    result, trace = run(backend, Planner(("valid",)))
    assert result.status == "stuck" and result.steps == result.planner_calls == 4
    assert result.context_recoveries == len(backend.takes) == 1
    assert result.semantic_steps == result.plan_reuses == 0
    assert all(not e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step")


def test_verified_recovery_does_not_replenish_episode_context_allowance():
    backend = ContextBackend((False, False, False, True, False, False, False))
    result, _trace = run(backend, Planner(("valid",)))
    assert result.status == "stuck" and result.steps == result.planner_calls == 7
    assert result.context_recoveries == len(backend.takes) == 1
    assert backend.completed == ["Ready-3"]


def test_disappearing_context_stops_before_an_extra_planner_call_or_input():
    backend = ContextBackend(context_current=False)
    result, _trace = run(backend, Planner(("valid",)))
    assert result.status == "stuck" and result.reason == "new recovery context no longer observed"
    assert result.steps == result.planner_calls == 3
    assert result.context_recoveries == len(backend.confirms) == 1


def test_invalid_final_recovery_plan_does_not_gain_another_planning_retry():
    backend = ContextBackend()
    result, _trace = run(backend, Planner(("valid", "valid", "valid", "invalid")))
    assert result.status == "planner_error" and result.reason == "final context recovery planning failed"
    assert result.steps == 3 and result.planner_calls == 4 and result.context_recoveries == 1


@pytest.mark.parametrize("input_count,success", [(0, True), (True, True), (1, False)])
def test_no_input_or_failed_receipt_cannot_request_an_extra_recovery(input_count, success):
    backend = ContextBackend(input_count=input_count, receipt_success=success)
    result, _trace = run(backend, Planner(("valid",)))
    assert result.status == "stuck" and result.steps == result.planner_calls == 3
    assert result.context_recoveries == 0 and backend.takes == []


@pytest.mark.parametrize("config", [{"max_steps": 3}, {"max_plans": 3}])
def test_context_recovery_never_extends_total_step_or_plan_budgets(config):
    backend = ContextBackend()
    result, _trace = run(backend, Planner(("valid",)), **config)
    assert result.steps == result.planner_calls == 3
    assert result.context_recoveries == 0 and backend.takes == []


def test_verified_wait_does_not_clear_exhausted_failure_streak():
    class WaitPlanner(Planner):
        def plan(self, *args, **kwargs):
            plan = super().plan(*args, **kwargs)
            if len(self.calls) == 4:
                return replace(plan, steps=(replace(plan.steps[0], operation="wait", value="250"),))
            return plan

    backend = ContextBackend((False, False, False, True))
    result, trace = run(backend, WaitPlanner(("valid",)))
    assert result.status == "stuck" and result.reason == "new context recovery produced no input"
    assert result.steps == result.planner_calls == 4 and result.semantic_steps == 0
    assert result.context_recoveries == 1
    assert [e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step"][-1]


def test_already_satisfied_recovery_step_cannot_reset_failures_or_trigger_another_plan():
    class SatisfiedPlanner(Planner):
        def plan(self, *args, **kwargs):
            plan = super().plan(*args, **kwargs)
            if len(self.calls) == 4:
                effect = AgentCondition("exists", AgentTarget(name="Edit", app="Editor"))
                return replace(plan, steps=(replace(plan.steps[0], postconditions=(effect,)),))
            return plan

    result, _trace = run(ContextBackend(), SatisfiedPlanner(("valid",)))
    assert result.status == "stuck" and result.reason == "new context recovery produced no input"
    assert result.steps == 3 and result.planner_calls == 4 and result.semantic_steps == 0


def test_final_context_recovery_target_disappears_and_ttl_cannot_request_a_fifth_plan(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("cua_jev.agent_runner.time.perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        "cua_jev.agent_runner.time.sleep",
        lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    )

    class DisappearingBackend(ContextBackend):
        hidden = False

        def observe(self):
            snapshot = super().observe()
            return replace(snapshot, controls=()) if self.hidden else snapshot

        def compile(self, step, planned, current):
            if not matching_controls(current, step.target):
                return ()
            return super().compile(step, planned, current)

    backend = DisappearingBackend()

    class GroundedRecoveryPlanner(Planner):
        def plan(self, goal, snapshot, history, *, max_actions):
            plan = super().plan(goal, snapshot, history, max_actions=max_actions)
            if len(self.calls) == 4:
                # The recovery query is valid at request time; the UI changes
                # while the model is answering, before the mandatory reread.
                assert len(matching_controls(snapshot, plan.steps[0].target)) == 1
                backend.hidden = True
            return plan

    planner = GroundedRecoveryPlanner(("valid",))
    result, trace = run(backend, planner, max_plans=10, plan_ttl_s=0.1)
    assert result.status == "stuck"
    assert result.context_recoveries == 1 and result.planner_calls == 4
    assert result.steps == len(backend.inputs) == 3
    assert [call["limit"] for call in planner.calls] == [6, 1, 1, 1]
    assert planner.calls[-1]["history"][-1]["action_failures"] == 3
    assert result.plan_reuses == result.semantic_steps == 0
    assert any(event["kind"] == "agent_recovery"
               and event["payload"].get("reason") == "awaiting_conditions"
               for event in trace.events)


def test_final_context_recovery_skipped_step_completion_regresses_before_fresh_check():
    class TransientCompletionBackend(ContextBackend):
        completion_reads = 0

        def observe(self):
            snapshot = super().observe()
            if self.completion_reads:
                self.completion_reads -= 1
                marker = AgentControl("goal", "Goal reached", "text", role="label", app="Editor")
                return replace(snapshot, controls=(*snapshot.controls, marker))
            return snapshot

    backend = TransientCompletionBackend()

    class CompletionPlanner(Planner):
        def plan(self, *args, **kwargs):
            plan = super().plan(*args, **kwargs)
            if len(self.calls) == 4:
                backend.completion_reads = 1
                existing = AgentCondition("exists", AgentTarget(name="Edit", app="Editor"))
                goal = AgentCondition("exists", AgentTarget(name="Goal reached", app="Editor"))
                return replace(
                    plan,
                    steps=(replace(plan.steps[0], postconditions=(existing,)),),
                    completion_conditions=(goal,),
                )
            return plan

    planner = CompletionPlanner(("valid",))
    result, trace = run(backend, planner, max_plans=10)
    assert result.status == "stuck"
    assert result.context_recoveries == 1 and result.planner_calls == 4
    assert result.steps == len(backend.inputs) == 3
    assert result.condition_skips == 1 and result.plan_completion_claims == 0
    assert result.semantic_steps == result.plan_reuses == 0
    assert len(backend.confirms) == 1
    assert not any(event["kind"] == "agent_plan_completion" for event in trace.events)
