from dataclasses import replace

import pytest

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentControl, AgentPlan, AgentSnapshot, AgentStep
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Risk, Verification
from cua_jev.trace import JsonlTrace


class Backend:
    def __init__(self, *, fail=False, change_scope=False):
        self.values = {}
        self.executed = []
        self.fail = fail
        self.change_scope = change_scope

    def observe(self):
        controls = tuple(
            AgentControl(f"e{i}", name, "input", value=self.values.get(name, ""))
            for i, name in enumerate(("Name", "City"))
        )
        return AgentSnapshot("https://local.test/", "Form", "Form", controls)

    def compile(self, step, planned, current):
        return (
            ActionCandidate(
                step.id,
                Channel.SCRIPT,
                "agent.browser.fill",
                step.label,
                arguments={"route": "dom", "value": step.value, "ref": step.ref},
                risk=Risk.LOCAL_WRITE,
                intent=step.id,
                verifier="agent.effect",
            ),
        )

    def execute(self, candidate, observation_id, decision_id):
        self.executed.append(candidate.id)
        if not self.fail:
            self.values[candidate.description] = candidate.arguments["value"]
        return ActionReceipt(
            observation_id,
            decision_id,
            candidate.id,
            candidate.channel,
            candidate.capability,
            not self.fail,
            0,
            0.001,
            output={
                "postcondition": {
                    "identity": list(
                        next(
                            c for c in self.observe().controls if c.label == candidate.description
                        ).identity()
                    ),
                    "value": self.values.get(candidate.description, ""),
                    "checked": False,
                }
            },
        )

    def verify(self, candidate, receipt):
        return Verification(receipt.success, "agent.effect")


class Planner:
    def __init__(self):
        self.calls = 0
        self.horizons = []
        self.usage_totals = {"total_tokens": 0}

    def plan(self, goal, snapshot, history, *, max_actions=6):
        self.calls += 1
        self.usage_totals["total_tokens"] += 25
        self.horizons.append(max_actions)
        steps = []
        for control, value in zip(snapshot.controls, ("Ada", "Hangzhou"), strict=True):
            if control.value != value:
                steps.append(
                    AgentStep(
                        control.label,
                        "fill",
                        control.ref,
                        value,
                        depends_on=(steps[-1].id,) if steps else (),
                        label=control.label,
                    )
                )
        return AgentPlan("Fill the form", tuple(steps[:max_actions]), done=not steps)


def done(snapshot):
    return tuple(c.value for c in snapshot.controls) == ("Ada", "Hangzhou")


def run(backend=None, planner=None, **config):
    backend, planner = backend or Backend(), planner or Planner()
    trace = JsonlTrace()
    result = AgentRunner(
        backend,
        planner,
        policy=DirectPlanPolicy(),
        config=AgentConfig(allow_actions=True, **config),
        trace=trace,
    ).run("Fill form", success=done)
    return result, backend, planner, trace


def test_dependency_plan_reused_across_value_changes():
    result, backend, planner, trace = run()
    assert result.success and result.steps == 2
    assert planner.calls == 1 and result.plan_reuses == 1
    assert result.jev_calls == 0 and result.singleton_decisions == 2
    assert backend.executed == ["Name", "City"]
    assert result.model_usage == {"total_tokens": 25}
    assert len([e for e in trace.events if e["kind"] == "verification"]) == 2


def test_vlm_baseline_calls_same_planner_for_every_action_without_jev():
    result, _, planner, _ = run(mode="vlm")
    assert result.success and result.steps == 2
    assert planner.calls == 2 and planner.horizons == [1, 1]
    assert result.jev_calls == 0 and result.plan_reuses == 0


def test_last_permitted_action_can_satisfy_independent_goal():
    assert run(max_steps=2)[0].success


def test_failed_effect_replans_and_stops_at_failure_budget():
    result, backend, planner, _ = run(Backend(fail=True))
    assert result.status == "stuck" and planner.calls == 3
    assert backend.executed == ["Name"] * 3
    assert result.replan_reasons["effect_failed"] == 2


def test_model_done_is_rejected_when_evaluator_disagrees():
    class LyingPlanner:
        def plan(self, *args, **kwargs):
            return AgentPlan("Done", (), done=True)

    result, backend, _, _ = run(planner=LyingPlanner())
    assert not result.success and result.status == "task_failure"
    assert result.steps == 0 and not backend.executed


def test_completion_without_evaluator_is_unverified():
    class Done:
        def plan(self, *args, **kwargs):
            return AgentPlan("Done", (), done=True)

    result = AgentRunner(Backend(), Done(), config=AgentConfig(mode="vlm")).run("Finish")
    assert result.status == "unverified" and not result.success


def test_structural_change_forces_fresh_plan():
    class ChangingBackend(Backend):
        def observe(self):
            snapshot = super().observe()
            return replace(snapshot, title="Changed" if self.executed else snapshot.title)

    result, _, planner, _ = run(ChangingBackend())
    assert result.success and planner.calls == 2
    assert result.replan_reasons["page_or_controls_changed"] == 1


def test_target_unavailable_is_replanned_boundedly():
    class MissingBackend(Backend):
        def compile(self, *args):
            return ()

    result, backend, _, _ = run(MissingBackend())
    assert result.status == "stuck" and not backend.executed
    assert result.planner_calls == 3


def test_planning_error_can_recover_and_does_not_leak_exception():
    class FlakyPlanner(Planner):
        def plan(self, *args, **kwargs):
            if not self.calls:
                self.calls += 1
                raise ValueError("secret response body")
            return super().plan(*args, **kwargs)

    result, _, _, trace = run(planner=FlakyPlanner())
    assert result.success and result.planner_calls == 2
    assert "secret response body" not in str(trace.events)


def test_policy_failure_is_counted_without_fallback():
    class Broken:
        def choose(self, *args):
            raise RuntimeError("secret network endpoint")

    backend = Backend()
    result = AgentRunner(backend, Planner(), policy=Broken(), config=AgentConfig(
        allow_actions=True, singleton_fast_path=False,
    )).run(
        "Fill"
    )
    assert result.status == "error" and result.jev_calls == 1
    assert not backend.executed and "secret" not in result.reason


def test_policy_diagnostics_trace_only_allowlisted_typed_fields():
    class Broken:
        last_failure = {"code": "connect_timeout", "attempts": 3, "retryable": True,
                        "http_status": "secret-url", "response": "secret-body"}

        def choose(self, *args):
            raise RuntimeError("secret endpoint")

    backend, trace = Backend(), JsonlTrace()
    result = AgentRunner(backend, Planner(), policy=Broken(), trace=trace,
                         config=AgentConfig(allow_actions=True, singleton_fast_path=False)).run("Fill")
    error = next(e["payload"] for e in trace.events if e["kind"] == "agent_error")
    assert error["policy_failure"] == {
        "code": "connect_timeout", "attempts": 3, "retryable": True, "http_status": None,
    }
    assert result.steps == 0 and not backend.executed
    assert "secret" not in str(trace.events)


def test_write_guard_is_still_enforced():
    backend = Backend()
    result = AgentRunner(backend, Planner(), policy=DirectPlanPolicy()).run("Fill", success=done)
    assert result.reason == "GuardRejected" and not backend.executed
    assert result.jev_calls == 0 and result.singleton_decisions == 1


def test_singleton_does_not_need_service_and_cannot_claim_a_jev_exchange():
    class Unavailable:
        last_exchange = {"response": "stale exchange from an earlier step"}

        def choose(self, *args):
            raise AssertionError("a single legal candidate does not need remote selection")

    backend, trace = Backend(), JsonlTrace()
    result = AgentRunner(backend, Planner(), policy=Unavailable(), trace=trace,
                         config=AgentConfig(allow_actions=True)).run("Fill form", success=done)
    assert result.success and result.jev_calls == 0 and result.singleton_decisions == 2
    assert not any(e["kind"] == "policy_exchange" for e in trace.events)
    decisions = [e["payload"] for e in trace.events if e["kind"] == "decision"]
    assert {d["model"] for d in decisions} == {"deterministic-singleton"}
    assert backend.executed == ["Name", "City"]


def test_done_hint_cannot_suppress_valid_steps_or_determine_success():
    class AdvisoryPlanner(Planner):
        def plan(self, *args, **kwargs):
            return replace(super().plan(*args, **kwargs), done=True)

    result, backend, _, _ = run(planner=AdvisoryPlanner())
    assert result.success and backend.executed == ["Name", "City"]


def test_usage_is_per_episode_when_planner_is_reused():
    planner = Planner()
    assert run(planner=planner)[0].model_usage == {"total_tokens": 25}
    assert run(planner=planner)[0].model_usage == {"total_tokens": 25}
    assert planner.usage_totals == {"total_tokens": 50}


def test_value_change_during_slow_planning_discards_plan_before_jev():
    backend = Backend()

    class SlowPlanner(Planner):
        def plan(self, *args, **kwargs):
            plan = super().plan(*args, **kwargs)
            if self.calls == 1:
                backend.values["City"] = "Unexpected"
            return plan

    result, _, planner, _ = run(backend, SlowPlanner())
    assert result.success and planner.calls == 2 and result.jev_calls == 0
    assert result.singleton_decisions == 2
    assert result.replan_reasons["changed_during_planning"] == 1


def test_action_resetting_completed_field_forces_replan_before_submit():
    class ResetBackend(Backend):
        def execute(self, candidate, observation_id, decision_id):
            if candidate.id == "Submit":
                assert self.values == {"Name": "Ada", "City": "Hangzhou"}
                self.executed.append("Submit")
                return ActionReceipt(
                    observation_id,
                    decision_id,
                    candidate.id,
                    candidate.channel,
                    candidate.capability,
                    True,
                    0,
                    0.001,
                )
            receipt = super().execute(candidate, observation_id, decision_id)
            if candidate.id == "City":
                self.values["Name"] = ""
            return receipt

    class SubmitPlanner(Planner):
        def plan(self, *args, **kwargs):
            plan = super().plan(*args, **kwargs)
            return replace(
                plan,
                steps=(
                    *plan.steps,
                    AgentStep("Submit", "click", depends_on=tuple(s.id for s in plan.steps), label="Submit"),
                ),
                done=False,
            )

    backend, planner = ResetBackend(), SubmitPlanner()
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), config=AgentConfig(allow_actions=True)
    ).run("Fill then submit", success=lambda _: "Submit" in backend.executed)
    assert result.success and backend.executed == ["Name", "City", "Name", "Submit"]
    assert planner.calls == 2 and result.replan_reasons["completed_state_changed"] == 1


def test_slow_jev_cannot_start_action_after_episode_deadline(monkeypatch):
    from cua_jev import agent_runner

    clock = [0.0]
    monkeypatch.setattr(agent_runner.time, "perf_counter", lambda: clock[0])

    class SlowPolicy(DirectPlanPolicy):
        def choose(self, *args):
            clock[0] = 10.0
            return super().choose(*args)

    backend = Backend()
    result = AgentRunner(
        backend, Planner(), policy=SlowPolicy(),
        config=AgentConfig(allow_actions=True, timeout_s=1, singleton_fast_path=False),
    ).run("Fill", success=done)
    assert result.status == "timeout" and result.jev_calls == 1
    assert result.steps == 0 and not backend.executed


def test_unplanned_changes_to_previously_correct_fields_invalidate_plan():
    backend = Backend()
    backend.values["City"] = "Hangzhou"
    planner = Planner()

    def external_edit(event):
        if event["event"] == "agent_step" and event["step"] == 1:
            backend.values["City"] = "Unexpected"

    result = AgentRunner(
        backend,
        planner,
        policy=DirectPlanPolicy(),
        progress=external_edit,
        config=AgentConfig(allow_actions=True),
    ).run("Fill form", success=done)
    assert result.success and planner.calls == 2
    assert result.replan_reasons["field_state_changed"] == 1


@pytest.mark.parametrize("operation", ["type_text", "hotkey", "key_press", "press"])
def test_desktop_keyboard_actions_require_new_planning(operation):
    class KeyboardPlanner(Planner):
        def plan(self, *args, **kwargs):
            plan = super().plan(*args, **kwargs)
            return replace(plan, steps=tuple(replace(s, operation=operation) for s in plan.steps))

    result, _, planner, _ = run(planner=KeyboardPlanner())
    assert result.success and planner.calls == 2 and result.plan_reuses == 0
    assert result.replan_reasons["visual_or_page_action"] == 1


@pytest.mark.parametrize(
    "kwargs", [{"mode": "unknown"}, {"plan_horizon": 13}, {"max_steps": 0}, {"timeout_s": float("nan")}]
)
def test_config_bounds(kwargs):
    with pytest.raises(ValueError):
        AgentConfig(**kwargs)
