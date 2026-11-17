"""Dispatch must see full bindings and cannot bypass execution guards."""

from dataclasses import replace

import pytest
from test_agent_runner import Backend, Planner, run

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.dispatch import DispatchEscalation, DispatchPolicy, identical_execution_candidates
from cua_jev.errors import GuardRejected
from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, Channel, Observation, Risk
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime


class Spy(DirectPlanPolicy):
    def __init__(self):
        self.calls = []
        self.last_exchange = {"prior": "exchange"}

    def choose(self, observation, candidates):
        self.calls.append((observation, candidates))
        return super().choose(observation, candidates)


def candidate():
    return ActionCandidate("a", Channel.SCRIPT, "agent.browser.fill", "Fill Name",
                           arguments={"ref": "e0", "value": "Ada", "route": "dom"},
                           expected={"value": "Ada"}, intent="name", risk=Risk.LOCAL_WRITE)


@pytest.mark.parametrize("aliases", [False, True])
def test_singleton_and_exact_aliases_skip_model_without_mutation(aliases):
    model, events = Spy(), []
    policy = DispatchPolicy(PublicDecisionPolicy(model), emit=lambda k, v: events.append((k, v)))
    first = candidate()
    offered = (first, replace(first, id="b")) if aliases else (first,)
    before = [c.to_dict() for c in offered]
    obs = Observation("Fill", "Name", {})
    chosen = policy.choose(obs, offered)
    assert chosen.candidate_id == "a" and chosen.observation_id == obs.observation_id
    assert not model.calls and policy.last_exchange == {}
    assert [c.to_dict() for c in offered] == before
    assert policy.equivalent_decisions == int(aliases)
    assert policy.singleton_decisions == int(not aliases)
    assert events[-1][1]["route"] == "rule"


@pytest.mark.parametrize("change", [
    {"arguments": {"ref": "e0", "value": "Bob", "route": "dom"}},
    {"arguments": {"ref": "e1", "value": "Ada", "route": "dom"}},
    {"arguments": {"ref": "e0", "value": "Ada", "route": "mouse"}},
    {"expected": {"value": "Bob"}}, {"intent": "another-step"},
    {"preconditions": ("ready",)}, {"risk": Risk.EXTERNAL_SIDE_EFFECT},
    {"capability": "agent.browser.click"}, {"channel": Channel.GUI},
    {"requires_confirmation": True},
])
def test_binding_differences_are_not_hidden_by_public_projection(change):
    model = Spy()
    policy = DispatchPolicy(PublicDecisionPolicy(model))
    first = candidate()
    offered = (first, replace(first, id="b", **change))
    assert not identical_execution_candidates(offered)
    policy.choose(Observation("Fill", "Name", {}), offered)
    assert len(model.calls) == 1
    assert all(c.arguments == {} and c.expected == {} for c in model.calls[0][1])
    assert policy.last_exchange == model.last_exchange
    policy.choose(Observation("Fill", "Name", {}), (first,))
    assert policy.last_exchange == {}  # no stale exchange on the next rule step


def test_explicit_property_confirmation_remains_model_routed():
    model = Spy()
    policy = DispatchPolicy(model)
    policy.choose(Observation("Check", "Property", {"explicit_property_confirmation": True}),
                  (candidate(),))
    assert len(model.calls) == 1 and policy.singleton_decisions == 0


def test_incomplete_evidence_escalates_before_any_model_selection():
    model = Spy()
    policy = DispatchPolicy(model)
    offered = (candidate(), replace(candidate(), id="b", intent="other"))
    with pytest.raises(DispatchEscalation, match="incomplete_observation"):
        policy.choose(Observation("Fill", "Name", {"a11y_complete": False}), offered)
    assert not model.calls


def test_rule_selection_still_passes_guard_before_execution():
    runtime = AgentRuntime(policy=DispatchPolicy(Spy()), guard=ActionGuard(),
                           executors=ExecutorRegistry())
    with pytest.raises(GuardRejected, match="local writes are disabled"):
        runtime.step(Observation("Fill", "Name", {}), (candidate(),))


def test_runner_counts_rule_decisions_and_keeps_effect_checks():
    result, backend, _, trace = run(adaptive_dispatch=True)
    assert result.success and result.singleton_decisions == 2 and result.jev_calls == 0
    assert backend.executed == ["Name", "City"]
    assert sum(e["kind"] == "verification" for e in trace.events) == 2


class IncompleteBackend(Backend):
    def __init__(self, *, resolve=False):
        super().__init__()
        self.reads = 0
        self.resolve = resolve

    def observe(self):
        self.reads += 1
        return replace(super().observe(), a11y_complete=self.resolve and self.reads >= 3)

    def compile(self, *args):
        first = super().compile(*args)[0]
        return (first, replace(first, id=first.id + "-other", description="Alternative route"))


def test_persistent_missing_evidence_is_bounded_without_model_choice_or_input():
    result, backend, planner, trace = run(IncompleteBackend(), adaptive_dispatch=True)
    assert result.status == "stuck" and not backend.executed and result.jev_calls == 0
    assert result.dispatch_refreshes == 1
    assert planner.horizons == [6, 1, 1]
    routes = [e["payload"]["route"] for e in trace.events if e["kind"] == "agent_dispatch"]
    assert routes.count("observe") == 1


def test_one_fresh_observation_can_resolve_evidence_without_replanning():
    result, backend, planner, _ = run(IncompleteBackend(resolve=True), adaptive_dispatch=True)
    assert result.success and result.dispatch_refreshes == 1
    assert planner.calls == 1 and backend.executed == ["Name", "City"]


def test_verified_progress_resets_dispatch_failure_streak_but_not_total_counters():
    class Multi(Backend):
        def compile(self, *args):
            first = super().compile(*args)[0]
            return (first, replace(first, id=first.id + "-other", description="Alternative route"))

    class Declining(DirectPlanPolicy):
        calls = 0

        def choose(self, observation, candidates):
            self.calls += 1
            if self.calls % 3:
                raise DispatchEscalation("uncertain_choice")
            return super().choose(observation, candidates)

    backend, planner = Multi(), Planner()
    result = AgentRunner(backend, planner, policy=Declining(), config=AgentConfig(
        allow_actions=True, adaptive_dispatch=True, max_failures=2,
    )).run("Fill form", success=lambda s: tuple(c.value for c in s.controls) == ("Ada", "Hangzhou"))
    assert result.success and backend.executed == ["Name", "City"]
    assert result.dispatch_refreshes == result.dispatch_s2_replans == 2
    assert planner.horizons == [6, 1, 6, 1]


def test_browser_invalid_plan_retries_with_h1_then_resumes_after_verified_input():
    class Flaky(Planner):
        profile = "browser"

        def plan(self, *args, **kwargs):
            if not self.calls:
                self.calls += 1
                self.horizons.append(kwargs["max_actions"])
                raise ValueError("invalid multi-action plan")
            return super().plan(*args, **kwargs)

    result, _, planner, _ = run(planner=Flaky(), adaptive_dispatch=True)
    assert result.success and planner.horizons == [6, 1, 6]


@pytest.mark.parametrize("kwargs", [{"adaptive_dispatch": 1},
                                   {"adaptive_dispatch": True, "mode": "vlm"}])
def test_adaptive_dispatch_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError, match="adaptive_dispatch"):
        AgentConfig(**kwargs)


class BrowserPlanner(Planner):
    profile = "browser"


def test_missing_browser_binding_refreshes_once_then_recovers_at_h1():
    class Missing(Backend):
        def compile(self, step, planned, current):
            if planner.calls < 2:
                return ()
            return super().compile(step, planned, current)
    planner = BrowserPlanner()
    result, backend, _, trace = run(Missing(), planner=planner, adaptive_dispatch=True)
    assert result.success and backend.executed == ["Name", "City"]
    assert planner.horizons == [6, 1, 6]
    assert result.dispatch_refreshes == result.dispatch_s2_replans == 1
    routes = [e["payload"] for e in trace.events if e["kind"] == "agent_dispatch"]
    assert [r["route"] for r in routes[:2]] == ["observe", "vlm"]


def test_transient_missing_browser_binding_does_not_call_the_planner_again():
    class Appearing(Backend):
        attempts = 0
        def compile(self, *args):
            self.attempts += 1
            return () if self.attempts == 1 else super().compile(*args)
    result, _, planner, _ = run(Appearing(), planner=BrowserPlanner(), adaptive_dispatch=True)
    assert result.success and planner.horizons == [6]
    assert result.dispatch_refreshes == 1 and result.dispatch_s2_replans == 0


def test_persistently_missing_browser_binding_has_a_finite_failure_budget():
    class Missing(Backend):
        def compile(self, *args):
            return ()
    result, backend, planner, _ = run(Missing(), planner=BrowserPlanner(), adaptive_dispatch=True,
                                      max_failures=2)
    assert result.status == "stuck" and not backend.executed
    assert planner.horizons == [6, 1] and result.dispatch_refreshes == 1


def test_failed_browser_input_uses_h1_repair_before_restoring_normal_horizon():
    class OnceFailed(Backend):
        def execute(self, *args):
            self.fail = not self.executed
            return super().execute(*args)
    result, backend, planner, trace = run(OnceFailed(), planner=BrowserPlanner(), adaptive_dispatch=True)
    assert result.success and backend.executed == ["Name", "Name", "City"]
    assert planner.horizons == [6, 1, 6]
    recovery = [e['payload'] for e in trace.events if e['kind'] == 'agent_recovery']
    assert recovery[0]['recovery_required'] is True
