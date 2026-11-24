"""Fresh candidate selection after a receipt-bound refusal; never retry an old input."""

import io
from dataclasses import replace
from functools import lru_cache

import pytest
from PIL import Image

from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image
from cua_jev.trace import JsonlTrace


@lru_cache(maxsize=1)
def initial():
    image = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(image, "PNG")
    controls = tuple(AgentControl(
        name, name, "input", role="checkbox", input_type="checkbox", app="Settings",
        box=(x, 100, x + 100, 200),
    ) for name, x in (("Alpha", 100), ("Beta", 600)))
    return AgentSnapshot("osworld://desktop", "Settings", "Settings", controls,
                         image=screenshot_image(image.getvalue()))


def checked(name):
    return AgentCondition("checked", AgentTarget(name=name, app="Settings"))


def check(name="Alpha", **kwargs):
    return AgentStep(name, "check", target=checked(name).target, postconditions=(checked(name),), **kwargs)


class Environment:
    def __init__(self):
        self.snapshot = initial()
        self.actions = []
        self.reads = 0
        self.fail_read = False
        self.fail_input = False

    def _get_obs(self):
        self.reads += 1
        if self.fail_read:
            raise RuntimeError("unavailable")
        return self.snapshot

    def step(self, action, pause=0):
        self.actions.append(action)
        if self.fail_input:
            raise RuntimeError("input outcome unknown")
        assert action["action_type"] == "CLICK"
        x = action["parameters"]["x"]
        control = next(c for c in self.snapshot.controls if int((c.box[0] + c.box[2]) * 800 / 2000) == x)
        self.snapshot = replace(self.snapshot, controls=tuple(
            replace(c, checked=not c.checked) if c.ref == control.ref else c for c in self.snapshot.controls
        ))
        return self.snapshot, 0, False, {}

    def move(self, name="Alpha"):
        self.snapshot = replace(self.snapshot, controls=tuple(
            replace(c, box=(c.box[0] + 10, c.box[1], c.box[2] + 10, c.box[3]))
            if c.label == name else c for c in self.snapshot.controls
        ))


class Backend(OSWorldBackend):
    def __init__(self, env, *, stale_at=()):
        super().__init__(env, pause=0)
        self.stale_at = set(stale_at)
        self.executions = 0
        self.executed_candidates = []

    def _snapshot(self, observation):
        return observation

    def execute(self, candidate, observation_id, decision_id):
        self.executions += 1
        self.executed_candidates.append(candidate)
        if self.executions in self.stale_at:
            self.env.move(candidate.arguments["bound_control"]["label"])
        return super().execute(candidate, observation_id, decision_id)


def refusal():
    env = Environment()
    backend = Backend(env)
    before = backend.observe()
    candidate = backend.compile(check(), before, before)[0]
    env.move()
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success
    assert receipt.output == {"failure_code": "stale_observation", "environment_steps": 0}
    assert env.actions == [] and not backend.verify(candidate, receipt).passed
    return env, backend, candidate, receipt


def test_only_exact_zero_input_stale_refusal_can_be_consumed_once():
    env, backend, candidate, receipt = refusal()
    assert backend.consume_stale_rejection(candidate, receipt)
    assert not backend.consume_stale_rejection(candidate, receipt)
    assert env.actions == [] and not backend.verify(candidate, receipt).passed


@pytest.mark.parametrize("change", ["receipt_copy", "output", "candidate", "backend"])
def test_refusal_binding_rejects_copies_mutation_and_other_backends(change):
    env, backend, candidate, receipt = refusal()
    if change == "receipt_copy":
        receipt = replace(receipt)
    elif change == "output":
        receipt.output["environment_steps"] = 1
    elif change == "candidate":
        candidate = replace(candidate, description="different")
    else:
        backend = Backend(env)
    assert not backend.consume_stale_rejection(candidate, receipt)


@pytest.mark.parametrize("change", [
    "observe", "failed_read", "compile", "input", "execute", "pending", "close",
])
def test_any_intervening_backend_operation_invalidates_a_refusal(change):
    env, backend, candidate, receipt = refusal()
    if change == "observe":
        backend.observe()
    elif change == "failed_read":
        env.fail_read = True
        with pytest.raises(RuntimeError):
            backend.observe()
    elif change == "compile":
        backend.compile(check(), env.snapshot, env.snapshot)
    elif change == "input":
        backend._step("CLICK", {"x": 128, "y": 90})
    elif change == "execute":
        backend.execute(candidate, "different", "different")
    elif change == "pending":
        backend.verify_pending(candidate, receipt)
    else:
        backend.close()
    assert not backend.consume_stale_rejection(candidate, receipt)


@pytest.mark.parametrize("failure", ["altered_candidate", "input_started"])
def test_other_failures_cannot_be_relabelled_as_zero_input_staleness(failure):
    env = Environment()
    backend = Backend(env)
    snapshot = backend.observe()
    candidate = backend.compile(check(), snapshot, snapshot)[0]
    if failure == "altered_candidate":
        candidate = replace(candidate, description="altered")
    else:
        env.fail_input = True
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success
    receipt.output.update(failure_code="stale_observation", environment_steps=0)
    assert not backend.consume_stale_rejection(candidate, receipt)


class Planner:
    def __init__(self):
        self.calls = []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.calls.append((snapshot, list(history), max_actions))
        remaining = [c.label for c in snapshot.controls if not c.checked][:max_actions]
        steps = tuple(check(name, depends_on=(remaining[i - 1],) if i else ())
                      for i, name in enumerate(remaining))
        return AgentPlan("Enable both requested options", steps, reusable=True,
                         completion_conditions=(checked("Alpha"), checked("Beta")))


class Policy:
    def __init__(self):
        self.selections = []

    def choose(self, observation, candidates):
        self.selections.append(candidates[0])
        return DirectPlanPolicy().choose(observation, candidates)


def run(*, stale_at=(1,), progress=None, independent=True, **settings):
    env = Environment()
    backend, planner, policy, trace = Backend(env, stale_at=stale_at), Planner(), Policy(), JsonlTrace()
    config = AgentConfig(allow_actions=True, singleton_fast_path=False, **{"max_steps": 6, **settings})
    success = (lambda s: all(c.checked for c in s.controls)) if independent else None
    result = AgentRunner(backend, planner, policy=policy, trace=trace, config=config, progress=progress).run(
        "Enable both options", success=success,
    )
    return result, env, backend, planner, policy, trace


def test_stale_refusal_preserves_stage_but_reobserves_and_reselects_changed_geometry():
    result, env, backend, planner, policy, trace = run()
    assert result.success and result.planner_calls == len(planner.calls) == 1
    assert result.steps == 3 and result.stale_recovery_attempts == 1 and result.plan_reuses == 1
    assert len(env.actions) == 2 and env.actions[0]["parameters"]["x"] == 128
    assert len(policy.selections) == 3
    assert policy.selections[0].id != policy.selections[1].id
    assert all(c.arguments == {} for c in policy.selections)  # Policy still sees only public candidates.
    assert backend.executed_candidates[0].arguments["bound_control"]["box"] != (
        backend.executed_candidates[1].arguments["bound_control"]["box"]
    )
    steps = [e["payload"] for e in trace.events if e["kind"] == "agent_step"]
    assert [s["verified"] for s in steps] == [False, True, True]
    assert backend.fresh_observe_calls >= 6


def test_second_stale_refusal_in_same_plan_requires_vlm_even_after_verified_progress():
    result, env, _backend, planner, _policy, _trace = run(stale_at=(1, 3))
    assert result.success and result.stale_recovery_attempts == 1
    assert len(planner.calls) == 2 and planner.calls[1][2] == 1
    assert len(env.actions) == 2 and result.steps == 4


def test_opt_in_rearms_only_after_verified_gui_progress_and_keeps_one_root_plan():
    result, env, _backend, planner, _policy, trace = run(stale_at=(1, 3), stale_recovery_on_progress=True)
    assert result.success and result.planner_calls == len(planner.calls) == 1
    assert result.stale_recovery_attempts == 2 and result.steps == 4 and len(env.actions) == 2
    assert all(c.checked for c in env.snapshot.controls)
    events = [e["payload"] for e in trace.events if e["kind"] == "agent_stale_recovery"]
    assert [e["stage"] for e in events] == ["requested", "rearmed_after_verified_input",
                                          "requested", "rearmed_after_verified_input"]
    assert all(e["per_progress_limit"] == 1 and e["task_success"] is False for e in events)


def test_opt_in_consecutive_refusals_still_require_vlm_without_intervening_verified_input():
    result, env, _backend, planner, _policy, trace = run(stale_at=(1, 2), stale_recovery_on_progress=True)
    assert result.success and result.stale_recovery_attempts == 1
    assert len(planner.calls) == 3 and planner.calls[1][2] == 1
    assert len(env.actions) == 2 and result.steps == 4
    events = [e["payload"] for e in trace.events if e["kind"] == "agent_stale_recovery"]
    assert [e["stage"] for e in events] == ["requested"]


@pytest.mark.parametrize("setting", [{"max_failures": 1}, {"max_steps": 1}, {"mode": "vlm"}])
def test_opt_in_preserves_baseline_and_original_input_failure_budgets(setting):
    result, env, _backend, _planner, _policy, _trace = run(stale_recovery_on_progress=True, **setting)
    assert result.stale_recovery_attempts == 0
    if setting.get("mode") != "vlm":
        assert not result.success and result.steps == 1 and not env.actions


def test_rearmed_allowance_never_resets_the_total_attempt_budget_after_progress():
    result, env, _backend, planner, _policy, _trace = run(
        stale_at=(1, 3), stale_recovery_on_progress=True, max_steps=3,
    )
    assert result.status == "max_steps" and result.steps == 3 and not result.success
    assert result.stale_recovery_attempts == 1 and len(planner.calls) == 1 and len(env.actions) == 1


@pytest.mark.parametrize("invalid", [None, 0, 1, "true"])
def test_progress_recovery_requires_an_explicit_boolean_config(invalid):
    with pytest.raises(ValueError, match="stale_recovery_on_progress"):
        AgentConfig(stale_recovery_on_progress=invalid)


@pytest.mark.parametrize("settings", [{"max_failures": 1}, {"max_steps": 1}, {"mode": "vlm"}])
def test_stale_recovery_does_not_extend_failure_or_action_budgets_or_the_baseline(settings):
    result, env, _backend, _planner, _policy, _trace = run(**settings)
    assert result.stale_recovery_attempts == 0
    if settings.get("mode") != "vlm":
        assert not result.success and len(env.actions) == 0 and result.steps == 1


@pytest.mark.parametrize("on_progress", [False, True])
@pytest.mark.parametrize("budget", ["episode", "plan"])
def test_retained_stage_must_still_pass_episode_and_plan_expiry_on_next_observation(
    monkeypatch, budget, on_progress,
):
    clock = [1000.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("time.time", lambda: clock[0])

    def progress(event):
        if event["event"] == "agent_stale_recovery":
            clock[0] += 6

    settings = {"timeout_s": 5} if budget == "episode" else {"plan_ttl_s": 5}
    result, env, _backend, planner, _policy, _trace = run(
        progress=progress, stale_recovery_on_progress=on_progress, **settings,
    )
    assert result.stale_recovery_attempts == 1
    if budget == "episode":
        assert result.status == "timeout" and len(env.actions) == 0 and len(planner.calls) == 1
    else:
        assert result.success and len(planner.calls) == 2
        assert result.replan_reasons["plan_expired"] == 1


def test_stale_snapshot_is_the_new_refusing_frame_not_the_original_candidate_frame():
    env, backend, candidate, receipt = refusal()
    assert backend.take_stale_preflight_snapshot(candidate, receipt) is None
    assert backend.consume_stale_rejection(candidate, receipt)
    cached = backend.take_stale_preflight_snapshot(candidate, receipt)
    assert cached is not None and cached[0] is env.snapshot and cached[0] is not initial()
    assert cached[0].controls[0].box != tuple(candidate.arguments["bound_control"]["box"])
    assert backend.take_stale_preflight_snapshot(candidate, receipt) is None
    assert not backend.verify(candidate, receipt).passed and env.actions == []


@pytest.mark.parametrize("change", ["receipt_copy", "output", "candidate", "backend"])
def test_decision_cache_rechecks_receipt_binding_after_consumption(change):
    env, backend, candidate, receipt = refusal()
    assert backend.consume_stale_rejection(candidate, receipt)
    if change == "receipt_copy":
        receipt = replace(receipt)
    elif change == "output":
        receipt.output["unexpected"] = True
    elif change == "candidate":
        candidate = replace(candidate, description="changed")
    else:
        backend = Backend(env)
    assert backend.take_stale_preflight_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("change", [
    "observe", "failed_read", "compile", "input", "execute", "pending", "close",
])
def test_intervening_operations_clear_the_consumed_stale_decision_frame(change):
    env, backend, candidate, receipt = refusal()
    assert backend.consume_stale_rejection(candidate, receipt)
    if change == "observe":
        backend.observe()
    elif change == "failed_read":
        env.fail_read = True
        with pytest.raises(RuntimeError):
            backend.observe()
    elif change == "compile":
        backend.compile(check(), env.snapshot, env.snapshot)
    elif change == "input":
        backend._step("CLICK", {"x": 128, "y": 90})
    elif change == "execute":
        backend.execute(candidate, "next", "next")
    elif change == "pending":
        backend.verify_pending(candidate, receipt)
    else:
        backend.close()
    assert backend.take_stale_preflight_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("age,eligible", [
    (0, True), (0.99, True), (1.0, True), (1.01, False), (-0.01, False),
])
def test_stale_frame_age_is_bounded_and_never_resurrected(monkeypatch, age, eligible):
    clock = [1000.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    env, backend, candidate, receipt = refusal()
    assert backend.consume_stale_rejection(candidate, receipt)
    clock[0] += age
    assert (backend.take_stale_preflight_snapshot(candidate, receipt) is not None) is eligible
    assert backend.take_stale_preflight_snapshot(candidate, receipt) is None
    assert env.actions == []


def test_stale_frame_age_includes_snapshot_processing(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])

    class SlowBackend(Backend):
        def _snapshot(self, observation):
            clock[0] += 1.01
            return observation

    env = Environment()
    backend = SlowBackend(env)
    snapshot = backend.observe()
    candidate = backend.compile(check(), snapshot, snapshot)[0]
    env.move()
    receipt = backend.execute(candidate, "one", "one")
    assert not receipt.success and backend.consume_stale_rejection(candidate, receipt)
    assert backend.take_stale_preflight_snapshot(candidate, receipt) is None


def test_stale_decision_frame_saves_a_read_but_preserves_new_selection_and_fresh_execution():
    result, env, backend, planner, policy, trace = run(independent=False)
    assert result.status == "unverified" and result.planner_calls == len(planner.calls) == 1
    assert result.stale_snapshot_reuses == 1 and result.post_snapshot_reuses == 1
    assert result.stale_recovery_attempts == 1 and result.steps == 3
    assert len(policy.selections) == 3 and len(env.actions) == 2
    assert backend.fresh_observe_calls == env.reads == 6
    events = [e["payload"] for e in trace.events if e["kind"] == "agent_observation_reuse"]
    assert [e["source"] for e in events] == ["stale_rejection_preflight", "verified_step_response"]
    verdicts = [e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step"]
    assert verdicts == [False, True, True]


@pytest.mark.parametrize("settings", [{"independent": True}, {"reuse_verified_post_snapshot": False}])
def test_independent_acceptance_and_explicit_optout_keep_fresh_observations(settings):
    result, env, backend, _planner, _policy, _trace = run(**{"independent": False, **settings})
    assert result.stale_recovery_attempts == 1 and result.stale_snapshot_reuses == 0
    assert backend.fresh_observe_calls == env.reads == 8
    assert len(env.actions) == 2


def test_progress_delay_expires_cache_without_discarding_the_valid_stage(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("time.time", lambda: clock[0])

    def progress(event):
        if event["event"] == "agent_stale_recovery":
            clock[0] += 1.01

    result, env, backend, planner, _policy, _trace = run(independent=False, progress=progress)
    assert result.status == "unverified" and len(planner.calls) == 1
    assert result.stale_recovery_attempts == 1 and result.stale_snapshot_reuses == 0
    assert backend.fresh_observe_calls == env.reads == 7
    assert len(env.actions) == 2


def test_another_movement_after_cached_selection_is_still_refused_before_any_input():
    result, env, backend, planner, policy, trace = run(independent=False, stale_at=(1, 2), max_steps=2)
    assert result.steps == 2 and result.status == "max_steps"
    assert result.stale_recovery_attempts == result.stale_snapshot_reuses == 1
    assert len(planner.calls) == 1 and len(policy.selections) == 2
    assert backend.environment_steps == 0 and env.actions == []
    receipts = [e["payload"] for e in trace.events if e["kind"] == "receipt"]
    assert all(not receipt["success"] and receipt["output"]["environment_steps"] == 0 for receipt in receipts)
