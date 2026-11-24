"""Decision reuse saves one read without caching preflight or completion evidence."""

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


@pytest.fixture
def snapshot():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, format="PNG")
    return AgentSnapshot("osworld://desktop", "Settings", "Settings", (
        AgentControl("a", "Alpha", "input", role="checkbox", input_type="checkbox", app="Settings",
                     box=(100, 100, 200, 200)),
        AgentControl("b", "Beta", "input", role="checkbox", input_type="checkbox", app="Settings",
                     box=(400, 100, 500, 200)),
    ), image=screenshot_image(buffer.getvalue()))


class Environment:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.actions = []
        self.reads = 0
        self.on_read = None
        self.on_step = None

    def _get_obs(self):
        self.reads += 1
        if self.on_read:
            self.on_read(self)
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        if self.on_step:
            return self.on_step(self, action), 0, False, {}
        assert action["action_type"] == "CLICK"
        ref = "a" if action["parameters"]["x"] < 250 else "b"
        self.snapshot = replace(self.snapshot, controls=tuple(
            replace(control, checked=not control.checked) if control.ref == ref else control
            for control in self.snapshot.controls
        ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def __init__(self, env):
        super().__init__(env, pause=0)
        self.take_calls = 0
        self.taken = []

    def _snapshot(self, observation):
        return observation

    def take_verified_post_snapshot(self, candidate, receipt):
        self.take_calls += 1
        result = super().take_verified_post_snapshot(candidate, receipt)
        if result:
            self.taken.append(result[0])
        return result


def checked(name):
    return AgentCondition("checked", AgentTarget(name=name, app="Settings"))


def check_step(name="Alpha", **kwargs):
    return AgentStep(name, "check", target=checked(name).target,
                     postconditions=(checked(name),), **kwargs)


def execute(backend, step=None):
    snapshot = backend.env.snapshot
    candidate = backend.compile(step or check_step(), snapshot, snapshot)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    return candidate, receipt


class Planner:
    def __init__(self):
        self.snapshots = []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.snapshots.append(snapshot)
        steps = []
        for name in ("Alpha", "Beta"):
            if not any(c.label == name and c.checked for c in snapshot.controls):
                steps.append(check_step(name, depends_on=(steps[-1].id,) if steps else ()))
        stage = tuple(steps[:max_actions])
        completion = (checked("Alpha"), checked("Beta")) if any(s.id == "Beta" for s in stage) else ()
        return AgentPlan("Enable the requested options", stage,
                         reusable=True, completion_conditions=completion)


def run(env, *, planner=None, progress=None, success=None, **kwargs):
    backend, planner, trace = Backend(env), planner or Planner(), JsonlTrace()
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), trace=trace, progress=progress,
        config=AgentConfig(allow_actions=True, max_steps=6, **kwargs),
    ).run("Enable Alpha and Beta", success=success)
    return result, backend, planner, trace


def test_receipt_bound_snapshot_is_actual_step_return_and_single_use(clock, snapshot):
    env, backend = Environment(snapshot), None
    backend = Backend(env)
    candidate, receipt = execute(backend)
    assert backend.verify(candidate, receipt).passed
    cached = backend.take_verified_post_snapshot(candidate, receipt)
    assert cached == (env.snapshot, clock.now)
    assert cached[0] is env.snapshot and cached[0] is not snapshot
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert env.reads == backend.fresh_observe_calls == 1


@pytest.mark.parametrize("alteration", ["copy", "id", "output", "candidate", "backend"])
def test_forged_or_mutated_receipt_cannot_supply_a_snapshot(clock, snapshot, alteration):
    env, backend = Environment(snapshot), None
    backend = Backend(env)
    candidate, receipt = execute(backend)
    if alteration == "copy":
        receipt = replace(receipt)
    elif alteration == "id":
        receipt = replace(receipt, receipt_id="forged")
    elif alteration == "output":
        receipt.output["environment_steps"] = 999
    elif alteration == "candidate":
        candidate = replace(candidate, description="altered")
    else:
        backend = Backend(env)
    assert backend.take_verified_post_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("invalidate", ["observe", "failed_observe", "input", "execute", "pending", "close"])
def test_any_new_observation_or_input_invalidates_previous_response(clock, snapshot, invalidate):
    env, backend = Environment(snapshot), None
    backend = Backend(env)
    candidate, receipt = execute(backend)
    if invalidate == "observe":
        backend.observe()
    elif invalidate == "failed_observe":
        def fail(_):
            raise RuntimeError("read failed")
        env.on_read = fail
        with pytest.raises(RuntimeError):
            backend.observe()
    elif invalidate == "input":
        backend._step("CLICK", {"x": 360, "y": 90})
    elif invalidate == "execute":
        # Replayed candidates fail before sending but must still discard old evidence.
        assert not backend.execute(candidate, "replayed", "replayed").success
    elif invalidate == "pending":
        backend.verify_pending(candidate, receipt)
    else:
        backend.close()
    assert backend.take_verified_post_snapshot(candidate, receipt) is None


@pytest.mark.parametrize("age,eligible", [(0.99, True), (1.01, False), (-0.1, False)])
def test_step_response_local_age_is_bounded(clock, snapshot, age, eligible):
    backend = Backend(Environment(snapshot))
    candidate, receipt = execute(backend)
    clock.advance(age)
    assert (backend.take_verified_post_snapshot(candidate, receipt) is not None) is eligible


def test_response_age_includes_local_snapshot_processing(clock, snapshot):
    class SlowBackend(Backend):
        def _snapshot(self, observation):
            if self.env.actions:
                clock.advance(1.01)
            return observation
    backend = SlowBackend(Environment(snapshot))
    candidate, receipt = execute(backend)
    assert backend.verify(candidate, receipt).passed
    assert backend.take_verified_post_snapshot(candidate, receipt) is None


def test_next_execution_publishes_only_its_own_final_response(clock, snapshot):
    backend = Backend(Environment(snapshot))
    first, first_receipt = execute(backend)
    second, second_receipt = execute(backend, check_step("Beta"))
    assert backend.verify(first, first_receipt).passed
    cached = backend.take_verified_post_snapshot(second, second_receipt)
    assert all(c.checked for c in cached[0].controls)
    assert backend.take_verified_post_snapshot(first, first_receipt) is None
    assert backend.env.reads == 2


def test_fill_keeps_both_focus_reads_and_reuses_only_final_typing_response(clock, snapshot):
    field = AgentControl("field", "Document", "input", role="textbox", app="Settings",
                         editable=True, focused=True, box=(100, 100, 400, 400))
    env = Environment(replace(snapshot, controls=(field,)))
    def typing(env, action):
        if action["action_type"] == "TYPING":
            env.snapshot = replace(env.snapshot, controls=(
                replace(field, value=action["parameters"]["text"]),
            ))
        return env.snapshot
    env.on_step = typing
    target = AgentTarget(name="Document", app="Settings")
    step = AgentStep("fill", "fill", target=target, value="requested text",
                     postconditions=(AgentCondition("value_equals", target, "requested text"),))
    backend = Backend(env)
    candidate, receipt = execute(backend, step)
    cached = backend.take_verified_post_snapshot(candidate, receipt)
    assert cached[0].controls[0].value == "requested text"
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert env.reads == backend.fresh_observe_calls == 3


@pytest.mark.parametrize("kind", ["unconfirmed", "failed", "partial", "wait", "noop"])
def test_failed_unconfirmed_or_no_input_actions_have_no_reusable_response(clock, snapshot, kind):
    env, backend = Environment(snapshot), None
    backend = Backend(env)
    step = check_step()
    if kind == "unconfirmed":
        env.on_step = lambda env, action: env.snapshot
    elif kind == "failed":
        def fail(env, action):
            raise RuntimeError("input failed")
        env.on_step = fail
    elif kind == "partial":
        step = AgentStep("fill", "visual_fill", box=(100, 100, 200, 200), value="test", label="Field")
        def fail_later(env, action):
            if len(env.actions) > 1:
                raise RuntimeError("second input failed")
            return env.snapshot
        env.on_step = fail_later
    elif kind == "wait":
        step = AgentStep("wait", "wait", value="250")
    else:
        env.snapshot = replace(snapshot, controls=(replace(snapshot.controls[0], checked=True),
                                                  snapshot.controls[1]))
    candidate, receipt = execute(backend, step)
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    if kind in {"failed", "partial"}:
        assert not receipt.success
    if kind in {"wait", "noop"}:
        assert receipt.success and receipt.output["environment_steps"] == 0


@pytest.mark.parametrize("mode,expected_reads", [("hybrid", 5), ("vlm", 6)])
def test_both_modes_save_one_decision_read_and_keep_completion_fresh(clock, snapshot, mode, expected_reads):
    baseline, baseline_backend, _, _ = run(
        Environment(snapshot), mode=mode, reuse_verified_post_snapshot=False,
    )
    result, backend, planner, trace = run(Environment(snapshot), mode=mode)
    assert result.status == baseline.status == "unverified"
    assert result.steps == baseline.steps == 2 and result.plan_completion_claims == 1
    assert result.post_snapshot_reuses == 1 and baseline.post_snapshot_reuses == 0
    assert result.fresh_observe_calls == expected_reads == backend.env.reads
    assert baseline.fresh_observe_calls == expected_reads + 1 == baseline_backend.env.reads
    assert result.runner_observe_calls == baseline.runner_observe_calls - 1
    assert backend.take_calls == 1 and len(backend.taken) == 1
    assert result.planner_calls == baseline.planner_calls == (1 if mode == "hybrid" else 2)
    assert len([e for e in trace.events if e["kind"] == "agent_observation_reuse"]) == 1
    if mode == "vlm":
        assert planner.snapshots[1] is backend.taken[0]


def test_independent_success_callback_always_gets_fresh_observations(clock, snapshot):
    result, backend, _, _ = run(
        Environment(snapshot), success=lambda state: all(c.checked for c in state.controls),
    )
    assert result.success and result.steps == 2
    assert result.post_snapshot_reuses == backend.take_calls == 0
    assert result.fresh_observe_calls == 6 and result.runner_observe_calls == 4


def test_post_vlm_observation_is_fresh_even_when_planner_used_a_cached_decision(clock, snapshot):
    env = Environment(snapshot)
    class MovingPlanner(Planner):
        def plan(self, goal, state, history, *, max_actions):
            result = super().plan(goal, state, history, max_actions=max_actions)
            if len(self.snapshots) == 2:
                moved = replace(env.snapshot.controls[1], box=(700, 500, 800, 600))
                env.snapshot = replace(env.snapshot, controls=(env.snapshot.controls[0], moved))
            return result
    result, backend, _, _ = run(env, mode="vlm", planner=MovingPlanner())
    assert result.post_snapshot_reuses == 1 and result.plan_completion_claims == 1
    assert env.actions[-1]["parameters"]["x"] == 600
    assert env.actions[-1]["parameters"]["y"] == 330
    assert result.fresh_observe_calls == backend.env.reads == 6


def test_empty_goal_declaration_after_cached_planning_still_checks_fresh(clock, snapshot):
    env = Environment(snapshot)
    class CompletionPlanner:
        calls = 0

        def plan(self, goal, state, history, *, max_actions):
            self.calls += 1
            if self.calls == 1:
                return AgentPlan("Enable Alpha", (check_step(),), reusable=True)
            env.snapshot = snapshot  # The effect was transient while the model was thinking.
            return AgentPlan("Claim completion", (), done=True, reusable=True,
                             completion_conditions=(checked("Alpha"),))
    result, backend, _, _ = run(env, planner=CompletionPlanner(), max_plans=2)
    assert result.post_snapshot_reuses == backend.take_calls == 1
    assert result.plan_completion_claims == 0 and result.status == "max_plans"
    assert len(env.actions) == 1


def test_transient_effect_cannot_be_used_as_final_goal_completion(clock, snapshot):
    env = Environment(snapshot)
    def regress_after_last_input(env):
        if len(env.actions) == 2:
            env.snapshot = replace(env.snapshot, controls=(env.snapshot.controls[0],
                                                          replace(env.snapshot.controls[1], checked=False)))
    env.on_read = regress_after_last_input
    result, backend, _, _ = run(env, max_plans=1)
    assert result.post_snapshot_reuses == 1
    assert result.plan_completion_claims == 0 and result.status == "max_plans"
    assert backend.take_calls == 1 and len(env.actions) == 2


def test_cached_decision_rejects_moved_target_then_recompiles_from_fresh_state(clock, snapshot):
    env = Environment(snapshot)
    def move_after_first_input(env):
        if len(env.actions) == 1:
            moved = replace(env.snapshot.controls[1], box=(600, 300, 700, 400))
            env.snapshot = replace(env.snapshot, controls=(env.snapshot.controls[0], moved))
    env.on_read = move_after_first_input
    result, backend, _, trace = run(env, max_plans=1)
    assert result.post_snapshot_reuses == 1 and len(env.actions) == 2
    assert result.steps == 3 and result.status == "unverified"
    assert result.stale_recovery_attempts == 1 and result.planner_calls == 1
    assert env.actions[-1]["parameters"] == {"x": 520, "y": 210, "button": "left", "num_clicks": 1}
    verdicts = [e["payload"]["verified"] for e in trace.events if e["kind"] == "agent_step"]
    assert verdicts == [True, False, True]
    refused = [e["payload"] for e in trace.events if e["kind"] == "receipt"][1]
    assert not refused["success"] and refused["output"]["environment_steps"] == 0
    assert backend.fresh_observe_calls == env.reads


def test_slow_progress_callback_expires_the_optional_snapshot(clock, snapshot):
    def progress(event):
        if event["event"] == "agent_step":
            clock.advance(1.01)
    result, backend, _, _ = run(Environment(snapshot), progress=progress)
    assert result.post_snapshot_reuses == 0 and result.fresh_observe_calls == 6
    assert backend.take_calls == 1


@pytest.mark.parametrize("constraint", ["episode", "plan"])
def test_cached_observation_never_waives_episode_or_plan_deadline(clock, snapshot, constraint):
    def progress(event):
        if event["event"] == "agent_step":
            clock.advance(0.5)
    limits = {"timeout_s": 0.5} if constraint == "episode" else {"plan_ttl_s": 0.5, "max_plans": 1}
    result, backend, _, _ = run(Environment(snapshot), progress=progress, **limits)
    assert len(backend.env.actions) == 1
    assert result.status == ("timeout" if constraint == "episode" else "max_plans")


def test_pending_effect_recovery_observes_fresh_and_does_not_publish_cache(clock, snapshot):
    env = Environment(snapshot)
    def delayed(env, action):
        previous = env.snapshot
        env.snapshot = replace(previous, controls=(replace(previous.controls[0], checked=True),
                                                   previous.controls[1]))
        return previous
    env.on_step = delayed
    class PendingPlanner:
        def plan(self, goal, snapshot, history, *, max_actions):
            return AgentPlan("Enable Alpha", (check_step(recovery="observe"),), reusable=True)
    result, backend, _, _ = run(env, planner=PendingPlanner(), max_plans=1)
    assert result.steps == 1 and result.recovery_observations == 1
    assert result.post_snapshot_reuses == backend.take_calls == 0
    assert len(env.actions) == 1 and env.reads == 5


def test_shell_progress_while_waiting_for_new_app_never_publishes_cache(clock, snapshot):
    launcher = AgentControl("launcher", "Editor launcher", "button", role="button", app="Shell",
                            box=(100, 100, 200, 200))
    before = replace(snapshot, controls=(launcher,))
    shell_change = replace(before, controls=(launcher, replace(
        launcher, ref="spinner", label="Starting", box=(300, 100, 400, 200),
    )))
    env = Environment(before)
    env.on_step = lambda env, action: shell_change
    step = AgentStep("launch", "click", label="Reach the editor", navigation=True,
                     target=AgentTarget(app="Shell"), recovery="observe",
                     postconditions=(AgentCondition("exists", AgentTarget(role="frame", app="Editor")),))
    backend = Backend(env)
    candidate, receipt = execute(backend, step)
    verified = backend.verify(candidate, receipt)
    assert verified.passed and verified.details["application_wait"]["eligible"]
    assert backend.take_verified_post_snapshot(candidate, receipt) is None


def test_pending_navigation_handoff_is_accepted_without_reusing_pending_snapshot(clock, snapshot):
    launcher = AgentControl("launcher", "Editor launcher", "button", role="button", app="Shell",
                            box=(100, 100, 200, 200))
    before = replace(snapshot, controls=(launcher,))
    frame = AgentControl("frame", "Editor", "container", role="frame", app="Editor",
                         window="Editor", active=True)
    save = AgentControl("save", "Save", "button", role="button", app="Editor", window="Editor",
                        active=True, box=(400, 100, 500, 200))
    appeared = replace(before, controls=(launcher, frame, save))
    env = Environment(before)
    def launch(env, action):
        env.snapshot = appeared
        return before
    env.on_step = launch
    class LaunchPlanner:
        def plan(self, goal, state, history, *, max_actions):
            condition = AgentCondition("exists", AgentTarget(name="Field", app="Editor"))
            step = AgentStep("launch", "click", label="Reach the field", navigation=True,
                             target=AgentTarget(app="Shell"), recovery="observe",
                             postconditions=(condition,))
            return AgentPlan("Reach field", (step,), reusable=True)
    def settled(event):
        if event["event"] == "agent_step" and event["verified"]:
            field = AgentControl("field", "Field", "input", role="textbox", app="Editor",
                                 editable=True, active=True, window="Editor", box=(500, 200, 600, 300))
            env.snapshot = replace(appeared, controls=(*appeared.controls, field))
    result, backend, _, trace = run(env, planner=LaunchPlanner(), progress=settled, max_plans=1)
    assert result.navigation_steps == 1 and result.recovery_observations == 1
    assert result.post_snapshot_reuses == backend.take_calls == 0
    assert len(env.actions) == 1 and result.condition_skips == 1
    assert any(e["kind"] == "agent_pending_effect" and e["payload"]["passed"] for e in trace.events)


def test_counter_reports_only_this_run_and_unknown_backend_has_no_invented_count(clock, snapshot):
    env, backend = Environment(snapshot), None
    backend = Backend(env)
    backend.observe()  # Driver/bootstrap observation is outside the agent episode.
    result = AgentRunner(backend, Planner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Enable Alpha and Beta")
    assert result.fresh_observe_calls == env.reads - 1 == 5
    class UnreportedBackend:
        observe = backend.observe
        compile = backend.compile
        execute = backend.execute
        verify = backend.verify
    result = AgentRunner(UnreportedBackend(), Planner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Options already set")
    assert result.fresh_observe_calls is None and result.runner_observe_calls > 0


def test_navigation_cycle_does_not_offer_its_response_for_reuse(clock, snapshot):
    button = AgentControl("toggle", "Configure", "button", role="button", app="Settings",
                          box=(100, 100, 200, 200))
    extra = replace(button, ref="new", label="Other", box=(300, 100, 400, 200))
    before = replace(snapshot, controls=(button,))
    expanded = replace(snapshot, controls=(button, extra))
    env = Environment(before)
    def toggle(env, action):
        env.snapshot = expanded if len(env.actions) % 2 else before
        return env.snapshot
    env.on_step = toggle
    class NavigationPlanner:
        def plan(self, goal, snapshot, history, *, max_actions):
            step = AgentStep("navigate", "click", target=AgentTarget(app="Settings"),
                             navigation=True, label="Find the requested setting",
                             postconditions=(AgentCondition("exists", AgentTarget(name="Missing")),))
            return AgentPlan("Find setting", (step,), reusable=True)
    result, backend, _, trace = run(env, planner=NavigationPlanner(), max_plans=1)
    assert result.post_snapshot_reuses == backend.take_calls == 1
    assert len(env.actions) == 2 and result.status == "max_plans"
    recovery = [e["payload"] for e in trace.events if e["kind"] == "agent_recovery"]
    assert any(event.get("reason") == "navigation_cycle" for event in recovery)


def test_configuration_rejects_non_boolean_switch():
    assert AgentConfig().reuse_verified_post_snapshot is True
    with pytest.raises(ValueError, match="reuse_verified_post_snapshot"):
        AgentConfig(reuse_verified_post_snapshot=1)
