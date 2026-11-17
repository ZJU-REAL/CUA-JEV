"""Receipt-bound application startup waits do not replay launchers or waive TTLs."""

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
def snapshots():
    png = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(png, format="PNG")
    image = screenshot_image(png.getvalue())
    launcher = AgentControl("launcher", "Editor launcher", "button", role="button", app="Shell",
                            box=(10, 100, 70, 160))

    def snapshot(controls):
        return AgentSnapshot("osworld://desktop", "Desktop", "", tuple(controls), image=image)

    before = snapshot((launcher,))
    frame = AgentControl("frame", "Editor", "container", role="frame", app="Editor",
                         window="Editor", active=True)
    save = AgentControl("save", "Save", "button", role="button", app="Editor",
                        window="Editor", active=True, box=(100, 100, 200, 200))
    return before, snapshot((launcher, frame, save))


class Environment:
    def __init__(self, before, states, clock, *, latency=0, sent=True, step_state=None):
        self.before, self.states, self.clock = before, list(states), clock
        self.latency, self.sent = latency, sent
        self.step_state = step_state
        self.inputs, self.pending_reads = [], 0

    def _get_obs(self):
        if not self.inputs:
            return self.before
        self.clock.advance(self.latency)
        index = min(self.pending_reads, len(self.states) - 1)
        self.pending_reads += 1
        return self.states[index]

    def step(self, action, pause):
        if not self.sent:
            raise RuntimeError("failed transport")
        self.inputs.append(action)
        return self.step_state or self.before, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def launch_step(*, navigation=False, expected_app="Editor", role="frame"):
    return AgentStep(
        "launch", "click", label="Reach the editor",
        target=AgentTarget(app="Shell") if navigation else AgentTarget(name="Editor launcher", app="Shell"),
        navigation=navigation, recovery="observe",
        postconditions=(AgentCondition("exists", AgentTarget(role=role, app=expected_app)),),
    )


class Planner:
    def __init__(self, step, *, repeat=False):
        self.step, self.repeat, self.calls = step, repeat, []

    def plan(self, goal, snapshot, history, *, max_actions):
        self.calls.append({"history": list(history), "max_actions": max_actions})
        if len(self.calls) > 1 and not self.repeat:
            return AgentPlan("Inspect the observed application", (), done=True)
        return AgentPlan("Open the application", (self.step,), reusable=True)


def run(env, step, *, repeat=False, **config):
    backend, planner, trace = Backend(env, pause=0), Planner(step, repeat=repeat), JsonlTrace()
    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), trace=trace,
        config=AgentConfig(allow_actions=True, plan_ttl_s=120, **config),
    ).run("Edit the requested document")
    return result, planner, trace, backend


def events(trace, kind):
    return [event["payload"] for event in trace.events if event["kind"] == kind]


def test_effect_returning_exactly_at_lease_deadline_does_not_revive_old_plan(clock, snapshots):
    before, appeared = snapshots
    env = Environment(before, [before] * 14 + [appeared], clock, latency=3)
    result, _planner, trace, _backend = run(env, launch_step(), max_steps=1)
    assert len(env.inputs) == 1 and result.recovery_observations == 15
    assert not events(trace, "agent_step")[0]["verified"]
    wait = events(trace, "agent_application_wait")[0]
    # Fifteen 1s intervals plus 3s observations reach the scheduling boundary.
    assert wait["elapsed_ms"] == 60000 and wait["reason"] == "application_wait_timeout"


def test_delayed_in_budget_application_is_verified_once(clock, snapshots):
    before, appeared = snapshots
    env = Environment(before, [before] * 14 + [appeared], clock, latency=2.9)
    result, _planner, trace, _backend = run(env, launch_step(), max_steps=1)
    assert len(env.inputs) == 1 and result.recovery_observations == 15
    assert events(trace, "agent_step")[0]["verified"]
    assert events(trace, "agent_application_wait")[0]["reason"] == "effect_confirmed"


@pytest.mark.parametrize("kind", ["same_app", "missing_app", "non_launcher", "failed_input"])
def test_only_successful_cross_app_launcher_extends_normal_recovery(clock, snapshots, kind):
    before, _appeared = snapshots
    step = launch_step(expected_app="Shell" if kind == "same_app" else "Editor")
    if kind == "same_app":
        step = replace(step, postconditions=(
            AgentCondition("exists", AgentTarget(name="Missing", app="Shell")),
        ))
    if kind == "missing_app":
        step = replace(step, postconditions=(AgentCondition("exists", AgentTarget(name="Missing")),))
    if kind == "non_launcher":
        before = replace(before, controls=(replace(before.controls[0], window="Existing"),))
    env = Environment(before, [before], clock, sent=kind != "failed_input")
    result, _planner, trace, _backend = run(env, step, max_steps=1)
    assert result.recovery_observations == (0 if kind == "failed_input" else 2)
    assert not events(trace, "agent_application_wait")
    assert len(env.inputs) == (0 if kind == "failed_input" else 1)


def test_wrong_window_role_stops_on_unique_new_app_and_repairs_predicate(clock, snapshots):
    before, appeared = snapshots
    env = Environment(before, [before, before, appeared], clock)
    result, planner, trace, backend = run(env, launch_step(role="window"))
    assert result.recovery_observations == 3 and len(env.inputs) == 1
    feedback = planner.calls[1]["history"][-1]
    assert planner.calls[1]["max_actions"] == 1
    assert feedback["reason"] == "application_appeared_predicate_unconfirmed"
    assert feedback["application_appearance"] == {
        "observed_app": "Editor", "observed_window": "Editor", "observed_role": "frame",
    }
    assert not events(trace, "agent_step")[0]["verified"]
    assert not backend._navigation_apps


def test_ambiguous_new_active_windows_end_wait_without_success(clock, snapshots):
    before, appeared = snapshots
    appeared = replace(appeared, controls=appeared.controls + (
        replace(appeared.controls[1], ref="second-frame", window="Editor 2", label="Editor 2"),
    ))
    env = Environment(before, [appeared], clock)
    result, planner, _trace, _backend = run(env, launch_step(role="window"))
    assert result.recovery_observations == 1 and len(env.inputs) == 1
    assert planner.calls[1]["history"][-1]["reason"] == "application_appearance_ambiguous"
    assert planner.calls[1]["history"][-1]["application_appearance"] == {}


def test_navigation_keeps_existing_valid_handoff_instead_of_forcing_replanning(clock, snapshots):
    before, appeared = snapshots
    env = Environment(before, [before, before, appeared], clock)
    step = launch_step(navigation=True, role="window")
    result, _planner, trace, backend = run(env, step, max_steps=1)
    assert result.navigation_steps == 1 and result.navigation_subgoals == 0
    assert events(trace, "agent_step")[0]["verified"]
    assert backend._navigation_apps[id(step)][1] == "Editor"
    assert events(trace, "agent_application_wait")[0]["reason"] == "effect_confirmed"


def shell_launch_progress(before):
    """Minimal v16 saved-state structure: Shell gains a menuitem before the app.

    Source step-001-a11y.xml SHA256:
    0237cb780f8d4565fc7706ccc4f712f94dd7dad711db00a7bcfe880fb23efdd8.
    Labels are generic; the regression is a new executable shell affordance,
    with no new application/frame yet, not a benchmark-specific identity.
    """
    return replace(before, controls=before.controls + (
        AgentControl("shell-app-menu", "Editor", "button", role="menuitem", app="Shell",
                     box=(100, 10, 200, 40)),
    ))


@pytest.mark.parametrize("initial_progress", [False, True])
def test_shell_progress_does_not_release_application_wait_before_exact_target(
    clock, snapshots, initial_progress,
):
    before, appeared = snapshots
    progress = shell_launch_progress(before)
    env = Environment(before, [progress, progress, appeared], clock,
                      step_state=progress if initial_progress else None)
    result, _planner, trace, _backend = run(env, launch_step(navigation=True), max_steps=1)
    # Both the first receipt and later pending receipts can report valid shell
    # navigation progress. Neither is evidence that the application arrived.
    assert events(trace, "verification")[0]["passed"] is initial_progress
    assert [row["passed"] for row in events(trace, "agent_pending_effect")] == [False, False, True]
    assert result.recovery_observations == 3 and env.pending_reads == 3
    assert len(env.inputs) == result.steps == 1
    assert result.navigation_steps == result.navigation_subgoals == 1
    assert events(trace, "agent_step")[0]["subgoal_complete"]
    assert events(trace, "agent_application_wait")[0]["attempts"] == 3


def test_shell_progress_preserves_twenty_observation_limit_without_extra_input(clock, snapshots):
    before, _appeared = snapshots
    progress = shell_launch_progress(before)
    env = Environment(before, [progress], clock, step_state=progress)
    result, _planner, trace, _backend = run(env, launch_step(navigation=True), max_steps=1)
    assert len(env.inputs) == result.steps == 1
    assert env.pending_reads == result.recovery_observations == 20
    assert not any(row["passed"] for row in events(trace, "agent_pending_effect"))
    assert not events(trace, "agent_step")[0]["verified"]
    assert events(trace, "agent_application_wait")[0]["reason"] == "observation_budget_exhausted"


def test_shell_progress_does_not_extend_episode_deadline(clock, snapshots):
    before, _appeared = snapshots
    progress = shell_launch_progress(before)
    env = Environment(before, [progress], clock, latency=2, step_state=progress)
    result, _planner, trace, _backend = run(env, launch_step(navigation=True), timeout_s=8)
    assert result.status == "timeout" and len(env.inputs) == 1
    assert result.recovery_observations == 3
    assert events(trace, "agent_application_wait")[0]["reason"] == "episode_timeout"
    assert not events(trace, "agent_step")[0]["verified"]


def test_shell_progress_then_authorized_handoff_keeps_navigation_without_claiming_goal(clock, snapshots):
    before, appeared = snapshots
    env = Environment(before, [shell_launch_progress(before), appeared], clock)
    step = launch_step(navigation=True, role="window")
    result, _planner, trace, backend = run(env, step, max_steps=1)
    assert [row["passed"] for row in events(trace, "agent_pending_effect")] == [False, True]
    assert result.recovery_observations == 2 and len(env.inputs) == 1
    assert result.navigation_steps == 1 and result.navigation_subgoals == 0
    assert backend._navigation_apps[id(step)][1] == "Editor"
    assert events(trace, "agent_step")[0]["verified"]
    assert not events(trace, "agent_step")[0]["subgoal_complete"]


def test_shell_progress_and_ambiguous_arrival_use_recovery_instead_of_relaunching(clock, snapshots):
    before, appeared = snapshots
    progress = shell_launch_progress(before)
    ambiguous = replace(appeared, controls=appeared.controls + (progress.controls[-1],
        replace(appeared.controls[1], ref="other-frame", label="Editor 2", window="Editor 2"),
    ))
    env = Environment(before, [progress, ambiguous], clock)
    result, planner, trace, _backend = run(env, launch_step(navigation=True, role="window"))
    assert result.recovery_observations == 2 and len(env.inputs) == 1
    assert not events(trace, "agent_step")[0]["verified"]
    assert planner.calls[1]["max_actions"] == 1
    assert planner.calls[1]["history"][-1]["reason"] == "application_appearance_ambiguous"
    assert events(trace, "agent_application_wait")[0]["reason"] == "application_appearance_ambiguous"


def test_ordinary_navigation_progress_without_application_wait_remains_verified(clock, snapshots):
    before, _appeared = snapshots
    env = Environment(before, [shell_launch_progress(before)], clock)
    result, _planner, trace, _backend = run(
        env, launch_step(navigation=True, expected_app="Shell"), max_steps=1,
    )
    assert len(env.inputs) == 1 and result.recovery_observations == 1
    assert events(trace, "agent_step")[0]["verified"]
    assert result.navigation_steps == 1 and result.navigation_subgoals == 0
    assert not events(trace, "agent_application_wait")


def test_twenty_observations_and_failed_launches_count_once_each(clock, snapshots):
    before, _appeared = snapshots
    env = Environment(before, [before], clock)
    result, _planner, trace, _backend = run(env, launch_step(), repeat=True)
    assert result.status == "stuck" and result.reason == "consecutive action failures"
    assert len(env.inputs) == result.steps == 3
    assert result.recovery_observations == 60
    assert [row["attempts"] for row in events(trace, "agent_application_wait")] == [20] * 3


@pytest.mark.parametrize("budget", ["episode", "plan", "lease"])
def test_wait_stops_at_original_deadline_without_replaying_input(clock, snapshots, budget):
    before, _appeared = snapshots
    env = Environment(before, [before], clock, latency=2)
    config = {"timeout_s": 8} if budget == "episode" else {}
    if budget == "plan":
        backend, planner, trace = Backend(env, pause=0), Planner(launch_step()), JsonlTrace()
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
                             config=AgentConfig(allow_actions=True, plan_ttl_s=8)).run("Open editor")
    else:
        result, _planner, trace, _backend = run(env, launch_step(), **config)
    wait = events(trace, "agent_application_wait")[0]
    assert wait["reason"] == {"episode": "episode_timeout", "plan": "plan_expired",
                              "lease": "application_wait_timeout"}[budget]
    assert len(env.inputs) == 1
    assert result.status == ("timeout" if budget == "episode" else "unverified")


def test_blocking_observe_returning_after_deadline_is_not_authority_for_more_input(clock, snapshots):
    before, appeared = snapshots
    env = Environment(before, [appeared], clock, latency=70)
    result, _planner, trace, _backend = run(env, launch_step(), timeout_s=10)
    wait = events(trace, "agent_application_wait")[0]
    assert result.status == "timeout" and len(env.inputs) == 1
    assert wait["attempts"] == 1 and wait["elapsed_ms"] == 71000
    assert wait["deadline_overrun_ms"] == 61000 and wait["reason"] == "episode_timeout"
    assert not events(trace, "agent_pending_effect")[0]["passed"]
    assert not events(trace, "agent_step")[0]["verified"]


def test_preflight_failure_does_not_create_application_wait_or_pending_receipt(clock, snapshots):
    before, _appeared = snapshots
    env = Environment(before, [before], clock)
    backend = Backend(env, pause=0)
    action = backend.compile(launch_step(), before, before)[0]
    env.before = replace(before, controls=(replace(before.controls[0], disabled=True),))
    receipt = backend.execute(action, "observation", "decision")
    assert not receipt.success and receipt.output["environment_steps"] == 0
    assert not backend.verify(action, receipt).details.get("application_wait")
    assert not backend.verify_pending(action, receipt).passed
    assert not env.inputs and not env.pending_reads and not backend._launch_wait_until


def test_backend_lease_cannot_be_reused_with_forged_receipt_or_after_twenty_reads(clock, snapshots):
    before, _appeared = snapshots
    env = Environment(before, [before], clock)
    backend = Backend(env, pause=0)
    action = backend.compile(launch_step(), before, before)[0]
    receipt = backend.execute(action, "observation", "decision")
    assert not backend.verify_pending(action, replace(receipt, ended_at=receipt.ended_at + 1)).passed
    assert not backend.verify_pending(replace(action, description="changed"), receipt).passed
    assert env.pending_reads == 0
    for _attempt in range(20):
        assert not backend.verify_pending(action, receipt).passed
    exhausted = backend.verify_pending(action, receipt)
    assert not exhausted.passed and "budget exhausted" in exhausted.details["reason"]
    assert env.pending_reads == 20 and len(env.inputs) == 1
    backend.close()
    assert not backend._launch_wait_until


@pytest.mark.parametrize("change", ["incomplete", "already_present"])
def test_incomplete_or_existing_application_cannot_authorize_long_wait(clock, snapshots, change):
    before, appeared = snapshots
    before = replace(before, a11y_complete=False) if change == "incomplete" else replace(
        before, controls=before.controls + (replace(appeared.controls[1], active=False),),
    )
    env = Environment(before, [before], clock)
    result, _planner, trace, _backend = run(env, launch_step(role="window"), max_steps=1)
    assert len(env.inputs) == 1 and result.recovery_observations == 2
    assert not events(trace, "agent_application_wait")
