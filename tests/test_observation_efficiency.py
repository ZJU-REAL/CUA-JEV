"""Observe fewer times without executing stale actions or accepting missing AX."""

import signal
import time
from types import SimpleNamespace

import pytest
from test_osworld_backend import Environment, png

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentPlan, AgentStep
from cua_jev.observation_budget import AccessibilityBudget
from cua_jev.osworld_backend import OSWorldBackend


def test_ax_timeout_escapes_sdk_retries_and_never_accumulates_requests():
    attempts = []

    def slow_sdk():
        for _ in range(5):
            attempts.append(1)
            try:
                time.sleep(2)
            except Exception:
                continue
        return "<desktop/>"

    controller = SimpleNamespace(get_accessibility_tree=slow_sdk)
    previous = signal.getsignal(signal.SIGALRM)
    budget = AccessibilityBudget(controller, 0.03)
    began = time.perf_counter()
    try:
        assert controller.get_accessibility_tree() is None
        assert time.perf_counter() - began < 0.5
        for _ in range(8):
            assert controller.get_accessibility_tree() is None
        assert len(attempts) == 1
        assert budget.report["timeouts"] == 1 and budget.report["skipped"] == 8
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
        assert signal.getsignal(signal.SIGALRM) == previous
    finally:
        budget.restore()
    assert controller.get_accessibility_tree is slow_sdk


def test_ax_budget_keeps_outer_watchdog_and_restores_after_reader_failure():
    def read():
        raise ValueError("reader failure")

    controller = SimpleNamespace(get_accessibility_tree=read)
    budget = AccessibilityBudget(controller, 0.1)
    try:
        signal.setitimer(signal.ITIMER_REAL, 10)
        assert controller.get_accessibility_tree() is None
        assert 9 < signal.getitimer(signal.ITIMER_REAL)[0] <= 10
        signal.setitimer(signal.ITIMER_REAL, 0)
        with pytest.raises(ValueError, match="reader failure"):
            controller.get_accessibility_tree()
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        budget.restore()


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 61])
def test_invalid_ax_budget_rejected_before_installation(value):
    controller = SimpleNamespace(get_accessibility_tree=lambda: "tree")
    original = controller.get_accessibility_tree
    with pytest.raises(ValueError):
        AccessibilityBudget(controller, value)
    assert controller.get_accessibility_tree is original


def run_immediate(*, defer, change=False, done=False):
    env = Environment()
    backend = OSWorldBackend(env, pause=0)

    class Planner:
        def plan(self, goal, snapshot, history, *, max_actions):
            if change:
                env.frame = png("red")
            if done:
                return AgentPlan("Done", (), done=True)
            return AgentPlan("Click Save", (AgentStep(
                "save", "visual_click", label="Save", box=(100, 100, 300, 300),
            ),))

    try:
        result = AgentRunner(
            backend, Planner(), config=AgentConfig(
                mode="vlm", max_steps=1, max_plans=1, allow_actions=True,
                defer_immediate_plan_refresh=defer,
            ),
        ).run("Click Save")
        return result, env, backend
    finally:
        backend.close()


def test_single_action_deferral_saves_a_full_read_but_preserves_input_and_verdict():
    old, old_env, old_backend = run_immediate(defer=False)
    new, new_env, new_backend = run_immediate(defer=True)
    assert old_env.calls == new_env.calls and len(new_env.calls) == 1
    assert old.status == new.status and old.steps == new.steps == 1
    assert old_backend.fresh_observe_calls == new_backend.fresh_observe_calls + 1
    assert new.post_plan_observations_deferred == 1


def test_deferred_plan_never_executes_after_screen_changes_during_planning():
    result, env, _ = run_immediate(defer=True, change=True)
    assert result.post_plan_observations_deferred == 1
    assert not env.calls and not result.success


def test_done_claim_still_requires_normal_completion_path():
    result, env, _ = run_immediate(defer=True, done=True)
    assert result.post_plan_observations_deferred == 0
    assert not result.success and not env.calls
