"""Receipt-bound launcher handoff, replayed without a VM or live model."""

import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentControl, AgentPlan, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import navigation_handoff_app, navigation_scope_controls
from cua_jev.osworld_backend import OSWorldBackend, _navigation_scope, screenshot_image
from cua_jev.policy import JevPolicy


@pytest.fixture
def recorded():
    data = json.loads((Path(__file__).parent / "fixtures" / "osworld_navigation_handoff_v11.json").read_text(
        encoding="utf-8"
    ))
    buffer = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(buffer, "PNG")
    image = screenshot_image(buffer.getvalue())
    snapshots = []
    for key in ("before", "after"):
        controls = []
        for raw in data[key]["controls"]:
            raw = dict(raw)
            for field in ("box", "ancestors", "options"):
                if raw[field] is not None:
                    raw[field] = tuple(raw[field])
            controls.append(AgentControl(**raw))
        snapshots.append(AgentSnapshot("osworld://desktop", data[key]["title"], "", tuple(controls), image))
    return tuple(snapshots)


def step_for(snapshot):
    launcher = next(c for c in snapshot.controls if c.tag == "button")
    return AgentStep(
        "navigate", "click", target=AgentTarget(app=launcher.app), navigation=True,
        label="Reach the folder operations", recovery="observe",
        postconditions=(AgentCondition("exists", AgentTarget(name="Rename")),),
    )


class Playback:
    def __init__(self, before, after):
        self.current, self.after = before, after
        self.actions = []
        self.failure = False

    def _get_obs(self):
        return {"snapshot": self.current}

    def step(self, action, pause=0):
        self.actions.append(action)
        if self.failure:
            raise RuntimeError("input rejected")
        self.current = self.after
        return self._get_obs(), "PRIVATE REWARD", True, {}


def backend_for(monkeypatch, before, after):
    env = Playback(before, after)
    backend = OSWorldBackend(env, pause=0)
    # Replay public controls extracted by the real accessibility parser. The
    # backend's candidate binding, input preflight and effect verifier stay real.
    monkeypatch.setattr(backend, "_snapshot", lambda observation: observation["snapshot"])
    return backend, env


def execute_launcher(backend, step, before):
    candidate = backend.compile(step, before, before)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    return candidate, receipt


def test_recorded_new_app_replaces_shell_scope_and_justifies_real_progress(recorded, monkeypatch):
    before, after = recorded
    step = step_for(before)
    assert _navigation_scope(before, step) == _navigation_scope(after, step)
    backend, env = backend_for(monkeypatch, before, after)
    candidate, receipt = execute_launcher(backend, step, before)
    verification = backend.verify(candidate, receipt)
    assert receipt.success and verification.passed
    assert verification.details["navigation_handoff"]
    assert verification.details["accessibility_progress"]
    assert not verification.details["postconditions_met"]
    candidates = backend.compile(step, before, after)
    assert {c.arguments["bound_control"]["label"] for c in candidates} == {"Recent", "Desktop"}
    assert all(c.arguments["bound_control"]["app"] == "org.gnome.Nautilus" for c in candidates)
    assert all(c.arguments["navigation_app"] == "org.gnome.Nautilus" for c in candidates)
    assert len(env.actions) == 1


def test_observe_or_compile_alone_never_authorizes_new_app(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    step = step_for(before)
    backend.compile(step, before, before)
    env.current = after
    assert backend.observe() is after
    offered = backend.compile(step, before, after)
    assert {c.arguments["bound_control"]["app"] for c in offered} == {step.target.app}
    assert env.actions == []


def test_failed_launcher_receipt_cannot_authorize_handoff(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    env.failure = True
    step = step_for(before)
    candidate, receipt = execute_launcher(backend, step, before)
    assert not receipt.success
    env.current = after
    assert not backend.verify_pending(candidate, receipt).passed
    offered = backend.compile(step, before, after)
    assert {c.arguments["bound_control"]["app"] for c in offered} == {step.target.app}


def test_delayed_receipt_observation_can_handoff_without_repeating_input(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, before)
    step = step_for(before)
    candidate, receipt = execute_launcher(backend, step, before)
    assert not backend.verify(candidate, receipt).passed
    env.current = after
    verification = backend.verify_pending(candidate, receipt)
    assert verification.passed and verification.details["navigation_handoff"]
    assert len(backend.compile(step, before, after)) == 2 and len(env.actions) == 1


@pytest.mark.parametrize("change", ["background", "existing", "ambiguous_app", "ambiguous_window",
                                   "incomplete", "no_legal_control", "no_window", "named_launcher_window",
                                   "narrow_scope", "menu_not_launcher"])
def test_handoff_rejects_unproven_or_ambiguous_application_transition(recorded, change):
    before, after = recorded
    step = step_for(before)
    launcher = next(c for c in before.controls if c.tag == "button")
    arrived = tuple(c for c in after.controls if c.app != launcher.app)
    if change == "background":
        after = replace(after, controls=tuple(replace(c, active=False) if c in arrived else c
                                              for c in after.controls))
    elif change == "existing":
        # A previously visible but inactive application becoming active is
        # intentionally conservative: it is not a newly observed application.
        before = replace(before, controls=before.controls + tuple(replace(c, active=False) for c in arrived))
    elif change == "ambiguous_app":
        after = replace(after, controls=after.controls + tuple(
            replace(c, app="Other arrival") for c in arrived
        ))
    elif change == "ambiguous_window":
        root = next(c for c in arrived if c.role == "frame")
        after = replace(after, controls=after.controls + (replace(root, ref="duplicate"),))
    elif change == "incomplete":
        after = replace(after, a11y_complete=False)
    elif change == "no_legal_control":
        after = replace(after, controls=tuple(replace(c, disabled=True) if c in arrived else c
                                              for c in after.controls))
    elif change == "no_window":
        after = replace(after, controls=tuple(c for c in after.controls if c.role != "frame"))
    elif change == "named_launcher_window":
        launcher = replace(launcher, window="Existing app window")
        before = replace(before, controls=(launcher,))
    elif change == "narrow_scope":
        step = replace(step, target=replace(step.target, window="Specific window"))
    elif change == "menu_not_launcher":
        launcher = replace(launcher, role="menuitem")
        before = replace(before, controls=(launcher,))
    assert navigation_handoff_app(before, after, step.target, launcher) == ""


@pytest.mark.parametrize("button,click_count", [("right", 1), ("left", 2)])
def test_nonlaunch_mouse_input_cannot_expand_scope(recorded, monkeypatch, button, click_count):
    before, after = recorded
    backend, _ = backend_for(monkeypatch, before, after)
    step = replace(step_for(before), button=button, click_count=click_count)
    _, receipt = execute_launcher(backend, step, before)
    assert receipt.success and not receipt.output["navigation_handoff"]
    assert not receipt.output["accessibility_progress"]


def test_handoff_follows_one_active_dialog_but_not_other_apps_or_inactive_windows(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    step = step_for(before)
    execute_launcher(backend, step, before)
    app = "org.gnome.Nautilus"
    root = next(c for c in after.controls if c.app == app and c.role == "frame")
    button = next(c for c in after.controls if c.label == "Desktop")
    dialog = (replace(root, ref="dialog", role="dialog", window="Rename", label="Rename"),
              replace(button, ref="confirm", label="Confirm", window="Rename"))
    current = replace(after, controls=tuple(replace(c, active=False) if c.app == app else c
                                            for c in after.controls) + dialog
                      + tuple(replace(c, app=app + ".unrelated") for c in dialog))
    env.current = current
    offered = backend.compile(step, before, current)
    assert len(offered) == 1 and offered[0].arguments["bound_control"]["label"] == "Confirm"
    assert offered[0].arguments["bound_control"]["app"] == app
    assert navigation_scope_controls(current, step.target, handoff_app=app) == dialog


def test_handoff_never_chains_to_a_second_application(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    step = step_for(before)
    execute_launcher(backend, step, before)
    selected = backend.compile(step, before, after)[0]
    new_app = tuple(replace(c, app="Second app") for c in after.controls if c.app != step.target.app)
    env.after = replace(after, controls=after.controls + new_app)
    receipt = backend.execute(selected, "obs", "decision")
    assert receipt.success and not receipt.output["navigation_handoff"]
    offered = backend.compile(step, before, env.current)
    assert {c.arguments["bound_control"]["app"] for c in offered} == {"org.gnome.Nautilus"}


def test_handoff_resets_for_new_plan_and_old_pending_receipt_cannot_restore_it(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    step = step_for(before)
    candidate, receipt = execute_launcher(backend, step, before)
    new_planned = replace(after)
    new_step = replace(step)
    offered = backend.compile(new_step, new_planned, after)
    assert {c.arguments["bound_control"]["app"] for c in offered} == {step.target.app}
    env.current = after
    backend.verify_pending(candidate, receipt)
    offered = backend.compile(new_step, new_planned, after)
    assert {c.arguments["bound_control"]["app"] for c in offered} == {step.target.app}


def test_handoff_candidate_keeps_full_freshness_checks(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    step = step_for(before)
    execute_launcher(backend, step, before)
    candidate = backend.compile(step, before, after)[0]
    target_ref = candidate.arguments["bound_control"]["ref"]
    env.current = replace(after, controls=tuple(
        replace(c, box=(10, 20, 40, 50)) if c.ref == target_ref else c for c in after.controls
    ))
    assert not backend.execute(candidate, "obs", "decision").success
    assert len(env.actions) == 1


def test_one_plan_and_two_jev_selections_cross_app_boundary(recorded, monkeypatch):
    before, after = recorded
    backend, env = backend_for(monkeypatch, before, after)
    step = step_for(before)
    calls, requests = [], []
    finished = replace(after, controls=after.controls + (
        AgentControl("done", "Rename", "text", role="label", app="org.gnome.Nautilus"),
    ))

    class Planner:
        def plan(self, *args, **kwargs):
            calls.append(kwargs["max_actions"])
            return AgentPlan("Reach folder operations", (step,), reusable=True)

    def answer(request):
        body = json.loads(request.content)
        requests.append(body)
        candidates = body["state"]["available_actions"]
        name = "Files" if not env.actions else "Desktop"
        selected = next(c for c in candidates if f"click {name} " in c["description"])
        if env.actions:
            env.after = finished
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected["id"], "confidence": 1,
            "probabilities": {c["id"]: float(c is selected) for c in candidates},
        }}})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = AgentRunner(
            backend, Planner(), policy=JevPolicy(api_key="test", client=client, retries=0),
            config=AgentConfig(allow_actions=True),
        ).run("Reach folder operations", success=lambda _: env.current is finished)
    assert result.success and calls == [6]
    assert result.steps == 2 and result.plan_reuses == 1 and result.navigation_subgoals == 1
    assert result.candidate_counts == (2, 2) and result.jev_calls == 2
    assert len(env.actions) == len(requests) == 2
    assert all("org.gnome.Nautilus" in c["description"] for c in requests[1]["state"]["available_actions"])
