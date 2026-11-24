"""Anonymous active-window continuation from observed XML; no VM or live policy."""

import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentPlan, AgentStep, AgentTarget
from cua_jev.desktop_semantics import executable_control, navigation_scope_controls
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import JevPolicy

SHELL = (
    '<desktop><application name="gnome-shell">'
    '<push-button name="Files" screencoord="(0,100)" size="(50,50)"/>'
    '<push-button name="Settings" screencoord="(0,160)" size="(50,50)"/>'
    "</application></desktop>"
)


@pytest.fixture
def recorded():
    data = json.loads(
        (Path(__file__).parent / "fixtures/osworld_anonymous_active_window_v24.json").read_bytes()
    )
    png = io.BytesIO()
    Image.new("RGB", tuple(data["screen_size"]), "white").save(png, "PNG")
    observations = [
        {"screenshot": png.getvalue(), "accessibility_tree": xml}
        for xml in (
            SHELL,
            *(data["frames"][name]["xml"] for name in ("desktop_selected", "context_menu", "rename_editor")),
        )
    ]
    parser = OSWorldBackend(object(), pause=0)
    snapshots = tuple(parser._snapshot(observation) for observation in observations)
    assert all(snapshot.a11y_complete for snapshot in snapshots)
    return observations, snapshots


def navigation():
    return AgentStep(
        "reach",
        "click",
        target=AgentTarget(app="gnome-shell"),
        navigation=True,
        label="Reach the directory name editor",
        recovery="observe",
        native_navigation_gestures=True,
        postconditions=(
            AgentCondition(
                "focused", AgentTarget(role="textbox", app="org.gnome.Nautilus", window="Desktop")
            ),
        ),
    )


def test_observed_popup_is_unique_active_window_and_only_its_sixteen_items_are_offered(recorded):
    _observations, (_shell, selected, popup, editor) = recorded
    scope, app = navigation().target, "org.gnome.Nautilus"
    roots = [control for control in popup.controls if control.role in {"frame", "window", "dialog"}]
    assert [(c.label, c.window, c.active) for c in roots] == [("Desktop", "Desktop", False), ("", "", True)]
    controls = navigation_scope_controls(popup, scope, handoff_app=app)
    executable = [c for c in controls if executable_control(c, "click")]
    assert len(executable) == 16 and all(c.role == "menuitem" and c.window == "" for c in executable)
    assert any(c.label == "Rename\u2026" for c in executable)
    assert not any("todo_list" in c.label for c in controls)
    assert navigation_scope_controls(selected, scope, handoff_app=app)
    assert any(c.editable for c in navigation_scope_controls(editor, scope, handoff_app=app))


@pytest.mark.parametrize(
    "change",
    ["second_active_root", "incomplete", "no_root", "all_background", "inconsistent_named_root", "wrong_app"],
)
def test_missing_ambiguous_or_unproven_window_evidence_offers_nothing(recorded, change):
    _observations, (_shell, _selected, popup, _editor) = recorded
    app = "org.gnome.Nautilus"
    root = next(c for c in popup.controls if c.active and c.role == "window")
    if change == "second_active_root":
        popup = replace(popup, controls=popup.controls + (replace(root, ref="second-root"),))
    elif change == "incomplete":
        popup = replace(popup, a11y_complete=False)
    elif change == "no_root":
        popup = replace(popup, controls=tuple(c for c in popup.controls if c.role != "window"))
    elif change == "all_background":
        popup = replace(popup, controls=tuple(replace(c, active=False) for c in popup.controls))
    elif change == "inconsistent_named_root":
        popup = replace(
            popup, controls=tuple(replace(c, label="Named") if c is root else c for c in popup.controls)
        )
    elif change == "wrong_app":
        app += ".unrelated"
    assert navigation_scope_controls(popup, navigation().target, handoff_app=app) == ()


def test_structurally_inactive_unnamed_sibling_and_other_app_are_excluded(recorded):
    _observations, (_shell, _selected, popup, _editor) = recorded
    # The parser derives active from the containing root, including unnamed
    # roots. Same blank window strings do not turn inactive siblings active.
    xml = (
        '<desktop><application name="App">'
        '<window active="true"><push-button name="Current" screencoord="(0,0)" size="(50,50)"/></window>'
        '<window active="false"><push-button name="Background" screencoord="(50,0)" size="(50,50)"/></window>'
        '</application><application name="App.unrelated"><window active="true">'
        '<push-button name="Foreign" screencoord="(100,0)" size="(50,50)"/></window></application></desktop>'
    )
    backend = OSWorldBackend(object(), pause=0)
    snapshot = backend._snapshot({"screenshot": _observations[0]["screenshot"], "accessibility_tree": xml})
    assert snapshot.a11y_complete
    controls = navigation_scope_controls(snapshot, navigation().target, handoff_app="App")
    assert [c.label for c in controls if executable_control(c, "click")] == ["Current"]
    assert not navigation_scope_controls(
        popup, navigation().target
    )  # No application grant from observe alone.


@pytest.mark.parametrize(
    "target",
    [
        AgentTarget(app="gnome-shell", window="Specific"),
        AgentTarget(app="gnome-shell", ancestor="Specific"),
        AgentTarget(app="gnome-shell", name="Specific"),
        AgentTarget(app="gnome-shell", match="contains"),
        AgentTarget(window="Specific"),
    ],
)
def test_handoff_resolution_cannot_ignore_an_explicit_narrow_scope(recorded, target):
    popup = recorded[1][2]
    assert navigation_scope_controls(popup, target, handoff_app="org.gnome.Nautilus") == ()


class Playback:
    def __init__(self, observations):
        self.observations, self.index, self.actions = observations, 0, []

    def _get_obs(self):
        return self.observations[self.index]

    def step(self, action, pause=0):
        self.actions.append(action)
        self.index += 1
        return self._get_obs(), "PRIVATE reward", False, {}


def test_one_plan_reuses_across_launcher_native_context_menu_and_anonymous_popup(recorded):
    observations, _snapshots = recorded
    env = Playback(observations)
    backend = OSWorldBackend(env, pause=0)
    calls, requests, target_sets = [], [], []
    step = navigation()

    class Planner:
        def plan(self, *_args, **kwargs):
            calls.append(kwargs["max_actions"])
            return AgentPlan(
                "Reach the directory name editor",
                (step,),
                reusable=True,
                completion_conditions=step.postconditions,
            )

    def answer(request):
        body = json.loads(request.content)
        candidates = body["state"]["available_actions"]
        requests.append(body)
        bindings = {c["id"]: backend._issued[c["id"]] for c in candidates}
        target_sets.append([bindings[c["id"]].target for c in candidates])
        wanted = (
            ("Files", "left")
            if env.index == 0
            else (("todo_list_Jan_1", "right") if env.index == 1 else ("Rename\u2026", "left"))
        )
        selected = next(
            c for c in candidates if (bindings[c["id"]].target.label, bindings[c["id"]].button) == wanted
        )
        return httpx.Response(
            200,
            json={
                "answers": {
                    "action": {
                        "choice": selected["id"],
                        "confidence": 1,
                        "probabilities": {c["id"]: float(c is selected) for c in candidates},
                    }
                }
            },
        )

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = AgentRunner(
            backend,
            Planner(),
            policy=JevPolicy(api_key="test", client=client, retries=0),
            config=AgentConfig(allow_actions=True),
        ).run("Reach editor")
    assert calls == [6] and result.planner_calls == 1 and result.steps == 3
    assert result.jev_calls == 3 and result.plan_reuses == 2
    assert result.candidate_counts[0] == 2 and result.candidate_counts[2] == 16
    assert len(env.actions) == len(requests) == 3 and env.index == 3
    assert env.actions[1]["parameters"]["button"] == "right"
    assert not result.success  # No independent evaluator; this is a local replay.
    assert all(c.window == "" and c.active and c.app == "org.gnome.Nautilus" for c in target_sets[2])


def test_property_stage_retains_receipt_handoff_and_binds_real_anonymous_editor(recorded):
    observations, snapshots = recorded
    env = Playback(observations)
    backend = OSWorldBackend(env, pause=0)
    planned = backend.observe()
    step = parse_plan({"intent": "pursue_property", "label": "Rename the requested directory to Reports",
                       "scope": {"app": "gnome-shell"},
                       "edit": {"operation": "fill", "value": "Reports"}},
                      planned, profile="desktop", max_actions=6).steps[0]
    for label, button in (("Files", "left"), ("todo_list_Jan_1", "right"), ("Rename…", "left")):
        current = backend.observe()
        choices = backend.compile(step, planned, current)
        chosen = next(c for c in choices if c.arguments["bound_control"]["label"] == label
                      and c.arguments["button"] == button)
        receipt = backend.execute(chosen, "obs", "decision")
        evidence = backend.verify(chosen, receipt)
        assert evidence.passed and evidence.details["postconditions_met"] is False
    current = backend.observe()
    edits = [c for c in backend.compile(step, planned, current) if c.arguments["operation"] == "fill"]
    assert len(edits) == 1 and edits[0].arguments["value"] == "Reports"
    target = edits[0].arguments["target"]
    assert target["name"] == "" and target["role"] == "textbox"
    assert target["app"] == "org.gnome.Nautilus" and target["window"] == "Desktop"
    # Observing the same application/editor without a sent launcher receipt
    # cannot authorize its mutation from a shell scope.
    unrelated_backend = OSWorldBackend(object(), pause=0)
    assert unrelated_backend.compile(step, planned, snapshots[-1]) == ()


def test_fresh_popup_root_or_target_change_blocks_previously_offered_input(recorded):
    observations, _snapshots = recorded
    env = Playback(observations)
    backend = OSWorldBackend(env, pause=0)
    step, planned = navigation(), backend.observe()
    first = next(
        c for c in backend.compile(step, planned, planned) if c.arguments["bound_control"]["label"] == "Files"
    )
    receipt = backend.execute(first, "obs", "decision")
    assert backend.verify(first, receipt).passed
    current = backend.observe()
    second = next(
        c
        for c in backend.compile(step, planned, current)
        if c.arguments["bound_control"]["label"] == "todo_list_Jan_1" and c.arguments["button"] == "right"
    )
    receipt = backend.execute(second, "obs", "decision")
    assert backend.verify(second, receipt).passed
    popup = backend.observe()
    offered = backend.compile(step, planned, popup)
    assert len(offered) == 16
    env.observations = [*observations[:2], observations[1], observations[3]]  # Popup closed before input.
    assert not backend.execute(offered[0], "obs", "decision").success
    assert len(env.actions) == 2
