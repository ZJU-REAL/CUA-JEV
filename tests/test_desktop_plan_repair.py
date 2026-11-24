"""Preparation replacement keeps real execution and its receipt with the runner."""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import replace

import pytest
from test_observed_input_geometry import STATE, VISIBLE, form, observed
from test_observed_input_geometry import png as png

from cua_jev.agent_types import (
    AgentCondition,
    AgentControl,
    AgentPlan,
    AgentPropertyEdit,
    AgentStep,
    AgentTarget,
)
from cua_jev.desktop_plan_repair import PrepareRepair, repair_property_prepare, suspend_property_prepare
from cua_jev.desktop_semantics import normalize_role


@pytest.fixture
def snapshot(png):
    root, frame, _parent, _proxy = form()
    ET.SubElement(frame, "page-tab", {"name": "User", **VISIBLE, STATE + "selected": "true"})
    _backend, current = observed(root, png)
    return current


def rebind(snapshot, *, controls=None, context=None, **changes):
    """Build a new fixture observation with its own genuine source hashes."""
    changed = replace(snapshot, controls=snapshot.controls if controls is None else tuple(controls),
                      **changes)
    context = context or snapshot.input_geometry[0]
    context = replace(context, snapshot_fingerprint=changed.fingerprint(),
                      jpeg_sha256=hashlib.sha256(changed.image.jpeg).hexdigest())
    return replace(changed, input_geometry=(context,))


def root_plan(*, confirm=False, conditions=()):
    prefix = AgentStep("prepare", "visual_fill", value="retention limit", label="Search current settings",
                       box=(302, 131, 820, 149))
    prop = AgentStep("property", "click", label="Set retained revisions",
                     target=AgentTarget(app="Editor"), depends_on=("prepare",), navigation=True,
                     native_navigation_gestures=True, recovery="observe",
                     property_edit=AgentPropertyEdit("set", "42", True, True),
                     property_prepare_dependency="prepare")
    steps = (prefix, prop)
    if confirm:
        steps += (AgentStep("confirm", "confirm_property", label="Submit the requested settings",
                            depends_on=("property",), recovery="observe",
                            property_confirmation_of="property"),)
    return AgentPlan("Set retained revisions", steps, reusable=True, completion_conditions=conditions,
                     defer_dialog_field_completion=True, complete_property_goal=True)


def suspend(plan, snapshot, current=None, **changes):
    return suspend_property_prepare(plan, snapshot, current or snapshot,
                                    **{"plan_at": 100.0, "ttl_s": 300.0, "horizon": 6,
                                       "now": 110.0, **changes})


def immediate(frame, **changes):
    action = AgentStep("fresh", "visual_fill", value=frame.original_plan.steps[0].value,
                       label="Current visible search input", box=(310, 132, 810, 148))
    return AgentPlan("Repair the current preparation", (action,), prepare_repair_token=frame.token, **changes)


@pytest.mark.parametrize("confirm", [False, True])
def test_replacement_preserves_actual_future_objects_conditions_and_original_lease(snapshot, confirm):
    condition = AgentCondition("checked", AgentTarget(name="User", role="checkbox", app="Editor"))
    old = root_plan(confirm=confirm, conditions=(condition,))
    frame = suspend(old, snapshot)
    assert frame is not None and re.fullmatch("[0-9a-f]{32}", frame.token)
    request = immediate(frame)
    repaired = repair_property_prepare(frame, request, snapshot, now=120)
    assert repaired.steps[0] == replace(request.steps[0], id="prepare")
    assert repaired.steps[0] is not old.steps[0]
    assert all(a is b for a, b in zip(repaired.steps[1:], old.steps[1:], strict=True))
    assert repaired.completion_conditions is old.completion_conditions
    assert repaired.complete_property_goal and repaired.defer_dialog_field_completion and repaired.reusable
    assert repaired.summary == old.summary and repaired.prepare_repair_token == ""
    assert frame.original_plan is old
    assert (frame.plan_at, frame.deadline, frame.horizon) == (100, 400, 6)
    # No execution occurs here: the returned prefix still requires ordinary issuance/guard/receipt.
    assert old.steps[0].box == (302, 131, 820, 149) and not hasattr(frame, "receipt")


def test_context_rechecks_the_current_source_and_never_supplies_an_old_box(snapshot):
    frame = suspend(root_plan(confirm=True), snapshot)
    context = frame.context(snapshot, now=130)
    assert context["token"] == frame.token and context["remaining_time_s"] == 270
    assert context["value"] == "retention limit" and context["operation"] == "visual_fill"
    assert context["retained_stage"]["scope"]["app"] == "Editor"
    assert context["retained_stage"]["edit"] == {"operation": "set", "value": "42"}
    assert context["retained_stage"]["confirm"] == {"label": "Submit the requested settings"}
    assert "box" not in json.dumps(context) and "input_ref" not in json.dumps(context)
    assert "no stable native PID" in context["owner_continuity"]
    context["retained_stage"]["edit"]["value"] = "999"
    assert frame.context(snapshot, now=130)["retained_stage"]["edit"]["value"] == "42"


@pytest.mark.parametrize("box", [(302, 131, 820, 149), (430, 132, 859, 160), (0, 100, 1000, 180)])
def test_actual_click_inside_raw_parent_accepts_wide_and_recorded_recovery_boxes(snapshot, box):
    # Actual v43 recovery clicks floor(1237.44,157.68) within the raw parent,
    # although the model rectangle extends past that parent. No input is sent.
    frame = suspend(root_plan(), snapshot)
    assert frame is not None
    request = immediate(frame)
    request = replace(request, steps=(replace(request.steps[0], box=box),))
    assert repair_property_prepare(frame, request, snapshot, now=120).steps[0].box == box
    assert "actual click pixel" in frame.context(snapshot, now=120)["geometry_relation"]


@pytest.mark.parametrize("box", [(0, 131, 500, 149), (800, 131, 990, 149),
                                  (302, 0, 820, 250), (302, 132, 820, 300),
                                  (290, 131, 312, 149), (800, 131, 841, 149)])
def test_actual_click_outside_the_parent_cannot_be_repaired(snapshot, box):
    old = root_plan()
    assert suspend(replace(old, steps=(replace(old.steps[0], box=box), *old.steps[1:])), snapshot) is None
    frame = suspend(old, snapshot)
    request = immediate(frame)
    with pytest.raises(ValueError):
        repair_property_prepare(frame, replace(request, steps=(replace(request.steps[0], box=box),)),
                                snapshot, now=120)


@pytest.mark.parametrize("box,accepted", [((250, 100, 350, 180), True),
                                           ((750, 100, 850, 180), False),
                                           ((302, 80, 820, 120), True),
                                           ((302, 180, 820, 220), False)])
def test_parent_pixel_edges_are_half_open_exactly_as_physical_clicks(snapshot, box, accepted):
    context = replace(snapshot.input_geometry[0], ancestor_rect_px=(576, 108, 1536, 216))
    changed = rebind(snapshot, context=context)
    frame = suspend(root_plan(), changed)
    assert frame is not None
    request = immediate(frame)
    request = replace(request, steps=(replace(request.steps[0], box=box),))
    if accepted:
        assert repair_property_prepare(frame, request, changed, now=120).steps[0].box == box
    else:
        with pytest.raises(ValueError):
            repair_property_prepare(frame, request, changed, now=120)


@pytest.mark.parametrize("change", ["app", "window", "ancestor", "identity", "input_rect", "parent_rect",
                                    "parent_role", "parent_class", "parent_label", "desktop", "tab"])
def test_changed_owner_input_or_selected_tab_refuses_context_and_replacement(snapshot, change):
    frame = suspend(root_plan(), snapshot)
    controls = list(snapshot.controls)
    context = snapshot.input_geometry[0]
    proxy = next(i for i, c in enumerate(controls) if c.ref == context.input_ref)
    if change in {"app", "window"}:
        field, value = ("app", "Editor Twin") if change == "app" else ("window", "Other Settings")
        controls = [replace(c, **{field: value}) if c.app == "Editor" else c for c in controls]
    elif change == "ancestor":
        controls[proxy] = replace(controls[proxy], ancestors=(*controls[proxy].ancestors, "panel: Other"))
    elif change == "identity":
        controls[proxy] = replace(controls[proxy], input_type="search")
    elif change == "tab":
        controls = [replace(c, label="Workspace") if normalize_role(c.role) == "tab" and c.selected else c
                    for c in controls]
    else:
        field, value = {
            "input_rect": ("input_rect_px", (724, 141, 725, 142)),
            "parent_rect": ("ancestor_rect_px", (579, 141, 1575, 161)),
            "parent_role": ("ancestor_role", "panel"),
            "parent_class": ("ancestor_class", "another wrapper"),
            "parent_label": ("ancestor_label", "Another search"),
            "desktop": ("desktop_size_px", (1921, 1080)),
        }[change]
        context = replace(context, **{field: value})
    changed = rebind(snapshot, controls=controls, context=context)
    assert suspend(frame.original_plan, snapshot, changed) is None
    assert frame.context(changed, now=120) is None
    with pytest.raises(ValueError):
        repair_property_prepare(frame, immediate(frame), changed, now=120)


def test_foreign_global_active_window_is_not_an_ambiguous_owner_in_this_app(snapshot):
    foreign = AgentControl("shell", "Desktop", "container", role="window", app="Shell",
                           window="Desktop", active=True, focused=True)
    changed = rebind(snapshot, controls=(*snapshot.controls, foreign))
    frame = suspend(root_plan(), snapshot, changed)
    assert frame is not None and frame.context(changed, now=120)


@pytest.mark.parametrize("kind", ["duplicate", "inactive", "disabled"])
def test_owner_frame_must_be_unique_and_current(snapshot, kind):
    owner = next(c for c in snapshot.controls if c.role == "frame" and c.app == "Editor")
    controls = (*snapshot.controls, replace(owner, ref="twin")) if kind == "duplicate" else tuple(
        replace(c, **({"active": False} if kind == "inactive" else {"disabled": True})) if c is owner else c
        for c in snapshot.controls)
    changed = rebind(snapshot, controls=controls)
    assert suspend(root_plan(), snapshot, changed) is None


def test_observation_local_refs_may_change_without_changing_owner_or_input_identity(snapshot):
    context = snapshot.input_geometry[0]
    controls = tuple(replace(c, ref="new-" + c.ref) for c in snapshot.controls)
    changed = rebind(snapshot, controls=controls,
                     context=replace(context, input_ref="new-" + context.input_ref))
    frame = suspend(root_plan(), snapshot, changed)
    assert frame is not None
    repaired = repair_property_prepare(frame, immediate(frame), changed, now=120)
    assert repaired.steps[1] is frame.original_plan.steps[1]


@pytest.mark.parametrize("kind", ["missing", "incomplete", "bad_hash", "stale_fingerprint"])
def test_source_binding_is_required_for_both_old_and_current_observations(snapshot, kind):
    context = snapshot.input_geometry[0]
    if kind == "missing":
        changed = replace(snapshot, input_geometry=())
    elif kind == "incomplete":
        changed = replace(snapshot, a11y_complete=False)
    elif kind == "bad_hash":
        changed = replace(snapshot, input_geometry=(replace(context, jpeg_sha256="0" * 64),))
    else:
        changed = replace(snapshot, text="An unbound observation")
    assert suspend(root_plan(), snapshot, changed) is None
    assert suspend(root_plan(), changed, snapshot) is None


@pytest.mark.parametrize("changes", [{"horizon": 1}, {"horizon": True}, {"horizon": 13},
                                      {"ttl_s": 0}, {"ttl_s": 301}, {"ttl_s": float("nan")},
                                      {"plan_at": float("inf")}, {"now": 99}, {"now": 400}])
def test_invalid_or_expired_original_lease_does_not_issue_a_frame(snapshot, changes):
    assert suspend(root_plan(), snapshot, **changes) is None


@pytest.mark.parametrize("now", [99, 400, 401, float("inf"), float("nan"), True])
def test_context_and_repair_never_refresh_the_original_deadline(snapshot, now):
    frame = suspend(root_plan(), snapshot)
    assert frame.context(snapshot, now=now) is None
    with pytest.raises(ValueError):
        repair_property_prepare(frame, immediate(frame), snapshot, now=now)


@pytest.mark.parametrize("kind", ["token", "deadline", "horizon", "scope", "value", "goal_condition"])
def test_frame_deep_binding_rejects_altered_source_authority(snapshot, kind):
    frame = suspend(root_plan(), snapshot)
    if kind == "token":
        forged = replace(frame, token="a" * 32)
    elif kind == "deadline":
        forged = replace(frame, deadline=399)
    elif kind == "horizon":
        forged = replace(frame, horizon=5)
    else:
        old = frame.original_plan
        prop = old.steps[1]
        if kind == "scope":
            prop = replace(prop, target=AgentTarget(app="Editor", window="Settings"))
            changed = replace(old, steps=(old.steps[0], prop))
        elif kind == "value":
            changed_prop = replace(prop, property_edit=replace(prop.property_edit, value="999"))
            changed = replace(old, steps=(old.steps[0], changed_prop))
        else:
            changed = replace(old, completion_conditions=(AgentCondition("window_contains", value="Other"),))
        forged = replace(frame, original_plan=changed)
    assert forged.context(snapshot, now=120) is None
    with pytest.raises(ValueError):
        repair_property_prepare(forged, immediate(forged), snapshot, now=120)


def test_unissued_public_frame_is_not_an_authorization(snapshot):
    frame = suspend(root_plan(), snapshot)
    unissued = PrepareRepair(frame.token, frame.original_plan, frame.plan_at, frame.deadline,
                             frame.horizon, frame.owner_key)
    assert unissued.context(snapshot, now=120) is None
    with pytest.raises(ValueError):
        repair_property_prepare(unissued, immediate(unissued), snapshot, now=120)


@pytest.mark.parametrize("kind", ["missing_token", "wrong_token", "done", "reusable", "goal_conditions",
                                    "complete_goal", "defer", "two_steps", "value", "operation", "target",
                                    "ref", "preconditions", "postconditions", "navigation"])
def test_repair_metadata_never_loosens_the_single_current_action_grammar(snapshot, kind):
    frame = suspend(root_plan(), snapshot)
    request = immediate(frame)
    step = request.steps[0]
    fields = {
        "missing_token": {"prepare_repair_token": ""}, "wrong_token": {"prepare_repair_token": "b" * 32},
        "done": {"done": True}, "reusable": {"reusable": True},
        "goal_conditions": {"completion_conditions": (AgentCondition("window_contains", value="Settings"),)},
        "complete_goal": {"complete_property_goal": True}, "defer": {"defer_dialog_field_completion": True},
        "two_steps": {"steps": (step, replace(step, id="extra"))},
    }
    if kind in fields:
        request = replace(request, **fields[kind])
    else:
        changes = {
            "value": {"value": step.value + " "}, "operation": {"operation": "visual_click"},
            "target": {"target": AgentTarget(name="Search", app="Editor")}, "ref": {"ref": "current"},
            "preconditions": {"preconditions": (AgentCondition("window_contains", value="Settings"),)},
            "postconditions": {"postconditions": (AgentCondition("window_contains", value="Settings"),)},
            "navigation": {"navigation": True},
        }[kind]
        request = replace(request, steps=(replace(step, **changes),))
    with pytest.raises(ValueError):
        repair_property_prepare(frame, request, snapshot, now=120)


@pytest.mark.parametrize("kind", ["prefix_conditions", "prefix_target", "prefix_ref", "prefix_unicode",
                                    "prefix_empty", "not_reusable", "done", "bad_dependency", "scope",
                                    "extra_step", "bad_confirmation"])
def test_only_the_compiler_prepared_property_shape_can_be_suspended(snapshot, kind):
    old = root_plan()
    prefix, prop = old.steps
    if kind == "not_reusable":
        old = replace(old, reusable=False)
    elif kind == "done":
        old = replace(old, done=True)
    elif kind == "extra_step":
        old = replace(old, steps=(*old.steps, AgentStep("foreign", "click")))
    elif kind == "bad_confirmation":
        old = root_plan(confirm=True)
        bad_confirm = replace(old.steps[-1], property_confirmation_of="foreign")
        old = replace(old, steps=(*old.steps[:-1], bad_confirm))
    elif kind == "bad_dependency":
        old = replace(old, steps=(prefix, replace(prop, depends_on=("foreign",))))
    elif kind == "scope":
        old = replace(old, steps=(prefix, replace(prop, target=AgentTarget(app="Other"))))
    else:
        change = {
            "prefix_conditions": {"preconditions": (AgentCondition("window_contains", value="Settings"),)},
            "prefix_target": {"target": AgentTarget(name="Search", app="Editor")},
            "prefix_ref": {"ref": "old"}, "prefix_unicode": {"value": "\u641c\u7d22"},
            "prefix_empty": {"value": ""},
        }[kind]
        old = replace(old, steps=(replace(prefix, **change), prop))
    assert suspend(old, snapshot) is None
