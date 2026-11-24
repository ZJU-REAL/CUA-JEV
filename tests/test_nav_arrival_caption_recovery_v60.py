'Regression tests for test nav arrival caption recovery v60.'

import copy
import io
from contextlib import contextmanager
from dataclasses import replace
from html import escape

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import _condition_facts
from cua_jev.agent_types import NativeDesktopEvidence
from cua_jev.desktop_navigation_policy import enable_full_legal_navigation_gestures
from cua_jev.desktop_semantics import condition_evidence, matching_controls
from cua_jev.desktop_target_diagnostics import observed_navigation_arrival_diagnostics
from cua_jev.osworld_backend import OSWorldBackend

APP = "Workbench"
CAPTION = "Publish Plan"
OBSERVED = "Publish Plan... Ctrl+Shift+P"


def menu_xml(labels=(OBSERVED,), *, shown=True, enabled=True, second_owner=False, owner_box=True):
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            '<menu-item name="' + escape(label, quote=True) + '" '
            'st:enabled="' + ("true" if enabled else "false") + '" '
            'st:sensitive="true" st:focusable="true" st:visible="true" '
            'st:showing="' + ("true" if shown else "false") + '" '
            'cp:screencoord="(30, ' + str(90 + 35 * index) + ')" cp:size="(300, 30)"/>'
        )
    other = (
        '<frame name="Other workspace" st:active="true" st:enabled="true" '
        'st:sensitive="true" st:visible="true" st:showing="true" '
        'cp:screencoord="(500, 20)" cp:size="(450, 700)"/>'
    ) if second_owner else ""
    owner_geometry = ' cp:screencoord="(20, 70)" cp:size="(340, 400)"' if owner_box else ""
    return (
        '<desktop-frame xmlns:st="https://accessibility.ubuntu.example.org/ns/state" '
        'xmlns:cp="https://accessibility.ubuntu.example.org/ns/component">'
        '<application name="' + APP + '" st:enabled="true" st:sensitive="true">'
        '<frame name="Workspace" st:enabled="true" st:sensitive="true" '
        'st:showing="true" st:visible="true" cp:screencoord="(0, 0)" cp:size="(1000, 800)">'
        '<push-button name="Tools" st:enabled="true" st:sensitive="true" '
        'st:showing="true" st:visible="true" cp:screencoord="(10, 10)" cp:size="(80, 30)"/>'
        '</frame><frame name="" st:active="true" st:enabled="true" st:sensitive="true" '
        'st:visible="true" st:showing="true"' + owner_geometry + '>'
        '<menu st:enabled="true" st:sensitive="true" st:showing="true" st:visible="true">'
        + "".join(rows) + '</menu></frame>' + other + '</application></desktop-frame>'
    )


class ReadonlyDesktop:
    def __init__(self, xml):
        self.xml, self.reads, self.inputs = xml, 0, 0
        buffer = io.BytesIO()
        Image.new("RGB", (1000, 800), "white").save(buffer, format="PNG")
        self.png = buffer.getvalue()

    def _get_obs(self):
        self.reads += 1
        return {"screenshot": self.png, "accessibility_tree": self.xml}

    def step(self, *_args, **_kwargs):
        self.inputs += 1
        raise AssertionError("Diagnostic evaluation must never send GUI input")


@contextmanager
def prepared(xml=None):
    desktop = ReadonlyDesktop(xml or menu_xml())
    backend = OSWorldBackend(desktop, pause=0)
    try:
        current = backend.observe()
        original = {
            "summary": "Reach requested menu view",
            "done": False,
            "steps": [{
                "id": "reach", "operation": "click", "navigation": True,
                "label": "Reach requested menu view", "target": {"app": APP},
                "postconditions": [{"kind": "exists", "target": {
                    "name": CAPTION, "role": "menuitem", "app": APP,
                }}], "recovery": "observe",
            }],
        }
        plan = parse_plan(original, current, max_actions=6, profile="desktop")
        plan = enable_full_legal_navigation_gestures(plan)
        _validate_desktop_execution_plan(plan, current, 6)
        step = plan.steps[0]
        offers = backend.compile(step, current, current)
        yield desktop, backend, current, step, offers
    finally:
        backend.close()


def diagnostic(current, step, offers):
    return observed_navigation_arrival_diagnostics(
        current, step, _condition_facts(step.postconditions, current), offers,
    )


def recompile(backend, current, step):
    return backend.compile(step, current, current)


def test_actual_OSWorld_nativeNone_menu_offer_explains_caption_without_rewriting_truth_or_input():
    with prepared() as (desktop, _backend, current, step, offers):
        menu = next(control for control in current.controls if control.label == OBSERVED)
        assert menu.native is None  # Actual ordinary Electron-like adapter route.
        assert menu.active and not menu.disabled and menu.window == ""
        evidence = condition_evidence(step.postconditions[0], current)
        assert evidence == {"outcome": "unknown", "reason": "target_missing"}
        assert not matching_controls(current, step.postconditions[0].target)
        before = current.to_dict(), step.to_dict(), [candidate.to_dict() for candidate in offers]
        reads_before = desktop.reads
        result = diagnostic(current, step, offers)
        assert result["code"] == "navigation_arrival_caption_mismatch"
        assert result["source"] == "current_delegated_navigation_snapshot"
        assert result["snapshot_fingerprint"] == current.fingerprint()
        item, = result["predicates"]
        shown, = item["observed_controls"]
        assert shown["label"] == OBSERVED and shown["ref"] == menu.ref
        assert shown["window"] == "" and item["current_menu_owner"]["window"] == ""
        assert item["retrieval"] == "menu_caption_and_shortcut_text"
        assert item["original_predicate"]["outcome"] == "unknown"
        assert item["original_predicate"]["target"]["name"] == CAPTION
        assert result["input_sent"] is result["input_authority"] is result["task_success"] is False
        assert result["conditions_rewritten"] is False
        assert condition_evidence(step.postconditions[0], current) == evidence
        assert (current.to_dict(), step.to_dict(), [candidate.to_dict() for candidate in offers]) == before
        assert desktop.reads == reads_before and desktop.inputs == 0


@pytest.mark.parametrize("label", ["Unrelated Option Ctrl+P", "Publish Plan Later Ctrl+P", "Publish Plan"])
def test_future_missing_or_a_now_exact_target_is_not_a_caption_mismatch(label):
    with prepared(menu_xml((label,))) as (desktop, _backend, current, step, offers):
        assert diagnostic(current, step, offers) is None
        assert desktop.inputs == 0


@pytest.mark.parametrize("changes", [
    {"role": "dialog"}, {"role": "button"}, {"match": "contains"},
    {"app": "Other"}, {"window": "Workspace"}, {"ancestor": "Unobserved section"},
])
def test_no_role_fuzzing_or_app_window_ancestor_expansion(changes):
    with prepared() as (desktop, _backend, current, step, offers):
        condition = replace(step.postconditions[0], target=replace(step.postconditions[0].target, **changes))
        revised = replace(step, postconditions=(condition,))
        assert diagnostic(current, revised, offers) is None
        assert desktop.inputs == 0


@pytest.mark.parametrize("shown,enabled", [(False, True), (True, False)])
def test_real_adapter_hidden_or_disabled_menu_never_supplies_trigger(shown, enabled):
    with prepared(menu_xml(shown=shown, enabled=enabled)) as (desktop, _backend, current, step, offers):
        assert diagnostic(current, step, offers) is None
        assert desktop.inputs == 0


@pytest.mark.parametrize("state", ["visible", "showing", "enabled", "sensitive"])
def test_explicit_negative_native_evidence_is_never_overridden_by_an_offer(state):
    with prepared() as (desktop, backend, current, step, _offers):
        menu = next(control for control in current.controls if control.label == OBSERVED)
        native = NativeDesktopEvidence(
            complete=True, enabled=True, sensitive=True, visible=True, showing=True,
        )
        revised_menu = replace(menu, native=replace(native, **{state: False}))
        revised = replace(current, controls=tuple(revised_menu if c.ref == menu.ref else c
                                                   for c in current.controls))
        offers = recompile(backend, revised, step)
        assert diagnostic(revised, step, offers) is None
        assert desktop.inputs == 0


@pytest.mark.parametrize("tamper", ["fingerprint", "ref", "bound_control", "box", "scope", "route"])
def test_current_offers_must_bind_full_current_state_control_scope_and_source(tamper):
    with prepared() as (desktop, _backend, current, step, offers):
        candidate = offers[0]
        args = copy.deepcopy(candidate.arguments)
        if tamper == "fingerprint":
            args["fingerprint"] = "0" * 64
        elif tamper == "ref":
            args["bound_control"]["ref"] = "unobserved-ref"
        elif tamper == "bound_control":
            args["bound_control"]["label"] = "invented caption"
        elif tamper == "box":
            args["box"] = (1, 1, 100, 100)
        elif tamper == "scope":
            args["target"] = {**args["target"], "app": "Other"}
        else:
            args["route"] = "unverified-adapter"
        altered = (replace(candidate, arguments=args), *offers[1:])
        assert diagnostic(current, step, altered) is None
        assert desktop.inputs == 0


def test_full_offers_cannot_be_a_truncated_or_duplicate_prefix():
    with prepared() as (desktop, _backend, current, step, offers):
        assert len(offers) > 1
        assert diagnostic(current, step, offers[:-1]) is None
        assert diagnostic(current, step, (offers[0], offers[0], *offers[1:])) is None
        assert desktop.inputs == 0


def test_two_real_active_roots_are_not_silently_assigned_one_owner():
    with prepared(menu_xml(second_owner=True)) as (desktop, _backend, current, step, offers):
        assert len([c for c in current.controls if c.app == APP and c.active and c.role == "frame"]) == 2
        assert diagnostic(current, step, offers) is None
        assert desktop.inputs == 0


def test_real_adapter_can_keep_nonexecuting_owner_without_box_and_legal_menu_geometry():
    # This exercises the actual adapter on XML with no owner component geometry.
    # Historical public trace controls omit box entirely; they do NOT prove
    # that the internal owner box in the closed real run was None.
    with prepared(menu_xml(owner_box=False)) as (desktop, _backend, current, step, offers):
        owner, = [control for control in current.controls
                  if control.app == APP and control.active and control.role == "frame"]
        menu = next(control for control in current.controls if control.label == OBSERVED)
        assert owner.box is None and owner.tag == "container" and owner.window == ""
        assert menu.box is not None and menu.native is None
        result = diagnostic(current, step, offers)
        item, = result["predicates"]
        assert item["current_menu_owner"] == {
            "ref": owner.ref, "role": owner.role, "app": owner.app, "window": owner.window,
        }
        assert item["observed_controls"][0]["ref"] == menu.ref
        assert result["input_sent"] is False and desktop.inputs == 0


@pytest.mark.parametrize("change", [{"app": "Other"}, {"window": "Other workspace"}])
def test_unboxed_owner_still_requires_actual_matching_app_and_window(change):
    with prepared(menu_xml(owner_box=False)) as (desktop, backend, current, step, _offers):
        owner, = [control for control in current.controls
                  if control.app == APP and control.active and control.role == "frame"]
        revised = replace(current, controls=tuple(replace(control, **change)
                                                   if control.ref == owner.ref else control
                                                   for control in current.controls))
        offers = recompile(backend, revised, step)
        assert diagnostic(revised, step, offers) is None
        assert desktop.inputs == 0


def test_unboxed_owner_cannot_borrow_a_menu_control_reference():
    with prepared(menu_xml(owner_box=False)) as (desktop, _backend, current, step, offers):
        owner, = [control for control in current.controls
                  if control.app == APP and control.active and control.role == "frame"]
        menu = next(control for control in current.controls if control.label == OBSERVED)
        revised = replace(current, controls=tuple(replace(control, ref=menu.ref)
                                                   if control.ref == owner.ref else control
                                                   for control in current.controls))
        assert diagnostic(revised, step, offers) is None
        assert desktop.inputs == 0


def test_unboxed_owner_does_not_remove_actual_menu_click_box_requirement():
    with prepared(menu_xml(owner_box=False)) as (desktop, backend, current, step, _offers):
        menu = next(control for control in current.controls if control.label == OBSERVED)
        revised = replace(current, controls=tuple(replace(control, box=None)
                                                   if control.ref == menu.ref else control
                                                   for control in current.controls))
        offers = recompile(backend, revised, step)
        assert diagnostic(revised, step, offers) is None
        assert desktop.inputs == 0


def test_duplicate_snapshot_refs_cannot_supply_an_owner_or_input_identity():
    with prepared() as (desktop, _backend, current, step, offers):
        revised = replace(current, controls=(*current.controls, current.controls[0]))
        assert diagnostic(revised, step, offers) is None
        assert desktop.inputs == 0


@pytest.mark.parametrize("field,value", [
    ("outcome", "true"), ("reason", "unsupported_state"), ("satisfied", True), ("kind", "expanded"),
])
def test_historical_or_fabricated_facts_do_not_override_current_original_condition(field, value):
    with prepared() as (desktop, _backend, current, step, offers):
        facts = _condition_facts(step.postconditions, current)
        facts[0][field] = value
        assert observed_navigation_arrival_diagnostics(current, step, facts, offers) is None
        assert desktop.inputs == 0


def test_incomplete_or_nonfull_frame_never_triggers():
    with prepared() as (desktop, _backend, current, step, offers):
        assert diagnostic(replace(current, a11y_complete=False), step, offers) is None
        assert diagnostic(replace(current, image=None), step, offers) is None
        assert desktop.inputs == 0


def test_only_ordinary_delegated_navigation_is_eligible():
    from cua_jev.agent_types import AgentPropertyEdit

    with prepared() as (desktop, _backend, current, step, offers):
        assert diagnostic(current, replace(step, navigation=False), offers) is None
        assert diagnostic(current, replace(step, property_edit=AgentPropertyEdit("check")), offers) is None
        assert diagnostic(current, replace(step, property_confirmation_of="property"), offers) is None
        assert desktop.inputs == 0


def test_bounded_display_preserves_complete_match_count_and_original_unknown():
    labels = (
        "Publish Plan... Ctrl+P", "Publish Plan... Alt+P", "Publish Plan... Shift+P",
        "Publish Plan... Ctrl+Shift+P", "Publish Plan... Alt+Shift+P", "Publish Plan... Super+P",
    )
    with prepared(menu_xml(labels)) as (desktop, _backend, current, step, offers):
        result = diagnostic(current, step, offers)
        item, = result["predicates"]
        assert item["match_count"] == len(labels) and item["truncated"] is True
        assert len(item["observed_controls"]) == 4
        assert item["original_predicate"]["outcome"] == "unknown"
        assert desktop.inputs == 0
