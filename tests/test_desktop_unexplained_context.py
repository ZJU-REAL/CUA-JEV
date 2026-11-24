"""Information predicates preserve exact roles, evidence and modal ownership."""

from dataclasses import replace

import pytest

from cua_jev.agent_types import (
    AgentCondition,
    AgentControl,
    AgentPropertyEdit,
    AgentSnapshot,
    AgentStep,
    AgentTarget,
)
from cua_jev.desktop_unexplained_context import unexplained_navigation_context
from cua_jev.models import Verification


def _owner(ref, name, *, role="dialog", active=True, modal=False):
    return AgentControl(ref, name, "container", role=role, app="Demo", window=name,
                        active=active, modal=modal, modal_owner_ref=ref if modal else "")


def _button(ref, owner, *, disabled=False):
    return AgentControl(ref, "Accept", "button", role="button", app="Demo", window=owner.window,
                        active=owner.active, box=(10, 10, 50, 40), disabled=disabled,
                        modal_owner_ref=owner.ref if owner.modal else "")


def _snapshot(*controls, complete=True):
    return AgentSnapshot("osworld://desktop", "Demo", "", tuple(controls), a11y_complete=complete)


def _case(*, arrival_role="frame"):
    original = _owner("old", "Main", role="frame")
    new = _owner("new", "Options")
    before = _snapshot(original, _button("open", original))
    after = _snapshot(new, _button("accept", new))
    arrival = AgentCondition("exists", AgentTarget(name="Options", role=arrival_role))
    step = AgentStep("reach", "click", target=AgentTarget(app="Demo"), navigation=True,
                     recovery="observe", postconditions=(arrival,))
    verified = Verification(True, "agent.effect", {"navigation": True, "accessibility_progress": True})
    return step, before, after, verified


def test_new_dialog_requests_information_without_relabeling_arrival():
    args = _case()
    signal = unexplained_navigation_context(*args, owner_app="Demo")
    assert signal["owner_role"] == "dialog"
    assert signal["arrival_evidence"][0]["outcome"] == "unknown"
    assert signal["task_success"] is False and signal["input_authority"] is False
    assert args[3].passed is True


def test_exact_arrival_never_adds_request():
    assert unexplained_navigation_context(*_case(arrival_role="dialog"), owner_app="Demo") is None


@pytest.mark.parametrize(
    "change", ["partial_before", "partial_after", "unverified", "wrong_app", "no_effects"],
)
def test_missing_evidence_does_not_invent_context(change):
    step, before, after, verified = _case()
    app = "Demo"
    if change == "partial_before":
        before = replace(before, a11y_complete=False)
    elif change == "partial_after":
        after = replace(after, a11y_complete=False)
    elif change == "unverified":
        verified = replace(verified, passed=False)
    elif change == "wrong_app":
        app = "Other"
    else:
        verified = replace(verified, details={"navigation": True, "accessibility_progress": False})
    assert unexplained_navigation_context(step, before, after, verified, owner_app=app) is None


def test_menu_focus_value_and_geometry_changes_are_not_new_owner():
    step, before, _after, verified = _case()
    root, button = before.controls
    after = _snapshot(root, replace(button, focused=True, value="changed", box=(40, 10, 80, 40)),
                      AgentControl("menu", "Action", "button", role="menuitem", app="Demo",
                                   window=root.window, box=(10, 60, 60, 90)))
    assert unexplained_navigation_context(step, before, after, verified, owner_app="Demo") is None


def test_title_only_same_kind_replacement_is_not_proven_new_owner():
    step, _before, after, verified = _case()
    old = _owner("old", "Previous title")
    before = _snapshot(old, _button("oldbutton", old))
    assert unexplained_navigation_context(step, before, after, verified, owner_app="Demo") is None


def test_ambiguous_or_disabled_context_does_not_bind():
    step, before, after, verified = _case()
    other = _owner("second", "Another options")
    ambiguous = _snapshot(*after.controls, other, _button("secondbutton", other))
    assert unexplained_navigation_context(step, before, ambiguous, verified, owner_app="Demo") is None
    disabled = _snapshot(after.controls[0], replace(after.controls[1], disabled=True))
    assert unexplained_navigation_context(step, before, disabled, verified, owner_app="Demo") is None


def test_old_active_dialog_plus_one_novel_dialog_is_still_ambiguous():
    step, _before, after, verified = _case()
    old = _owner("old", "Already open")
    before = _snapshot(old, _button("oldbutton", old))
    current = _snapshot(*before.controls, *after.controls)
    assert unexplained_navigation_context(step, before, current, verified, owner_app="Demo") is None


def test_property_and_empty_arrival_remain_outside_initial_rule():
    step, before, after, verified = _case()
    for unsupported in (replace(step, postconditions=()),
                        replace(step, property_edit=AgentPropertyEdit("fill", "requested"))):
        assert unexplained_navigation_context(unsupported, before, after, verified, owner_app="Demo") is None


def test_reported_modal_frame_uses_current_owner_ancestry_not_parent_box():
    step, before, _after, verified = _case()
    modal = _owner("modal", "", role="frame", modal=True)
    after = _snapshot(modal, _button("accept", modal))
    assert unexplained_navigation_context(step, before, after, verified, owner_app="Demo") is not None
