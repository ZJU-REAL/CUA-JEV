"""Public diagnostic retrieval must not become an executable menu-label alias."""

import copy
import json
from dataclasses import asdict, replace

import httpx
import pytest
from test_desktop_target_diagnostics import control, history, snapshot

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentCondition, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, executable_controls, matching_controls
from cua_jev.desktop_target_diagnostics import observed_target_diagnostics


def menu(ref, label, **changes):
    return control(ref=ref, label=label, role="menuitem", app="Editor", window="", active=True, **changes)


def parse_current_click(current, *, target=None, ref=None):
    action = {"operation": "click", **({"target": asdict(target)} if target is not None else {"ref": ref})}
    if target is not None:
        action["postconditions"] = [{"kind": "absent", "target": asdict(target)}]
    raw = {"summary": "Explicit current menu item", "done": False, "steps": [{"id": "s1", **action}]}

    def respond(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        planner = MultimodalPlanner(
            "https://offline.invalid", "fixture", api_key="offline", client=client,
            profile="desktop", desktop_contract="property_intent_v52",
        )
        try:
            return planner.plan("Use the explicit current menu item", current, [], max_actions=1)
        except ValueError as exc:
            raise ValueError(planner.last_validation_error or str(exc)) from None


@pytest.mark.parametrize("label", [
    "Save As...", "Save As…", "Save As... Ctrl+Shift+S", "Save As… Ctrl+Shift+S", "Save As Ctrl+Shift+S",
])
def test_public_retrieval_preserves_full_label_ref_and_original_exact_input_boundary(label):
    current = snapshot(menu("observed-save", label))
    target = AgentTarget(name="Save As", role="menuitem", app="Editor")
    previous = history(target)
    before = copy.deepcopy(previous), current.to_dict()
    condition = AgentCondition("exists", target)
    evidence = condition_evidence(condition, current)
    assert evidence == {"outcome": "unknown", "reason": "target_missing"}
    assert matching_controls(current, target) == executable_controls(current, "click", target) == ()

    result = observed_target_diagnostics(current, previous)
    item, = result["predicates"]
    shown, = item["observed_controls"]
    assert item["retrieval"] == "menu_caption_and_shortcut_text"
    assert item["match_count"] == 1 and item["truncated"] is False
    assert shown["label"] == label and shown["ref"] == "observed-save" and shown["window"] == ""
    assert shown["operations"] == ["click"]
    assert item["historical_predicate"]["target"] == asdict(target)
    assert item["historical_predicate"]["outcome"] == "unknown"
    assert condition_evidence(condition, current) == evidence
    with pytest.raises(ValueError, match="no current executable control"):
        parse_current_click(current, target=target)

    explicit = replace(target, name=shown["label"])
    assert len(parse_current_click(current, target=explicit).steps) == 1
    assert parse_current_click(current, ref=shown["ref"]).steps[0].ref == "observed-save"
    assert (previous, current.to_dict()) == before


def test_public_retrieval_keeps_all_owner_constraints_and_disabled_peers():
    base = menu("first", "Save As... Ctrl+Shift+S")
    base = replace(base, window="Main", ancestors=("menu: File",))
    current = snapshot(
        base, replace(base, ref="disabled", label="Save As…", disabled=True),
        replace(base, ref="other-app", app="Other"),
        replace(base, ref="other-window", window="Other"),
        replace(base, ref="other-ancestor", ancestors=("menu: Other",)),
    )
    target = AgentTarget(name="Save As", role="menuitem", app="Editor", window="Main", ancestor="File")
    item, = observed_target_diagnostics(current, history(target))["predicates"]
    assert item["match_count"] == 2 and item["truncated"] is False
    assert [item["ref"] for item in item["observed_controls"]] == ["first", "disabled"]
    assert [item["operations"] for item in item["observed_controls"]] == [["click"], []]
    assert all(item["matches_requested_ancestor"] for item in item["observed_controls"])
    assert item["historical_predicate"]["target"] == asdict(target)
    with pytest.raises(ValueError, match="no current executable control"):
        parse_current_click(current, target=target)


def test_diagnostic_ambiguity_never_selects_an_enabled_peer_or_authorizes_a_disabled_ref():
    label = "Save As... Ctrl+Shift+S"
    current = snapshot(menu("one", label), menu("two", label), menu("disabled", label, disabled=True))
    target = AgentTarget(name="Save As", role="menuitem", app="Editor")
    item, = observed_target_diagnostics(current, history(target))["predicates"]
    assert item["match_count"] == 3
    assert [item["operations"] for item in item["observed_controls"]] == [["click"], ["click"], []]
    with pytest.raises(ValueError, match="multiple executable controls"):
        parse_current_click(current, target=replace(target, name=label))
    with pytest.raises(ValueError, match="unavailable or sensitive"):
        parse_current_click(current, ref="disabled")


def test_anonymous_menu_does_not_inherit_the_snapshot_or_requested_window():
    current = snapshot(menu("anonymous", "Save As... Ctrl+Shift+S"))
    target = AgentTarget(name="Save As", role="menuitem", app="Editor", window="Main")
    assert current.title == "Main" and current.controls[0].window == ""
    assert observed_target_diagnostics(current, history(target)) is None
    assert matching_controls(current, replace(target, name=current.controls[0].label)) == ()


def test_incomplete_AX_cannot_supply_caption_retrieval_or_change_the_historical_unknown():
    current = snapshot(menu("observed", "Save As… Ctrl+Shift+S"), complete=False)
    previous = history(AgentTarget(name="Save As", role="menuitem", app="Editor"))
    before = copy.deepcopy(previous), current.to_dict()
    assert observed_target_diagnostics(current, previous) is None
    assert (previous, current.to_dict()) == before


@pytest.mark.parametrize("label", ["Save... As Ctrl+Shift+S", "Save Workspace As...", "Save As...."])
def test_public_retrieval_does_not_expand_to_internal_or_different_captions(label):
    current = snapshot(menu("unrelated", label))
    target = AgentTarget(name="Save As", role="menuitem", app="Editor")
    assert observed_target_diagnostics(current, history(target)) is None


def test_display_bound_preserves_total_match_count_and_cannot_imply_input_uniqueness():
    label = "Save As... Ctrl+Shift+S"
    current = snapshot(*(menu(f"observed-{index}", label) for index in range(6)))
    target = AgentTarget(name="Save As", role="menuitem", app="Editor")
    item, = observed_target_diagnostics(current, history(target))["predicates"]
    assert item["match_count"] == 6 and item["truncated"] is True
    assert len(item["observed_controls"]) == 4
    assert [shown["ref"] for shown in item["observed_controls"]] == [
        f"observed-{index}" for index in range(4)
    ]
    assert len(matching_controls(current, replace(target, name=label))) == 6
    with pytest.raises(ValueError, match="multiple executable controls"):
        parse_current_click(current, target=replace(target, name=label))
