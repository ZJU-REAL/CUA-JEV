"""Shortcut-like menu text is diagnostic retrieval, never a new target alias."""

import copy
import json
from dataclasses import asdict, replace

import pytest

from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, executable_controls
from cua_jev.desktop_target_diagnostics import observed_target_diagnostics

TARGET = AgentTarget(name="New document", role="menuitem", app="Editor")


def control(label="New document Ctrl+N", **changes):
    return replace(AgentControl("item", label, "button", role="menuitem", app="Editor", window="",
                                selectable=True, box=(10, 10, 200, 60)), **changes)


def state(*controls, complete=True):
    return AgentSnapshot("osworld://desktop", "Editor", "private page text", tuple(controls),
                         a11y_complete=complete)


def history(target=TARGET):
    return [{"action": "Open menu", "operation": "click", "verified": False,
             "postconditions": [{"kind": "exists", "target": asdict(target), "outcome": "unknown",
                                  "reason": "target_missing"}]}]


def test_observed_full_menu_label_explains_missing_predicate_without_changing_truth():
    actual = control()
    current, previous = state(actual), history()
    original = copy.deepcopy(previous), current.to_dict()
    predicate = AgentCondition("exists", TARGET)
    assert condition_evidence(predicate, current) == {"outcome": "unknown", "reason": "target_missing"}
    report = observed_target_diagnostics(current, previous)
    item, = report["predicates"]
    assert item["retrieval"] == "menu_caption_and_shortcut_text"
    assert item["historical_predicate"]["target"] == asdict(TARGET)
    shown, = item["observed_controls"]
    assert shown["label"] == actual.label and shown["window"] == ""
    assert shown["operations"] == ["click"]
    assert "NOT a matching alias" in report["guidance"]
    assert not executable_controls(current, "click", TARGET)
    assert executable_controls(current, "click", replace(TARGET, name=shown["label"])) == (actual,)
    assert condition_evidence(predicate, current)["outcome"] == "unknown"
    assert (previous, current.to_dict()) == original


@pytest.mark.parametrize("suffix", ["Ctrl+N", "Alt+Ctrl+N", "Ctrl+Shift+S", "Ctrl+K Ctrl+N"])
def test_bounded_shortcut_text_retrieval_keeps_the_actual_full_label(suffix):
    actual = control("New document " + suffix)
    report = observed_target_diagnostics(state(actual), history())
    assert report["predicates"][0]["observed_controls"][0]["label"] == actual.label


@pytest.mark.parametrize("label,role", [
    ("New document soon", "menuitem"), ("New document Ctrl+", "menuitem"),
    ("New document N", "menuitem"), ("New document Ctrl+N extra", "menuitem"),
    ("New documents Ctrl+N", "menuitem"), ("New document Ctrl+N", "textbox"),
    ("New document Ctrl+N Ctrl+O Ctrl+P Ctrl+Q", "menuitem"),
])
def test_no_arbitrary_fuzzy_caption_or_nonmenu_expansion(label, role):
    assert observed_target_diagnostics(state(control(label, role=role)), history()) is None


@pytest.mark.parametrize("changes", [
    {"app": "Another"}, {"window": "NamedParent"}, {"ancestor": "Different context"},
])
def test_requested_ownership_filters_are_never_erased(changes):
    target = replace(TARGET, **changes)
    assert observed_target_diagnostics(state(control()), history(target)) is None


def test_exact_name_diagnostic_is_preserved_without_caption_expansion():
    actual = control("New document")
    report = observed_target_diagnostics(state(actual, control(ref="legend")), history())
    item, = report["predicates"]
    assert item["match_count"] == 1 and item["observed_controls"][0]["ref"] == actual.ref
    assert "retrieval" not in item


def test_disabled_and_ambiguous_caption_peers_are_reported_without_false_uniqueness():
    controls = [control("New document Ctrl+N", ref="a"),
                control("New document Alt+N", ref="b", disabled=True)]
    item, = observed_target_diagnostics(state(*controls), history())["predicates"]
    assert item["match_count"] == 2 and len(item["observed_controls"]) == 2
    assert item["observed_controls"][1]["operations"] == []
    assert not executable_controls(state(*controls), "click", TARGET)


def test_private_matching_peer_suppresses_the_entire_retrieval():
    current = state(control(), control(ref="private", input_type="password"))
    assert observed_target_diagnostics(current, history()) is None


def test_output_stays_bounded_and_does_not_include_private_values():
    controls = [control(ref=f"item{i}", value="private value", edit_buffer="private buffer")
                for i in range(7)]
    report = observed_target_diagnostics(state(*controls), history())
    item, = report["predicates"]
    assert item["match_count"] == 7 and item["truncated"] and len(item["observed_controls"]) == 4
    rendered = json.dumps(report)
    assert all(text not in rendered for text in ("private value", "private buffer", "private page text"))
    assert observed_target_diagnostics(state(*controls, complete=False), history()) is None


def test_new_action_ends_caption_retrieval_context():
    previous = history() + [{"action": "New action", "verified": True}]
    assert observed_target_diagnostics(state(control()), previous) is None
