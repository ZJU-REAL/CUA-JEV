"""Condition evidence distinguishes unsupported or unavailable state from false."""

from dataclasses import replace

import pytest

from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, condition_holds, observable_condition_kinds


def snapshot(*controls, complete=True, text="Current settings", title="Settings"):
    return AgentSnapshot("osworld://desktop", title, text, tuple(controls), a11y_complete=complete)


def control(**changes):
    values = dict(ref="c0", label="Option", tag="container", role="checkbox", input_type="checkbox")
    return AgentControl(**(values | changes))


@pytest.mark.parametrize(("kind", "changes", "expected"), [
    ("checked", {"checked": True}, True),
    ("checked", {"checked": False}, False),
    ("unchecked", {"checked": False}, True),
    ("unchecked", {"checked": True}, False),
    ("selected", {"selected": True}, True),
    ("selected", {"selectable": True}, False),
    ("expanded", {"expanded": True}, True),
    ("expanded", {"expandable": True}, False),
])
def test_observed_state_is_true_or_false_independent_of_action_legality(kind, changes, expected):
    state = snapshot(control(disabled=True, box=None, **changes))
    condition = AgentCondition(kind, AgentTarget(name="Option"))
    evidence = condition_evidence(condition, state)
    assert evidence == {
        "outcome": "true" if expected else "false",
        "reason": "condition_met" if expected else "state_mismatch",
    }
    assert condition_holds(condition, state) is expected


@pytest.mark.parametrize(("state", "condition", "reason"), [
    (snapshot(), AgentCondition("checked", AgentTarget(name="Missing")), "target_missing"),
    (snapshot(control(), control(ref="c1")), AgentCondition("checked", AgentTarget(name="Option")),
     "ambiguous_target"),
    (snapshot(control(role="listitem", input_type="", tag="button")),
     AgentCondition("checked", AgentTarget(name="Option")), "unsupported_state"),
    (snapshot(control(role="label", input_type="", tag="text")),
     AgentCondition("focused", AgentTarget(name="Option")), "unsupported_state"),
    (snapshot(control(role="button", input_type="", tag="button")),
     AgentCondition("value_equals", AgentTarget(name="Option"), ""), "unsupported_state"),
    (snapshot(complete=False), AgentCondition("absent", AgentTarget(name="Missing")),
     "incomplete_observation"),
    (snapshot(text=""), AgentCondition("absent", AgentTarget(name="Missing")), "incomplete_observation"),
    (snapshot(), AgentCondition("exists", AgentTarget(name="Missing")), "target_missing"),
    (snapshot(), AgentCondition("checked", AgentTarget(app="Only scope")), "invalid_condition"),
    (snapshot(control()), AgentCondition("invented", AgentTarget(name="Option")), "invalid_condition"),
])
def test_unavailable_ambiguous_and_unsupported_conditions_are_unknown(state, condition, reason):
    assert condition_evidence(condition, state) == {"outcome": "unknown", "reason": reason}
    assert not condition_holds(condition, state)


def test_condition_capabilities_disambiguate_same_name_label_without_claiming_input_support():
    readonly = control(checked=True, disabled=True)
    label = control(ref="label", tag="text", role="label", input_type="", checked=True)
    supported = set(observable_condition_kinds(readonly))
    assert {"exists", "absent", "checked", "unchecked"} <= supported
    assert not {"focused", "value_equals"} & supported
    assert not {"checked", "unchecked", "focused", "value_equals"} & set(observable_condition_kinds(label))
    condition = AgentCondition("checked", AgentTarget(name="Option"))
    assert condition_evidence(condition, snapshot(readonly, label))["outcome"] == "true"
    assert condition_evidence(condition, snapshot(readonly, label, replace(readonly, ref="other")))[
        "outcome"
    ] == "unknown"


@pytest.mark.parametrize(("kind", "capability"), [("selected", "selectable"), ("expanded", "expandable")])
def test_optional_state_defaults_require_capability_evidence(kind, capability):
    ordinary = control(tag="text", role="label", input_type="")
    condition = AgentCondition(kind, AgentTarget(name="Option"))
    assert kind not in observable_condition_kinds(ordinary)
    assert condition_evidence(condition, snapshot(ordinary)) == {
        "outcome": "unknown", "reason": "unsupported_state"
    }
    capable = replace(ordinary, **{capability: True})
    assert kind in observable_condition_kinds(capable)
    assert condition_evidence(condition, snapshot(capable))["outcome"] == "false"
    positive_legacy = replace(ordinary, **{kind: True})
    assert kind in observable_condition_kinds(positive_legacy)
    assert condition_evidence(condition, snapshot(positive_legacy))["outcome"] == "true"


def test_exists_and_absent_preserve_boolean_semantics_for_multiple_or_incomplete_observations():
    target = AgentTarget(name="Option")
    present = snapshot(control(), control(ref="c1"), complete=False)
    assert condition_holds(AgentCondition("exists", target), present)
    assert condition_evidence(AgentCondition("absent", target), present)["outcome"] == "false"
    absent = snapshot(control(label="Other"))
    assert condition_holds(AgentCondition("absent", target), absent)
    assert condition_evidence(AgentCondition("absent", target), absent)["outcome"] == "true"


def test_text_window_and_empty_values_have_value_free_evidence():
    state = snapshot(control(tag="input", role="textbox", input_type="text", value="", editable=True))
    cases = [
        (AgentCondition("value_equals", AgentTarget(name="Option"), ""), "true"),
        (AgentCondition("value_equals", AgentTarget(name="Option"), "private expected"), "false"),
        (AgentCondition("text_contains", value="Current"), "true"),
        (AgentCondition("text_contains", value="private missing"), "false"),
        (AgentCondition("window_contains", value="Settings"), "true"),
        (AgentCondition("window_contains", value="private missing"), "false"),
    ]
    for condition, expected in cases:
        evidence = condition_evidence(condition, state)
        assert evidence["outcome"] == expected
        assert set(evidence) == {"outcome", "reason"} and "private" not in str(evidence)
        assert condition_holds(condition, state) == (expected == "true")
    assert condition_evidence(
        AgentCondition("window_contains", value="Settings"), snapshot(title="OSWorld desktop")
    )["outcome"] == "unknown"
    assert condition_evidence(
        AgentCondition("text_contains", value="Missing"), snapshot(complete=False)
    )["outcome"] == "unknown"
