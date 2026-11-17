"""Display-only line-break hints must not establish a false absence."""

from dataclasses import replace

import pytest

from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, executable_controls, matching_controls


def state(*controls):
    return AgentSnapshot("osworld://desktop", "Desktop", "Visible desktop", tuple(controls))


def item(label, **kwargs):
    return AgentControl("d1", label, "button", "listitem", box=(100, 100, 200, 200),
                        app="Files", window="Desktop", selectable=True, **kwargs)


def test_zero_width_wrap_hint_cannot_make_existing_file_absent():
    # The v11 OS rename trace exposed U+200B inside a displayed file label.
    raw = item("todo_\u200blist_\u200bJan_1")
    snapshot = state(raw)
    target = AgentTarget(name="todo_list_Jan_1", app="Files")
    assert condition_evidence(AgentCondition("absent", target), snapshot)["outcome"] == "false"
    assert condition_evidence(AgentCondition("exists", target), snapshot)["outcome"] == "true"
    assert executable_controls(snapshot, "click", target) == (raw,)
    assert raw.identity()[0] == "todo_\u200blist_\u200bJan_1"
    assert snapshot.to_dict()["controls"][0]["label"] == raw.label
    changed = replace(raw, label="todo_\u200blist_\u200bJan_2")
    assert condition_evidence(AgentCondition("absent", target), state(changed))["outcome"] == "true"


def test_display_equivalent_controls_remain_ambiguous_for_state_predicates():
    first = item("my_\u200bfile", selected=True)
    second = replace(first, ref="d2", label="my_file", selected=False)
    target = AgentTarget(name="my_file")
    assert matching_controls(state(first, second), target) == (first, second)
    assert condition_evidence(AgentCondition("selected", target), state(first, second)) == {
        "outcome": "unknown", "reason": "ambiguous_target",
    }


@pytest.mark.parametrize("character", ["\u200c", "\u200d", "\ufe0f", "\u2060", "\u00ad"])
def test_other_unicode_characters_are_not_erased(character):
    assert not matching_controls(state(item(f"my{character}file")), AgentTarget(name="myfile"))


@pytest.mark.parametrize("kind", ["exists", "absent", "selected", "text_contains", "window_contains"])
def test_invisible_only_predicates_fail_closed(kind):
    condition = (AgentCondition(kind, value="\u200b") if kind.endswith("contains")
                 else AgentCondition(kind, AgentTarget(name="\u200b")))
    assert condition_evidence(condition, state(item("a"))) == {
        "outcome": "unknown", "reason": "invalid_condition",
    }


def test_invisible_only_scope_cannot_match_every_application():
    assert not executable_controls(state(item("a")), "click", AgentTarget(app="\u200b"), allow_scope=True)


@pytest.mark.parametrize("field", ["role", "app", "window", "ancestor"])
def test_nonempty_invisible_qualifier_is_not_silently_ignored(field):
    target = AgentTarget(name="Save", **{field: "\u200b"})
    snapshot = state(item("Save"))
    assert not matching_controls(snapshot, target)
    assert condition_evidence(AgentCondition("absent", target), snapshot) == {
        "outcome": "unknown", "reason": "invalid_condition",
    }


def test_exact_field_value_is_not_display_normalized():
    field = AgentControl("f1", "Name", "input", "textbox", editable=True, value="my\u200bfile")
    assert condition_evidence(AgentCondition("value_equals", AgentTarget(name="Name"), "myfile"),
                              state(field))["outcome"] == "false"
