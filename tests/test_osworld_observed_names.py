"""Observed names retain evidence independently of the planner's query budget."""

from __future__ import annotations

import hashlib
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import pytest
from PIL import Image

from cua_jev.agent_planner import _grounded_navigation_scopes, parse_plan
from cua_jev.agent_types import AgentCondition, AgentTarget
from cua_jev.desktop_semantics import condition_evidence
from cua_jev.osworld_backend import OSWorldBackend


def observe(xml, *, size=(800, 600)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")

    class Saved:
        def _get_obs(self):
            return {"screenshot": buffer.getvalue(), "accessibility_tree": xml}

    backend = OSWorldBackend(Saved(), pause=0)
    try:
        return backend.observe()
    finally:
        backend.close()


def tree(*, app="Example", window="Editor", ancestor="Group", name="Control", body=""):
    return f'''<desktop><application name={quoteattr(app)}>
      <frame name={quoteattr(window)} active="true"><section name={quoteattr(ancestor)}>
        <button name={quoteattr(name)} screencoord="(20, 20)" size="(100, 40)"/>
        <paragraph>{escape(body)}</paragraph>
      </section></frame></application></desktop>'''


def missing(snapshot):
    return condition_evidence(AgentCondition("absent", AgentTarget(name="Definitely missing")), snapshot)


def test_long_observed_names_are_consistent_in_control_application_window_and_ancestors():
    app, window, ancestor, name = (prefix * count for prefix, count in [("A", 300), ("W", 400),
                                                                       ("S", 500), ("B", 600)])
    snapshot = observe(tree(app=app, window=window, ancestor=ancestor, name=name))
    target = next(control for control in snapshot.controls if control.role == "button")
    assert snapshot.a11y_complete and snapshot.title == window
    assert target.label == name and target.app == app and target.window == window
    assert target.ancestors == (f"frame: {window}", f"section: {ancestor}")
    assert all(value in snapshot.text for value in (app, window, ancestor, name))
    assert missing(snapshot)["outcome"] == "true"


@pytest.mark.parametrize("field", ["app", "window", "ancestor", "name", "body"])
def test_actual_suffix_after_240_changes_fingerprint_and_freshness_scope(field):
    first = observe(tree(**{field: "x" * 275 + "A"}))
    second = observe(tree(**{field: "x" * 275 + "B"}))
    assert first.a11y_complete and second.a11y_complete
    assert first.fingerprint() != second.fingerprint() and first.scope_key() != second.scope_key()


@pytest.mark.parametrize("field", ["app", "window", "ancestor", "name", "body"])
def test_exact_two_thousand_is_complete_but_any_lost_suffix_stays_incomplete(field):
    at_limit = observe(tree(**{field: "a" * 2000}))
    too_long = observe(tree(**{field: "a" * 2000 + "Z"}))
    assert at_limit.a11y_complete and not too_long.a11y_complete
    assert "a" * 2000 in too_long.text and "a" * 2000 + "Z" not in too_long.text
    assert missing(too_long) == {"outcome": "unknown", "reason": "incomplete_observation"}
    target = next(control for control in too_long.controls if control.role == "button")
    if field == "name":
        assert target.label == "a" * 2000
    elif field == "app":
        assert target.app == "a" * 2000
    elif field == "window":
        assert target.window == too_long.title == "a" * 2000
    elif field == "ancestor":
        assert target.ancestors[-1] == "section: " + "a" * 2000


@pytest.mark.parametrize("field", ["app", "window", "ancestor", "name", "body"])
def test_raw_whitespace_length_is_not_mistaken_for_lost_normalized_evidence(field):
    normalized = "alpha beta"
    expanded = " " * 1500 + "alpha" + " \n\t " * 300 + "beta" + " " * 1500
    snapshot = observe(tree(**{field: expanded}))
    reference = observe(tree(**{field: normalized}))
    assert snapshot.a11y_complete
    assert snapshot.to_dict() == reference.to_dict()
    assert snapshot.scope_key() == reference.scope_key()


@pytest.mark.parametrize("label_source", ["descendant", "sibling", "editable_row"])
@pytest.mark.parametrize("length", [276, 1000, 1001, 2000, 2001])
def test_associated_names_use_the_same_observation_budget(label_source, length):
    label = "L" * length
    if label_source == "descendant":
        branch = (f'<button screencoord="(0, 0)" size="(300, 40)">'
                  f'<label name={quoteattr(label)} screencoord="(0, 0)" size="(100, 40)"/></button>')
        role = "button"
    elif label_source == "sibling":
        branch = (f'<panel screencoord="(0, 0)" size="(300, 40)">'
                  f'<label name={quoteattr(label)} screencoord="(0, 0)" size="(100, 40)"/>'
                  '<check-box screencoord="(200, 0)" size="(30, 40)"/></panel>')
        role = "checkbox"
    else:
        branch = (f'<panel screencoord="(0, 0)" size="(300, 40)">'
                  f'<label name={quoteattr(label)} screencoord="(0, 0)" size="(100, 40)"/>'
                  '<entry editable="true" screencoord="(100, 0)" size="(200, 40)"/></panel>')
        role = "textbox"
    snapshot = observe(f'<desktop><application name="Example"><frame name="Editor" active="true">'
                       f'{branch}</frame></application></desktop>')
    target = next(control for control in snapshot.controls if control.role == role)
    assert target.label == label[:2000]
    assert snapshot.a11y_complete is (length <= 2000)


def test_normalization_does_not_hide_an_actual_suffix_loss():
    raw = " " * 1500 + "x" * 1999 + " y"
    snapshot = observe(tree(name=raw))
    assert not snapshot.a11y_complete
    target = next(control for control in snapshot.controls if control.role == "button")
    assert target.label == "x" * 1999 + " "
    assert missing(snapshot)["outcome"] == "unknown"


@pytest.mark.parametrize("budget", ["controls", "nodes", "total_text", "raw_xml"])
def test_existing_whole_observation_budgets_remain_fail_closed(budget):
    if budget == "controls":
        xml = "<desktop>" + "".join(f'<label name="label{i}"/>' for i in range(513)) + "</desktop>"
    elif budget == "nodes":
        xml = "<desktop>" + "<panel/>" * 6001 + "</desktop>"
    elif budget == "total_text":
        xml = "<desktop>" + (f'<label name="{"x" * 999}"/>' * 97) + "</desktop>"
    else:
        xml = "<desktop>" + " " * 2_000_000 + "</desktop>"
    snapshot = observe(xml)
    assert not snapshot.a11y_complete and len(snapshot.controls) <= 512
    assert len(snapshot.text) <= 96000 and missing(snapshot)["outcome"] == "unknown"


@pytest.mark.parametrize("field,value", [("name", "x" * 241), ("role", "x" * 81)])
def test_larger_observation_budget_does_not_expand_target_query_schema(field, value):
    snapshot = observe(tree(name="x" * 300))
    raw = {"summary": "Query remains bounded", "done": False, "steps": [
        {"id": "s1", "operation": "click", "target": {field: value},
         "postconditions": [{"kind": "focused", "target": {"name": "Control"}}]},
    ]}
    with pytest.raises(ValueError, match="semantic target field"):
        parse_plan(raw, snapshot, profile="desktop")


def test_action_label_and_input_value_limits_remain_unchanged():
    snapshot = observe(tree())
    raw = {"summary": "Bounded label", "done": False, "steps": [
        {"id": "s1", "operation": "click", "label": "x" * 241,
         "target": {"name": "Control"},
         "postconditions": [{"kind": "focused", "target": {"name": "Control"}}]},
    ]}
    with pytest.raises(ValueError, match="invalid label"):
        parse_plan(raw, snapshot, profile="desktop")
    raw["steps"][0] = {"id": "s1", "operation": "type_text", "value": "x" * 2001}
    with pytest.raises(ValueError, match="invalid value"):
        parse_plan(raw, snapshot, profile="desktop")


def test_recorded_vscode_section_is_retained_whole_without_overriding_completeness():
    path = Path(__file__).parent / "fixtures" / "osworld_long_name_v16.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert fixture["source"]["xml_sha256"] == (
        "84a55f97a57ef48cf68dcb7df679e9a450f06b1f50c4f4a8c1ca3d505f7b8bb0"
    )
    raw = fixture["xml"]
    assert hashlib.sha256(raw.encode()).hexdigest() == fixture["fixture_xml_sha256"]
    root = ET.fromstring(raw)
    section = next(node for node in root.iter() if hashlib.sha256(
        node.attrib.get("name", "").encode(),
    ).hexdigest() == "c20e5dfa1ea9878df12b17539f9ec6e92abd36bc9c023ff45481bcfadff3a21a")
    label = section.attrib["name"]
    assert len(label) == 276
    assert len(list(section.iter())) == fixture["source"]["section_nodes"] == 15
    assert [child.tag for child in section] == [
        "paragraph", "push-button", "paragraph", "paragraph", "push-button", "paragraph",
    ]
    snapshot = observe(raw, size=tuple(fixture["screen_size"]))
    assert snapshot.a11y_complete
    assert any(control.label == label and control.role == "section" for control in snapshot.controls)
    buttons = {control.label for control in snapshot.controls if control.role == "button"}
    assert buttons == {"Open Folder", "Clone Repository", "add a folder", "read our docs"}
    scopes = _grounded_navigation_scopes(snapshot)
    assert {"target": {"app": "code"}, "candidate_count": 4} in scopes


def test_recorded_full_vscode_retains_long_name_text_and_controls_after_former_cutoff():
    path = Path(__file__).parent / "fixtures/osworld_vscode_large_observation.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture["xml"].encode()).hexdigest() == fixture["xml_sha256"]
    assert fixture["source"]["sha256"] == fixture["xml_sha256"]
    snapshot = observe(fixture["xml"], size=tuple(fixture["screen_size"]))
    assert snapshot.a11y_complete and len(snapshot.controls) == 351
    assert snapshot.title == fixture["title"]
    assert len(snapshot.text) == fixture["text_chars"] == 25391
    long_control = next(control for control in snapshot.controls if len(control.label) == 1804)
    assert hashlib.sha256(long_control.label.encode()).hexdigest() == fixture["long_control_name_sha256"]
    assert long_control.label in snapshot.text
    assert [control.label for control in snapshot.controls[347:]] == fixture["tail_labels"]
    assert fixture["tail_labels"] == ["Ubuntu Software", "Help", "Trash", "Show Applications"]
    assert missing(snapshot)["outcome"] == "true"
    scopes = _grounded_navigation_scopes(snapshot)
    assert any(s["target"] == {"app": "code"} and 1 < s["candidate_count"] <= 255 for s in scopes)
    root = ET.fromstring(fixture["xml"])
    node = next(n for n in root.iter() if " ".join(n.attrib.get("name", "").split()) == long_control.label)
    node.set("name", long_control.label[:-1] + "Z")
    changed = observe(ET.tostring(root, encoding="unicode"), size=tuple(fixture["screen_size"]))
    assert changed.a11y_complete
    assert changed.fingerprint() != snapshot.fingerprint() and changed.scope_key() != snapshot.scope_key()


def test_long_common_prefix_never_hides_a_second_editable_field():
    prefix = "field" * 300
    xml = ('<desktop><application name="Example"><frame name="Editor" active="true">'
           f'<entry name="{prefix}A" editable="true" value="desired" '
           'screencoord="(20, 20)" size="(100, 40)"/>'
           f'<entry name="{prefix}B" editable="true" value="other" '
           'screencoord="(20, 80)" size="(100, 40)"/>'
           '</frame></application></desktop>')
    snapshot = observe(xml)
    assert snapshot.a11y_complete
    fields = [control for control in snapshot.controls if control.role == "textbox"]
    assert [control.label for control in fields] == [prefix + "A", prefix + "B"]
    ready = AgentCondition("ready_to_fill", AgentTarget(role="textbox", app="Example"), "desired")
    assert condition_evidence(ready, snapshot) == {"outcome": "unknown", "reason": "ambiguous_target"}
