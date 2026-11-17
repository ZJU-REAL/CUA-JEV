"""Bounded form-row label association, replayed without a VM or evaluator."""

from __future__ import annotations

import copy
import io
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PIL import Image

from cua_jev.agent_types import AgentCondition, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, executable_controls
from cua_jev.osworld_backend import OSWorldBackend, _accessibility_observation


def snapshot(xml, size=(800, 600)):
    controls = []
    title, text, _ = _accessibility_observation(xml, screen_size=size, controls_out=controls)
    return AgentSnapshot("osworld://desktop", title, text, tuple(controls))


def xml_text(root):
    return ET.tostring(root, encoding="unicode")


@pytest.fixture
def recorded():
    return json.loads((Path(__file__).parent / "fixtures" / "osworld_editable_rows_v11.json").read_text(
        encoding="utf-8"
    ))


def row_tree():
    return ET.fromstring('''<desktop><application name="Editor"><dialog name="Options" active="true">
      <panel id="table" name="Settings" screencoord="(10, 10)" size="(600, 100)">
        <label id="label" name="History limit:" screencoord="(20, 20)" size="(180, 30)"/>
        <filler id="branch" screencoord="(210, 20)" size="(300, 30)">
          <spin-button id="field" name="" editable="true" focusable="true" value="5.0"
            screencoord="(210, 20)" size="(100, 30)">5</spin-button>
        </filler>
        <label id="other-label" name="Threads:" screencoord="(20, 70)" size="(180, 30)"/>
        <filler id="other-branch" screencoord="(210, 70)" size="(300, 30)">
          <spin-button id="other-field" name="" editable="true" focusable="true" value="2.0"
            screencoord="(210, 70)" size="(100, 30)">2</spin-button>
        </filler>
      </panel></dialog></application></desktop>''')


def element(root, name):
    return next(node for node in root.iter() if node.get("id") == name)


def first_field(root):
    return next(control for control in snapshot(xml_text(root)).controls if control.value == "5.0")


def test_recorded_five_anonymous_fields_acquire_their_own_observed_row_labels(recorded):
    observed = snapshot(recorded["xml"], tuple(recorded["screen_size"]))
    fields = [control for control in observed.controls if control.role == "spinbutton"]
    assert [(c.label, c.value, c.box) for c in fields] == [
        ("Number of threads to use:", "2.0", (302, 695, 338, 727)),
        ("Maximum new image size:", "128.0", (302, 621, 352, 653)),
        ("Tile cache size:", "1997486.0", (302, 584, 352, 616)),
        ("Maximum undo memory:", "499372.0", (302, 547, 352, 579)),
        ("Minimal number of undo levels:", "5.0", (302, 510, 352, 542)),
    ]
    # XML text=100 is only the edit buffer of the *threads* widget. Preserve
    # its authoritative value=2.0; label recovery must not invent a commit.
    assert "100" in recorded["xml"] and fields[0].focused
    target = AgentTarget(name="Minimal number of undo levels:", role="spinbutton", app="gimp")
    assert executable_controls(observed, "fill", target) == (fields[-1],)
    assert condition_evidence(AgentCondition("value_equals", target, "5.0"), observed)["outcome"] == "true"
    assert condition_evidence(AgentCondition("value_equals", target, "100"), observed)["outcome"] == "false"
    assert [c.label for c in observed.controls if c.role == "combobox"] == [
        "fast", "Mebibyte", "Kibibyte", "Kibibyte",
    ]


def test_fixture_records_exact_raw_provenance_without_requiring_local_run_artifacts(recorded):
    assert recorded["source"].endswith("/vlm/step-003-a11y.xml")
    assert recorded["raw_sha256"] == "652297c2e45ba9b12daa331a08ad5b75217031eab1c99065d279a2f6504c3599"
    assert len(ET.fromstring(recorded["xml"]).findall(".//spin-button")) == 5


def test_recorded_mapping_is_invariant_to_child_order(recorded):
    root = ET.fromstring(recorded["xml"])
    for node in root.iter():
        node[:] = reversed(list(node))
    before = snapshot(recorded["xml"], tuple(recorded["screen_size"]))
    after = snapshot(xml_text(root), tuple(recorded["screen_size"]))
    assert {c.box: (c.label, c.value) for c in before.controls if c.role == "spinbutton"} == {
        c.box: (c.label, c.value) for c in after.controls if c.role == "spinbutton"
    }


@pytest.mark.parametrize("role", ["spin-button", "entry", "text-entry", "text"])
def test_plain_anonymous_editable_roles_can_use_a_unique_form_row(role):
    root = row_tree()
    element(root, "field").tag = role
    field = first_field(root)
    assert field.label == "History limit:"
    assert field.editable and field.tag == "input"


@pytest.mark.parametrize("depth", [0, 1])
def test_nested_transparent_wrapper_and_named_noneditable_units_keep_unique_field(depth):
    root = row_tree()
    branch, field = element(root, "branch"), element(root, "field")
    if depth:
        branch.remove(field)
        inner = ET.SubElement(branch, "filler", screencoord="(210, 20)", size="(250, 30)")
        inner.append(field)
    else:
        inner = branch
    ET.SubElement(inner, "combo-box", name="Units", screencoord="(320, 20)", size="(100, 30)")
    assert first_field(root).label == "History limit:"
    assert next(c for c in snapshot(xml_text(root)).controls if c.role == "combobox").label == "Units"


@pytest.mark.parametrize("part,attribute,value", [
    ("label", "screencoord", "(20, 21)"),
    ("label", "size", "(180, 29)"),
    ("label", "screencoord", "(220, 20)"),
    ("label", "screencoord", "(0, 20)"),
    ("label", "visible", "false"),
    ("label", "editable", "true"),
    ("label", "focusable", "true"),
    ("branch", "focusable", "true"),
    ("branch", "editable", "true"),
    ("branch", "screencoord", "(215, 20)"),
    ("branch", "size", "(300, 31)"),
    ("table", "focusable", "true"),
    ("table", "size", "(400, 100)"),
    ("field", "editable", "false"),
    ("field", "size", "(100, 29)"),
])
def test_missing_or_ambiguous_structural_row_evidence_does_not_guess(part, attribute, value):
    root = row_tree()
    element(root, part).set(attribute, value)
    assert first_field(root).label == ""


@pytest.mark.parametrize("part", ["label", "branch", "table", "field"])
def test_every_association_component_requires_observed_bounds(part):
    root = row_tree()
    element(root, part).attrib.pop("screencoord")
    assert first_field(root).label == ""


@pytest.mark.parametrize("duplicate", ["label", "branch", "field"])
def test_same_row_duplicate_label_or_editable_owner_fails_closed(duplicate):
    root = row_tree()
    parent = element(root, "branch" if duplicate == "field" else "table")
    parent.append(copy.deepcopy(element(root, duplicate)))
    assert first_field(root).label == ""


def test_second_editable_with_a_different_role_is_still_ambiguous():
    root = row_tree()
    ET.SubElement(element(root, "branch"), "entry", name="Other", editable="true",
                  screencoord="(320, 20)", size="(100, 30)")
    assert first_field(root).label == ""


@pytest.mark.parametrize("change", ["editable", "nested_input", "overlap", "different_row", "unnamed"])
def test_units_selector_exception_cannot_hide_another_input_or_bad_geometry(change):
    root = row_tree()
    selector = ET.SubElement(element(root, "branch"), "combo-box", name="Units",
                             screencoord="(320, 20)", size="(100, 30)")
    if change == "editable":
        selector.set("editable", "true")
    elif change == "nested_input":
        ET.SubElement(selector, "entry", editable="true", screencoord="(320, 20)", size="(80, 30)")
    elif change == "overlap":
        selector.set("screencoord", "(290, 20)")
    elif change == "different_row":
        selector.set("size", "(100, 29)")
    else:
        selector.set("name", "")
    assert first_field(root).label == ""


def test_hidden_duplicate_does_not_compete_with_visible_row():
    root = row_tree()
    duplicate = copy.deepcopy(element(root, "label"))
    duplicate.set("visible", "false")
    element(root, "table").append(duplicate)
    assert first_field(root).label == "History limit:"


def test_native_accessible_name_always_wins():
    root = row_tree()
    element(root, "field").set("name", "Explicit field name")
    assert first_field(root).label == "Explicit field name"


@pytest.mark.parametrize("boundary", ["dialog", "button", "viewport"])
def test_label_search_never_crosses_nontransparent_boundaries(boundary):
    root = row_tree()
    element(root, "branch").tag = boundary
    assert first_field(root).label == ""


def test_label_search_has_a_fixed_wrapper_depth_limit():
    root = row_tree()
    branch, field = element(root, "branch"), element(root, "field")
    branch.remove(field)
    for _ in range(2):
        branch = ET.SubElement(branch, "filler", screencoord="(210, 20)", size="(300, 30)")
    branch.append(field)
    assert first_field(root).label == ""


def test_duplicate_branch_in_a_different_row_is_not_a_competing_owner():
    root = row_tree()
    assert first_field(root).label == "History limit:"
    fields = [c for c in snapshot(xml_text(root)).controls if c.role == "spinbutton"]
    assert [c.label for c in fields] == ["History limit:", "Threads:"]


def test_derived_label_preserves_freshness_and_does_not_add_a_commit_action():
    root = row_tree()
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, "PNG")

    class Environment:
        def __init__(self):
            self.calls = []

        def _get_obs(self):
            return {"screenshot": buffer.getvalue(), "accessibility_tree": xml_text(root)}

        def step(self, action, pause):
            self.calls.append(action)
            return self._get_obs(), 0, False, {}

    env = Environment()
    backend = OSWorldBackend(env, pause=0)
    try:
        before = backend.observe()
        step = AgentStep("edit", "fill", value="100", target=AgentTarget(name="History limit:"))
        candidates = backend.compile(step, before, before)
        assert len(candidates) == 1
        # Changing only the label association invalidates this previously bound
        # candidate before any click, typing, or implicit commit can be sent.
        element(root, "label").set("name", "Different setting:")
        receipt = backend.execute(candidates[0], "obs", "decision")
        assert not receipt.success and env.calls == []
    finally:
        backend.close()
