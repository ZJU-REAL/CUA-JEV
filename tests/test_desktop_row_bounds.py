"""Aggregate tree/list rows use their own label, never a descendant's center."""

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


def snapshot(root, size=(800, 600)):
    controls = []
    title, text, _ = _accessibility_observation(
        ET.tostring(root, encoding="unicode"), screen_size=size, controls_out=controls,
    )
    return AgentSnapshot("osworld://desktop", title, text, tuple(controls))


def row_tree():
    return ET.fromstring('''<desktop><application name="Files"><frame name="Folders" active="true">
      <tree-item id="parent" name="Projects" selectable="true" expanded="true"
        screencoord="(100, 100)" size="(300, 120)">
        <label id="label" name="Projects" screencoord="(140, 105)" size="(160, 20)"/>
        <panel id="children" screencoord="(100, 130)" size="(300, 90)">
          <tree-item id="child" name="Archive" selectable="true"
            screencoord="(100, 130)" size="(300, 30)">
            <label name="Archive" screencoord="(150, 135)" size="(160, 20)"/>
          </tree-item>
          <tree-item id="other" name="Current" selectable="true"
            screencoord="(100, 160)" size="(300, 30)"/>
        </panel>
      </tree-item></frame></application></desktop>''')


def element(root, name):
    return next(node for node in root.iter() if node.get("id") == name)


def parent_control(root):
    return next(control for control in snapshot(root).controls if control.label == "Projects")


@pytest.fixture
def recorded():
    return json.loads((Path(__file__).parent / "fixtures/osworld_tree_row_v11.json").read_text(
        encoding="utf-8",
    ))


def test_recorded_parent_label_avoids_the_trash_row_without_changing_row_identity(recorded):
    root = ET.fromstring(recorded["xml"])
    observed = snapshot(root, tuple(recorded["screen_size"]))
    parent = next(c for c in observed.controls if c.label == "Local Folders" and c.role == "treeitem")
    assert parent.box == (86, 391, 191, 406)
    assert parent.selectable and parent.expandable and parent.expanded
    assert parent.app == "Thunderbird" and parent.tag == "button"
    trash = next(c for c in observed.controls if c.label == "Trash" and c.role == "treeitem")
    assert parent.box[3] < trash.box[1]
    target = AgentTarget(name="Local Folders", role="treeitem", app="Thunderbird")
    assert executable_controls(observed, "click", target) == (parent,)
    assert condition_evidence(AgentCondition("exists", target), observed)["outcome"] == "true"
    assert recorded["raw_sha256"] == "891b9071916ed858d43ac06e9a74dc7021e1b866e3e5053816ec558e4add4182"
    assert any("/ns/action}" in key for node in root.iter() for key in node.attrib)


@pytest.mark.parametrize("role", ["tree-item", "list-item"])
def test_only_own_exact_label_rectangle_is_used_for_aggregate_row(role):
    root = row_tree()
    element(root, "parent").tag = role
    parent = parent_control(root)
    assert parent.box == (175, 175, 375, 208)
    assert parent.label == "Projects"


def test_reordering_children_and_same_named_descendant_cannot_steal_own_label():
    root = row_tree()
    child = element(root, "child")
    child.set("name", "Projects")
    child[0].set("name", "Projects")
    expected = parent_control(root).box
    for node in root.iter():
        node[:] = reversed(list(node))
    parent = next(c for c in snapshot(root).controls if c.label == "Projects" and c.expanded)
    assert parent.box == expected == (175, 175, 375, 208)


def test_collapsed_hidden_subtree_keeps_ordinary_row_bounds_then_expansion_uses_label():
    root = row_tree()
    row, children = element(root, "parent"), element(root, "children")
    row.set("expanded", "false")
    row.set("size", "(300, 30)")
    children.set("visible", "false")
    assert parent_control(root).box == (125, 167, 500, 217)
    row.set("expanded", "true")
    row.set("size", "(300, 120)")
    children.attrib.pop("visible")
    assert parent_control(root).box == (175, 175, 375, 208)


@pytest.mark.parametrize("part,field,value", [
    ("label", "name", "Project"),
    ("label", "visible", "false"),
    ("label", "editable", "true"),
    ("label", "focusable", "true"),
    ("label", "screencoord", "(140, 135)"),
    ("label", "screencoord", "(99, 105)"),
    ("label", "size", "(161, 120)"),
    ("child", "screencoord", "(140, 105)"),
    ("child", "size", "(301, 120)"),
    ("parent", "name", ""),
])
def test_no_trusted_own_region_leaves_row_observable_but_without_click_box(part, field, value):
    root = row_tree()
    element(root, part).set(field, value)
    observed = snapshot(root)
    parent = next(c for c in observed.controls if c.expanded)
    assert parent.box is None
    target = AgentTarget(role="treeitem", app="Files")
    assert parent not in executable_controls(observed, "click", target)
    assert condition_evidence(AgentCondition("exists", target), observed)["outcome"] == "true"


@pytest.mark.parametrize("part", ["parent", "label", "child", "children"])
def test_missing_geometry_never_restores_aggregate_center(part):
    root = row_tree()
    element(root, part).attrib.pop("screencoord")
    assert parent_control(root).box is None


def test_multiple_own_labels_are_ambiguous_even_when_they_have_same_text_and_box():
    root = row_tree()
    element(root, "parent").append(copy.deepcopy(element(root, "label")))
    assert parent_control(root).box is None


@pytest.mark.parametrize("focusable", [False, True])
def test_section_with_only_click_ancestor_is_parent_region_not_a_new_action(focusable):
    root = row_tree()
    label = element(root, "label")
    label.tag = "section"
    label.set("name", "")
    label.text = "Projects"
    label.set("focusable", str(focusable).lower())
    label.set("{https://example.test/ns/action}click-ancestor_desc", "Click ancestor")
    assert parent_control(root).box == (175, 175, 375, 208)
    assert len(executable_controls(snapshot(root), "click", AgentTarget(name="Projects"))) == 1


@pytest.mark.parametrize("action", ["click", "press", "activate"])
def test_independently_actionable_label_cannot_redirect_parent_click(action):
    root = row_tree()
    label = element(root, "label")
    label.set("{https://example.test/ns/action}click-ancestor_desc", "Click ancestor")
    label.set(f"{{https://example.test/ns/action}}{action}_desc", "Own action")
    assert parent_control(root).box is None


def test_no_label_search_crosses_an_independent_control_boundary():
    root = row_tree()
    row, label = element(root, "parent"), element(root, "label")
    row.remove(label)
    button = ET.SubElement(row, "button", name="Other", screencoord="(120, 100)", size="(200, 30)")
    button.append(label)
    assert parent_control(root).box is None


def test_child_row_behind_interactive_wrapper_still_invalidates_aggregate_center():
    root = row_tree()
    element(root, "children").set("focusable", "true")
    element(root, "parent").remove(element(root, "label"))
    assert parent_control(root).box is None


def test_nested_control_overlapping_label_invalidates_own_point():
    root = row_tree()
    ET.SubElement(element(root, "parent"), "button", name="More",
                  screencoord="(140, 105)", size="(100, 20)")
    assert parent_control(root).box is None


def test_visible_grandchild_outside_its_row_cannot_overlap_own_label_unnoticed():
    root = row_tree()
    ET.SubElement(element(root, "child"), "tree-item", name="Unexpected grandchild", selectable="true",
                  screencoord="(140, 105)", size="(100, 20)")
    assert parent_control(root).box is None


def test_focusable_label_exception_is_limited_to_parent_delegating_section():
    root = row_tree()
    label = element(root, "label")
    label.set("focusable", "true")
    label.set("{https://example.test/ns/action}click-ancestor_desc", "Click ancestor")
    assert parent_control(root).box is None


def test_nonaggregate_row_and_other_control_roles_keep_their_observed_bounds():
    root = row_tree()
    row = element(root, "parent")
    row.tag = "button"
    assert parent_control(root).box == (125, 167, 500, 367)
    row.tag = "tree-item"
    row.remove(element(root, "children"))
    assert parent_control(root).box == (125, 167, 500, 367)


def test_bounded_scan_does_not_fall_back_to_center_when_descendants_are_too_deep():
    root = row_tree()
    row, children = element(root, "parent"), element(root, "children")
    row.remove(children)
    branch = row
    for _ in range(7):
        branch = ET.SubElement(branch, "panel", screencoord="(100, 130)", size="(300, 90)")
    branch.append(children)
    assert parent_control(root).box is None


class Environment:
    def __init__(self, root):
        self.root, self.calls = root, []
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buffer, "PNG")
        self.png = buffer.getvalue()

    def _get_obs(self):
        return {"screenshot": self.png, "accessibility_tree": ET.tostring(self.root, encoding="unicode")}

    def step(self, action, pause):
        self.calls.append(action)
        return self._get_obs(), 0, False, {}


@pytest.mark.parametrize("button,count", [("left", 1), ("right", 1), ("left", 2)])
def test_typed_click_uses_own_label_and_preserves_explicit_button_count(button, count):
    env = Environment(row_tree())
    backend = OSWorldBackend(env, pause=0)
    try:
        before = backend.observe()
        step = AgentStep("row", "click", target=AgentTarget(name="Projects", role="treeitem"),
                         button=button, click_count=count)
        candidates = backend.compile(step, before, before)
        assert len(candidates) == 1
        receipt = backend.execute(candidates[0], "obs", "decision")
        assert receipt.success
        action = env.calls[0]
        assert action["action_type"] == "CLICK"
        assert action["parameters"] == {"x": 220, "y": 114, "button": button, "num_clicks": count}
        assert action["parameters"]["y"] < 130  # The nested child row starts here.
    finally:
        backend.close()


def test_moving_label_or_changing_nested_geometry_invalidates_fresh_preflight():
    root = row_tree()
    env, step = Environment(root), AgentStep("row", "click", target=AgentTarget(name="Projects"))
    backend = OSWorldBackend(env, pause=0)
    try:
        before = backend.observe()
        candidate = backend.compile(step, before, before)[0]
        element(root, "child").set("screencoord", "(140, 105)")
        assert not backend.execute(candidate, "obs", "decision").success
        assert env.calls == []
    finally:
        backend.close()
