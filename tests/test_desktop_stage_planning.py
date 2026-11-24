"""Hybrid stages and safe validation feedback without changing action authority."""

from __future__ import annotations

import copy
import io
import json

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _desktop_instructions, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.vision import WindowImage


@pytest.fixture
def snapshot():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, format="JPEG")
    return AgentSnapshot("osworld://desktop", "Editor", "Editor", (
        AgentControl("name", "Name", "input", role="textbox", app="Editor", editable=True,
                     box=(100, 100, 300, 200)),
        AgentControl("menu", "Menu", "button", role="menuitem", app="Editor", box=(100, 200, 300, 300)),
    ), image=WindowImage(buffer.getvalue(), bytes(64 * 64), (0, 0, 800, 600)))


def predicate(kind, name, **fields):
    return {"kind": kind, "target": {"name": name, "app": "Editor"}, **fields}


def fill(step_id="fill", name="Name", value="private-stage-content", **extra):
    return {"id": step_id, "operation": "fill", "target": {"name": name, "app": "Editor"},
            "value": value, "postconditions": [predicate("value_equals", name, value=value)], **extra}


def commit(name="Save"):
    return {"id": "commit", "operation": "click", "target": {"name": name, "app": "Editor"},
            "depends_on": ["fill"],
            "preconditions": [predicate("value_equals", "Name", value="private-stage-content")],
            "postconditions": [predicate("exists", "Saved item")]}


def plan(*steps):
    return {"summary": "Complete current stage", "steps": list(steps), "done": False}


def planner_for(raw):
    def response(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    return MultimodalPlanner("https://model.example/v1", "vlm", profile="desktop",
                             client=httpx.Client(transport=httpx.MockTransport(response)))


@pytest.mark.parametrize("name", ["Save", "Submit", "Menu"])
def test_persistent_guard_does_not_allow_an_intermediate_commit_or_menu(snapshot, name):
    raw = plan(fill(), commit(name), fill("next-object", "Other", depends_on=["commit"]))
    original = copy.deepcopy(raw)
    # Wire compatibility is unchanged; the live hybrid contract enforces stages.
    assert parse_plan(raw, snapshot, profile="desktop").reusable
    planner = planner_for(raw)
    with pytest.raises(ValueError):
        planner.plan("Create two separate objects", snapshot, [])
    message = planner.last_validation_error
    assert message.startswith("step 2 (steps[1]): guarded exact click must end the current stage")
    assert "done=false" in message and "replan the remaining work" in message
    assert "Do not relabel the commit as navigation" in message
    assert "invent a same-target local predicate" in message
    assert "whole-goal completion_conditions" in message
    assert "private-stage-content" not in message
    assert planner.calls == 1 and planner.last_raw is None and raw == original


def test_legal_current_stage_can_finish_at_guarded_commit_without_claiming_whole_goal(snapshot):
    planner = planner_for(plan(fill(), commit()))
    result = planner.plan("Create two separate objects", snapshot, [])
    assert result.reusable and not result.done and len(result.steps) == 2
    assert result.completion_conditions == ()
    assert result.steps[-1].preconditions[0].kind == "value_equals"
    assert not result.steps[-1].navigation


def test_real_same_target_local_effect_still_allows_later_edit(snapshot):
    target = {"name": "Name", "app": "Editor"}
    raw = plan({"id": "focus", "operation": "click", "target": target,
                "preconditions": [predicate("value_equals", "Name", value="old")],
                "postconditions": [{"kind": "focused", "target": target}]},
               fill(depends_on=["focus"]))
    assert len(planner_for(raw).plan("Edit text", snapshot, []).steps) == 2


def test_stage_guidance_is_front_loaded_and_keeps_navigation_with_jev(snapshot):
    instructions = _desktop_instructions(6, snapshot)
    prefix = instructions[:1000]
    assert prefix.startswith("PLAN ONE MEANINGFUL DESKTOP STAGE.")
    assert "Jev selects navigation controls" in prefix and "FIRST choose" in prefix
    assert "completion=goal only if this verified edit" in instructions
    assert "otherwise stage, particularly when confirmation remains" in instructions
    assert "Navigation step: operation=click,navigation=true" in instructions
    example_text = instructions.split("Example schema only: ")[1]
    example, _end = json.JSONDecoder().raw_decode(example_text)
    compiled = parse_plan(example, snapshot, profile="desktop")
    assert compiled.steps[0].navigation and not compiled.done
    assert compiled.steps[1].operation == "check" and compiled.completion_conditions
    assert _desktop_instructions(1, snapshot).startswith("IMMEDIATE ACTION CONTRACT:")


@pytest.mark.parametrize("operation,value,canonical", [
    ("key_press", "F2", "f2"), ("key_press", "Enter", "enter"),
    ("key_press", "ESC", "esc"), ("hotkey", "Ctrl+Shift+S", "ctrl+shift+s"),
    ("hotkey", "ALT+F4", "alt+f4"),
])
def test_ascii_key_case_variants_canonicalize_before_allowlist_validation(
    snapshot, operation, value, canonical,
):
    raw = plan({"id": "key", "operation": operation, "value": value})
    result = parse_plan(raw, snapshot, profile="desktop")
    assert result.steps[0].value == canonical and raw["steps"][0]["value"] == value


@pytest.mark.parametrize("operation,value", [
    ("key_press", "Ctrl+S"), ("key_press", "Return"), ("key_press", " Enter"),
    ("key_press", "Enter "), ("key_press", "İ"), ("key_press", "Ｆ2"),
    ("key_press", "F25"), ("hotkey", "CTRL+Ctrl+S"), ("hotkey", "Ctrl+ Shift+S"),
    ("hotkey", "Control+S"), ("hotkey", "Ｃtrl+S"), ("hotkey", "Ctrl+UNKNOWN"),
    ("key_press", "private-unrecognized-key"), ("hotkey", "ctrl+private-sensitive-value"),
    ("key_press", {"private": "private-non-string-value"}),
])
def test_invalid_keys_keep_indexed_static_shape_diagnostic_without_echo(snapshot, operation, value):
    raw = plan(fill(), {"id": "key", "operation": operation, "value": value,
                        "preconditions": [predicate("exists", "Name")],
                        "postconditions": [predicate("exists", "Result")]})
    planner = planner_for(raw)
    with pytest.raises(ValueError):
        planner.plan("Edit document", snapshot, [])
    message = planner.last_validation_error
    assert "steps[1].value:" in message and operation in message
    assert ('"enter"' if operation == "key_press" else '"ctrl+shift+s"') in message
    assert "private" not in message
    assert planner.last_raw is None and planner.calls == 1


@pytest.mark.parametrize("operation", ["type_text", "fill"])
def test_literal_input_case_is_unchanged(snapshot, operation):
    raw = plan(fill(value="MiXeD Case")) if operation == "fill" else plan(
        {"id": "literal", "operation": "type_text", "value": "MiXeD Case"},
    )
    assert parse_plan(raw, snapshot, profile="desktop").steps[0].value == "MiXeD Case"


def test_legacy_press_remains_case_sensitive(snapshot):
    raw = plan({"id": "key", "operation": "press", "value": "Enter"})
    assert parse_plan(raw, snapshot, profile="desktop").steps[0].value == "Enter"
    raw["steps"][0]["value"] = "enter"
    with pytest.raises(ValueError, match="press requires an allowed key"):
        parse_plan(raw, snapshot, profile="desktop")


def test_prompt_requests_lowercase_and_retains_single_action_case_guidance(snapshot):
    prompt = _desktop_instructions(6, snapshot)
    assert "hotkey: lowercase modifiers then one key" in prompt
    assert "key_press: one key" in prompt
    immediate = _desktop_instructions(1, snapshot)
    assert "ASCII case variants in key_press/hotkey are normalized" in immediate
    assert "whitespace and key aliases are not repaired" in immediate


def visual(step_id="visual"):
    return {"id": step_id, "operation": "visual_click", "label": "Visible control",
            "box": [100, 200, 300, 400], "box_space": "normalized_1000"}


@pytest.mark.parametrize("raw,existing_prefix", [
    (plan(fill(), visual()), "semantic desktop plans cannot mix refs, visual coordinates"),
    (plan(visual(), visual("future")), "desktop plans allow at most one visual action"),
])
def test_future_visual_coordinates_stay_rejected_with_single_visual_recovery_guidance(snapshot, raw,
                                                                                     existing_prefix):
    original = copy.deepcopy(raw)
    with pytest.raises(ValueError) as error:
        parse_plan(raw, snapshot, profile="desktop")
    assert str(error.value).startswith(existing_prefix)
    assert "one immediate visual_click or visual_fill" in str(error.value)
    assert "current screenshot" in str(error.value)
    assert raw == original
