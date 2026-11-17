"""Versioned prompt bases plus the shared numeric-edit diagnostic guidance."""

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import (
    MultimodalPlanner,
    _desktop_instructions,
    _validate_desktop_execution_plan,
    parse_plan,
)
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.desktop_semantics import conditions_hold
from cua_jev.desktop_stage_prompt import PENDING_NUMERIC_EDIT_GUIDANCE
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image


def state(populated=True, *, image=False):
    controls = (AgentControl("menu", "Settings", "button", role="menuitem", app="Editor",
                             box=(100, 100, 200, 150)),) if populated else ()
    screenshot = None
    if image:
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buffer, "PNG")
        screenshot = screenshot_image(buffer.getvalue())
    return AgentSnapshot("osworld://desktop", "Editor", "Current editor", controls, image=screenshot)


@pytest.mark.parametrize("populated", [False, True])
def test_single_action_preserves_v17_base_and_adds_shared_numeric_guidance(populated):
    # These hashes were generated from the frozen source whose SHA is retained
    # in the fixture; the regression does not depend on ignored runs artifacts.
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "desktop_single_action_prompt_v17.json").read_text(encoding="utf-8"))
    expected = next(case for case in fixture["cases"] if case["has_controls"] == populated)
    prompt = _desktop_instructions(1, state(populated))
    assert prompt.endswith(PENDING_NUMERIC_EDIT_GUIDANCE)
    assert prompt.count(PENDING_NUMERIC_EDIT_GUIDANCE) == 1
    base = prompt.removesuffix(PENDING_NUMERIC_EDIT_GUIDANCE)
    assert len(base) == expected["characters"]
    assert hashlib.sha256(base.encode("utf-8")).hexdigest() == expected["sha256"]
    assert "reach_edit" not in prompt


@pytest.mark.parametrize("limit", [2, 6, 12])
def test_stage_first_hybrid_matches_v23_golden_and_keeps_numeric_guidance(limit):
    fixtures = Path(__file__).parent / "fixtures"
    versioned = json.loads((fixtures / "desktop_stage_prompt_v23.json").read_text(encoding="utf-8"))
    prompt = _desktop_instructions(limit, state())
    assert prompt.endswith(PENDING_NUMERIC_EDIT_GUIDANCE)
    assert prompt.count(PENDING_NUMERIC_EDIT_GUIDANCE) == 1
    base = prompt.removesuffix(PENDING_NUMERIC_EDIT_GUIDANCE)
    assert hashlib.sha256(base.encode("utf-8")).hexdigest() == versioned["prompt_sha256"][str(limit)]
    assert prompt.startswith("PLAN ONE MEANINGFUL DESKTOP STAGE.")
    assert "do not return reach_edit or repeat that edit" in prompt
    assert "PREFER reach_edit even when" in prompt
    assert "A nearby static label with no fill operation is not that field's name" in prompt


@pytest.mark.parametrize("format_name", ["intent", "navigation", "keyboard", "visual"])
def test_production_transport_accepts_intent_and_existing_legacy_fallbacks(format_name):
    raw = {
        "intent": {"intent": "reach_edit", "label": "Enable backups", "scope": {"app": "Editor"},
                   "edit": {"operation": "check", "target": {"name": "Backups", "app": "Editor"}},
                   "completion": "stage"},
        "navigation": {"summary": "Reach preferences", "done": False, "steps": [{
            "id": "nav", "operation": "click", "navigation": True, "label": "Reach preferences",
            "target": {"app": "Editor"},
            "postconditions": [{"kind": "exists", "target": {"name": "Preferences", "app": "Editor"}}],
        }]},
        "keyboard": {"summary": "Open command palette", "done": False, "steps": [{
            "id": "keys", "operation": "hotkey", "value": "ctrl+shift+p",
            "preconditions": [{"kind": "window_contains", "value": "Editor"}],
            "postconditions": [{"kind": "exists", "target": {"name": "Command", "app": "Editor"}}],
        }]},
        "visual": {"summary": "Open visible item", "done": False, "steps": [{
            "id": "visual", "operation": "visual_click", "label": "Visible item",
            "box_space": "normalized_1000", "box": [100, 100, 250, 200],
        }]},
    }[format_name]
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner = MultimodalPlanner("https://model.invalid/v1", "test", client=client, profile="desktop")
        plan = planner.plan("Perform the requested stage", state(image=True), [], max_actions=6)
    assert requests[0]["messages"][0]["content"].startswith("PLAN ONE MEANINGFUL DESKTOP STAGE.")
    assert len(requests) == 1 and not plan.done
    assert len(plan.steps) == (2 if format_name == "intent" else 1)
    assert not plan.completion_conditions
    assert plan.reusable is (format_name != "visual")


@pytest.mark.parametrize("recovery", [False, True])
def test_baseline_and_recovery_transport_share_numeric_guidance_with_preserved_base(recovery):
    requests = []
    raw = {"summary": "Observe", "done": False, "steps": [
        {"id": "wait", "operation": "wait", "value": "250"},
    ]}

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as client:
        planner = MultimodalPlanner("https://model.invalid/v1", "test", client=client, profile="desktop")
        planner.plan("Observe", state(), [{"recovery_required": True}] if recovery else [],
                     max_actions=6 if recovery else 1)
    prompt = requests[0]["messages"][0]["content"]
    assert prompt.startswith("RECOVERY PRIORITY:" if recovery else "IMMEDIATE ACTION CONTRACT:")
    assert "reach_edit" not in prompt
    immediate = prompt[prompt.index("IMMEDIATE ACTION CONTRACT:"):].removesuffix(
        " This call is limited to the ONE immediate next action."
    )
    assert immediate.count(PENDING_NUMERIC_EDIT_GUIDANCE) == 1
    immediate = immediate.removesuffix(PENDING_NUMERIC_EDIT_GUIDANCE)
    fixture = json.loads((Path(__file__).parent / "fixtures" /
                          "desktop_single_action_prompt_v17.json").read_text(encoding="utf-8"))
    expected = next(case for case in fixture["cases"] if case["has_controls"])
    assert hashlib.sha256(immediate.encode("utf-8")).hexdigest() == expected["sha256"]


@pytest.mark.parametrize("already_filled", [False, True])
def test_current_anonymous_edit_and_final_commit_use_existing_guarded_contract(already_filled):
    snapshot = state(image=True)
    snapshot = replace(snapshot, title="Properties", controls=(
        AgentControl("field", "", "input", role="textbox", app="Editor", window="Properties",
                     editable=True, focused=True, active=True,
                     value="Report" if already_filled else "Draft", box=(100, 100, 500, 200)),
        AgentControl("apply", "Apply", "button", role="button", app="Editor", window="Properties",
                     disabled=not already_filled, active=True, box=(600, 100, 700, 200)),
    ))
    field = {"role": "textbox", "app": "Editor", "window": "Properties"}
    fill = {
        "id": "edit", "operation": "fill", "target": field, "value": "Report",
        "preconditions": [{"kind": "focused", "target": field}],
        "postconditions": [{"kind": "value_equals", "target": field, "value": "Report"}],
    }
    commit = {
        "id": "apply", "operation": "click",
        "target": {"name": "Apply", "role": "button", "app": "Editor", "window": "Properties"},
        "depends_on": [] if already_filled else ["edit"],
        "preconditions": [{"kind": "value_equals", "target": field, "value": "Report"}],
        "postconditions": [{"kind": "absent", "target": field},
                           {"kind": "exists", "target": {"name": "Report", "app": "Editor",
                                                          "window": "Main"}}],
    }
    plan = parse_plan({"summary": "Apply the property", "done": False,
                       "steps": [commit] if already_filled else [fill, commit]},
                      snapshot, max_actions=6, profile="desktop")
    _validate_desktop_execution_plan(plan, snapshot, 6)
    backend = OSWorldBackend(object(), pause=0)
    assert len(plan.steps) == (1 if already_filled else 2)
    assert len(backend.compile(plan.steps[0], snapshot, snapshot)) == 1
    if not already_filled:
        assert backend.compile(plan.steps[-1], snapshot, snapshot) == ()
    after_fill = replace(snapshot, controls=(replace(snapshot.controls[0], value="Report"),
                                             replace(snapshot.controls[1], disabled=False)))
    final_step = plan.steps[-1]
    assert conditions_hold(final_step.preconditions, after_fill)
    assert not conditions_hold(final_step.postconditions, after_fill)
    assert len(backend.compile(final_step, snapshot, after_fill)) == 1
    ambiguous = replace(after_fill, controls=(*after_fill.controls,
                                             replace(after_fill.controls[0], ref="duplicate")))
    assert backend.compile(final_step, snapshot, ambiguous) == ()
    final = replace(snapshot, title="Main", controls=(
        AgentControl("result", "Report", "text", role="label", app="Editor", window="Main"),
    ))
    assert conditions_hold(final_step.postconditions, final)
    assert backend.environment_steps == backend.fresh_observe_calls == 0
