"""V52 stopping/confirmation choices are explicit; older contracts remain unchanged."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner, VisualInspectionRequested, parse_plan
from cua_jev.desktop_output_schema import desktop_response_format
from cua_jev.desktop_property_protocol import completion_policy_property_protocol_instructions


@pytest.fixture
def native_initial(initial):
    # The preparation fixture deliberately has no native Search pointer target.
    # Current-field edits need real executable geometry under the existing guard.
    return replace(initial, controls=tuple(
        replace(control, box=(100, 100, 400, 200)) if control.ref == "search" else control
        for control in initial.controls
    ))


def property_response(*, completion="stage", confirm="none", key="none", value="on"):
    return {
        "intent": "pursue_property", "label": "Requested setting", "scope": {"app": "Editor"},
        "edit": {"operation": "set", "value": value}, "numeric_finalize": {"key": key},
        "confirm": confirm, "completion": completion,
    }


def semantic_response(kind):
    if kind == "pursue_property":
        return property_response()
    if kind == "navigate":
        return {"intent": kind, "label": "Requested destination", "scope": {"app": "Editor"},
                "arrival": [{"kind": "exists", "target": {"name": "Requested destination"}}],
                "completion": "stage"}
    return {"intent": "edit_fields", "label": "Edit current field", "completion": "stage",
            "edits": [{"operation": "fill", "target": {"name": "Search", "app": "Editor"},
                       "value": "requested text"}]}


def parse(raw, snapshot, *, policy=True, limit=6):
    return parse_plan(
        raw, snapshot, max_actions=limit, profile="desktop", allow_property_goal_conditions=True,
        allow_advisory_completion=True, inherit_current_action_label=True, allow_guarded_commit_only=True,
        allow_property_preparation=True, allow_property_value_binding=True, allow_dialog_commit_effect=True,
        allow_observed_property_goal=True, allow_current_commit_refs=True,
        allow_property_numeric_finalization=True, require_numeric_finalize_choice=True,
        require_explicit_stage_policy=policy,
    )


def schema(*, contract="property_intent_v52", limit=6, image=True):
    return desktop_response_format(
        limit, property_contract=contract, image_uploaded=image,
    )["json_schema"]["schema"]


@pytest.mark.parametrize("kind", ["pursue_property", "navigate", "edit_fields"])
@pytest.mark.parametrize("completion", ["stage", "goal"])
def test_each_semantic_stage_requires_and_preserves_its_completion_choice(native_initial, kind, completion):
    raw = semantic_response(kind) | {"completion": completion}
    assert Draft202012Validator(schema()).is_valid(raw)
    chosen = parse(raw, native_initial)
    if kind == "pursue_property":
        assert chosen.complete_property_goal is (completion == "goal")
    else:
        assert bool(chosen.completion_conditions) is (completion == "goal")
    assert chosen.done is False


@pytest.mark.parametrize("kind", ["pursue_property", "navigate", "edit_fields"])
@pytest.mark.parametrize("invalid", [None, True, [], {}, "all", "Goal"])
def test_bad_completion_is_rejected_by_schema_and_parser_without_type_errors(initial, kind, invalid):
    raw = semantic_response(kind) | {"completion": invalid}
    assert not Draft202012Validator(schema()).is_valid(raw)
    with pytest.raises(ValueError):
        parse(raw, initial)


@pytest.mark.parametrize("kind", ["pursue_property", "navigate", "edit_fields"])
def test_omission_is_new_only_requirement_and_old51_keeps_stage_default(native_initial, kind):
    raw = semantic_response(kind)
    raw.pop("completion")
    if kind == "pursue_property":
        raw.pop("confirm")
    assert Draft202012Validator(schema(contract="property_intent_v51")).is_valid(raw)
    original = parse(raw, native_initial, policy=False)
    assert not original.complete_property_goal and not original.completion_conditions
    assert not Draft202012Validator(schema()).is_valid(raw)
    with pytest.raises(ValueError):
        parse(raw, native_initial)


def test_literal_none_compiles_no_confirmation_or_numeric_key_and_preserves_old_plan(initial):
    raw = property_response()
    original_raw = copy.deepcopy(raw)
    original_raw.pop("confirm")
    chosen = parse(raw, initial)
    assert chosen == parse(original_raw, initial, policy=False)
    assert len(chosen.steps) == 1 and chosen.steps[0].property_edit.numeric_finalize_key == ""
    assert all(not step.property_confirmation_of and step.operation != "key_press" for step in chosen.steps)
    assert chosen.complete_property_goal is False
    with pytest.raises(ValueError):
        parse(raw, initial, policy=False)  # Old51 never accepts the new literal.


@pytest.mark.parametrize("invalid", [None, False, {}, "", "NONE", {"label": ""},
                                     {"label": None}, {"label": "Submit", "operation": "click"}])
def test_confirmation_choice_has_no_null_or_inferred_input_form(initial, invalid):
    raw = property_response(confirm=invalid)
    assert not Draft202012Validator(schema()).is_valid(raw)
    with pytest.raises(ValueError):
        parse(raw, initial)


def test_property_confirmation_cannot_be_omitted_and_label_remains_explicit(initial):
    raw = property_response()
    raw.pop("confirm")
    assert not Draft202012Validator(schema()).is_valid(raw)
    with pytest.raises(ValueError):
        parse(raw, initial)
    raw["confirm"] = {"label": "Submit requested form"}
    chosen = parse(raw, initial)
    assert len(chosen.steps) == 2
    assert chosen.steps[1].property_confirmation_of == chosen.steps[0].id
    assert chosen.steps[1].label == "Submit requested form"
    with pytest.raises(ValueError):
        parse(raw | {"confirm": {"label": "   "}}, initial)


@pytest.mark.parametrize("key", ["enter", "tab"])
def test_numeric_strategy_keeps_original_reserved_step_budget(initial, key):
    raw = property_response(key=key, value="42", confirm={"label": "Submit"})
    assert parse(raw, initial, limit=3).steps[0].property_edit.numeric_finalize_key == key
    with pytest.raises(ValueError, match="reserved original logical step"):
        parse(raw, initial, limit=2)
    with pytest.raises(ValueError, match="bounded numeric"):
        parse(raw | {"edit": {"operation": "set", "value": "on"}}, initial)


def test_explicit_goal_conditions_are_retained_with_either_completion_scope(initial):
    conditions = [{"kind": "window_contains", "value": "Main"}]
    for completion in ("stage", "goal"):
        chosen = parse(property_response(completion=completion) | {"goal_conditions": conditions}, initial)
        assert len(chosen.completion_conditions) == 1
        assert chosen.completion_conditions[0].kind == "window_contains"
        assert chosen.completion_conditions[0].value == "Main"
        assert chosen.complete_property_goal is (completion == "goal")


def request(snapshot, raw, *, contract="property_intent_v52", limit=6, observation="native_first_v51"):
    captured = []

    def answer(req):
        captured.append(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner(
            "https://offline.invalid", "fixture", api_key="offline", client=client, profile="desktop",
            desktop_contract=contract, desktop_output_format="json_schema_v50",
            desktop_planning_observation=observation,
        )
        try:
            result = planner.plan("Apply the requested setting", snapshot, [], max_actions=limit)
        except (ValueError, VisualInspectionRequested) as error:
            result = error
    assert planner.calls == len(captured) == 1
    return result, captured[0], planner


def test_planner52_selects_its_own_decoder_and_native_policy(initial):
    result, wire, planner = request(initial, property_response(completion="goal"))
    assert not isinstance(result, Exception) and result.complete_property_goal
    body = json.loads(wire)
    assert body["response_format"]["json_schema"]["name"] == "cua_desktop_stage_v52"
    assert [part["type"] for part in body["messages"][1]["content"]] == ["text"]
    assert planner.last_request_provenance == {
        "image_uploaded": False, "planning_observation": "native_first_v51",
        "max_actions": 6, "a11y_complete": True,
    }
    for field in ("completion", "confirm"):
        raw = property_response()
        raw.pop(field)
        failed, _wire, _planner = request(initial, raw)
        assert isinstance(failed, ValueError)


def test_act_current_is_only_current_action_not_automatic_goal(initial):
    raw = {"intent": "act_current", "label": "Current Help", "action": {"operation": "click", "ref": "help"}}
    chosen = parse(raw, initial)
    assert len(chosen.steps) == 1 and not chosen.done and not chosen.complete_property_goal
    assert not chosen.completion_conditions
    with pytest.raises(ValueError):
        parse(raw | {"completion": "goal"}, initial)


def test_no_image52_cannot_smuggle_visual_prepare_but_explicit_inspect_is_typed(initial):
    raw = property_response() | {"prepare": {
        "operation": "visual_click", "label": "Visible object", "box": [100, 100, 300, 200],
    }}
    assert not Draft202012Validator(schema(image=False)).is_valid(raw)
    failed, _wire, _planner = request(initial, raw)
    assert isinstance(failed, ValueError)
    inspected, _wire, planner = request(initial, {"intent": "inspect_visual", "label": "Need visual context"})
    assert isinstance(inspected, VisualInspectionRequested) and planner.calls == 1


def test_single_action52_is_still_exact_full_image51_wire(initial):
    raw = {"summary": "Use current Help", "done": False,
           "steps": [{"id": "s1", "operation": "click", "ref": "help"}]}
    chosen, wire, model = request(initial, raw, limit=1)
    original, original_wire, _ = request(initial, raw, contract="property_intent_v51", limit=1)
    assert chosen == original and wire == original_wire
    assert model.last_request_image_uploaded is True
    assert desktop_response_format(1, property_contract="property_intent_v52") == (
        desktop_response_format(1, property_contract="property_intent_v51")
    )


def test_new_schema_is_fresh_and_does_not_mutate_old51_or_default50_grammar():
    old51 = desktop_response_format(6, property_contract="property_intent_v51", image_uploaded=False)
    old50 = desktop_response_format(6)
    schema(image=False)["anyOf"][0]["required"].append("untrusted mutation")
    assert old51 == desktop_response_format(6, property_contract="property_intent_v51", image_uploaded=False)
    assert old50 == desktop_response_format(6)
    assert "untrusted mutation" not in schema(image=False)["anyOf"][0]["required"]


@pytest.mark.parametrize("limit", [6, 12])
@pytest.mark.parametrize("box", ["normalized_1000", "image_pixels"])
@pytest.mark.parametrize("image", [False, True])
def test_short_protocol_has_both_generic_examples_and_explicit_non_authority(limit, box, image):
    text = completion_policy_property_protocol_instructions(limit, visual_box_space=box, image_uploaded=image)
    assert len(text) <= 7000
    assert '"confirm":"none"' in text and '"numeric_finalize":{"key":"enter"}' in text
    assert "stage|goal" in text and "none compiles no confirmation" in text


def test_old_contract_http_and_plan_wire_match_frozen51_in_clean_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = root / "runs/v51-frozen/src"
    if not frozen.is_dir():
        pytest.skip("local frozen51 source is required for independent historical compatibility")
    code = r'''
import hashlib, io, json, sys
from dataclasses import asdict
from pathlib import Path
import httpx
from PIL import Image
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.osworld_backend import screenshot_image
module_path=Path(sys.modules['cua_jev.agent_planner'].__file__).resolve()
assert module_path.is_relative_to(Path(sys.argv[1]).resolve())
data=io.BytesIO(); Image.new('RGB',(800,600),'white').save(data,'PNG')
snapshot=AgentSnapshot('osworld://desktop','Main','',(
    AgentControl('help','Help','button',role='button',app='Editor',window='Main',active=True,
                 box=(100,100,200,200)),),image=screenshot_image(data.getvalue()))
facts=[]
for contract,fmt,observation in [
 ('property_intent_v33','unconstrained_v1','full_multimodal_v1'),
 ('property_intent_v49','json_schema_v50','full_multimodal_v1'),
 ('property_intent_v51','json_schema_v50','native_first_v51')]:
 for limit in (1,6):
  raw=({'summary':'Current Help','done':False,'steps':[{'id':'s1','operation':'click','ref':'help'}]}
       if limit==1 else {'intent':'pursue_property','label':'Requested setting','scope':{'app':'Editor'},
                        'edit':{'operation':'set','value':'42'}})
  if limit>1 and contract=='property_intent_v51': raw['numeric_finalize']={'key':'none'}
  captured=[]
  def answer(request):
   captured.append(request.content)
   return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(raw)}}]})
  with httpx.Client(transport=httpx.MockTransport(answer)) as client:
   planner=MultimodalPlanner('https://offline.invalid','fixture',api_key='offline',client=client,
      profile='desktop',desktop_contract=contract,desktop_output_format=fmt,
      desktop_planning_observation=observation)
   plan=planner.plan('Apply requested setting',snapshot,[],max_actions=limit)
  assert len(captured)==planner.calls==1
  facts.append({'contract':contract,'limit':limit,'HTTP_sha256':hashlib.sha256(captured[0]).hexdigest(),
                'plan':asdict(plan)})
print(json.dumps(facts,sort_keys=True))
'''
    observations = []
    for source in (frozen, root / "src"):
        completed = subprocess.run(
            [sys.executable, "-B", "-c", code, str(source)], cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=True, timeout=30,
        )
        observations.append(json.loads(completed.stdout))
    assert observations[0] == observations[1]
