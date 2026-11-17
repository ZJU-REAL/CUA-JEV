"""Guidance composition and the real planner transport preserve original55 authority."""

import base64
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from test_desktop_output_schema_contract import observed as observed
from test_semantic_tools_planner import flat, model, packet
from test_semantic_tools_v55 import capture, repair_wire

from cua_jev import desktop_semantic_tools_v56 as new
from cua_jev.agent_planner import _desktop_instructions
from cua_jev.desktop_output_schema import desktop_response_format
from cua_jev.desktop_property_protocol import completion_policy_property_protocol_instructions
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_semantic_tools_v55 import decode_semantic_declaration as decode55
from cua_jev.desktop_semantic_tools_v55 import encode_semantic_declaration as encode55

TOKEN = "ab" * 16
OPTIONS = {"profile": "desktop", "contract": "property_intent_v52",
           "observation": "full_multimodal_v1", "visual_box_space": "normalized_1000"}


def _body(snapshot, limit, *, recovery=False, repair=None):
    system = (_desktop_instructions(1, snapshot) if limit == 1 else
              completion_policy_property_protocol_instructions(limit))
    if recovery:
        system = "RECOVERY PRIORITY: retained fixture context. " + system
    if repair:
        system += " PREPARATION REPAIR: retained fixture offer token=" + repair
    return {
        "model": "offline-strong", "stream": False, "enable_thinking": True,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": [
            {"type": "text", "text": "Unchanged observed state/history"},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(snapshot.image.jpeg).decode("ascii"),
            }},
        ]}],
        "response_format": desktop_response_format(
            limit, property_contract="property_intent_v52", image_uploaded=True, prepare_repair_token=repair,
        ),
    }


@pytest.mark.parametrize("limit,recovery,repair", [(1, False, None), (1, True, None),
                                                  (1, True, TOKEN), (6, False, None)])
def test_only_allowed_guidance_slices_change_and_actual_hashes_follow(observed, limit, recovery, repair):
    body = _body(observed, limit, recovery=recovery, repair=repair)
    before = copy.deepcopy(body)
    old, old_meta = encode55(body, max_actions=limit, **OPTIONS)
    encoded, metadata = new.encode_semantic_declaration(body, max_actions=limit, **OPTIONS)
    assert body == before
    expected = copy.deepcopy(old)
    expected["messages"][0]["content"] += new.ATOMIC_VISUAL_FILL_GUIDANCE
    if limit > 1:
        expected["messages"][0]["content"] += new.FUTURE_ROLE_GUIDANCE
    visual = next(t["function"] for t in expected["tools"] if t["function"]["name"] == new.VISUAL_TOOL)
    visual["description"] += new.VISUAL_DESCRIPTION_SUFFIX
    assert encoded == expected
    assert len(encoded["tools"]) == (5 if limit == 1 else 8)
    assert encoded["tool_choice"] == "auto" and encoded["parallel_tool_calls"] is False
    assert encoded["messages"][1:] == old["messages"][1:]
    assert metadata["response_transport"] == metadata["requested_format"] == new.FORMAT
    assert metadata["ancestor_response_transport"] == "semantic_tools_v55"
    assert metadata["request_body_sha256"] == _digest(encoded)
    assert metadata["ancestor_request_sha256"] == _digest(old)
    assert metadata["declaration_tools_sha256"] == _digest(encoded["tools"])
    assert metadata["ancestor_declaration_schema_sha256"] == _digest(old["tools"])
    assert metadata["projected_system_sha256"] == _digest(encoded["messages"][0]["content"])
    assert metadata["parameter_schema_sha256"] == old_meta["parameter_schema_sha256"]
    assert metadata["ancestor_request_provenance"] == old_meta
    if repair:
        for tool in encoded["tools"]:
            original = next(t for t in old["tools"] if t["function"]["name"] == tool["function"]["name"])
            assert tool["function"]["parameters"] == original["function"]["parameters"]


@pytest.mark.parametrize("limit", [1, 6])
def test_original_encoder_called_once_without_other_body_changes(observed, monkeypatch, limit):
    calls = []

    def ancestor(body, **kwargs):
        calls.append(kwargs)
        return encode55(body, **kwargs)

    monkeypatch.setattr(new, "encode_ancestor", ancestor)
    new.encode_semantic_declaration(_body(observed, limit), max_actions=limit, **OPTIONS)
    assert calls == [{"max_actions": limit, **OPTIONS}]


@pytest.mark.parametrize("limit", [1, 6])
def test_visual_fill_mapping_literal_value_and_coordinates_are_exact55(limit):
    header = {"summary": "Replace current text", "id": "s1", "done": False} if limit == 1 else {
        "label": "Replace current text",
    }
    raw = {**header, "operation": "visual_fill", "action_label": "Current visible text box",
           "value": "literal value", "box": [100, 100, 300, 160], "box_space": "normalized_1000"}
    payload = packet(raw, name=new.VISUAL_TOOL)
    before = copy.deepcopy(payload)
    expected, old_meta = decode55(payload, max_actions=limit)
    mapped, metadata = new.decode_semantic_declaration(payload, max_actions=limit)
    assert mapped == expected and payload == before
    assert metadata["ancestor_mapping_provenance"] == old_meta
    assert metadata["response_transport"] == new.FORMAT and metadata["input_authority"] is False


def test_repair_and_reusable_stage_mapping_remain_exact55():
    raw = {"summary": "Repair current preparation", "id": "s1", "done": False,
           "operation": "visual_fill", "action_label": "Current visible text box", "value": "literal query",
           "box": [100, 100, 300, 160], "box_space": "normalized_1000", "prepare_repair": TOKEN}
    payload = packet(raw, name=new.VISUAL_TOOL)
    expected, _ = decode55(payload, max_actions=1, prepare_repair_token=TOKEN)
    actual, _ = new.decode_semantic_declaration(payload, max_actions=1, prepare_repair_token=TOKEN)
    assert actual == expected
    stage = packet(flat(), name="declare_property_stage")
    assert new.decode_semantic_declaration(stage, max_actions=6)[0] == decode55(stage, max_actions=6)[0]


@pytest.mark.parametrize("failure", ["refusal", "multiple", "mixed_visual", "nonfinite"])
def test_original55_rejections_are_not_repaired(failure):
    raw = {"summary": "Current action", "id": "s1", "done": False,
           "operation": "visual_fill", "action_label": "Visible field", "value": "literal",
           "box": [100, 100, 300, 160], "box_space": "normalized_1000"}
    payload = packet(raw, name=new.VISUAL_TOOL)
    message = payload["choices"][0]["message"]
    if failure == "refusal":
        message["refusal"] = "fixture"
    elif failure == "multiple":
        message["tool_calls"] *= 2
    elif failure == "mixed_visual":
        payload = packet({**raw, "ref": "not-authorized"}, name=new.VISUAL_TOOL)
    else:
        message["tool_calls"][0]["function"]["arguments"] = '{"operation":NaN}'
    with pytest.raises(ValueError) as old:
        decode55(payload, max_actions=1)
    with pytest.raises(ValueError) as wrapped:
        new.decode_semantic_declaration(payload, max_actions=1)
    assert str(wrapped.value) == str(old.value)


@pytest.mark.parametrize("limit", [1, 6])
def test_real_planner_selects56_once_and_preserves55_plan_and_unchanged_parameters(observed, limit):
    old_plan, old_body, _ = capture(observed, limit=limit)
    plan, body, planner = capture(observed, limit=limit, output_format=new.FORMAT)
    assert plan == old_plan
    assert planner.calls == 1 and planner.last_request_image_uploaded is True
    assert planner.last_request_provenance["response_transport"] == new.FORMAT
    assert planner.last_response_transport["ancestor_response_transport"] == "semantic_tools_v55"
    assert planner.last_response_transport["stage_mapping_applied"] is True
    for old_tool, tool in zip(old_body["tools"], body["tools"], strict=True):
        assert tool["function"]["name"] == old_tool["function"]["name"]
        assert tool["function"]["parameters"] == old_tool["function"]["parameters"]
    assert body["messages"][0]["content"].startswith(old_body["messages"][0]["content"])
    if limit > 1:
        assert new.FUTURE_ROLE_GUIDANCE in body["messages"][0]["content"]


def test_real_planner56_repair_offer_remains_explicit_and_original(observed):
    raw = repair_wire()
    context = {"token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}, "value": "original"}}
    old, _, _ = capture(observed, raw, new.VISUAL_TOOL, limit=1, context=context)
    plan, body, planner = capture(observed, raw, new.VISUAL_TOOL, limit=1,
                                  context=context, output_format=new.FORMAT)
    assert plan == old and plan.prepare_repair_token == TOKEN
    assert planner.calls == 1
    schema = next(t["function"]["parameters"] for t in body["tools"]
                  if t["function"]["name"] == new.VISUAL_TOOL)
    assert schema["properties"]["prepare_repair"] == {"const": TOKEN}


def test_actual56_planner_preserves_full_legal_navigation_policy(observed):
    raw = {"label": "Reach current view", "scope": {"app": "Editor"}, "completion": "stage",
           "arrival": [{"kind": "exists", "target": {"name": "Future control"}}]}
    plan, _, _ = capture(observed, raw, "declare_navigation_stage", output_format=new.FORMAT)
    assert plan.steps[0].full_legal_navigation_gestures is True


def test_actual56_http_error_is_one_call_without_fallback_or_private_response(observed):
    calls = []

    def reject(request):
        calls.append(request)
        return httpx.Response(400, text="private provider diagnostic")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        planner = model(client, desktop_output_format=new.FORMAT)
        with pytest.raises(RuntimeError):
            planner.plan("Use current state", observed, [], max_actions=6)
    assert len(calls) == planner.calls == 1
    assert planner.last_request_provenance["response_transport"] == new.FORMAT
    assert planner.last_response_transport is None
    assert "private" not in (planner.last_validation_error or "")


def test_existing55_request_bytes_and_plans_equal_frozen55_in_clean_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = root / "runs/v55-frozen/src"
    if not frozen.is_dir():
        pytest.skip("independent frozen55 source required for old55 equality")
    code = r'''
import hashlib,io,json,sys
from dataclasses import asdict
from pathlib import Path
import httpx
from PIL import Image
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentControl,AgentSnapshot
from cua_jev.osworld_backend import screenshot_image
origin=Path(sys.modules['cua_jev.agent_planner'].__file__).resolve()
assert origin.is_relative_to(Path(sys.argv[1]).resolve())
data=io.BytesIO();Image.new('RGB',(800,600),'white').save(data,'PNG')
snapshot=AgentSnapshot('osworld://desktop','Main','',(
 AgentControl('help','Help','button',role='button',app='Editor',window='Main',active=True,
              box=(100,100,200,200)),),image=screenshot_image(data.getvalue()))
facts=[];token='ab'*16
for requested,recovering,repair in ((1,False,False),(6,False,False),(6,True,False),(1,True,True)):
 limit=1 if recovering else requested; requests=[]
 raw=({'summary':'Help','id':'s1','done':False,'operation':'click','ref':'help'} if limit==1 else
      {'label':'Reach view','scope':{'app':'Editor'},'arrival':[{'kind':'exists','target':{'name':'Future'}}],
       'completion':'stage'})
 name='declare_native_current_action' if limit==1 else 'declare_navigation_stage'
 if repair:
  name='declare_visual_current_action'
  raw={'summary':'Repair','id':'repair','done':False,'operation':'visual_fill','value':'query',
       'action_label':'Current field','box_space':'normalized_1000','box':[100,100,300,180],
       'prepare_repair':token}
 def answer(request):
  requests.append(request.content)
  return httpx.Response(200,json={'choices':[{'message':{'tool_calls':[{'type':'function',
    'function':{'name':name,'arguments':json.dumps(raw)}}]}}]})
 with httpx.Client(transport=httpx.MockTransport(answer)) as client:
  planner=MultimodalPlanner('https://offline.invalid','fixture',api_key='offline',client=client,
    profile='desktop',desktop_contract='property_intent_v52',desktop_output_format='semantic_tools_v55',
    desktop_planning_observation='full_multimodal_v1',enable_thinking=True,request_timeout_s=90)
  if repair: planner.set_prepare_repair_context({'token':token,'remaining_stage':{'scope':{'app':'Editor'}}})
  history=[{'recovery_required':True,'reason':'effect_failed'}] if recovering else []
  plan=planner.plan('Use current state',snapshot,history,max_actions=requested)
 assert len(requests)==planner.calls==1
 exported=asdict(plan);exported['steps']=[step.to_dict() for step in plan.steps]
 facts.append({'request_sha256':hashlib.sha256(requests[0]).hexdigest(),'plan':exported})
print(json.dumps(facts,sort_keys=True))
'''
    outputs = []
    for source in (frozen, root / "src"):
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, str(source)], cwd=tmp_path,
            env={**os.environ, "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, text=True, check=True, timeout=30,
        )
        outputs.append(json.loads(result.stdout))
    assert outputs[0] == outputs[1]
