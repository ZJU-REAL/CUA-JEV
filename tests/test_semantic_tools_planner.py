"""Offline MockTransport obligations for the opt-in semantic declaration wire."""

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.desktop_semantic_tools import FORMAT, _digest, _property_from_flat, semantic_stage_instructions
from cua_jev.desktop_tool_declaration import FORMAT as ORIGINAL_FORMAT

TOKEN = "ab" * 16


@pytest.fixture
def snapshot(initial):
    return replace(initial, controls=tuple(
        replace(control, box=(100, 100, 400, 200)) if control.ref == "search" else control
        for control in initial.controls
    ))


def flat():
    return {"label": "Requested text change", "scope": {"app": "Editor"}, "completion": "goal",
            "edit_operation": "fill", "edit_value": "Requested literal", "numeric_finalize_key": "none",
            "confirmation_label": "Submit requested change"}


def immediate():
    return {"summary": "Current field", "done": False,
            "steps": [{"id": "s1", "operation": "fill", "ref": "search", "value": "requested text"}]}


def packet(raw, name="declare_property_stage", content=None):
    return {"usage": {"prompt_tokens": 20, "completion_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 4, "text_tokens": 3}},
            "choices": [{"message": {"content": content, "tool_calls": [{"type": "function",
                "function": {"name": name, "arguments": json.dumps(raw)}}]}}]}


def model(client, **options):
    return MultimodalPlanner(
        "https://offline.invalid", "strong-fixture", api_key="offline", client=client,
        profile=options.pop("profile", "desktop"),
        desktop_contract=options.pop("desktop_contract", "property_intent_v52"),
        desktop_output_format=options.pop("desktop_output_format", FORMAT),
        desktop_planning_observation=options.pop("desktop_planning_observation", "full_multimodal_v1"),
        enable_thinking=True, request_timeout_s=90, **options,
    )


def capture(snapshot, raw, *, limit=6, output_format=FORMAT, context=None):
    requests = []

    def answer(request):
        requests.append(request)
        if output_format == "json_schema_v50":
            response = {"choices": [{"message": {"content": json.dumps(raw)}}]}
        else:
            name = ("declare_property_stage" if output_format == FORMAT and limit > 1
                    else "declare_desktop_stage")
            response = packet(raw, name=name, content="Ignored private narration")
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=output_format)
        if context is not None:
            planner.set_prepare_repair_context(context)
        plan = planner.plan("Apply the requested setting", snapshot, [], max_actions=limit)
    assert len(requests) == planner.calls == 1
    return plan, requests[0], planner


def test_H6_named_flat_schema_full_image_original_parser_and_safe_provenance(snapshot):
    raw = flat()
    plan, request, planner = capture(snapshot, raw)
    body = json.loads(request.content)
    tools = {tool["function"]["name"]: tool["function"] for tool in body["tools"]}
    assert len(tools) == 5
    assert Draft202012Validator(tools["declare_property_stage"]["parameters"]).is_valid(raw)
    assert "response_format" not in body and all("strict" not in tool for tool in tools.values())
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["enable_thinking"] is True and "thinking_budget" not in body
    assert [part["type"] for part in body["messages"][1]["content"]] == ["text", "image_url"]
    assert request.extensions["timeout"]["read"] == 90
    assert plan.reusable and plan.complete_property_goal
    assert plan.steps[0].property_edit.numeric_finalize_key == ""
    assert plan.steps[1].property_confirmation_of == "property"
    provenance = planner.last_request_provenance
    assert provenance["requested_format"] == provenance["response_transport"] == FORMAT
    assert provenance["declaration_schema_sha256"] == _digest(body["tools"])
    assert provenance["declaration_argument_sha256"] == hashlib.sha256(json.dumps(raw).encode()).hexdigest()
    assert provenance["mapped_object_sha256"] == _digest(
        {"intent": "pursue_property", **_property_from_flat(raw)},
    )
    assert provenance["stage_mapping_applied"] is True and provenance["image_uploaded"] is True
    assert type(provenance["http_wall_ms"]) is float and provenance["http_wall_ms"] >= 0
    assert planner.last_response_transport["http_wall_ms"] == planner.last_http_wall_ms
    assert planner.last_response_transport["accompanying_content_present"] is True
    safe_metadata = json.dumps(provenance) + json.dumps(planner.last_response_transport)
    assert "Ignored private narration" not in safe_metadata
    assert planner.last_usage["completion_details_reasoning_tokens"] == 4
    assert planner.last_usage["completion_details_text_tokens"] == 3
    assert planner.last_usage == planner.usage_totals


@pytest.mark.parametrize("repair", [False, True])
def test_H1_request_bytes_prompt_schema_decoder_and_plan_equal_original_52(snapshot, repair):
    raw, context = immediate(), None
    if repair:
        raw = {"summary": "Repair current preparation", "done": False, "prepare_repair": TOKEN,
               "steps": [{"id": "s1", "operation": "visual_fill", "label": "Visible current search field",
                          "box": [100, 100, 400, 200], "value": "requested query"}]}
        context = {"token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}}}
    old_plan, old_request, _ = capture(snapshot, raw, limit=1, output_format=ORIGINAL_FORMAT, context=context)
    plan, request, planner = capture(snapshot, raw, limit=1, context=context)
    assert request.content == old_request.content and plan == old_plan
    assert planner.last_request_provenance["requested_format"] == FORMAT
    assert planner.last_request_provenance["response_transport"] == ORIGINAL_FORMAT
    assert planner.last_response_transport["response_transport"] == ORIGINAL_FORMAT
    assert planner.last_request_provenance["stage_mapping_applied"] is False
    assert json.loads(request.content)["tools"][0]["function"]["name"] == "declare_desktop_stage"


def test_H6_only_projected_system_and_tool_transport_change_from_original_request(snapshot):
    raw = flat()
    original = {"intent": "pursue_property", **_property_from_flat(raw)}
    old_plan, old_request, _ = capture(snapshot, original, output_format="json_schema_v50")
    plan, request, _ = capture(snapshot, raw)
    original_body, body = json.loads(old_request.content), json.loads(request.content)
    from cua_jev.desktop_property_protocol import completion_policy_property_protocol_instructions

    base = completion_policy_property_protocol_instructions(6, image_uploaded=True)
    projected = semantic_stage_instructions(6)
    assert original_body["messages"][0]["content"].startswith(base)
    assert body["messages"][0]["content"] == projected + original_body["messages"][0]["content"][len(base):]
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        body.pop(key)
    body["messages"][0]["content"] = original_body["messages"][0]["content"]
    body["response_format"] = original_body["response_format"]
    assert body == original_body and plan == old_plan


@pytest.mark.parametrize("kind", [
    "absent", "multiple", "wrong_name", "invalid_JSON", "array", "old_fields", "text_enter",
])
def test_bad_wire_or_original_numeric_guard_has_one_HTTP_no_content_fallback(snapshot, kind):
    secret = "private-provider-content-never-log"
    payload = packet(flat(), content=secret)
    message = payload["choices"][0]["message"]
    function = message["tool_calls"][0]["function"]
    if kind == "absent":
        message.pop("tool_calls")
        message["content"] = json.dumps({"intent": "pursue_property", **_property_from_flat(flat())})
    elif kind == "multiple":
        message["tool_calls"] *= 2
    elif kind == "wrong_name":
        function["name"] = secret
    elif kind == "invalid_JSON":
        function["arguments"] = "{" + secret
    elif kind == "array":
        function["arguments"] = "[]"
    elif kind == "old_fields":
        function["arguments"] = json.dumps(flat() | {"confirm": secret})
    else:
        function["arguments"] = json.dumps(flat() | {"numeric_finalize_key": "enter"})
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client)
        with pytest.raises(ValueError) as error:
            planner.plan("Apply setting", snapshot, [], max_actions=6)
    assert len(requests) == planner.calls == 1
    assert secret not in str(error.value) + planner.last_validation_error
    assert planner.last_usage["completion_tokens"] == 7
    assert planner.last_request_provenance["http_wall_ms"] >= 0
    assert (planner.last_response_transport is not None) == (kind == "text_enter")
    if kind == "text_enter":
        assert "bounded numeric edit" in planner.last_validation_error


@pytest.mark.parametrize("failure", ["HTTP", "timeout"])
def test_HTTP_wall_is_recorded_on_single_transport_failure_without_raw_error(snapshot, failure):
    requests, secret = [], "private-provider-body-or-endpoint"

    def reject(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        return httpx.Response(400, text=secret)

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        planner = model(client)
        with pytest.raises(RuntimeError, match="multimodal planning request failed") as error:
            planner.plan("Apply setting", snapshot, [], max_actions=6)
    assert len(requests) == planner.calls == 1 and planner.last_usage == {}
    assert planner.last_response_transport is None and planner.last_request_provenance["http_wall_ms"] >= 0
    assert planner.last_http_wall_ms == planner.last_request_provenance["http_wall_ms"]
    assert secret not in str(error.value) + planner.last_validation_error


def test_expired_deadline_does_not_send_or_retain_previous_HTTP_metadata(snapshot):
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(flat()))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client)
        planner.plan("Apply setting", snapshot, [], max_actions=6)
        planner.set_episode_deadline(time.perf_counter() - 1)
        with pytest.raises(TimeoutError, match="expired before model request"):
            planner.plan("Apply setting", snapshot, [], max_actions=6)
        assert planner.last_request_provenance is None and planner.last_http_wall_ms is None
    assert len(requests) == planner.calls == 1


@pytest.mark.parametrize("options", [
    {"profile": "browser", "desktop_contract": "legacy_v23"},
    {"desktop_contract": "property_intent_v51"},
    {"desktop_planning_observation": "native_first_v51"},
])
def test_opt_in_rejects_unsupported_configuration_before_any_HTTP(options):
    requests = []
    with httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request))) as client:
        with pytest.raises(ValueError):
            model(client, **options)
    assert requests == []


def test_old_contracts_default_JSON_and_H1_tool_wire_equal_frozen52_in_clean_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = root / "runs/v52-frozen/src"
    if not frozen.is_dir():
        pytest.skip("independent frozen52 source is required for historical wire comparison")
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
cases=[('property_intent_v33','unconstrained_v1','full_multimodal_v1',h) for h in (1,6)]
cases += [('property_intent_v49','json_schema_v50','full_multimodal_v1',h) for h in (1,6)]
cases += [('property_intent_v51','json_schema_v50','native_first_v51',h) for h in (1,6)]
cases += [('property_intent_v52','tool_declaration_v52','full_multimodal_v1',1)]
for contract,fmt,observation,limit in cases:
 raw=(
  {'summary':'Current Help','done':False,'steps':[{'id':'s1','operation':'click','ref':'help'}]} if limit==1
  else {'intent':'act_current','label':'Current Help','action':{'operation':'click','ref':'help'}})
 captured=[]
 def answer(request):
  captured.append(request.content)
  message=({'tool_calls':[{'type':'function','function':{'name':'declare_desktop_stage',
             'arguments':json.dumps(raw)}}]} if fmt=='tool_declaration_v52' else {'content':json.dumps(raw)})
  return httpx.Response(200,json={'choices':[{'message':message}]})
 with httpx.Client(transport=httpx.MockTransport(answer)) as client:
  planner=MultimodalPlanner('https://offline.invalid','fixture',api_key='offline',client=client,
   profile='desktop',desktop_contract=contract,desktop_output_format=fmt,
   desktop_planning_observation=observation)
  plan=planner.plan('Use current Help',snapshot,[],max_actions=limit)
 assert len(captured)==planner.calls==1
 facts.append({'contract':contract,'limit':limit,'format':fmt,
   'HTTP_sha256':hashlib.sha256(captured[0]).hexdigest(),'plan':asdict(plan)})
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
