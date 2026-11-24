"""Opt-in54 regression for actual generation wire, original parser and accounting."""

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
from test_desktop_tool_declaration_planner import Backend, Environment
from test_osworld_comparison import comparison, row, stats, write_summary
from test_semantic_tools_planner import initial as initial
from test_semantic_tools_planner import model, packet
from test_semantic_tools_planner import snapshot as snapshot

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentControl
from cua_jev.desktop_semantic_tools import _digest
from cua_jev.desktop_semantic_tools_v54 import FORMAT
from cua_jev.trace import JsonlTrace

TOKEN = "ab" * 16
SHAPE_CASES = json.loads(
    (Path(__file__).parent / "fixtures/semantic_v54_shape_matrix.json").read_text(encoding="utf-8")
)["cases"]


def current(limit):
    if limit == 1:
        return {"summary": "Use current Help", "done": False,
                "steps": [{"id": "s1", "operation": "click", "ref": "help"}]}, "declare_desktop_stage"
    return (
        {"label": "Use current Help", "action": {"operation": "click", "ref": "help"}},
        "declare_current_action",
    )


def capture(snapshot, raw=None, name=None, *, limit=6, output_format=FORMAT, context=None):
    requests = []
    if raw is None:
        raw, name = current(limit)

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=output_format)
        if context is not None:
            planner.set_prepare_repair_context(context)
        plan = planner.plan("Use the requested desktop state", snapshot, [], max_actions=limit)
    assert len(requests) == planner.calls == 1
    return plan, json.loads(requests[0].content), planner


@pytest.mark.parametrize("limit,ancestor", [(1, "tool_declaration_v52"), (6, "semantic_tools_v53")])
def test_actual_tools_and_parameter_hashes_follow_projection_and_descriptions(snapshot, limit, ancestor):
    _plan, body, planner = capture(snapshot, limit=limit)
    metadata = planner.last_request_provenance
    assert metadata["response_transport"] == metadata["requested_format"] == FORMAT
    assert metadata["ancestor_response_transport"] == ancestor
    assert (
        metadata["declaration_tools_sha256"]
        == metadata["declaration_schema_sha256"]
        == _digest(body["tools"])
    )
    assert metadata["parameter_schema_sha256"] == {
        tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in body["tools"]
    }
    assert metadata["declaration_tools_sha256"] != metadata["ancestor_declaration_schema_sha256"]
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["enable_thinking"] is True and "thinking_budget" not in body


@pytest.mark.parametrize("kind", ["mixed_ref", "missing_effect", "visual_completion"])
def test_actual_generation_schema_excludes_the_three_observed_invalid_shapes(snapshot, kind):
    limit = 1 if kind == "visual_completion" else 6
    _plan, body, _planner = capture(snapshot, limit=limit)
    name = "declare_desktop_stage" if limit == 1 else "declare_current_action"
    parameters = next(t["function"]["parameters"] for t in body["tools"] if t["function"]["name"] == name)
    target = {"name": "Help", "app": "Editor"}
    if kind == "mixed_ref":
        bad = {"label": "Current action", "action": {"operation": "click", "ref": "help",
               "postconditions": [{"kind": "exists", "target": target}]}}
    elif kind == "missing_effect":
        bad = {"label": "Current action", "action": {"operation": "check", "target": target}}
    else:
        bad = {"summary": "Current visible action", "done": False,
               "steps": [{"id": "s1", "operation": "visual_click", "label": "Visible control",
                          "box": [100, 100, 200, 200]}],
               "completion_conditions": [{"kind": "exists", "target": target}]}
    assert not Draft202012Validator(parameters).is_valid(bad)


def test_semantic_preparation_with_an_explicit_real_effect_remains_legal(snapshot):
    target = {"name": "Prerequisite", "role": "checkbox", "app": "Editor", "window": "Main"}
    snapshot = replace(snapshot, controls=(*snapshot.controls, AgentControl(
        "prerequisite", "Prerequisite", "input", role="checkbox", input_type="checkbox",
        app="Editor", window="Main", active=True, box=(500, 100, 650, 200),
    )))
    raw = {"label": "Requested checked state", "scope": {"app": "Editor"},
           "edit_operation": "check", "numeric_finalize_key": "none", "confirmation_label": "none",
           "completion": "goal", "prepare": {"operation": "check", "target": target,
           "postconditions": [{"kind": "checked", "target": target}]}}
    plan, body, _planner = capture(snapshot, raw, "declare_property_stage")
    parameters = body["tools"][0]["function"]["parameters"]
    assert Draft202012Validator(parameters).is_valid(raw)
    assert plan.steps[0].operation == "check" and plan.steps[0].postconditions[0].kind == "checked"
    assert plan.steps[1].property_edit.operation == "check"


@pytest.mark.parametrize("repair", [False, True])
def test_H1_original_body_and_strong_prompt_except_explicitly_projected_parameters(snapshot, repair):
    raw, name = current(1)
    context = None
    if repair:
        raw = {"summary": "Repair current preparation", "done": False, "prepare_repair": TOKEN,
               "steps": [{"id": "s1", "operation": "visual_fill", "label": "Current search field",
                          "box": [100, 100, 400, 200], "value": "requested query"}]}
        context = {"token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}}}
    plan, body, planner = capture(snapshot, raw, name, limit=1, context=context)
    original, original_body, _original_planner = capture(
        snapshot, raw, name, limit=1, context=context, output_format="semantic_tools_v53",
    )
    assert plan == original
    parameters = body["tools"][0]["function"]["parameters"]
    assert Draft202012Validator(parameters).is_valid(raw)
    if repair:
        assert parameters["properties"]["prepare_repair"] == {"const": TOKEN}
    restored = copy.deepcopy(body)
    restored["tools"][0]["function"]["parameters"] = original_body["tools"][0]["function"]["parameters"]
    assert restored == original_body  # New54 schema wire itself intentionally differs.
    assert planner.last_response_transport["ancestor_response_transport"] == "tool_declaration_v52"


@pytest.mark.parametrize("failure", ["timeout", "HTTP", "nonfinite", "multiple"])
def test_original_decoder_or_HTTP_errors_do_not_retry_or_fallback(snapshot, failure):
    requests = []
    raw, name = current(6)

    def answer(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private endpoint detail", request=request)
        if failure == "HTTP":
            return httpx.Response(400, text="private provider body")
        payload = packet(raw, name=name)
        if failure == "nonfinite":
            payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"extra":NaN}'
        else:
            payload["choices"][0]["message"]["tool_calls"] *= 2
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=FORMAT)
        with pytest.raises((RuntimeError, ValueError)):
            planner.plan("Use current Help", snapshot, [], max_actions=6)
    assert len(requests) == planner.calls == 1
    assert planner.last_request_provenance["response_transport"] == FORMAT
    assert planner.last_request_provenance["http_wall_ms"] >= 0
    assert planner.last_response_transport is None
    assert "private" not in planner.last_validation_error


@pytest.mark.parametrize("limit", [1, 6])
def test_runner_records_actual54_and_ancestor_transport_even_for_H1(snapshot, limit):
    env, trace, requests = Environment(snapshot), JsonlTrace(), []
    raw, name = current(limit)

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(raw, name=name))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=FORMAT)
        result = AgentRunner(Backend(env, pause=0), planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, max_steps=1, max_plans=1, max_failures=2,
                               plan_horizon=limit, timeout_s=30)).run("Use current Help")
    assert len(requests) == planner.calls == result.planner_calls == len(env.actions) == 1
    event = next(e["payload"] for e in trace.events if e["kind"] == "agent_model_output_transport")
    assert event["response_transport"] == event["requested_format"] == FORMAT
    assert event["ancestor_response_transport"] == (
        "tool_declaration_v52" if limit == 1 else "semantic_tools_v53"
    )
    assert event["declaration_tools_sha256"] == _digest(json.loads(requests[0].content)["tools"])
    assert event["generation_projection_format"] == FORMAT


def test_old53_valid_requests_and_plans_equal_frozen53_in_clean_processes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    frozen = next((p for p in (root.parent / "v53-frozen/src", root / "runs/v53-frozen/src")
                   if p.is_dir()), None)
    if frozen is None:
        pytest.skip("independent frozen53 source required for old-wire comparison")
    code = r'''
import hashlib,io,json,sys
from dataclasses import asdict
from pathlib import Path
import httpx
from PIL import Image
from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentControl,AgentSnapshot
from cua_jev.osworld_backend import screenshot_image
assert (
 Path(sys.modules['cua_jev.agent_planner'].__file__).resolve().is_relative_to(
  Path(sys.argv[1]).resolve()
 )
)
data=io.BytesIO();Image.new('RGB',(800,600),'white').save(data,'PNG')
snapshot=AgentSnapshot('osworld://desktop','Main','',(
 AgentControl('help','Help','button',role='button',app='Editor',window='Main',active=True,
              box=(100,100,200,200)),),image=screenshot_image(data.getvalue()))
facts=[]
for limit in (1,6):
 raw=({'summary':'Help','done':False,'steps':[{'id':'s1','operation':'click','ref':'help'}]} if limit==1 else
      {'label':'Help','action':{'operation':'click','ref':'help'}})
 name='declare_desktop_stage' if limit==1 else 'declare_current_action'
 requests=[]
 def answer(request):
  requests.append(request.content)
  return httpx.Response(200,json={'choices':[{'message':{'tool_calls':[{'type':'function',
    'function':{'name':name,'arguments':json.dumps(raw)}}]}}]})
 with httpx.Client(transport=httpx.MockTransport(answer)) as client:
  planner=MultimodalPlanner('https://offline.invalid','fixture',api_key='offline',client=client,
    profile='desktop',desktop_contract='property_intent_v52',desktop_output_format='semantic_tools_v53',
    desktop_planning_observation='full_multimodal_v1',enable_thinking=True,request_timeout_s=90)
  plan=planner.plan('Use current Help',snapshot,[],max_actions=limit)
 assert len(requests)==planner.calls==1
 facts.append({'limit':limit,'request_sha256':hashlib.sha256(requests[0]).hexdigest(),'plan':asdict(plan)})
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


@pytest.mark.parametrize("baseline_format", [FORMAT, "semantic_tools_v53"])
def test_comparison_accepts_new54_but_excludes_old53_format_mismatch(tmp_path, baseline_format):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", desktop_output_format=FORMAT),
        row("vlm", desktop_output_format=baseline_format),
    ]))])
    configuration = report["pairs"][0]["configuration"]
    assert not configuration["invalid"]
    if baseline_format == FORMAT:
        assert configuration["matched"]["desktop_output_format"] == FORMAT
        assert report["pairs"][0]["eligible"]
        assert stats(report)["agent_ms"]["n"] == 1
    else:
        assert configuration["mismatches"]["desktop_output_format"] == {
            "hybrid": FORMAT, "vlm": baseline_format,
        }
        assert not report["pairs"][0]["eligible"]
        assert stats(report)["agent_ms"]["n"] == 0


@pytest.mark.parametrize("unknown", [False, True])
def test_same_snapshot_precondition_feedback_exposes_no_target_or_expected_value(snapshot, unknown):
    secret = "private predicate literal"
    target = {"name": secret if unknown else "Search", "app": "Editor"}
    predicate = ({"kind": "focused", "target": target} if unknown else
                 {"kind": "value_equals", "target": target, "value": secret})
    raw = {"summary": "Guarded current input", "done": False, "steps": [{
        "id": "s1", "operation": "key_press", "value": "enter",
        "preconditions": [{"kind": "exists", "target": {"name": "Help", "app": "Editor"}}, predicate],
        "postconditions": [{"kind": "exists", "target": {"name": "Help", "app": "Editor"}}],
    }]}
    plan = parse_plan(raw, snapshot, max_actions=1, profile="desktop")
    with pytest.raises(ValueError, match="immediate action preconditions do not hold now") as error:
        _validate_desktop_execution_plan(plan, snapshot, 1)
    message = str(error.value)
    assert "first_nontrue_precondition_index0=1" in message and secret not in message
    assert ("kind=focused, outcome=unknown, reason=target_missing" if unknown else
            "kind=value_equals, outcome=false, reason=state_mismatch") in message


@pytest.mark.parametrize("case", SHAPE_CASES, ids=lambda case: case["id"])
def test_design_matrix_against_actual53_and54_request_parameters(snapshot, case):
    location = case["at"]
    limit = 1 if location.startswith("H1_root") else 6
    context = None
    raw = name = None
    if location == "H1_root_with_matching_repair_offer":
        raw = case["instance"]
        name = "declare_desktop_stage"
        token = case["offered_prepare_repair_token"]
        assert raw["prepare_repair"] == token
        context = {"token": token, "remaining_stage": {"scope": {"app": "Editor"}}}
    schemas = []
    for output_format in ("semantic_tools_v53", FORMAT):
        _plan, body, _planner = capture(
            snapshot, raw, name, limit=limit, output_format=output_format, context=context,
        )
        if location == "action":
            tool_name, definition = "declare_current_action", "action"
        elif location == "unchanged_$defs.commit":
            tool_name, definition = "declare_field_edits_stage", "commit"
        else:
            assert location in {"H1_root", "H1_root_with_matching_repair_offer"}
            tool_name, definition = "declare_desktop_stage", None
        parameters = next(tool["function"]["parameters"] for tool in body["tools"]
                          if tool["function"]["name"] == tool_name)
        schema = parameters if definition is None else {
            "$schema": parameters["$schema"], "$ref": "#/$defs/" + definition,
            "$defs": parameters["$defs"],
        }
        schemas.append(schema)
    assert Draft202012Validator(schemas[0]).is_valid(case["instance"])
    assert Draft202012Validator(schemas[1]).is_valid(case["instance"]) is case["projected_generation_allows"]
