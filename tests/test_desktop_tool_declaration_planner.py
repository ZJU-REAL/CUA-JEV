"""One HTTP declaration preserves desktop schemas, parser authority and accounting."""

import json
import time
from dataclasses import replace

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.desktop_output_schema import desktop_response_format
from cua_jev.desktop_tool_declaration import FORMAT, FUNCTION, TOOL_GUIDANCE, function_parameters
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.trace import JsonlTrace

TOKEN = "ab" * 16


@pytest.fixture
def snapshot(initial):
    return replace(initial, controls=tuple(
        replace(control, box=(100, 100, 400, 200)) if control.ref == "search" else control
        for control in initial.controls
    ))


def property_plan():
    return {"intent": "pursue_property", "label": "Requested numeric setting", "scope": {"app": "Editor"},
            "edit": {"operation": "set", "value": "42"}, "numeric_finalize": {"key": "enter"},
            "confirm": {"label": "Submit requested form"}, "completion": "goal"}


def immediate():
    return {"summary": "Current field", "done": False,
            "steps": [{"id": "s1", "operation": "fill", "ref": "search", "value": "requested text"}]}


def packet(raw, *, content=None):
    return {"usage": {"prompt_tokens": 20, "completion_tokens": 7,
                      "completion_tokens_details": {"reasoning_tokens": 4, "text_tokens": 3}},
            "choices": [{"message": {"content": content, "tool_calls": [{"type": "function",
                "function": {"name": FUNCTION, "arguments": json.dumps(raw)}}]}}]}


def model(client, **options):
    return MultimodalPlanner(
        "https://offline.invalid", "strong-fixture", api_key="offline", client=client,
        profile="desktop", desktop_contract="property_intent_v52",
        desktop_output_format=options.pop("desktop_output_format", FORMAT),
        desktop_planning_observation="full_multimodal_v1", enable_thinking=True,
        request_timeout_s=90, **options,
    )


def capture(snapshot, raw, *, limit=6, output_format=FORMAT, context=None, content=None):
    requests = []

    def answer(request):
        requests.append(request)
        response = (packet(raw, content=content) if output_format == FORMAT else
                    {"choices": [{"message": {"content": json.dumps(raw)}}]})
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client, desktop_output_format=output_format)
        if context is not None:
            planner.set_prepare_repair_context(context)
        plan = planner.plan("Apply the requested setting", snapshot, [], max_actions=limit)
    assert len(requests) == planner.calls == 1
    return plan, requests[0], planner


@pytest.mark.parametrize("limit", [6, 1])
def test_actual_H6_and_H1_schema_full_image_thinking_usage_and_single_HTTP(snapshot, limit):
    raw = property_plan() if limit > 1 else immediate()
    plan, request, planner = capture(snapshot, raw, limit=limit, content="Ignored narration")
    body = json.loads(request.content)
    expected = desktop_response_format(limit, property_contract="property_intent_v52", image_uploaded=True)
    parameters = body["tools"][0]["function"]["parameters"]
    assert parameters == function_parameters(expected["json_schema"]["schema"])
    assert Draft202012Validator(parameters).is_valid(raw)
    assert body["tools"][0]["function"]["name"] == FUNCTION and len(body["tools"]) == 1
    assert "response_format" not in body and "strict" not in body["tools"][0]["function"]
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert body["enable_thinking"] is True and "thinking_budget" not in body
    assert [part["type"] for part in body["messages"][1]["content"]] == ["text", "image_url"]
    assert request.extensions["timeout"]["read"] == 90
    assert planner.last_request_image_uploaded is True
    assert planner.last_request_provenance["response_transport"] == FORMAT
    assert planner.last_response_transport["accompanying_content_present"] is True
    assert "Ignored narration" not in json.dumps(planner.last_response_transport)
    assert planner.last_usage["completion_details_reasoning_tokens"] == 4
    assert planner.last_usage["completion_details_text_tokens"] == 3
    assert planner.usage_totals == planner.last_usage
    if limit > 1:
        assert plan.reusable and plan.complete_property_goal
        assert plan.steps[0].property_edit.numeric_finalize_key == "enter"
        assert plan.steps[1].property_confirmation_of == "property"
    else:
        assert len(plan.steps) == 1 and plan.steps[0].operation == "fill"


def test_complete_immediate_grammar_keeps_current_guarded_keyboard(snapshot):
    raw = {"summary": "Type in current focused field", "done": False,
           "steps": [{"id": "s1", "operation": "type_text", "value": "requested text",
                      "preconditions": [{"kind": "focused", "target": {"name": "Search"}}],
                      "postconditions": [{"kind": "value_equals", "target": {"name": "Search"},
                                          "value": "requested text"}]}]}
    plan, request, _planner = capture(snapshot, raw, limit=1)
    parameters = json.loads(request.content)["tools"][0]["function"]["parameters"]
    assert Draft202012Validator(parameters).is_valid(raw)
    assert plan.prepare_repair_token == "" and plan.steps[0].operation == "type_text"
    assert plan.steps[0].preconditions and plan.steps[0].postconditions


def test_original_prepare_repair_token_only_authorizes_current_visual_fill_grammar(snapshot):
    raw = {"summary": "Repair current preparation", "done": False, "prepare_repair": TOKEN,
           "steps": [{"id": "s1", "operation": "visual_fill", "label": "Visible search editor",
                      "box": [100, 100, 400, 200], "value": "requested query"}]}
    context = {"token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}}}
    plan, request, _planner = capture(snapshot, raw, limit=1, context=context)
    parameters = json.loads(request.content)["tools"][0]["function"]["parameters"]
    assert parameters["properties"]["prepare_repair"] == {"const": TOKEN}
    assert Draft202012Validator(parameters).is_valid(raw)
    assert plan.prepare_repair_token == TOKEN and plan.steps[0].operation == "visual_fill"


@pytest.mark.parametrize("limit", [6, 1])
def test_tool_body_is_only_reviewed_transport_delta_from_original_schema_wire(snapshot, limit):
    raw = property_plan() if limit > 1 else immediate()
    original_plan, original_request, _ = capture(snapshot, raw, limit=limit, output_format="json_schema_v50")
    plan, tool_request, _ = capture(snapshot, raw, limit=limit)
    original, changed = json.loads(original_request.content), json.loads(tool_request.content)
    for key in ("tools", "tool_choice", "parallel_tool_calls"):
        changed.pop(key)
    changed["messages"][0]["content"] = changed["messages"][0]["content"].removesuffix(TOOL_GUIDANCE)
    changed["response_format"] = original["response_format"]
    assert changed == original and plan == original_plan


@pytest.mark.parametrize("kind", ["absent", "multiple", "wrong_name", "invalid_JSON", "array", "semantic"])
def test_wire_and_semantic_failures_are_one_original_failure_without_content_fallback(snapshot, kind):
    secret = "private-provider-content-never-log"
    response = packet(property_plan(), content=secret)
    message = response["choices"][0]["message"]
    function = message["tool_calls"][0]["function"]
    if kind == "absent":
        message.pop("tool_calls")
        message["content"] = json.dumps(property_plan())
    elif kind == "multiple":
        message["tool_calls"] *= 2
    elif kind == "wrong_name":
        function["name"] = secret
    elif kind == "invalid_JSON":
        function["arguments"] = "{" + secret
    elif kind == "array":
        function["arguments"] = "[]"
    else:
        function["arguments"] = json.dumps({"intent": "pursue_property", "label": secret})
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client)
        with pytest.raises(ValueError, match="invalid multimodal planning response") as error:
            planner.plan("Apply setting", snapshot, [], max_actions=6)
    assert len(requests) == planner.calls == 1
    assert secret not in str(error.value) + planner.last_validation_error
    assert planner.last_usage["completion_tokens"] == 7


@pytest.mark.parametrize("failure", ["HTTP", "timeout"])
def test_transport_errors_preserve_timeout_path_without_fallback_or_raw_error(snapshot, failure):
    requests = []
    secret = "private-endpoint-body-do-not-log"

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
    assert planner.last_response_transport is None
    assert secret not in str(error.value) + planner.last_validation_error


def test_expired_deadline_sends_no_HTTP_and_clears_previous_transport(snapshot):
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(property_plan()))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client)
        planner.plan("Apply setting", snapshot, [], max_actions=6)
        planner.set_episode_deadline(time.perf_counter() - 1)
        with pytest.raises(TimeoutError, match="expired before model request"):
            planner.plan("Apply setting", snapshot, [], max_actions=6)
        assert planner.last_response_transport is None and planner.last_request_provenance is None
    assert len(requests) == planner.calls == 1


class Environment:
    def __init__(self, snapshot):
        self.snapshot, self.actions = snapshot, []

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        self.snapshot = replace(self.snapshot, text="Observed Help action effect")
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


@pytest.mark.parametrize("valid", [True, False])
def test_runner_records_wire_provenance_and_existing_failure_count_without_hidden_request(snapshot, valid):
    env, trace, requests = Environment(snapshot), JsonlTrace(), []
    raw = {"intent": "act_current", "label": "Current Help", "action": {"operation": "click", "ref": "help"}}

    def answer(request):
        requests.append(request)
        payload = packet(raw, content="private-accompanying-content") if valid else {
            "choices": [{"message": {"content": json.dumps(raw)}}]}
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = model(client)
        result = AgentRunner(Backend(env, pause=0), planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, max_steps=1, max_plans=1,
                               max_failures=2, plan_horizon=6, timeout_s=30)).run("Use current Help")
    assert len(requests) == result.planner_calls == planner.calls == 1
    transport = [event["payload"] for event in trace.events
                 if event["kind"] == "agent_model_output_transport"]
    assert len(transport) == 1 and transport[0]["response_transport"] == FORMAT
    assert transport[0]["image_uploaded"] is True and transport[0]["max_actions"] == 6
    assert "private-accompanying-content" not in json.dumps(trace.events)
    if valid:
        assert len(env.actions) == 1 and transport[0]["declaration"]["accompanying_content_present"]
    else:
        assert env.actions == [] and transport[0]["declaration"] is None
        assert any(event["kind"] == "agent_recovery" and event["payload"].get("reason") == "invalid_plan"
                   for event in trace.events)
