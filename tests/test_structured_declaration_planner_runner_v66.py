"""Mock HTTP and memory-backend integration; no real API or desktop input."""

import copy
import hashlib
import json
from dataclasses import replace

import httpx
import pytest
from test_prepare_repair_runner import Clock, Desktop, root_plan
from test_prepare_repair_runner import png as png
from test_semantic_tools_v57 import current
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import packet as function_packet
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_contract_v60 import declaration
from test_stage_program_planning_preference_v61 import requested as function_requested
from test_stage_program_runner_v60 import MemoryFields, events, field_member

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.desktop_plan_repair import repair_property_prepare, suspend_property_prepare
from cua_jev.desktop_stage_program import StageProgramPlan
from cua_jev.trace import JsonlTrace

FORMAT = "semantic_json_v66"
PROVENANCE_KEYS = (
    "wire_declaration_json", "wire_declaration_sha256", "normalized_declaration_json",
    "normalized_declaration_sha256", "wire_stage_mapping", "codec_version", "codec_source_hashes",
)


def packet(raw, name="declare_native_current_action"):
    content = json.dumps({"tool": name, "arguments": raw}, ensure_ascii=False, indent=2)
    return {"usage": {"prompt_tokens": 20, "completion_tokens": 7}, "choices": [{
        "finish_reason": "stop", "message": {"content": content},
    }]}


def requested(observed, raw, tool, *, fmt=FORMAT, limit=6, planning_options=None):
    if fmt != FORMAT or limit == 1 or (planning_options or {}).get("whole_goal_recheck"):
        return function_requested(observed, raw, tool, fmt=fmt, limit=limit,
                                  planning_options=planning_options)
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=packet(raw, tool))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        instance = make_planner(client, fmt=fmt)
        plan = instance.plan("Complete the known requested desktop result", observed, [],
                             max_actions=limit, **(planning_options or {}))
    assert instance.calls == len(requests) == 1
    return plan, requests[0], instance


def wire(source=None):
    result = copy.deepcopy(source or declaration())
    for member in result["stages"]:
        del member["arguments"]["completion"]
    return result


class RecordingPlanner(MultimodalPlanner):
    def _parse_program_stage(self, raw, snapshot, max_actions):
        self.compiled.append((copy.deepcopy(raw), snapshot, max_actions))
        return super()._parse_program_stage(raw, snapshot, max_actions)


def test_actual_HTTP_wire_uses_production_parser_first_compile_and_honest_hashes(observed):
    raw = wire()
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=packet(raw, "declare_stage_program"))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT, cls=RecordingPlanner)
        planner.compiled = []
        plan = planner.plan("Reach the requested panel and fill its requested query",
                            observed, [], max_actions=6)
    assert isinstance(plan, StageProgramPlan) and len(requests) == planner.calls == 1
    assert len(planner.compiled) == 1 and planner.compiled[0][1] is observed
    assert planner.compiled[0][0]["intent"] == "navigate" and planner.compiled[0][2] == 6
    assert plan.initial_plan.reusable and not plan.initial_plan.complete_property_goal
    assert not plan.initial_plan.completion_conditions
    transport, public = planner.last_response_transport, plan.program.to_dict()
    assert public["source_declaration_status"] == "codec_normalized_canonical_json"
    for key in PROVENANCE_KEYS:
        assert public[key] == transport[key]
    assert json.loads(transport["wire_declaration_json"]) == raw
    normalized = json.loads(transport["normalized_declaration_json"])
    assert all(member["arguments"]["completion"] == "stage" for member in normalized["stages"])
    assert transport["wire_declaration_sha256"] != transport["normalized_declaration_sha256"]
    for prefix in ("wire", "normalized"):
        assert hashlib.sha256(transport[prefix + "_declaration_json"].encode()).hexdigest() == (
            transport[prefix + "_declaration_sha256"])
    actual_arguments = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert transport["argument_sha256"] == hashlib.sha256(actual_arguments.encode()).hexdigest()
    assert transport["argument_sha256"] == transport["canonical_arguments_sha256"]
    assert transport["argument_hash_kind"] == "adapter_canonical_arguments"
    assert transport["transport_kind"] == "json_schema_content"
    assert transport["provider_tool_call_count"] == transport["tool_call_count"] == 0
    assert transport["internal_adapter"]["kind"] == "synthetic_legacy_mapper_payload_not_provider"
    actual_content = packet(raw, "declare_stage_program")["choices"][0]["message"]["content"]
    assert transport["observed_content_utf8_sha256"] == hashlib.sha256(actual_content.encode()).hexdigest()
    assert transport["observed_content_utf8_sha256"] != transport["argument_sha256"]
    body = json.loads(requests[0].content)
    assert not any(key in body for key in ("tools", "tool_choice", "parallel_tool_calls"))
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["enable_thinking"] is True and "thinking_budget" not in body
    assert len(transport["offered_tool_names"]) == 6
    assert transport["normalized_declaration_sha256"] == plan.program.declaration_sha256
    assert planner.last_request_provenance["max_actions"] == 6
    assert planner.last_request_provenance["declaration_argument_sha256"] == transport["argument_sha256"]


class ProductionMemoryFields(MemoryFields):
    def compile(self, step, planned, current):
        # The legacy memory receipt fixture keys values by candidate description.
        # Bind that fixture label to the actual semantic target; do not replace
        # production parsing, target validation, current proof or final verifier.
        if step.operation == "fill":
            step = replace(step, label=step.target.name)
        return super().compile(step, planned, current)


def test_program_public_provenance_reaches_runner_and_future_activation_from_fresh_frame():
    backend = ProductionMemoryFields(reveal_after_first=True)
    raw = wire({"label": "Fill both requested fields", "stages": [
        field_member("first", ("Name",)), field_member("second", ("City",)),
    ]})
    conditions = [{"kind": "value_equals", "target": {"name": name, "role": "textbox",
                    "app": "Editor", "window": "Main"}, "value": "requested-" + name}
                  for name in ("Name", "City")]
    responses = iter([packet(raw, "declare_stage_program"), function_packet(
        {"label": "Both requested values observed", "conditions": conditions}, "declare_completion")])
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=next(responses))

    trace = JsonlTrace()
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT, cls=RecordingPlanner)
        planner.compiled = []
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, mode="hybrid", plan_horizon=6,
                whole_goal_recheck=True, max_plans=2, max_failures=3, max_steps=6)).run(
                    "Fill Name with requested-Name and City with requested-City")
    assert len(requests) == planner.calls == result.planner_calls == 2
    assert backend.executed == ["Name", "City"] and backend.environment_steps == result.steps == 2
    assert len(planner.compiled) == 2
    assert planner.compiled[0][1] is not planner.compiled[1][1]
    assert "City" not in {control.label for control in planner.compiled[0][1].controls}
    assert "City" in {control.label for control in planner.compiled[1][1].controls}
    declared = events(trace, "agent_program_declared")
    assert len(declared) == 1
    public = declared[0]["declaration"]
    assert public["source_declaration_status"] == "codec_normalized_canonical_json"
    assert json.loads(public["wire_declaration_json"]) == raw
    assert public["wire_declaration_sha256"] != public["normalized_declaration_sha256"]
    completed = events(trace, "agent_program_stage_completed")
    assert [item["stage_index"] for item in completed] == [0, 1]
    assert len({item["program_id"] for item in completed}) == 1
    assert events(trace, "agent_program_completed")[0]["whole_goal_review_still_required"] is True
    assert not result.success  # A fake receipt or program never supplies an official evaluator.


def test_actual_normal_context_request_sends_zero_input_and_uses_original_fresh_H1():
    backend = ProductionMemoryFields()
    answers = iter([packet({"reason": "missing_future_semantics"}, "declare_current_context_needed"),
                    function_packet({"label": "Observed field", "conditions": [{"kind": "exists", "target": {
                        "name": "Name", "role": "textbox", "app": "Editor", "window": "Main"}}]},
                        "declare_completion")])
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid" and backend.environment_steps == 0
        requests.append(request)
        return httpx.Response(200, json=next(answers))

    trace = JsonlTrace()
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT)
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, mode="hybrid", plan_horizon=6,
                max_plans=2, max_failures=3)).run("Inspect the currently observed Name field")
    assert len(requests) == planner.calls == result.planner_calls == 2
    assert backend.environment_steps == result.steps == result.plan_reuses == 0
    handoff = events(trace, "agent_high_level_context_requested")
    assert len(handoff) == 1 and handoff[0]["logical_attempts"] == 0
    assert handoff[0]["positive_input_credit"] is False and handoff[0]["next_max_actions"] == 1
    assert planner.last_request_provenance["max_actions"] == 1
    assert planner.last_request_provenance["normal_high_level_phase"] is False
    assert planner.last_request_provenance["declaration_tool_count"] == 5
    assert not events(trace, "agent_program_declared")


@pytest.mark.parametrize("limit,formal", [(1, False), (1, True), (6, True)])
@pytest.mark.parametrize("tool", ["declare_native_current_action", "declare_completion"])
def test_H1_and_formal_review_actual_request_bytes_remain_exact58(observed, limit, formal, tool):
    raw = ({"label": "Observed completion", "conditions": [{"kind": "text_contains", "value": "Editor"}]}
           if tool == "declare_completion" else current(limit, tool))
    options = {"whole_goal_recheck": True} if formal else {}
    expected, old, _ = requested(observed, raw, tool, fmt="semantic_tools_v58", limit=limit,
                                 planning_options=options)
    actual, new, _ = requested(observed, raw, tool, fmt=FORMAT, limit=limit, planning_options=options)
    assert actual == expected and new.content == old.content


def test_real_prepare_repair_H1_actual_request_is_exact58(png):
    backend = Desktop(png, Clock(), confirm=True)
    planned = backend.snapshot
    original = root_plan(planned, confirm=True)
    prop = original.steps[1]
    prop = replace(prop, property_edit=replace(prop.property_edit, numeric_finalize_key="enter"))
    original = replace(original, steps=(original.steps[0], prop, *original.steps[2:]))
    backend.drift()
    frame = suspend_property_prepare(original, planned, backend.snapshot, plan_at=100.0,
        ttl_s=60.0, horizon=len(original.steps) + 1, now=101.0)
    assert frame is not None
    context = frame.context(backend.snapshot, now=102.0)
    history = [{"reason": "no_executable_target", "recovery_required": True,
                "prepare_repair_offered": True, "feedback": "No input was sent."}]
    raw = {"summary": "Replace the unexecuted preparation", "id": "fresh", "done": False,
        "operation": "visual_fill", "action_label": "Fresh current input", "box_space": "normalized_1000",
        "box": [302, 131, 820, 149], "value": context["value"], "prepare_repair": frame.token}
    captured = []
    for fmt in ("semantic_tools_v58", FORMAT):
        requests = []

        def answer(request, requests=requests):
            assert request.url.host == "offline.invalid"
            requests.append(request)
            return httpx.Response(200, json=function_packet(raw, "declare_visual_current_action"))

        with httpx.Client(transport=httpx.MockTransport(answer)) as client:
            planner = make_planner(client, fmt=fmt)
            planner.set_prepare_repair_context(context)
            plan = planner.plan("Complete the requested query", backend.snapshot, history, max_actions=6)
        assert planner.calls == len(requests) == 1
        captured.append((plan, requests[0], planner))
    expected, ancestor, _ = captured[0]
    actual, request, planner = captured[1]
    assert request.content == ancestor.content and actual == expected
    assert actual.prepare_repair_token == frame.token and not actual.reusable and not actual.done
    assert len(actual.steps) == 1 and actual.steps[0].operation == "visual_fill"
    assert actual.steps[0].value == context["value"] and not actual.completion_conditions
    body = json.loads(request.content)
    state = json.loads(body["messages"][1]["content"][0]["text"])
    assert state["prepare_repair_offer"] == context and state["recent_actions"] == history
    schema = next(item["function"]["parameters"] for item in body["tools"]
                  if item["function"]["name"] == "declare_visual_current_action")
    assert schema["properties"]["prepare_repair"] == {"const": frame.token}
    assert planner.last_request_provenance["max_actions"] == 1
    assert planner.last_request_provenance["normal_high_level_phase"] is False
    repaired = repair_property_prepare(frame, actual, backend.snapshot, now=102.0)
    assert repaired.steps[1:] == original.steps[1:]
    assert repaired.steps[1].property_edit.numeric_finalize_key == "enter"
    assert context["remaining_time_s"] == 58.0 and frame.deadline == 160.0
    assert backend.actions == []


def test_old62_illegal_goal_member_is_still_rejected_before_any_stage_activation(observed):
    raw = copy.deepcopy(declaration())
    raw["stages"][1]["arguments"]["completion"] = "goal"
    with pytest.raises(ValueError):
        requested(observed, raw, "declare_stage_program", fmt="semantic_tools_v62")


@pytest.mark.parametrize("key,value", [("completion", "stage"), ("completion", "goal"),
                                      ("goal_conditions", [])])
def test_new_explicit_member_completion_fields_rejected_before_activation(observed, key, value):
    raw = wire()
    raw["stages"][1]["arguments"][key] = value
    with pytest.raises(ValueError):
        requested(observed, raw, "declare_stage_program", fmt=FORMAT)


def test_old64_normal_function_request_and_actual_argument_hash_are_unchanged(observed):
    raw = wire()
    plan, request, instance = function_requested(observed, raw, "declare_stage_program",
                                                fmt="semantic_tools_v64")
    assert isinstance(plan, StageProgramPlan)
    body = json.loads(request.content)
    assert len(body["tools"]) == 6 and body["tool_choice"] == "auto"
    assert body["parallel_tool_calls"] is False and "response_format" not in body
    assert instance.last_response_transport["argument_sha256"] == hashlib.sha256(
        json.dumps(raw).encode()).hexdigest()
    assert instance.last_response_transport["wire_argument_hash_status"] == (
        "observed_original_text_not_reconstructed")


@pytest.mark.parametrize("content", [
    "PRIVATE_PLAIN_RESPONSE", "Here is the declaration: {}", "```json\n{}\n```",
    '{"tool":"unknown_declaration","arguments":{}}',
    '{"tool":"declare_current_context_needed","arguments":{"reason":"unknown"}}',
    '{"tool":"declare_current_context_needed","arguments":{"reason":"missing_future_semantics"},"extra":1}',
])
def test_rejected_actual_JSON_content_sends_zero_input_and_preserves_failure_budget(content):
    backend = ProductionMemoryFields()
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
                                                      "message": {"content": content}}]})

    trace = JsonlTrace()
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT, cls=RecordingPlanner)
        planner.compiled = []
        result = AgentRunner(backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, mode="hybrid", plan_horizon=6,
                max_plans=1, max_failures=1)).run("Fill the requested two fields")
    assert len(requests) == planner.calls == result.planner_calls == 1
    assert result.steps == backend.environment_steps == result.plan_reuses == 0
    assert backend.executed == [] and planner.compiled == [] and not result.success
    assert not events(trace, "agent_program_declared")
    rejection = planner.last_declaration_rejection
    assert rejection == {"version": "bounded_json_content_declaration_rejection_v66",
        "stage": "json_content_adapter", "code": "json_content_declaration_rejected",
        "untrusted": True, "input_authority": False, "schema_validated": False}
    assert content not in json.dumps(rejection)
