"""INERT mock-only checks for bounded program diagnostic facts, never authority."""

import copy
import json

import httpx
import pytest
from test_semantic_tools_v57 import current, packet
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_contract_v60 import declaration

from cua_jev import desktop_declaration_rejection as rejection
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.desktop_declaration_envelope import declaration_envelope_facts
from cua_jev.desktop_semantic_tools_v60 import FORMAT, PROGRAM_TOOL
from cua_jev.desktop_stage_program import StageProgramPlan
from cua_jev.trace import JsonlTrace

PRIVATE = "PRIVATE_PROGRAM_ARGUMENT_LABEL_ID_VALUE_b719"
NATIVE = "declare_native_current_action"
GOAL = "Inspect the original requested generic desktop state."


def private_program():
    raw = declaration()
    raw["label"] = PRIVATE
    for index, stage in enumerate(raw["stages"]):
        stage["id"] = PRIVATE + str(index)
        stage["arguments"]["label"] = PRIVATE
    raw["stages"][1]["arguments"]["edits"][0]["value"] = PRIVATE
    return raw


def response(arguments, name=PROGRAM_TOOL):
    payload = packet({}, name, content=PRIVATE)
    payload.update(id=PRIVATE, headers={"Authorization": PRIVATE})
    call = payload["choices"][0]["message"]["tool_calls"][0]
    call["id"] = PRIVATE
    call["function"]["arguments"] = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return payload


def function_facts(envelope):
    message = envelope["choices"]["items"][0]["message"]["shape"]
    return message["tool_calls"]["items"][0]["function"]["shape"]


def assert_trace_only(facts):
    assert facts["untrusted"] is True
    assert facts["input_authority"] is facts["schema_validated"] is False
    rendered = json.dumps(facts)
    assert PRIVATE not in rendered
    assert "Authorization" not in rendered and "sha256" not in rendered
    assert "accepted" not in facts and "actions" not in facts


@pytest.fixture
def actual_tools(observed):
    tools = {}
    for context, fmt, options in (
        ("normal", FORMAT, {}),
        ("formal-review", FORMAT, {"whole_goal_recheck": True}),
        ("ancestor", "semantic_tools_v58", {}),
    ):
        requests = []

        def answer(request, *, collected=requests):
            collected.append(json.loads(request.content))
            return httpx.Response(200, json=packet(current(6), NATIVE))

        with httpx.Client(transport=httpx.MockTransport(answer)) as client:
            planner = make_planner(client, fmt=fmt)
            planner.plan(GOAL, observed, [], max_actions=6, **options)
        assert len(requests) == planner.calls == 1
        tools[context] = requests[0]["tools"]
    assert len(tools["normal"]) == 9
    assert tools["formal-review"] == tools["ancestor"] and len(tools["ancestor"]) == 8
    return tools


def test_program_name_is_observable_only_when_actually_offered_and_unknown_names_stay_private(actual_tools):
    payload = response(private_program())
    before = copy.deepcopy(payload)
    for context, expected in (("normal", PROGRAM_TOOL), ("formal-review", "unoffered")):
        names = tuple(item["function"]["name"] for item in actual_tools[context])
        facts = declaration_envelope_facts(payload, offered_function_names=names)
        assert function_facts(facts)["name"]["enum"] == expected
        assert_trace_only(facts)
    unknown = response({}, PRIVATE)
    facts = declaration_envelope_facts(unknown, offered_function_names=(PRIVATE, PROGRAM_TOOL))
    assert function_facts(facts)["name"]["enum"] == "unoffered"
    assert_trace_only(facts)
    assert payload == before


def test_complete_program_keeps_safe_envelope_and_shape_inspection_is_not_validation(observed, actual_tools):
    payload, requests = response(private_program()), []
    original = copy.deepcopy(payload)

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT)
        wrapped = planner.plan(GOAL, observed, [], max_actions=6)
    assert isinstance(wrapped, StageProgramPlan) and len(requests) == planner.calls == 1
    assert planner.last_declaration_rejection is None
    envelope = planner.last_response_transport["declaration_envelope"]
    assert function_facts(envelope)["name"]["enum"] == PROGRAM_TOOL
    assert_trace_only(envelope)
    # A manual shape read remains untrusted even when its text is a complete
    # declaration. It neither accepts a program nor validates any member/field.
    facts = rejection.declaration_rejection_facts(
        payload, offered_tools=actual_tools["normal"], error=ValueError(PRIVATE),
    )
    assert facts["argument_parse_status"] == "parsed" and facts["root_type"] == "object"
    assert facts["code"] == "adapter_value_error"
    assert facts["top_level"]["known_required_missing_count"] == 0
    assert facts["top_level"]["known_fields"]["label"] == {"present": True, "type": "string"}
    assert facts["top_level"]["known_fields"]["stages"] == {
        "present": True, "type": "array", "count_capped": 2, "overflow": False,
        "item_types_prefix": {"object": 2}, "item_type_prefix_complete": True,
    }
    assert_trace_only(facts)
    assert payload == original


def rejected_cases():
    extra = private_program()
    extra[PRIVATE] = PRIVATE
    future_authority = private_program()
    future_authority["stages"][1]["arguments"]["edits"][0]["ref"] = PRIVATE
    invalid_policy = private_program()
    invalid_policy["stages"][0]["recovery"] = PRIVATE
    return [
        (extra, "program_structure", "program_outer_fields_rejected", "parsed"),
        ({"label": PRIVATE, "stages": []},
         "program_structure", "program_stage_count_rejected", "parsed"),
        ({"label": PRIVATE, "stages": PRIVATE},
         "program_structure", "program_stage_count_rejected", "parsed"),
        ({"label": PRIVATE, "stages": [{PRIVATE: PRIVATE}, {}]},
         "program_structure", "program_stage_fields_rejected", "parsed"),
        (future_authority, "program_structure",
         "future_stage_requires_semantic_declarations_without_current_inputs", "parsed"),
        (invalid_policy, "program_structure", "program_recovery_policy", "parsed"),
        ('{"label":"' + PRIVATE + '","label":"' + PRIVATE + '","stages":[]}',
         "declaration_arguments", "program_duplicate_json_field", "parsed"),
        ('{"label":"' + PRIVATE + '","stages":[',
         "declaration_arguments", "argument_decode_rejected", "invalid_json"),
    ]


class ObserveOnlyBackend:
    environment_steps = 0

    def __init__(self, snapshot):
        self.snapshot = snapshot

    def observe(self):
        return self.snapshot

    def compile(self, *_args):
        pytest.fail("A rejected declaration must never reach candidate compilation")

    def execute(self, *_args):
        pytest.fail("Diagnostics cannot send an input")

    def verify(self, *_args):
        pytest.fail("A rejected declaration must never reach effect verification")


class NeverChoose:
    def choose(self, *_args):
        pytest.fail("Diagnostics cannot request an action choice")


@pytest.mark.parametrize("arguments,stage,code,status", rejected_cases())
def test_actual_program_rejection_retains_one_call_safe_trace_and_zero_inputs(
    observed, arguments, stage, code, status,
):
    payload, requests = response(arguments), []
    original = copy.deepcopy(payload)

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    backend, trace = ObserveOnlyBackend(observed), JsonlTrace()
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT)
        result = AgentRunner(backend, planner, policy=NeverChoose(), trace=trace, config=AgentConfig(
            allow_actions=True, whole_goal_recheck=True, plan_horizon=6,
            max_plans=1, max_steps=25, max_failures=1, timeout_s=420,
        )).run(GOAL)
    facts = planner.last_declaration_rejection
    assert len(requests) == planner.calls == result.planner_calls == 1
    assert result.steps == backend.environment_steps == 0 and not result.success
    assert (facts["stage"], facts["code"], facts["argument_parse_status"]) == (stage, code, status)
    assert_trace_only(facts)
    assert function_facts(planner.last_response_structure)["name"]["enum"] == PROGRAM_TOOL
    recoveries = [event["payload"] for event in trace.events if event["kind"] == "agent_recovery"]
    assert len(recoveries) == 1 and recoveries[0]["declaration_rejection"] == facts
    assert not any(event["kind"] in {"candidates", "receipt", "agent_step", "agent_program_declared"}
                   for event in trace.events)
    assert PRIVATE not in json.dumps(trace.events)
    assert payload == original


def test_program_stages_report_only_bounded_root_shape_and_never_member_values(actual_tools):
    payload = response({"label": PRIVATE, "stages": [{PRIVATE: PRIVATE}] * 40, PRIVATE: PRIVATE})
    facts = rejection.declaration_rejection_facts(
        payload, offered_tools=actual_tools["normal"], error=ValueError("program_outer_fields"),
    )
    stages = facts["top_level"]["known_fields"]["stages"]
    assert stages == {"present": True, "type": "array", "count_capped": 32, "overflow": True,
                      "item_types_prefix": {"object": 8}, "item_type_prefix_complete": False}
    assert facts["top_level"]["unknown_key_count_capped"] == 1
    assert len(json.dumps(facts)) < 8000
    assert_trace_only(facts)


@pytest.mark.parametrize("error,code", [
    (ValueError("program_stage_count_" + PRIVATE), "adapter_value_error"),
    (TypeError("program_stage_count"), "adapter_type_error"),
])
def test_unknown_or_wrong_type_error_never_masquerades_as_fixed_program_rejection(actual_tools, error, code):
    facts = rejection.declaration_rejection_facts(
        response(private_program()), offered_tools=actual_tools["normal"], error=error,
    )
    assert facts["stage"] == "declaration_adapter" and facts["code"] == code
    assert_trace_only(facts)


def test_unoffered_program_or_unknown_name_does_not_parse_arguments(actual_tools, monkeypatch):
    program, unknown = response(private_program()), response({}, PRIVATE)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Only a known actually offered schema permits diagnostic JSON inspection")

    monkeypatch.setattr(rejection.json, "loads", forbidden)
    for payload, tools in ((program, actual_tools["formal-review"]), (unknown, actual_tools["normal"])):
        facts = rejection.declaration_rejection_facts(payload, offered_tools=tools, error=ValueError(PRIVATE))
        assert facts["argument_parse_status"] == "known_offered_schema_unavailable"
        assert_trace_only(facts)


def test_all_original_function_diagnostics_are_identical_with_or_without_program_offer(actual_tools):
    original_names = tuple(item["function"]["name"] for item in actual_tools["ancestor"])
    new_names = tuple(item["function"]["name"] for item in actual_tools["normal"])
    for name in original_names:
        payload = response({"label": PRIVATE, "value": PRIVATE, "id": PRIVATE, PRIVATE: PRIVATE}, name)
        before = copy.deepcopy(payload)
        assert declaration_envelope_facts(payload, offered_function_names=original_names) == (
            declaration_envelope_facts(payload, offered_function_names=new_names)
        )
        old = rejection.declaration_rejection_facts(
            payload, offered_tools=actual_tools["ancestor"], error=ValueError("unknown declaration function"),
        )
        new = rejection.declaration_rejection_facts(
            payload, offered_tools=actual_tools["normal"], error=ValueError("unknown declaration function"),
        )
        assert old == new and old["code"] == "function_not_allowed"
        assert_trace_only(new)
        assert payload == before


def test_program_rejection_diagnostic_does_not_enter_next_request_and_clears_on_valid_old_route(observed):
    requests = []
    replies = iter([response({"label": PRIVATE, "stages": []}), packet(current(6), NATIVE)])

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=next(replies))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = make_planner(client, fmt=FORMAT)
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            planner.plan(GOAL, observed, [], max_actions=6)
        assert planner.last_declaration_rejection["code"] == "program_stage_count_rejected"
        actual = planner.plan(GOAL, observed, [], max_actions=6)
    assert actual.steps[0].ref == "help" and len(requests) == planner.calls == 2
    assert planner.last_declaration_rejection is None
    assert requests[0].content == requests[1].content
    assert PRIVATE not in requests[1].content.decode("utf-8")
    assert "declaration_rejection" not in requests[1].content.decode("utf-8")


def test_program_diagnostic_internal_failure_keeps_fixed_safe_fallback(actual_tools, monkeypatch):
    def fail(*_args):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(rejection, "_facts", fail)
    facts = rejection.declaration_rejection_facts(
        response(private_program()), offered_tools=actual_tools["normal"], error=ValueError(PRIVATE),
    )
    assert facts["code"] == "adapter_rejected" and facts["details_status"] == "unavailable"
    assert_trace_only(facts)
