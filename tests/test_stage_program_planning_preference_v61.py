"""Mock-only regressions for program preference without restricting valid routes.

INERT draft: not collected or executed until explicitly promoted after source60.
"""

import copy
import json
from dataclasses import replace

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_prepare_repair_runner import Clock, Desktop, root_plan
from test_prepare_repair_runner import png as png
from test_semantic_tools_v57 import FIELD, current, packet
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import planner as make_planner
from test_stage_program_contract_v60 import declaration, parse_original, query_panel

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentPlan
from cua_jev.desktop_plan_repair import repair_property_prepare, suspend_property_prepare
from cua_jev.desktop_semantic_tools_v60 import FORMAT, PROGRAM_GUIDANCE, PROGRAM_TOOL
from cua_jev.desktop_stage_program import StageProgramPlan

GOAL = "Complete the known requested query and the following requested result."
ANCESTOR = "semantic_tools_v58"
CURRENT_TOOLS = (
    "declare_native_current_action",
    "declare_semantic_current_action",
    "declare_visual_current_action",
    "declare_key_page_action",
)


def requested(observed, raw, tool, *, fmt=FORMAT, limit=6, planning_options=None):
    requests = []

    def answer(request):
        assert request.url.host == "offline.invalid"
        requests.append(request)
        return httpx.Response(200, json=packet(raw, tool))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        instance = make_planner(client, fmt=fmt)
        plan = instance.plan(GOAL, observed, [], max_actions=limit, **(planning_options or {}))
    assert len(requests) == instance.calls == 1
    return plan, requests[0], instance


def test_H6_prefers_program_first_without_changing_any_original_tool_or_other_request_field(observed):
    raw = current(6)
    _, previous, _ = requested(observed, raw, CURRENT_TOOLS[0], fmt=ANCESTOR)
    _, request, instance = requested(observed, raw, CURRENT_TOOLS[0])
    old, body = json.loads(previous.content), json.loads(request.content)
    names = [tool["function"]["name"] for tool in body["tools"]]
    assert len(old["tools"]) == 8 and len(names) == len(set(names)) == 9
    assert names[0] == PROGRAM_TOOL
    assert body["tools"][1:] == old["tools"]  # Original relative order AND entire schemas.
    assert body["tool_choice"] == "auto" and body["parallel_tool_calls"] is False
    assert instance.last_request_provenance["normal_program_eligible"] is True
    assert instance.last_request_provenance["program_tool_added"] is True
    assert instance.last_request_provenance["declaration_tool_count"] == 9
    schema = body["tools"][0]["function"]["parameters"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(declaration())

    # Check only the intended policy, not an entire brittle prompt snapshot.
    guidance = PROGRAM_GUIDANCE.casefold()
    assert "prefer declare_stage_program" in guidance
    assert "two or more" in guidance and "programs are optional" in guidance
    assert "not that future ui is already visible" in guidance
    assert "one property stage" in guidance and "complete result" in guidance
    assert "atomic/single-stage" in guidance
    assert body["messages"][0]["content"].endswith(PROGRAM_GUIDANCE)
    restored = copy.deepcopy(body)
    restored["tools"].pop(0)
    restored["messages"][0]["content"] = restored["messages"][0]["content"][:-len(PROGRAM_GUIDANCE)]
    assert restored == old  # Full image, state, history, model and all other fields remain exact.


@pytest.mark.parametrize("tool", [*CURRENT_TOOLS, "declare_completion"])
@pytest.mark.parametrize("options", [{}, {"whole_goal_recheck": True}], ids=["normal", "formal-review"])
def test_H1_current_and_completion_actual_requests_are_exact58_in_both_contexts(observed, tool, options):
    raw = ({"label": "Observed completion", "conditions": [{"kind": "text_contains", "value": "Editor"}]}
           if tool == "declare_completion" else current(1, tool))
    original, old_request, _ = requested(observed, raw, tool, fmt=ANCESTOR, limit=1, planning_options=options)
    actual, request, instance = requested(observed, raw, tool, limit=1, planning_options=options)
    assert request.content == old_request.content
    assert actual == original and isinstance(actual, AgentPlan)
    assert PROGRAM_TOOL not in {item["function"]["name"] for item in json.loads(request.content)["tools"]}
    assert instance.last_response_transport["strong_H1_mapping_exact_ancestor58"] is True
    expected_eligible = not options.get("whole_goal_recheck", False)
    assert instance.last_request_provenance["normal_program_eligible"] is expected_eligible
    assert instance.last_request_provenance["program_tool_added"] is False
    assert instance.last_request_provenance["declaration_tool_count"] == 5


def test_real_prepare_repair_context_keeps_actual_recovery_H1_exact58_and_retained_stage(png):
    # Reuse the real public-artifact repair fixture, including its numeric-key reservation.
    env = Desktop(png, Clock(), confirm=True)
    planned = env.snapshot
    original = root_plan(planned, confirm=True)
    prop = original.steps[1]
    prop = replace(prop, property_edit=replace(prop.property_edit, numeric_finalize_key="enter"))
    original = replace(original, steps=(original.steps[0], prop, *original.steps[2:]))
    horizon = len(original.steps) + 1
    env.drift()
    frame = suspend_property_prepare(
        original, planned, env.snapshot, plan_at=100.0, ttl_s=60.0, horizon=horizon, now=101.0,
    )
    assert frame is not None
    context = frame.context(env.snapshot, now=102.0)
    assert context is not None
    before = copy.deepcopy(context)
    history = [{"reason": "no_executable_target", "recovery_required": True,
                "prepare_repair_offered": True, "feedback": "No input was sent."}]
    raw = {
        "summary": "Replace the unexecuted preparation", "id": "fresh", "done": False,
        "operation": "visual_fill", "action_label": "Fresh current input",
        "box_space": "normalized_1000", "box": [302, 131, 820, 149],
        "value": context["value"], "prepare_repair": frame.token,
    }
    captured = []
    for fmt in (ANCESTOR, FORMAT):
        requests = []

        def answer(request, requests=requests):
            assert request.url.host == "offline.invalid"
            requests.append(request)
            return httpx.Response(200, json=packet(raw, "declare_visual_current_action"))

        with httpx.Client(transport=httpx.MockTransport(answer)) as client:
            instance = make_planner(client, fmt=fmt)
            instance.set_prepare_repair_context(context)
            # Public recovery history narrows the requested H6 to the original H1 path.
            plan = instance.plan(GOAL, env.snapshot, history, max_actions=6)
        assert len(requests) == instance.calls == 1
        captured.append((plan, requests[0], instance))
    ancestor, old_request, _ = captured[0]
    actual, request, instance = captured[1]
    assert request.content == old_request.content and actual == ancestor
    assert isinstance(actual, AgentPlan) and not isinstance(actual, StageProgramPlan)
    assert actual.prepare_repair_token == frame.token and not actual.done and not actual.reusable
    assert actual.completion_conditions == () and len(actual.steps) == 1
    assert actual.steps[0].operation == "visual_fill" and actual.steps[0].value == context["value"]
    body = json.loads(request.content)
    state = json.loads(body["messages"][1]["content"][0]["text"])
    assert state["prepare_repair_offer"] == before == context
    assert state["recent_actions"] == history
    assert PROGRAM_TOOL not in {tool["function"]["name"] for tool in body["tools"]}
    assert PROGRAM_GUIDANCE not in body["messages"][0]["content"]
    schema = next(tool["function"]["parameters"] for tool in body["tools"]
                  if tool["function"]["name"] == "declare_visual_current_action")
    assert schema["properties"]["prepare_repair"] == {"const": frame.token}
    Draft202012Validator(schema).validate(raw)
    assert instance.last_request_provenance["max_actions"] == 1
    assert instance.last_request_provenance["normal_program_eligible"] is False
    assert instance.last_request_provenance["program_tool_added"] is False
    assert instance.last_request_provenance["declaration_tool_count"] == 5
    assert instance.last_response_transport["strong_H1_mapping_exact_ancestor58"] is True
    repaired = repair_property_prepare(frame, actual, env.snapshot, now=102.0)
    assert repaired.steps[1:] == original.steps[1:]
    assert repaired.steps[1].property_edit.numeric_finalize_key == "enter"
    assert context["remaining_time_s"] == 58.0 and frame.deadline == 160.0 and frame.horizon == horizon
    assert env.actions == []


def alternatives():
    navigation = declaration()["stages"][0]["arguments"]
    property_goal = {
        "label": "One complete requested property result",
        "scope": {"app": "Editor"},
        "edit_operation": "fill",
        "edit_value": "requested literal",
        "numeric_finalize_key": "none",
        "confirmation_label": "none",
        "completion": "goal",
    }
    fields = {
        "label": "Fill the currently observed requested field",
        "edits": [{"operation": "fill", "target": FIELD, "value": "requested literal"}],
        "completion": "stage",
    }
    return [
        ("declare_navigation_stage", navigation),
        ("declare_property_stage", property_goal),
        ("declare_field_edits_stage", fields),
        *((tool, current(6, tool)) for tool in CURRENT_TOOLS),
        ("declare_completion", {
            "label": "Fresh currently observed completion",
            "conditions": [{"kind": "text_contains", "value": "Editor"}],
        }),
    ]


@pytest.mark.parametrize(
    "tool,raw", alternatives(), ids=lambda value: value if isinstance(value, str) else None,
)
def test_program_priority_keeps_all_original_legal_fallback_routes(observed, tool, raw):
    unchanged = copy.deepcopy(raw)
    original, _, _ = requested(observed, raw, tool, fmt=ANCESTOR)
    actual, request, instance = requested(observed, raw, tool)
    assert isinstance(actual, AgentPlan) and not isinstance(actual, StageProgramPlan)
    assert actual == original and raw == unchanged
    assert json.loads(request.content)["tools"][0]["function"]["name"] == PROGRAM_TOOL
    assert instance.last_response_transport["program_declaration"] is False
    if tool == "declare_property_stage":
        # A complete property is not split merely for preference.
        assert actual.complete_property_goal is True
    if tool in CURRENT_TOOLS:
        assert len(actual.steps) == 1 and not actual.steps[0].navigation


@pytest.mark.parametrize("tool", [*CURRENT_TOOLS, "declare_completion"])
def test_formal_H6_review_keeps_original_eight_tool_request_and_valid_current_or_completion(observed, tool):
    raw = ({"label": "Fresh whole-goal completion",
            "conditions": [{"kind": "text_contains", "value": "Editor"}]}
           if tool == "declare_completion" else current(6, tool))
    options = {"whole_goal_recheck": True}
    original, old_request, _ = requested(observed, raw, tool, fmt=ANCESTOR, planning_options=options)
    actual, request, instance = requested(observed, raw, tool, planning_options=options)
    body = json.loads(request.content)
    assert request.content == old_request.content and actual == original
    assert len(body["tools"]) == 8
    assert PROGRAM_TOOL not in {item["function"]["name"] for item in body["tools"]}
    assert PROGRAM_GUIDANCE not in body["messages"][0]["content"]
    assert instance.last_request_provenance["normal_program_eligible"] is False
    assert instance.last_request_provenance["program_tool_added"] is False
    assert instance.last_request_provenance["declaration_tool_count"] == 8


def test_formal_review_still_rejects_unsolicited_program_without_retry_or_action(observed):
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(declaration(), PROGRAM_TOOL))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        instance = make_planner(client, fmt=FORMAT)
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            instance.plan(GOAL, observed, [], max_actions=6, whole_goal_recheck=True)
    assert len(requests) == instance.calls == 1
    assert PROGRAM_TOOL not in {item["function"]["name"] for item in json.loads(requests[0].content)["tools"]}
    assert instance.last_validation_error


def test_inband_reminder_keeps_normal_H6_program_permission_and_original_other_request_fields(observed):
    options = {"whole_goal_inband_epoch": 0}
    _, old_request, _ = requested(
        observed, current(6), CURRENT_TOOLS[0], fmt=ANCESTOR, planning_options=options,
    )
    actual, request, instance = requested(
        observed, declaration(), PROGRAM_TOOL, planning_options=options,
    )
    assert isinstance(actual, StageProgramPlan)
    body, previous = json.loads(request.content), json.loads(old_request.content)
    assert body["tools"][0]["function"]["name"] == PROGRAM_TOOL
    assert body["tools"][1:] == previous["tools"]
    assert instance.last_request_provenance["whole_goal_inband_review"] == {
        "version": "all_original_goal_review_v1", "input_epoch": 0,
    }
    assert instance.last_request_provenance["normal_program_eligible"] is True
    assert instance.last_request_provenance["program_tool_added"] is True
    assert instance.last_request_provenance["declaration_tool_count"] == 9
    restored = copy.deepcopy(body)
    restored["tools"].pop(0)
    assert restored["messages"][0]["content"].endswith(PROGRAM_GUIDANCE)
    restored["messages"][0]["content"] = restored["messages"][0]["content"][:-len(PROGRAM_GUIDANCE)]
    assert restored == previous


class RecordingOriginalStageParser(MultimodalPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parsed_stage_frames = []

    def _parse_program_stage(self, raw, snapshot, max_actions):
        self.parsed_stage_frames.append((copy.deepcopy(raw), snapshot, max_actions))
        return super()._parse_program_stage(raw, snapshot, max_actions)


def test_known_future_intent_compiles_only_on_actual_live_activation(observed):
    requests = []

    def answer(request):
        requests.append(request)
        return httpx.Response(200, json=packet(declaration(), PROGRAM_TOOL))

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        instance = make_planner(client, fmt=FORMAT, cls=RecordingOriginalStageParser)
        wrapped = instance.plan(GOAL, observed, [], max_actions=6)
        assert isinstance(wrapped, StageProgramPlan)
        assert len(instance.parsed_stage_frames) == 1
        assert instance.parsed_stage_frames[0][1] is observed
        assert instance.parsed_stage_frames[0][0]["intent"] == "navigate"
        assert all(control.label != "Requested query" for control in observed.controls)
        usage = copy.deepcopy(instance.usage_totals)
        provenance = copy.deepcopy(instance.last_request_provenance)
        now = query_panel(observed)
        activated = instance.activate_program_stage(wrapped.program, 1, now, remaining_actions=5)
        assert len(instance.parsed_stage_frames) == 2
        assert instance.parsed_stage_frames[1][1] is now
        assert instance.parsed_stage_frames[1][2] == 6
        assert activated == parse_original(wrapped.program.stages[1].parser_object(), now, 6)
        assert activated.steps[0].target.name == "Requested query"
        assert len(requests) == instance.calls == 1
        assert instance.usage_totals == usage and instance.last_request_provenance == provenance


@pytest.mark.parametrize("damage", ["missing", "wrong_app", "duplicate", "incomplete"])
def test_program_priority_keeps_original_future_activation_validation(
    observed, damage,
):
    wrapped, _, instance = requested(observed, declaration(), PROGRAM_TOOL)
    now = query_panel(observed)
    if damage == "missing":
        now = replace(now, controls=now.controls[:1])
    elif damage == "wrong_app":
        now = replace(now, controls=(now.controls[0], replace(now.controls[-1], app="Elsewhere")))
    elif damage == "duplicate":
        now = replace(now, controls=(*now.controls, replace(now.controls[-1], ref="ambiguous-new-query")))
    else:
        now = replace(now, a11y_complete=False)
    with pytest.raises(ValueError):
        instance.activate_program_stage(wrapped.program, 1, now, remaining_actions=5)
    assert instance.calls == 1  # Validation cannot create a replacement model request.


@pytest.mark.parametrize("damage", ["cross_target", "false_guard"])
def test_program_priority_keeps_original_atomic_target_and_current_state_guards(observed, damage):
    raw = current(6, "declare_semantic_current_action")
    if damage == "cross_target":
        raw["postconditions"][0]["target"] = {
            "name": "Other", "role": "button", "app": "Editor", "window": "Main",
        }
    else:
        raw["preconditions"] = [{"kind": "value_equals", "target": FIELD, "value": "not current"}]
    validations = []
    for fmt in (ANCESTOR, FORMAT):
        requests = []

        def answer(request, *, collected=requests):
            collected.append(request)
            return httpx.Response(200, json=packet(raw, "declare_semantic_current_action"))

        with httpx.Client(transport=httpx.MockTransport(answer)) as client:
            instance = make_planner(client, fmt=fmt)
            with pytest.raises(ValueError, match="invalid multimodal planning response"):
                instance.plan(GOAL, observed, [], max_actions=6)
        assert len(requests) == instance.calls == 1
        validations.append(instance.last_validation_error)
    assert validations[0] == validations[1] and validations[0]
