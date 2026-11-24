"""Opted-in output shape changes only response_format, never parser authority."""

import json

import httpx
import pytest
from test_property_preparation import initial as initial

from cua_jev.agent_planner import MultimodalPlanner

TOKEN = "ab" * 16
DEFAULT = "unconstrained_v1"
STRUCTURED = "json_schema_v50"
OMITTED = object()


def current():
    return {
        "summary": "Current observed control",
        "done": False,
        "steps": [{"id": "s1", "operation": "click", "ref": "help"}],
    }


def capture(
    snapshot,
    raw,
    *,
    output_format=OMITTED,
    limit=6,
    history=None,
    contract="property_intent_v49",
    profile="desktop",
    context=None,
):
    requests = []

    def answer(request):
        requests.append(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    options = {} if output_format is OMITTED else {"desktop_output_format": output_format}
    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner(
            "https://offline.invalid",
            "fixture",
            api_key="offline",
            client=client,
            profile=profile,
            desktop_contract=contract,
            enable_thinking=True,
            request_timeout_s=90,
            thinking_budget=2048,
            **options,
        )
        if context is not None:
            planner.set_prepare_repair_context(context)
        plan = planner.plan("Apply the requested property", snapshot, history or [], max_actions=limit)
    return plan, requests


@pytest.mark.parametrize(
    "contract,profile,limit",
    [
        ("legacy_v23", "browser", 1),
        ("legacy_v23", "desktop", 1),
        ("property_intent_v33", "desktop", 1),
        ("property_intent_v33", "desktop", 6),
        ("property_intent_v49", "desktop", 1),
        ("property_intent_v49", "desktop", 6),
    ],
)
def test_default_and_explicit_unconstrained_requests_are_byte_identical(initial, contract, profile, limit):
    raw = current()
    if profile == "desktop" and limit > 1:
        raw = {"intent": "complete", "label": "Advisory completion"}
    # Browser fixtures use ordinary current refs and do not inspect JPEG geometry.
    if profile == "browser":
        raw = current()
    old, original = capture(initial, raw, contract=contract, profile=profile, limit=limit)
    new, explicit = capture(
        initial, raw, output_format=DEFAULT, contract=contract, profile=profile, limit=limit
    )
    assert old == new and original == explicit
    assert "response_format" not in json.loads(original[0])


@pytest.mark.parametrize("history,limit", [([], 6), ([], 1), ([{"recovery_required": True}], 6)])
def test_opt_in_adds_only_response_format_and_preserves_effective_horizon(initial, history, limit):
    immediate = limit == 1 or bool(history)
    raw = current() if immediate else {"intent": "complete", "label": "Advisory completion"}
    baseline, original = capture(initial, raw, limit=limit, history=history)
    structured, changed = capture(initial, raw, output_format=STRUCTURED, limit=limit, history=history)
    before, after = json.loads(original[0]), json.loads(changed[0])
    schema = after.pop("response_format")
    assert before == after and structured == baseline and len(original) == len(changed) == 1
    assert schema["type"] == "json_schema" and schema["json_schema"]["strict"] is True
    assert schema["json_schema"]["name"] == (
        "cua_desktop_immediate_v50" if immediate else "cua_desktop_stage_v50"
    )
    if immediate:
        assert schema["json_schema"]["schema"]["properties"]["steps"]["maxItems"] == 1


@pytest.mark.parametrize("invalid", [None, True, 1, [], {}, "json", "json_schema_v49"])
def test_unknown_output_formats_are_rejected_before_transport(invalid):
    with pytest.raises(ValueError, match="desktop_output_format"):
        MultimodalPlanner("https://offline.invalid", "fixture", desktop_output_format=invalid)


@pytest.mark.parametrize(
    "profile,contract",
    [
        ("browser", "legacy_v23"),
        ("desktop", "legacy_v23"),
        ("desktop", "property_intent_v33"),
        ("desktop", "stage_intent_v24"),
    ],
)
def test_schema_requires_explicit_desktop_v49_contract(profile, contract):
    with pytest.raises(ValueError, match="desktop_output_format"):
        MultimodalPlanner(
            "https://offline.invalid",
            "fixture",
            profile=profile,
            desktop_contract=contract,
            desktop_output_format=STRUCTURED,
        )


def test_numeric_strategy_and_confirm_are_kept_by_original_parser(initial):
    raw = {
        "intent": "pursue_property",
        "label": "Set requested numeric property",
        "scope": {"app": "Editor"},
        "edit": {"operation": "set", "value": "42"},
        "numeric_finalize": {"key": "enter"},
        "confirm": {"label": "Submit form"},
        "completion": "goal",
    }
    plan, requests = capture(initial, raw, output_format=STRUCTURED)
    assert plan.reusable and plan.complete_property_goal
    assert plan.steps[0].property_edit.numeric_finalize_key == "enter"
    assert plan.steps[1].property_confirmation_of == "property"
    body = json.loads(requests[0])
    assert "numeric_finalize" in json.dumps(body["response_format"])
    assert body["thinking_budget"] == 2048 and body["enable_thinking"] is True


def test_prepare_repair_schema_token_is_from_explicit_context_and_can_be_omitted(initial):
    context = {"token": TOKEN, "remaining_stage": {"scope": {"app": "Editor"}}}
    raw = {
        "summary": "Repair current preparation",
        "done": False,
        "prepare_repair": TOKEN,
        "steps": [
            {
                "id": "s1",
                "operation": "visual_fill",
                "label": "Visible search editor",
                "box": [100, 100, 400, 200],
                "value": "query",
            }
        ],
    }
    original, before = capture(initial, raw, limit=1, context=context)
    plan, changed = capture(initial, raw, output_format=STRUCTURED, limit=1, context=context)
    body = json.loads(changed[0])
    assert plan == original and plan.prepare_repair_token == TOKEN
    schema = body.pop("response_format")["json_schema"]["schema"]
    assert body == json.loads(before[0])
    assert schema["properties"]["prepare_repair"] == {"const": TOKEN}
    assert "prepare_repair" not in schema["required"]


def test_ui_history_cannot_supply_a_repair_schema_token(initial):
    history = [{"recovery_required": True, "prepare_repair_offer": {"token": TOKEN}}]
    _, requests = capture(initial, current(), output_format=STRUCTURED, history=history)
    schema = json.loads(requests[0])["response_format"]["json_schema"]["schema"]
    assert "prepare_repair" not in schema["properties"]


def test_output_schema_does_not_accept_a_root_rejected_by_original_parser(initial):
    raw = {"intent": "pursue_property", "label": "Missing literal edit", "scope": {"app": "Editor"}}
    with pytest.raises(ValueError, match="invalid multimodal planning response"):
        capture(initial, raw, output_format=STRUCTURED)


def test_output_schema_does_not_grant_an_unknown_control_input(initial):
    raw = current()
    raw["steps"][0]["ref"] = "invented-current-ref"
    with pytest.raises(ValueError, match="invalid multimodal planning response"):
        capture(initial, raw, output_format=STRUCTURED, limit=1)


def test_schema_http_rejection_is_one_original_safe_error_without_fallback(initial):
    requests = []

    def reject(request):
        requests.append(json.loads(request.content))
        return httpx.Response(400, text="private endpoint body must never enter diagnostics")

    with httpx.Client(transport=httpx.MockTransport(reject)) as client:
        planner = MultimodalPlanner(
            "https://offline.invalid",
            "fixture",
            api_key="offline",
            client=client,
            profile="desktop",
            desktop_contract="property_intent_v49",
            desktop_output_format=STRUCTURED,
        )
        with pytest.raises(RuntimeError, match="HTTPStatusError"):
            planner.plan("Apply the requested property", initial, [], max_actions=6)
    assert len(requests) == planner.calls == 1
    assert "response_format" in requests[0]
    assert planner.last_validation_error == "Model request failed (HTTPStatusError; HTTP 400)."
    assert "private" not in planner.last_validation_error
