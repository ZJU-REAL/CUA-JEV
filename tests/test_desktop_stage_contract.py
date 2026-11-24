import json

import httpx
import pytest

from cua_jev.agent_planner import MultimodalPlanner
from cua_jev.agent_types import AgentControl, AgentSnapshot


@pytest.fixture
def snapshot():
    return AgentSnapshot(
        "osworld://desktop",
        "Editor",
        "",
        (
            AgentControl(
                "one",
                "Alias",
                "input",
                role="textbox",
                editable=True,
                input_type="text",
                app="Editor",
                window="Properties",
                focused=True,
                box=(10, 10, 100, 40),
            ),
        ),
    )


def call(raw, snapshot, *, contract="stage_intent_v24", limit=6, history=None):
    requests = []

    def answer(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})

    client = httpx.Client(transport=httpx.MockTransport(answer))
    planner = MultimodalPlanner(
        "https://unused.invalid",
        "same-model",
        profile="desktop",
        client=client,
        enable_thinking=False,
        desktop_contract=contract,
    )
    try:
        result = planner.plan("Set the requested property", snapshot, history or [], max_actions=limit)
        return result, requests, planner
    finally:
        client.close()


def immediate():
    return {
        "summary": "Set alias",
        "done": False,
        "steps": [
            {
                "id": "s1",
                "operation": "fill",
                "target": {"name": "Alias", "app": "Editor"},
                "value": "new",
                "postconditions": [
                    {"kind": "value_equals", "target": {"name": "Alias", "app": "Editor"}, "value": "new"}
                ],
            }
        ],
    }


def test_new_contract_requests_explicit_stage_and_uses_existing_compiler(snapshot):
    raw = {
        "intent": "navigate_edit",
        "label": "Set future property",
        "scope": {"app": "Editor"},
        "edit": {"operation": "fill", "target": {"name": "Later", "app": "Editor"}, "value": "new"},
    }
    result, requests, planner = call(raw, snapshot)
    assert len(result.steps) == 2 and result.steps[0].native_navigation_gestures
    assert planner.calls == len(requests) == 1
    body = requests[0]
    assert "explicit intent" in body["messages"][0]["content"]
    assert "CURRENT ANONYMOUS FIELDS" in body["messages"][0]["content"]
    assert "PENDING NUMERIC EDITS" in body["messages"][0]["content"]
    assert "response_format" not in body and body["enable_thinking"] is False


@pytest.mark.parametrize("history,limit", [([], 1), ([{"recovery_required": True}], 6)])
def test_single_action_and_recovery_request_bodies_remain_identical(history, limit, snapshot):
    old, old_requests, _ = call(immediate(), snapshot, contract="legacy_v23", history=history, limit=limit)
    new, new_requests, _ = call(immediate(), snapshot, history=history, limit=limit)
    assert new_requests == old_requests and new == old


@pytest.mark.parametrize(
    "raw",
    [
        immediate(),
        {
            "intent": "reach_edit",
            "label": "Old intent",
            "scope": {"app": "Editor"},
            "edit": {"operation": "fill", "target": {"name": "Later", "app": "Editor"}, "value": "new"},
        },
        [{"intent": "complete"}],
        {"intent": []},
    ],
)
def test_normal_stage_contract_rejects_legacy_scripts_or_mixed_shapes_without_retry(raw, snapshot):
    with pytest.raises(ValueError, match="invalid multimodal"):
        call(raw, snapshot)


@pytest.mark.parametrize(
    "contract,profile",
    [([], "desktop"), ({}, "desktop"), ("unknown", "desktop"), ("stage_intent_v24", "browser")],
)
def test_invalid_or_browser_contract_does_not_open_a_model_client(contract, profile):
    with pytest.raises(ValueError, match="desktop planning contract"):
        MultimodalPlanner("https://unused.invalid", "model", profile=profile, desktop_contract=contract)
