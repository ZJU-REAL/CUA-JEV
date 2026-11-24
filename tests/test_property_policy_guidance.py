"""Property guidance does not narrow candidates or treat matching values as semantic success."""

import json

import httpx
import pytest

from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.policy import JevPolicy


def choose(state, layout="shared_fields_v2"):
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(200, json={"answers": {"action": {
            "choice": "one", "probabilities": {"one": 1.0}, "confidence": 1.0,
        }}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="offline", request_layout=layout, client=client)
        decision = policy.choose(Observation("goal", "stage", state), [
            ActionCandidate("one", Channel.GUI, "click", "Click the current control"),
        ])
    assert decision.candidate_id == "one"
    return calls[0]


@pytest.mark.parametrize("layout", ["full_v1", "deduplicated_v1", "shared_fields_v2"])
@pytest.mark.parametrize("operation", ["fill", "check", "uncheck"])
def test_property_goal_marks_only_selection_semantics_and_keeps_original_option_id(layout, operation):
    body = choose({"ready_steps": [{"property_goal": {"operation": operation, "has_value": True}}]}, layout)
    question = body["questions"]["action"]
    assert list(question["criteria"]) == ["one"]
    assert "For that SAME requested field" in question["instructions"]
    assert "prerequisites or navigation" in question["instructions"]
    assert "not merely to a field" in question["instructions"]


@pytest.mark.parametrize("state", [
    None, [], "opaque", {}, {"ready_steps": None}, {"ready_steps": "opaque"},
    {"ready_steps": [None, {}, {"property_goal": None}]},
    {"ready_steps": [{"property_goal": {"operation": []}}]},
    {"ready_steps": [{"property_goal": {"operation": True}}]},
    {"ready_steps": [{"property_goal": {"operation": "SCRIPT"}}]},
])
def test_legacy_or_untyped_public_data_does_not_change_question_guidance(state):
    body = choose(state, "full_v1")
    assert "This is a property stage" not in body["questions"]["action"]["instructions"]
    assert body["state"]["state"] == state
