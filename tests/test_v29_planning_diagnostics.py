"""V29 response framing and public numeric facts never invent input fields."""

import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _response_object
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.desktop_semantics import numeric_editor_status
from cua_jev.osworld_backend import screenshot_image


def number(value="100.0", buffer="100"):
    return AgentControl("field", "History levels:", "input", role="spinbutton", input_type="number",
                        value=value, edit_buffer=buffer, editable=True, focused=True,
                        app="Editor", window="Preferences", box=(100, 100, 300, 200))


@pytest.fixture
def observed():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buffer, "PNG")
    return AgentSnapshot("osworld://desktop", "Preferences", "", (number(),),
                         image=screenshot_image(buffer.getvalue()))


@pytest.mark.parametrize("value,buffer,status", [
    ("100.0", "100", "committed"), ("100", "+100.00", "committed"), ("5.0", "100", "pending"),
    ("100", "50", "pending"), ("NaN", "NaN", "unknown"), ("Infinity", "Infinity", "unknown"),
    ("100", "100px", "unknown"), ("1e9999999", "1e9999999", "unknown"),
])
def test_numeric_status_uses_bounded_committed_numbers_instead_of_string_format_or_buffer_presence(
    value, buffer, status,
):
    assert numeric_editor_status(number(value, buffer)) == status


@pytest.mark.parametrize("changes", [
    {"role": "textbox"}, {"editable": False}, {"focused": False}, {"disabled": True},
    {"input_type": "password"}, {"input_type": "file"}, {"box": None}, {"edit_buffer": None},
])
def test_numeric_status_omits_private_non_numeric_and_unsupported_controls(changes):
    assert numeric_editor_status(replace(number(), **changes)) is None


def test_opt_in_extracts_exactly_one_json_fence_without_using_surrounding_explanation():
    text = 'The view is open.\n\n```json\n{"intent":"complete","label":"visible result"}\n```\n'
    assert _response_object(text, allow_fenced_explanation=True) == {
        "intent": "complete", "label": "visible result",
    }
    with pytest.raises(ValueError):
        _response_object(text)


@pytest.mark.parametrize("text", [
    'First {"intent":"complete"}.\n```json\n{"intent":"navigate"}\n```',
    'First ["different choice"].\n```json\n{"intent":"navigate"}\n```',
    '```json\n{"intent":"complete"}\n```\n```json\n{"intent":"navigate"}\n```',
    'Explanation\n```python\n{"intent":"complete"}\n```',
    'Explanation\n```json\n{"intent":"complete","intent":"navigate"}\n```',
    'Explanation\n```json\n{"intent":"complete"}\n```\n```other',
])
def test_ambiguous_multiple_non_json_or_duplicate_object_frames_remain_rejected(text):
    with pytest.raises(ValueError):
        _response_object(text, allow_fenced_explanation=True)


def request(snapshot, *, contract="property_intent_v29", limit=6, history=(), response=None):
    calls = []
    response = response or '{"summary":"done","steps":[],"done":true}'

    def answer(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": response}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner("https://offline.invalid/v1", "mock", client=client,
                                    profile="desktop", desktop_contract=contract)
        try:
            plan = planner.plan("Set history levels to100", snapshot, list(history), max_actions=limit)
        except ValueError:
            plan = None
    return plan, calls[0]


def test_numeric_status_is_diagnostic_and_keeps_immediate_and_recovery_requests(observed):
    done = '{"intent":"complete","label":"complete"}'
    _plan, body = request(observed, response=done)
    state = json.loads(body["messages"][1]["content"][0]["text"])
    assert state["numeric_editor_states"] == [{"ref": "field", "status": "committed"}]
    for limit, history in ((1, []), (6, [{"recovery_required": True}])):
        bodies = [request(observed, contract=c, limit=limit, history=history)[1]
                  for c in ("legacy_v23", "stage_intent_v24", "property_intent_v25",
                            "property_intent_v28", "property_intent_v29", "property_intent_v30",
                            "property_intent_v31")]
        assert all(b == bodies[0] for b in bodies)
    _plan, partial = request(replace(observed, a11y_complete=False), response=done)
    assert "numeric_editor_states" not in json.loads(partial["messages"][1]["content"][0]["text"])


def test_v29_accepts_a_unique_framed_response_but_never_fills_in_a_missing_keyboard_key(observed):
    complete = 'Visible result.\n```json\n{"intent":"complete","label":"result"}\n```'
    assert request(observed, response=complete)[0].done
    assert request(observed, contract="property_intent_v28", response=complete)[0] is None
    missing = {"intent": "act_current", "label": "Finalize field", "action": {
        "operation": "key_press", "preconditions": [{"kind": "focused", "target": {
            "name": "History levels:", "app": "Editor",
        }}], "postconditions": [{"kind": "value_equals", "target": {
            "name": "History levels:", "app": "Editor",
        }, "value": "100"}],
    }}
    assert request(observed, response="Explanation\n```json\n" + json.dumps(missing) + "\n```")[0] is None
