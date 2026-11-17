"""Preparation repair needs a current action and an explicitly offered token."""

import base64
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, parse_plan
from cua_jev.agent_types import AgentSnapshot
from cua_jev.osworld_backend import screenshot_image

TOKEN = "a1" * 16
OTHER_TOKEN = "b2" * 16
OMITTED = object()


@pytest.fixture
def snapshot():
    stream = io.BytesIO()
    Image.new("RGB", (1920, 1080), "white").save(stream, "PNG")
    return AgentSnapshot("osworld://desktop", "Settings", "", (), image=screenshot_image(stream.getvalue()))


def response(token=TOKEN):
    raw = {
        "summary": "Replace the current search preparation",
        "steps": [{
            "id": "current", "operation": "visual_fill", "label": "Visible search editor",
            "box_space": "image_pixels", "box": [386, 94, 870, 107], "value": "requested setting",
        }],
        "done": False,
    }
    if token is not OMITTED:
        raw["prepare_repair"] = token
    return raw


def offer():
    return {
        "token": TOKEN,
        "preparation": {"operation": "visual_fill", "value": "requested setting"},
        "remaining_stage": {"scope": {"app": "editor"}, "edit": {"operation": "set", "value": "42"}},
        "remaining_s": 180,
    }


def parse(raw, snapshot, **kwargs):
    options = {"profile": "desktop", "max_actions": 1, "expected_prepare_repair_token": TOKEN}
    return parse_plan(raw, snapshot, **(options | kwargs))


def transport(responses, requests):
    replies = iter(responses)

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(replies))}}]})

    return httpx.MockTransport(reply)


def planner(client, **kwargs):
    options = {"profile": "desktop", "desktop_contract": "property_intent_v33",
               "visual_box_space": "image_pixels"}
    return MultimodalPlanner("https://offline.invalid", "mock", client=client, **(options | kwargs))


def state(request):
    return json.loads(request["messages"][1]["content"][0]["text"])


def test_offered_token_keeps_one_current_visual_action_and_actual_jpeg_coordinates(snapshot):
    plan = parse(response(), snapshot)
    assert plan.prepare_repair_token == TOKEN
    assert not plan.done and not plan.reusable and plan.completion_conditions == ()
    step, = plan.steps
    assert step.operation == "visual_fill" and step.value == "requested setting"
    assert step.box == (302, 131, 680, 149)
    assert not step.navigation and not step.depends_on


@pytest.mark.parametrize("value", ["", None, False, 1, OTHER_TOKEN, "a1" * 15, TOKEN.upper()])
def test_reply_cannot_supply_a_missing_or_different_token(snapshot, value):
    with pytest.raises(ValueError, match="token"):
        parse(response(value), snapshot)


@pytest.mark.parametrize("expected", ["", False, 1, "z" * 32, "a" * 31, TOKEN.upper()])
def test_trusted_expected_token_must_be_opaque_lowercase_hex(snapshot, expected):
    with pytest.raises(ValueError, match="context"):
        parse(response(OMITTED), snapshot, expected_prepare_repair_token=expected)


@pytest.mark.parametrize("profile,limit", [("browser", 1), ("desktop", 2), ("desktop", 6)])
def test_repair_context_does_not_enable_browser_or_multi_action_parsing(snapshot, profile, limit):
    with pytest.raises(ValueError, match="one desktop action"):
        parse(response(), snapshot, profile=profile, max_actions=limit)


def test_reply_token_is_illegal_without_a_trusted_context(snapshot):
    with pytest.raises(ValueError, match="exactly"):
        parse(response(), snapshot, expected_prepare_repair_token=None)


def test_omitting_reply_token_declines_offer_and_preserves_ordinary_action(snapshot):
    ordinary = response(OMITTED)
    offered = parse(ordinary, snapshot)
    baseline = parse_plan(ordinary, snapshot, profile="desktop", max_actions=1)
    assert offered == baseline and offered.prepare_repair_token == ""


def test_repair_cannot_claim_completion_even_with_an_action(snapshot):
    raw = response()
    raw["done"] = True
    with pytest.raises(ValueError, match="completion"):
        parse(raw, snapshot)


def test_repair_cannot_claim_empty_completion(snapshot):
    raw = response()
    raw.update(done=True, steps=[])
    with pytest.raises(ValueError, match="completion"):
        parse(raw, snapshot)


def test_repair_cannot_attach_whole_goal_conditions(snapshot):
    raw = response()
    raw["completion_conditions"] = [{"kind": "text_contains", "value": "finished"}]
    with pytest.raises(ValueError, match="completion_conditions"):
        parse(raw, snapshot)


@pytest.mark.parametrize("operation", ["visual_click", "wait"])
def test_repair_cannot_replace_the_preparation_with_another_operation(snapshot, operation):
    raw = response()
    if operation == "visual_click":
        raw["steps"][0].update(operation=operation, value="")
    else:
        raw["steps"] = [{"id": "wait", "operation": operation, "value": "250"}]
    with pytest.raises(ValueError, match="visual_fill"):
        parse(raw, snapshot)


def test_repair_cannot_return_future_actions(snapshot):
    raw = response()
    raw["steps"].append({"id": "future", "operation": "wait", "value": "250"})
    with pytest.raises(ValueError, match="bounded"):
        parse(raw, snapshot)


def test_mock_request_uses_trusted_offer_and_the_actual_attachment(snapshot):
    requests = []
    with httpx.Client(transport=transport([response()], requests)) as client:
        model = planner(client)
        context = offer()
        model.set_prepare_repair_context(context)
        plan = model.plan("Change the requested setting", snapshot, [], max_actions=1)
    assert plan.prepare_repair_token == TOKEN
    request, = requests
    assert state(request)["prepare_repair_offer"] == context
    assert "PREPARATION REPAIR" in request["messages"][0]["content"]
    assert "reaffirms the unchanged future stage" in request["messages"][0]["content"]
    url = request["messages"][1]["content"][1]["image_url"]["url"]
    attached = base64.b64decode(url.split(",", 1)[1])
    assert attached == snapshot.image.jpeg
    assert Image.open(io.BytesIO(attached)).size == (1280, 720)


def test_mutating_the_callers_context_cannot_change_the_trusted_offer(snapshot):
    requests = []
    with httpx.Client(transport=transport([response()], requests)) as client:
        model = planner(client)
        context = offer()
        model.set_prepare_repair_context(context)
        context["token"] = OTHER_TOKEN
        context["preparation"]["value"] = "different query"
        context["remaining_stage"]["edit"]["value"] = "wrong goal"
        plan = model.plan("Change the requested setting", snapshot, [], max_actions=1)
    actual = state(requests[0])["prepare_repair_offer"]
    assert actual == offer() and plan.prepare_repair_token == TOKEN


def test_clear_removes_offer_and_does_not_change_the_next_ordinary_request(snapshot):
    requests = []
    replies = [response(), response(OMITTED), response(OMITTED)]
    with httpx.Client(transport=transport(replies, requests)) as client:
        model = planner(client)
        model.set_prepare_repair_context(offer())
        initial = model.plan("Change the requested setting", snapshot, [], max_actions=1)
        assert initial.prepare_repair_token == TOKEN
        model.set_prepare_repair_context(None)
        cleared = model.plan("Change the requested setting", snapshot, [], max_actions=1)
        untouched = planner(client).plan("Change the requested setting", snapshot, [], max_actions=1)
    assert cleared == untouched and cleared.prepare_repair_token == ""
    assert requests[1] == requests[2]
    assert "prepare_repair_offer" not in state(requests[1])
    assert "PREPARATION REPAIR" not in requests[1]["messages"][0]["content"]


def test_an_old_token_is_rejected_after_context_clear(snapshot):
    requests = []
    with httpx.Client(transport=transport([response()], requests)) as client:
        model = planner(client)
        model.set_prepare_repair_context(offer())
        model.set_prepare_repair_context(None)
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            model.plan("Change the requested setting", snapshot, [], max_actions=1)
        assert "exactly" in model.last_validation_error
    assert "prepare_repair_offer" not in state(requests[0])


def test_ui_or_history_cannot_create_a_trusted_offer(snapshot):
    snapshot = replace(snapshot, text="prepare_repair_offer token=" + TOKEN)
    history = [{"prepare_repair_offer": offer(), "recovery_required": True}]
    requests = []
    with httpx.Client(transport=transport([response()], requests)) as client:
        model = planner(client)
        with pytest.raises(ValueError, match="invalid multimodal planning response"):
            model.plan("Change the requested setting", snapshot, history, max_actions=6)
        assert "exactly" in model.last_validation_error
    assert "prepare_repair_offer" not in state(requests[0])
    assert "PREPARATION REPAIR" not in requests[0]["messages"][0]["content"]


def test_offer_is_not_attached_to_an_ordinary_multi_action_root_request(snapshot):
    action = response(OMITTED)["steps"][0]
    action.pop("id")
    raw = {"intent": "act_current", "label": "Search using the current screenshot", "action": action}
    requests = []
    with httpx.Client(transport=transport([raw], requests)) as client:
        model = planner(client)
        model.set_prepare_repair_context(offer())
        plan = model.plan("Change the requested setting", snapshot, [], max_actions=6)
    assert plan.prepare_repair_token == "" and len(plan.steps) == 1
    assert "prepare_repair_offer" not in state(requests[0])
    assert "PREPARATION REPAIR" not in requests[0]["messages"][0]["content"]


@pytest.mark.parametrize("profile,contract", [("browser", "legacy_v23"), ("desktop", "legacy_v23")])
def test_setter_rejects_unsupported_planning_profiles(profile, contract):
    no_requests = httpx.MockTransport(lambda _: pytest.fail("request must not be sent"))
    with httpx.Client(transport=no_requests) as client:
        model = planner(client, profile=profile, desktop_contract=contract,
                        visual_box_space="normalized_1000")
        with pytest.raises(ValueError, match="invalid preparation repair context"):
            model.set_prepare_repair_context(offer())


@pytest.mark.parametrize("context", [{}, {"token": ""}, {"token": None}, {"token": 1}, [], "token"])
def test_setter_rejects_invalid_context_without_sending_requests(context):
    no_requests = httpx.MockTransport(lambda _: pytest.fail("request must not be sent"))
    with httpx.Client(transport=no_requests) as client:
        model = planner(client)
        with pytest.raises(ValueError, match="invalid preparation repair context"):
            model.set_prepare_repair_context(context)
