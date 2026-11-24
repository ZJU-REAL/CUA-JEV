import copy
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, parse_plan
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.vision import WindowImage


@pytest.fixture
def case():
    controls = tuple(AgentControl(f"e{i}", f"Option {i}", "input", input_type="checkbox")
                     for i in range(8)) + (AgentControl("submit", "Submit", "button"),)
    image = io.BytesIO()
    Image.new("RGB", (64, 64)).save(image, format="JPEG")
    snapshot = AgentSnapshot("https://local.test/", "Form", "Form", controls,
                             image=WindowImage(image.getvalue(), bytes(64 * 64), (0, 0, 64, 64)))
    steps = [{"id": f"s{i}", "operation": "check", "ref": f"e{i}"} for i in range(8)]
    steps.append({"id": "submit", "operation": "click", "ref": "submit",
                  "depends_on": [s["id"] for s in steps]})
    return snapshot, {"summary": "Check required options and submit", "steps": steps, "done": False}


def test_validated_prefix_defers_submit_and_does_not_mutate_original(case):
    snapshot, raw = case
    original = copy.deepcopy(raw)
    result = parse_plan(raw, snapshot, allow_browser_edit_prefix=True)
    assert len(result.steps) == 6 and not result.done
    assert all(s.operation == "check" for s in result.steps)
    assert raw == original
    with pytest.raises(ValueError):
        parse_plan(raw, snapshot)
    with pytest.raises(ValueError):
        parse_plan(raw, snapshot, max_actions=1, allow_browser_edit_prefix=True)


@pytest.mark.parametrize("bad", ["unknown_target", "future_dependency", "extra_field", "unsafe_prefix"])
def test_invalid_suffix_or_prefix_cannot_be_hidden_by_segmentation(case, bad):
    snapshot, raw = case
    if bad == "unknown_target":
        raw["steps"][-1]["ref"] = "not-observed"
    elif bad == "future_dependency":
        raw["steps"][0]["depends_on"] = ["s7"]
    elif bad == "extra_field":
        raw["steps"][-1]["code"] = "untrusted"
    else:
        raw["steps"][0] = {"id": "s0", "operation": "click", "ref": "submit"}
    with pytest.raises(ValueError):
        parse_plan(raw, snapshot, allow_browser_edit_prefix=True)


def test_segmentation_has_absolute_validation_cap(case):
    snapshot, raw = case
    raw["steps"] *= 8
    with pytest.raises(ValueError, match="exceeds 64"):
        parse_plan(raw, snapshot, allow_browser_edit_prefix=True)


def test_validation_budget_can_exceed_execution_budget_without_queueing_suffix(case):
    snapshot, raw = case
    snapshot = replace(snapshot, controls=tuple(
        AgentControl(f"e{i}", f"Option {i}", "input", input_type="checkbox") for i in range(63)
    ) + (AgentControl("submit", "Submit", "button"),))
    steps = [{"id": f"s{i}", "operation": "check", "ref": f"e{i}"} for i in range(63)]
    steps.append({"id": "submit", "operation": "click", "ref": "submit",
                  "depends_on": [s["id"] for s in steps]})
    raw["steps"] = steps
    result = parse_plan(raw, snapshot, allow_browser_edit_prefix=True)
    assert len(result.steps) == 6 and all(s.operation == "check" for s in result.steps)
    assert not result.done
    with pytest.raises(ValueError):
        parse_plan(raw, snapshot, max_actions=64)
    raw["steps"][-1]["depends_on"].append("invented")
    with pytest.raises(ValueError, match="depends_on"):
        parse_plan(raw, snapshot, allow_browser_edit_prefix=True)


def test_prefix_parser_rejects_untrusted_validation_tokens_and_nonboolean_flag(case):
    snapshot, raw = case
    with pytest.raises(ValueError, match="validation context"):
        parse_plan(raw, snapshot, _browser_validation_token=True)
    with pytest.raises(ValueError, match="boolean"):
        parse_plan(raw, snapshot, allow_browser_edit_prefix=1)


def test_real_planner_transport_segments_once_and_resets_diagnostics(case):
    snapshot, raw = case
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        reply = raw if len(requests) == 1 else {**raw, "steps": raw["steps"][:1]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    planner = MultimodalPlanner("https://model.test/v1", "vlm", api_key="test",
                               client=httpx.Client(transport=httpx.MockTransport(handler)),
                               browser_edit_prefix=True)
    try:
        result = planner.plan("Check all required options", snapshot, [])
        assert len(result.steps) == 6 and len(requests) == 1
        assert planner.last_plan_segmentation == {
            "received_steps": 9, "retained_steps": 6,
            "entire_response_validated": True, "suffix_queued": False,
        }
        planner.plan("Continue", snapshot, [])
        assert planner.last_plan_segmentation is None
        assert "next executable" in requests[0]["messages"][0]["content"]
    finally:
        planner.close()
