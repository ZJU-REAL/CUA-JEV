"""Regression tests for actual policy v6."""

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError, asdict, fields, replace

import httpx
import pytest

from cua_jev import jev_layout_selection
from cua_jev.jev_argument_encoding import restore_argument_bases, share_argument_bases
from cua_jev.jev_control_description_encoding import restore_public_control_descriptions
from cua_jev.jev_description_encoding import restore_description_fragments, share_description_fragments
from cua_jev.models import ActionCandidate, Channel, Observation, Risk
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.policy import JevEncodingDiagnostics, JevPolicy, JevPublicEncodingDiagnostics, JevRequestError

KEY = "OFFLINE_PRIVATE_KEY"
URL = "https://offline.invalid/v1/systemone"
MODEL = "jev-1.13.0"
OLD_FIELDS = (
    "requested_layout", "actual_layout", "argument_base_applied", "candidate_count",
    "marked_candidate_count", "argument_base_count", "compact_utf8_body_bytes",
    "layout_encoding_wall_ms", "description_fragment_applied",
    "description_marked_candidate_count", "description_fragment_count",
)


def wire(value, *, ascii=False):
    return json.dumps(value, ensure_ascii=ascii, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def anchor(control):
    return (control["label"] + " [" + control["ref"] + "; " + control["role"] + "; "
            + control["app"] + "; " + control["window"] + "; "
            + " / ".join(control["ancestors"]) + "]")


def workload(kind):
    if kind == "v6":
        controls, actions = [], []
        for index in range(60):
            control = {"ref": f"c{index}", "label": f"Observed widget {index}: " + f"item{index} " * 24,
                       "role": "button", "app": "ObservedApp", "window": "Observed frame",
                       "ancestors": ["frame: current observed owner", f"panel: public tools {index}"],
                       "disabled": False, "checked": False}
            controls.append(control)
            for gesture in ("", " (right button, 1 click(s))", " (left button, 2 click(s))"):
                actions.append(ActionCandidate(
                    f"choice-{index}:{len(actions) % 3}", Channel.GUI, "agent.osworld.click",
                    "click " + anchor(control) + gesture
                    + " toward navigation goal: current public goal via OSWorld input",
                    arguments={"private": "PRIVATE_EXECUTOR_ARGUMENT", "ref": control["ref"]},
                    expected={"private": "PRIVATE_EXPECTED_VALUE"}, risk=Risk.LOCAL_WRITE, intent="navigate",
                ))
    elif kind == "v5":
        controls = []  # No anchor: this must retain the useful original v5 option.
        common = "Observed public context and current legal action with exact retained facts. " * 40
        actions = [ActionCandidate(f"choice-{i}", Channel.GUI, "agent.click", common + str(i),
                                   arguments={"private": "PRIVATE_EXECUTOR_ARGUMENT"}) for i in range(9)]
    else:
        controls = []
        actions = [ActionCandidate("only", Channel.CONTROL, "agent.read", "Read")]
    return Observation("Complete the public current goal", "Current public state",
                       {"controls": controls, "text": "PRIVATE_VISUAL_TEXT"}, "jev-vlm-agent"), actions


def response(actions, *, invalid=False):
    chosen = "invented" if invalid else actions[-1].id
    return {"answers": {"action": {"choice": chosen,
            "probabilities": {item.id: float(item.id == chosen) for item in actions}, "confidence": 0.875}}}


def capture(layout, observed, actions):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response(actions))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key=KEY, api_url=URL, model=MODEL, client=client,
                           request_layout=layout, retries=0)
        decision = PublicDecisionPolicy(policy).choose(observed, actions)
    assert len(requests) == 1
    return json.loads(requests[0].content), decision, policy.last_exchange, requests[0]


def restore_selected(body):
    restored = copy.deepcopy(body)
    encoding = restored["state"]["encoding"]
    if encoding["version"] == "shared_fields_v6":
        return restore_public_control_descriptions(restored)
    if "description_fragment_candidates" in encoding:
        restored = restore_description_fragments(restored)
    if "argument_base_candidates" in restored["state"]["encoding"]:
        restored = restore_argument_bases(restored)
    return restored


@pytest.mark.parametrize("kind,actual", [("v6", "shared_fields_v6"),
                                         ("v5", "shared_fields_v5"), ("v4", "shared_fields_v4")])
def test_actual_policy_selects_truthful_layout_once_and_restores_exact_public_v4(kind, actual):
    observed, actions = workload(kind)
    local_before = wire([action.to_dict() for action in actions])
    state_before = wire(observed.to_dict())
    baseline, _, _, baseline_request = capture("shared_fields_v4", observed, actions)
    body, decision, exchange, request = capture("shared_fields_v6", observed, actions)
    diag = exchange["request_encoding"]
    encoding = body["state"]["encoding"]
    assert encoding["version"] == diag["actual_layout"] == actual
    assert (
        diag["requested_layout"] == "shared_fields_v6"
        and diag["version"] == "jev_request_encoding_cost_v2"
    )
    assert wire(restore_selected(body)) == wire(baseline)
    assert list(body["questions"]["action"]["criteria"]) == [action.id for action in actions]
    assert diag["compact_utf8_body_bytes"] == len(wire(body))
    assert diag["byte_metric_is_captured_HTTP_bytes"] is False
    assert diag["argument_base_applied"] == bool(encoding.get("argument_base_candidates"))
    assert diag["marked_candidate_count"] == len(encoding.get("argument_base_candidates", []))
    assert diag["argument_base_count"] == len(encoding.get("argument_bases", {}))
    assert diag["description_fragment_applied"] == bool(encoding.get("description_fragment_candidates"))
    assert diag["description_marked_candidate_count"] == len(
        encoding.get("description_fragment_candidates", [])
    )
    assert diag["description_fragment_count"] == len(encoding.get("description_fragments", []))
    assert diag["control_description_applied"] is (actual == "shared_fields_v6")
    assert diag["control_description_marked_candidate_count"] == len(
        encoding.get("control_description_candidates", [])
    )
    assert diag["control_description_template_count"] == len(
        encoding.get("control_description_templates", [])
    )
    selection = diag["independent_layout_selection"]
    assert "generated_v5_argument_base_applied" in selection
    assert "generated_v5_description_fragment_applied" in selection
    assert "generated_v6_public_control_description_applied" in selection
    assert "v5_description_fragment_applied" not in selection
    assert selection["actual_layout"] == actual
    assert selection["selected_compact_UTF8_bytes"] <= selection["complete_v5_compact_UTF8_bytes"]
    assert selection["selected_compact_UTF8_bytes"] <= selection["complete_v6_compact_UTF8_bytes"]
    assert body["model"] == MODEL and request.url == baseline_request.url
    assert request.headers["Authorization"] == baseline_request.headers["Authorization"] == f"Bearer {KEY}"
    assert decision.candidate_id == actions[-1].id and decision.confidence == 0.875
    assert exchange["request_hash"] == hashlib.sha256(wire(body, ascii=True)).hexdigest()
    assert "PRIVATE_" not in wire(body).decode()
    assert wire([action.to_dict() for action in actions]) == local_before
    assert wire(observed.to_dict()) == state_before


def test_selector_receives_actual_v4_body_before_any_http(monkeypatch):
    observed, actions = workload("v6")
    baseline, _, _, _ = capture("shared_fields_v4", observed, actions)
    original_selector = jev_layout_selection.select_public_request_encoding
    requests, bases = [], []

    def inspect_base(body, **callbacks):
        assert not requests
        assert body["state"]["encoding"]["version"] == "shared_fields_v4"
        assert "available_actions" not in body["state"]
        bases.append(copy.deepcopy(body))
        return original_selector(body, **callbacks)

    monkeypatch.setattr(jev_layout_selection, "select_public_request_encoding", inspect_base)
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=response(actions))
    )) as client:
        policy = JevPolicy(api_key=KEY, api_url=URL, model=MODEL, client=client,
                           request_layout="shared_fields_v6", retries=0)
        PublicDecisionPolicy(policy).choose(observed, actions)
    assert len(requests) == len(bases) == 1 and wire(bases[0]) == wire(baseline)


def test_requested_v5_stays_on_original_body_and_dataclass_asdict_shape(monkeypatch):
    observed, actions = workload("v5")
    expected, _, _, _ = capture("shared_fields_v4", observed, actions)
    share_argument_bases(expected)
    share_description_fragments(expected)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Old requested-v5 entered the new selector")

    monkeypatch.setattr(jev_layout_selection, "select_public_request_encoding", forbidden)
    body, _, exchange, _ = capture("shared_fields_v5", observed, actions)
    assert wire(body) == wire(expected)
    assert exchange["request_encoding"]["version"] == "jev_request_encoding_cost_v1"
    assert "control_description_applied" not in exchange["request_encoding"]
    assert "independent_layout_selection" not in exchange["request_encoding"]
    assert tuple(field.name for field in fields(JevEncodingDiagnostics)) == OLD_FIELDS
    original = JevEncodingDiagnostics("shared_fields_v5", "shared_fields_v4", False,
                                      1, 0, 0, 100, 1.0)
    assert tuple(asdict(original)) == OLD_FIELDS


def test_success_then_capacity_error_uses_second_call_diag_and_no_capacity_retry():
    first, big = workload("v6")
    second, small = workload("v4")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return (httpx.Response(200, json=response(big)) if len(requests) == 1 else
                httpx.Response(400, json={"error": {"code": "max_tokens_exceeded",
                                                     "message": "PRIVATE_PROVIDER_TEXT"}}))

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key=KEY, api_url=URL, model=MODEL, client=client,
                           request_layout="shared_fields_v6", retries=3)
        public = PublicDecisionPolicy(policy)
        public.choose(first, big)
        first_diag = copy.deepcopy(policy.last_exchange["request_encoding"])
        with pytest.raises(JevRequestError) as caught:
            public.choose(second, small)
    failure = caught.value.failure
    assert len(requests) == 2 and failure.attempts == 1
    assert failure.observation_id == second.observation_id != first.observation_id
    assert failure.request_hash == hashlib.sha256(wire(requests[-1], ascii=True)).hexdigest()
    assert failure.code == "http_status" and failure.http_status == 400
    assert failure.service_code == "max_tokens_exceeded" and failure.retryable is False
    assert failure.network_retryable is False
    diag = failure.to_dict()["request_encoding"]
    assert diag == policy.last_exchange["request_encoding"] == policy.last_failure["request_encoding"]
    assert first_diag["candidate_count"] == len(big) and diag["candidate_count"] == 1
    assert diag["requested_layout"] == "shared_fields_v6" and diag["actual_layout"] == "shared_fields_v4"
    assert diag["control_description_applied"] is False
    assert diag["description_fragment_applied"] is False and diag["argument_base_applied"] is False
    assert isinstance(failure.request_encoding, JevPublicEncodingDiagnostics)
    with pytest.raises(FrozenInstanceError):
        failure.request_encoding.control_description_template_count = 99
    diag["independent_layout_selection"]["actual_layout"] = "MUTATED_COPY"
    assert (
        failure.to_dict()["request_encoding"]["independent_layout_selection"]["actual_layout"]
        == "shared_fields_v4"
    )
    assert "PRIVATE_" not in wire(failure.to_dict()).decode()


def test_invalid_model_choice_keeps_current_v6_diagnostics_without_second_http():
    observed, actions = workload("v6")
    requests = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=response(actions, invalid=True))
    )) as client:
        policy = JevPolicy(api_key=KEY, api_url=URL, model=MODEL, client=client,
                           request_layout="shared_fields_v6", retries=3)
        with pytest.raises(JevRequestError) as caught:
            PublicDecisionPolicy(policy).choose(observed, actions)
    failure = caught.value.failure
    assert len(requests) == failure.attempts == 1 and failure.code == "invalid_response"
    assert failure.retryable is False
    assert failure.to_dict()["request_encoding"] == policy.last_failure["request_encoding"]
    assert failure.to_dict()["request_encoding"] == policy.last_exchange["request_encoding"]


def test_utf8_selection_hash_and_original_ascii_request_hash_keep_their_serializations():
    observed, actions = workload("v6")
    observed = replace(observed, task=observed.task + " 中文目标")
    body, _, exchange, _ = capture("shared_fields_v6", observed, actions)
    selection = exchange["request_encoding"]["independent_layout_selection"]
    assert selection["selected_sha256"] == hashlib.sha256(wire(body)).hexdigest()
    assert exchange["request_hash"] == hashlib.sha256(wire(body, ascii=True)).hexdigest()
    assert selection["selected_sha256"] != exchange["request_hash"]
