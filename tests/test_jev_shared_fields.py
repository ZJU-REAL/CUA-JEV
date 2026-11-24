"""Lossless wire compaction must preserve choices, types and public-data boundaries."""

import copy
import json
from dataclasses import replace

import httpx
import pytest

from cua_jev.errors import PolicyError
from cua_jev.models import ActionCandidate, Channel, Observation, Risk
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.policy import JevPolicy


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def capture(observation, candidates, *, layout="shared_fields_v2", public=False, choice=None):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        ids = list(body["questions"]["action"]["criteria"])
        selected = choice or ids[-1]
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected,
            "probabilities": {key: float(key == ids[-1]) for key in ids},
            "confidence": 1.0,
        }}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="offline", client=client, request_layout=layout, retries=0)
        decision = (PublicDecisionPolicy(policy) if public else policy).choose(observation, candidates)
    return requests[0], decision


def expand(body):
    encoding = body["state"]["encoding"]
    assert encoding["version"] in {"shared_fields_v2", "shared_fields_v3", "shared_fields_v4"}
    candidates = [
        {"id": key, **encoding["candidate_defaults"], **fields}
        for key, fields in body["questions"]["action"]["criteria"].items()
    ]
    defaults = copy.deepcopy(encoding["control_defaults"])
    state = copy.deepcopy(body["state"]["state"])
    if "ancestor_paths" in encoding:
        def resolve(value):
            path, result, seen = value["path_ref"], [], set()
            while path is not None:
                assert path not in seen
                seen.add(path)
                node = encoding["ancestor_paths"][path]
                result.append(node["text"])
                path = node["parent"]
            return result[::-1]

        if encoding["ancestor_path_default"]:
            defaults["ancestors"] = resolve(defaults["ancestors"])
        for index in encoding["ancestor_path_controls"]:
            state["controls"][index]["ancestors"] = resolve(state["controls"][index]["ancestors"])
    if isinstance(state.get("controls"), list) and all(isinstance(c, dict) for c in state["controls"]):
        state["controls"] = [{**defaults, **control} for control in state["controls"]]
    return state, candidates


def candidates(count):
    return [ActionCandidate(
        f"action:{i}", Channel.GUI, "desktop.click", f"Click current choice {i}",
        arguments={"ref": f"d{i}", "nested": [False, 0, None]},
        risk=Risk.LOCAL_WRITE, preconditions=("fresh", "enabled"),
        verifier="state-check", expected={"changed": i},
        requires_confirmation=True, intent="navigate",
    ) for i in range(count)]


@pytest.mark.parametrize("count", [1, 2, 107, 255])
@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3", "shared_fields_v4"])
def test_all_options_and_512_controls_round_trip_without_touching_local_execution_inputs(count, layout):
    controls = [{"ref": f"d{i}", "label": f"Choice {i}", "role": "button",
                 "disabled": False, "checked": False, "has_value": False,
                 "active": True, "ancestors": ["frame: Main"], "box": [i, 0, i + 1, 20]}
                for i in range(512)]
    state = {"controls": controls, "title": "Main", "a11y_complete": True,
             "encoding": {"unrelated": "keep"}, "nested": {"controls": [{"different": True}]}}
    obs = Observation("goal", "subgoal", state)
    actions = candidates(count)
    before_state = canonical(obs.to_dict())
    before_actions = canonical([c.to_dict() for c in actions])
    body, decision = capture(obs, actions, layout=layout)
    decoded_state, decoded_actions = expand(body)
    assert canonical(decoded_state) == canonical(state)
    assert canonical(decoded_actions) == before_actions
    assert canonical(obs.to_dict()) == before_state
    assert canonical([c.to_dict() for c in actions]) == before_actions
    assert "available_actions" not in body["state"]
    assert list(body["questions"]["action"]["criteria"]) == [c.id for c in actions]
    assert decision.candidate_id == actions[-1].id
    assert set(decision.probabilities) == {c.id for c in actions}
    assert all("description" in fields for fields in body["questions"]["action"]["criteria"].values())
    assert all("ref" in c and "label" in c for c in body["state"]["state"]["controls"])


@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3"])
def test_shared_equality_never_merges_false_zero_null_missing_or_nested_type_differences(layout):
    controls = [
        {"ref": "d0", "label": "A", "flag": False, "nullable": None, "optional": "",
         "nested": {"state": False}, "shared": {"a": False, "b": [0, None]}},
        {"ref": "d1", "label": "B", "flag": 0, "nullable": None,
         "nested": {"state": 0}, "shared": {"b": [0, None], "a": False}},
    ]
    actions = candidates(2)
    obs = Observation("goal", "stage", {"controls": controls})
    body, _ = capture(obs, actions, layout=layout)
    defaults = body["state"]["encoding"]["control_defaults"]
    assert set(defaults) == {"nullable", "shared"}
    deltas = body["state"]["state"]["controls"]
    assert type(deltas[0]["flag"]) is bool and type(deltas[1]["flag"]) is int
    assert "optional" in deltas[0] and "optional" not in deltas[1]
    state, decoded = expand(body)
    assert canonical(state) == canonical(obs.state)
    assert canonical(decoded) == canonical([c.to_dict() for c in actions])
    assert "arguments" not in body["state"]["encoding"]["candidate_defaults"]


@pytest.mark.parametrize("controls", [None, [], [{"ref": "only"}], [None, {"ref": "d1"}], "opaque"])
@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3"])
def test_unsupported_empty_and_singleton_control_shapes_are_preserved(controls, layout):
    state = {"controls": controls, "other": {"controls": []}}
    body, _ = capture(Observation("goal", "stage", state), candidates(2), layout=layout)
    assert body["state"]["encoding"]["control_defaults"] == {}
    assert canonical(expand(body)[0]) == canonical(state)


@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3"])
def test_absent_controls_are_not_invented(layout):
    body, _ = capture(Observation("goal", "stage", {"title": "Main"}), candidates(2), layout=layout)
    assert expand(body)[0] == {"title": "Main"}


@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3", "shared_fields_v4"])
def test_public_redaction_precedes_factoring_and_never_puts_private_literals_in_defaults(layout):
    actions = [ActionCandidate(str(i), Channel.GUI, "fill", f"Fill field {i}",
                               arguments={"value": "private literal"},
                               expected={"value": "private literal"}) for i in range(2)]
    obs = Observation("goal", "stage", {"text": "private OCR", "visual_text": ["private visual"],
                                        "tool_results": ["private result"], "controls": []})
    before = canonical(obs.to_dict())
    body, _ = capture(obs, actions, public=True, layout=layout)
    assert "private" not in canonical(body)
    assert body["state"]["encoding"]["candidate_defaults"]["arguments"] == {}
    assert body["state"]["encoding"]["candidate_defaults"]["expected"] == {}
    assert canonical(obs.to_dict()) == before
    assert actions[0].arguments == {"value": "private literal"}


@pytest.mark.parametrize("count", [0, 256])
@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3", "shared_fields_v4"])
def test_original_candidate_cap_is_enforced_without_truncation(count, layout):
    with pytest.raises(PolicyError, match="1 and 255"):
        capture(Observation("goal", "stage", {}), candidates(count), layout=layout)


@pytest.mark.parametrize("layout", ["shared_fields_v2", "shared_fields_v3", "shared_fields_v4"])
def test_duplicate_and_unknown_choices_remain_invalid(layout):
    actions = candidates(2)
    with pytest.raises(PolicyError, match="unique"):
        capture(Observation("goal", "stage", {}), [actions[0], actions[0]], layout=layout)
    with pytest.raises(PolicyError, match="invalid Jev"):
        capture(Observation("goal", "stage", {}), actions, choice="not-offered", layout=layout)


def test_large_repeated_state_request_is_smaller_without_erasing_candidate_information():
    state = {"controls": [{"ref": f"d{i}", "label": f"Choice {i}", "tag": "button",
                           "role": "button", "app": "App", "disabled": False, "active": True,
                           "checked": False, "selected": False, "editable": False,
                           "ancestors": ["frame: Main"]} for i in range(107)]}
    obs, actions = Observation("goal", "stage", state), candidates(107)
    original, _ = capture(obs, actions, layout="deduplicated_v1")
    compact, _ = capture(obs, actions)
    assert len(canonical(compact)) < len(canonical(original)) * 0.7
    assert canonical(expand(compact)) == canonical((state, [c.to_dict() for c in actions]))


def test_modal_defaults_preserve_typed_overrides_and_never_create_a_missing_field():
    values = [False, 0, False, None]
    controls = [{"ref": f"d{i}", "label": "same", "flag": value, "nested": {"flag": value},
                 **({"optional": False} if i in {0, 2} else {})}
                for i, value in enumerate(values)]
    obs = Observation("goal", "stage", {"controls": controls, "nested": {"controls": controls}})
    before = canonical(obs.to_dict())
    body, _ = capture(obs, candidates(4), layout="shared_fields_v3")
    defaults = body["state"]["encoding"]["control_defaults"]
    assert defaults == {"flag": False, "nested": {"flag": False}}
    deltas = body["state"]["state"]["controls"]
    assert type(deltas[1]["flag"]) is int and deltas[1]["flag"] == 0
    assert deltas[3]["flag"] is None
    assert "optional" not in defaults and "optional" not in deltas[1]
    assert all("ref" in control and "label" in control for control in deltas)
    assert canonical(expand(body)[0]) == canonical(obs.state)
    assert canonical(obs.to_dict()) == before


def test_modal_candidate_defaults_keep_risk_confirmation_intent_and_every_option_exact():
    actions = candidates(4)
    actions[1] = replace(actions[1], risk=Risk.DESTRUCTIVE, requires_confirmation=False, intent="edit")
    before = canonical([a.to_dict() for a in actions])
    body, decision = capture(Observation("goal", "stage", {}), actions, layout="shared_fields_v3")
    defaults = body["state"]["encoding"]["candidate_defaults"]
    changed = body["questions"]["action"]["criteria"][actions[1].id]
    assert defaults["risk"] == "local_write" and defaults["requires_confirmation"] is True
    assert changed["risk"] == "destructive" and changed["requires_confirmation"] is False
    assert changed["intent"] == "edit" and "description" in changed
    assert canonical(expand(body)[1]) == before and canonical([a.to_dict() for a in actions]) == before
    assert set(decision.probabilities) == {action.id for action in actions}


def test_modal_ties_preserve_zero_and_false_as_different_wire_values():
    controls = [{"ref": f"d{i}", "label": "same", "flag": value}
                for i, value in enumerate([0, False, 0, False])]
    body, _ = capture(Observation("goal", "stage", {"controls": controls}), candidates(2),
                      layout="shared_fields_v3")
    defaults = body["state"]["encoding"]["control_defaults"]
    assert type(defaults["flag"]) is int
    overrides = body["state"]["state"]["controls"]
    assert overrides[1]["flag"] is overrides[3]["flag"] is False
    assert canonical(expand(body)[0]["controls"]) == canonical(controls)


def test_modal_layout_saves_mixed_control_repetition_that_v2_must_repeat():
    controls = [{"ref": f"d{i}", "label": f"Field {i}", "app": "Editor" if i < 100 else "Shell",
                 "window": "Document" if i < 100 else "", "ancestors": ["section: " + "a" * 600],
                 "focused": i == 20, "active": i < 100, "editable": i == 20} for i in range(107)]
    controls[106]["ancestors"] = ["shell"]
    obs = Observation("goal", "stage", {"controls": controls})
    v2, _ = capture(obs, candidates(107), layout="shared_fields_v2")
    v3, _ = capture(obs, candidates(107), layout="shared_fields_v3")
    assert len(canonical(v3)) < len(canonical(v2))
    assert canonical(expand(v2)) == canonical(expand(v3))
    assert "Defaults are a representation choice" in v3["questions"]["action"]["instructions"]


def test_ancestor_paths_preserve_order_duplicates_defaults_and_all_control_records():
    prefix = "section: " + "observed content " * 80
    controls = [{"ref": f"d{i}", "label": f"Choice {i}", "app": "Editor",
                 "ancestors": [prefix, "repeated", "repeated", "same" if i < 100 else f"leaf {i}"]}
                for i in range(200)]
    state = {"controls": controls, "other": {"controls": copy.deepcopy(controls)}}
    observation, actions = Observation("goal", "stage", state), candidates(107)
    before = canonical((observation.to_dict(), [c.to_dict() for c in actions]))
    v3, _ = capture(observation, actions, layout="shared_fields_v3")
    v4, decision = capture(observation, actions, layout="shared_fields_v4")
    encoding = v4["state"]["encoding"]
    assert encoding["ancestor_path_default"] is True
    assert encoding["ancestor_path_controls"] == list(range(100, 200))
    assert sum(node["text"] == prefix for node in encoding["ancestor_paths"].values()) == 1
    assert len(canonical(v4)) < len(canonical(v3))
    assert canonical(expand(v4)) == canonical(expand(v3))
    assert canonical((observation.to_dict(), [c.to_dict() for c in actions])) == before
    assert list(v4["questions"]["action"]["criteria"]) == [c.id for c in actions]
    assert decision.candidate_id == actions[-1].id and len(decision.probabilities) == 107


def test_unmarked_path_shaped_data_and_unsupported_ancestor_types_stay_literal():
    prefix = "frame: " + "long visible name " * 80
    controls = [{"ref": f"d{i}", "label": f"Choice {i}", "ancestors": [prefix, f"section {i}"]}
                for i in range(100)]
    unusual = [{"path_ref": "p0"}, None, False, 0, "literal", [False], [None], []]
    controls.extend({"ref": f"u{i}", "label": "literal", "ancestors": value}
                    for i, value in enumerate(unusual))
    controls.append({"ref": "missing", "label": "No observed ancestors field"})
    state = {"controls": controls, "nested": {"ancestors": {"path_ref": "p0"}}}
    body, _ = capture(Observation("goal", "stage", state), candidates(2), layout="shared_fields_v4")
    assert "ancestor_paths" in body["state"]["encoding"]
    assert body["state"]["encoding"]["ancestor_path_controls"] == list(range(100))
    assert canonical(expand(body)[0]) == canonical(state)
    assert "ancestors" not in expand(body)[0]["controls"][-1]


def test_tiny_path_data_is_left_literal_when_references_would_increase_its_size():
    state = {"controls": [{"ref": "d0", "label": "Small", "ancestors": ["Main"]}]}
    body, _ = capture(Observation("goal", "stage", state), candidates(2), layout="shared_fields_v4")
    assert "ancestor_paths" not in body["state"]["encoding"]
    assert canonical(expand(body)[0]) == canonical(state)


@pytest.mark.parametrize("code,expected", [
    ("max_tokens_exceeded", "max_tokens_exceeded"), ("private_code", None),
])
def test_actual_provider_error_type_is_classified_without_echoing_provider_messages(code, expected):
    def handle(request):
        return httpx.Response(400, json={"detail": {"error_type": code, "message": "private_value"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        policy = JevPolicy(api_key="offline", client=client, retries=0, request_layout="shared_fields_v4")
        with pytest.raises(PolicyError, match="HTTP 400") as error:
            policy.choose(Observation("goal", "stage", {}), candidates(2))
    assert policy.last_failure.get("service_code") == expected
    assert "private" not in str(error.value) and "private" not in canonical(policy.last_failure)
