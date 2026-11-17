"""Exact reversible v7 values, conservative ancestor fallback, local-only ID maps."""

import copy
from dataclasses import replace

import pytest

from cua_jev.jev_control_description_encoding import _anchor, share_public_control_descriptions
from cua_jev.jev_wire_handle_encoding import (
    ALL_MARKER,
    MASK,
    _wire,
    restore_wire_handle_request,
    select_wire_handle_encoding,
    validate_wire_provider_response,
)
from cua_jev.models import ActionCandidate, Channel, Observation


def public_inputs(count=24):
    """Fixed public-only offers with all original gesture distinctions in descriptions."""
    controls, candidates = [], []
    for index in range(count):
        control = {"ref": f"d{index}", "label": f"Observed 文件 {index}", "role": "button",
                   "app": "Observed App", "window": "Observed current public window",
                   "ancestors": ["frame: public window", "panel: current tools"],
                   "checked": False, "value_length": 0, "extra": None}
        controls.append(control)
        for gesture, suffix in enumerate(("", " (right button, 1 click(s))", " (left button, 2 click(s))")):
            candidates.append(ActionCandidate(
                f"reach:desktop:{3 * index + gesture:08x}", Channel.GUI, "agent.osworld.click",
                "click " + _anchor(control) + suffix + " toward navigation goal: Open the requested view",
                intent="reach", arguments={},
            ))
    observation = Observation("Open the requested view", "Reach the current target", {
        "controls": controls, "title": "Observed current public window", "a11y_complete": True,
        "typed_facts": [False, 0, 0.0, None, "0", [], {}],
        "ready_steps": [{"completion_conditions": [{"kind": "exists", "outcome": "unknown",
                                                     "target": {"name": "Requested view"}}]}],
        "navigation_memory": {"version": "observed_transition_v1", "edges": []},
    }, source="jev-vlm-agent", observation_id="fixed-public-observation", created_at=0.0)
    return observation, tuple(candidates)


def v6_body(count=24):
    observation, candidates = public_inputs(count)
    body = {"model": "jev-1.13.0", "state": {
        "observation_id": observation.observation_id, "state": copy.deepcopy(observation.state),
        "encoding": {"version": "shared_fields_v4", "candidate_defaults": {}, "control_defaults": {}},
    }, "questions": {"action": {"instructions": "Layout shared_fields_v4: original complete guidance",
        "criteria": {item.id: {key: value for key, value in item.to_dict().items() if key != "id"}
                     for item in candidates}}}}
    assert share_public_control_descriptions(body)
    return body


def select(body, **kwargs):
    return select_wire_handle_encoding(
        body, offered_ids=list(body["questions"]["action"]["criteria"]),
        observation_id=body["state"]["observation_id"], **kwargs,
    )


def restored(selection):
    return restore_wire_handle_request(
        selection.body, selection.alias, offered_ids=selection.alias.original_ids,
        observation_id=selection.alias.observation_id,
    )


def test_full_ordered_body_values_controls_and_all_gestures_restore_exactly():
    body = v6_body()
    before = _wire(body)
    controls = body["state"]["state"]["controls"]
    result = select(body)
    assert result.alias is not None
    assert result.body["state"]["encoding"]["version"] == "shared_fields_v7"
    assert result.body["state"]["encoding"][ALL_MARKER] is True
    assert MASK not in result.body["state"]["encoding"]
    assert _wire(restored(result)) == before
    assert body["state"]["state"]["controls"] is controls and _wire(body) == before
    assert _wire(result.body["state"]["state"]) == _wire(body["state"]["state"])
    assert len(_wire(result.body)) < len(before)
    assert [record["description"][1] for record in list(
        result.body["questions"]["action"]["criteria"].values()
    )[:3]] == [0, 1, 2]
    diagnostics = dict(result.diagnostics)
    assert diagnostics["selected_compact_UTF8_bytes"] == len(_wire(result.body))
    assert diagnostics["alias_count"] == 72 and diagnostics["token_fit_proven"] is False
    assert diagnostics["self_contained_encoding"] is False


@pytest.mark.parametrize("problem", ["extra_value", "extra_key", "embedded", "wire_collision",
                                     "partial_mask", "reversed_mask", "semantic_ID", "intent"])
def test_unknown_references_semantic_IDs_and_partial_masks_keep_entire_ancestor(problem):
    body = v6_body()
    criteria = body["questions"]["action"]["criteria"]
    candidate_id = next(iter(criteria))
    if problem == "extra_value":
        body["state"]["state"]["history"] = {"candidate_id": candidate_id}
    elif problem == "extra_key":
        body["state"]["state"]["literal"] = {candidate_id: "unchanged"}
    elif problem == "embedded":
        body["state"]["state"]["literal"] = f"prefix:{candidate_id}:suffix"
    elif problem == "wire_collision":
        body["state"]["state"]["literal"] = "h0"
    elif problem == "partial_mask":
        body["state"]["encoding"][MASK].pop()
    elif problem == "reversed_mask":
        body["state"]["encoding"][MASK].reverse()
    elif problem == "semantic_ID":
        replacement = "important semantic action"
        body["questions"]["action"]["criteria"] = {
            replacement if key == candidate_id else key: value for key, value in criteria.items()
        }
        body["state"]["encoding"][MASK][0] = replacement
    else:
        criteria[candidate_id]["intent"] = "different semantic intent"
    before = _wire(body)
    result = select(body)
    assert result.body is body and result.alias is None and _wire(body) == before
    assert dict(result.diagnostics)["fallback_reason"]


@pytest.mark.parametrize("description", [
    {"control_ref": "d0", "template_id": False},
    {"control_ref": "d0", "template_id": 0, "label_hint": None},
    {"control_ref": "d0", "template_id": 0, "label_hint": "incorrect public label"},
    {"control_ref": "d0", "template_id": 0, "unknown": "literal"},
    {"template_id": 0, "control_ref": "d0"},
    {"control_ref": "missing", "template_id": 0},
    "a literal nonencoded description",
])
def test_not_strict_valid_two_or_three_fields_returns_exact_v6(description):
    body = v6_body()
    next(iter(body["questions"]["action"]["criteria"].values()))["description"] = description
    before = _wire(body)
    result = select(body)
    assert result.body is body and result.alias is None and _wire(body) == before


def test_absent_hint_nested_reference_looking_literals_and_zero_types_are_exact():
    body = v6_body()
    record = next(iter(body["questions"]["action"]["criteria"].values()))
    del record["description"]["label_hint"]
    body["state"]["state"]["deep_literal"] = [
        {"description": {"control_ref": "literal", "template_id": 0}},
        {"control_description_all": True, "wire_criterion_handles": None},
    ]
    result = select(body)
    assert result.alias is not None
    assert len(next(iter(result.body["questions"]["action"]["criteria"].values()))["description"]) == 2
    assert _wire(restored(result)) == _wire(body)


@pytest.mark.parametrize("version", ["shared_fields_v4", "shared_fields_v5"])
def test_old_selected_ancestor_layout_is_reported_honestly_without_changing_body(version):
    body = v6_body()
    body["state"]["encoding"]["version"] = version
    before = _wire(body)
    result = select(body)
    assert result.body is body and result.alias is None and _wire(body) == before
    assert dict(result.diagnostics)["actual_layout"] == version
    assert dict(result.diagnostics)["fallback_reason"] == "ancestor_not_v6"


def test_profit_strict_boundary_and_no_profit_preserve_object_references():
    body = v6_body()
    controls = body["state"]["state"]["controls"]
    reference = select(body)
    saving = len(_wire(body)) - len(_wire(reference.body))
    assert select(body, minimum_savings=saving - 1).alias is not None
    result = select(body, minimum_savings=saving)
    assert result.alias is None and result.body is body
    assert body["state"]["state"]["controls"] is controls
    assert dict(result.diagnostics)["fallback_reason"] == "complete_body_no_profit"


@pytest.mark.parametrize("problem", ["tuple", "all_marker", "instruction", "map_order", "hash", "obs"])
def test_corruption_or_wrong_request_binding_is_rejected_without_guessing(problem):
    result = select(v6_body())
    body, alias = copy.deepcopy(result.body), result.alias
    if problem == "tuple":
        next(iter(body["questions"]["action"]["criteria"].values()))["description"][1] = False
    elif problem == "all_marker":
        body["state"]["encoding"][ALL_MARKER] = 1
    elif problem == "instruction":
        body["questions"]["action"]["instructions"] += " altered"
    elif problem == "map_order":
        alias = replace(alias, original_ids=tuple(reversed(alias.original_ids)))
    elif problem == "hash":
        alias = replace(alias, ancestor_ordered_sha256="0" * 64)
    else:
        alias = replace(alias, observation_id="different-observation")
    with pytest.raises(ValueError):
        restore_wire_handle_request(body, alias, offered_ids=result.alias.original_ids,
                                    observation_id=result.alias.observation_id)


def test_duplicate_application_and_nonfinite_input_are_rejected_before_mutation():
    result = select(v6_body())
    before = _wire(result.body)
    with pytest.raises(ValueError):
        select_wire_handle_encoding(result.body, offered_ids=result.alias.handles,
                                    observation_id=result.alias.observation_id)
    assert _wire(result.body) == before
    body = v6_body()
    controls = body["state"]["state"]["controls"]
    body["state"]["state"]["nonfinite"] = float("nan")
    with pytest.raises(ValueError):
        select(body)
    assert body["state"]["state"]["controls"] is controls


@pytest.mark.parametrize("raw", [b'{"answers":{},"answers":{}}',
                                  b'{"answer":{"choice":"h0","choice":"h1"}}',
                                  b'{"probabilities":{"h0":0,"h0":1}}',
                                  b'{"constant":NaN}', b'{"broken":'])
def test_provider_duplicate_keys_nonfinite_or_invalid_JSON_are_rejected(raw):
    with pytest.raises(ValueError):
        validate_wire_provider_response(raw)


def test_probability_order_and_types_map_only_to_current_original_offers():
    result = select(v6_body())
    alias = result.alias
    probabilities = {key: float(key == alias.handles[-1]) for key in reversed(alias.handles)}
    answer = {"choice": alias.handles[-1], "confidence": .95, "probabilities": probabilities}
    before = _wire(answer)
    mapped = alias.restore_answer(answer, offered_ids=alias.original_ids, observation_id=alias.observation_id)
    assert mapped["choice"] == alias.original_ids[-1]
    assert list(mapped["probabilities"]) == list(reversed(alias.original_ids))
    assert _wire(answer) == before
    for bad in (
        {**answer, "choice": alias.original_ids[-1]},
        {**answer, "choice": "invented"}, {**answer, "confidence": True},
        {**answer, "probabilities": {**probabilities, "invented": 0}}, {**answer, "extra": None},
    ):
        with pytest.raises(ValueError):
            alias.restore_answer(bad, offered_ids=alias.original_ids, observation_id=alias.observation_id)
