"""Lossless JSON roundtrips and conservative argument-sharing boundaries."""

import copy
import json

import pytest

from cua_jev.jev_argument_encoding import restore_argument_bases, share_argument_bases

_ABSENT = object()
_GESTURES = ({"button": "left", "click_count": 1},
             {"button": "right", "click_count": 1},
             {"button": "left", "click_count": 2})


def _wire(value):
    # Independent full-JSON oracle: Python equality would conflate False and 0.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _body(variants=(False,), *, operation="click", padding=4096, gestures=_GESTURES):
    criteria = {}
    for turn, gesture in enumerate(gestures):
        for group, fact in enumerate(variants):
            key = ("a0", "third", "first")[turn] if turn < 3 else f"extra-{turn}"
            key = f"{key}-g{group}"
            arguments = {
                "operation": operation, "padding": "雪🙂" * padding,
                "bound_control": {"ref": "d1", "label": "Undo", "value": "5.0"},
                "typed": [None, False, 0, True, 1, 1.0, -0.0, ["b", "a"]],
                "literal": {"deep": [{"base_ref": "a0", "gesture": {"button": "right"},
                                      "ancestors": {"path_ref": "p0"}}]},
                **copy.deepcopy(gesture),
            }
            if fact is not _ABSENT:
                arguments["fact"] = copy.deepcopy(fact)
            criteria[key] = {"description": f"{key}: literal description 雪🙂",
                             "arguments": arguments, "risk": "local_write"}
    for operation in ("fill", "type_text", "keyPress", "check", "select_option"):
        criteria[operation] = {
            "description": f"mutation {operation}",
            "arguments": {"operation": operation, "value": False,
                          "button": None, "click_count": 0, "nested": {"base_ref": "a0"}},
        }
    criteria.update({
        "inherited": {"description": "inherits whole default arguments"},
        "null": {"description": "explicit null", "arguments": None},
        "empty": {"description": "whole empty override", "arguments": {}},
        "lookalike": {"description": "unmarked literal wrapper",
                      "arguments": {"base_ref": "a0", "gesture": {}}},
    })
    return {
        "model": "offline-fixture", "temperature": 0, "strict": False,
        "state": {
            "task": "literal task", "observation_id": "original-observation", "source": "fixture",
            "state": {"title": "Preferences", "controls": [
                {"ref": "d1", "label": "Undo", "ancestors": {"path_ref": "p0"}},
                {"ref": "d2", "label": "Literal", "data": {"path_ref": "not-a-path"}},
            ], "nested": [None, False, 0, {"base_ref": "a0"}]},
            "available_actions": [{"id": "original", "arguments": {"base_ref": "a0"}}],
            "encoding": {
                "version": "shared_fields_v4",
                "candidate_defaults": {"channel": "gui", "arguments": {
                    "operation": "click", "button": "right", "value": "inherited-only"}},
                "control_defaults": {"active": False, "ancestors": {"path_ref": "p0"}},
                "ancestor_paths": {"p0": {"parent": None, "text": "dialog: Preferences"}},
                "ancestor_path_default": True, "ancestor_path_controls": [0],
            },
        },
        "questions": {"action": {"type": "choice", "criteria": criteria,
                                  "instructions": "Layout shared_fields_v4: original strong guidance 雪🙂"}},
    }


def _refs(value, path=()):
    result = {path: value} if isinstance(value, (dict, list)) else {}
    entries = (value.items() if isinstance(value, dict) else
               enumerate(value) if isinstance(value, list) else ())
    for key, item in entries:
        result.update(_refs(item, (*path, key)))
    return result


def _snapshot(body):
    return repr(body), _refs(body)


def _assert_untouched(body, snapshot):
    representation, references = snapshot
    assert repr(body) == representation
    current = _refs(body)
    assert current.keys() == references.keys()
    assert all(current[path] is value for path, value in references.items())


def _encode(body):
    original = copy.deepcopy(body)
    order = list(original["questions"]["action"]["criteria"])
    descriptions = [item["description"] for item in original["questions"]["action"]["criteria"].values()]
    assert share_argument_bases(body) is True
    criteria = body["questions"]["action"]["criteria"]
    assert list(criteria) == order
    assert [item["description"] for item in criteria.values()] == descriptions
    encoded_snapshot = _snapshot(body)
    restored = restore_argument_bases(body)
    assert _wire(restored) == _wire(original)
    assert list(restored["questions"]["action"]["criteria"]) == order
    _assert_untouched(body, encoded_snapshot)
    return original


@pytest.mark.parametrize("operation", ["click", "visual_click"])
def test_complete_body_roundtrip_and_nonclick_default_literals(operation):
    body = _body(operation=operation)
    original = _encode(body)
    encoding = body["state"]["encoding"]
    assert encoding["version"] == "shared_fields_v5"
    for key in ("candidate_defaults", "control_defaults", "ancestor_paths",
                "ancestor_path_default", "ancestor_path_controls"):
        assert _wire(encoding[key]) == _wire(original["state"]["encoding"][key])
    marked = encoding["argument_base_candidates"]
    assert marked == [f"{key}-g0" for key in ("a0", "third", "first")]
    wrapped = body["questions"]["action"]["criteria"][marked[0]]["arguments"]
    literal = encoding["argument_bases"][wrapped["base_ref"]]["literal"]
    original_literal = original["questions"]["action"]["criteria"][marked[0]]["arguments"]["literal"]
    assert _wire(literal) == _wire(original_literal)
    for key, record in original["questions"]["action"]["criteria"].items():
        if key not in marked:
            assert _wire(body["questions"]["action"]["criteria"][key]) == _wire(record)


@pytest.mark.parametrize("variants", [
    (False, 0, True, 1, 1.0, -0.0),
    (_ABSENT, None, "", [], {}),
    ({"bound_control": {"ref": "same", "value": "x"}},
     {"bound_control": {"ref": "same", "value": "y"}}),
])
def test_typed_distinct_bases_are_not_conflated(variants):
    body = _body(variants)
    _encode(body)
    encoding = body["state"]["encoding"]
    assert len(encoding["argument_bases"]) == len(variants)
    criteria = body["questions"]["action"]["criteria"]
    refs = [criteria[f"a0-g{group}"]["arguments"]["base_ref"] for group in range(len(variants))]
    assert len(set(refs)) == len(variants)


def test_gesture_missing_null_and_literal_types_do_not_gain_defaults():
    gestures = (*_GESTURES, {}, {"button": None}, {"click_count": None},
                {"button": False, "click_count": True})
    body = _body(gestures=gestures)
    _encode(body)
    criteria = body["questions"]["action"]["criteria"]
    marked = body["state"]["encoding"]["argument_base_candidates"]
    assert len(marked) == len(gestures)
    assert len(body["state"]["encoding"]["argument_bases"]) == 1
    for key, gesture in zip(marked, gestures, strict=True):
        assert _wire(criteria[key]["arguments"]["gesture"]) == _wire(gesture)


@pytest.mark.parametrize("case", ["no_groups", "tiny_group", "strong_threshold"])
def test_unprofitable_fallback_retains_entire_body_and_all_object_references(case):
    body = _body(padding=0, gestures=({},)) if case == "no_groups" else _body(padding=0)
    if case == "strong_threshold":
        body = _body()
    before = _snapshot(body)
    wire_before = _wire(body)
    threshold = 10**9 if case == "strong_threshold" else 0
    assert share_argument_bases(body, minimum_savings=threshold) is False
    assert _wire(body) == wire_before
    _assert_untouched(body, before)


def test_exact_savings_threshold_counts_complete_utf8_body():
    body = _body()
    trial = copy.deepcopy(body)
    assert share_argument_bases(trial) is True
    savings = len(_wire(body)) - len(_wire(trial))
    assert savings > 0
    before = _snapshot(body)
    assert share_argument_bases(body, minimum_savings=savings) is False
    _assert_untouched(body, before)
    assert share_argument_bases(body, minimum_savings=savings - 1) is True
    assert _wire(restore_argument_bases(body)) == _wire(restore_argument_bases(trial))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"),
                                 {1: "nonstring key"}, b"unsupported"])
def test_non_json_or_nonfinite_anywhere_rejects_without_mutation(bad):
    body = _body()
    body["state"]["state"]["outside_eligible_arguments"] = bad
    before = _snapshot(body)
    with pytest.raises(ValueError):
        share_argument_bases(body)
    _assert_untouched(body, before)


@pytest.mark.parametrize("case", ["missing_state", "bad_encoding", "bad_record", "bad_prefix"])
def test_malformed_request_rejects_before_any_mutation(case):
    body = _body()
    if case == "missing_state":
        del body["state"]
    elif case == "bad_encoding":
        body["state"]["encoding"] = []
    elif case == "bad_record":
        body["questions"]["action"]["criteria"]["bad"] = None
    else:
        body["questions"]["action"]["instructions"] = "Layout shared_fields_v3: wrong pass"
    before = _snapshot(body)
    with pytest.raises((ValueError, KeyError, TypeError)):
        share_argument_bases(body)
    _assert_untouched(body, before)


@pytest.mark.parametrize("collision", ["already_encoded", "argument_bases", "argument_base_candidates"])
def test_second_encoding_or_existing_marker_is_rejected_without_mutation(collision):
    body = _body()
    if collision == "already_encoded":
        assert share_argument_bases(body) is True
    else:
        body["state"]["encoding"][collision] = {}
    before = _snapshot(body)
    with pytest.raises(ValueError):
        share_argument_bases(body)
    _assert_untouched(body, before)


@pytest.mark.parametrize("corruption", [
    "version", "missing_table", "missing_mask", "duplicate_mask", "unknown_mask", "mask_order",
    "unknown_base", "unused_base", "base_operation", "base_gesture", "wrapper_extra",
    "gesture_extra", "instruction_envelope", "nonfinite_base",
])
def test_restore_rejects_marker_mask_and_base_corruption_without_repair(corruption):
    body = _body()
    assert share_argument_bases(body) is True
    encoding = body["state"]["encoding"]
    mask = encoding["argument_base_candidates"]
    arguments = body["questions"]["action"]["criteria"][mask[0]]["arguments"]
    base = encoding["argument_bases"][arguments["base_ref"]]
    question = body["questions"]["action"]
    damage = {
        "version": lambda: encoding.update(version="shared_fields_v4"),
        "missing_table": lambda: encoding.pop("argument_bases"),
        "missing_mask": lambda: encoding.pop("argument_base_candidates"),
        "duplicate_mask": lambda: mask.append(mask[0]),
        "unknown_mask": lambda: mask.__setitem__(0, "absent-candidate"),
        "mask_order": lambda: mask.reverse(),
        "unknown_base": lambda: arguments.update(base_ref="absent-base"),
        "unused_base": lambda: encoding["argument_bases"].update(unused=copy.deepcopy(base)),
        "base_operation": lambda: base.update(operation="fill"),
        "base_gesture": lambda: base.update(button="right"),
        "wrapper_extra": lambda: arguments.update(value="smuggled override"),
        "gesture_extra": lambda: arguments["gesture"].update(operation="fill"),
        "instruction_envelope": lambda: question.update(instructions=question["instructions"] + "tampered"),
        "nonfinite_base": lambda: base.update(literal=float("nan")),
    }
    damage[corruption]()
    before = _snapshot(body)
    with pytest.raises(ValueError):
        restore_argument_bases(body)
    _assert_untouched(body, before)
