"""Regression tests for jev layout selection boundaries."""

import copy
import json

import pytest
from test_jev_control_description_encoding import body, wire

from cua_jev.jev_control_description_encoding import (
    MAX_HINT_BYTES,
    _anchor,
    share_public_control_descriptions,
)
from cua_jev.jev_layout_selection import select_public_request_encoding


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def selector_body():
    return {
        "model": "same-model",
        "state": {"state": {"controls": [], "typed_public_values": [False, 0, None, "0"]},
                  "encoding": {"version": "shared_fields_v4"}},
        "questions": {"action": {"type": "choice", "instructions": "Original public instructions",
                                  "criteria": {"first": {"description": "z" * 2048},
                                               "second": {"description": "z" * 2048}}}},
    }


class Branches:
    suffix = " EXACT_TEST_ONLY_SECOND_LAYER"

    def __init__(self, before, *, v5_delta=None, v6_delta=None, fragments=False):
        self.base_length = len(compact(before))
        self.v5_delta, self.v6_delta = v5_delta, v6_delta
        self.fragments = fragments
        self.events = []
        self.captured = {}

    def encode(self, value, version, delta):
        if delta is None:
            return False
        encoding = value["state"]["encoding"]
        assert encoding == {"version": "shared_fields_v4"}
        assert value["state"]["state"]["typed_public_values"] == [False, 0, None, "0"]
        for record in value["questions"]["action"]["criteria"].values():
            literal = record["description"]
            assert type(literal) is str and literal and set(literal) == {"z"}
            record["description"] = {"fixture_character": "z", "fixture_count": len(literal)}
        encoding.update(version=version, fixture_padding="")
        needed = self.base_length + delta - len(compact(value))
        assert needed >= 0
        encoding["fixture_padding"] = "p" * needed
        assert len(compact(value)) == self.base_length + delta
        self.captured[version] = copy.deepcopy(value)
        return True

    def share_args(self, value):
        self.events.append("share_args")
        return self.encode(value, "shared_fields_v5", self.v5_delta)

    def share_fragments(self, value):
        self.events.append("share_fragments")
        if not self.fragments:
            return False
        assert value["state"]["encoding"]["version"] == "shared_fields_v5"
        value["state"]["encoding"]["fixture_fragment"] = True
        value["questions"]["action"]["instructions"] += self.suffix
        self.captured["shared_fields_v5"] = copy.deepcopy(value)
        return True

    def share_v6(self, value):
        self.events.append("share_v6")
        return self.encode(value, "shared_fields_v6", self.v6_delta)

    def restore_fragments(self, value):
        self.events.append("restore_fragments")
        restored = copy.deepcopy(value)
        assert restored["state"]["encoding"].pop("fixture_fragment") is True
        question = restored["questions"]["action"]
        assert question["instructions"].endswith(self.suffix)
        question["instructions"] = question["instructions"][:-len(self.suffix)]
        return restored

    def restore(self, value, version):
        restored = copy.deepcopy(value)
        encoding = restored["state"]["encoding"]
        assert encoding["version"] == version and "fixture_fragment" not in encoding
        encoding.pop("fixture_padding")
        encoding["version"] = "shared_fields_v4"
        for record in restored["questions"]["action"]["criteria"].values():
            encoded = record["description"]
            assert set(encoded) == {"fixture_character", "fixture_count"}
            record["description"] = encoded["fixture_character"] * encoded["fixture_count"]
        return restored

    def restore_args(self, value):
        self.events.append("restore_args")
        return self.restore(value, "shared_fields_v5")

    def restore_v6(self, value):
        self.events.append("restore_v6")
        return self.restore(value, "shared_fields_v6")

    def apply(self, value, **overrides):
        options = {
            "share_argument_bases": self.share_args,
            "restore_argument_bases": self.restore_args,
            "share_description_fragments": self.share_fragments,
            "restore_description_fragments": self.restore_fragments,
            "share_public_control_descriptions": self.share_v6,
            "restore_public_control_descriptions": self.restore_v6,
        }
        return select_public_request_encoding(value, **{**options, **overrides})


@pytest.mark.parametrize("v5,v6,expected", [
    (-200, None, "shared_fields_v5"), (None, -200, "shared_fields_v6"),
    (-200, -300, "shared_fields_v6"), (-300, -200, "shared_fields_v5"),
    (-200, -200, "shared_fields_v5"),
])
def test_independent_single_branches_exact_full_body_size_and_established_v5_tie(v5, v6, expected):
    request = selector_body()
    before = copy.deepcopy(request)
    branches = Branches(before, v5_delta=v5, v6_delta=v6)
    diagnostics = branches.apply(request)
    assert diagnostics["actual_layout"] == expected
    assert compact(request) == compact(branches.captured[expected])
    assert diagnostics["selected_compact_UTF8_bytes"] == len(compact(request))
    assert len(compact(request)) < len(compact(before))
    assert diagnostics["requested_layout"] == "shared_fields_v6"
    assert "requested_layout" not in json.dumps(request)
    assert branches.events[:3] == ["share_args", "share_fragments", "share_v6"]
    assert list(request["questions"]["action"]["criteria"]) == ["first", "second"]


@pytest.mark.parametrize("v5,v6", [(None, None), (0, 0), (None, 0), (0, None)])
def test_equal_to_baseline_is_not_profit_and_preserves_nested_original_identities(v5, v6):
    request = selector_body()
    before = compact(request)
    controls = request["state"]["state"]["controls"]
    criteria = request["questions"]["action"]["criteria"]
    diagnostics = Branches(request, v5_delta=v5, v6_delta=v6).apply(request)
    assert diagnostics["actual_layout"] == "shared_fields_v4" and compact(request) == before
    assert request["state"]["state"]["controls"] is controls
    assert request["questions"]["action"]["criteria"] is criteria


def test_complete_old_two_stage_branch_runs_once_and_restores_fragment_before_arguments():
    request = selector_body()
    branches = Branches(request, v5_delta=-500, fragments=True)
    diagnostics = branches.apply(request)
    assert diagnostics["actual_layout"] == "shared_fields_v5"
    assert branches.events == ["share_args", "share_fragments", "share_v6",
                               "restore_fragments", "restore_args"]
    assert compact(request) == compact(branches.captured["shared_fields_v5"])


@pytest.mark.parametrize("problem", ["growing_body", "typed_value", "candidate_order", "false_mutation",
                                     "non_boolean_flag", "nonfinite_body"])
def test_invalid_generated_branch_is_rejected_without_mutating_base_or_silent_other_branch_fallback(problem):
    request = selector_body()
    before = compact(request)
    branches = Branches(request, v5_delta=-500, v6_delta=-300)
    overrides = {}
    if problem == "growing_body":
        branches.v6_delta = 1
    elif problem == "typed_value":
        def corrupt_restore(value):
            restored = branches.restore_v6(value)
            restored["state"]["state"]["typed_public_values"][0] = 0  # false and zero stay distinct.
            return restored
        overrides["restore_public_control_descriptions"] = corrupt_restore
    elif problem == "candidate_order":
        def reversed_choices(value):
            result = branches.share_v6(value)
            question = value["questions"]["action"]
            question["criteria"] = dict(reversed(list(question["criteria"].items())))
            return result
        overrides["share_public_control_descriptions"] = reversed_choices
    elif problem == "false_mutation":
        def wrong_false(value):
            branches.share_v6(value)
            return False
        overrides["share_public_control_descriptions"] = wrong_false
    elif problem == "non_boolean_flag":
        def wrong_flag(value):
            branches.share_v6(value)
            return 1
        overrides["share_public_control_descriptions"] = wrong_flag
    else:
        def nonfinite(value):
            branches.share_v6(value)
            value["state"]["state"]["typed_public_values"].append(float("nan"))
            return True
        overrides["share_public_control_descriptions"] = nonfinite
    with pytest.raises(ValueError):
        branches.apply(request, **overrides)
    assert compact(request) == before


def no_hint_body():
    request = body()
    for index, control in enumerate(request["state"]["state"]["controls"]):
        control["label"] = f"Observed {index} " + "x" * (MAX_HINT_BYTES + 1)
        request["questions"]["action"]["criteria"][f"option{index}"]["description"] = (
            "click " + _anchor(control) + " literal observed suffix"
        )
    return request


def test_actual_codec_threshold_is_strict_at_complete_body_savings_without_optional_hint_ambiguity():
    original = no_hint_body()
    reference = copy.deepcopy(original)
    assert share_public_control_descriptions(reference, minimum_savings=0)
    assert all("label_hint" not in record["description"]
               for record in reference["questions"]["action"]["criteria"].values())
    savings = len(wire(original)) - len(wire(reference))
    assert savings > 1
    below = copy.deepcopy(original)
    assert share_public_control_descriptions(below, minimum_savings=savings - 1)
    assert wire(below) == wire(reference)
    for threshold in (savings, savings + 1):
        request = copy.deepcopy(original)
        controls = request["state"]["state"]["controls"]
        criteria = request["questions"]["action"]["criteria"]
        assert share_public_control_descriptions(request, minimum_savings=threshold) is False
        assert wire(request) == wire(original)
        assert request["state"]["state"]["controls"] is controls
        assert request["questions"]["action"]["criteria"] is criteria


@pytest.mark.parametrize("threshold", [False, True, -1, 0.0, "0", None])
def test_actual_codec_profit_threshold_requires_a_nonnegative_exact_integer(threshold):
    request = body()
    before = wire(request)
    with pytest.raises(ValueError):
        share_public_control_descriptions(request, minimum_savings=threshold)
    assert wire(request) == before
