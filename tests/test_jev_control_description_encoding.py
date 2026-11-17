"""Regression tests for jev control description encoding."""

import copy
import json

import pytest

from cua_jev.jev_control_description_encoding import (
    MAX_HINT_BYTES,
    _anchor,
    restore_public_control_descriptions,
    share_public_control_descriptions,
)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def body(count=60):
    controls, criteria = [], {}
    for i in range(count):
        control = {"ref": f"c{i}", "label": f"Observed widget {i}", "role": "button",
                   "app": "ObservedApp", "window": "Observed application frame with repeated context",
                   "ancestors": ["frame: observed public frame", "panel: observed public tools"],
                   "checked": False, "value_length": 0, "extra": None}
        controls.append(control)
        criteria[f"option{i}"] = {
            "description": "click " + _anchor(control)
                           + " toward navigation goal: public literal goal via OSWorld input",
            "intent": "public-step", "arguments": {}, "expected": {},
            "extra_values": [False, 0, None, "0"],
        }
    return {"model": "UNCHANGED", "state": {"state": {"controls": controls},
            "encoding": {"version": "shared_fields_v4", "candidate_defaults": {},
                         "control_defaults": {}}},
            "questions": {"action": {"criteria": criteria,
                "instructions": "Layout shared_fields_v4: retained original instructions"}}}


def test_exact_json_roundtrip_ids_order_values_and_entire_body_profit():
    original = body()
    encoded = copy.deepcopy(original)
    assert share_public_control_descriptions(encoded)
    assert len(wire(encoded)) < len(wire(original))
    assert wire(restore_public_control_descriptions(encoded)) == wire(original)
    assert list(encoded["questions"]["action"]["criteria"]) == list(
        original["questions"]["action"]["criteria"]
    )
    assert encoded["state"]["state"]["controls"] == original["state"]["state"]["controls"]
    assert all(record["arguments"] == {} for record in encoded["questions"]["action"]["criteria"].values())


@pytest.mark.parametrize("field,value", [
    ("label", None), ("role", False), ("app", 0), ("window", None), ("ancestors", None),
    ("ancestors", {"path_ref": "literal-unmarked"}),
])
def test_missing_unknown_typed_or_unmarked_reference_fields_keep_candidate_literal(field, value):
    encoded = body()
    original = encoded["questions"]["action"]["criteria"]["option0"]["description"]
    encoded["state"]["state"]["controls"][0][field] = value
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"]["description"] == original


@pytest.mark.parametrize("identical", [True, False])
def test_duplicate_public_ref_even_identical_is_ambiguous_and_left_literal(identical):
    encoded = body()
    control = copy.deepcopy(encoded["state"]["state"]["controls"][0])
    if not identical:
        control["label"] += " conflicting"
    encoded["state"]["state"]["controls"].append(control)
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"]["description"] == (
        before["questions"]["action"]["criteria"]["option0"]["description"]
    )
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


def test_truncated_public_label_is_not_completed_from_description_or_private_arguments():
    encoded = body()
    encoded["state"]["state"]["controls"][0]["label"] = "Observed"
    # This private-looking literal must neither be read nor exposed as a reconstruction source.
    encoded["questions"]["action"]["criteria"]["option0"]["arguments"] = {"bound_control": "do not use"}
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"] == (
        before["questions"]["action"]["criteria"]["option0"]
    )
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


def test_prefix_suffix_unicode_whitespace_quotes_and_ref_looking_literals_are_exact():
    encoded = body()
    record = encoded["questions"]["action"]["criteria"]["option0"]
    record["description"] = (
        '前缀\t"{control_ref:c9}"\n' + _anchor(encoded["state"]["state"]["controls"][0]) + '  后缀\r\n'
    )
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


def test_multiple_anchors_including_one_repeated_cannot_select_an_arbitrary_control():
    encoded = body()
    controls = encoded["state"]["state"]["controls"]
    literal = _anchor(controls[0]) * 2 + _anchor(controls[1])
    encoded["questions"]["action"]["criteria"]["option0"]["description"] = literal
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"]["description"] == literal


def test_explicit_local_mask_never_rewrites_defaults_or_unmarked_ref_looking_dict():
    encoded = body()
    encoding = encoded["state"]["encoding"]
    encoding["candidate_defaults"]["description"] = {"control_ref": "c0", "template_id": 0}
    record = encoded["questions"]["action"]["criteria"]["option0"]
    record["description"] = {"control_ref": "c0", "template_id": 0}
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)
    assert "option0" not in encoded["state"]["encoding"]["control_description_candidates"]


def test_v4_defaults_and_marked_ancestor_pool_overlay_preserve_all_facts():
    encoded = body()
    encoding = encoded["state"]["encoding"]
    controls = encoded["state"]["state"]["controls"]
    defaults = {key: controls[0][key] for key in ("role", "app", "window", "ancestors", "checked")}
    encoding["control_defaults"] = copy.deepcopy(defaults)
    for control in controls:
        for key in defaults:
            del control[key]
    encoding["ancestor_paths"] = {
        "p0": {"parent": None, "text": defaults["ancestors"][0]},
        "p1": {"parent": "p0", "text": defaults["ancestors"][1]},
    }
    encoding["ancestor_path_default"] = True
    encoding["ancestor_path_controls"] = []
    encoding["control_defaults"]["ancestors"] = {"path_ref": "p1"}
    # Explicit override beats the default; it is not silently repaired from the template.
    controls[0]["ancestors"] = []
    controls[1]["checked"] = None
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert type(encoded["questions"]["action"]["criteria"]["option0"]["description"]) is str
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


@pytest.mark.parametrize("problem", ["cycle", "bool_index", "unknown", "partial_mask"])
def test_invalid_v4_ancestor_metadata_declines_without_mutation(problem):
    encoded = body()
    encoding = encoded["state"]["encoding"]
    encoding.update(ancestor_paths={"p": {"parent": None, "text": "public"}},
                    ancestor_path_default=False, ancestor_path_controls=[0])
    encoded["state"]["state"]["controls"][0]["ancestors"] = {"path_ref": "p"}
    if problem == "cycle":
        encoding["ancestor_paths"]["p"]["parent"] = "p"
    elif problem == "bool_index":
        encoding["ancestor_path_controls"] = [False]
    elif problem == "unknown":
        encoded["state"]["state"]["controls"][0]["ancestors"]["path_ref"] = "missing"
    else:
        del encoding["ancestor_path_default"]
    before = wire(encoded)
    assert share_public_control_descriptions(encoded) is False and wire(encoded) == before


def test_hints_are_exact_short_public_labels_and_decoder_rejects_tampering():
    encoded = body()
    assert share_public_control_descriptions(encoded)
    envelope = encoded["questions"]["action"]["criteria"]["option0"]["description"]
    assert envelope["label_hint"] == "Observed widget 0"
    assert len(envelope["label_hint"].encode()) <= MAX_HINT_BYTES
    envelope["label_hint"] = "not the public label"
    with pytest.raises(ValueError):
        restore_public_control_descriptions(encoded)


def test_long_public_label_has_no_hint_and_is_never_shortened_for_matching():
    encoded = body()
    control = encoded["state"]["state"]["controls"][0]
    control["label"] = "Exact public label " + "界" * 40
    encoded["questions"]["action"]["criteria"]["option0"]["description"] = (
        "click " + _anchor(control) + " literal suffix"
    )
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert "label_hint" not in encoded["questions"]["action"]["criteria"]["option0"]["description"]
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


@pytest.mark.parametrize("value", [False, -1, 999, "0"])
def test_template_index_is_an_exact_bounded_integer(value):
    encoded = body()
    assert share_public_control_descriptions(encoded)
    encoded["questions"]["action"]["criteria"]["option0"]["description"]["template_id"] = value
    with pytest.raises(ValueError):
        restore_public_control_descriptions(encoded)


@pytest.mark.parametrize(
    "drift", ["mask_order", "mask_unknown", "extra_envelope", "instruction", "unused_template"]
)
def test_decoder_rejects_ambiguous_or_partial_protocol_envelopes(drift):
    encoded = body()
    assert share_public_control_descriptions(encoded)
    encoding = encoded["state"]["encoding"]
    if drift == "mask_order":
        encoding["control_description_candidates"].reverse()
    elif drift == "mask_unknown":
        encoding["control_description_candidates"].append("unknown-option")
    elif drift == "extra_envelope":
        encoded["questions"]["action"]["criteria"]["option0"]["description"]["extra"] = None
    elif drift == "instruction":
        encoded["questions"]["action"]["instructions"] += " extra"
    else:
        encoding["control_description_templates"].append({"prefix": "unused", "suffix": ""})
    with pytest.raises(ValueError):
        restore_public_control_descriptions(encoded)


@pytest.mark.parametrize("threshold", [0, 10000000])
def test_no_full_body_profit_leaves_request_and_nested_identity_untouched(threshold):
    encoded = body(1) if threshold == 0 else body()
    controls = encoded["state"]["state"]["controls"]
    criteria = encoded["questions"]["action"]["criteria"]
    before = wire(encoded)
    assert share_public_control_descriptions(encoded, minimum_savings=threshold) is False
    assert wire(encoded) == before
    assert encoded["state"]["state"]["controls"] is controls
    assert encoded["questions"]["action"]["criteria"] is criteria


def test_v5_or_existing_own_markers_are_not_stacked_and_never_mutate():
    for drift in ("version", "marker"):
        encoded = body()
        if drift == "version":
            encoded["state"]["encoding"]["version"] = "shared_fields_v5"
        else:
            encoded["state"]["encoding"]["control_description_candidates"] = []
        before = wire(encoded)
        with pytest.raises(ValueError):
            share_public_control_descriptions(encoded)
        assert wire(encoded) == before


def test_nonfinite_value_rejects_before_any_mutation():
    encoded = body()
    encoded["state"]["state"]["numeric"] = float("nan")
    controls = encoded["state"]["state"]["controls"]
    with pytest.raises(ValueError):
        share_public_control_descriptions(encoded)
    assert encoded["state"]["state"]["controls"] is controls
    assert encoded["state"]["encoding"]["version"] == "shared_fields_v4"


def test_empty_public_window_with_a_nonempty_original_anchor_stays_literal_until_exact_match():
    encoded = body()
    original = encoded["questions"]["action"]["criteria"]["option0"]["description"]
    encoded["state"]["state"]["controls"][0]["window"] = ""
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"]["description"] == original
    assert "option0" not in encoded["state"]["encoding"]["control_description_candidates"]
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)
