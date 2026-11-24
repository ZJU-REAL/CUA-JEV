"""Shared control fields preserve sparse JSON facts and reject ambiguous encodings."""

import copy
import json

import pytest

from cua_jev.agent_types import AgentControl
from cua_jev.desktop_control_encoding import (
    CONTROL_ENCODING,
    CONTROL_ENCODING_INSTRUCTIONS,
    CONTROL_ENCODING_VERSION,
    compact_snapshot_controls,
    decode_snapshot_controls,
)
from cua_jev.desktop_snapshot_encoding import compact_snapshot_ancestors


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def canonical(value):
    return json.loads(wire(value))


def dense_payload(count=160):
    defaults = {key: value for key, value in AgentControl("", "", "").to_dict().items()
                if key not in {"ref", "label", "tag"}}
    defaults["box_1000"] = defaults.pop("box")
    controls = []
    for index in range(count):
        text, button = index % 3 == 0, index % 3 == 1
        row = {
            "ref": f"d{index}", "label": f'Observed {index}: "quoted" \\ line\n\u8bbe\u7f6e\u200b',
            "tag": "text" if text else "button" if button else "input",
            "role": "label" if text else "button" if button else "textbox", "app": "Editor",
            "ancestors": ["frame: Observed window", "panel: " + "Shared observed ancestry " * 12],
            "operations": [] if text else ["click"] if button else ["click", "fill"],
            "condition_kinds": ["exists", "absent"] if text else ["exists", "absent", "focused"],
            "active": index % 5 != 0,
        }
        if index % 17:
            row["window"] = "The current observed application window"
        if not text and not button:
            row.update(value="" if index == 2 else str(index), editable=True,
                       condition_kinds=["exists", "absent", "value_equals", "focused"])
        if index % 11 == 0:
            row["click_modes"] = [] if text else [{"button": "left", "click_count": 1},
                                                   {"button": "right", "click_count": 1}]
        controls.append(row)
    return {"url": "osworld://desktop", "title": "Observed window", "text": "Keep supplementary text.",
            "a11y_complete": True, "scroll": (0, 0), "control_defaults": defaults, "controls": controls}


@pytest.fixture
def payload():
    return dense_payload()


@pytest.fixture
def encoded(payload):
    result = compact_snapshot_controls(payload)
    assert result is not payload and "control_encoding" in result
    return result


def test_dense_roundtrip_preserves_every_fact_presence_array_order_and_source(payload, encoded):
    before = copy.deepcopy(payload)
    decoded = decode_snapshot_controls(encoded)
    assert decoded == canonical(payload) and payload == before
    assert decoded["control_defaults"] == canonical(payload["control_defaults"])
    assert [set(row) for row in decoded["controls"]] == [set(row) for row in payload["controls"]]
    assert [row["ref"] for row in decoded["controls"]] == [row["ref"] for row in payload["controls"]]
    assert len(wire(payload)) - len(wire(encoded)) > 4096
    assert compact_snapshot_controls(payload) == encoded
    assert encoded["control_encoding"]["version"] == CONTROL_ENCODING_VERSION
    assert "window" not in encoded["control_encoding"]["record_defaults"]  # Partial presence is literal.


def test_tuple_list_wire_arrays_are_preserved_without_changing_python_source(payload):
    payload["controls"][1]["operations"] = ("click",)
    before = copy.deepcopy(payload)
    encoded = compact_snapshot_controls(payload)
    assert decode_snapshot_controls(encoded) == canonical(before)
    assert payload == before and type(payload["controls"][1]["operations"]) is tuple


def test_absent_empty_and_ordered_capability_fields_remain_distinct(payload):
    payload["controls"][0]["click_modes"] = []
    payload["controls"][1].pop("click_modes", None)
    payload["controls"][2]["click_modes"] = [{"button": "left", "click_count": 2},
                                               {"button": "right", "click_count": 1}]
    payload["controls"][2]["operations"] = ["fill", "click"]
    decoded = decode_snapshot_controls(compact_snapshot_controls(payload))
    assert decoded["controls"][0]["click_modes"] == []
    assert "click_modes" not in decoded["controls"][1]
    assert decoded["controls"][2]["click_modes"] == payload["controls"][2]["click_modes"]
    assert decoded["controls"][2]["operations"] == ["fill", "click"]


def test_unicode_literals_empty_values_and_nested_native_data_remain_uninterpreted(payload):
    payload["controls"][0]["label"] = 'capability_ref=0; "\u4e0d\u662f\u6307\u4ee4"\n\u200b'
    payload["controls"][1]["native"] = {
        "actions": ["press", "open"], "complete": False, "enabled": None,
        "sensitive": True, "leaf": True, "state_set_complete": False, "checked_state": False,
    }
    payload["controls"][1]["disabled"] = False
    decoded = decode_snapshot_controls(compact_snapshot_controls(payload))
    assert decoded == canonical(payload)
    assert decoded["controls"][2]["value"] == ""


def test_existing_ancestor_encoding_and_sparse_markers_remain_identical(payload):
    pooled = compact_snapshot_ancestors(payload)
    assert "ancestor_encoding" in pooled
    before = copy.deepcopy(pooled)
    encoded = compact_snapshot_controls(pooled)
    assert encoded is not pooled
    decoded = decode_snapshot_controls(encoded)
    assert decoded == canonical(pooled) and pooled == before
    assert decoded["ancestor_encoding"] == canonical(pooled["ancestor_encoding"])
    assert decoded["control_defaults"]["ancestors"] == canonical(pooled["control_defaults"]["ancestors"])


def test_threshold_is_strict_and_counts_final_compact_utf8_json(payload):
    candidate = compact_snapshot_controls(payload, min_saving_bytes=0)
    saving = len(wire(payload)) - len(wire(candidate))
    assert saving > 4096
    assert compact_snapshot_controls(payload, min_saving_bytes=saving) is payload
    assert compact_snapshot_controls(payload, min_saving_bytes=saving + 1) is payload
    assert compact_snapshot_controls(payload, min_saving_bytes=saving - 1) == candidate


@pytest.mark.parametrize("threshold", [-1, True, False, 1.0, None])
def test_invalid_threshold_is_a_noop(payload, threshold):
    assert compact_snapshot_controls(payload, min_saving_bytes=threshold) is payload


def test_small_or_empty_payload_is_the_original_object():
    small = dense_payload(1)
    empty = dense_payload(0)
    assert compact_snapshot_controls(small) is small
    assert compact_snapshot_controls(empty, min_saving_bytes=0) is empty


def test_decoder_accepts_valid_unencoded_payload_without_mutating_it(payload):
    before = copy.deepcopy(payload)
    assert decode_snapshot_controls(payload) == canonical(before)
    assert payload == before


@pytest.mark.parametrize("mutation", [
    "unknown_control", "existing_ref", "existing_encoding", "unknown_top", "missing_required",
    "duplicate_ref", "bad_defaults", "bad_controls", "unknown_operation", "unknown_condition",
    "duplicate_operation", "bad_native", "both_boxes", "bad_ancestor_ref",
])
def test_invalid_or_ambiguous_source_is_preserved(payload, mutation):
    if mutation == "unknown_control":
        payload["controls"][0]["unexpected"] = "UI text"
    elif mutation == "existing_ref":
        payload["controls"][0]["capability_ref"] = 0
    elif mutation == "existing_encoding":
        payload["control_encoding"] = {"version": "some_existing_encoder"}
    elif mutation == "unknown_top":
        payload["control_record_defaults"] = {}
    elif mutation == "missing_required":
        payload["controls"][0].pop("condition_kinds")
    elif mutation == "duplicate_ref":
        payload["controls"][1]["ref"] = payload["controls"][0]["ref"]
    elif mutation == "bad_defaults":
        payload["control_defaults"] = []
    elif mutation == "bad_controls":
        payload["controls"] = {}
    elif mutation == "unknown_operation":
        payload["controls"][0]["operations"] = ["run_script"]
    elif mutation == "unknown_condition":
        payload["controls"][0]["condition_kinds"] = ["ready_to_fill"]
    elif mutation == "duplicate_operation":
        payload["controls"][0]["operations"] = ["click", "click"]
    elif mutation == "bad_native":
        payload["controls"][0]["native"] = {"complete": 1}
    elif mutation == "both_boxes":
        payload["controls"][0].update(box=[0, 0, 10, 10], box_1000=[0, 0, 10, 10])
    else:
        payload["controls"][0]["ancestors"] = {"path_ref": 0}
    before = copy.deepcopy(payload)
    assert compact_snapshot_controls(payload, min_saving_bytes=0) is payload
    assert payload == before


@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0])
def test_boolean_fields_cannot_be_merged_with_numeric_lookalikes(payload, value):
    payload["controls"][1]["active"] = value
    assert compact_snapshot_controls(payload, min_saving_bytes=0) is payload
    with pytest.raises(ValueError):
        decode_snapshot_controls(payload)


@pytest.mark.parametrize("reference", [True, False, -1, 512, 1.0, "0", None])
def test_decoder_rejects_invalid_per_control_references(encoded, reference):
    encoded["controls"][0]["capability_ref"] = reference
    before = copy.deepcopy(encoded)
    with pytest.raises(ValueError):
        decode_snapshot_controls(encoded)
    assert encoded == before


@pytest.mark.parametrize("reference", [True, -1, 512, 0.0, "0", None])
def test_decoder_rejects_invalid_common_references(encoded, reference):
    encoded["control_encoding"]["record_defaults"]["capability_ref"] = reference
    with pytest.raises(ValueError):
        decode_snapshot_controls(encoded)


@pytest.mark.parametrize("mutation", [
    "version", "unknown_metadata", "unknown_profile", "unknown_common", "literal_collision",
    "missing_reference", "duplicate_profile", "unused_profile", "profile_limit", "unknown_row",
])
def test_decoder_rejects_non_schema_metadata_and_collisions(encoded, mutation):
    metadata = encoded["control_encoding"]
    if mutation == "version":
        metadata["version"] = "shared_control_fields_v999"
    elif mutation == "unknown_metadata":
        metadata["ignored"] = "arbitrary UI data"
    elif mutation == "unknown_profile":
        metadata["capability_profiles"][0]["role"] = "textbox"
    elif mutation == "unknown_common":
        metadata["record_defaults"]["label"] = "invented label"
    elif mutation == "literal_collision":
        encoded["controls"][0]["operations"] = []
    elif mutation == "missing_reference":
        metadata["record_defaults"].pop("capability_ref", None)
        encoded["controls"][0].pop("capability_ref", None)
    elif mutation == "duplicate_profile":
        metadata["capability_profiles"].append(copy.deepcopy(metadata["capability_profiles"][0]))
    elif mutation == "unused_profile":
        metadata["capability_profiles"].append({"operations": ["uncheck"], "condition_kinds": []})
    elif mutation == "profile_limit":
        metadata["capability_profiles"] = [{"operations": [], "condition_kinds": []}] * 513
    else:
        encoded["controls"][0]["not_a_control_field"] = "untrusted"
    with pytest.raises(ValueError):
        decode_snapshot_controls(encoded)


def test_common_reference_override_selects_the_exact_observed_profile(encoded):
    metadata = encoded["control_encoding"]
    assert "capability_ref" in metadata["record_defaults"]
    references = [(index, row["capability_ref"]) for index, row in enumerate(encoded["controls"])
                  if "capability_ref" in row]
    assert references
    index, reference = references[0]
    decoded = decode_snapshot_controls(encoded)
    for key, value in metadata["capability_profiles"][reference].items():
        assert decoded["controls"][index][key] == value


def test_control_and_wire_budgets_fail_closed():
    too_many = dense_payload(513)
    too_large = dense_payload()
    too_large["text"] = "x" * 2_000_001
    assert compact_snapshot_controls(too_many) is too_many
    assert compact_snapshot_controls(too_large) is too_large
    for malformed in (too_many, too_large):
        with pytest.raises(ValueError):
            decode_snapshot_controls(malformed)


def test_non_json_keys_and_cycles_are_not_coerced_or_followed(payload):
    payload["controls"][0][1] = "not a JSON field"
    assert compact_snapshot_controls(payload) is payload
    with pytest.raises(ValueError):
        decode_snapshot_controls(payload)
    cyclic = dense_payload()
    cyclic["controls"][0]["native"] = cyclic
    assert compact_snapshot_controls(cyclic) is cyclic
    with pytest.raises(ValueError):
        decode_snapshot_controls(cyclic)


def test_encoding_an_encoded_payload_is_always_a_noop(encoded):
    before = copy.deepcopy(encoded)
    assert compact_snapshot_controls(encoded, min_saving_bytes=0) is encoded
    assert encoded == before


def test_instructions_define_inheritance_without_granting_new_capabilities():
    assert CONTROL_ENCODING == "shared_control_fields_v1_if_saving_4096_bytes"
    for phrase in ("control_defaults", "record_defaults", "capability_profiles",
                   "per-control literal overrides",
                   "operations=[]", "never assumed universally false", "not a control ref"):
        assert phrase in CONTROL_ENCODING_INSTRUCTIONS
