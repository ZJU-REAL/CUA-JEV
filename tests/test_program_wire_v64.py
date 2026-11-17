"""The v64 member wire normalizes fixed syntax without extending authority."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_semantic_tools_v57 import current, packet
from test_semantic_tools_v57 import observed as observed
from test_stage_program_contract_v60 import declaration, parse_original, query_panel
from test_stage_program_planning_preference_v61 import requested

from cua_jev import desktop_semantic_tools_v62 as previous
from cua_jev import desktop_semantic_tools_v64 as new
from cua_jev.desktop_phase_handoff_v62 import HighLevelContextRequest
from cua_jev.desktop_program_wire_v64 import (
    CODEC_VERSION,
    SOURCE_MODULES,
    bind_program_provenance,
    canonical_json,
)
from cua_jev.desktop_semantic_tools_v60 import decode_semantic_declaration as decode60
from cua_jev.desktop_stage_program import activate_program_stage, parse_stage_program

TOOL = "declare_stage_program"


def wire_declaration():
    raw = copy.deepcopy(declaration())
    for member in raw["stages"]:
        del member["arguments"]["completion"]
    return raw


def actual_input(observed, monkeypatch, *, limit=6, formal=False):
    captured = []
    original = previous.encode_semantic_declaration

    def capture(body, **options):
        captured.append((copy.deepcopy(body), copy.deepcopy(options)))
        return original(body, **options)

    raw, tool = declaration(), TOOL
    if limit == 1:
        raw, tool = current(1), "declare_native_current_action"
    elif formal:
        raw, tool = {"label": "Observed editor", "conditions": [
            {"kind": "text_contains", "value": "Editor"},
        ]}, "declare_completion"
    with monkeypatch.context() as patch:
        patch.setattr(previous, "encode_semantic_declaration", capture)
        requested(observed, raw, tool, fmt=previous.FORMAT, limit=limit,
                  planning_options={"whole_goal_recheck": True} if formal else {})
    assert len(captured) == 1
    return captured[0]


@pytest.fixture
def offered(observed, monkeypatch):
    body, options = actual_input(observed, monkeypatch)
    old, _ = previous.encode_semantic_declaration(body, **options)
    actual, provenance = new.encode_semantic_declaration(body, **options)
    return old, actual, provenance


def test_normal_schema_only_changes_program_member_wire_and_local_guidance(offered):
    old, actual, provenance = offered

    # v68 corrects only normal-H6 property documentation. Derive that expected
    # exception independently; every other tool and request field stays exact.
    def expected_annotations(value):
        if isinstance(value, dict):
            return {
                key: (item.replace(new.LEGACY_PROPERTY_OPERATIONS, new.NORMAL_PROPERTY_OPERATIONS)
                      if key == "description" and isinstance(item, str) else expected_annotations(item))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [expected_annotations(item) for item in value]
        return value

    annotated_old = copy.deepcopy(old)
    annotated_old["tools"] = expected_annotations(annotated_old["tools"])
    assert actual["tools"][1:] == annotated_old["tools"][1:]
    assert len(actual["tools"]) == 6
    restored = copy.deepcopy(actual)
    restored["tools"][0] = annotated_old["tools"][0]
    restored["messages"][0]["content"] = annotated_old["messages"][0]["content"]
    assert restored == annotated_old  # Model, image, history and remaining request fields stay exact.
    schema = actual["tools"][0]["function"]["parameters"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(wire_declaration())
    for name, member in schema["$defs"].items():
        if name.startswith("program_stage_arguments_"):
            assert "completion" not in member["properties"]
            assert "completion" not in member["required"]
            assert "goal_conditions" not in member["properties"]
            assert member["additionalProperties"] is False
    system = actual["messages"][0]["content"]
    assert previous.PROGRAM_MEMBER_GUIDANCE not in system
    assert new.OLD_PROGRAM_GUIDANCE not in system
    assert "FIRST_KNOWN_RESULT" not in system and "completion=stage, no goal_conditions" not in system
    assert "EVERY member" in system and "already visible entry control" in system
    assert provenance["response_transport"] == "semantic_tools_v64"
    assert provenance["program_wire_codec_version"] == CODEC_VERSION


@pytest.mark.parametrize("index", [0, 1], ids=["first", "last"])
@pytest.mark.parametrize("key,value", [
    ("completion", "stage"), ("completion", "goal"), ("completion", None),
    ("completion", ""), ("goal_conditions", []), ("goal_conditions", None),
    ("goal_conditions", {}), ("goal_conditions", ""),
])
def test_explicit_member_fields_are_rejected_even_when_stage_empty_or_null(offered, index, key, value):
    raw = wire_declaration()
    raw["stages"][index]["arguments"][key] = value
    before = copy.deepcopy(raw)
    schema = offered[1]["tools"][0]["function"]["parameters"]
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(raw)
    code = "completion" if key == "completion" else "goal_conditions"
    with pytest.raises(ValueError, match="^program_wire_forbids_member_" + code + "$"):
        new.decode_semantic_declaration(packet(raw, TOOL), max_actions=6)
    assert raw == before


def test_valid_wire_runs_original_decode_and_has_recomputable_separate_provenance():
    raw = wire_declaration()
    payload = packet(raw, TOOL)
    original_argument_text = payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
    before = copy.deepcopy(payload)
    mapped, provenance = new.decode_semantic_declaration(payload, max_actions=6)
    original, _ = decode60(packet(declaration(), TOOL), max_actions=6)
    assert mapped == original and payload == before
    wire_json = canonical_json(raw)
    normalized_json = canonical_json(declaration())
    assert provenance["wire_declaration_json"] == wire_json
    assert provenance["normalized_declaration_json"] == normalized_json
    for key, text in (("wire", wire_json), ("normalized", normalized_json)):
        assert provenance[key + "_declaration_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    original_hash = hashlib.sha256(original_argument_text.encode()).hexdigest()
    assert provenance["argument_sha256"] == provenance["observed_wire_argument_sha256"] == original_hash
    assert provenance["normalized_argument_sha256"] == provenance["normalized_declaration_sha256"]
    assert original_hash != provenance["wire_declaration_sha256"]  # Original text was not canonical.
    assert provenance["program_declaration_sha256"] == provenance["normalized_declaration_sha256"]
    assert provenance["wire_stage_mapping"] == [
        {"index": index, "id": member["id"], "tool": member["tool"]}
        for index, member in enumerate(raw["stages"])
    ]
    assert set(provenance["codec_source_hashes"]) == {"src/cua_jev/" + name for name in SOURCE_MODULES}
    for name in SOURCE_MODULES:
        actual_hash = hashlib.sha256(Path(new.__file__).with_name(name).read_bytes()).hexdigest()
        assert provenance["codec_source_hashes"]["src/cua_jev/" + name] == actual_hash
    program = parse_stage_program({key: value for key, value in mapped.items() if key != "intent"},
                                  max_actions=6)
    bound = bind_program_provenance(program, provenance)
    emitted = bound.to_dict()
    assert bound.declaration_json == normalized_json and bound.stages == program.stages
    assert emitted["source_declaration_status"] == "codec_normalized_canonical_json"
    assert "original_canonical_json" not in json.dumps(emitted)
    for key in ("wire_declaration_json", "wire_declaration_sha256", "normalized_declaration_json",
                "normalized_declaration_sha256", "wire_stage_mapping", "codec_version",
                "codec_source_hashes"):
        assert emitted[key] == provenance[key]


@pytest.mark.parametrize("key", [
    "wire_declaration_sha256", "normalized_declaration_json", "normalized_declaration_sha256",
    "wire_stage_mapping", "codec_source_hashes",
])
def test_provenance_binding_rejects_tampering(key):
    _mapped, provenance = new.decode_semantic_declaration(packet(wire_declaration(), TOOL), max_actions=6)
    program = parse_stage_program(declaration(), max_actions=6)
    altered = copy.deepcopy(provenance)
    altered[key] = "tampered"
    with pytest.raises(ValueError, match="^program_wire_provenance_mismatch$"):
        bind_program_provenance(program, altered)


def test_binding_cannot_attach_valid_wire_to_another_normalized_program():
    _mapped, provenance = new.decode_semantic_declaration(packet(wire_declaration(), TOOL), max_actions=6)
    raw = declaration()
    raw["label"] = "Different declaration"
    with pytest.raises(ValueError, match="^program_wire_normalized_parser_input_mismatch$"):
        bind_program_provenance(parse_stage_program(raw, max_actions=6), provenance)


@pytest.mark.parametrize("damage", ["future_ref", "duplicate_id", "current_tool", "empty_arrival"])
def test_normalization_keeps_original_semantic_and_future_authority_guards(damage):
    raw = wire_declaration()
    if damage == "future_ref":
        raw["stages"][1]["arguments"]["edits"][0]["target"]["ref"] = "invented-ref"
    elif damage == "duplicate_id":
        raw["stages"][1]["id"] = raw["stages"][0]["id"]
    elif damage == "current_tool":
        raw["stages"][1]["tool"] = "declare_native_current_action"
    else:
        raw["stages"][0]["arguments"]["arrival"] = []
    normalized = copy.deepcopy(raw)
    for member in normalized["stages"]:
        member["arguments"]["completion"] = "stage"
    with pytest.raises(ValueError) as old_failure:
        decode60(packet(normalized, TOOL), max_actions=6)
    with pytest.raises(ValueError) as new_failure:
        new.decode_semantic_declaration(packet(raw, TOOL), max_actions=6)
    assert str(new_failure.value) == str(old_failure.value)


def test_original_fresh_activation_guards_still_reject_unavailable_future_field(observed):
    mapped, provenance = new.decode_semantic_declaration(packet(wire_declaration(), TOOL), max_actions=6)
    program = bind_program_provenance(parse_stage_program(
        {key: value for key, value in mapped.items() if key != "intent"}, max_actions=6,
    ), provenance)
    initial = activate_program_stage(program, 0, observed, remaining_actions=6, parse_stage=parse_original)
    assert initial.reusable and initial.steps[0].navigation
    with pytest.raises(ValueError):
        activate_program_stage(program, 1, observed, remaining_actions=5, parse_stage=parse_original)
    current_frame = query_panel(observed)
    plan = activate_program_stage(program, 1, current_frame, remaining_actions=5, parse_stage=parse_original)
    assert plan.steps[0].value == "requested literal" and not plan.complete_property_goal
    with pytest.raises(ValueError, match="program_activation_requires_complete_current_observation"):
        activate_program_stage(program, 1, replace(current_frame, a11y_complete=False),
                               remaining_actions=5, parse_stage=parse_original)


@pytest.mark.parametrize("limit,formal", [(1, False), (1, True), (6, True)])
def test_H1_and_formal_requests_are_exact_previous_ancestor58(observed, monkeypatch, limit, formal):
    body, options = actual_input(observed, monkeypatch, limit=limit, formal=formal)
    old, _ = previous.encode_semantic_declaration(body, **options)
    actual, _ = new.encode_semantic_declaration(body, **options)
    assert actual == old
    assert new.PROGRAM_GUIDANCE not in actual["messages"][0]["content"]
    raw = current(1) if limit == 1 else {
        "label": "Observed editor", "conditions": [{"kind": "text_contains", "value": "Editor"}],
    }
    tool = "declare_native_current_action" if limit == 1 else "declare_completion"
    expected, _ = previous.decode_semantic_declaration(packet(raw, tool), max_actions=limit,
                                                      normal_program_eligible=options["normal_program_eligible"])
    mapped, _ = new.decode_semantic_declaration(packet(raw, tool), max_actions=limit,
                                               normal_program_eligible=options["normal_program_eligible"])
    assert mapped == expected


def test_single_property_prepare_and_context_keep_original_mapping():
    raw = {"label": "Complete requested value", "scope": {"app": "Editor"},
           "edit_operation": "fill", "edit_value": "requested literal", "numeric_finalize_key": "none",
           "confirmation_label": "none", "completion": "goal",
           "prepare": {"operation": "click", "ref": "help"}}
    payload = packet(raw, "declare_property_stage")
    assert new.decode_semantic_declaration(payload, max_actions=6)[0] == (
        previous.decode_semantic_declaration(payload, max_actions=6)[0]
    )
    context = packet({"reason": "missing_future_semantics"}, "declare_current_context_needed")
    mapped, transport = new.decode_semantic_declaration(context, max_actions=6)
    assert isinstance(mapped, HighLevelContextRequest) and transport["context_request"]
    assert transport["mapping_applied"] is False and transport["input_authority"] is False


def test_original_program_output_dict_remains_exact_without_v64_binding():
    raw = declaration()
    program = parse_stage_program(raw, max_actions=6)
    assert program.to_dict() == {
        "label": program.summary, "stages": [stage.to_dict() for stage in program.stages],
        "max_actions": 6, "declaration_sha256": program.declaration_sha256,
        "source_declaration": raw, "source_declaration_status": "original_canonical_json",
        "final_review": "original_fresh_whole_goal_review_required",
        "future_stages_are_uncompiled_declarations": True,
    }


@pytest.mark.parametrize("arguments,error", [
    ('{"label":"one","label":"two","stages":[]}', "duplicate program JSON field"),
    ('{"label":NaN,"stages":[]}', "invalid declaration argument JSON"),
    ('{"label":1e9999,"stages":[]}', "invalid declaration argument JSON"),
])
def test_duplicate_and_nonfinite_json_never_reaches_normalized_program(arguments, error):
    payload = packet(wire_declaration(), TOOL)
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    with pytest.raises(ValueError, match="^" + error + "$"):
        new.decode_semantic_declaration(payload, max_actions=6)
