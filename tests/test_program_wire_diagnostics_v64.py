"""New wire rejection reasons stay bounded and do not serialize model values."""

import copy
import json

import pytest
from test_semantic_tools_v57 import observed as observed
from test_semantic_tools_v57 import packet
from test_stage_program_contract_v60 import declaration
from test_stage_program_planning_preference_v61 import requested

from cua_jev import desktop_semantic_tools_v64 as wire_codec
from cua_jev.desktop_declaration_rejection import declaration_rejection_facts

PRIVATE = "UNTRUSTED_FIELD_VALUE_NOT_FOR_LOGS_6ad4"


@pytest.fixture
def offered(observed):
    wire = copy.deepcopy(declaration())
    for member in wire["stages"]:
        member["arguments"].pop("completion")
    _plan, request, _planner = requested(
        observed, wire, "declare_stage_program", fmt="semantic_tools_v64",
    )
    return json.loads(request.content)["tools"]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("key,value", [
    ("completion", "stage"), ("completion", "goal"), ("completion", None),
    ("goal_conditions", []), ("goal_conditions", None),
])
def test_explicit_member_control_fields_are_rejected_and_diagnosed_without_values(offered, index, key, value):
    wire = copy.deepcopy(declaration())
    wire["label"] = PRIVATE
    for member in wire["stages"]:
        member["arguments"].pop("completion")
        member["arguments"]["label"] = PRIVATE
    wire["stages"][index]["arguments"][key] = value
    payload = packet(wire, "declare_stage_program", content=PRIVATE)
    before = copy.deepcopy(payload)
    code = "program_wire_forbids_member_" + key
    with pytest.raises(ValueError, match="^" + code + "$") as failure:
        wire_codec.decode_semantic_declaration(payload, max_actions=6)
    facts = declaration_rejection_facts(payload, offered_tools=offered, error=failure.value)
    assert facts["stage"] == "program_wire" and facts["code"] == code
    assert facts["input_authority"] is False and facts["schema_validated"] is False
    assert facts["program_members"]["members"][index][key]["present"] is True
    assert PRIVATE not in json.dumps(facts)
    assert payload == before


@pytest.mark.parametrize("code", [
    "program_wire_arguments_must_be_object", "program_wire_requires_original_stage_program",
    "program_wire_codec_version", "program_wire_provenance_mismatch",
    "program_wire_normalized_parser_input_mismatch",
])
def test_provenance_failure_codes_do_not_copy_exception_details(code):
    facts = declaration_rejection_facts({}, offered_tools=None, error=ValueError(code))
    assert facts["stage"] == "program_wire" and facts["code"] == code
    unknown = declaration_rejection_facts({}, offered_tools=None, error=ValueError(code + PRIVATE))
    assert unknown["code"] == "adapter_value_error"
    assert PRIVATE not in json.dumps(unknown)
