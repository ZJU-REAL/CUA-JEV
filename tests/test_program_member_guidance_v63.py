"""Clarify the existing stage-only member rule without changing any guard."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_semantic_tools_v57 import current, packet
from test_semantic_tools_v57 import observed as observed
from test_stage_program_contract_v60 import declaration
from test_stage_program_planning_preference_v61 import requested

from cua_jev import desktop_declaration_rejection as rejection
from cua_jev.desktop_semantic_tools_v62 import PROGRAM_MEMBER_GUIDANCE, decode_semantic_declaration

FORMAT = "semantic_tools_v62"
TOOL = "declare_stage_program"
PRIVATE = "PRIVATE_MEMBER_LABEL_REF_VALUE_NO_LEAK_8a19"


@pytest.fixture
def offered(observed):
    _plan, request, _planner = requested(observed, declaration(), TOOL, fmt=FORMAT)
    return json.loads(request.content)


@pytest.mark.parametrize("index", [0, 1], ids=["first", "last"])
@pytest.mark.parametrize("change", [
    "missing", "goal", "other", "nonstring", "arguments_string", "empty_goal",
])
def test_original_member_guard_and_safe_diagnostic_distinguish_failures(offered, index, change):
    raw = declaration()
    raw["label"] = PRIVATE
    for member in raw["stages"]:
        member["id"] = PRIVATE + member["id"]
        member["arguments"]["label"] = PRIVATE
    arguments = raw["stages"][index]["arguments"]
    if change == "missing":
        del arguments["completion"]
    elif change == "arguments_string":
        raw["stages"][index]["arguments"] = PRIVATE
    elif change == "empty_goal":
        arguments["goal_conditions"] = []
    else:
        arguments["completion"] = {"goal": "goal", "other": PRIVATE, "nonstring": {PRIVATE: PRIVATE}}[change]
    payload = packet(raw, TOOL, content=PRIVATE)
    payload["headers"] = {"Authorization": PRIVATE}
    before = copy.deepcopy(payload)
    schema = next(item["function"]["parameters"] for item in offered["tools"]
                  if item["function"]["name"] == TOOL)
    # It was already illegal in the wire grammar; the guard is not relaxed.
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(raw)
    with pytest.raises(ValueError, match="^program_members_cannot_claim_whole_goal$") as failure:
        decode_semantic_declaration(payload, max_actions=6)
    facts = rejection.declaration_rejection_facts(
        payload, offered_tools=offered["tools"], error=failure.value,
    )
    member = facts["program_members"]["members"][index]
    assert member["index"] == index and facts["code"] == "program_members_cannot_claim_whole_goal"
    assert facts["stage"] == "program_structure" and facts["input_authority"] is False
    if change == "arguments_string":
        assert member["arguments"]["type"] == "string"
        assert member["inspection"] == "arguments_not_object" and "completion" not in member
    else:
        assert member["completion"]["enum"] == {
            "missing": "missing", "goal": "goal", "other": "other", "nonstring": "other",
            "empty_goal": "stage",
        }[change]
        assert member["completion"]["present"] is (change != "missing")
        assert member["goal_conditions"]["present"] is (change == "empty_goal")
    rendered = json.dumps(facts)
    assert PRIVATE not in rendered and "Authorization" not in rendered and "sha256" not in rendered
    assert facts["schema_validated"] is False and payload == before


def test_member_facts_are_bounded_and_do_not_visit_unknown_member_values(offered):
    raw = declaration()
    raw["stages"] = [copy.deepcopy(raw["stages"][0]) for _ in range(20)]
    for member in raw["stages"]:
        member[PRIVATE] = {PRIVATE: PRIVATE}
    facts = rejection.declaration_rejection_facts(packet(raw, TOOL), offered_tools=offered["tools"],
                                                error=ValueError("program_stage_count"))
    assert facts["program_members"]["count_capped"] == 12 and facts["program_members"]["overflow"] is True
    assert len(facts["program_members"]["members"]) == 12
    assert PRIVATE not in json.dumps(facts)


def test_new_diagnostic_failure_cannot_mask_original_rejection(offered, monkeypatch):
    def fail(_raw):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(rejection, "_program_member_completion_facts", fail)
    facts = rejection.declaration_rejection_facts(packet(declaration(), TOOL), offered_tools=offered["tools"],
                                                error=ValueError("program_members_cannot_claim_whole_goal"))
    assert (facts["code"] == "program_members_cannot_claim_whole_goal"
            and facts["stage"] == "program_structure")
    assert facts["program_members"] == {"status": "unavailable"}
    assert PRIVATE not in json.dumps(facts)


def test_normal_program_guidance_targets_every_member_but_single_property_keeps_goal_capability(offered):
    system = offered["messages"][0]["content"]
    assert system.endswith(PROGRAM_MEMBER_GUIDANCE)
    assert "INCLUDING THE LAST" in PROGRAM_MEMBER_GUIDANCE
    assert "OMIT goal_conditions entirely" in PROGRAM_MEMBER_GUIDANCE
    tools = {item["function"]["name"]: item["function"] for item in offered["tools"]}
    assert "including the last" in tools[TOOL]["description"]
    assert tools["declare_property_stage"]["parameters"]["properties"]["completion"]["enum"] == [
        "stage", "goal",
    ]
    Draft202012Validator(tools[TOOL]["parameters"]).validate(declaration())


@pytest.mark.parametrize("limit,formal", [(1, False), (1, True), (6, True)])
def test_H1_and_formal_actual_wire_bodies_stay_exact_original58(observed, limit, formal):
    tool = "declare_completion" if limit == 6 else "declare_native_current_action"
    raw = ({"label": "Observed editor", "conditions": [{"kind": "text_contains", "value": "Editor"}]}
           if limit == 6 else current(1, tool))
    options = {"whole_goal_recheck": True} if formal else {}
    expected, old, _ = requested(observed, raw, tool, fmt="semantic_tools_v58", limit=limit,
                                 planning_options=options)
    actual, new, _ = requested(observed, raw, tool, fmt=FORMAT, limit=limit, planning_options=options)
    assert actual == expected and new.content == old.content
    assert PROGRAM_MEMBER_GUIDANCE not in json.loads(new.content)["messages"][0]["content"]
