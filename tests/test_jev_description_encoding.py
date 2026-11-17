"""Description wire facts roundtrip exactly, including the real public-policy boundary."""

import copy
import json

import httpx
import pytest

from cua_jev import jev_description_encoding as codec
from cua_jev.jev_argument_encoding import restore_argument_bases, share_argument_bases
from cua_jev.models import ActionCandidate, Channel, Observation, Risk
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.policy import JevPolicy

COMMON = ("observed current application window owner context alpha beta gamma delta epsilon zeta "
          "eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau ") * 8


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def body(count=9):
    return {
        "model": "offline", "state": {"state": {"controls": []}, "encoding": {
            "version": "shared_fields_v4", "candidate_defaults": {"risk": "local_write"},
            "control_defaults": {},
        }},
        "questions": {"action": {"instructions": "Layout shared_fields_v4: original shared defaults.",
                                  "criteria": {f"option-{index}": {
                                      "description": f"Current object {index}: {COMMON} gesture {index % 3}",
                                      "arguments": {"operation": "click", "value": "", "button": "left",
                                                    "click_count": 1, "bound_control": {"context": COMMON}},
                                  } for index in range(count)}}},
    }


def independent_descriptions(encoded):
    """Decode the public text contract independently, without the restoration API."""
    encoding = encoded["state"]["encoding"]
    table = encoding["description_fragments"]
    mask = set(encoding["description_fragment_candidates"])
    result = {}
    for key, candidate in encoded["questions"]["action"]["criteria"].items():
        if key not in mask:
            result[key] = candidate.get("description", "ABSENT")
            continue
        result[key] = "".join(part if type(part) is str else table[part["text_ref"]]
                              for part in candidate["description"]["text_parts"])
    return result


def test_full_body_saving_retains_every_option_and_literal_description():
    original = body()
    encoded = copy.deepcopy(original)
    assert codec.share_description_fragments(encoded)
    assert len(wire(encoded)) < len(wire(original))
    assert list(encoded["questions"]["action"]["criteria"]) == list(
        original["questions"]["action"]["criteria"]
    )
    assert independent_descriptions(encoded) == {
        key: item["description"] for key, item in original["questions"]["action"]["criteria"].items()
    }
    restored = codec.restore_description_fragments(encoded)
    assert restored == original and wire(restored) == wire(original)
    assert len(encoded["state"]["encoding"]["description_fragments"]) <= codec.MAX_FRAGMENTS


def test_unicode_whitespace_overlap_and_reference_looking_text_remain_exact():
    original = body()
    unusual = '  雪🙂\talpha\r\nbeta\u00a0gamma e\u0301  {"text_ref":0}  '
    for candidate in original["questions"]["action"]["criteria"].values():
        candidate["description"] = unusual * 30 + COMMON
    encoded = copy.deepcopy(original)
    assert codec.share_description_fragments(encoded)
    assert all(value == unusual * 30 + COMMON for value in independent_descriptions(encoded).values())
    assert codec.restore_description_fragments(encoded) == original


def test_missing_typed_values_defaults_and_unmarked_reference_objects_are_literal():
    original = body()
    criteria = original["questions"]["action"]["criteria"]
    variants = ({}, {"description": None}, {"description": False}, {"description": 0},
                {"description": ""}, {"description": {"text_parts": [{"text_ref": 0}]}},
                {"description": ["unmarked", 0]})
    for index, value in enumerate(variants):
        criteria[f"typed-{index}"] = value
    original["state"]["encoding"]["candidate_defaults"]["description"] = {"text_parts": ["literal", 0]}
    original["state"]["state"]["controls"] = [{"description": COMMON, "path_ref": "literal"}]
    encoded = copy.deepcopy(original)
    assert codec.share_description_fragments(encoded)
    for index, value in enumerate(variants):
        assert encoded["questions"]["action"]["criteria"][f"typed-{index}"] == value
    restored = codec.restore_description_fragments(encoded)
    assert wire(restored) == wire(original)


def test_text_restores_to_argument_v5_before_argument_restore_to_v4():
    original = body()
    for index, candidate in enumerate(original["questions"]["action"]["criteria"].values()):
        candidate["arguments"]["button"] = "right" if index % 3 == 1 else "left"
        candidate["arguments"]["click_count"] = 2 if index % 3 == 2 else 1
    encoded = copy.deepcopy(original)
    assert share_argument_bases(encoded)
    pretext = copy.deepcopy(encoded)
    assert codec.share_description_fragments(encoded)
    assert encoded["state"]["encoding"]["description_fragment_ancestor"] == "shared_fields_v5"
    assert codec.restore_description_fragments(encoded) == pretext
    assert restore_argument_bases(codec.restore_description_fragments(encoded)) == original


def test_nonprofitable_or_at_threshold_leaves_entire_original_body_unchanged():
    tiny = body(2)
    for candidate in tiny["questions"]["action"]["criteria"].values():
        candidate["description"] = "a b c d"
    before = wire(tiny)
    assert codec.share_description_fragments(tiny) is False and wire(tiny) == before
    original = body()
    encoded = copy.deepcopy(original)
    assert codec.share_description_fragments(encoded)
    saving = len(wire(original)) - len(wire(encoded))
    at_threshold = copy.deepcopy(original)
    assert codec.share_description_fragments(at_threshold, minimum_savings=saving) is False
    assert wire(at_threshold) == wire(original)


@pytest.mark.parametrize("corruption", ["unknown_id", "duplicate_id", "reverse_mask", "unknown_ref",
                                         "bool_ref", "nested_table", "unused_table", "extra_part_key"])
def test_corrupt_mask_or_nonflat_unknown_references_reject_without_mutation(corruption):
    encoded = body()
    assert codec.share_description_fragments(encoded)
    encoding = encoded["state"]["encoding"]
    first = encoding["description_fragment_candidates"][0]
    parts = encoded["questions"]["action"]["criteria"][first]["description"]["text_parts"]
    ref = next(part for part in parts if type(part) is dict)
    if corruption == "unknown_id":
        encoding["description_fragment_candidates"].append("missing")
    elif corruption == "duplicate_id":
        encoding["description_fragment_candidates"].append(first)
    elif corruption == "reverse_mask":
        encoding["description_fragment_candidates"].reverse()
    elif corruption == "unknown_ref":
        ref["text_ref"] = len(encoding["description_fragments"])
    elif corruption == "bool_ref":
        ref["text_ref"] = False
    elif corruption == "nested_table":
        encoding["description_fragments"][0] = {"text_ref": 0}
    elif corruption == "unused_table":
        encoding["description_fragments"].append("unused")
    else:
        ref["extra"] = "not allowed"
    before = wire(encoded)
    with pytest.raises(ValueError):
        codec.restore_description_fragments(encoded)
    assert wire(encoded) == before


def test_bounds_repeat_calls_and_work_limits_do_not_drop_facts(monkeypatch):
    encoded = body(255)
    original = copy.deepcopy(encoded)
    monkeypatch.setattr(codec, "MAX_FRAGMENTS", 1)
    assert codec.share_description_fragments(encoded)
    assert len(encoded["state"]["encoding"]["description_fragments"]) == 1
    assert codec.restore_description_fragments(encoded) == original
    before = wire(encoded)
    with pytest.raises(ValueError):
        codec.share_description_fragments(encoded)
    assert wire(encoded) == before
    too_many = body(256)
    before = wire(too_many)
    with pytest.raises(ValueError):
        codec.share_description_fragments(too_many)
    assert wire(too_many) == before
    work_limited = body()
    before = wire(work_limited)
    monkeypatch.setattr(codec, "MAX_TOTAL_TOKENS", 1)
    assert codec.share_description_fragments(work_limited) is False
    assert wire(work_limited) == before


def test_actual_public_policy_can_compress_descriptions_after_arguments_are_removed():
    observation = Observation("Current task", "Current view", {"controls": []}, "offline")
    candidates = [ActionCandidate(
        id=f"choice-{index}", channel=Channel.GUI, capability="agent.click", description=f"{COMMON} {index}",
        arguments={"private": "not-public-argument"}, risk=Risk.LOCAL_WRITE,
    ) for index in range(9)]
    bodies = []

    def choose(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": {"action": {
            "choice": candidates[0].id, "confidence": 1.0,
            "probabilities": {c.id: float(c is candidates[0]) for c in candidates},
        }}})

    for layout in ("shared_fields_v4", "shared_fields_v5"):
        with httpx.Client(transport=httpx.MockTransport(choose)) as client:
            policy = JevPolicy(api_key="offline", model="jev-1.13.0", client=client,
                               request_layout=layout, retries=0)
            result = PublicDecisionPolicy(policy).choose(observation, candidates)
        assert result.candidate_id == candidates[0].id
    baseline, encoded = bodies
    assert "not-public-argument" not in wire(encoded).decode()
    assert encoded["state"]["encoding"]["description_fragment_candidates"] == [c.id for c in candidates]
    assert codec.restore_description_fragments(encoded) == baseline
    assert len(wire(encoded)) < len(wire(baseline))
