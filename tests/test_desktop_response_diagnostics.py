"""Structure diagnostics explain rejected output without echoing model data."""

import hashlib
import json

import pytest

from cua_jev.desktop_response_diagnostics import response_structure_facts


def encoded(facts):
    return json.dumps(facts, ensure_ascii=True, sort_keys=True, allow_nan=False)


def test_known_extra_root_fields_are_visible_without_claiming_they_are_legal():
    raw = {"intent": "act_current", "label": "PRIVATE_LABEL", "completion": "stage",
           "stage": "PRIVATE_STAGE", "action": {"operation": "click", "ref": "PRIVATE_REF"}}
    facts = response_structure_facts(raw)
    root = facts["paths"]["root"]
    assert root["fields"]["completion"] == root["fields"]["stage"] == "string"
    assert root["enums"] == {"completion": "stage", "intent": "act_current"}
    assert facts["paths"]["action"]["enums"] == {"operation": "click"}
    assert facts["untrusted"] and not facts["schema_validated"] and not facts["input_authority"]
    assert facts["status"] == "observed"
    assert "PRIVATE" not in encoded(facts)


def test_misplaced_repair_metadata_is_reported_by_path_without_echoing_any_token():
    token = "SECRET-REPAIR-TOKEN-7c30de1182e14747"
    raw = {"summary": "PRIVATE_SUMMARY", "done": False,
           "steps": [{"operation": "visual_fill", "prepare_repair": token,
                      "action": {"operation": "fill", "value": "PRIVATE_DATA"}}],
           "action": {"operation": "click", "prepare_repair": token},
           "prepare": {"operation": "visual_fill", "prepare_repair": token},
           "edit": {"operation": "set", "value": "PRIVATE_FINAL"},
           "commit": {"operation": "key_press", "value": "PRIVATE_KEY"},
           "confirm": {"label": "PRIVATE_CONFIRM", "prepare_repair": token}}
    facts = response_structure_facts(raw)
    paths = facts["paths"]
    assert "prepare_repair" not in paths["root"]["fields"]
    for path in ("steps[0]", "action", "prepare", "confirm"):
        assert paths[path]["fields"]["prepare_repair"] == "string"
    assert paths["edit"]["enums"]["operation"] == "set"
    assert paths["commit"]["enums"]["operation"] == "key_press"
    assert paths["steps[0].action"]["enums"]["operation"] == "fill"
    assert token not in encoded(facts) and "PRIVATE" not in encoded(facts)


def test_user_fields_unknown_names_coordinates_urls_and_credentials_never_escape():
    secret = "sk-PRIVATE-CREDENTIAL-do-not-log"
    raw = {"intent": "PRIVATE_INTENT", "label": secret, "value": "PRIVATE_VALUE",
           "ref": "PRIVATE_REF", "name": "PRIVATE_NAME", "target": {"name": "PRIVATE_TARGET"},
           "box": [913, 927, 963, 981], "prepare_repair": "PRIVATE_TOKEN",
           "PRIVATE_UNKNOWN_FIELD": {"PRIVATE_KEY": "https://PRIVATE.example/?key=" + secret},
           "action": {"operation": secret, "label": secret}}
    facts = response_structure_facts(raw)
    text = encoded(facts)
    assert "PRIVATE" not in text and "https://" not in text and secret not in text
    assert facts["paths"]["root"]["unknown_field_count"] == 1
    assert facts["paths"]["root"]["fields"]["target"] == "object"
    assert facts["paths"]["root"]["fields"]["box"] == "array"
    assert facts["paths"]["root"]["enums"]["intent"] == "other"
    assert facts["paths"]["action"]["enums"]["operation"] == "other"


def test_model_metadata_cannot_override_diagnostic_authority_flags():
    facts = response_structure_facts({"input_authority": True, "schema_validated": True,
                                      "status": "accepted", "untrusted": False})
    assert not facts["input_authority"] and not facts["schema_validated"]
    assert facts["untrusted"] and facts["status"] == "observed"
    assert facts["paths"]["root"]["unknown_field_count"] == 4


@pytest.mark.parametrize("value,kind,enum", [(False, "boolean", False), (True, "boolean", True),
                                           (0, "number", None), (1, "number", None),
                                           ("true", "string", None), (None, "null", None)])
def test_bool_and_number_types_are_not_coerced(value, kind, enum):
    facts = response_structure_facts({"done": value})["paths"]["root"]
    assert facts["fields"]["done"] == kind
    if enum is None:
        assert "enums" not in facts
    else:
        assert facts["enums"]["done"] is enum


@pytest.mark.parametrize("raw,kind", [(None, "null"), (False, "boolean"), (23, "number"),
                                     (1.5, "number"), ("PRIVATE_JSON_SCALAR", "string"), ([], "array")])
def test_json_roots_are_diagnosed_without_accepting_the_agent_schema(raw, kind):
    facts = response_structure_facts(raw)
    assert facts["paths"]["root"] == {"type": kind}
    assert not facts["schema_validated"] and not facts["input_authority"]
    assert "PRIVATE" not in encoded(facts)


def test_structure_digest_is_canonical_and_does_not_fingerprint_redacted_values():
    a = response_structure_facts({"intent": "act_current", "label": "A" * 40,
                                   "action": {"operation": "click", "ref": "B" * 20}})
    b = response_structure_facts({"action": {"ref": "C" * 20, "operation": "click"},
                                   "label": "D" * 40, "intent": "act_current"})
    assert a == b
    clean = {key: value for key, value in a.items() if key != "structure_sha256"}
    canonical = json.dumps(clean, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    assert a["structure_sha256"] == hashlib.sha256(canonical).hexdigest()


def test_representable_shared_subtrees_are_not_mistaken_for_cycles():
    action = {"operation": "click", "ref": "PRIVATE_REF"}
    facts = response_structure_facts({"action": action, "prepare": action})
    assert facts["status"] == "observed"
    assert facts["paths"]["action"] == facts["paths"]["prepare"]


@pytest.mark.parametrize("shape", ["list", "dict", "indirect"])
def test_cycles_are_rejected_without_recursion_or_echoing_partial_data(shape):
    if shape == "list":
        raw = []
        raw.append(raw)
    elif shape == "dict":
        raw = {}
        raw["PRIVATE_CYCLE_KEY"] = raw
    else:
        raw = {"action": []}
        raw["action"].append({"PRIVATE_CYCLE_KEY": raw})
    facts = response_structure_facts(raw)
    assert facts["error_type"] == "cyclic_structure" and facts["status"] == "unavailable"
    assert "paths" not in facts and "PRIVATE" not in encoded(facts)


@pytest.mark.parametrize("shape", ["wide_list", "wide_dict", "deep", "string", "escaped", "unicode", "int"])
def test_large_inputs_stop_with_fixed_bounded_facts(shape):
    if shape == "wide_list":
        raw = [None] * 10000
    elif shape == "wide_dict":
        raw = {"PRIVATE_FIELD_" + str(index): None for index in range(10000)}
    elif shape == "deep":
        raw = None
        for _ in range(100):
            raw = {"PRIVATE_CHILD": raw}
    elif shape == "string":
        raw = "PRIVATE" * 20000
    elif shape == "escaped":
        raw = "\x00" * 20000
    elif shape == "unicode":
        raw = "\U0001f642" * 30000
    else:
        raw = 1 << 100000
    facts = response_structure_facts(raw)
    assert facts["status"] == "unavailable" and facts["error_type"] == "budget_exceeded"
    assert "paths" not in facts and "PRIVATE" not in encoded(facts)
    assert len(encoded(facts)) < 500


class Hostile:
    def __str__(self):
        raise AssertionError("must not stringify an unknown object")

    def __repr__(self):
        raise AssertionError("must not repr an unknown object")

    def to_dict(self):
        raise AssertionError("must not call unknown object methods")


class HostileDict(dict):
    def __iter__(self):
        raise AssertionError("must not iterate a dict subclass")

    def __len__(self):
        raise AssertionError("must not measure a dict subclass")

    def items(self):
        raise AssertionError("must not inspect a dict subclass")


class HostileList(list):
    def __iter__(self):
        raise AssertionError("must not iterate a list subclass")

    def __len__(self):
        raise AssertionError("must not measure a list subclass")


class HostileStr(str):
    def __str__(self):
        raise AssertionError("must not stringify a str subclass")


class HostileMeta(type):
    def __eq__(cls, other):
        raise AssertionError("must not compare unknown metaclasses")

    def __hash__(cls):
        raise AssertionError("must not hash unknown metaclasses")


class HostileClass(metaclass=HostileMeta):
    pass


def test_unknown_objects_and_builtin_subclasses_never_get_their_methods_called():
    for value in (Hostile(), HostileDict(), HostileList(), HostileStr("PRIVATE"), HostileClass()):
        facts = response_structure_facts({"target": value})
        assert facts["error_type"] == "non_json_value" and "paths" not in facts
        assert "PRIVATE" not in encoded(facts) and "Hostile" not in encoded(facts)


@pytest.mark.parametrize("value", [b"PRIVATE_BYTES", ("PRIVATE_TUPLE",), {"PRIVATE_SET"},
                                    float("nan"), float("inf"), float("-inf"), "\ud800"])
def test_non_json_values_are_rejected_even_in_an_unreported_subtree(value):
    facts = response_structure_facts({"PRIVATE_UNKNOWN": {"PRIVATE_INNER": value}})
    assert facts["error_type"] == "non_json_value" and "paths" not in facts
    assert "PRIVATE" not in encoded(facts)


def test_non_string_or_subclass_keys_are_rejected_without_echoing_the_key():
    for raw in ({23: "PRIVATE"}, {HostileStr("PRIVATE_KEY"): None}):
        facts = response_structure_facts(raw)
        assert facts["error_type"] == "non_json_key" and "PRIVATE" not in encoded(facts)


def test_successful_facts_stay_bounded_when_many_fixed_fields_are_repeated():
    block = {key: "PRIVATE_DATA" for key in (
        "intent", "operation", "label", "name", "ref", "value", "target", "box",
        "completion", "prepare_repair", "recovery", "button", "box_space")}
    raw = {"action": block, "prepare": block, "edit": block, "commit": block, "confirm": block,
           "steps": [{**block, "action": block}]}
    facts = response_structure_facts(raw)
    assert facts["status"] == "observed" and len(encoded(facts)) < 10000
    assert "PRIVATE" not in encoded(facts)
