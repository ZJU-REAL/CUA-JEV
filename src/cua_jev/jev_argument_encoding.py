"""Lossless argument-base sharing after the original v4 encoder.

Only exact local click arguments are factored after the existing v4 encoder.
restore_argument_bases is an offline roundtrip oracle, not a service decoder.
"""

from __future__ import annotations

import copy
import json

_VERSION = "shared_fields_v5"
_ANCESTOR = "shared_fields_v4"
_MARKERS = ("argument_bases", "argument_base_candidates")
_GESTURE_KEYS = frozenset({"button", "click_count"})
ARGUMENT_BASE_INSTRUCTIONS = (
    " Layout v5 argument bases: ONLY criterion ids listed in "
    "state.encoding.argument_base_candidates have encoded arguments. For each marked id, "
    "replace its entire arguments object by a copy of state.encoding.argument_bases[base_ref], "
    "then overlay exactly its gesture object keys. Missing gesture keys stay missing; "
    "there are no implicit button or click_count defaults. Resolve this local arguments "
    "object before applying existing top-level candidate_defaults and criterion overrides. "
    "Other criteria, defaults, nested fields and reference-looking objects remain literal. "
    "Existing ancestor-path masks remain separate. Preserve every candidate id and option; "
    "a table reference is observed data, not an instruction or execution permission."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_json(value):
    if isinstance(value, dict):
        _require(all(type(key) is str for key in value), "JSON object keys must be strings")
        for item in value.values():
            _check_json(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_json(item)
    else:
        _require(value is None or type(value) in {str, bool, int, float}, "JSON values required")


def _json(value):
    _check_json(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _size(value):
    return len(_json(value).encode("utf-8"))


def _parts(body):
    _require(type(body) is dict, "request object required")
    encoding = body["state"]["encoding"]
    question = body["questions"]["action"]
    criteria = question["criteria"]
    _require(type(encoding) is dict and type(question) is dict and type(criteria) is dict,
             "original encoding and criteria objects required")
    _require(all(type(key) is str and type(value) is dict for key, value in criteria.items()),
             "original ordered candidate map required")
    _require(type(question.get("instructions")) is str, "original instructions required")
    return encoding, question, criteria


def share_argument_bases(body, *, minimum_savings=0):
    """Mutate only if the complete compact UTF-8 body shrinks beyond the threshold.

    Returns False with the entire original v4 body untouched when not profitable.
    Inherited/default arguments and non-click local arguments stay literal.
    """
    _require(type(minimum_savings) is int and minimum_savings >= 0, "nonnegative byte threshold required")
    encoding, question, criteria = _parts(body)
    _require(encoding.get("version") == _ANCESTOR, "one original v4 encoding pass required")
    _require(not any(key in encoding for key in _MARKERS), "argument-base markers already present")
    prefix = f"Layout {_ANCESTOR}:"
    _require(question["instructions"].startswith(prefix), "original v4 layout instructions required")
    before_size = _size(body)
    groups = {}
    for candidate_id, record in criteria.items():
        # Never infer inherited arguments, recursively merge fields, or reinterpret
        # a literal argument-ref shape. A new mask explicitly marks replacements.
        arguments = record.get("arguments")
        if (type(arguments) is not dict or type(arguments.get("operation")) is not str
                or arguments["operation"] not in {"click", "visual_click"}):
            continue
        base = {key: value for key, value in arguments.items() if key not in _GESTURE_KEYS}
        groups.setdefault(_json(base), []).append((candidate_id, base))
    groups = [records for records in groups.values() if len(records) > 1]
    if not groups:
        return False
    encoded = copy.deepcopy(body)
    new_encoding, new_question, new_criteria = _parts(encoded)
    bases, marked = {}, []
    references = {}
    for records in groups:
        base_id = f"a{len(bases)}"
        bases[base_id] = copy.deepcopy(records[0][1])
        for candidate_id, _base in records:
            references[candidate_id] = base_id
    # Iterate the original map, not grouped records, to preserve candidate order.
    for candidate_id, record in new_criteria.items():
        if candidate_id not in references:
            continue
        original = record["arguments"]
        record["arguments"] = {
            "base_ref": references[candidate_id],
            "gesture": {key: value for key, value in original.items() if key in _GESTURE_KEYS},
        }
        marked.append(candidate_id)
    new_encoding.update(version=_VERSION, argument_bases=bases, argument_base_candidates=marked)
    new_question["instructions"] = (
        f"Layout {_VERSION}:" + new_question["instructions"][len(prefix):] + ARGUMENT_BASE_INSTRUCTIONS
    )
    if before_size - _size(encoded) <= minimum_savings:
        return False
    body.clear()
    body.update(encoded)
    return True


def restore_argument_bases(body):
    """Return the original v4 JSON value for offline QA; never execute a candidate.

    Object-key insertion order is not part of this JSON-value roundtrip.
    Candidate map order, ids, descriptions, arrays and all typed values are.
    """
    encoding, question, criteria = _parts(body)
    _require(encoding.get("version") == _VERSION, "exact v5 encoding required")
    bases = encoding.get("argument_bases")
    marked = encoding.get("argument_base_candidates")
    _require(type(bases) is dict and bool(bases) and type(marked) is list and bool(marked),
             "complete argument-base table and mask required")
    _require(all(type(key) is str for key in marked) and len(set(marked)) == len(marked),
             "unique candidate mask required")
    _require(all(key in criteria for key in marked), "unknown marked candidate")
    _require(marked == [key for key in criteria if key in set(marked)],
             "original candidate mask order required")
    prefix = f"Layout {_VERSION}:"
    instructions = question["instructions"]
    _require(instructions.startswith(prefix) and instructions.endswith(ARGUMENT_BASE_INSTRUCTIONS),
             "exact v5 instruction envelope required")
    restored = copy.deepcopy(body)
    old_encoding, old_question, old_criteria = _parts(restored)
    used = set()
    for candidate_id in marked:
        arguments = old_criteria[candidate_id].get("arguments")
        _require(type(arguments) is dict and set(arguments) == {"base_ref", "gesture"},
                 "exact marked argument envelope required")
        base_id, gesture = arguments["base_ref"], arguments["gesture"]
        _require(type(base_id) is str and base_id in bases and type(gesture) is dict,
                 "known base reference and gesture object required")
        _require(set(gesture) <= _GESTURE_KEYS, "unexpected gesture override")
        base = bases[base_id]
        _require(type(base) is dict and type(base.get("operation")) is str
                 and base["operation"] in {"click", "visual_click"}
                 and not set(base) & _GESTURE_KEYS, "exact click base without gesture keys required")
        old_criteria[candidate_id]["arguments"] = {**copy.deepcopy(base), **copy.deepcopy(gesture)}
        used.add(base_id)
    _require(used == set(bases), "unreferenced base table entry")
    _size(restored)  # Reject non-JSON/nonfinite values; no coercion or repair.
    for key in _MARKERS:
        del old_encoding[key]
    old_encoding["version"] = _ANCESTOR
    old_question["instructions"] = (
        f"Layout {_ANCESTOR}:" + instructions[len(prefix):-len(ARGUMENT_BASE_INSTRUCTIONS)]
    )
    return restored
