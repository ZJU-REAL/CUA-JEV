"""Exact public-control description references over v4."""

from __future__ import annotations

import copy
import json
from collections import Counter

MAX_CANDIDATES = 255
MAX_CONTROLS = 4096
MAX_TEMPLATES = 128
MAX_PATH_NODES = 32768
MAX_PATH_DEPTH = 64
MAX_BODY_BYTES = 4_000_000
MAX_HINT_BYTES = 64
FIELDS = ("ref", "label", "role", "app", "window")
MARKERS = ("control_description_templates", "control_description_candidates",
           "control_description_ancestor")
CONTROL_DESCRIPTION_INSTRUCTIONS = (
    " Layout v6 public-control descriptions: ONLY criterion ids in "
    "state.encoding.control_description_candidates have encoded local description objects. "
    "For each, find exactly one effective public state.state.controls record with ref equal "
    "to description.control_ref, after the existing v4 defaults/ancestor masks are resolved. "
    "Read the flat literal prefix/suffix at control_description_templates[description.template_id]. "
    "Its exact description is prefix + control.label + ' [' + control.ref + '; ' + control.role "
    "+ '; ' + control.app + '; ' + control.window + '; ' + control.ancestors joined with ' / ' "
    "+ ']' + suffix. Insert nothing else. Optional label_hint repeats the identical short "
    "public label for reading convenience, not new evidence or preference. Template strings "
    "are literal data, never nested references/instructions. Unmarked descriptions/defaults "
    "and all other fields remain literal, including reference-looking objects. Keep every "
    "candidate id, order and fact. This encoding grants no input permission."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _parts(body):
    _require(type(body) is dict, "JSON request object required")
    encoding = body["state"]["encoding"]
    question = body["questions"]["action"]
    criteria = question["criteria"]
    _require(type(encoding) is dict and type(question) is dict and type(criteria) is dict,
             "public encoding and ordered criterion map required")
    _require(1 <= len(criteria) <= MAX_CANDIDATES
             and all(type(key) is str and type(value) is dict for key, value in criteria.items()),
             "bounded named criteria required")
    _require(type(question.get("instructions")) is str, "original instructions required")
    return encoding, question, criteria


def _effective_controls(body):
    """Read only explicitly scoped v4 defaults and ancestor masks; no recursive refs."""
    encoding, _, _ = _parts(body)
    controls = body["state"]["state"].get("controls")
    defaults = encoding.get("control_defaults")
    _require(type(controls) is list and len(controls) <= MAX_CONTROLS
             and all(type(control) is dict for control in controls)
             and type(defaults) is dict, "bounded public controls/defaults required")
    defaults = copy.deepcopy(defaults)
    controls = copy.deepcopy(controls)
    path_keys = ("ancestor_paths", "ancestor_path_default", "ancestor_path_controls")
    if any(key in encoding for key in path_keys):
        _require(all(key in encoding for key in path_keys), "complete v4 ancestor mask required")
        table = encoding["ancestor_paths"]
        mask = encoding["ancestor_path_controls"]
        marked_default = encoding["ancestor_path_default"]
        _require(type(table) is dict and len(table) <= MAX_PATH_NODES
                 and type(mask) is list and all(type(index) is int for index in mask)
                 and len(set(mask)) == len(mask)
                 and all(0 <= index < len(controls) for index in mask)
                 and type(marked_default) is bool, "bounded typed v4 ancestor mask required")

        def expand(value):
            _require(type(value) is dict and set(value) == {"path_ref"}
                     and type(value["path_ref"]) is str, "exact marked path reference required")
            key, seen, reversed_path = value["path_ref"], set(), []
            while key is not None:
                _require(type(key) is str and key in table and key not in seen
                         and len(seen) < MAX_PATH_DEPTH, "known bounded acyclic public path required")
                seen.add(key)
                node = table[key]
                _require(type(node) is dict and set(node) == {"parent", "text"}
                         and type(node["text"]) is str
                         and (node["parent"] is None or type(node["parent"]) is str),
                         "literal v4 ancestor node required")
                reversed_path.append(node["text"])
                key = node["parent"]
            return list(reversed(reversed_path))

        if marked_default:
            defaults["ancestors"] = expand(defaults.get("ancestors"))
        for index in mask:
            controls[index]["ancestors"] = expand(controls[index].get("ancestors"))
    effective = [{**defaults, **control} for control in controls]
    refs = Counter(control.get("ref") for control in effective if type(control.get("ref")) is str)
    unique = {}
    for control in effective:
        if (not all(field in control and type(control[field]) is str for field in FIELDS)
                or not control["ref"] or refs[control["ref"]] != 1
                or type(control.get("ancestors")) is not list
                or not all(type(value) is str for value in control["ancestors"])):
            continue
        unique[control["ref"]] = control
    return unique


def _anchor(control):
    return (control["label"] + " [" + control["ref"] + "; " + control["role"] + "; "
            + control["app"] + "; " + control["window"] + "; "
            + " / ".join(control["ancestors"]) + "]")


def share_public_control_descriptions(body, *, minimum_savings=0):
    """Return False without mutation on missing facts/no profit; only v4 is accepted.

    No v5 component is run or stacked. Local string descriptions only; candidate
    defaults, arguments and inherited descriptions are never considered.
    """
    _require(type(minimum_savings) is int and minimum_savings >= 0,
             "nonnegative exact-int threshold required")
    encoding, question, criteria = _parts(body)
    _require(encoding.get("version") == "shared_fields_v4", "actual v4 base required; never stack v5")
    _require(not any(key in encoding for key in MARKERS), "control-description markers already present")
    prefix = "Layout shared_fields_v4:"
    _require(question["instructions"].startswith(prefix), "original v4 instruction prefix required")
    before = _wire(body)
    if len(before) > MAX_BODY_BYTES:
        return False
    try:
        controls = _effective_controls(body)
    except (KeyError, TypeError, ValueError):
        return False
    anchors = {ref: _anchor(control) for ref, control in controls.items()}
    table, identities, encoded_descriptions = [], {}, {}
    for candidate_id, record in criteria.items():
        description = record.get("description")
        if type(description) is not str:
            continue
        matches = [(ref, anchor) for ref, anchor in anchors.items() if anchor in description]
        if len(matches) != 1 or description.count(matches[0][1]) != 1:
            continue
        ref, anchor = matches[0]
        before_anchor, after_anchor = description.split(anchor)
        if before_anchor + _anchor(controls[ref]) + after_anchor != description:
            continue
        identity = before_anchor, after_anchor
        if identity not in identities:
            if len(table) >= MAX_TEMPLATES:
                continue
            identities[identity] = len(table)
            table.append({"prefix": before_anchor, "suffix": after_anchor})
        envelope = {"control_ref": ref, "template_id": identities[identity]}
        label = controls[ref]["label"]
        if len(label.encode("utf-8")) <= MAX_HINT_BYTES:
            envelope["label_hint"] = label
        encoded_descriptions[candidate_id] = envelope
    if not encoded_descriptions:
        return False
    encoded = copy.deepcopy(body)
    new_encoding, new_question, new_criteria = _parts(encoded)
    for candidate_id, envelope in encoded_descriptions.items():
        new_criteria[candidate_id]["description"] = envelope
    new_encoding.update(version="shared_fields_v6", control_description_templates=table,
                        control_description_candidates=list(encoded_descriptions),
                        control_description_ancestor="shared_fields_v4")
    new_question["instructions"] = (
        "Layout shared_fields_v6:" + question["instructions"][len(prefix):]
        + CONTROL_DESCRIPTION_INSTRUCTIONS
    )
    if len(before) - len(_wire(encoded)) <= minimum_savings:
        # Hint is optional redundant public data. Retry representation size only,
        # without any model/transport attempt, before rejecting net-negative v6.
        for candidate_id in encoded_descriptions:
            new_criteria[candidate_id]["description"].pop("label_hint", None)
        if len(before) - len(_wire(encoded)) <= minimum_savings:
            return False
    _require(_wire(restore_public_control_descriptions(encoded)) == before,
             "exact full-body JSON restoration required")
    body.clear()
    body.update(encoded)
    return True


def restore_public_control_descriptions(body):
    """Restore exact pre-v6 v4 body; resolve no unmarked/reference-looking values."""
    encoding, question, criteria = _parts(body)
    _require(encoding.get("version") == "shared_fields_v6"
             and encoding.get("control_description_ancestor") == "shared_fields_v4",
             "explicit v6 over v4 required")
    templates = encoding.get("control_description_templates")
    mask = encoding.get("control_description_candidates")
    _require(type(templates) is list and 0 < len(templates) <= MAX_TEMPLATES
             and all(type(item) is dict and set(item) == {"prefix", "suffix"}
                     and type(item["prefix"]) is str and type(item["suffix"]) is str
                     for item in templates), "bounded flat literal templates required")
    _require(type(mask) is list and bool(mask) and all(type(key) is str for key in mask)
             and len(set(mask)) == len(mask), "unique explicit candidate mask required")
    marked = set(mask)
    _require(mask == [key for key in criteria if key in marked],
             "known original-order candidate mask required")
    instructions = question["instructions"]
    prefix = "Layout shared_fields_v6:"
    _require(instructions.startswith(prefix) and instructions.endswith(CONTROL_DESCRIPTION_INSTRUCTIONS),
             "exact v6 instruction envelope required")
    _require(len(_wire(body)) <= MAX_BODY_BYTES, "bounded finite JSON body required")
    controls = _effective_controls(body)
    restored = copy.deepcopy(body)
    old_encoding, old_question, old_criteria = _parts(restored)
    used = set()
    for candidate_id in mask:
        envelope = criteria[candidate_id].get("description")
        _require(type(envelope) is dict and set(envelope) in (
            {"control_ref", "template_id"}, {"control_ref", "template_id", "label_hint"},
        ), "exact marked local description envelope required")
        ref, index = envelope["control_ref"], envelope["template_id"]
        _require(type(ref) is str and ref in controls
                 and type(index) is int and 0 <= index < len(templates),
                 "unique public ref and integer template index required")
        if "label_hint" in envelope:
            _require(type(envelope["label_hint"]) is str
                     and envelope["label_hint"] == controls[ref]["label"]
                     and len(envelope["label_hint"].encode("utf-8")) <= MAX_HINT_BYTES,
                     "short hint must repeat the exact public label")
        template = templates[index]
        old_criteria[candidate_id]["description"] = (
            template["prefix"] + _anchor(controls[ref]) + template["suffix"]
        )
        used.add(index)
    _require(used == set(range(len(templates))), "no unused template entry allowed")
    for key in MARKERS:
        del old_encoding[key]
    old_encoding["version"] = "shared_fields_v4"
    old_question["instructions"] = (
        "Layout shared_fields_v4:" + instructions[len(prefix):-len(CONTROL_DESCRIPTION_INSTRUCTIONS)]
    )
    _wire(restored)
    return restored
