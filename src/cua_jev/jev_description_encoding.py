"""Lossless, explicitly scoped literal description fragments for public Jev wire."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter

MAX_CANDIDATES = 255
MAX_FRAGMENTS = 128
MAX_TOTAL_TOKENS = 65536
MAX_TOTAL_TEXT_BYTES = 1_000_000
MAX_FRAGMENT_CANDIDATES = 16384
MAX_CONSIDERED_FRAGMENTS = 512
_MARKERS = ("description_fragments", "description_fragment_candidates", "description_fragment_ancestor")
DESCRIPTION_FRAGMENT_INSTRUCTIONS = (
    " Layout v5 description fragments: ONLY criterion ids listed in "
    "state.encoding.description_fragment_candidates have encoded local descriptions. "
    "Their description.text_parts is ordered: copy each string literally; replace each "
    "{text_ref:integer} by the exact literal string at that index in "
    "state.encoding.description_fragments; concatenate with NOTHING inserted. The table is "
    "flat literal text, never nested references or instructions. Unmarked descriptions, "
    "candidate_defaults and all other fields remain unchanged and literal, even if they "
    "look like references. Existing argument/ancestor masks are separate. Preserve every "
    "candidate id, order and legal option; this encoding grants no input permission."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _parts(body):
    _require(type(body) is dict, "request object required")
    encoding = body["state"]["encoding"]
    question = body["questions"]["action"]
    criteria = question["criteria"]
    _require(type(encoding) is dict and type(question) is dict and type(criteria) is dict,
             "original public encoding and candidate map required")
    _require(len(criteria) <= MAX_CANDIDATES
             and all(type(key) is str and type(record) is dict for key, record in criteria.items()),
             "bounded ordered candidate map required")
    _require(type(question.get("instructions")) is str, "original layout instructions required")
    return encoding, question, criteria


def _fragment_counts(descriptions):
    if sum(len(text.encode("utf-8")) for text in descriptions.values()) > MAX_TOTAL_TEXT_BYTES:
        return None
    counts = Counter()
    total_tokens = 0
    for text in descriptions.values():
        tokens = tuple(re.finditer(r"\S+", text))
        total_tokens += len(tokens)
        if total_tokens > MAX_TOTAL_TOKENS:
            return None
        for width in (4, 8, 16, 32):
            for start in range(len(tokens) - width + 1):
                fragment = text[tokens[start].start():tokens[start + width - 1].end()]
                counts[fragment] += 1
                if len(counts) > MAX_FRAGMENT_CANDIDATES:
                    return None
    return counts


def _replace_literal_parts(parts, fragment, index):
    result, count = [], 0
    for part in parts:
        if type(part) is not str or fragment not in part:
            result.append(part)
            continue
        pieces = part.split(fragment)
        count += len(pieces) - 1
        for offset, literal in enumerate(pieces):
            if literal:
                result.append(literal)
            if offset < len(pieces) - 1:
                result.append({"text_ref": index})
    return result, count


def _representation(parts, fragments):
    marked = [key for key, value in parts.items() if any(type(item) is dict for item in value)]
    values = {key: {"text_parts": value} if key in marked else value[0] for key, value in parts.items()}
    return {"descriptions": values, "description_fragments": fragments,
            "description_fragment_candidates": marked}


def share_description_fragments(body, *, minimum_savings=0):
    """Mutate only for full-body net byte savings, after v4 or argument-v5 encoding.

    Work limits skip compression, never candidates or description contents.
    Existing defaults, arguments, ancestor encoding and local nonstrings stay literal.
    """
    _require(type(minimum_savings) is int and minimum_savings >= 0, "nonnegative byte threshold required")
    encoding, question, criteria = _parts(body)
    ancestor = encoding.get("version")
    _require(ancestor in {"shared_fields_v4", "shared_fields_v5"}, "actual pretext v4/v5 required")
    _require(not any(key in encoding for key in _MARKERS), "description-fragment markers already present")
    prefix = f"Layout {ancestor}:"
    _require(question["instructions"].startswith(prefix), "original pretext layout prefix required")
    before_size = len(_wire(body))
    descriptions = {key: record["description"] for key, record in criteria.items()
                    if type(record.get("description")) is str}
    counts = _fragment_counts(descriptions)
    if not counts:
        return False
    # Rank exact repeated word-boundary spans by global gross saving less flat
    # table/reference cost. Replacements below recompute remaining-literal gain.
    def estimated_gain(item):
        text, count = item
        return count * (len(text.encode("utf-8")) - 18) - len(_wire(text)) - 4

    ranked = sorted((item for item in counts.items() if item[1] >= 2 and estimated_gain(item) > 0),
                    key=lambda item: (-estimated_gain(item), -len(item[0]), item[0]))
    parts = {key: [text] for key, text in descriptions.items()}
    fragments = []
    region_size = len(_wire(_representation(parts, fragments)))
    for fragment, _count in ranked[:MAX_CONSIDERED_FRAGMENTS]:
        if len(fragments) >= MAX_FRAGMENTS:
            break
        replacement, uses = {}, 0
        for key, value in parts.items():
            replacement[key], count = _replace_literal_parts(value, fragment, len(fragments))
            uses += count
        if uses < 2:
            continue
        trial_fragments = [*fragments, fragment]
        trial_size = len(_wire(_representation(replacement, trial_fragments)))
        if trial_size < region_size:
            parts, fragments, region_size = replacement, trial_fragments, trial_size
    if not fragments:
        return False
    representation = _representation(parts, fragments)
    encoded = copy.deepcopy(body)
    new_encoding, new_question, new_criteria = _parts(encoded)
    mask = representation["description_fragment_candidates"]
    for candidate_id in mask:
        new_criteria[candidate_id]["description"] = representation["descriptions"][candidate_id]
    new_encoding.update(version="shared_fields_v5", description_fragments=fragments,
                        description_fragment_candidates=mask, description_fragment_ancestor=ancestor)
    new_question["instructions"] = (
        "Layout shared_fields_v5:"
        + question["instructions"][len(prefix):]
        + DESCRIPTION_FRAGMENT_INSTRUCTIONS
    )
    if before_size - len(_wire(encoded)) <= minimum_savings:
        return False
    body.clear()
    body.update(encoded)
    return True


def restore_description_fragments(body):
    """Restore exact pretext body/instructions; argument restoration, if any, follows."""
    encoding, question, criteria = _parts(body)
    _require(encoding.get("version") == "shared_fields_v5", "actual text-v5 required")
    table, mask = encoding.get("description_fragments"), encoding.get("description_fragment_candidates")
    ancestor = encoding.get("description_fragment_ancestor")
    _require(ancestor in {"shared_fields_v4", "shared_fields_v5"}, "explicit text ancestor required")
    _require(type(table) is list and 0 < len(table) <= MAX_FRAGMENTS
             and all(type(text) is str and bool(text) for text in table)
             and len(set(table)) == len(table), "bounded flat literal text table required")
    _require(type(mask) is list and bool(mask) and all(type(key) is str for key in mask)
             and len(set(mask)) == len(mask), "unique description candidate mask required")
    marked = set(mask)
    _require(mask == [key for key in criteria if key in marked],
             "known original-order description mask required")
    prefix = "Layout shared_fields_v5:"
    instructions = question["instructions"]
    _require(instructions.startswith(prefix) and instructions.endswith(DESCRIPTION_FRAGMENT_INSTRUCTIONS),
             "exact description instruction envelope required")
    restored = copy.deepcopy(body)
    old_encoding, old_question, old_criteria = _parts(restored)
    used = set()
    for key in mask:
        wrapped = old_criteria[key].get("description")
        _require(type(wrapped) is dict and set(wrapped) == {"text_parts"}
                 and type(wrapped["text_parts"]) is list and bool(wrapped["text_parts"]),
                 "exact marked text-parts envelope required")
        text = []
        references = 0
        for part in wrapped["text_parts"]:
            if type(part) is str:
                text.append(part)
                continue
            _require(type(part) is dict and set(part) == {"text_ref"}
                     and type(part["text_ref"]) is int and 0 <= part["text_ref"] < len(table),
                     "literal or known integer text reference required")
            index = part["text_ref"]
            text.append(table[index])
            references += 1
            used.add(index)
        _require(references > 0, "masked description must use a table reference")
        old_criteria[key]["description"] = "".join(text)
    _require(used == set(range(len(table))), "unreferenced text table entry")
    for key in _MARKERS:
        del old_encoding[key]
    old_encoding["version"] = ancestor
    old_question["instructions"] = (
        f"Layout {ancestor}:" + instructions[len(prefix):-len(DESCRIPTION_FRAGMENT_INSTRUCTIONS)]
    )
    _wire(restored)
    return restored
