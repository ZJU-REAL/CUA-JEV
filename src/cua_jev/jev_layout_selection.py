"""Choose a profitable public request encoding from one actual v4 body."""

from __future__ import annotations

import copy
import hashlib
import json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _wire(body):
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _ids(body):
    criteria = body["questions"]["action"]["criteria"]
    _require(type(criteria) is dict and 1 <= len(criteria) <= 255
             and all(type(key) is str for key in criteria), "bounded ordered criteria required")
    return list(criteria)


def select_public_request_encoding(
    body, *, share_argument_bases, restore_argument_bases,
    share_description_fragments, restore_description_fragments,
    share_public_control_descriptions, restore_public_control_descriptions,
):
    """Only requested-v6 uses this path; old requested-v5 remains byte-for-byte as before.

    V5 means the COMPLETE original two-stage v5 branch, including its existing
    argument and description encoders/instructions. V6 is independently built
    from v4. Wire metadata/instructions all participate in size comparisons.
    """
    _require(body["state"]["encoding"].get("version") == "shared_fields_v4",
             "one original actual v4 base required")
    baseline = _wire(body)
    original_ids = _ids(body)
    # Neither branch can mutate the common base or the other branch.
    old_v5, new_v6 = copy.deepcopy(body), copy.deepcopy(body)
    argument_applied = share_argument_bases(old_v5)
    description_applied = share_description_fragments(old_v5)
    control_applied = share_public_control_descriptions(new_v6)
    _require(all(type(flag) is bool for flag in (
        argument_applied, description_applied, control_applied,
    )), "explicit encoder results required")

    # Roundtrip checks are offline representation guards, never a new request.
    restored_v5 = old_v5
    if description_applied:
        restored_v5 = restore_description_fragments(restored_v5)
    if argument_applied:
        restored_v5 = restore_argument_bases(restored_v5)
    restored_v6 = (restore_public_control_descriptions(new_v6)
                   if control_applied else new_v6)
    for candidate, restored in ((old_v5, restored_v5), (new_v6, restored_v6)):
        _require(_ids(candidate) == _ids(restored) == original_ids
                 and _wire(restored) == baseline, "exact JSON facts and candidate order required")
    v5_actual = old_v5["state"]["encoding"].get("version")
    v6_actual = new_v6["state"]["encoding"].get("version")
    _require(v5_actual == ("shared_fields_v5" if argument_applied or description_applied
                           else "shared_fields_v4"), "original v5 branch actual layout mismatch")
    _require(v6_actual == ("shared_fields_v6" if control_applied else "shared_fields_v4"),
             "independent v6 branch actual layout mismatch")
    v5_bytes, v6_bytes = _wire(old_v5), _wire(new_v6)
    _require(len(v5_bytes) <= len(baseline) and len(v6_bytes) <= len(baseline),
             "encoder must not report a growing complete body")
    # Tie among profitable branches keeps the established v5 representation.
    # Neither branch is eligible unless the complete body is STRICTLY smaller.
    choices = [(len(v5_bytes), 0, old_v5, v5_actual),
               (len(v6_bytes), 1, new_v6, v6_actual)]
    profitable = [choice for choice in choices if choice[0] < len(baseline)]
    selected, actual = (body, "shared_fields_v4")
    if profitable:
        _, _, selected, actual = min(profitable, key=lambda choice: choice[:2])
    selected_bytes = _wire(selected)
    diagnostics = {
        "version": "independent_v4_v5_v6_selection_v1",
        "requested_layout": "shared_fields_v6", "actual_layout": actual,
        "base_compact_UTF8_bytes": len(baseline),
        "complete_v5_compact_UTF8_bytes": len(v5_bytes),
        "complete_v6_compact_UTF8_bytes": len(v6_bytes),
        "selected_compact_UTF8_bytes": len(selected_bytes),
        "generated_v5_argument_base_applied": argument_applied,
        "generated_v5_description_fragment_applied": description_applied,
        "generated_v6_public_control_description_applied": control_applied,
        "base_sha256": hashlib.sha256(baseline).hexdigest(),
        "selected_sha256": hashlib.sha256(selected_bytes).hexdigest(),
        "byte_serialization": "finite_sorted_compact_UTF8_JSON_ensure_ascii_false",
        "captured_HTTP_bytes": False, "token_fit_proven": False,
    }
    # Diagnostics stay outside the wire. Every selected wire table, mask,
    # version and instruction was already included in the comparison.
    if selected is not body:
        body.clear()
        body.update(selected)
    return diagnostics
