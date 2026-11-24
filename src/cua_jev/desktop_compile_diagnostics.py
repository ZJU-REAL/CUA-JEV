"""Read-only facts about the snapshots supplied to a visual compile attempt."""

from __future__ import annotations

import hashlib
import json

from .agent_types import AgentSnapshot, AgentStep
from .vision import SAMPLE_EDGE, WindowImage, changed_fraction

_SAMPLE_CHANGE_THRESHOLD = 0.01


def _checked_hash(value: object) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("invalid snapshot digest")
    return value


def _snapshot_facts(snapshot: AgentSnapshot) -> dict:
    if not isinstance(snapshot, AgentSnapshot) or type(snapshot.a11y_complete) is not bool:
        raise TypeError("invalid snapshot")
    controls = [control.to_dict() for control in snapshot.controls]
    result = {
        "scope_key": _checked_hash(snapshot.scope_key()),
        "a11y_complete": snapshot.a11y_complete,
        "control_count": len(controls),
        "controls_sha256": hashlib.sha256(json.dumps(
            controls, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        "fingerprint": _checked_hash(snapshot.fingerprint()),
        "image_present": snapshot.image is not None,
    }
    image = snapshot.image
    if image is None:
        return result
    if (not isinstance(image, WindowImage)
            or not isinstance(image.jpeg, (bytes, bytearray))
            or not isinstance(image.sample, (bytes, bytearray, memoryview))
            or len(image.sample) != SAMPLE_EDGE * SAMPLE_EDGE
            or not isinstance(image.region, (tuple, list)) or len(image.region) != 4
            or any(type(value) is not int or abs(value) > 2**63 - 1 for value in image.region)
            or image.region[2] <= 0 or image.region[3] <= 0):
        raise ValueError("invalid image evidence")
    result.update(
        image_region=list(image.region),
        jpeg_sha256=hashlib.sha256(image.jpeg).hexdigest(),
        sample_sha256=hashlib.sha256(image.sample).hexdigest(),
    )
    return result


def unavailable_visual_facts(
    step: AgentStep, planned: AgentSnapshot, current: AgentSnapshot,
) -> dict | None:
    """Describe supplied evidence without observing, recompiling or granting input.

    ``fresh_equivalent`` mirrors the existing backend predicate for valid public
    snapshots, including its equality-only completeness check. It does not say
    which compile branch actually returned no candidate. Malformed evidence or
    failed introspection leaves that equivalence unknown and exposes no error
    message, label, literal input or other observed text.
    """
    if not isinstance(step, AgentStep) or step.operation not in ("visual_click", "visual_fill"):
        return None
    result = {
        "source": "same_memory_visual_compile_facts_v1",
        "read_only": True,
        "changes_candidates": False,
        "diagnosis_scope": "snapshot_facts_not_full_compile_rejection_reason",
        "operation": step.operation,
        "sample_change_threshold": _SAMPLE_CHANGE_THRESHOLD,
        "fresh_equivalent": None,
    }
    try:
        result["planned"] = _snapshot_facts(planned)
        result["current"] = _snapshot_facts(current)
        before, after = result["planned"], result["current"]
        result.update(
            scope_equal=before["scope_key"] == after["scope_key"],
            a11y_complete_equal=planned.a11y_complete == current.a11y_complete,
            controls_equal=planned.controls == current.controls,
            images_present=planned.image is not None and current.image is not None,
            image_region_equal=None,
            sample_changed_fraction=None,
        )
        if result["images_present"]:
            result["image_region_equal"] = planned.image.region == current.image.region
            # These are the actual in-memory samples used by the backend. JPEG
            # bytes are hashed for provenance and are never decoded or resampled.
            result["sample_changed_fraction"] = changed_fraction(planned.image.sample, current.image.sample)
        result["fresh_equivalent"] = bool(
            result["scope_equal"] and result["a11y_complete_equal"] and result["controls_equal"]
            and result["images_present"] and result["image_region_equal"]
            and result["sample_changed_fraction"] <= _SAMPLE_CHANGE_THRESHOLD
        )
    except Exception as error:
        # Restrict even custom exception class names to a fixed public vocabulary.
        result["fresh_equivalent"] = None
        result["error_type"] = next((name for cls, name in (
            (TypeError, "TypeError"), (ValueError, "ValueError"), (AttributeError, "AttributeError"),
            (OverflowError, "OverflowError"), (RuntimeError, "RuntimeError"),
        ) if isinstance(error, cls)), "Exception")
    return result
