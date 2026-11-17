"""Read-only geometry of an actual input proxy's immediate structural parent."""

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class ObservedInputGeometry:
    input_ref: str
    input_label: str
    input_role: str
    input_rect_px: tuple[int, int, int, int]
    ancestor_role: str
    ancestor_label: str
    ancestor_class: str
    ancestor_rect_px: tuple[int, int, int, int]
    desktop_size_px: tuple[int, int]
    jpeg_sha256: str = ""
    snapshot_fingerprint: str = ""


INPUT_GEOMETRY_GUIDANCE = (
    " observed_input_ancestor_geometry is read-only context from the actual focused input "
    "proxy's immediate structural parent. Its rectangle belongs to that parent, NOT to a "
    "verified editable control. Match it against the CURRENT screenshot before explicitly "
    "choosing a visual action. It never adds an operation to the input ref; operations=[] "
    "still forbids semantic input. rect_image_pixels is an inward-rounded rectangle in the "
    "attached JPEG, not normalized_1000 or desktop pixels."
)


def input_geometry_payload(snapshot, image_size: tuple[int, int]) -> dict | None:
    """Project only source-bound context; never change executable control geometry."""
    contexts = snapshot.input_geometry
    image = snapshot.image
    if not snapshot.a11y_complete or image is None or len(contexts) != 1:
        return None
    context = contexts[0]
    if not isinstance(context, ObservedInputGeometry):
        return None
    desktop = context.desktop_size_px
    if (len(desktop) != 2 or len(image_size) != 2
            or any(type(x) is not int or x <= 0 for x in (*desktop, *image_size))
            or image.region != (0, 0, *desktop)
            or hashlib.sha256(image.jpeg).hexdigest() != context.jpeg_sha256
            or snapshot.fingerprint() != context.snapshot_fingerprint):
        return None
    matches = tuple(c for c in snapshot.controls if c.ref == context.input_ref)
    if len(matches) != 1:
        return None
    control = matches[0]
    if (control.box is not None or control.disabled or not control.focused
            or not control.editable or not control.active
            or control.label != context.input_label or control.role != context.input_role):
        return None
    for rect in (context.input_rect_px, context.ancestor_rect_px):
        if (len(rect) != 4 or any(type(x) is not int for x in rect)
                or not 0 <= rect[0] < rect[2] <= desktop[0]
                or not 0 <= rect[1] < rect[3] <= desktop[1]):
            return None
    inside, outer = context.input_rect_px, context.ancestor_rect_px
    if not (outer[0] <= inside[0] < inside[2] <= outer[2]
            and outer[1] <= inside[1] < inside[3] <= outer[3]):
        return None
    projected = [Fraction(value * target, source)
                 for value, target, source in zip(outer, image_size * 2, desktop * 2, strict=True)]
    box = [math.ceil(projected[0]), math.ceil(projected[1]),
           math.floor(projected[2]), math.floor(projected[3])]
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    payload = {
        "source": "current_native_accessibility_immediate_parent",
        "input_ref": context.input_ref, "input_label": context.input_label,
        "input_role": context.input_role, "input_rect_desktop_pixels": list(inside),
        "ancestor": {"role": context.ancestor_role, "label": context.ancestor_label,
                     "class": context.ancestor_class, "distance": 1},
        "ancestor_rect_desktop_pixels": list(outer), "desktop_size_px": list(desktop),
        "attachment_sha256": context.jpeg_sha256, "attachment_size_px": list(image_size),
        "snapshot_fingerprint": context.snapshot_fingerprint,
        "rect_image_pixels": box, "projection": "exact_ratio_then_inward_rounding",
        "grants_input_capability": False,
    }
    return payload if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= 4000 else None
