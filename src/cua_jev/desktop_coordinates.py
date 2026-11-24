"""Explicit visual coordinate guidance; no inference from the numeric box values."""


def visual_coordinate_guidance(preferred: str) -> str:
    if preferred == "normalized_1000":
        # Preserve the historical single-action prompt exactly by default.
        return (
            'PREFER box_space="normalized_1000": locate the target '
            "directly on a native 0..1000 coordinate scale over the ENTIRE computer screenshot and give "
            "box=[left,top,right,bottom] as four integers. Do not first estimate pixels and manually "
            "calculate a conversion. The upper-left is (0,0), image center is (500,500), and bottom-right "
            "is (1000,1000), regardless of screenshot resolution. Accessibility observations may include "
            "box_1000=[left,top,right,bottom] measured "
            "from a currently showing control. When that label and visible target match, use its "
            'reported box with box_space="normalized_1000" instead of estimating new coordinates. '
            "normalized_1000 is also the legacy default if box_space is omitted. image_pixels is an "
            "optional alternative ONLY if your coordinates actually use the attached image's integer "
            "pixel positions within screenshot_geometry.width_px/height_px; explicitly declare "
            'box_space="image_pixels" in that case. Never relabel normalized coordinates as pixels. '
            "Coordinate spaces are NEVER inferred. "
        )
    if preferred != "image_pixels":
        raise ValueError("unsupported visual box space preference")
    return (
        'PREFER box_space="image_pixels": locate the target directly in integer pixel positions '
        "of the ATTACHED JPEG, using screenshot_geometry.width_px/height_px, and return "
        "box=[left,top,right,bottom]. Do not normalize or calculate screen-resolution coordinates. "
        "The executor converts this explicit pixel box to the real desktop. Choose a box wholly "
        "inside the intended visible control; do not include an adjacent menu row. "
        "Accessibility box_1000 measurements remain normalized: use those unchanged ONLY with "
        'box_space="normalized_1000". Never label pixel positions normalized_1000 or vice versa. '
        "Both spaces remain supported, the legacy omitted-space default is normalized_1000, "
        "and coordinate spaces are NEVER inferred. "
    )


def compact_visual_box_rule(preferred: str) -> str:
    if preferred == "normalized_1000":
        return "native 0..1000 coordinates over the entire image, width/height>=5"
    if preferred == "image_pixels":
        return (
            "integer pixel positions over the attached JPEG using screenshot_geometry.width_px/height_px; "
            "after conversion each side must span >=5/1000 of that image"
        )
    raise ValueError("unsupported visual box space preference")
