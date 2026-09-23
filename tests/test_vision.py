import pytest

from cua_jev.vision import VisualTarget, WindowImage, changed_fraction


def test_visual_target_parser_rejects_ambiguous_or_invalid_boxes():
    assert VisualTarget.from_response({"targets": []}) == ()
    with pytest.raises(ValueError, match="too many"):
        VisualTarget.from_response({"targets": [
            {"label": "Button", "box": [0, 0, 20, 20]} for _ in range(13)
        ]})
    with pytest.raises(ValueError, match="duplicate"):
        VisualTarget.from_response({"targets": [
            {"label": "Button", "box": [0, 0, 20, 20]},
            {"label": "Button", "box": [0, 0, 20, 20]},
        ]})
    with pytest.raises(ValueError, match="four integers"):
        VisualTarget.from_response({"targets": [
            {"label": "Button", "box": [0, 0, True, 20]},
        ]})


def test_window_image_is_bounded_and_change_fraction_detects_effect():
    image = WindowImage(b"\xff\xd8\xff\xd9", bytes(64 * 64), (10, 20, 200, 100))
    assert image.region == (10, 20, 200, 100)
    assert changed_fraction(image.sample, bytes([255]) * (64 * 64)) == 1.0
    with pytest.raises(ValueError, match="JPEG"):
        WindowImage(b"not-an-image", bytes(64 * 64), (10, 20, 200, 100))
