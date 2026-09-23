import pytest

from cua_jev.vision import VisualScene, VisualTarget, WindowImage, changed_fraction


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


def test_visual_scene_contains_bounded_text_and_grounded_targets():
    scene = VisualScene.from_response({
        "summary": "A settings dialog with a Save button.",
        "visible_text": ["Settings", "Save"],
        "targets": [{"label": "Save", "box": [400, 700, 600, 780]}],
    })
    assert scene.targets[0].ref == "v0"
    assert scene.to_dict()["visible_text"] == ["Settings", "Save"]
    with pytest.raises(ValueError, match="summary"):
        VisualScene.from_response({"summary": "x" * 1001, "targets": []})
    with pytest.raises(ValueError, match="visible text"):
        VisualScene.from_response({"visible_text": ["x"] * 25, "targets": []})
