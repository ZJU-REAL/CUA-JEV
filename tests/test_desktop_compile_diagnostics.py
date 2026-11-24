"""Compile diagnostics preserve the existing visual freshness decision exactly."""

import json
from dataclasses import replace

import pytest

from cua_jev.agent_types import AgentControl, AgentSnapshot, AgentStep
from cua_jev.desktop_compile_diagnostics import unavailable_visual_facts
from cua_jev.osworld_backend import _fresh
from cua_jev.vision import SAMPLE_EDGE, WindowImage

PIXELS = SAMPLE_EDGE * SAMPLE_EDGE


def observed(*, changed_pixels=0, pixel_delta=255, jpeg=b"\xff\xd8fixture", image=True):
    sample = bytes([pixel_delta]) * changed_pixels + bytes(PIXELS - changed_pixels)
    return AgentSnapshot(
        "osworld://desktop", "Editor", "Observed text", (
            AgentControl("field", "Existing field", "input", role="textbox", value="old",
                         app="Editor", window="Settings", box=(100, 100, 400, 200), editable=True),
            AgentControl("button", "Existing button", "button", role="button",
                         app="Editor", window="Settings", box=(500, 100, 600, 200)),
        ), image=WindowImage(jpeg, sample, (0, 0, 1920, 1080)) if image else None,
        semantic_text=("Actual retained public structure",),
    )


def visual(operation="visual_fill"):
    return AgentStep("prepare", operation, value="query" if operation == "visual_fill" else "",
                     box=(302, 131, 820, 149), label="Current visible input")


@pytest.mark.parametrize("operation", ["visual_click", "visual_fill"])
def test_identical_snapshots_report_the_same_freshness_without_mutating_inputs(operation):
    planned, current = observed(), observed()
    before = (planned.to_dict(), current.to_dict(), planned.fingerprint(), current.fingerprint())
    facts = unavailable_visual_facts(visual(operation), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is True
    assert facts["scope_equal"] and facts["controls_equal"] and facts["images_present"]
    assert facts["sample_changed_fraction"] == 0
    assert facts["sample_change_threshold"] == 0.01
    assert facts["planned"]["control_count"] == facts["current"]["control_count"] == 2
    assert facts["planned"]["controls_sha256"] == facts["current"]["controls_sha256"]
    assert facts["read_only"] is True and facts["changes_candidates"] is False
    assert (planned.to_dict(), current.to_dict(), planned.fingerprint(), current.fingerprint()) == before


@pytest.mark.parametrize("changed_pixels,pixel_delta,expected", [
    (40, 255, True), (41, 255, False), (PIXELS, 19, True), (41, 20, False),
])
def test_actual_samples_preserve_both_the_per_pixel_and_fraction_thresholds(
    changed_pixels, pixel_delta, expected,
):
    planned = observed()
    current = observed(changed_pixels=changed_pixels, pixel_delta=pixel_delta)
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is expected
    assert facts["controls_equal"] and facts["scope_equal"]
    assert facts["planned"]["jpeg_sha256"] == facts["current"]["jpeg_sha256"]
    assert facts["sample_changed_fraction"] == (changed_pixels / PIXELS if pixel_delta >= 20 else 0)


def test_different_jpeg_bytes_do_not_replace_the_existing_sample_evidence():
    planned, current = observed(), observed(jpeg=b"\xff\xd8different fixture")
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is True
    assert facts["planned"]["jpeg_sha256"] != facts["current"]["jpeg_sha256"]
    assert facts["planned"]["sample_sha256"] == facts["current"]["sample_sha256"]
    assert facts["sample_changed_fraction"] == 0


@pytest.mark.parametrize("change", ["value", "order"])
def test_control_changes_are_distinguished_from_scope_and_pixel_changes(change):
    planned = observed()
    controls = ((replace(planned.controls[0], value="new"), planned.controls[1]) if change == "value"
                else tuple(reversed(planned.controls)))
    current = replace(planned, controls=controls)
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is False
    assert facts["scope_equal"] and facts["sample_changed_fraction"] == 0
    assert not facts["controls_equal"]
    assert facts["planned"]["controls_sha256"] != facts["current"]["controls_sha256"]


def test_scope_change_can_be_identified_even_when_all_controls_and_pixels_match():
    planned = observed()
    current = replace(planned, semantic_text=("Different actual public structure",))
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is False
    assert not facts["scope_equal"] and facts["controls_equal"]
    assert facts["sample_changed_fraction"] == 0


@pytest.mark.parametrize("planned_complete,current_complete,expected", [
    (True, False, False), (False, False, True),
])
def test_completeness_is_compared_exactly_as_the_existing_backend(
    planned_complete, current_complete, expected,
):
    planned, current = replace(observed(), a11y_complete=planned_complete), replace(
        observed(), a11y_complete=current_complete,
    )
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is expected
    assert facts["planned"]["a11y_complete"] is planned_complete
    assert facts["current"]["a11y_complete"] is current_complete
    assert facts["a11y_complete_equal"] is (planned_complete == current_complete)


@pytest.mark.parametrize("before_image,after_image", [(True, False), (False, True), (False, False)])
def test_missing_images_are_reported_without_an_attempt_to_compute_pixel_change(before_image, after_image):
    planned, current = observed(image=before_image), observed(image=after_image)
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is False
    assert facts["planned"]["image_present"] is before_image
    assert facts["current"]["image_present"] is after_image
    assert facts["sample_changed_fraction"] is None and facts["image_region_equal"] is None
    assert "error_type" not in facts


def test_changed_desktop_region_is_separate_from_sample_change():
    planned = observed()
    current = replace(planned, image=replace(planned.image, region=(0, 0, 1280, 720)))
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is _fresh(planned, current) is False
    assert facts["image_region_equal"] is False
    assert facts["sample_changed_fraction"] == 0


@pytest.mark.parametrize("invalid", [None, object()])
def test_invalid_snapshot_cannot_escape_the_diagnostic_or_claim_freshness(invalid):
    facts = unavailable_visual_facts(visual(), observed(), invalid)
    assert facts["fresh_equivalent"] is None and facts["error_type"] == "TypeError"
    assert len(json.dumps(facts)) < 2000


def test_corrupt_sample_is_unknown_and_is_never_reconstructed_from_jpeg():
    planned, current = observed(), observed()
    # Deliberately simulate corrupted in-memory evidence after construction.
    object.__setattr__(current.image, "sample", b"bad")
    facts = unavailable_visual_facts(visual(), planned, current)
    assert facts["fresh_equivalent"] is None and facts["error_type"] == "ValueError"
    assert "sample_changed_fraction" not in facts


def test_unexpected_introspection_errors_expose_neither_messages_nor_custom_class_names():
    sensitive = "private-source-value"
    error_class = type(sensitive, (Exception,), {})

    class BrokenSnapshot(AgentSnapshot):
        def scope_key(self):
            raise error_class(sensitive)

    current = BrokenSnapshot("osworld://desktop", "Editor", "", ())
    facts = unavailable_visual_facts(visual(), observed(), current)
    assert facts["fresh_equivalent"] is None and facts["error_type"] == "Exception"
    assert sensitive not in json.dumps(facts)


def test_facts_include_hashes_instead_of_observed_text_or_requested_literals():
    secret = "private observed text and typed literal"
    snapshot = replace(observed(), text=secret, controls=(replace(observed().controls[0], value=secret),))
    facts = unavailable_visual_facts(replace(visual(), value=secret, label=secret), snapshot, snapshot)
    assert secret not in json.dumps(facts)
    assert facts["fresh_equivalent"] is True


@pytest.mark.parametrize("step", [AgentStep("click", "click"), AgentStep("typing", "type_text"), None])
def test_nonvisual_actions_receive_no_diagnostic_and_do_not_inspect_invalid_snapshots(step):
    assert unavailable_visual_facts(step, object(), object()) is None
