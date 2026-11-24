"""Compare the requested planning policy without inventing missing history."""

import pytest
from test_osworld_comparison import comparison, row, stats, write_summary


@pytest.mark.parametrize("mode", ["full_multimodal_v1", "native_first_v51"])
def test_same_explicit_planning_policy_is_matched(tmp_path, mode):
    path = write_summary(tmp_path, [
        row("hybrid", desktop_planning_observation=mode), row("vlm", desktop_planning_observation=mode),
    ])
    facts = comparison.summarize([("trial", path)])["pairs"][0]["configuration"]
    assert facts["matched"]["desktop_planning_observation"] == mode


def test_different_planning_policy_cannot_create_a_speed_pair(tmp_path):
    path = write_summary(tmp_path, [
        row("hybrid", desktop_planning_observation="native_first_v51"),
        row("vlm", desktop_planning_observation="full_multimodal_v1"),
    ])
    report = comparison.summarize([("trial", path)])
    assert "desktop_planning_observation" in report["pairs"][0]["configuration"]["mismatches"]
    assert stats(report)["agent_ms"]["n"] == 0


@pytest.mark.parametrize("invalid", [False, 1, "native_first", [], {}])
def test_invalid_policy_is_never_reported_as_matching(tmp_path, invalid):
    path = write_summary(tmp_path, [
        row("hybrid", desktop_planning_observation=invalid),
        row("vlm", desktop_planning_observation=invalid),
    ])
    report = comparison.summarize([("trial", path)])
    assert "desktop_planning_observation" in report["pairs"][0]["configuration"]["invalid"]
    assert stats(report)["agent_ms"]["n"] == 0


def test_missing_policy_stays_unknown_and_explicit_null_does_not_inherit(tmp_path):
    original = write_summary(tmp_path, [row("hybrid"), row("vlm")], name="old.json")
    facts = comparison.summarize([("old", original)])["pairs"][0]["configuration"]
    assert facts["unknown"]["desktop_planning_observation"] == ["hybrid", "vlm"]
    current = write_summary(tmp_path, [row("hybrid", desktop_planning_observation=None), row("vlm")],
                            experiment_settings={"desktop_planning_observation": "native_first_v51"})
    facts = comparison.summarize([("current", current)])["pairs"][0]["configuration"]
    assert facts["unknown"]["desktop_planning_observation"] == ["hybrid"]
    assert "desktop_planning_observation" not in facts["matched"]
