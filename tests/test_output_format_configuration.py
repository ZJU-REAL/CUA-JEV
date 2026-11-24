"""Response format differences and unknown history cannot certify equal runs."""

import pytest
from test_osworld_comparison import comparison, row, stats, write_summary


@pytest.mark.parametrize("output_format", ["unconstrained_v1", "json_schema_v50"])
def test_requested_format_matches_only_the_same_explicit_setting(tmp_path, output_format):
    path = write_summary(tmp_path, [
        row("hybrid", desktop_output_format=output_format),
        row("vlm", desktop_output_format=output_format),
    ])
    report = comparison.summarize([("trial", path)])
    assert report["pairs"][0]["configuration"]["matched"]["desktop_output_format"] == output_format


def test_different_response_formats_exclude_the_speed_pair_without_losing_successes(tmp_path):
    path = write_summary(tmp_path, [
        row("hybrid", desktop_output_format="json_schema_v50"),
        row("vlm", desktop_output_format="unconstrained_v1"),
    ])
    report = comparison.summarize([("trial", path)])
    assert "desktop_output_format" in report["pairs"][0]["configuration"]["mismatches"]
    assert stats(report)["agent_ms"]["n"] == 0
    assert all(report["counts_all_records"][mode]["official_successes"] == 1 for mode in ("vlm", "hybrid"))


@pytest.mark.parametrize("invalid", [True, 1, "json_schema", {}, []])
def test_invalid_format_never_certifies_matching_configuration(tmp_path, invalid):
    path = write_summary(tmp_path, [
        row("hybrid", desktop_output_format=invalid), row("vlm", desktop_output_format=invalid),
    ])
    report = comparison.summarize([("trial", path)])
    assert "desktop_output_format" in report["pairs"][0]["configuration"]["invalid"]
    assert stats(report)["agent_ms"]["n"] == 0


def test_historical_missing_response_format_remains_unknown(tmp_path):
    path = write_summary(tmp_path, [row("hybrid"), row("vlm")])
    report = comparison.summarize([("trial", path)])
    facts = report["pairs"][0]["configuration"]
    assert facts["unknown"]["desktop_output_format"] == ["hybrid", "vlm"]
    assert "desktop_output_format" not in facts["matched"]


def test_explicit_unknown_row_format_does_not_inherit_shared_format(tmp_path):
    path = write_summary(tmp_path, [row("hybrid", desktop_output_format=None), row("vlm")],
                         experiment_settings={"desktop_output_format": "json_schema_v50"})
    report = comparison.summarize([("trial", path)])
    facts = report["pairs"][0]["configuration"]
    assert facts["unknown"]["desktop_output_format"] == ["hybrid"]
    assert "desktop_output_format" not in facts["matched"]
