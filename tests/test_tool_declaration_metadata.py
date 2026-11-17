"""Tool declarations remain explicit run metadata and never hide format conflicts."""

import json

import pytest
import test_osworld_remote_driver as remote_helpers
from test_osworld_comparison import comparison, row, stats, write_summary

pytestmark = pytest.mark.parametrize("output_format", ["tool_declaration_v52", "semantic_tools_v53"])


@pytest.fixture
def remote_driver(monkeypatch, tmp_path):
    return remote_helpers.driver.__wrapped__(monkeypatch, tmp_path)


def test_tool_format_cli_forwarding_and_actual_summary_record_both_modes(
    remote_driver, monkeypatch, tmp_path, output_format,
):
    calls = remote_helpers.install_episode_fakes(monkeypatch, remote_driver)
    argv = remote_helpers.arguments(tmp_path, modes=["vlm", "hybrid"], **{
        "desktop-stage-contract": "property_intent_v52",
        "desktop-output-format": output_format,
        "desktop-planning-observation": "full_multimodal_v1",
        "model-thinking": "on",
        "model-request-timeout": "90",
    })
    assert remote_driver.main(argv) == 0
    assert len(calls["planners"]) == 3  # Configuration check plus one fake per mode.
    for planner in calls["planners"]:
        assert planner.settings["desktop_output_format"] == output_format
        assert planner.settings["desktop_contract"] == "property_intent_v52"
        assert planner.settings["desktop_planning_observation"] == "full_multimodal_v1"
        assert planner.settings["enable_thinking"] is True
        assert planner.settings["thinking_budget"] is None
        assert planner.settings["request_timeout_s"] == 90
    summary = json.loads((tmp_path / "run/summary.json").read_text("utf-8"))
    for settings in [summary["experiment_settings"], *(r["experiment_settings"] for r in summary["runs"])]:
        assert settings["desktop_output_format"] == output_format
        assert settings["desktop_planning_observation"] == "full_multimodal_v1"
        assert settings["model_thinking"] == "on"
        assert settings["model_thinking_budget_tokens"] == "default"
        assert settings["model_request_timeout_s"] == 90
    assert {r["mode"] for r in summary["runs"]} == {"hybrid", "vlm"}
    assert all("--desktop-output-format" not in env.command for env in calls["environments"])


def test_same_explicit_tool_format_is_legal_but_does_not_certify_full_equality(tmp_path, output_format):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", desktop_output_format=output_format), row("vlm", desktop_output_format=output_format),
    ]))])
    pair = report["pairs"][0]
    assert pair["configuration"]["matched"]["desktop_output_format"] == output_format
    assert not pair["configuration"]["invalid"] and pair["eligible"]
    assert stats(report)["agent_ms"]["n"] == 1
    assert pair["configuration"]["complete_configuration_equality_verified"] is False
    assert report["policy"]["causal_speedup_established"] is False


@pytest.mark.parametrize("baseline_format", ["json_schema_v50", "unconstrained_v1"])
def test_tool_format_mismatch_excludes_ratios_without_losing_successes(
    tmp_path, baseline_format, output_format,
):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", agent=1, desktop_output_format=output_format),
        row("vlm", agent=1000, desktop_output_format=baseline_format),
    ]))])
    pair = report["pairs"][0]
    assert pair["configuration"]["mismatches"]["desktop_output_format"] == {
        "hybrid": output_format, "vlm": baseline_format,
    }
    assert not pair["configuration"]["invalid"] and not pair["eligible"]
    assert "configuration_mismatch" in pair["exclusions"]
    assert pair["speedup"] == {"agent_ms": None, "episode_ms": None}
    assert stats(report)["agent_ms"]["n"] == stats(report)["episode_ms"]["n"] == 0
    assert all(report["counts_all_records"][mode]["official_successes"] == 1
               for mode in ("hybrid", "vlm"))


def test_missing_baseline_format_remains_unknown_not_tool_format_equality(tmp_path, output_format):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", desktop_output_format=output_format), row("vlm"),
    ]))])
    configuration = report["pairs"][0]["configuration"]
    assert configuration["unknown"]["desktop_output_format"] == ["vlm"]
    assert "desktop_output_format" not in configuration["matched"]
    assert "desktop_output_format" not in configuration["mismatches"]
    assert configuration["complete_configuration_equality_verified"] is False


def test_matching_tool_format_keeps_failed_record_and_never_creates_speedup(tmp_path, output_format):
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", agent=1, success=False, desktop_output_format=output_format),
        row("vlm", agent=1000, desktop_output_format=output_format),
    ]))])
    pair = report["pairs"][0]
    assert pair["configuration"]["matched"]["desktop_output_format"] == output_format
    assert not pair["eligible"] and stats(report)["agent_ms"]["n"] == 0
    assert report["counts_all_records"]["hybrid"]["records"] == 1
    assert report["counts_all_records"]["hybrid"]["official_successes"] == 0
    assert report["records"][0]["failures"]


def test_mixed_declaration_versions_remain_ineligible(tmp_path, output_format):
    other = "tool_declaration_v52" if output_format == "semantic_tools_v53" else "semantic_tools_v53"
    report = comparison.summarize([("trial", write_summary(tmp_path, [
        row("hybrid", desktop_output_format=output_format), row("vlm", desktop_output_format=other),
    ]))])
    pair = report["pairs"][0]
    assert not pair["eligible"]
    assert "configuration_mismatch" in pair["exclusions"]
    assert stats(report)["agent_ms"]["n"] == 0
