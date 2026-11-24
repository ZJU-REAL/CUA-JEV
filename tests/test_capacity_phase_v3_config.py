"""Wire/configuration propagation only; never launch an OSWorld episode."""

import argparse
import importlib.util
from pathlib import Path

from test_osworld_comparison import comparison, row, write_summary


def test_cli_explicit_policy_and_master_are_recorded_without_default_change():
    script = Path(__file__).parents[1] / "scripts/run_osworld_smoke.py"
    spec = importlib.util.spec_from_file_location("isolated_capacity_smoke", script)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    parser = argparse.ArgumentParser()
    smoke.add_plan_arguments(parser)
    assert parser.parse_args([]).jev_capacity_recovery_policy == "vlm_once_v1"
    args = parser.parse_args(["--jev-capacity-recovery-policy", "vlm_phase_v3"])
    assert smoke.experiment_settings(args)["jev_capacity_recovery"] == "disabled"
    args = parser.parse_args(["--jev-capacity-recovery-policy", "vlm_phase_v3", "--jev-capacity-recovery"])
    settings = smoke.experiment_settings(args)
    assert settings["jev_capacity_recovery"] == settings["jev_capacity_recovery_policy"] == "vlm_phase_v3"


def test_comparer_accepts_v3_but_keeps_mismatched_policy_visible(tmp_path):
    settings = {"jev_capacity_recovery": "vlm_phase_v3", "jev_capacity_recovery_policy": "vlm_phase_v3"}
    rows = [row(mode, experiment_settings=settings) for mode in ("vlm", "hybrid")]
    report = comparison.summarize([("isolated", write_summary(tmp_path, rows))])
    assert report["pairs"][0]["configuration"]["invalid"] == {}
    rows[0]["experiment_settings"] = {**settings, "jev_capacity_recovery_policy": "vlm_phase_v2"}
    report = comparison.summarize([("isolated", write_summary(tmp_path, rows))])
    assert "jev_capacity_recovery_policy" in report["pairs"][0]["configuration"]["mismatches"]
