"""Explicit protocol/layout settings retain fair pairs and existing defaults."""

import argparse

import pytest
from test_osworld_comparison import comparison
from test_osworld_driver import driver


def options(*flags):
    parser = argparse.ArgumentParser()
    driver.add_plan_arguments(parser)
    return parser.parse_args(flags)


def configured(format_name):
    args = options("--desktop-output-format", format_name,
                   "--desktop-stage-contract", "property_intent_v52")
    return {"config": driver.experiment_settings(args)}


def test_original_format_default_remains_unconstrained():
    assert options().desktop_output_format == "unconstrained_v1"


def test_actual_smoke_parser_and_settings_record_explicit_new_format():
    args = options("--desktop-output-format", "semantic_tools_v58",
                   "--desktop-stage-contract", "property_intent_v52")
    assert args.desktop_output_format == "semantic_tools_v58"
    assert driver.experiment_settings(args)["desktop_output_format"] == "semantic_tools_v58"


def test_comparer_accepts_same58_and_keeps57_a_distinct_protocol():
    same = comparison.compare_config(configured("semantic_tools_v58"), configured("semantic_tools_v58"))
    assert not same["invalid"] and not same["mismatches"]
    assert same["matched"]["desktop_output_format"] == "semantic_tools_v58"
    mixed = comparison.compare_config(configured("semantic_tools_v58"), configured("semantic_tools_v57"))
    assert "desktop_output_format" in mixed["mismatches"]


@pytest.mark.parametrize("format_name", ["semantic_tools_v55", "semantic_tools_v56", "semantic_tools_v57"])
def test_prior_formats_still_parse_and_keep_their_own_metadata(format_name):
    assert configured(format_name)["config"]["desktop_output_format"] == format_name


def test_v7_layout_remains_explicit_and_old_layout_default_stays_full():
    assert options().jev_request_layout == "full_v1"
    args = options("--jev-request-layout", "shared_fields_v7")
    assert driver.experiment_settings(args)["jev_request_layout"] == "shared_fields_v7"


def test_comparer_does_not_mix_v7_and_v6_even_with_identical_task_scores():
    base = configured("semantic_tools_v58")
    left = {"config": {**base["config"], "jev_request_layout": "shared_fields_v7"}}
    right = {"config": {**base["config"], "jev_request_layout": "shared_fields_v6"}}
    mixed = comparison.compare_config(left, right)
    assert "jev_request_layout" in mixed["mismatches"]
    assert "jev_request_layout" not in mixed["matched"]
    same = comparison.compare_config(left, left)
    assert not same["invalid"] and not same["mismatches"]
    assert same["matched"]["jev_request_layout"] == "shared_fields_v7"


@pytest.mark.parametrize("flag,value", [
    ("--desktop-output-format", "semantic_tools_v59"),
    ("--jev-request-layout", "shared_fields_v8"),
])
def test_future_unknown_protocol_values_are_not_silently_accepted(flag, value):
    with pytest.raises(SystemExit):
        options(flag, value)
