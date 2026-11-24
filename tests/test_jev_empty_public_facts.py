"""Regression tests for jev empty public facts."""

import copy

import pytest
from test_jev_control_description_encoding import body, wire

from cua_jev.jev_control_description_encoding import (
    _anchor,
    restore_public_control_descriptions,
    share_public_control_descriptions,
)


@pytest.mark.parametrize("field", ["label", "role", "app", "window"])
def test_explicit_empty_string_remains_exact_and_can_be_referenced(field):
    encoded = body()
    for index, control in enumerate(encoded["state"]["state"]["controls"]):
        control[field] = ""
        encoded["questions"]["action"]["criteria"][f"option{index}"]["description"] = (
            "click " + _anchor(control) + " literal current goal via OSWorld input"
        )
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert len(encoded["state"]["encoding"]["control_description_candidates"]) == 60
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


@pytest.mark.parametrize("field", ["label", "role", "app", "window", "ancestors"])
def test_missing_field_is_not_inferred_as_empty(field):
    encoded = body()
    before_description = encoded["questions"]["action"]["criteria"]["option0"]["description"]
    del encoded["state"]["state"]["controls"][0][field]
    before = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"]["description"] == before_description
    assert wire(restore_public_control_descriptions(encoded)) == wire(before)


@pytest.mark.parametrize("value", [None, False, 0, [], {}])
def test_nonstring_value_never_becomes_an_empty_string(value):
    encoded = body()
    encoded["state"]["state"]["controls"][0]["window"] = value
    original = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert encoded["questions"]["action"]["criteria"]["option0"]["description"] == (
        original["questions"]["action"]["criteria"]["option0"]["description"]
    )
    assert wire(restore_public_control_descriptions(encoded)) == wire(original)


def test_ref_still_must_be_nonempty_and_unique():
    encoded = body()
    encoded["state"]["state"]["controls"][0]["ref"] = ""
    original = copy.deepcopy(encoded)
    assert share_public_control_descriptions(encoded)
    assert type(encoded["questions"]["action"]["criteria"]["option0"]["description"]) is str
    assert wire(restore_public_control_descriptions(encoded)) == wire(original)
