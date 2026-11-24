"""Committed spinbutton values use exact, bounded ASCII numeric equivalence."""

from __future__ import annotations

import io
from dataclasses import replace
from decimal import localcontext

import pytest
from PIL import Image

from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, condition_holds, value_matches
from cua_jev.osworld_backend import OSWorldBackend


def control(value, *, role="spinbutton"):
    return AgentControl("numeric", "Number", "input", role=role, input_type="number", value=value,
                        editable=True, app="Editor")


def evidence(actual, expected, *, role="spinbutton", controls=None, text=""):
    condition = AgentCondition("value_equals", AgentTarget(name="Number", app="Editor"), expected)
    snapshot = AgentSnapshot("osworld://desktop", "Editor", text,
                             (control(actual, role=role),) if controls is None else controls)
    result = condition_evidence(condition, snapshot)
    assert condition_holds(condition, snapshot) is (result["outcome"] == "true")
    return result


@pytest.mark.parametrize("actual,expected", [
    ("50.0", "50"), ("050", "50"), ("+50.00", "5e1"), (".5", "0.50"),
    ("50.", "50"), ("-0.0", "+0"), ("-1.2500", "-125e-2"), ("5E+1", "50.0"),
    ("1e-1000", "1.0e-1000"), ("1e1000", "1.0e1000"),
])
def test_spinbutton_same_number_different_format_is_equal(actual, expected):
    assert evidence(actual, expected) == {"outcome": "true", "reason": "condition_met"}


@pytest.mark.parametrize("actual,expected", [
    ("50.00000000000000000000000000001", "50"),
    ("0.10000000000000000000000000001", "0.1"),
    ("9007199254740993", "9007199254740992"),
    ("1e-1000", "0"), ("-50", "50"),
])
def test_no_float_rounding_or_tolerance_can_hide_a_different_committed_value(actual, expected):
    with localcontext() as context:
        context.prec = 3
        assert evidence(actual, expected) == {"outcome": "false", "reason": "state_mismatch"}


@pytest.mark.parametrize("role", ["textbox", "entry", "combobox", "button", "slider"])
@pytest.mark.parametrize("actual,expected", [("050", "50"), ("50.0", "50")])
def test_other_controls_keep_literal_equality_even_when_input_type_is_number(role, actual, expected):
    assert not value_matches(control(actual, role=role), expected)
    assert evidence(actual, expected, role=role)["outcome"] == "false"


@pytest.mark.parametrize("role", ["spin-button", "SPIN BUTTON", "spin_button"])
def test_role_normalization_limits_equivalence_to_real_spinbutton_aliases(role):
    assert evidence("50.0", "50", role=role)["outcome"] == "true"


@pytest.mark.parametrize("actual,expected", [
    ("NaN", "nan"), ("sNaN", "NaN"), ("Infinity", "Inf"), ("-Infinity", "Infinity"),
    ("50,0", "50"), ("1,000", "1000"), ("50px", "50"), ("50%", "0.5"),
    ("５０", "50"), ("٥٠", "50"), ("50\u200b", "50"),
    (" 50", "50"), ("50 ", "50"), ("50\n", "50"), ("0x32", "50"),
    ("5_0", "50"), ("", "0"), ("+", "0"), (".", "0"), ("1e", "1"),
])
def test_invalid_or_localized_numeric_syntax_cannot_create_new_equivalence(actual, expected):
    assert evidence(actual, expected)["outcome"] == "false"


@pytest.mark.parametrize("value", ["NaN", "50px", "50,0", "", "1e9999999999", "1" * 129])
def test_unsupported_syntax_preserves_existing_exact_string_semantics(value):
    assert evidence(value, value)["outcome"] == "true"


@pytest.mark.parametrize("actual,expected", [
    ("1e1001", "10e1000"), ("1e-1001", ".1e-1000"),
    ("1e00001", "10"), ("1e99999999999999999999999", "10e99999999999999999999998"),
    ("0" * 128 + "1", "1"), ("1." + "0" * 127, "1"),
])
def test_numeric_size_and_exponent_bounds_fail_without_expanding_values(actual, expected):
    assert evidence(actual, expected)["outcome"] == "false"


def test_duplicate_numeric_targets_remain_unknown_even_if_both_values_equal():
    first = control("50.0")
    second = replace(first, ref="second", value="050")
    assert evidence("unused", "50", controls=(first, second)) == {
        "outcome": "unknown", "reason": "ambiguous_target",
    }


def test_changed_display_text_does_not_replace_old_committed_value():
    # The real source separates ns:value:value from valuetext and node text.
    # Only the committed value enters AgentControl.value; display text stays here.
    assert evidence("80.0", "50", text="spin-button Number: text=50 valuetext=50") == {
        "outcome": "false", "reason": "state_mismatch",
    }


def test_static_label_does_not_make_one_real_numeric_input_ambiguous():
    field = control("50.0")
    label = AgentControl("label", "Number", "text", role="label", app="Editor")
    assert evidence("unused", "50", controls=(field, label))["outcome"] == "true"


class NumericEnvironment:
    def __init__(self, *, role="spin-button", immediate=False):
        self.role, self.immediate = role, immediate
        self.committed, self.display = "80.0", "80"
        self.actions = []
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "white").save(buffer, format="PNG")
        self.frame = buffer.getvalue()

    def _get_obs(self):
        return {"screenshot": self.frame, "accessibility_tree": f'''
            <desktop xmlns:v="https://accessibility.ubuntu.example.org/ns/value"
                     xmlns:a="https://accessibility.ubuntu.example.org/ns/attributes">
              <application name="Editor"><frame name="Editor" active="true">
                <{self.role} name="Number" editable="true" focused="true" enabled="true"
                 screencoord="(100, 100)" size="(200, 40)" a:text-input-type="number"
                 v:value="{self.committed}" a:valuetext="{self.display}">{self.display}</{self.role}>
              </frame></application>
            </desktop>'''}

    def step(self, action, pause):
        self.actions.append(action)
        if action["action_type"] == "TYPING":
            self.display = action["parameters"]["text"]
            if self.immediate:
                self.committed = "50.0"
        return self._get_obs(), 0, False, {}


def execute_numeric_fill(env):
    backend = OSWorldBackend(env, pause=0)
    before = backend.observe()
    target = AgentTarget(name="Number", app="Editor")
    step = AgentStep("fill-number", "fill", target=target, value="50", recovery="observe",
                     postconditions=(AgentCondition("value_equals", target, "50"),))
    action = backend.compile(step, before, before)[0]
    receipt = backend.execute(action, "observation", "decision")
    assert receipt.success and receipt.output["environment_steps"] == 3
    assert [item["action_type"] for item in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    return backend, action, receipt


@pytest.mark.parametrize("role,expected", [("spin-button", True), ("entry", False)])
def test_immediate_fill_verification_uses_committed_numeric_equivalence_only_for_spinbutton(role, expected):
    env = NumericEnvironment(role=role, immediate=True)
    backend, action, receipt = execute_numeric_fill(env)
    verification = backend.verify(action, receipt)
    assert verification.passed is expected
    assert verification.details["postconditions_met"] is expected
    assert receipt.output["postcondition"]["value"] == "50.0"
    assert env.display == "50" and len(env.actions) == 3
    backend.close()


@pytest.mark.parametrize("role,expected", [("spin-button", True), ("entry", False)])
def test_pending_fill_waits_for_committed_value_without_repeating_input(role, expected):
    env = NumericEnvironment(role=role)
    backend, action, receipt = execute_numeric_fill(env)
    # TYPING updated visible text/valuetext only; committed numeric state is old.
    assert env.display == "50" and env.committed == "80.0"
    assert receipt.output["postcondition"]["value"] == "80.0"
    assert not backend.verify(action, receipt).passed
    assert not backend.verify_pending(action, receipt).passed
    env.committed = "50.0"
    pending = backend.verify_pending(action, receipt)
    assert pending.passed is expected and pending.details["postconditions_met"] is expected
    assert backend.verify(action, receipt) == pending
    assert len(env.actions) == 3
    backend.close()
