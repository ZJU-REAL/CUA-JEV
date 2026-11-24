"""A repaired preparation retains the declared key and its original budget."""

from dataclasses import replace

import pytest
from test_prepare_repair_runner import Clock, Desktop, root_plan
from test_prepare_repair_runner import png as png

from cua_jev.agent_types import AgentPlan
from cua_jev.desktop_plan_repair import repair_property_prepare, suspend_property_prepare


@pytest.mark.parametrize("key", ["enter", "tab"])
@pytest.mark.parametrize("confirm", [False, True])
def test_prepare_repair_preserves_explicit_numeric_policy_and_reserved_action(png, key, confirm):
    env = Desktop(png, Clock(), confirm=confirm)
    original = root_plan(env.snapshot, confirm=confirm)
    prop = original.steps[1]
    prop = replace(prop, property_edit=replace(prop.property_edit, numeric_finalize_key=key))
    plan = replace(original, steps=(original.steps[0], prop, *original.steps[2:]))
    required = len(plan.steps) + 1
    options = {"plan_at": 100.0, "ttl_s": 60.0, "now": 101.0}
    assert suspend_property_prepare(plan, env.snapshot, env.snapshot, horizon=required - 1, **options) is None
    frame = suspend_property_prepare(plan, env.snapshot, env.snapshot, horizon=required, **options)
    assert frame is not None
    offer = frame.context(env.snapshot, now=102.0)
    assert offer["retained_stage"]["numeric_finalize"] == {"key": key}
    assert offer["remaining_time_s"] == 58.0
    immediate = AgentPlan("Current preparation", (original.steps[0],), prepare_repair_token=frame.token)
    repaired = repair_property_prepare(frame, immediate, env.snapshot, now=102.0)
    assert repaired.steps[1:] == plan.steps[1:]
    assert repaired.steps[1].property_edit.numeric_finalize_key == key
    assert frame.deadline == 160.0 and frame.horizon == required


def test_prepare_repair_does_not_offer_an_undeclared_numeric_key(png):
    env = Desktop(png, Clock())
    plan = root_plan(env.snapshot)
    frame = suspend_property_prepare(
        plan, env.snapshot, env.snapshot, plan_at=100.0, ttl_s=60.0,
        horizon=len(plan.steps), now=101.0,
    )
    assert frame is not None
    assert "numeric_finalize" not in frame.context(env.snapshot, now=102.0)["retained_stage"]
