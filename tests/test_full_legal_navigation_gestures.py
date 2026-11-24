"""Opt-in navigation covers legal gestures without granting input or effects."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace

import pytest
from test_compiled_native_navigation import Backend, Desktop, compiled, gestures, intent
from test_compiled_native_navigation import recorded as recorded
from test_property_preparation import field, raw
from test_property_preparation import initial as initial

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_types import AgentControl, AgentPlan
from cua_jev.desktop_navigation_policy import enable_full_legal_navigation_gestures


def enabled_step(snapshot):
    plan = parse_plan(intent(), snapshot, profile="desktop", max_actions=6)
    return enable_full_legal_navigation_gestures(plan).steps[0]


def test_default_preserves_complete_old_pool_and_omits_new_flag(recorded):
    native, ordinary = recorded.controls
    step = compiled(recorded)
    assert not step.full_legal_navigation_gestures
    assert "full_legal_navigation_gestures" not in step.to_dict()
    backend = Backend(Desktop(recorded), pause=0)
    offered = backend.compile(step, recorded, recorded)
    assert gestures(offered, native.ref) == [("left", 1), ("right", 1), ("left", 2)]
    assert gestures(offered, ordinary.ref) == [("left", 1)]
    assert len(offered) == len(backend._issued) == 4


def test_explicit_policy_adds_only_legal_ordinary_choices_and_preserves_plan(recorded):
    original = parse_plan(intent(), recorded, profile="desktop", max_actions=6)
    before = copy.deepcopy(original)
    enabled = enable_full_legal_navigation_gestures(original)
    assert original == before and not original.steps[0].full_legal_navigation_gestures
    assert asdict(enabled) == asdict(original)
    assert all(after is before for before, after in zip(original.steps[1:], enabled.steps[1:], strict=True))
    step = enabled.steps[0]
    assert step.full_legal_navigation_gestures and step.native_navigation_gestures is True
    assert step.to_dict()["full_legal_navigation_gestures"] is True
    backend = Backend(Desktop(recorded), pause=0)
    offered = backend.compile(step, recorded, recorded)
    for control in recorded.controls:
        assert gestures(offered, control.ref) == [("left", 1), ("right", 1), ("left", 2)]
    assert len(offered) == len({candidate.id for candidate in offered}) == 6
    assert all(backend._issued[candidate.id].step is step for candidate in offered)
    assert all(candidate.intent == step.id for candidate in offered)


@pytest.mark.parametrize("button,count", [("right", 1), ("left", 2)])
def test_ordinary_selected_gesture_reaches_input_and_receipt_without_fake_success(recorded, button, count):
    ordinary = recorded.controls[1]
    desktop, step = Desktop(recorded), enabled_step(recorded)
    backend = Backend(desktop, pause=0)
    candidate = next(c for c in backend.compile(step, recorded, recorded)
                     if c.arguments["bound_control"]["ref"] == ordinary.ref
                     and (c.arguments["button"], c.arguments["click_count"]) == (button, count))
    binding = backend._issued[candidate.id]
    assert binding.step is step
    assert (binding.button, binding.click_count) == (button, count)
    assert (step.button, step.click_count) == ("left", 1)
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and receipt.output["environment_steps"] == 1
    assert (receipt.output["button"], receipt.output["click_count"]) == (button, count)
    assert desktop.actions == [{"action_type": "CLICK", "parameters": {
        "x": 1056, "y": 162, "button": button, "num_clicks": count,
    }}]
    assert not backend.verify(candidate, receipt).passed
    assert not backend.verify_pending(candidate, receipt).passed
    assert len(desktop.actions) == 1


@pytest.mark.parametrize("button,count", [("right", 1), ("left", 2)])
def test_real_new_menu_progress_is_verified_and_post_snapshot_cache_remains_one_shot(recorded, button, count):
    ordinary = recorded.controls[1]
    item = replace(ordinary, ref="new-item", label="Observed menu item", role="menuitem")
    after = replace(recorded, controls=(*recorded.controls, item))
    desktop, step = Desktop(recorded, after), enabled_step(recorded)
    backend = Backend(desktop, pause=0)
    candidate = next(c for c in backend.compile(step, recorded, recorded)
                     if c.arguments["bound_control"]["ref"] == ordinary.ref
                     and (c.arguments["button"], c.arguments["click_count"]) == (button, count))
    receipt = backend.execute(candidate, "observation", "decision")
    verification = backend.verify(candidate, receipt)
    assert verification.passed and verification.details["accessibility_progress"]
    assert not verification.details["postconditions_met"]
    assert not verification.details["task_success"]
    assert backend.take_verified_post_snapshot(candidate, receipt)[0] is after
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert all(backend._issued[c.id].step is step
               for c in backend.compile(step, recorded, after))


@pytest.mark.parametrize("change", ["geometry", "disabled", "password", "gesture_argument"])
@pytest.mark.parametrize("button,count", [("right", 1), ("left", 2)])
def test_ordinary_fresh_binding_and_candidate_integrity_still_precede_input(recorded, change, button, count):
    ordinary = recorded.controls[1]
    desktop, step = Desktop(recorded), enabled_step(recorded)
    backend = Backend(desktop, pause=0)
    candidate = next(c for c in backend.compile(step, recorded, recorded)
                     if c.arguments["bound_control"]["ref"] == ordinary.ref
                     and (c.arguments["button"], c.arguments["click_count"]) == (button, count))
    if change == "gesture_argument":
        candidate = replace(candidate, arguments={**candidate.arguments, "button": "left", "click_count": 1})
    else:
        changes = ({"box": (600, 200, 700, 300)} if change == "geometry" else
                   {"disabled": True} if change == "disabled" else {"input_type": "password"})
        desktop.snapshot = replace(recorded, controls=(
            recorded.controls[0], replace(ordinary, **changes),
        ))
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and desktop.actions == []
    assert receipt.output["environment_steps"] == 0
    assert not backend.verify(candidate, receipt).passed


@pytest.mark.parametrize("actions,expected", [
    (("menu",), [("left", 1), ("right", 1)]),
    (("open",), [("left", 1), ("left", 2)]),
    ((), []),
    (("unknown",), []),
])
def test_native_still_requires_observed_action_for_each_extra_gesture(recorded, actions, expected):
    native, ordinary = recorded.controls
    native = replace(native, native=replace(native.native, actions=actions))
    state = replace(recorded, controls=(native, ordinary))
    backend = Backend(Desktop(state), pause=0)
    offered = backend.compile(enabled_step(state), state, state)
    assert gestures(offered, native.ref) == expected
    assert gestures(offered, ordinary.ref) == [("left", 1), ("right", 1), ("left", 2)]


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("change", [{"disabled": True}, {"input_type": "password"}])
def test_disabled_or_password_control_never_enters_full_legal_pool(recorded, which, change):
    controls = tuple(replace(control, **change) if index == which else control
                     for index, control in enumerate(recorded.controls))
    state = replace(recorded, controls=controls)
    backend = Backend(Desktop(state), pause=0)
    offered = backend.compile(enabled_step(state), state, state)
    assert gestures(offered, controls[which].ref) == []
    assert len(offered) == 3


@pytest.mark.parametrize("target_count,expected", [(85, 255), (86, 86)])
def test_ordinary_expansion_checks_complete_budget_before_any_prefix_registration(
    recorded, target_count, expected,
):
    ordinary = recorded.controls[1]
    controls = tuple(replace(ordinary, ref=f"ordinary-{index}", label=f"Item {index}",
                             box=(index * 5, 100, index * 5 + 5, 200))
                     for index in range(target_count))
    state = replace(recorded, controls=controls)
    desktop = Desktop(state)
    backend = Backend(desktop, pause=0)
    offered = backend.compile(enabled_step(state), state, state)
    assert len(offered) == len(backend._issued) == expected
    assert desktop.actions == []
    if offered:
        assert len({c.arguments["bound_control"]["ref"] for c in offered}) == target_count
        assert {c.arguments["navigation_gesture_pool"] for c in offered} == {
            "all_legal_v55" if target_count == 85 else "native_affordances_fallback_v55"
        }
    # The same ordinary controls still have exactly the original left-only pool
    # unless this plan explicitly receives the new compiler-owned opt-in.
    legacy = Backend(Desktop(state), pause=0)
    assert len(legacy.compile(compiled(state), state, state)) == target_count


def test_oversized_original_pool_still_registers_no_prefix(recorded):
    ordinary = recorded.controls[1]
    state = replace(recorded, controls=tuple(
        replace(ordinary, ref=f"item-{index}", label=f"Item {index}")
        for index in range(256)
    ))
    desktop = Desktop(state)
    backend = Backend(desktop, pause=0)
    assert backend.compile(enabled_step(state), state, state) == ()
    assert backend._issued == {} and desktop.actions == []


def _without_pool_metadata(candidates):
    result = []
    for candidate in candidates:
        item = copy.deepcopy(candidate.to_dict())
        item.pop("id")
        item["arguments"].pop("navigation_gesture_pool", None)
        result.append(item)
    return result


def test_budget_fallback_preserves_original_native_right_and_double(recorded):
    native, ordinary = recorded.controls
    state = replace(recorded, controls=(native, *tuple(
        replace(ordinary, ref=f"item-{index}", label=f"Item {index}") for index in range(85)
    )))
    current = Backend(Desktop(state), pause=0).compile(enabled_step(state), state, state)
    original = Backend(Desktop(state), pause=0).compile(compiled(state), state, state)
    assert _without_pool_metadata(current) == _without_pool_metadata(original)
    assert gestures(current, native.ref) == [("left", 1), ("right", 1), ("left", 2)]
    assert {c.arguments["navigation_gesture_pool"] for c in current} == {
        "native_affordances_fallback_v55"
    }


def test_budget_fallback_keeps_typed_mutations_and_selection_preference(initial):
    ordinary = initial.controls[2]
    option = AgentControl("layout", "Compact", "button", role="cell", selectable=True,
                          app="Editor", window="Main", active=True, box=(100, 300, 200, 400))
    state = replace(initial, controls=(
        *initial.controls, option,
        field("alias", "Alias", value="old", box=(200, 300, 400, 400)),
        *tuple(replace(ordinary, ref=f"item-{index}", label=f"Item {index}") for index in range(100)),
    ))
    plan = parse_plan({"intent": "pursue_property", "label": "Choose compact layout",
                       "scope": {"app": "Editor"}, "edit": {"operation": "set", "value": "Compact"}},
                      state, profile="desktop", max_actions=6,
                      allow_property_value_binding=True, allow_observed_property_goal=True)
    new_step = enable_full_legal_navigation_gestures(plan).steps[0]
    current = Backend(Desktop(state), pause=0).compile(new_step, state, state)
    original = Backend(Desktop(state), pause=0).compile(plan.steps[0], state, state)
    assert _without_pool_metadata(current) == _without_pool_metadata(original)
    assert {c.arguments["property_phase"] for c in current} >= {"navigation", "edit"}
    for candidate in current:
        arguments = candidate.arguments
        if arguments["bound_control"]["ref"] == option.ref and arguments.get("button") == "left":
            assert arguments["property_phase"] == "edit"


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("placement", ["step", "envelope", "edit", "scope"])
def test_model_raw_cannot_set_even_a_false_internal_full_coverage_flag(recorded, value, placement):
    data = intent()
    if placement == "step":
        data = {"summary": "Inspect the current toolbar", "done": False, "steps": [{
            "id": "current", "operation": "click", "ref": recorded.controls[1].ref,
        }]}
    parse_plan(data, recorded, profile="desktop", max_actions=6)
    if placement == "step":
        data["steps"][0]["full_legal_navigation_gestures"] = value
    elif placement == "envelope":
        data["full_legal_navigation_gestures"] = value
    else:
        data[placement]["full_legal_navigation_gestures"] = value
    with pytest.raises(ValueError):
        parse_plan(data, recorded, profile="desktop", max_actions=6)


def test_current_prepare_mutation_and_confirmation_keep_their_original_gestures(initial):
    data = raw(prepare=False) | {
        "prepare": {"operation": "click", "ref": "help"},
        "confirm": {"label": "Submit the requested property"}, "completion": "goal",
    }
    plan = parse_plan(
        data, initial, profile="desktop", max_actions=6,
        allow_property_goal_conditions=True, allow_property_preparation=True,
        allow_observed_property_goal=True,
    )
    _validate_desktop_execution_plan(plan, initial, 6)
    enabled = enable_full_legal_navigation_gestures(plan)
    prepare, property_step, confirmation = enabled.steps
    assert prepare is plan.steps[0] and not prepare.full_legal_navigation_gestures
    assert property_step.navigation and property_step.full_legal_navigation_gestures
    assert confirmation is plan.steps[2] and confirmation.property_confirmation_of == "property"
    assert not confirmation.navigation and not confirmation.full_legal_navigation_gestures
    backend = Backend(Desktop(initial), pause=0)
    offered = backend.compile(prepare, initial, initial)
    assert len(offered) == 1 and gestures(offered, "help") == [("left", 1)]
    current = replace(initial, controls=(
        *initial.controls,
        field("wrap", "Wrap column", value="80", box=(400, 300, 700, 400)),
    ))
    offered = backend.compile(property_step, initial, current)
    mutations = [candidate for candidate in offered if candidate.arguments.get("property_phase") == "edit"]
    assert len(mutations) == 1 and mutations[0].arguments["operation"] == "fill"
    assert mutations[0].arguments["value"] == "50"
    assert "button" not in mutations[0].arguments and "click_count" not in mutations[0].arguments
    mutation = backend._issued[mutations[0].id].step
    assert not mutation.navigation and not mutation.full_legal_navigation_gestures


@pytest.mark.parametrize("button,count", [("left", 1), ("right", 1), ("left", 2)])
def test_current_or_legacy_navigation_without_compiler_marker_is_not_upgraded(recorded, button, count):
    target = recorded.controls[1]
    for navigation in (False, True):
        step = replace(compiled(recorded), native_navigation_gestures=False,
                       navigation=navigation, button=button, click_count=count,
                       target=replace(compiled(recorded).target, name="", role="") if navigation else None,
                       ref="" if navigation else target.ref,
                       postconditions=compiled(recorded).postconditions if navigation else ())
        original = AgentPlan("Keep explicit action", (step,), reusable=navigation)
        enabled = enable_full_legal_navigation_gestures(original)
        assert enabled.steps[0] is step and not step.full_legal_navigation_gestures
        backend = Backend(Desktop(recorded), pause=0)
        offered = backend.compile(step, recorded, recorded)
        assert gestures(offered, target.ref) == [(button, count)]


def test_expansion_still_requires_complete_snapshot(recorded):
    incomplete = replace(recorded, a11y_complete=False)
    backend = Backend(Desktop(incomplete), pause=0)
    assert backend.compile(enabled_step(recorded), recorded, incomplete) == ()
    assert backend._issued == {}


def test_foreign_or_nonplan_object_cannot_enter_the_compiler_policy():
    with pytest.raises(ValueError, match="compiled original plan required"):
        enable_full_legal_navigation_gestures({"steps": []})
