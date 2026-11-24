"""Explicit already-guarded commits; all inputs run only against local fixtures."""

import copy
import io
import json
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image


@pytest.fixture
def observed():
    data = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(data, "PNG")
    controls = (
        AgentControl("dialog", "Preferences", "container", role="dialog", app="Editor",
                     window="Preferences", active=True, box=(50, 50, 950, 950)),
        AgentControl("field", "History levels:", "input", role="spinbutton", input_type="number",
                     value="100.0", edit_buffer="100", editable=True, focused=True, active=True,
                     app="Editor", window="Preferences", box=(100, 100, 300, 200)),
        AgentControl("ok", "OK", "button", role="button", app="Editor", window="Preferences",
                     active=True, box=(500, 800, 650, 900)),
    )
    return AgentSnapshot("osworld://desktop", "Preferences", "", controls,
                         image=screenshot_image(data.getvalue()))


def raw():
    return {"intent": "edit_fields", "label": "Confirm current settings", "edits": [],
            "completion": "goal", "commit": {
                "operation": "click", "target": {"name": "OK", "app": "Editor"},
                "preconditions": [{"kind": "value_equals", "target": {
                    "name": "History levels:", "role": "spinbutton", "app": "Editor",
                }, "value": "100"}],
                "postconditions": [{"kind": "absent", "target": {
                    "name": "Preferences", "role": "dialog", "app": "Editor",
                }}],
            }}


def parse(data, observed, **kwargs):
    return parse_plan(data, observed, profile="desktop", max_actions=6,
                      allow_guarded_commit_only=True, **kwargs)


class Environment:
    def __init__(self, snapshot):
        self.snapshot, self.actions = snapshot, []

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        assert action["action_type"] == "CLICK"
        self.snapshot = replace(self.snapshot, title="Main", controls=(
            AgentControl("main", "Main", "container", role="frame", app="Editor", window="Main",
                         active=True, box=(50, 50, 950, 950)),
        ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def test_explicit_guarded_commit_closes_form_without_repeating_the_fill_or_a_second_plan(observed):
    plan = parse(raw(), observed)
    assert len(plan.steps) == 1 and plan.steps[0].operation == "click"
    assert {c.kind for c in plan.steps[0].preconditions} == {"value_equals", "ready_to_fill"}
    assert plan.completion_conditions == plan.steps[0].postconditions

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    env = Environment(observed)
    result = AgentRunner(Backend(env, pause=0), Planner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Set history to100")
    assert result.status == "unverified" and not result.success
    assert result.planner_calls == 1 and result.plan_completion_claims == 1
    assert len(env.actions) == 1 and env.actions[0]["action_type"] == "CLICK"


@pytest.mark.parametrize("change", ["value", "pending", "duplicate", "readonly", "incomplete"])
def test_no_commit_is_compiled_for_wrong_pending_ambiguous_or_unusable_field_state(observed, change):
    controls = list(observed.controls)
    if change == "value":
        controls[1] = replace(controls[1], value="5.0")
    elif change == "pending":
        controls[1] = replace(controls[1], edit_buffer="5")
    elif change == "duplicate":
        controls.append(replace(controls[1], ref="other", value="5.0"))
    elif change == "readonly":
        controls[1] = replace(controls[1], editable=False, tag="container")
    else:
        observed = replace(observed, a11y_complete=False)
    with pytest.raises(ValueError):
        parse(raw(), replace(observed, controls=tuple(controls)))


@pytest.mark.parametrize("change", ["value", "pending", "button", "dialog", "incomplete"])
def test_changes_after_planning_are_rejected_before_any_input(observed, change):
    env, plan = Environment(observed), parse(raw(), observed)
    backend = Backend(env, pause=0)
    candidate = backend.compile(plan.steps[0], observed, observed)[0]
    controls = list(observed.controls)
    if change == "value":
        controls[1] = replace(controls[1], value="5.0")
    elif change == "pending":
        controls[1] = replace(controls[1], edit_buffer="5")
    elif change == "button":
        controls[2] = replace(controls[2], box=(600, 800, 750, 900))
    elif change == "dialog":
        controls = []
    else:
        observed = replace(observed, a11y_complete=False)
    env.snapshot = replace(observed, controls=tuple(controls))
    receipt = backend.execute(candidate, "observation", "decision")
    assert not receipt.success and env.actions == []


@pytest.mark.parametrize("change", ["no_guard", "focus_only", "missing_commit", "scope_only", "contextless"])
def test_empty_edits_never_infer_a_field_commit_or_persistent_guard(observed, change):
    data = raw()
    if change == "no_guard":
        data["commit"]["preconditions"] = []
    elif change == "focus_only":
        data["commit"]["preconditions"] = [{"kind": "focused", "target": {
            "name": "History levels:", "app": "Editor",
        }}]
    elif change == "missing_commit":
        del data["commit"]
    elif change == "scope_only":
        data["commit"]["preconditions"][0]["target"] = {"app": "Editor"}
    else:
        data["commit"]["preconditions"][0]["target"].pop("app")
    with pytest.raises(ValueError):
        parse(data, observed)


def test_old_protocol_keeps_rejecting_empty_edits_and_new_protocol_never_fabricates_completion(observed):
    with pytest.raises(ValueError, match="at least one current field"):
        parse_plan(raw(), observed, profile="desktop", max_actions=6)
    data = raw()
    data["completion"] = "stage"
    plan = parse(data, observed)
    assert not plan.completion_conditions
    assert plan.steps[0].postconditions


def ref_raw():
    data = raw()
    data["commit"]["target"] = {"ref": "ok", "role": "button", "app": "Editor", "window": "Preferences"}
    data["commit"]["preconditions"][0]["target"] = {
        "ref": "field", "role": "spinbutton", "app": "Editor", "window": "Preferences",
    }
    return data


def test_current_commit_references_resolve_to_actual_queries_without_rewriting_response(observed):
    data = ref_raw()
    before = copy.deepcopy(data)
    plan = parse(data, observed, allow_current_commit_refs=True)
    assert data == before
    assert plan.steps[0].target.name == "OK" and not plan.steps[0].ref
    assert plan.steps[0].preconditions[0].target.name == "History levels:"
    assert plan.steps[0].postconditions[0].kind == "absent"
    env = Environment(observed)
    backend = Backend(env, pause=0)
    candidate, = backend.compile(plan.steps[0], observed, observed)
    receipt = backend.execute(candidate, "observation", "decision")
    assert backend.verify(candidate, receipt).passed and len(env.actions) == 1


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_app", "wrong_role", "wrong_window",
                                    "wrong_value", "pending_buffer", "incomplete", "semantic_duplicate"])
def test_current_commit_ref_never_resolves_by_desired_state_or_ignores_qualifiers(observed, change):
    data = ref_raw()
    controls = list(observed.controls)
    if change == "missing":
        data["commit"]["target"]["ref"] = "future"
    elif change == "duplicate":
        controls.append(controls[2])
    elif change == "wrong_app":
        data["commit"]["target"]["app"] = "Foreign"
    elif change == "wrong_role":
        data["commit"]["target"]["role"] = "menuitem"
    elif change == "wrong_window":
        data["commit"]["target"]["window"] = "Foreign"
    elif change == "wrong_value":
        controls[1] = replace(controls[1], value="5", edit_buffer="5")
    elif change == "pending_buffer":
        controls[1] = replace(controls[1], edit_buffer="5")
    elif change == "incomplete":
        observed = replace(observed, a11y_complete=False)
    else:
        controls.append(replace(controls[1], ref="different", value="5", edit_buffer="5"))
    with pytest.raises(ValueError):
        parse(data, replace(observed, controls=tuple(controls)), allow_current_commit_refs=True)


def test_ref_only_current_commit_uses_actual_ownership_and_keeps_observed_field_identity(observed):
    data = ref_raw()
    data["commit"]["target"] = {"ref": "ok"}
    data["commit"]["preconditions"][0]["target"] = {"ref": "field"}
    plan = parse(data, observed, allow_current_commit_refs=True)
    assert plan.steps[0].target.app == "Editor" and plan.steps[0].target.window == "Preferences"
    guard = plan.steps[0].preconditions[0]
    assert guard.target.name == "History levels:" and guard.target.role == "spinbutton"
    assert guard.value == "100" and not plan.steps[0].ref


@pytest.mark.parametrize("case", ["old", "future_effect", "edit_fields", "nonpersistent"])
def test_ref_extension_is_only_current_commit_button_and_persistent_guards(observed, case):
    data = ref_raw()
    if case == "future_effect":
        data["commit"]["postconditions"][0]["target"]["ref"] = "future-dialog"
    elif case == "edit_fields":
        data["edits"] = [{"operation": "fill", "value": "100", "target":
                          data["commit"]["preconditions"][0]["target"]}]
    elif case == "nonpersistent":
        data["commit"]["preconditions"][0]["kind"] = "focused"
        del data["commit"]["preconditions"][0]["value"]
    with pytest.raises(ValueError):
        parse(data, observed, allow_current_commit_refs=case != "old")


@pytest.mark.parametrize("kind", ["checked", "unchecked", "selected"])
def test_guarded_commit_binds_boolean_or_selected_state_without_rewriting_it(observed, kind):
    field = replace(observed.controls[1], tag="input" if kind != "selected" else "button",
                    role="checkbox" if kind != "selected" else "cell", editable=False,
                    input_type="checkbox" if kind != "selected" else "", value="", edit_buffer=None,
                    checked=kind == "checked", selected=kind == "selected", selectable=kind == "selected")
    data = raw()
    data["commit"]["preconditions"] = [{"kind": kind, "target": {
        "name": field.label, "role": field.role, "app": field.app,
    }}]
    plan = parse(data, replace(observed, controls=(observed.controls[0], field, observed.controls[2])))
    assert len(plan.steps) == 1 and plan.steps[0].operation == "click"


def test_expanded_readiness_never_truncates_an_over_budget_commit_guard_set(observed):
    data = raw()
    data["commit"]["preconditions"] *= 8
    with pytest.raises(ValueError, match="condition budget"):
        parse(data, observed)


def test_already_true_field_value_cannot_also_stand_in_for_the_unobserved_commit_effect(observed):
    data = raw()
    data["commit"]["postconditions"] = copy.deepcopy(data["commit"]["preconditions"])
    with pytest.raises(ValueError, match="unmet committed-result"):
        parse(data, observed)


def test_one_native_caption_colon_can_be_rebound_only_to_the_unique_actual_guard_field(observed):
    data = raw()
    data["commit"]["preconditions"][0]["target"]["name"] = "History levels"
    untouched = copy.deepcopy(data)
    plan = parse(data, observed)
    assert data == untouched
    assert {c.target.name for c in plan.steps[0].preconditions} == {"History levels:"}


@pytest.mark.parametrize("change", ["duplicate", "different_label", "missing_context", "missing_role"])
def test_caption_rebinding_never_uses_expected_value_or_guesses_other_name_changes(observed, change):
    data = raw()
    target = data["commit"]["preconditions"][0]["target"]
    target["name"] = "History levels"
    if change == "duplicate":
        observed = replace(observed, controls=observed.controls + (
            replace(observed.controls[1], ref="second", label="History levels :", value="5.0"),
        ))
    elif change == "different_label":
        target["name"] = "History count"
    elif change == "missing_context":
        target.pop("app")
    else:
        target.pop("role")
    with pytest.raises(ValueError):
        parse(data, observed)


@pytest.mark.parametrize("contract,accepted", [("property_intent_v25", False), ("property_intent_v28", True)])
def test_production_planner_enables_commit_only_for_the_explicit_new_protocol(observed, contract, accepted):
    def answer(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw())}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner("https://offline.invalid/v1", "mock", client=client,
                                    profile="desktop", desktop_contract=contract)
        if accepted:
            plan = planner.plan("Set history to100", observed, [], max_actions=6)
            assert plan.steps[0].operation == "click"
        else:
            with pytest.raises(ValueError):
                planner.plan("Set history to100", observed, [], max_actions=6)


@pytest.mark.parametrize("profile,limit,flag", [("browser", 6, True), ("desktop", 1, True),
                                               ("desktop", 6, 1), ("desktop", 6, "yes")])
def test_commit_only_requires_explicit_desktop_multi_action_opt_in(observed, profile, limit, flag):
    with pytest.raises(ValueError, match="desktop multi-action"):
        parse_plan(copy.deepcopy(raw()), observed, profile=profile, max_actions=limit,
                   allow_guarded_commit_only=flag)
