"""An explicit guarded current click may derive its owning dialog's absence."""

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
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, "PNG")
    common = {"app": "Editor", "window": "Options", "active": True}
    controls = (
        AgentControl("dialog", "Options", "container", role="dialog", box=(50, 50, 950, 950), **common),
        AgentControl("field", "Kept revisions:", "input", role="spinbutton", input_type="number",
                     value="100.0", edit_buffer="100", editable=True, focused=True,
                     box=(100, 100, 300, 200), ancestors=("dialog: Options",), **common),
        AgentControl("confirm", "Use changes", "button", role="button", box=(500, 800, 700, 900),
                     ancestors=("dialog: Options",), **common),
    )
    return AgentSnapshot("osworld://desktop", "Options", "", controls,
                         image=screenshot_image(buffer.getvalue()))


def raw():
    return {"intent": "edit_fields", "label": "Persist the current revision setting", "edits": [],
            "completion": "goal", "commit": {
                "operation": "click", "target": {"name": "Use changes", "role": "button",
                                                    "app": "Editor", "window": "Options"},
                "preconditions": [{"kind": "value_equals", "target": {
                    "name": "Kept revisions:", "role": "spinbutton", "app": "Editor", "window": "Options",
                }, "value": "100.0"}],
            }}


def parse(data, observed, *, opt_in=True):
    return parse_plan(data, observed, profile="desktop", max_actions=6,
                      allow_guarded_commit_only=True, allow_dialog_commit_effect=opt_in)


class Environment:
    def __init__(self, snapshot, *, close=True, complete=True):
        self.snapshot, self.actions, self.close, self.complete = snapshot, [], close, complete

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        assert action["action_type"] == "CLICK"
        if self.close:
            self.snapshot = replace(self.snapshot, title="Main", a11y_complete=self.complete, controls=(
                AgentControl("main", "Main", "container", role="frame", app="Editor", window="Main",
                             active=True, box=(50, 50, 950, 950)),
            ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def test_derive_only_observed_effect_and_preserve_exact_raw_response(observed):
    data, original = raw(), raw()
    plan = parse(data, observed)
    step, = plan.steps
    assert data == original and "postconditions" not in data["commit"]
    assert step.operation == "click" and step.target.name == "Use changes" and not step.value
    assert step.derived_dialog_commit_effect
    assert [c.kind for c in step.preconditions] == ["value_equals", "ready_to_fill", "exists"]
    post, = step.postconditions
    assert post.kind == "absent" and post.target.role == "dialog"
    assert (post.target.name, post.target.app, post.target.window) == ("Options", "Editor", "Options")
    assert plan.completion_conditions == step.postconditions
    assert step.to_dict()["derived_dialog_commit_effect"] is True
    explicit = copy.deepcopy(data)
    explicit["commit"]["postconditions"] = [{"kind": "absent", "target": {"name": "Options",
                                                                                   "app": "Editor"}}]
    assert "derived_dialog_commit_effect" not in parse(explicit, observed).steps[0].to_dict()


def test_one_explicit_click_and_no_refill_or_automatic_confirmation(observed):
    plan = parse(raw(), observed)

    class Planner:
        def plan(self, *args, **kwargs):
            return plan

    env = Environment(observed)
    result = AgentRunner(Backend(env, pause=0), Planner(), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True)).run("Set retained revisions to100")
    assert result.planner_calls == 1 and result.plan_completion_claims == 1
    assert result.status == "unverified" and not result.success
    assert [a["action_type"] for a in env.actions] == ["CLICK"]


@pytest.mark.parametrize("change", ["missing_dialog", "duplicate_dialog", "inactive_dialog",
                                    "button_ancestry", "field_ancestry", "foreign_field", "duplicate_button",
                                    "wrong_value", "pending_buffer", "incomplete"])
def test_no_effect_derivation_without_current_dialog_ownership_and_guard_proof(observed, change):
    controls = list(observed.controls)
    if change == "missing_dialog":
        controls.pop(0)
    elif change == "duplicate_dialog":
        controls.append(replace(controls[0], ref="second-dialog", disabled=True))
    elif change == "inactive_dialog":
        controls[0] = replace(controls[0], active=False)
    elif change == "button_ancestry":
        controls[2] = replace(controls[2], ancestors=())
    elif change == "field_ancestry":
        controls[1] = replace(controls[1], ancestors=())
    elif change == "foreign_field":
        controls[1] = replace(controls[1], window="Another", ancestors=("dialog: Another",))
    elif change == "duplicate_button":
        controls.append(replace(controls[2], ref="other-button", disabled=True))
    elif change == "wrong_value":
        controls[1] = replace(controls[1], value="5.0")
    elif change == "pending_buffer":
        controls[1] = replace(controls[1], edit_buffer="5")
    else:
        observed = replace(observed, a11y_complete=False)
    with pytest.raises(ValueError):
        parse(raw(), replace(observed, controls=tuple(controls)))


@pytest.mark.parametrize("change", ["field", "button", "dialog", "incomplete"])
def test_new_plan_still_rejects_changed_observation_before_any_input(observed, change):
    plan, env = parse(raw(), observed), Environment(observed)
    backend = Backend(env, pause=0)
    candidate, = backend.compile(plan.steps[0], observed, observed)
    controls = list(observed.controls)
    if change == "field":
        controls[1] = replace(controls[1], value="5.0", edit_buffer="5")
    elif change == "button":
        controls[2] = replace(controls[2], box=(600, 800, 800, 900))
    elif change == "dialog":
        controls[0] = replace(controls[0], active=False)
    else:
        observed = replace(observed, a11y_complete=False)
    env.snapshot = replace(observed, controls=tuple(controls))
    receipt = backend.execute(candidate, "obs", "decision")
    assert not receipt.success and receipt.output["environment_steps"] == 0 and not env.actions


@pytest.mark.parametrize("close,complete", [(False, True), (True, False)])
def test_observed_closure_and_complete_post_tree_are_both_required(observed, close, complete):
    plan, env = parse(raw(), observed), Environment(observed, close=close, complete=complete)
    backend = Backend(env, pause=0)
    candidate, = backend.compile(plan.steps[0], observed, observed)
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert receipt.success and len(env.actions) == 1
    assert not verification.passed and not verification.details["postconditions_met"]


@pytest.mark.parametrize("conditions", [None, [], False, "absent", [{"kind": "not-supported"}]])
def test_explicit_invalid_conditions_are_never_replaced(observed, conditions):
    data = raw()
    data["commit"]["postconditions"] = conditions
    with pytest.raises(ValueError):
        parse(data, observed)
    assert data["commit"]["postconditions"] == conditions


@pytest.mark.parametrize("change", ["keyboard", "right", "double", "no_target", "no_guard",
                                    "missing_button_window", "missing_field_window", "marker"])
def test_no_inferred_action_key_target_or_guard(observed, change):
    data = raw()
    if change == "keyboard":
        data["commit"].update(operation="key_press")
        data["commit"].pop("target")
    elif change == "right":
        data["commit"]["button"] = "right"
    elif change == "double":
        data["commit"]["click_count"] = 2
    elif change == "no_target":
        data["commit"].pop("target")
    elif change == "no_guard":
        data["commit"].pop("preconditions")
    elif change == "missing_button_window":
        data["commit"]["target"].pop("window")
    elif change == "missing_field_window":
        data["commit"]["preconditions"][0]["target"].pop("window")
    else:
        data["commit"]["derived_dialog_commit_effect"] = True
    with pytest.raises(ValueError):
        parse(data, observed)


def test_no_derivation_for_future_unedited_fields_or_when_opt_in_is_disabled(observed):
    with pytest.raises(ValueError, match="postconditions"):
        parse(raw(), observed, opt_in=False)
    data = raw()
    data["edits"] = [{"operation": "fill", "value": "100", "target": {
        "name": "Kept revisions:", "app": "Editor", "window": "Options",
    }}]
    with pytest.raises(ValueError, match="postconditions"):
        parse(data, observed)


def test_derivation_requires_guards_from_the_actual_owning_dialog(observed):
    field = replace(observed.controls[1], window="Another", ancestors=("dialog: Another",))
    observed = replace(observed, controls=(observed.controls[0], field, observed.controls[2]))
    data = raw()
    data["commit"]["preconditions"][0]["target"]["window"] = "Another"
    with pytest.raises(ValueError, match="same dialog"):
        parse(data, observed)


def test_derived_existence_guard_consumes_the_existing_condition_budget(observed):
    data = raw()
    data["commit"]["preconditions"] *= 7
    with pytest.raises(ValueError, match="condition budget"):
        parse(data, observed)


def test_explicit_unmet_result_is_preserved_even_if_closure_could_be_derived(observed):
    data = raw()
    condition = copy.deepcopy(data["commit"]["preconditions"][0])
    condition["value"] = "5"
    data["commit"]["postconditions"] = [condition]
    plan = parse(data, observed)
    step, = plan.steps
    assert not step.derived_dialog_commit_effect
    assert step.postconditions[0].kind == "value_equals" and step.postconditions[0].value == "5"
    env = Environment(observed)
    backend = Backend(env, pause=0)
    choice, = backend.compile(step, observed, observed)
    receipt = backend.execute(choice, "obs", "decision")
    assert receipt.success and not backend.verify(choice, receipt).passed


def test_compiler_provenance_is_not_accepted_in_model_authored_steps(observed):
    with pytest.raises(ValueError):
        parse_plan({"summary": "bad provenance", "done": False, "steps": [{
            "id": "one", "operation": "click", "target": {"name": "Use changes", "app": "Editor"},
            "derived_dialog_commit_effect": True,
        }]}, observed, profile="desktop")


def request(observed, data, *, contract="property_intent_v31", limit=6, history=()):
    calls = []

    def answer(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner("https://offline.invalid/v1", "model", client=client,
                                    profile="desktop", desktop_contract=contract)
        plan = planner.plan("Set retained revisions to100", observed, list(history), max_actions=limit)
    return plan, calls


def test_production_gate_and_exact_missing_condition_response(observed):
    plan, calls = request(observed, raw())
    assert plan.steps[0].derived_dialog_commit_effect
    assert "CURRENT GUARDED DIALOG COMMIT" in calls[0]["messages"][0]["content"]
    for contract in ("property_intent_v28", "property_intent_v29", "property_intent_v30"):
        with pytest.raises(ValueError, match="invalid multimodal"):
            request(observed, raw(), contract=contract)


@pytest.mark.parametrize("limit,history", [(1, ()), (6, ({"recovery_required": True},))])
def test_immediate_and_recovery_requests_remain_identical(observed, limit, history):
    data = {"summary": "Stop", "steps": [], "done": True}
    bodies = [request(observed, data, contract=c, limit=limit, history=history)[1]
              for c in ("legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
                        "property_intent_v29", "property_intent_v30", "property_intent_v31")]
    assert all(body == bodies[0] for body in bodies)


def test_invalid_opt_in_never_authorizes_a_baseline_or_browser_extension(observed):
    for kwargs in ({"allow_dialog_commit_effect": 1},
                   {"allow_dialog_commit_effect": True, "max_actions": 1},
                   {"allow_dialog_commit_effect": True, "profile": "browser"}):
        with pytest.raises(ValueError):
            parse_plan(raw(), observed, **kwargs)
