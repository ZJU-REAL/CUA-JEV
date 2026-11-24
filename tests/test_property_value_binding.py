"""Bind unknown literal editors to actual fields/options; all execution is local."""

import copy
import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentControl
from cua_jev.desktop_property import pending_dialog_field_completion
from cua_jev.models import Decision, Risk
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import JevPolicy


def raw(value="light"):
    return {"intent": "pursue_property", "label": "Change the requested appearance to light",
            "scope": {"app": "gimp"}, "edit": {"operation": "set", "value": value}}


def compile_stage(observed, data=None, **kwargs):
    plan = parse_plan(data or raw(), observed, profile="desktop", max_actions=6,
                      allow_property_value_binding=True, **kwargs)
    _validate_desktop_execution_plan(plan, observed, 6)
    return plan


@pytest.fixture
def options():
    path = Path(__file__).parent / "fixtures/osworld_selectable_options_v30.json"
    fixture = json.loads(path.read_bytes())
    buffer = io.BytesIO()
    Image.new("RGB", tuple(fixture["screen_size"]), "white").save(buffer, "PNG")
    return OSWorldBackend(object(), pause=0)._snapshot({
        "screenshot": buffer.getvalue(), "accessibility_tree": fixture["xml"],
    })


class Environment:
    def __init__(self, observed, opened=None, *, effect=True):
        self.snapshot, self.opened, self.effect = observed, opened, effect
        self.actions = []

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        if action["action_type"] == "CLICK" and self.effect:
            if self.opened is not None:
                self.snapshot, self.opened = self.opened, None
            else:
                x, y = action["parameters"]["x"], action["parameters"]["y"]
                w, h = self.snapshot.image.region[2:]
                clicked = [c for c in self.snapshot.controls if c.box
                           and c.box[0] * w / 1000 <= x < c.box[2] * w / 1000
                           and c.box[1] * h / 1000 <= y < c.box[3] * h / 1000]
                self.snapshot = replace(self.snapshot, controls=tuple(
                    replace(c, selected=c in clicked) if c.selectable else
                    replace(c, focused=c in clicked) if c.editable else c
                    for c in self.snapshot.controls
                ))
        elif action["action_type"] == "TYPING" and self.effect:
            self.snapshot = replace(self.snapshot, controls=tuple(
                replace(c, value=action["parameters"]["text"]) if c.focused else c
                for c in self.snapshot.controls
            ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def edits(backend, observed, data=None):
    step = parse_plan(data or raw(), observed, profile="desktop", max_actions=6,
                      allow_property_value_binding=True).steps[0]
    return [c for c in backend.compile(step, observed, observed)
            if c.arguments.get("property_phase") in {"edit", "confirm"}]


def light(observed):
    return next(c for c in observed.controls if c.label == "Light")


def test_recorded_selectable_cell_becomes_verified_selection_instead_of_text_entry(options):
    actual = light(options)
    assert actual.role == "cell" and actual.selectable and not actual.editable
    assert actual.box and not actual.selected
    env = Environment(options)
    backend = Backend(env, pause=0)
    choices = edits(backend, options)
    assert len(choices) == 1
    chosen = choices[0]
    assert chosen.capability == "agent.osworld.click"
    assert chosen.description.startswith("select option Light ")
    assert chosen.arguments["bound_control"] == actual.to_dict()
    receipt = backend.execute(chosen, "observation", "decision")
    verified = backend.verify(chosen, receipt)
    assert receipt.success and verified.passed and verified.details["postconditions_met"]
    assert not verified.details["task_success"]
    assert [a["action_type"] for a in env.actions] == ["CLICK"]
    assert backend.bound_property_step(chosen, receipt).postconditions[0].kind == "selected"


def test_modern_literal_option_uses_one_effect_checked_left_click_instead_of_a_navigation_duplicate(options):
    env = Environment(options)
    backend = Backend(env, pause=0)
    actual = light(options)
    old = compile_stage(options)
    modern = compile_stage(options, allow_observed_property_goal=True)

    def choices(plan):
        return [c for c in backend.compile(plan.steps[0], options, options)
                if c.arguments.get("bound_control", {}).get("ref") == actual.ref
                and c.arguments.get("button", "left") == "left"
                and c.arguments.get("click_count", 1) == 1]

    assert len(choices(old)) == 2
    selected, = choices(modern)
    assert selected.description.startswith("select option Light ")
    assert selected.arguments["property_phase"] == "edit"
    receipt = backend.execute(selected, "observation", "decision")
    assert backend.verify(selected, receipt).passed and len(env.actions) == 1
    assert light(env.snapshot).selected
    assert backend.bound_property_step(selected, receipt).postconditions[0].kind == "selected"


def test_ambiguous_literal_options_keep_navigation_without_inventing_a_bound_selection(options):
    actual = light(options)
    current = replace(options, controls=(actual, replace(actual, ref="duplicate", selected=True)))
    plan = compile_stage(current, allow_observed_property_goal=True)
    backend = Backend(Environment(current), pause=0)
    candidates = backend.compile(plan.steps[0], current, current)
    assert candidates and all(c.arguments["property_phase"] == "navigation" for c in candidates)
    assert not any(c.description.startswith("select option ") for c in candidates)


def test_literal_text_fill_does_not_remove_a_distinct_focus_navigation_choice(options):
    field = AgentControl("field", "Alias", "input", role="textbox", input_type="text", editable=True,
                         value="old", app="gimp", window="Main", active=True, box=(100, 100, 400, 200))
    current = replace(options, controls=(field,))
    plan = compile_stage(current, raw("draft"), allow_observed_property_goal=True)
    backend = Backend(Environment(current), pause=0)
    choices = backend.compile(plan.steps[0], current, current)
    assert {c.capability for c in choices} == {"agent.osworld.click", "agent.osworld.fill"}


def test_model_cannot_enable_private_selection_preference(options):
    data = raw()
    data["edit"]["prefer_verified_selection"] = True
    with pytest.raises(ValueError):
        compile_stage(options, data, allow_observed_property_goal=True)


def test_one_plan_keeps_navigation_and_selection_under_the_same_budget(options):
    initial = replace(options, title="Main", text="", controls=tuple(
        AgentControl(name.lower(), name, "button", role="button", app="gimp", window="Main",
                     active=True, box=(10, y, 80, y + 50))
        for name, y in [("Preferences", 100), ("Help", 300)]
    ))
    env = Environment(initial, options)
    calls, events = [], []

    class Planner:
        def plan(self, *args, **kwargs):
            calls.append(kwargs["max_actions"])
            assert len(calls) == 1
            return compile_stage(initial)

    class Choice:
        def choose(self, observation, candidates):
            chosen = next(c for c in candidates if c.description.startswith(
                "select option Light " if env.opened is None else "click Preferences "
            ))
            return Decision(observation.observation_id, chosen.id,
                            {c.id: float(c is chosen) for c in candidates}, 1, "local-fixture", 0)

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Choice(), progress=events.append,
                         config=AgentConfig(allow_actions=True)).run(
        "Change appearance to light", success=lambda _: env.opened is None and light(env.snapshot).selected,
    )
    assert calls == [6] and result.planner_calls == 1 and result.plan_reuses == 1
    assert result.jev_calls == 2 and all(n > 1 for n in result.candidate_counts)
    assert result.property_edits == result.property_subgoals == 1
    assert [e for e in events if e.get("event") == "agent_property_stage"][0]["whole_goal_proven"] is False
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK"]


def test_same_literal_template_uses_real_editable_field_when_its_editor_is_text(options):
    field = AgentControl("field", "Retained revisions", "input", role="textbox", editable=True,
                         value="5", box=(100, 100, 300, 200), app="gimp", window="Main", active=True)
    observed = replace(options, controls=(field,))
    env = Environment(observed)
    backend = Backend(env, pause=0)
    choice, = edits(backend, observed, raw("100"))
    assert choice.capability == "agent.osworld.fill"
    receipt = backend.execute(choice, "obs", "choice")
    assert backend.verify(choice, receipt).passed
    assert env.snapshot.controls[0].value == "100"
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]


@pytest.mark.parametrize("change", ["static", "unselectable", "unboxed", "disabled", "no_click", "foreign"])
def test_no_selection_edit_without_actual_current_option_capabilities(options, change):
    actual = light(options)
    updated = {
        "static": replace(actual, tag="label", role="label", native=None, selectable=False),
        "unselectable": replace(actual, selectable=False),
        "unboxed": replace(actual, box=None),
        "disabled": replace(actual, disabled=True),
        "no_click": replace(actual, tag="label"),
        "foreign": replace(actual, app="another-app"),
    }[change]
    observed = replace(options, controls=(updated,))
    assert edits(Backend(Environment(observed), pause=0), observed) == []


@pytest.mark.parametrize("competitor_selected,competitor_disabled", [(False, False), (True, False),
                                                                   (False, True), (True, True)])
def test_expected_selected_state_or_executability_never_resolves_duplicate_identity(
    options, competitor_selected, competitor_disabled,
):
    actual = light(options)
    twin = replace(actual, ref="duplicate", selected=competitor_selected, disabled=competitor_disabled)
    observed = replace(options, controls=(actual, twin))
    assert edits(Backend(Environment(observed), pause=0), observed) == []


def test_already_selected_option_is_read_only_confirmation_and_does_not_click_again(options):
    actual = light(options)
    observed = replace(options, controls=(replace(actual, selected=True),))
    env = Environment(observed)
    backend = Backend(env, pause=0)
    choice, = edits(backend, observed)
    assert choice.risk == Risk.READ_ONLY and choice.arguments["property_phase"] == "confirm"
    receipt = backend.execute(choice, "obs", "choice")
    assert backend.verify(choice, receipt).passed and not env.actions


@pytest.mark.parametrize("changed", ["box", "selected", "removed", "incomplete"])
def test_changed_option_is_rejected_before_any_click(options, changed):
    env = Environment(options)
    backend = Backend(env, pause=0)
    choice, = edits(backend, options)
    actual = light(options)
    update = replace(actual, box=(100, 100, 200, 200)) if changed == "box" else replace(actual, selected=True)
    env.snapshot = replace(options, a11y_complete=False) if changed == "incomplete" else replace(
        options, controls=tuple(update if c == actual else c for c in options.controls
                                if c != actual or changed != "removed"),
    )
    receipt = backend.execute(choice, "obs", "choice")
    assert not receipt.success and receipt.output["environment_steps"] == 0 and not env.actions


def test_click_that_does_not_select_is_not_property_completion(options):
    env = Environment(options, effect=False)
    backend = Backend(env, pause=0)
    choice, = edits(backend, options)
    receipt = backend.execute(choice, "obs", "choice")
    verification = backend.verify(choice, receipt)
    assert receipt.success and not verification.passed and not verification.details["postconditions_met"]
    assert len(env.actions) == 1


def test_partial_post_observation_cannot_confirm_unique_selection(options):
    class PartialEnvironment(Environment):
        def step(self, action, pause):
            super().step(action, pause)
            self.snapshot = replace(self.snapshot, a11y_complete=False)
            return self.snapshot, 0, False, {}

    env = PartialEnvironment(options)
    backend = Backend(env, pause=0)
    choice, = edits(backend, options)
    receipt = backend.execute(choice, "obs", "choice")
    verification = backend.verify(choice, receipt)
    assert receipt.success and light(env.snapshot).selected
    assert not verification.passed and not verification.details["postconditions_met"]


def request(observed, data, *, contract="property_intent_v30", limit=6, history=()):
    requests = []

    def answer(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner("https://offline.invalid/v1", "model", client=client,
                                    profile="desktop", desktop_contract=contract)
        plan = planner.plan("Set the requested property", observed, list(history), max_actions=limit)
    return plan, requests


def test_new_operation_requires_opt_in_and_never_enters_legacy_action_protocol(options):
    data = raw()
    original = copy.deepcopy(data)
    plan, calls = request(options, data)
    assert plan.steps[0].property_edit.operation == "set" and plan.defer_dialog_field_completion
    assert data == original and "DEFER WIDGET TYPE" in calls[0]["messages"][0]["content"]
    inherited, _ = request(options, data, contract="property_intent_v31")
    assert inherited.steps[0].property_edit.operation == "set"
    for contract in ("property_intent_v25", "property_intent_v28", "property_intent_v29"):
        with pytest.raises(ValueError, match="invalid multimodal"):
            request(options, data, contract=contract)
    with pytest.raises(ValueError, match="edit operation"):
        parse_plan(data, options, profile="desktop", max_actions=6)
    with pytest.raises(ValueError):
        parse_plan({"summary": "bad", "done": False, "steps": [{
            "id": "one", "operation": "set", "ref": light(options).ref, "value": "light",
        }]}, options, profile="desktop")
    with pytest.raises(ValueError):
        parse_plan({"summary": "bad predicate", "done": False, "steps": [{
            "id": "one", "operation": "click", "target": {"name": "Light", "app": "gimp"},
            "preconditions": [{"kind": "ready_to_select", "target": {"name": "Light", "app": "gimp"}}],
        }]}, options, profile="desktop")


@pytest.mark.parametrize("value", ["", "\x00", "\u4e2d", None, False, 100])
def test_literal_input_contract_remains_bounded_ascii(options, value):
    with pytest.raises(ValueError):
        compile_stage(options, raw(value))


@pytest.mark.parametrize("limit,history", [(1, ()), (6, ({"recovery_required": True},))])
def test_new_contract_keeps_baseline_and_recovery_requests_identical(options, limit, history):
    immediate = {"summary": "stop", "steps": [], "done": True}
    bodies = [request(options, immediate, contract=c, limit=limit, history=history)[1]
              for c in ("legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
                        "property_intent_v29", "property_intent_v30", "property_intent_v31")]
    assert all(body == bodies[0] for body in bodies)


def test_no_input_or_option_can_be_derived_from_incomplete_observation(options):
    with pytest.raises(ValueError, match="complete observation"):
        compile_stage(replace(options, a11y_complete=False))
    step = compile_stage(options).steps[0]
    assert Backend(Environment(options), pause=0).compile(step, options,
                                                        replace(options, a11y_complete=False)) == ()


def test_set_literal_is_not_confused_with_a_different_option_or_substring(options):
    for value in ("ligh", "bright light", "nonexistent"):
        assert edits(Backend(Environment(options), pause=0), options, raw(value)) == []
    assert len(edits(Backend(Environment(options), pause=0), options, raw(" Light  "))) == 1


def test_invalid_opt_in_and_baseline_usage_are_rejected(options):
    for kwargs in ({"allow_property_value_binding": 1},
                   {"allow_property_value_binding": True, "max_actions": 1},
                   {"allow_property_value_binding": True, "profile": "browser"}):
        with pytest.raises(ValueError):
            parse_plan(raw(), options, **kwargs)


def test_recorded_hidden_options_remain_unavailable_before_their_view_is_opened():
    fixture = json.loads((Path(__file__).parent / "fixtures/osworld_hidden_options_v30.json").read_bytes())
    buffer = io.BytesIO()
    Image.new("RGB", tuple(fixture["screen_size"]), "white").save(buffer, "PNG")
    observed = OSWorldBackend(object(), pause=0)._snapshot({
        "screenshot": buffer.getvalue(), "accessibility_tree": fixture["xml"],
    })
    assert not any(c.label == "Light" for c in observed.controls)
    assert edits(Backend(Environment(observed), pause=0), observed) == []


def test_selected_option_in_open_dialog_does_not_prove_persistence(options):
    actual = light(options)
    observed = replace(options, controls=tuple(replace(c, selected=True) if c == actual else c
                                               for c in options.controls))
    data = raw() | {"goal_conditions": [{
        "kind": "selected", "target": {"name": "Light", "app": "gimp", "window": "Preferences"},
    }]}
    plan = compile_stage(observed, data, allow_property_preparation=True,
                         allow_property_goal_conditions=True)
    assert pending_dialog_field_completion(plan, observed)["reason"] == "field_effect_in_open_dialog"


def test_private_literal_stays_local_and_policy_can_distinguish_real_property_edits(options):
    literal, calls = "PRIVATE-LITERAL-V30", []
    field = AgentControl("field", "Alias", "input", role="textbox", editable=True,
                         value="old", box=(100, 100, 300, 200), app="gimp", window="Main", active=True)
    observed = replace(options, controls=(field,))
    env = Environment(observed)

    class Planner:
        def plan(self, *_args, **_kwargs):
            return compile_stage(observed, raw(literal) | {"label": "Set requested alias"})

    def answer(request):
        body = json.loads(request.content)
        calls.append(body)
        candidates = body["state"]["available_actions"]
        chosen = next(c for c in candidates if c["capability"] == "agent.osworld.fill")
        return httpx.Response(200, json={"answers": {"action": {
            "choice": chosen["id"], "confidence": 1,
            "probabilities": {c["id"]: float(c is chosen) for c in candidates},
        }}})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = AgentRunner(Backend(env, pause=0), Planner(),
                             policy=JevPolicy(api_key="local", client=client),
                             config=AgentConfig(allow_actions=True, max_steps=1)).run("Set requested alias")
    assert result.property_edits == 1 and len(calls) == 1 and literal not in json.dumps(calls)
    assert "select option" in calls[0]["questions"]["action"]["instructions"]
    assert env.actions[-1]["parameters"]["text"] == literal
