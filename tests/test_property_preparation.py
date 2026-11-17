"""One current preparation followed by fresh semantic actions; no live service/VM."""

import copy
import io
import json
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.desktop_property import pending_dialog_field_completion
from cua_jev.desktop_property_protocol import property_protocol_instructions
from cua_jev.models import Decision
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image


def field(ref, label, *, value="", box=None, window="Main", **extra):
    return AgentControl(ref, label, "input", role="textbox", input_type="text", editable=True,
                        value=value, box=box, app="Editor", window=window, active=True, **extra)


@pytest.fixture
def initial():
    data = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(data, "PNG")
    return AgentSnapshot("osworld://desktop", "Main", "", (
        AgentControl("root", "Main", "container", role="frame", app="Editor", window="Main",
                     active=True, box=(0, 0, 1000, 1000)),
        field("search", "Search", value="old query", focused=True),
        AgentControl("help", "Help", "button", role="button", app="Editor", window="Main",
                     active=True, box=(800, 500, 900, 600)),
    ), image=screenshot_image(data.getvalue()))


def raw(*, prepare=True):
    data = {"intent": "pursue_property", "label": "Set wrap column to50", "scope": {"app": "Editor"},
            "edit": {"operation": "fill", "value": "50"}, "goal_conditions": [{
                "kind": "value_equals", "target": {"name": "Wrap column", "app": "Editor"}, "value": "50",
            }]}
    if prepare:
        data["prepare"] = {"operation": "visual_fill", "label": "Current search input",
                           "box_space": "normalized_1000", "box": [100, 100, 400, 200],
                           "value": "word wrap column"}
    return data


def compile(data, snapshot, *, v29=True):
    plan = parse_plan(data, snapshot, max_actions=6, profile="desktop", allow_property_goal_conditions=True,
                      allow_property_preparation=v29, allow_guarded_commit_only=True)
    _validate_desktop_execution_plan(plan, snapshot, 6)
    return plan


class Environment:
    def __init__(self, snapshot, *, preparation_effect=True):
        self.snapshot, self.actions = snapshot, []
        self.opened = False
        self.focus = "search"
        self.preparation_effect = preparation_effect

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause):
        self.actions.append(action)
        kind = action["action_type"]
        if kind == "CLICK":
            x, y = action["parameters"]["x"], action["parameters"]["y"]
            if not self.opened:
                self.focus = "search"
            else:
                self.focus = next((c.ref for c in self.snapshot.controls if c.box and c.editable
                                   and c.box[0] * 0.8 <= x < c.box[2] * 0.8
                                   and c.box[1] * 0.6 <= y < c.box[3] * 0.6), "")
            self.snapshot = replace(self.snapshot, controls=tuple(
                replace(c, focused=c.ref == self.focus) for c in self.snapshot.controls
            ))
        elif kind == "TYPING" and self.preparation_effect:
            value = action["parameters"]["text"]
            if self.focus == "search":
                self.opened = True
                self.snapshot = replace(self.snapshot, controls=(
                    self.snapshot.controls[0], field("search", "Search", value=value, focused=True),
                    field("wrap", "Wrap column", value="80", box=(400, 300, 700, 400)),
                    field("cache", "Cache size", value="20", box=(400, 500, 700, 600)),
                ))
            else:
                self.snapshot = replace(self.snapshot, controls=tuple(
                    replace(c, value=value) if c.ref == self.focus else c for c in self.snapshot.controls
                ))
        return self.snapshot, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


class Choice:
    def __init__(self, env, *, move_before_execution=False):
        self.env = env
        self.move_before_execution = move_before_execution

    def choose(self, observation, candidates):
        chosen = next(c for c in candidates if c.capability == "agent.osworld.fill"
                      and c.description.startswith("fill Wrap column "))
        if self.move_before_execution:
            self.env.snapshot = replace(self.env.snapshot, controls=tuple(
                replace(c, box=(450, 300, 750, 400)) if c.ref == "wrap" else c
                for c in self.env.snapshot.controls
            ))
        return Decision(observation.observation_id, chosen.id,
                        {c.id: float(c is chosen) for c in candidates}, 1.0, "offline-choice", 0)


def run(env, plan, *, move=False, change_during_plan=False):
    class Planner:
        def plan(self, *_args, **_kwargs):
            if change_during_plan:
                env.snapshot = replace(env.snapshot, title="Changed", controls=())
            return plan

    return AgentRunner(Backend(env, pause=0), Planner(), policy=Choice(env, move_before_execution=move),
                       config=AgentConfig(allow_actions=True, max_plans=1, max_failures=1)).run(
        "Set wrap column to50",
    )


def test_one_plan_prepares_query_then_chooses_current_field_for_distinct_final_value(initial):
    data = raw()
    before = copy.deepcopy(data)
    plan = compile(data, initial)
    assert data == before
    assert plan.steps[0].value == "word wrap column" and plan.steps[1].property_edit.value == "50"
    assert plan.steps[1].depends_on == ("prepare",) and plan.steps[1].box is None
    env = Environment(initial)
    result = run(env, plan)
    assert result.status == "unverified" and result.planner_calls == 1
    assert result.steps == 2 and result.plan_reuses == 1
    assert result.candidate_counts[0] == 1 and result.candidate_counts[1] > 1
    assert result.semantic_steps == result.property_edits == 1
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column", "50",
    ]
    assert [(a["parameters"]["x"], a["parameters"]["y"]) for a in env.actions
            if a["action_type"] == "CLICK"] == [(200, 90), (440, 210)]


@pytest.mark.parametrize("failure", ["changed_during_plan", "no_preparation_effect", "moved_field"])
def test_failed_or_stale_preparation_and_newly_moved_field_never_type_the_final_literal(initial, failure):
    env = Environment(initial, preparation_effect=failure != "no_preparation_effect")
    result = run(env, compile(raw(), initial), move=failure == "moved_field",
                 change_during_plan=failure == "changed_during_plan")
    assert not result.success
    assert not any(a["action_type"] == "TYPING" and a["parameters"]["text"] == "50" for a in env.actions)
    if failure == "changed_during_plan":
        assert env.actions == []


@pytest.mark.parametrize("change", [
    {"id": "injected"}, {"depends_on": ["future"]}, {"navigation": True}, {"steps": []},
    {"property_edit": {"operation": "fill"}}, {"box_space": "future_pixels"},
    {"value": ""}, {"value": "\x00"},
])
def test_prepare_never_accepts_dependencies_future_schemas_or_invalid_literals(initial, change):
    data = raw()
    data["prepare"].update(change)
    with pytest.raises(ValueError):
        compile(data, initial)


@pytest.mark.parametrize("prepare", [None, [], [raw()["prepare"], raw()["prepare"]]])
def test_preparation_is_exactly_one_current_action_object(initial, prepare):
    with pytest.raises(ValueError):
        compile(raw() | {"prepare": prepare}, initial)


def test_root_prompt_prepare_example_compiles_without_an_action_wrapper(initial):
    prompt = property_protocol_instructions(6)
    marker = '"prepare":'
    offset = prompt.index(marker) + len(marker)
    preparation, _end = json.JSONDecoder().raw_decode(prompt[offset:])
    assert set(preparation) == {"operation", "ref"}
    assert preparation["operation"] == "click" and preparation["ref"] == "CURRENT_REF"
    preparation["ref"] = "help"
    data = raw(prepare=False) | {
        "prepare": preparation, "confirm": {"label": "Submit the requested property"}, "completion": "goal",
    }
    plan = parse_plan(
        data, initial, profile="desktop", max_actions=6, allow_property_goal_conditions=True,
        allow_property_preparation=True, allow_observed_property_goal=True,
    )
    _validate_desktop_execution_plan(plan, initial, 6)
    assert plan.steps[0].operation == "click" and plan.steps[0].ref == "help"
    assert plan.steps[1].property_edit.value == "50"
    assert plan.steps[2].property_confirmation_of == "property" and plan.complete_property_goal
    assert plan.completion_conditions[0].value == "50"


@pytest.mark.parametrize("preparation", [
    {"operation": "click", "ref": "help"}, raw()["prepare"],
])
def test_prepare_action_wrapper_remains_a_strict_error_without_input(initial, preparation):
    with pytest.raises(ValueError, match="missing required stage fields: operation"):
        compile(raw() | {"prepare": {"action": preparation}}, initial)


def test_existing_protocol_and_legacy_mixed_visual_wire_remain_strict(initial):
    with pytest.raises(ValueError):
        compile(raw(), initial, v29=False)
    with pytest.raises(ValueError):
        parse_plan({"summary": "mixed", "done": False, "steps": [
            {"id": "one", **raw()["prepare"]},
            {"id": "two", "operation": "fill", "target": {"name": "Wrap column", "app": "Editor"},
             "value": "50", "postconditions": raw()["goal_conditions"]},
        ]}, initial, max_actions=6, profile="desktop", allow_property_preparation=True)


@pytest.mark.parametrize("profile,limit,flag", [("browser", 6, True), ("desktop", 1, True),
                                               ("desktop", 6, 1), ("desktop", 6, "yes")])
def test_preparation_requires_explicit_desktop_multi_action_opt_in(initial, profile, limit, flag):
    with pytest.raises(ValueError, match="desktop multi-action"):
        parse_plan(raw(), initial, profile=profile, max_actions=limit, allow_property_preparation=flag)


def test_preparation_cannot_name_an_unobserved_future_application(initial):
    data = raw() | {"scope": {"app": "Foreign"}}
    with pytest.raises(ValueError, match="currently observed"):
        compile(data, initial)


class DialogEnvironment(Environment):
    def __init__(self, snapshot):
        super().__init__(snapshot)
        self.opened, self.focus, self.closed = True, "", False

    def step(self, action, pause):
        if action["action_type"] == "CLICK" and action["parameters"]["y"] >= 480:
            self.actions.append(action)
            self.closed = True
            self.snapshot = replace(self.snapshot, title="Main", controls=(
                AgentControl("main", "Main", "container", role="frame", app="Editor", window="Main",
                             active=True, box=(0, 0, 1000, 1000)),
            ))
            return self.snapshot, 0, False, {}
        return super().step(action, pause)


def dialog_snapshot(initial, *, role="dialog", value="80"):
    return replace(initial, title="Preferences", controls=(
        AgentControl("dialog", "Preferences", "container", role=role, app="Editor", window="Preferences",
                     active=True, box=(0, 0, 1000, 1000)),
        field("wrap", "Wrap column", value=value, window="Preferences", box=(400, 300, 700, 400)),
        AgentControl("ok", "OK", "button", role="button", app="Editor", window="Preferences",
                     active=True, box=(800, 800, 900, 900)),
    ))


@pytest.mark.parametrize("v29,calls,closed", [(False, 1, False), (True, 2, True)])
def test_field_effect_in_a_dialog_defers_automatic_completion_until_an_explicit_commit(
    initial, v29, calls, closed,
):
    observed = dialog_snapshot(initial)
    env, events = DialogEnvironment(observed), []

    class Planner:
        calls = 0

        def plan(self, _goal, current, history, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return compile(raw(prepare=False), current, v29=v29)
            assert history[-1]["reason"] == "property_confirmation_required"
            return compile({"intent": "edit_fields", "label": "Explicitly confirm the current value",
                            "edits": [], "completion": "goal", "commit": {
                                "operation": "click", "target": {"name": "OK", "app": "Editor"},
                                "preconditions": raw(prepare=False)["goal_conditions"],
                                "postconditions": [{"kind": "absent", "target": {
                                    "name": "Preferences", "role": "dialog", "app": "Editor",
                                }}],
                            }}, current)

    planner = Planner()
    result = AgentRunner(Backend(env, pause=0), planner, policy=Choice(env), progress=events.append,
                         config=AgentConfig(allow_actions=True)).run("Set wrap column to50")
    assert result.status == "unverified" and not result.success
    assert result.planner_calls == calls and env.closed is closed
    assert result.plan_completion_claims == 1
    deferred = [e for e in events if e["event"] == "agent_property_confirmation_required"]
    assert bool(deferred) is v29
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == ["50"]


def test_non_dialog_setting_can_still_finish_on_its_fresh_value_without_an_extra_plan(initial):
    observed = dialog_snapshot(initial, role="frame", value="50")
    plan = compile(raw(prepare=False), observed)
    assert pending_dialog_field_completion(plan, observed) is None
    assert plan.defer_dialog_field_completion is True


def test_false_or_unknown_goal_conditions_are_preserved_without_dialog_completion_inference(initial):
    observed = dialog_snapshot(initial, value="80")
    plan = compile(raw(prepare=False), observed)
    assert pending_dialog_field_completion(plan, observed) is None
    unknown = replace(observed, controls=(observed.controls[0], observed.controls[-1]))
    assert pending_dialog_field_completion(plan, unknown) is None
    old = compile(raw(prepare=False), observed, v29=False)
    assert plan.completion_conditions == old.completion_conditions


def test_an_unrelated_app_dialog_is_not_treated_as_the_current_property_confirmation(initial):
    observed = dialog_snapshot(initial, role="frame", value="50")
    observed = replace(observed, controls=observed.controls + (
        replace(observed.controls[0], ref="foreign", role="dialog", app="Foreign"),
    ))
    assert pending_dialog_field_completion(compile(raw(prepare=False), observed), observed) is None


@pytest.mark.parametrize("key,value", [("property_prepare_dependency", "prepare"),
                                       ("defer_dialog_field_completion", False)])
def test_model_cannot_override_internal_preparation_or_completion_markers(initial, key, value):
    with pytest.raises(ValueError):
        compile(raw() | {key: value}, initial)


def test_navigation_to_selected_option_in_a_dialog_also_defers_field_only_goal_completion(initial):
    observed = dialog_snapshot(initial, value="50")
    option = replace(observed.controls[1], tag="button", role="cell", editable=False,
                     input_type="", selectable=True, selected=True)
    observed = replace(observed, controls=(observed.controls[0], option, observed.controls[-1]))
    data = {"intent": "navigate", "label": "Choose requested option", "scope": {"app": "Editor"},
            "arrival": [{"kind": "selected", "target": {"name": "Wrap column", "app": "Editor"}}],
            "completion": "goal"}
    plan = compile(data, observed)
    assert plan.defer_dialog_field_completion is True
    assert pending_dialog_field_completion(plan, observed)["reason"] == "field_effect_in_open_dialog"


def test_preparation_and_property_share_the_original_total_action_budget(initial):
    plan, env = compile(raw(), initial), Environment(initial)

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Choice(env),
                         config=AgentConfig(allow_actions=True, max_steps=1)).run("Set wrap to50")
    assert result.steps == 1 and result.status == "max_steps"
    assert [a["parameters"]["text"] for a in env.actions if a["action_type"] == "TYPING"] == [
        "word wrap column",
    ]


def test_compiler_private_dependency_cannot_skip_or_replace_the_initial_preparation(initial):
    plan = compile(raw(), initial)
    with pytest.raises(ValueError, match="earlier current preparation"):
        _validate_desktop_execution_plan(replace(plan, steps=(plan.steps[1],)), initial, 6)
    altered = replace(plan.steps[1], depends_on=("foreign",))
    assert Backend(Environment(initial), pause=0).compile(altered, initial, initial) == ()
