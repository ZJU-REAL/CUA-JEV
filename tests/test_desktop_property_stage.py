"""Declarative property stages use real bindings; all policy/environment work here is offline."""

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
from cua_jev.agent_types import AgentControl, AgentPropertyEdit, AgentSnapshot, AgentTarget
from cua_jev.models import Decision, Risk
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image
from cua_jev.policy import JevPolicy


@pytest.fixture
def image():
    output = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(output, "PNG")
    return screenshot_image(output.getvalue())


def control(ref, name, *, field=False, y=100, **extra):
    return AgentControl(ref, name, "input" if field else "button",
                        role="textbox" if field else "button",
                        input_type="text" if field else "", editable=field,
                        box=(100, y, 300, y + 60), app="Editor", window="Main", active=True, **extra)


def snapshot(image, controls):
    return AgentSnapshot("osworld://desktop", "Main", "", tuple(controls), image=image)


def raw(operation="fill", value="100"):
    return {"intent": "pursue_property", "label": "Set the minimum undo history to 100",
            "scope": {"app": "Editor"}, "edit": {"operation": operation,
                                                      **({"value": value} if operation == "fill" else {})}}


def stage(observation, data=None):
    plan = parse_plan(data or raw(), observation, profile="desktop", max_actions=6)
    _validate_desktop_execution_plan(plan, observation, 6)
    return plan


class Desktop:
    """A local typed-input fixture, never a VM or independent OSWorld evaluator."""

    def __init__(self, observation, opened=None, *, pending=False):
        self.snapshot, self.opened, self.pending = observation, opened, pending
        self.actions = []

    def _get_obs(self):
        return self.snapshot

    def step(self, action, pause=0):
        self.actions.append(action)
        if action["action_type"] == "CLICK":
            if self.opened is not None:
                self.snapshot, self.opened = self.opened, None
            else:
                x, y = action["parameters"]["x"], action["parameters"]["y"]
                width, height = self.snapshot.image.region[2:]
                self.snapshot = replace(self.snapshot, controls=tuple(
                    replace(c, focused=c.editable and c.box[0] * width / 1000 <= x < c.box[2] * width / 1000
                            and c.box[1] * height / 1000 <= y < c.box[3] * height / 1000)
                    if c.box else c for c in self.snapshot.controls
                ))
        elif action["action_type"] == "TYPING":
            value = action["parameters"]["text"]
            self.snapshot = replace(self.snapshot, controls=tuple(
                replace(c, **({"edit_buffer": value} if self.pending else {"value": value}))
                if c.focused else c for c in self.snapshot.controls
            ))
        return self.snapshot, "PRIVATE reward", False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


class Select:
    def choose(self, observation, candidates):
        fields = [c for c in candidates if c.capability == "agent.osworld.fill"
                  and "Minimal" in c.description]
        chosen = fields[0] if fields else next(
            c for c in candidates if c.description.startswith("click Preferences ")
        )
        return Decision(observation.observation_id, chosen.id,
                        {c.id: float(c is chosen) for c in candidates}, 1.0, "scripted-fixture", 0)


def test_one_plan_navigates_then_binds_actual_field_without_a_guessed_future_label(image):
    initial = snapshot(image, [control("prefs", "Preferences"), control("help", "Help", y=300)])
    opened = snapshot(image, [control("field", "Minimal number of undo levels:", field=True, value="5"),
                              control("cache", "Cache size", field=True, y=300, value="20"),
                              control("ok", "OK", y=500)])
    env, calls = Desktop(initial, opened), []
    plan = stage(initial)
    assert not plan.steps[0].postconditions
    assert plan.steps[0].property_edit == AgentPropertyEdit("fill", "100")

    class Planner:
        def plan(self, *args, **kwargs):
            calls.append(kwargs["max_actions"])
            assert len(calls) == 1
            return plan

    backend = Backend(env, pause=0)
    result = AgentRunner(backend, Planner(), policy=Select(), config=AgentConfig(allow_actions=True)).run(
        "Set minimum undo history to 100",
        success=lambda _s: next((c.value for c in env.snapshot.controls if c.ref == "field"), None) == "100",
    )
    assert calls == [6] and result.planner_calls == 1 and result.plan_reuses == 1
    assert result.steps == 2 and result.candidate_counts == (2, 5)
    assert result.navigation_steps == 1 and result.navigation_subgoals == 0
    assert result.property_edits == result.property_subgoals == 1 and result.property_confirmations == 0
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK", "HOTKEY", "TYPING"]
    assert all(a["action_type"] != "PRESS" for a in env.actions)  # No implicit Enter/commit.


def test_navigation_change_does_not_claim_property_completion_and_no_change_does_not_pass(image):
    initial = snapshot(image, [control("prefs", "Preferences")])
    opened = snapshot(image, [control("help", "Help")])
    for after, expected in ((opened, True), (initial, False)):
        env = Desktop(initial, after)
        backend = Backend(env, pause=0)
        step = stage(initial).steps[0]
        candidate = backend.compile(step, initial, initial)[0]
        receipt = backend.execute(candidate, "obs", "decision")
        evidence = backend.verify(candidate, receipt)
        assert receipt.success and evidence.passed is expected
        assert evidence.details["postconditions_met"] is False
        assert evidence.details["task_success"] is False


def test_already_matching_field_is_a_fresh_read_only_confirmation_with_zero_inputs(image):
    observed = snapshot(image, [control("field", "Minimal number of undo levels:", field=True, value="100")])
    env, step = Desktop(observed), stage(observed).steps[0]
    backend = Backend(env, pause=0)
    candidate = next(c for c in backend.compile(step, observed, observed)
                     if c.arguments["operation"] == "fill")
    assert candidate.risk == Risk.READ_ONLY and candidate.arguments["property_phase"] == "confirm"
    receipt = backend.execute(candidate, "obs", "decision")
    evidence = backend.verify(candidate, receipt)
    assert evidence.passed and evidence.details["postconditions_met"]
    assert evidence.details["property_confirmation"] is True and not evidence.details["task_success"]
    assert env.actions == [] and receipt.output["environment_steps"] == 0
    actual = backend.bound_property_step(candidate, receipt)
    assert actual.operation == "fill" and actual.target.name == "Minimal number of undo levels:"
    assert backend.bound_property_step(replace(candidate, description="altered"), receipt) is None
    assert backend.bound_property_step(candidate, replace(receipt, success=False)) is None
    backend.close()
    assert backend.bound_property_step(candidate, receipt) is None


def test_private_literal_stays_local_while_jev_sees_property_operation_and_real_field_choices(image):
    literal = "PRIVATE-INPUT-LITERAL"
    observed = snapshot(image, [control("field", "Alias", field=True, value="old")])
    data = raw(value=literal)
    data["label"] = "Set the requested alias"
    plan = stage(observed, data)
    env, requests = Desktop(observed), []

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    def answer(req):
        body = json.loads(req.content)
        requests.append(body)
        actions = body["state"]["available_actions"]
        selected = next(c for c in actions if c["capability"] == "agent.osworld.fill")
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected["id"], "confidence": 1.0,
            "probabilities": {c["id"]: float(c is selected) for c in actions},
        }}})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        result = AgentRunner(Backend(env, pause=0), Planner(),
                             policy=JevPolicy(api_key="offline", client=client),
                             config=AgentConfig(allow_actions=True, max_steps=1)).run(
            "Set the requested alias",
        )
    assert result.property_edits == 1 and literal not in json.dumps(requests)
    assert env.actions[-1] == {"action_type": "TYPING", "parameters": {"text": literal}}


@pytest.mark.parametrize("buffer_updates", [False, True])
def test_committed_value_with_conflicting_numeric_buffer_is_not_a_read_only_confirmation(
    image, buffer_updates,
):
    field = replace(control("field", "Minimal undo history", field=True, value="100", focused=True),
                    role="spinbutton", input_type="number", edit_buffer="5")
    observed = snapshot(image, [field])

    class NumericDesktop(Desktop):
        def step(self, action, pause=0):
            if action["action_type"] == "TYPING" and not buffer_updates:
                self.actions.append(action)
                return self.snapshot, 0, False, {}
            return super().step(action, pause)

    env = NumericDesktop(observed, pending=True)
    backend, step = Backend(env, pause=0), stage(observed).steps[0]
    candidate = next(c for c in backend.compile(step, observed, observed)
                     if c.arguments["operation"] == "fill")
    assert candidate.risk == Risk.LOCAL_WRITE and candidate.arguments["property_phase"] == "edit"
    receipt = backend.execute(candidate, "obs", "decision")
    verification = backend.verify(candidate, receipt)
    assert verification.passed is buffer_updates
    assert verification.details["postconditions_met"] is buffer_updates
    assert verification.details["property_confirmation"] is False
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]


@pytest.mark.parametrize("change", ["geometry", "value", "disabled", "incomplete", "removed"])
def test_preflight_rejects_changed_or_missing_selected_field_before_any_input(image, change):
    field = control("field", "Minimal undo history", field=True, value="5")
    observed = snapshot(image, [field])
    backend, step = Backend(Desktop(observed), pause=0), stage(observed).steps[0]
    candidate = next(c for c in backend.compile(step, observed, observed)
                     if c.arguments["operation"] == "fill")
    altered = {"geometry": {"box": (300, 100, 500, 160)}, "value": {"value": "6"},
               "disabled": {"disabled": True}}
    backend.env.snapshot = replace(observed, a11y_complete=False) if change == "incomplete" else replace(
        observed, controls=() if change == "removed" else (replace(field, **altered[change]),)
    )
    receipt = backend.execute(candidate, "obs", "decision")
    assert not receipt.success and backend.env.actions == []
    assert receipt.output["failure_code"] == "stale_observation"


def test_same_name_fields_are_not_disambiguated_by_desired_value_and_foreign_fields_are_excluded(image):
    first = control("one", "Shared", field=True, value="100")
    second = replace(first, ref="two", value="5", box=(100, 300, 300, 360))
    foreign = replace(first, ref="foreign", app="Foreign", label="Minimal undo history")
    observed = snapshot(image, [first, second, foreign])
    backend, step = Backend(Desktop(observed), pause=0), stage(observed).steps[0]
    choices = backend.compile(step, observed, observed)
    assert choices and all(c.arguments["operation"] == "click" for c in choices)
    assert all(c.arguments["bound_control"]["app"] == "Editor" for c in choices)
    # Actual observed ancestor context may distinguish fields; value alone may not.
    second = replace(second, ancestors=("panel: Other",))
    observed = replace(observed, controls=(first, second, foreign))
    choices = backend.compile(step, observed, observed)
    fields = [c for c in choices if c.arguments["operation"] == "fill"]
    assert len(fields) == 1 and fields[0].arguments["bound_control"]["ref"] == "two"


@pytest.mark.parametrize("count,expected", [(253, 255), (254, 0)])
def test_combined_navigation_and_edit_cap_has_no_truncated_or_registered_prefix(image, count, expected):
    observed = snapshot(image, [*(control(f"button{i}", f"Action {i}") for i in range(count)),
                                control("field", "Minimal undo history", field=True, value="5")])
    plan = parse_plan(raw(), observed, profile="desktop", max_actions=6)
    backend = Backend(Desktop(observed), pause=0)
    choices = backend.compile(plan.steps[0], observed, observed)
    assert len(choices) == expected and len(backend._issued) == expected
    if not expected:
        with pytest.raises(ValueError, match="bounded current candidate"):
            _validate_desktop_execution_plan(plan, observed, 6)


@pytest.mark.parametrize("change", [
    {"edit": {"operation": "fill", "value": ""}}, {"edit": {"operation": "fill", "value": "\u4e2d"}},
    {"edit": {"operation": "check", "value": "true"}}, {"edit": {"operation": "SCRIPT"}},
    {"edit": {"operation": "fill", "value": "x", "target": {"name": "Guessed"}}},
    {"completion": "goal"}, {"commit": {"operation": "key_press", "value": "enter"}},
    {"scope": {"name": "Guessed", "app": "Editor"}}, {"property_edit": {"operation": "fill"}},
])
def test_invalid_templates_and_hidden_marker_injection_are_rejected(image, change):
    observed = snapshot(image, [control("prefs", "Preferences")])
    with pytest.raises(ValueError):
        stage(observed, raw() | change)


def test_internal_template_cannot_override_execution_guards_or_expand_observed_scope(image):
    observed = snapshot(image, [control("prefs", "Preferences")])
    step = stage(observed).steps[0]
    backend = Backend(Desktop(observed), pause=0)
    for altered in (replace(step, navigation=False), replace(step, operation="fill"),
                    replace(step, property_edit={"operation": "fill"}),
                    replace(step, target=AgentTarget(app="Foreign")),
                    replace(step, property_edit=AgentPropertyEdit("hotkey", "enter"))):
        assert backend.compile(altered, observed, observed) == ()
    assert backend.compile(step, observed, replace(observed, a11y_complete=False)) == ()


@pytest.mark.parametrize("variant", ["gtk_settings", "shell_shown_unchecked", "shell_shown_checked"])
def test_observed_switch_routes_keep_their_actual_navigation_and_property_capabilities(image, variant):
    fixture = json.loads((Path(__file__).parent / "fixtures/osworld_native_toggle.json").read_bytes())
    png = io.BytesIO()
    Image.new("RGB", tuple(fixture["screen_size"]), "white").save(png, "PNG")
    parser = OSWorldBackend(object(), pause=0)
    parsed = parser._snapshot({"screenshot": png.getvalue(),
                               "accessibility_tree": fixture["variants"][variant]["xml"]})
    native = next(c for c in parsed.controls if c.input_type == "checkbox")
    # A synthetic one-control desktop retains this recorded producer evidence.
    observed = snapshot(parsed.image, [native])
    data = raw("check")
    data["scope"] = {"app": native.app}
    step = stage(observed, data).steps[0]
    backend = Backend(Desktop(observed), pause=0)
    choices = backend.compile(step, observed, observed)
    mutations = [c for c in choices if c.arguments["operation"] == "check"]
    assert len(mutations) == 1 and "navigation" not in mutations[0].arguments
    assert len(choices) == (1 if native.native is not None else 2)
    unknown = replace(observed, controls=(replace(
        native, native=replace(native.native, checked_state=None),
    ),)) if native.native is not None else replace(observed, a11y_complete=False)
    assert backend.compile(step, observed, unknown) == ()


def test_recorded_numeric_buffer_remains_uncommitted_and_never_gets_automatic_enter(image):
    fixture = json.loads((Path(__file__).parent / "fixtures/osworld_pending_numeric_edit.json").read_bytes())
    png = io.BytesIO()
    Image.new("RGB", tuple(fixture["screen_size"]), "white").save(png, "PNG")
    parser = OSWorldBackend(object(), pause=0)
    observed = parser._snapshot({"screenshot": png.getvalue(), "accessibility_tree": fixture["xml"]})
    data = raw()
    data["scope"] = {"app": "gimp"}
    plan = stage(observed, data)
    env, events = Desktop(observed, pending=True), []
    backend = Backend(env, pause=0)

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    result = AgentRunner(backend, Planner(), policy=Select(), progress=events.append,
                         config=AgentConfig(allow_actions=True, max_steps=1)).run("Set undo levels to 100")
    assert not result.success and result.property_subgoals == 0
    feedback = [e for e in events if e.get("event") == "agent_recovery" and "pending_numeric_edit" in e]
    assert feedback and feedback[-1]["operation"] == "fill"
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]
    assert next(c for c in env.snapshot.controls if c.editable).value == "5.0"
    assert not any(e.get("event") == "agent_property_stage" for e in events)


@pytest.mark.parametrize("initial_value", ["5", "100"])
def test_verified_property_reports_selected_field_without_whole_goal_or_repeated_stage_claim(
    image, initial_value,
):
    observed = snapshot(image, [control("field", "Minimal undo history", field=True, value=initial_value)])
    env, events = Desktop(observed), []
    plan = stage(observed)

    class Planner:
        calls = 0

        def plan(self, _goal, current, history, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return plan
            feedback = history[-1]["property_stage_result"]
            assert feedback["status"] == "selected_field_verified"
            assert feedback["whole_goal_proven"] is False and "value" not in feedback
            assert feedback["target"]["name"] == "Minimal undo history"
            assert "Do not repeat" in feedback["next"] and "guarding" in feedback["next"]
            return parse_plan({"intent": "complete", "label": "Requested field is set", "conditions": [{
                "kind": "value_equals", "target": {"name": "Minimal undo history", "app": "Editor"},
                "value": "100",
            }]}, current, max_actions=6, profile="desktop")

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Select(), progress=events.append,
                         config=AgentConfig(allow_actions=True)).run("Set undo history to100")
    assert result.planner_calls == 2 and not result.success
    reports = [e for e in events if e.get("event") == "agent_property_stage"]
    assert len(reports) == 1 and reports[0]["whole_goal_proven"] is False


def request(data, observed, *, contract="property_intent_v25", limit=6, history=()):
    requests = []

    def answer(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(data)}}]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as client:
        planner = MultimodalPlanner("https://offline.invalid/v1", "mock", client=client,
                                    profile="desktop", desktop_contract=contract)
        plan = planner.plan("Change the property", observed, list(history), max_actions=limit)
    return plan, requests


def test_new_protocol_offers_property_intent_and_keeps_old_contract_separate(image):
    observed = snapshot(image, [control("prefs", "Preferences")])
    plan, calls = request(raw(), observed)
    assert plan.steps[0].property_edit is not None
    assert "pursue_property" in calls[0]["messages"][0]["content"]
    assert "PENDING NUMERIC EDITS" in calls[0]["messages"][0]["content"]
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(raw(), observed, contract="stage_intent_v24")
    old = {"intent": "navigate_edit", "label": "Future field", "scope": {"app": "Editor"},
           "edit": {"operation": "fill", "target": {"name": "Later", "app": "Editor"}, "value": "100"}}
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(old, observed)


@pytest.mark.parametrize("limit,history", [(1, []), (6, [{"recovery_required": True}])])
def test_baseline_and_recovery_request_bytes_stay_identical_across_all_contracts(image, limit, history):
    observed = snapshot(image, [control("field", "Minimal", field=True, value="5")])
    immediate = {"summary": "Fill current field", "done": False, "steps": [{
        "id": "one", "operation": "fill", "target": {"name": "Minimal", "app": "Editor"}, "value": "100",
        "postconditions": [{"kind": "value_equals", "target": {"name": "Minimal", "app": "Editor"},
                            "value": "100"}],
    }]}
    requests = [request(copy.deepcopy(immediate), observed, contract=c, limit=limit, history=history)[1]
                for c in ("legacy_v23", "stage_intent_v24", "property_intent_v25", "property_intent_v28",
                          "property_intent_v29", "property_intent_v30", "property_intent_v31")]
    assert all(item == requests[0] for item in requests)


def goal_condition():
    return {"kind": "value_equals", "target": {"name": "Minimal undo history", "app": "Editor"},
            "value": "100"}


def goal_stage(observed, conditions=None):
    data = raw() | {"goal_conditions": conditions if conditions is not None else [goal_condition()]}
    plan = parse_plan(data, observed, profile="desktop", max_actions=6, allow_property_goal_conditions=True)
    _validate_desktop_execution_plan(plan, observed, 6)
    return plan


@pytest.mark.parametrize("with_goals,expected_calls", [(False, 2), (True, 1)])
def test_read_only_goal_predicates_avoid_a_separate_completion_plan_after_navigation(
    image, with_goals, expected_calls,
):
    initial = snapshot(image, [control("prefs", "Preferences"), control("help", "Help", y=300)])
    opened = snapshot(image, [control("field", "Minimal undo history", field=True, value="5"),
                              control("cache", "Cache size", field=True, y=300, value="20")])
    env, events = Desktop(initial, opened), []

    class Planner:
        def plan(self, _goal, current, _history, **_kwargs):
            if any(c.editable and c.value == "100" for c in current.controls):
                assert not with_goals
                complete = {"intent": "complete", "label": "Goal met", "conditions": [goal_condition()]}
                return parse_plan(complete, current, profile="desktop", max_actions=6)
            return goal_stage(current) if with_goals else stage(current)

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Select(), progress=events.append,
                         config=AgentConfig(allow_actions=True)).run("Set minimum undo history to 100")
    assert result.status == "unverified" and not result.success
    assert result.planner_calls == expected_calls and result.plan_reuses == 1
    assert result.steps == 2 and result.candidate_counts == (2, 4)
    assert result.property_edits == result.property_subgoals == result.plan_completion_claims == 1
    assert [a["action_type"] for a in env.actions] == ["CLICK", "CLICK", "HOTKEY", "TYPING"]
    claims = [e for e in events if e["event"] == "agent_plan_completion"]
    assert len(claims) == 1 and claims[0]["task_success"] is False


@pytest.mark.parametrize("kind", ["false_value", "missing_field", "pending_save", "incomplete", "regressed"])
def test_field_verification_never_substitutes_for_fresh_whole_goal_evidence(image, kind):
    field = control("field", "Minimal undo history", field=True, value="5")
    controls = [field, control("save", "Save", y=300)] if kind == "pending_save" else [field]
    observed = snapshot(image, controls)
    env, events = Desktop(observed), []
    condition = goal_condition()
    if kind == "false_value":
        condition["value"] = "200"
    elif kind == "missing_field":
        condition["target"]["name"] = "Unobserved property"
    conditions = [condition]
    if kind in {"pending_save", "incomplete"}:
        conditions.append({"kind": "absent", "target": {"name": "Save", "app": "Editor"}})
    plan = goal_stage(observed, conditions)

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    def progress(event):
        events.append(event)
        # Invalidate the effect only AFTER execution/verification have succeeded.
        # The cached post-action frame must not complete the task.
        if event["event"] == "agent_step" and event["verified"]:
            if kind == "incomplete":
                env.snapshot = replace(env.snapshot, a11y_complete=False)
            elif kind == "regressed":
                env.snapshot = replace(env.snapshot, controls=(replace(env.snapshot.controls[0], value="5"),))

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Select(), progress=progress,
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Set undo history to 100")
    assert result.status == "max_plans" and result.property_subgoals == 1
    assert result.plan_completion_claims == 0 and not result.success
    assert not any(e["event"] == "agent_plan_completion" for e in events)
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]


def test_pending_numeric_buffer_cannot_complete_even_when_read_only_goal_predicate_holds(image):
    field = replace(control("field", "Minimal undo history", field=True, value="5", focused=True),
                    role="spinbutton", input_type="number", edit_buffer="5")
    observed = snapshot(image, [field])
    env = Desktop(observed, pending=True)
    plan = goal_stage(observed, [{"kind": "exists", "target": {"name": field.label, "app": "Editor"}}])

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Select(),
                         config=AgentConfig(allow_actions=True, max_steps=1)).run("Set undo history to 100")
    assert result.property_subgoals == result.plan_completion_claims == 0 and not result.success
    assert [a["action_type"] for a in env.actions] == ["CLICK", "HOTKEY", "TYPING"]


def test_goal_predicates_never_choose_a_field_or_expand_input_scope(image):
    observed = snapshot(image, [control("prefs", "Preferences"),
                                control("field", "Minimal undo history", field=True, value="5"),
                                replace(control("foreign", "Other property", field=True), app="Other")])
    old = stage(observed)
    new = goal_stage(observed, [{"kind": "checked", "target": {"name": "Future", "app": "Other"}}])
    assert old.steps == new.steps
    backend = Backend(Desktop(observed), pause=0)
    before = backend.compile(old.steps[0], observed, observed)
    after = backend.compile(new.steps[0], observed, observed)
    assert tuple(replace(c, id="stable") for c in before) == tuple(replace(c, id="stable") for c in after)
    assert len(before) == 3
    assert all(c.arguments["bound_control"]["app"] == "Editor" for c in after)


@pytest.mark.parametrize("conditions", [[], None, {}, "done", [True],
    [{"kind": "ready_to_fill", "target": {"name": "Field"}, "value": "100"}],
    [{"kind": "checked", "target": {"name": "Field", "checked": True}}],
    [{"kind": "value_equals", "target": {"name": "Field"}}],
    [{"kind": "exists", "target": {"app": "Editor"}}],
    [{"kind": "exists", "target": {"name": "Field"}}] * 9,
])
def test_opt_in_goal_conditions_preserve_original_condition_grammar(image, conditions):
    observed = snapshot(image, [control("prefs", "Preferences")])
    with pytest.raises(ValueError):
        parse_plan(raw() | {"goal_conditions": conditions}, observed, profile="desktop",
                   max_actions=6, allow_property_goal_conditions=True)


@pytest.mark.parametrize("profile,limit,flag", [("desktop", 1, True), ("browser", 6, True),
                                               ("desktop", 6, 1), ("desktop", 6, "yes")])
def test_goal_conditions_require_explicit_desktop_multi_action_opt_in(image, profile, limit, flag):
    with pytest.raises(ValueError, match="desktop multi-action"):
        parse_plan(raw(), snapshot(image, [control("prefs", "Preferences")]), profile=profile,
                   max_actions=limit, allow_property_goal_conditions=flag)


def test_v28_protocol_accepts_read_only_goals_while_v25_remains_strict(image):
    observed = snapshot(image, [control("prefs", "Preferences")])
    data = raw() | {"goal_conditions": [goal_condition()]}
    plan, calls = request(data, observed, contract="property_intent_v28")
    assert plan.completion_conditions and plan.steps == stage(observed).steps
    prompt = calls[0]["messages"][0]["content"]
    assert '"goal_conditions"' in prompt and "read-only whole-goal predicates" in prompt
    assert "NEW observation" in prompt
    assert "ONLY these fields; no target, arrival, completion" not in prompt
    assert "No future target query or arrival predicate" not in prompt
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(data, observed, contract="property_intent_v25")
    with pytest.raises(ValueError, match="requires exactly"):
        parse_plan(data, observed, profile="desktop", max_actions=6)


def test_v29_production_request_accepts_current_preparation_and_keeps_v28_strict(image):
    observed = snapshot(image, [control("prefs", "Preferences")])
    data = raw() | {"prepare": {"operation": "visual_fill", "value": "retained revisions",
                               "box_space": "normalized_1000", "box": [100, 100, 300, 200]}}
    plan, calls = request(data, observed, contract="property_intent_v29")
    assert len(plan.steps) == 2 and plan.steps[0].value == "retained revisions"
    assert plan.steps[1].property_edit.value == "100" and plan.defer_dialog_field_completion
    assert "prepare.value" in calls[0]["messages"][0]["content"]
    with pytest.raises(ValueError):
        request(data, observed, contract="property_intent_v28")


@pytest.mark.parametrize("operation", ["check", "uncheck"])
def test_v28_accepts_only_the_explicit_empty_unused_boolean_value_without_mutating_response(image, operation):
    observed = snapshot(image, [control("prefs", "Preferences")])
    data = raw(operation) | {"edit": {"operation": operation, "value": ""}}
    original = copy.deepcopy(data)
    plan, _ = request(data, observed, contract="property_intent_v28")
    assert data == original and plan.steps[0].property_edit.value == ""
    with pytest.raises(ValueError):
        request(data, observed, contract="property_intent_v25")


@pytest.mark.parametrize("value", [None, False, "on", "100"])
def test_boolean_property_value_never_authorizes_a_nonempty_or_nontyped_literal(image, value):
    data = raw("check") | {"edit": {"operation": "check", "value": value}}
    with pytest.raises(ValueError):
        request(data, snapshot(image, [control("prefs", "Preferences")]), contract="property_intent_v28")


@pytest.mark.parametrize("independent_rejects", [False, True])
def test_v28_screenshot_completion_has_same_unverified_stop_as_direct_vlm(image, independent_rejects):
    observed = snapshot(image, [control("prefs", "Preferences")])
    raw_complete = {"intent": "complete", "label": "The requested new name is visible in the screenshot"}
    hybrid, _ = request(raw_complete, observed, contract="property_intent_v28")
    direct = parse_plan({"summary": raw_complete["label"], "steps": [], "done": True},
                        observed, profile="desktop", max_actions=1)
    assert hybrid == direct and hybrid.done and not hybrid.completion_conditions

    class Planner:
        def __init__(self, plan):
            self.plan_to_return = plan

        def plan(self, *_args, **_kwargs):
            return self.plan_to_return

    results = []
    for mode, plan in (("hybrid", hybrid), ("vlm", direct)):
        env = Desktop(observed)
        result = AgentRunner(Backend(env, pause=0), Planner(plan), policy=Select(),
                             config=AgentConfig(mode=mode, allow_actions=True, max_plans=1)).run(
            "Rename an object", success=(lambda _snapshot: False) if independent_rejects else None,
        )
        results.append(result)
        assert not result.success and result.plan_completion_claims == 0
        assert result.planner_calls == 1 and result.steps == 0 and env.actions == []
    assert results[0].status == results[1].status == ("max_plans" if independent_rejects else "unverified")
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(raw_complete, observed, contract="property_intent_v25")


@pytest.mark.parametrize("condition", [
    {"kind": "exists", "target": {"name": "Unobservable new name", "app": "Editor"}},
    {"kind": "value_equals", "target": {"name": "Minimal undo history", "app": "Editor"}, "value": "100"},
])
def test_v28_never_discards_false_or_unknown_declared_completion_predicates(image, condition):
    observed = snapshot(image, [control("field", "Minimal undo history", field=True, value="5")])
    plan, _ = request({"intent": "complete", "label": "Alleged completion", "conditions": [condition]},
                      observed, contract="property_intent_v28")
    assert plan.completion_conditions

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    env = Desktop(observed)
    result = AgentRunner(Backend(env, pause=0), Planner(), policy=Select(),
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Change the field")
    assert result.status == "max_plans" and not result.success
    assert result.plan_completion_claims == 0 and env.actions == []


@pytest.mark.parametrize("extra", [{"conditions": []}, {"conditions": None}, {"evidence": "invented field"},
                                   {"done": True}, {"action": {"operation": "click"}}])
def test_v28_advisory_completion_rejects_malformed_evidence_and_hidden_actions(image, extra):
    with pytest.raises(ValueError, match="invalid multimodal"):
        request({"intent": "complete", "label": "Visible completed goal", **extra},
                snapshot(image, [control("prefs", "Preferences")]), contract="property_intent_v28")


@pytest.mark.parametrize("profile,limit,flag", [("desktop", 1, True), ("browser", 6, True),
                                               ("desktop", 6, 1), ("desktop", 6, "yes")])
def test_advisory_completion_opt_in_is_only_for_desktop_multi_action_protocol(image, profile, limit, flag):
    with pytest.raises(ValueError, match="desktop multi-action"):
        parse_plan({"intent": "complete", "label": "Visible goal"}, snapshot(image, []), profile=profile,
                   max_actions=limit, allow_advisory_completion=flag)


@pytest.mark.parametrize("operation", ["visual_click", "visual_fill"])
def test_v28_visual_action_inherits_its_stage_description_without_inventing_input(image, operation):
    observed = snapshot(image, [])
    action = {"operation": operation, "box": [358, 126, 740, 155], "box_space": "normalized_1000"}
    if operation == "visual_fill":
        action["value"] = "query"
    data = {"intent": "act_current", "label": "Search current settings", "action": action}
    original = copy.deepcopy(data)
    inherited, _ = request(data, observed, contract="property_intent_v28")
    explicit, _ = request({**data, "action": {**action, "label": data["label"]}},
                          observed, contract="property_intent_v28")
    assert inherited == explicit and inherited.steps[0].box == (358, 126, 740, 155)
    assert data == original
    with pytest.raises(ValueError, match="invalid multimodal"):
        request(data, observed, contract="property_intent_v25")


@pytest.mark.parametrize("label", [None, "", "x" * 241, True])
def test_v28_never_overwrites_an_explicit_invalid_visual_action_label(image, label):
    with pytest.raises(ValueError, match="invalid multimodal"):
        request({"intent": "act_current", "label": "Stage description", "action": {
            "operation": "visual_click", "box": [100, 100, 300, 200], "label": label,
        }}, snapshot(image, []), contract="property_intent_v28")


@pytest.mark.parametrize("profile,limit,flag", [("desktop", 1, True), ("browser", 6, True),
                                               ("desktop", 6, 1), ("desktop", 6, "yes")])
def test_current_action_label_inheritance_requires_desktop_protocol_opt_in(image, profile, limit, flag):
    with pytest.raises(ValueError, match="desktop multi-action"):
        parse_plan({}, snapshot(image, []), profile=profile, max_actions=limit,
                   inherit_current_action_label=flag)
