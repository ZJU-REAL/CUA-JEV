"""Explicit editing intent compiles to real navigation and verified mutation."""

import copy
import io
import json
from dataclasses import replace
from xml.sax.saxutils import quoteattr

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import MultimodalPlanner, _desktop_instructions, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentControl, AgentSnapshot, AgentTarget
from cua_jev.desktop_semantics import condition_evidence, condition_holds, conditions_hold
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.policy import JevPolicy
from cua_jev.trace import JsonlTrace


def intent(operation="check", *, completion="goal"):
    edit = {"operation": operation, "target": {"name": "Option", "app": "Fixture"}}
    if operation == "fill":
        edit["value"] = "new"
    return {"intent": "reach_edit", "label": "Set the requested option",
            "scope": {"app": "Fixture"}, "edit": edit, "completion": completion}


def control(**changes):
    return AgentControl(**({
        "ref": "field", "label": "Option", "tag": "input", "role": "checkbox",
        "input_type": "checkbox", "app": "Fixture", "window": "Settings",
        "box": (100, 100, 300, 180),
    } | changes))


def snapshot(*controls, complete=True):
    return AgentSnapshot("osworld://desktop", "Settings", "Settings", tuple(controls),
                         a11y_complete=complete)


def menu_snapshot():
    return snapshot(control(ref="menu", label="Settings", tag="button", role="menuitem", input_type=""))


def compiled(raw=None, state=None):
    return parse_plan(intent() if raw is None else raw, state or menu_snapshot(), profile="desktop")


@pytest.mark.parametrize("operation,effect", [("check", "checked"), ("uncheck", "unchecked"),
                                             ("fill", "value_equals")])
def test_future_edit_compiles_to_navigation_readiness_then_verified_mutation(operation, effect):
    state = menu_snapshot()
    plan = compiled(intent(operation), state)
    reach, edit = plan.steps
    assert plan.reusable and not plan.done
    assert reach.navigation and reach.target == AgentTarget(app="Fixture")
    assert edit.operation == operation and edit.depends_on == (reach.id,)
    assert reach.postconditions == edit.preconditions
    assert reach.postconditions[0].kind == f"ready_to_{operation}"
    assert edit.postconditions[0].kind == effect
    assert edit.postconditions[1] == reach.postconditions[0]
    assert plan.completion_conditions == edit.postconditions
    assert condition_evidence(reach.postconditions[0], state) == {
        "outcome": "unknown", "reason": "target_missing",
    }


def test_stage_default_does_not_claim_entire_goal_or_invent_commit():
    raw = intent("fill")
    del raw["completion"]
    plan = compiled(raw)
    assert not plan.completion_conditions and not plan.done
    assert [s.operation for s in plan.steps] == ["click", "fill"]
    assert all(not s.ref and s.box is None for s in plan.steps)


@pytest.mark.parametrize("change", [
    lambda r: r.update(extra=True),
    lambda r: r.update(steps=[]),
    lambda r: r.update(intent="other"),
    lambda r: r.update(label=" "),
    lambda r: r.update(label="x" * 241),
    lambda r: r.update(completion=True),
    lambda r: r.update(completion="done"),
    lambda r: r["scope"].update(name="Menu"),
    lambda r: r["scope"].update(role="button"),
    lambda r: r.update(scope={"ancestor": "Settings"}),
    lambda r: r["edit"].update(operation="click"),
    lambda r: r["edit"].update(value="private-value"),
    lambda r: r["edit"].update(button="right"),
    lambda r: r["edit"].update(target={"role": "checkbox", "app": "Fixture"}),
    lambda r: r["edit"].update(target={"name": "Option"}),
    lambda r: r["edit"]["target"].update(selector="private-value"),
    lambda r: r["edit"]["target"].update(name="x" * 241),
])
def test_stage_rejects_mixed_or_ambiguous_schema_without_echoing_values(change):
    raw = intent()
    change(raw)
    with pytest.raises(ValueError) as error:
        compiled(raw)
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize("value", [None, "", "\x00private", "秘密", "x" * 2001])
def test_fill_retains_existing_literal_input_constraints(value):
    raw = intent("fill")
    raw["edit"]["value"] = value
    with pytest.raises(ValueError):
        compiled(raw)


@pytest.mark.parametrize("role", [
    "checkbox", "check-box", "check-menu-item", "toggle-button", "switch", "toggle",
    "radio", "radio-button", "radio-menu-item", "CHECKBOX",
])
def test_stage_fill_rejects_explicit_checkable_role_without_changing_legacy(role):
    raw = intent("fill")
    raw["edit"]["target"]["role"] = role
    raw["edit"]["value"] = "private-value"
    with pytest.raises(ValueError, match="role conflicts with fill") as error:
        compiled(raw)
    assert "private-value" not in str(error.value)
    # The new envelope is stricter; do not silently broaden this change into
    # legacy schema behavior or translate the request into another operation.
    legacy = {"summary": "Legacy", "done": False, "steps": [{
        "id": "edit", **raw["edit"], "postconditions": [{
            "kind": "value_equals", "target": raw["edit"]["target"], "value": "private-value",
        }],
    }]}
    assert compiled(legacy).steps[0].operation == "fill"


@pytest.mark.parametrize("role", ["", "textbox", "entry", "spinbutton", "spin-button"])
def test_stage_fill_still_allows_unknown_future_role_and_text_roles(role):
    raw = intent("fill")
    if role:
        raw["edit"]["target"]["role"] = role
    assert compiled(raw).steps[1].operation == "fill"


def test_live_planner_reports_stage_fill_role_conflict_without_echoing_literal():
    raw = intent("fill")
    raw["edit"]["target"]["role"] = "checkbox"
    raw["edit"]["value"] = "private-value"
    planner = planner_for(raw, [])
    with pytest.raises(ValueError):
        planner.plan("Set the requested field", menu_snapshot(), [])
    assert "reach_edit.edit.target.role conflicts with fill" in planner.last_validation_error
    assert "private-value" not in planner.last_validation_error


@pytest.mark.parametrize("profile,limit", [("desktop", 1), ("browser", 1), ("browser", 6)])
def test_baseline_recovery_and_browser_cannot_accept_stage_envelope(profile, limit):
    with pytest.raises(ValueError, match="only for desktop multi-action"):
        parse_plan(intent(), menu_snapshot(), profile=profile, max_actions=limit)


@pytest.mark.parametrize("location", ["preconditions", "postconditions", "completion_conditions"])
def test_model_cannot_author_internal_readiness_predicates(location):
    target = {"name": "Option", "app": "Fixture"}
    raw = {"summary": "Edit", "done": False, "steps": [{
        "id": "edit", "operation": "check", "target": target,
        "postconditions": [{"kind": "checked", "target": target}],
    }]}
    destination = raw if location == "completion_conditions" else raw["steps"][0]
    destination[location] = [{"kind": "ready_to_check", "target": target}]
    with pytest.raises(ValueError, match="unsupported condition kind"):
        parse_plan(raw, menu_snapshot(), profile="desktop")


def test_legacy_exact_click_is_not_silently_changed_to_navigation():
    raw = {"summary": "Click", "done": False, "steps": [{
        "id": "click", "operation": "click", "target": {"name": "Settings", "app": "Fixture"},
        "postconditions": [{"kind": "focused", "target": {"name": "Settings", "app": "Fixture"}}],
    }]}
    plan = compiled(raw)
    assert not plan.steps[0].navigation and plan.steps[0].target.name == "Settings"


@pytest.mark.parametrize("operation", ["check", "uncheck", "fill"])
def test_readiness_requires_unique_capable_target_before_testing_its_value(operation):
    step = compiled(intent(operation)).steps[0]
    field = control(checked=operation == "check")
    if operation == "fill":
        field = control(role="textbox", input_type="text", editable=True, value="new")
    label = control(ref="label", tag="text", role="label", input_type="", box=None)
    desired = snapshot(label, field)
    assert condition_holds(step.postconditions[0], desired)
    duplicate = replace(field, ref="other", checked=not field.checked, value="wrong")
    ambiguous = condition_evidence(step.postconditions[0], snapshot(label, field, duplicate))
    assert ambiguous == {"outcome": "unknown", "reason": "ambiguous_target"}
    assert condition_evidence(step.postconditions[0], snapshot(label)) == {
        "outcome": "unknown", "reason": "unsupported_state",
    }


def test_readonly_desired_state_skips_but_wrong_state_is_not_actionable():
    ready = compiled().steps[0].postconditions[0]
    readonly = control(disabled=True, box=None, checked=True)
    assert condition_holds(ready, snapshot(readonly))
    assert condition_evidence(ready, snapshot(replace(readonly, checked=False))) == {
        "outcome": "false", "reason": "state_mismatch",
    }
    assert condition_holds(ready, snapshot(control(checked=False)))


@pytest.mark.parametrize("operation", ["check", "uncheck", "fill"])
def test_partial_tree_cannot_prove_readiness_effect_or_compiled_goal(operation):
    field = control(checked=operation == "check")
    if operation == "fill":
        field = control(role="textbox", input_type="text", editable=True, value="new")
    plan = compiled(intent(operation))
    state = snapshot(field, complete=False)
    # Ordinary legacy evidence stays compatible; the compiled plan adds its
    # stricter readiness proof to BOTH effect and whole-goal stopping checks.
    assert condition_holds(plan.steps[1].postconditions[0], state)
    assert condition_evidence(plan.steps[0].postconditions[0], state) == {
        "outcome": "unknown", "reason": "incomplete_observation",
    }
    assert not conditions_hold(plan.steps[1].postconditions, state)
    assert not conditions_hold(plan.completion_conditions, state)


def test_numeric_committed_value_and_textbox_rules_are_reused():
    raw = intent("fill")
    raw["edit"]["value"] = "50"
    plan = compiled(raw)
    field = control(role="spinbutton", input_type="number", editable=False, disabled=True,
                    box=None, value="50.0")
    assert conditions_hold(plan.completion_conditions, snapshot(field))
    assert not conditions_hold(plan.completion_conditions, snapshot(replace(field, role="textbox")))
    assert not conditions_hold(plan.completion_conditions, snapshot(replace(field, value="49")))


class EditingDesktop:
    """Three real navigation states followed by one actual input target."""

    def __init__(self, operation="check", *, ignore_edit=False):
        self.operation, self.ignore_edit = operation, ignore_edit
        self.phase, self.focused, self.value = 0, False, "old"
        self.checked = operation == "uncheck"
        self.actions = []

    def _get_obs(self):
        label = '<label name="Option"/>'
        if self.phase < 3:
            name = ("Configure", "Preferences", "Advanced")[self.phase]
            content = (
                '<button name="Help" enabled="true" screencoord="(20, 50)" size="(140, 40)"/>'
                f'<button name="{name}" enabled="true" screencoord="(400, 50)" size="(140, 40)"/>'
            )
        elif self.operation == "fill":
            content = (
                f'<entry name="Option" enabled="true" editable="true" value={quoteattr(self.value)} '
                f'focused="{str(self.focused).lower()}" screencoord="(220, 220)" size="(180, 40)"/>'
            )
        else:
            content = (
                f'<check-box name="Option" enabled="true" checked="{str(self.checked).lower()}" '
                'screencoord="(220, 220)" size="(180, 40)"/>'
            )
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), (255 - self.phase * 30, 230, 220)).save(buffer, "PNG")
        return {"screenshot": buffer.getvalue(), "accessibility_tree": (
            '<desktop><application name="Fixture"><frame name="Settings" active="true">'
            + label + content + '</frame></application></desktop>'
        )}

    def step(self, action, pause=0):
        self.actions.append(action)
        if self.phase < 3:
            assert action["action_type"] == "CLICK" and action["parameters"]["x"] > 300
            self.phase += 1
        elif self.operation == "fill":
            if action["action_type"] == "CLICK":
                self.focused = True
            elif action["action_type"] == "TYPING" and not self.ignore_edit:
                self.value = action["parameters"]["text"]
            else:
                assert action["action_type"] == "HOTKEY" or self.ignore_edit
        elif not self.ignore_edit:
            assert action["action_type"] == "CLICK"
            self.checked = not self.checked
        return self._get_obs(), "PRIVATE REWARD", False, {}


def planner_for(raw, requests):
    def answer(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(raw)}}]})
    return MultimodalPlanner("https://model.invalid/v1", "test", api_key="test",
                             profile="desktop", client=httpx.Client(transport=httpx.MockTransport(answer)))


@pytest.mark.parametrize("operation", ["check", "uncheck", "fill"])
def test_one_real_planner_call_drives_three_jev_hops_and_exact_edit(operation):
    desktop, requests, jev_requests = EditingDesktop(operation), [], []
    planner = planner_for(intent(operation), requests)
    backend = OSWorldBackend(desktop, pause=0)

    def choose(request):
        body = json.loads(request.content)
        jev_requests.append(body)
        candidates = body["state"]["available_actions"]
        name = ("Configure", "Preferences", "Advanced")[desktop.phase]
        selected = next(c for c in candidates if f" {name} [" in c["description"])
        assert all(not candidate.get("arguments") for candidate in candidates)
        return httpx.Response(200, json={"answers": {"action": {
            "choice": selected["id"], "confidence": 1,
            "probabilities": {c["id"]: float(c is selected) for c in candidates},
        }}})

    with httpx.Client(transport=httpx.MockTransport(choose)) as client:
        result = AgentRunner(backend, planner, policy=JevPolicy(api_key="test", client=client, retries=0),
                             config=AgentConfig(allow_actions=True)).run("Set Option")
    backend.close()
    assert result.status == "unverified" and not result.success
    assert result.planner_calls == len(requests) == 1
    assert result.steps == 4 and result.plan_reuses == 3
    assert result.candidate_counts == (2, 2, 2, 1)
    assert result.jev_calls == len(jev_requests) == 3 and result.singleton_decisions == 1
    assert result.plan_completion_claims == 1
    assert desktop.value == "new" if operation == "fill" else desktop.checked == (operation == "check")
    prompt = requests[0]["messages"][0]["content"]
    assert prompt.startswith("PLAN ONE MEANINGFUL DESKTOP STAGE.")
    assert prompt.index("reach_edit") < prompt.index("LEGACY format:")


def test_failed_actual_edit_does_not_claim_goal_or_repeat_input_during_pending_checks():
    desktop, requests, trace = EditingDesktop(ignore_edit=True), [], JsonlTrace()
    desktop.phase = 3
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(backend, planner_for(intent(), requests), policy=DirectPlanPolicy(), trace=trace,
                         config=AgentConfig(allow_actions=True, max_failures=1)).run("Set Option")
    backend.close()
    assert result.status == "stuck" and result.plan_completion_claims == 0
    assert len(desktop.actions) == result.steps == 1
    assert result.recovery_observations == 2
    assert not any(e["kind"] == "agent_plan_completion" for e in trace.events)


def test_post_input_partial_observation_cannot_verify_or_complete_stage(monkeypatch):
    desktop, requests = EditingDesktop(), []
    desktop.phase = 3
    backend = OSWorldBackend(desktop, pause=0)
    original = backend._snapshot
    monkeypatch.setattr(backend, "_snapshot", lambda observation: replace(
        original(observation), a11y_complete=not desktop.checked,
    ))
    result = AgentRunner(backend, planner_for(intent(), requests), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True, max_failures=1)).run("Set Option")
    backend.close()
    assert desktop.checked and len(desktop.actions) == 1
    assert result.status == "stuck" and result.plan_completion_claims == 0
    assert result.semantic_steps == 0 and result.recovery_observations == 2


def test_live_validator_rejects_incomplete_initial_ready_and_goal_state():
    requests = []
    planner = planner_for(intent(), requests)
    with pytest.raises(ValueError):
        planner.plan("Set Option", snapshot(control(checked=True), complete=False), [])
    assert "active navigation scope" in planner.last_validation_error


def test_prompt_and_parser_keep_single_action_and_browser_legacy_contracts():
    prompt = _desktop_instructions(1, menu_snapshot())
    assert "reach_edit" not in prompt and "ready_to_" not in prompt
    raw = {"summary": "Check", "done": False, "steps": [{
        "id": "one", "operation": "check", "ref": "field",
    }]}
    state = snapshot(control())
    assert parse_plan(copy.deepcopy(raw), state, 1).steps[0].operation == "check"
    assert parse_plan(copy.deepcopy(raw), state, 1, profile="desktop").steps[0].operation == "check"


def test_fresh_top_level_completion_does_not_accept_later_partial_tree(monkeypatch):
    desktop, requests = EditingDesktop(), []
    desktop.phase = 3
    backend = OSWorldBackend(desktop, pause=0)
    observe = backend.observe
    monkeypatch.setattr(backend, "observe", lambda: replace(
        observe(), a11y_complete=not desktop.checked,
    ))
    # Step's returned observation is complete and verifies the input. Only the
    # later independent loop observation becomes incomplete.
    result = AgentRunner(backend, planner_for(intent(), requests), policy=DirectPlanPolicy(),
                         config=AgentConfig(allow_actions=True, max_plans=1)).run("Set Option")
    backend.close()
    assert result.steps == result.semantic_steps == len(desktop.actions) == 1
    assert result.plan_completion_claims == 0 and not result.success


def test_stage_completion_requires_later_planning_instead_of_whole_goal_claim():
    desktop, requests = EditingDesktop(), []
    desktop.phase = 3
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner_for(intent(completion="stage"), requests), policy=DirectPlanPolicy(),
        config=AgentConfig(allow_actions=True, max_plans=1),
    ).run("Set Option then save another object")
    backend.close()
    assert result.steps == result.semantic_steps == 1 and desktop.checked
    assert result.plan_completion_claims == 0 and not result.success


def test_navigation_hops_still_consume_the_existing_horizon():
    class ChooseUseful(DirectPlanPolicy):
        def choose(self, observation, candidates):
            return super().choose(observation, sorted(candidates, key=lambda c: "Help" in c.description))

    desktop, requests = EditingDesktop(), []
    backend = OSWorldBackend(desktop, pause=0)
    result = AgentRunner(
        backend, planner_for(intent(), requests), policy=ChooseUseful(),
        config=AgentConfig(allow_actions=True, plan_horizon=2, max_plans=1),
    ).run("Set Option")
    backend.close()
    assert len(desktop.actions) == result.steps == 2 and desktop.phase == 2
    assert not desktop.checked and result.plan_completion_claims == 0


def test_compiled_future_app_uses_existing_receipt_wait_and_handoff(monkeypatch):
    # Advance only a local fake clock; this tests passive waiting beyond the
    # normal two observes without sleeping or sending another launcher input.
    clock = [1000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    monkeypatch.setattr("time.sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    image = OSWorldBackend(EditingDesktop(), pause=0).observe().image
    launcher = control(ref="launcher", label="Editor launcher", tag="button", role="button",
                       input_type="", app="Shell", window="")
    before = replace(snapshot(launcher, replace(launcher, ref="help", label="Help")), image=image)
    frame = control(ref="frame", label="Editor", tag="container", role="frame", input_type="",
                    app="Fixture", window="Editor", box=None, active=True)
    menu = control(ref="settings", label="Settings", tag="button", role="button", input_type="",
                   app="Fixture", window="Editor", active=True)
    appeared = replace(snapshot(launcher, frame, menu), image=image)
    field = replace(control(), window="Editor", active=True)
    editing = replace(snapshot(launcher, frame, field), image=image)

    class Environment:
        phase, reads = 0, 0

        def __init__(self):
            self.actions = []

        def _get_obs(self):
            if self.phase == 1:
                self.reads += 1
                if self.reads <= 2:
                    return before
                return appeared
            if self.phase == 2:
                return editing
            if self.phase == 3:
                return replace(editing, controls=(launcher, frame, replace(field, checked=True)))
            return before

        def step(self, action, pause=0):
            self.actions.append(action)
            self.phase += 1
            # Initial submission returns the shell; the new app only becomes
            # visible during the THIRD receipt-bound passive observation.
            observation = before if self.phase == 1 else self._get_obs()
            return observation, None, False, {}

    env, requests, trace = Environment(), [], JsonlTrace()
    backend = OSWorldBackend(env, pause=0)
    monkeypatch.setattr(backend, "_snapshot", lambda observation: observation)
    raw = intent()
    raw["scope"] = {"app": "Shell"}

    class ChooseLauncher(DirectPlanPolicy):
        def choose(self, observation, candidates):
            return super().choose(observation, sorted(candidates, key=lambda c: "Help" in c.description))

    result = AgentRunner(backend, planner_for(raw, requests), policy=ChooseLauncher(), trace=trace,
                         config=AgentConfig(allow_actions=True)).run("Enable the editor option")
    backend.close()
    assert result.status == "unverified" and result.plan_completion_claims == 1
    assert result.planner_calls == 1 and result.steps == len(env.actions) == 3
    assert result.recovery_observations == 3 and result.plan_reuses == 2
    assert result.candidate_counts == (2, 1, 1)
    waits = [e["payload"] for e in trace.events if e["kind"] == "agent_application_wait"]
    assert len(waits) == 1 and waits[0]["reason"] == "effect_confirmed"
    assert waits[0]["attempts"] == 3
