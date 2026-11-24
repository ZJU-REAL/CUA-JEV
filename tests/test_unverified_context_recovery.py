"""Unknown effects may justify fresh inspection, never input or success claims."""

from __future__ import annotations

import io
import json
from dataclasses import replace

import pytest
from PIL import Image

from cua_jev.agent_planner import _validate_desktop_execution_plan, parse_plan
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot, AgentStep, AgentTarget
from cua_jev.osworld_backend import OSWorldBackend, screenshot_image
from cua_jev.trace import JsonlTrace


@pytest.fixture
def states():
    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buffer, "PNG")
    button = AgentControl("button", "Settings", "button", role="button", app="Editor",
                          window="Document", active=True, box=(100, 100, 200, 200))
    before = AgentSnapshot("osworld://desktop", "Document", "Document", (button,),
                           image=screenshot_image(buffer.getvalue()))
    # The old window's button is still reported, but the actual new editable
    # field is active in the same application's dialog. Its value is private.
    field = AgentControl("field", "", "input", role="textbox", app="Editor", window="Settings",
                         ancestors=("dialog: Settings",), active=True, focused=True, editable=True,
                         box=(300, 300, 600, 400), value="PRIVATE_CURRENT_TEXT")
    after = replace(before, controls=(replace(button, active=False), field))
    return before, after


class Desktop:
    def __init__(self, before, after):
        self.current, self.after = before, after
        self.actions, self.reads = [], 0
        self.fail = False

    def _get_obs(self):
        self.reads += 1
        return self.current

    def step(self, action, pause):
        self.actions.append(action)
        if self.fail:
            raise RuntimeError("input failed")
        self.current = self.after
        return self.current, 0, False, {}


class Backend(OSWorldBackend):
    def _snapshot(self, observation):
        return observation


def step(*, postconditions=None, operation="click", navigation=False):
    missing = AgentCondition("exists", AgentTarget(name="Expected dialog", role="window", app="Editor"))
    target = AgentTarget(app="Editor") if navigation else AgentTarget(name="Settings", app="Editor")
    return AgentStep("open", operation, label="Reach settings", navigation=navigation,
                     target=target,
                     postconditions=postconditions or (missing,), recovery="observe")


def execute(before, after, *, requested=None, fail=False, preflight=None):
    env = Desktop(before, after)
    env.fail = fail
    backend = Backend(env, pause=0)
    candidate = backend.compile(requested or step(), before, before)[0]
    if preflight is not None:
        env.current = preflight
    receipt = backend.execute(candidate, "observation", "decision")
    return backend, env, candidate, receipt


def test_actual_new_anonymous_field_yields_private_ticket_but_stays_unverified(states):
    before, after = states
    backend, env, candidate, receipt = execute(before, after)
    verification = backend.verify(candidate, receipt)
    assert receipt.success and receipt.output["environment_steps"] == 1
    assert not verification.passed and not verification.details["postconditions_met"]
    ticket = backend.take_unverified_recovery_context(candidate, receipt)
    assert set(ticket) == {"context_fingerprint", "new_control_count", "reason"}
    assert len(ticket["context_fingerprint"]) == 64 and ticket["new_control_count"] == 1
    assert ticket["reason"] == "new_actionable_context_after_unconfirmed_input"
    assert "PRIVATE" not in json.dumps(ticket)
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    fresh = backend.observe()
    reads = env.reads
    assert backend.confirm_unverified_recovery_context(ticket, fresh)
    assert env.reads == reads and len(env.actions) == 1
    assert not backend.confirm_unverified_recovery_context(ticket, fresh)
    assert not backend.verify(candidate, receipt).passed
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


@pytest.mark.parametrize("change", ["label", "window", "ancestors", "ref", "geometry", "focus",
                                   "value", "selected", "active", "pixels"])
def test_nonstructural_changes_cannot_buy_a_recovery(states, change):
    before, _ = states
    button = before.controls[0]
    changes = {
        "label": {"label": "New label"}, "window": {"window": "New window title"},
        "ancestors": {"ancestors": ("panel: New title",)}, "ref": {"ref": "new-ref"},
        "geometry": {"box": (400, 200, 500, 300)}, "focus": {"focused": True},
        "value": {"value": "PRIVATE_VALUE"}, "selected": {"selectable": True, "selected": True},
        "active": {"active": True}, "pixels": {},
    }
    if change == "active":
        before = replace(before, controls=(replace(button, active=False),))
    after = replace(before, controls=(replace(button, **changes[change]),))
    if change == "pixels":
        buffer = io.BytesIO()
        Image.new("RGB", (800, 600), "gray").save(buffer, "PNG")
        after = replace(after, image=screenshot_image(buffer.getvalue()))
    backend, _, candidate, receipt = execute(before, after)
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


@pytest.mark.parametrize("change", ["before_partial", "after_partial", "no_app", "different_app",
                                   "background", "readonly", "no_geometry", "disabled"])
def test_new_context_requires_complete_same_app_active_executable_evidence(states, change):
    before, after = states
    if change == "before_partial":
        before = replace(before, a11y_complete=False)
    elif change == "after_partial":
        after = replace(after, a11y_complete=False)
    elif change == "no_app":
        before = replace(before, controls=(replace(before.controls[0], app=""),))
    else:
        field = after.controls[1]
        changes = {"different_app": {"app": "Other Editor"}, "background": {"active": False},
                   "readonly": {"tag": "text", "role": "label", "editable": False},
                   "no_geometry": {"box": None}, "disabled": {"disabled": True}}
        after = replace(after, controls=(after.controls[0], replace(field, **changes[change])))
    requested = step()
    if change == "no_app":
        requested = replace(requested, target=AgentTarget(name="Settings"))
    backend, _, candidate, receipt = execute(before, after, requested=requested)
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


def test_observed_false_postcondition_prevents_ticket_even_alongside_unknown(states):
    before, after = states
    checkbox = AgentControl("checkbox", "Enabled", "input", role="checkbox", input_type="checkbox",
                            app="Editor", window="Settings", active=True, box=(600, 100, 700, 200))
    after = replace(after, controls=(*after.controls, checkbox))
    requested = step(postconditions=(*step().postconditions, AgentCondition(
        "checked", AgentTarget(name="Enabled", app="Editor"),
    )))
    backend, _, candidate, receipt = execute(before, after, requested=requested)
    assert [f["outcome"] for f in backend.verify(candidate, receipt).details["postcondition_evidence"]] == [
        "unknown", "false",
    ]
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


def test_true_conditions_do_not_hide_remaining_unknown_or_turn_it_into_success(states):
    before, after = states
    field = AgentCondition("exists", AgentTarget(role="textbox", app="Editor"))
    requested = step(postconditions=(*step().postconditions, field))
    backend, _, candidate, receipt = execute(before, after, requested=requested)
    assert not backend.verify(candidate, receipt).passed
    assert backend.take_unverified_recovery_context(candidate, receipt)["new_control_count"] == 1


@pytest.mark.parametrize("kind", ["failed", "stale", "already_verified", "navigation_progress"])
def test_failed_unsent_or_verified_receipts_do_not_authorize_unknown_recovery(states, kind):
    before, after = states
    requested = step()
    if kind == "already_verified":
        condition = AgentCondition("exists", AgentTarget(role="textbox", app="Editor"))
        requested = step(postconditions=(condition,))
    elif kind == "navigation_progress":
        requested = step(navigation=True)
    backend, env, candidate, receipt = execute(
        before, after, requested=requested, fail=kind == "failed",
        preflight=after if kind == "stale" else None,
    )
    assert backend.take_unverified_recovery_context(candidate, receipt) is None
    if kind == "stale":
        assert env.actions == []
    if kind in {"already_verified", "navigation_progress"}:
        assert backend.verify(candidate, receipt).passed


@pytest.mark.parametrize("change", ["receipt_copy", "receipt_output", "candidate", "backend"])
def test_receipt_and_candidate_binding_cannot_be_forged(states, change):
    backend, env, candidate, receipt = execute(*states)
    if change == "receipt_copy":
        receipt = replace(receipt)
    elif change == "receipt_output":
        receipt.output["environment_steps"] = 2
    elif change == "candidate":
        candidate = replace(candidate, description="altered")
    else:
        backend = Backend(env, pause=0)
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


@pytest.mark.parametrize("action", ["observe", "new_execute", "input", "close"])
def test_new_activity_invalidates_unconsumed_evidence(states, action):
    backend, _, candidate, receipt = execute(*states)
    if action == "observe":
        backend.observe()
    elif action == "new_execute":
        assert not backend.execute(candidate, "replay", "replay").success
    elif action == "input":
        backend._step("CLICK", {"x": 1, "y": 1, "button": "left", "num_clicks": 1})
    else:
        backend.close()
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


def test_pending_uses_original_receipt_before_and_latest_complete_after(states):
    before, after = states
    backend, env, candidate, receipt = execute(before, before)
    env.current = after
    verification = backend.verify_pending(candidate, receipt)
    assert not verification.passed and verification.details["observation_recovery"] == 1
    ticket = backend.take_unverified_recovery_context(candidate, receipt)
    assert ticket["new_control_count"] == 1
    assert backend.take_verified_post_snapshot(candidate, receipt) is None
    assert backend.confirm_unverified_recovery_context(ticket, backend.observe())
    # Taking the original ticket consumes its source. Further pending checks
    # cannot manufacture a second recovery coupon for the same input.
    backend.verify_pending(candidate, receipt)
    assert backend.take_unverified_recovery_context(candidate, receipt) is None
    assert len(env.actions) == 1


@pytest.mark.parametrize("after_kind", ["gone", "incomplete", "inactive"])
def test_later_pending_observation_removes_old_context_evidence(states, after_kind):
    before, after = states
    backend, env, candidate, receipt = execute(before, after)
    env.current = {
        "gone": before,
        "incomplete": replace(after, a11y_complete=False),
        "inactive": replace(after, controls=tuple(replace(c, active=False) for c in after.controls)),
    }[after_kind]
    backend.verify_pending(candidate, receipt)
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


@pytest.mark.parametrize("change", ["copied_ticket", "mutated_ticket", "old_snapshot", "copied_snapshot",
                                   "second_observe", "new_input", "pending", "execute", "closed"])
def test_confirmation_requires_original_ticket_and_exactly_next_fresh_observation(states, change):
    backend, _, candidate, receipt = execute(*states)
    ticket = backend.take_unverified_recovery_context(candidate, receipt)
    fresh = backend.observe()
    if change == "copied_ticket":
        ticket = dict(ticket)
    elif change == "mutated_ticket":
        ticket["new_control_count"] += 1
    elif change == "old_snapshot":
        fresh = states[0]
    elif change == "copied_snapshot":
        fresh = replace(fresh)
    elif change == "second_observe":
        fresh = backend.observe()
    elif change == "new_input":
        backend._step("CLICK", {"x": 1, "y": 1, "button": "left", "num_clicks": 1})
    elif change == "pending":
        backend.verify_pending(candidate, receipt)
    elif change == "execute":
        backend.execute(candidate, "replayed", "replayed")
    else:
        backend.close()
    assert not backend.confirm_unverified_recovery_context(ticket, fresh)


@pytest.mark.parametrize("change", ["gone", "partial", "inactive", "unrelated", "value"])
def test_confirmation_checks_new_structure_without_requiring_entire_desktop_static(states, change):
    before, after = states
    backend, env, candidate, receipt = execute(before, after)
    ticket = backend.take_unverified_recovery_context(candidate, receipt)
    if change == "gone":
        env.current = before
    elif change == "partial":
        env.current = replace(after, a11y_complete=False)
    elif change == "inactive":
        env.current = replace(after, controls=tuple(replace(c, active=False) for c in after.controls))
    elif change == "unrelated":
        other = replace(before.controls[0], ref="other", app="Other", label="Unrelated")
        env.current = replace(after, controls=(*after.controls, other))
    else:
        changed = replace(after.controls[1], value="CHANGED")
        env.current = replace(after, controls=(after.controls[0], changed))
    assert backend.confirm_unverified_recovery_context(ticket, backend.observe()) is (
        change in {"unrelated", "value"}
    )


def test_no_input_transition_never_earns_recovery(states):
    before, after = states
    class NoInputBackend(Backend):
        def _perform(self, *args, **kwargs):
            self.env.current = after
    env = Desktop(before, after)
    backend = NoInputBackend(env, pause=0)
    candidate = backend.compile(step(), before, before)[0]
    receipt = backend.execute(candidate, "observation", "decision")
    assert receipt.success and not backend.verify(candidate, receipt).passed
    assert receipt.output["environment_steps"] == 0 and env.actions == []
    assert backend.take_unverified_recovery_context(candidate, receipt) is None


@pytest.mark.parametrize("change", ["before_read", "failed_read", "mutated_receipt"])
def test_confirmation_cannot_skip_fresh_read_or_ignore_receipt_changes(states, change):
    backend, env, candidate, receipt = execute(*states)
    ticket = backend.take_unverified_recovery_context(candidate, receipt)
    fresh = states[1]
    if change == "failed_read":
        def failed():
            raise RuntimeError("read failed")
        env._get_obs = failed
        with pytest.raises(RuntimeError):
            backend.observe()
    elif change == "mutated_receipt":
        fresh = backend.observe()
        receipt.output["environment_steps"] = 0
    assert not backend.confirm_unverified_recovery_context(ticket, fresh)


def test_confirmation_requires_added_multiplicity_not_just_an_old_similar_control(states):
    before, after = states
    field = after.controls[1]
    before = replace(before, controls=(*before.controls, replace(field, ref="existing")))
    after = replace(after, controls=(*after.controls, replace(field, ref="second")))
    backend, env, candidate, receipt = execute(before, after)
    ticket = backend.take_unverified_recovery_context(candidate, receipt)
    assert ticket["new_control_count"] == 1
    env.current = replace(after, controls=after.controls[:2])
    assert not backend.confirm_unverified_recovery_context(ticket, backend.observe())


@pytest.mark.parametrize("disappears", [False, True])
def test_real_runner_recovers_third_unknown_click_with_one_fresh_anonymous_field_edit(
    states, monkeypatch, disappears,
):
    monkeypatch.setattr("time.sleep", lambda _: None)
    before, after = states
    prior_edit = AgentControl("prior", "Prior edit", "input", role="textbox", app="Editor",
                              window="Document", value="committed", active=True)
    before = replace(before, controls=(*before.controls, prior_edit))
    after = replace(after, controls=(*after.controls, replace(prior_edit, active=False)))

    class SlowDialog(Desktop):
        def step(self, action, pause):
            self.actions.append(action)
            if len(self.actions) == 3:
                self.current = after
            if action["action_type"] == "TYPING":
                self.current = replace(self.current, controls=tuple(
                    replace(control, value=action["parameters"]["text"])
                    if control.ref == "field" else control for control in self.current.controls
                ))
            return self.current, 0, False, {}

    class Planning:
        def __init__(self):
            self.calls = []

        def plan(self, goal, snapshot, history, *, max_actions):
            self.calls.append({"max_actions": max_actions, "history": list(history)})
            if len(self.calls) <= 3:
                raw = {
                    "summary": "Commit the existing edit and open its settings", "done": False,
                    "steps": [{
                        "id": "open", "operation": "click", "target": {"name": "Settings", "app": "Editor"},
                        "preconditions": [{"kind": "value_equals", "target": {
                            "name": "Prior edit", "role": "textbox", "app": "Editor", "window": "Document",
                        }, "value": "committed"}],
                        "postconditions": [{"kind": "exists", "target": {
                            "name": "Expected dialog", "role": "window", "app": "Editor",
                        }}], "recovery": "observe",
                    }],
                }
            else:
                assert len(self.calls) == 4 and max_actions == 1
                target = {"role": "textbox", "app": "Editor", "window": "Settings"}
                effect = {"kind": "value_equals", "target": target, "value": "Requested edit"}
                raw = {"summary": "Edit the uniquely observed unnamed field", "done": False,
                       "steps": [{"id": "edit", "operation": "fill", "target": target,
                                  "value": "Requested edit",
                                  "preconditions": [{"kind": "exists", "target": target}],
                                  "postconditions": [effect], "recovery": "observe"}],
                       "completion_conditions": [effect]}
            plan = parse_plan(raw, snapshot, profile="desktop", max_actions=max_actions)
            _validate_desktop_execution_plan(plan, snapshot, max_actions)
            return plan

    env, planner, trace = SlowDialog(before, after), Planning(), JsonlTrace()
    backend = Backend(env, pause=0)

    def progress(event):
        if disappears and event["event"] == "agent_context_recovery":
            env.current = before

    result = AgentRunner(
        backend, planner, policy=DirectPlanPolicy(), trace=trace, progress=progress,
        config=AgentConfig(allow_actions=True, max_failures=3, max_steps=6, max_plans=5),
    ).run("Commit the edit and set the newly opened field to Requested edit")
    steps = [event["payload"] for event in trace.events if event["kind"] == "agent_step"]
    recovery = [event["payload"] for event in trace.events if event["kind"] == "agent_context_recovery"]
    assert [item["verified"] for item in steps[:3]] == [False, False, False]
    assert result.context_recoveries == len(recovery) == 1 and result.post_snapshot_reuses == 0
    assert recovery[0]["action_failures"] == 3 and not recovery[0]["verified"]
    assert not recovery[0]["task_success"] and result.recovery_observations == 6
    if disappears:
        assert result.status == "stuck" and result.reason == "new recovery context no longer observed"
        assert result.planner_calls == len(planner.calls) == result.steps == 3
        assert len(env.actions) == backend.environment_steps == 3
        assert all(action["action_type"] == "CLICK" for action in env.actions)
    else:
        assert result.planner_calls == len(planner.calls) == result.steps == 4
        assert [call["max_actions"] for call in planner.calls] == [6, 1, 1, 1]
        assert planner.calls[-1]["history"][-1]["action_failures"] == 3
        assert [item["verified"] for item in steps] == [False, False, False, True]
        assert [a["action_type"] for a in env.actions] == ["CLICK"] * 4 + ["HOTKEY", "TYPING"]
        assert env.actions[-2]["parameters"] == {"keys": ["ctrl", "a"]}
        assert env.actions[-1]["parameters"] == {"text": "Requested edit"}
        assert len(env.actions) == backend.environment_steps == 6
        assert result.status == "unverified" and result.plan_completion_claims == 1
        assert next(c for c in env.current.controls if c.ref == "field").value == "Requested edit"
        input_counts = [record[1].output["environment_steps"] for record in backend._verified.values()]
        assert input_counts == [1, 1, 1, 3]
