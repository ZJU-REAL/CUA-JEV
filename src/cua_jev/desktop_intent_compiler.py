"""Compile explicit stage declarations into existing validated desktop primitives."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent_types import AgentCondition, AgentPlan, AgentSnapshot, AgentStep
from .desktop_semantics import (
    conditions_hold,
    executable_controls,
    matching_controls,
    normalize_role,
    observable_condition_kinds,
)

STAGE_INTENTS = frozenset({"navigate", "navigate_edit", "edit_fields", "act_current", "complete"})
PROPERTY_STAGE_INTENTS = (STAGE_INTENTS - {"navigate_edit"}) | {"pursue_property"}


def _shape(raw: Any, required: set[str], optional: set[str], path: str) -> None:
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: stage fields must be a JSON object")
    missing = required - set(raw)
    if missing:
        raise ValueError(f"{path}: missing required stage fields: {', '.join(sorted(missing))}")
    if set(raw) - required - optional:
        raise ValueError(f"{path}: unsupported stage fields; allowed fields: "
                         f"{', '.join(sorted(required | optional))}")


def _current_dialog_effect(commit: dict, snapshot: AgentSnapshot, *, allow_native_modal_owners: bool = False):
    """Derive only an observed dialog's absence for an already explicit click.

    Field guards and their numeric readiness still pass the ordinary commit-only
    compiler below; no command, button, target or field value is supplied here.
    """
    from .agent_planner import _target

    if (not snapshot.a11y_complete or commit.get("operation") != "click"
            or "target" not in commit or "ref" in commit or commit.get("value", "")
            or commit.get("button", "left") != "left" or commit.get("click_count", 1) != 1):
        raise ValueError("derived commit effect requires a complete current explicit single left click")
    target = _target(commit["target"], path="edit_fields.commit.target")
    if not (target.app and target.window):
        raise ValueError("derived commit effect requires an explicit app and window on the commit target")
    buttons = executable_controls(snapshot, "click", target)
    if (len(buttons) != 1 or matching_controls(snapshot, target) != buttons
            or normalize_role(buttons[0].role) != "button"):
        raise ValueError("derived commit effect requires one uniquely identified current button")
    button = buttons[0]
    structural = allow_native_modal_owners is True and bool(button.modal_owner_ref)
    dialogs = (tuple(c for c in snapshot.controls if c.ref == button.modal_owner_ref
                     and c.modal is True and c.tag == "container"
                     and (c.app, c.window) == (button.app, button.window)) if structural else
               tuple(c for c in snapshot.controls if normalize_role(c.role) == "dialog"
                     and (c.app, c.window) == (button.app, button.window)))
    focused_owner = bool(len(dialogs) == 1 and structural and any(
        c.focused and c.modal_owner_ref == dialogs[0].ref for c in snapshot.controls
    ))
    if (len(dialogs) != 1 or dialogs[0].disabled or not (dialogs[0].active or focused_owner)
            or not dialogs[0].app or not dialogs[0].window):
        raise ValueError("derived commit effect requires one current active owning dialog")
    dialog = dialogs[0]
    if not structural and (not dialog.label or f"{dialog.role}: {dialog.label}" not in button.ancestors):
        raise ValueError("derived commit effect requires observed button ancestry in that dialog")
    if structural:
        observed_target = _current_ref_target({"ref": dialog.ref}, snapshot)
    else:
        observed_target = {"role": dialog.role, "app": dialog.app, "window": dialog.window}
        if dialog.label:
            observed_target["name"] = dialog.label
    predicate_target = _target(observed_target, path="derived_dialog_effect.target")
    if matching_controls(snapshot, predicate_target) != (dialog,):
        raise ValueError("derived commit effect cannot uniquely identify the observed dialog")
    return dialog, predicate_target, {"kind": "absent", "target": observed_target}


def _current_ref_target(raw: dict, snapshot: AgentSnapshot) -> dict:
    """Resolve a current guard reference once, without using its desired state."""
    from dataclasses import asdict

    from .agent_planner import _target
    from .agent_types import AgentTarget

    if (not snapshot.a11y_complete or not isinstance(raw, dict)
            or set(raw) - {"ref", "name", "role", "app", "window", "ancestor", "match"}
            or not isinstance(raw.get("ref"), str) or not raw["ref"]):
        raise ValueError("current commit references require a complete observation and current ref")
    matches = tuple(c for c in snapshot.controls if c.ref == raw["ref"])
    if len(matches) != 1:
        raise ValueError("current commit reference must identify exactly one current control")
    control = matches[0]
    qualifiers = {k: v for k, v in raw.items() if k != "ref"}
    if not qualifiers.get("name") and not qualifiers.get("role"):
        qualifiers["role"] = control.role
    if control not in matching_controls(snapshot, _target(qualifiers)):
        raise ValueError("current commit reference contradicts its observed qualifiers")
    if not control.app or not control.window or control.input_type == "password":
        raise ValueError("current commit reference lacks usable public ownership evidence")
    for ancestor in ("", *reversed(control.ancestors)):
        fields = (control.label, control.role, control.app, control.window, ancestor)
        if any(len(s) > 240 for s in fields) or len(control.role) > 80:
            continue
        target = AgentTarget(name=control.label, role=control.role, app=control.app,
                             window=control.window, ancestor=ancestor)
        if (target.name or target.role) and matching_controls(snapshot, target) == (control,):
            return asdict(target)
    raise ValueError("current commit reference has no unique observed semantic query")


def _resolve_current_commit_refs(commit: dict, snapshot: AgentSnapshot) -> dict:
    """Only current button and persistent guards; future effects stay untouched."""
    result = dict(commit)
    if isinstance(result.get("target"), dict) and "ref" in result["target"]:
        result["target"] = _current_ref_target(result["target"], snapshot)
    guards = result.get("preconditions")
    if isinstance(guards, list):
        result["preconditions"] = [
            {**guard, "target": _current_ref_target(guard["target"], snapshot)}
            if (isinstance(guard, dict) and guard.get("kind") in {
                "value_equals", "checked", "unchecked", "selected",
            } and isinstance(guard.get("target"), dict) and "ref" in guard["target"])
            else guard for guard in guards
        ]
    return result


def compile_stage_intent(raw: Any, snapshot: AgentSnapshot, max_actions: int = 6, *,
                         allow_property_goal_conditions: bool = False,
                         allow_advisory_completion: bool = False,
                         inherit_current_action_label: bool = False,
                         allow_guarded_commit_only: bool = False,
                         allow_property_preparation: bool = False,
                         allow_property_value_binding: bool = False,
                         allow_dialog_commit_effect: bool = False,
                         allow_observed_property_goal: bool = False,
                         allow_current_commit_refs: bool = False,
                         allow_property_numeric_finalization: bool = False,
                         require_numeric_finalize_choice: bool = False,
                         require_explicit_stage_policy: bool = False) -> AgentPlan:
    if type(max_actions) is not int or not 2 <= max_actions <= 12:
        raise ValueError("stage intent requires desktop multi-action planning")
    if type(allow_advisory_completion) is not bool:
        raise ValueError("advisory completion requires explicit compiler opt-in")
    if type(inherit_current_action_label) is not bool:
        raise ValueError("current action label inheritance requires explicit compiler opt-in")
    if type(allow_guarded_commit_only) is not bool:
        raise ValueError("guarded commit-only stages require explicit compiler opt-in")
    if type(allow_property_preparation) is not bool:
        raise ValueError("property preparation requires explicit compiler opt-in")
    if type(allow_property_value_binding) is not bool:
        raise ValueError("property value binding requires explicit compiler opt-in")
    if type(allow_dialog_commit_effect) is not bool:
        raise ValueError("dialog commit effect requires explicit compiler opt-in")
    if type(allow_observed_property_goal) is not bool or type(allow_current_commit_refs) is not bool:
        raise ValueError("observed property goal and current references require explicit compiler opt-in")
    if type(allow_property_numeric_finalization) is not bool:
        raise ValueError("numeric property finalization requires explicit compiler opt-in")
    if (type(require_numeric_finalize_choice) is not bool
            or (require_numeric_finalize_choice and not allow_property_numeric_finalization)):
        raise ValueError("explicit numeric strategy choice requires numeric finalization opt-in")
    if (type(require_explicit_stage_policy) is not bool
            or (require_explicit_stage_policy and not require_numeric_finalize_choice)):
        raise ValueError("explicit stage completion requires numeric strategy opt-in")
    if (
        not isinstance(raw, dict)
        or not isinstance(raw.get("intent"), str)
        or raw["intent"] not in STAGE_INTENTS | PROPERTY_STAGE_INTENTS
    ):
        raise ValueError("unsupported desktop stage intent")
    from .agent_planner import MAX_CONDITIONS, _target, _validate_desktop_execution_plan, parse_plan

    intent = raw["intent"]
    if require_explicit_stage_policy and intent in {"pursue_property", "navigate", "edit_fields"}:
        if (not isinstance(raw.get("completion"), str)
                or raw["completion"] not in {"stage", "goal"}):
            raise ValueError("semantic stage requires explicit completion stage or goal")
    if intent == "pursue_property":
        from .desktop_property import compile_property_stage

        return compile_property_stage(raw, snapshot, allow_goal_conditions=allow_property_goal_conditions,
                                      allow_preparation=allow_property_preparation, max_actions=max_actions,
                                      allow_value_binding=allow_property_value_binding,
                                      allow_observed_goal=allow_observed_property_goal,
                                      allow_property_numeric_finalization=allow_property_numeric_finalization,
                                      require_numeric_finalize_choice=require_numeric_finalize_choice,
                                      require_explicit_stage_policy=require_explicit_stage_policy)
    common = {"intent", "label"}

    def legacy(steps, *, done=False, completion=()):
        plan = parse_plan(
            {
                "summary": raw["label"],
                "steps": steps,
                "done": done,
                "completion_conditions": list(completion),
            },
            snapshot,
            max_actions=max_actions,
            profile="desktop",
        )
        return replace(plan, defer_dialog_field_completion=(
            allow_property_preparation and bool(steps) and bool(completion)
        ))

    if intent == "complete":
        if allow_advisory_completion and "conditions" not in raw:
            _shape(raw, common, set(), "complete")
            # Same unverified stopping declaration as the single-action VLM's
            # empty done response. No condition is fabricated or discarded;
            # the caller's independent task evaluator still determines success.
            return legacy([], done=True)
        _shape(raw, common | {"conditions"}, set(), "complete")
        if not isinstance(raw["conditions"], list) or not raw["conditions"]:
            raise ValueError("complete requires explicit whole-goal evidence conditions")
        return legacy([], done=True, completion=raw["conditions"])

    if intent == "navigate":
        _shape(raw, common | {"scope", "arrival"}, {"completion"}, "navigate")
        if not isinstance(raw["arrival"], list) or not raw["arrival"]:
            raise ValueError("navigate requires explicit observable arrival conditions")
        completion = raw.get("completion", "stage")
        if not isinstance(completion, str) or completion not in {"stage", "goal"}:
            raise ValueError("navigate.completion must be stage or goal")
        plan = legacy(
            [
                {
                    "id": "reach",
                    "operation": "click",
                    "navigation": True,
                    "label": raw["label"],
                    "target": raw["scope"],
                    "postconditions": raw["arrival"],
                    "recovery": "observe",
                }
            ],
            completion=raw["arrival"] if completion == "goal" else (),
        )
        return replace(plan, steps=(replace(plan.steps[0], native_navigation_gestures=True),))

    if intent == "navigate_edit":
        _shape(raw, common | {"scope", "edit"}, {"completion"}, "navigate_edit")
        from .desktop_stage_compiler import compile_reach_edit

        plan = compile_reach_edit({**raw, "intent": "reach_edit"}, snapshot, max_actions=max_actions)
        mutation = plan.steps[1]
        present = matching_controls(snapshot, mutation.target)
        if present:
            if executable_controls(snapshot, mutation.operation, mutation.target):
                raise ValueError(
                    "navigate_edit target is current; declare edit_fields with the current control"
                )
            raise ValueError(
                "navigate_edit target currently names no operation-capable control; a static label, "
                "filename or current value is not an editable field name. Use current operation/role/context "
                "evidence, navigate toward an explicit arrival, or one grounded act_current"
            )
        return plan

    if intent == "act_current":
        _shape(raw, common | {"action"}, set(), "act_current")
        _shape(
            raw["action"],
            {"operation"},
            {
                "target",
                "ref",
                "value",
                "label",
                "box",
                "box_space",
                "button",
                "click_count",
                "preconditions",
                "postconditions",
                "recovery",
            },
            "act_current.action",
        )
        action = raw["action"]
        if (inherit_current_action_label and action["operation"] in {"visual_click", "visual_fill"}
                and "label" not in action):
            # The stage already supplies the description. Only metadata is
            # inherited: coordinates, values, targets and operations stay explicit.
            action = {**action, "label": raw["label"]}
        plan = legacy([{"id": "act", **action}])
        _validate_desktop_execution_plan(plan, snapshot, 1)
        return plan

    _shape(raw, common | {"edits"}, {"commit", "completion"}, "edit_fields")
    if not isinstance(raw["edits"], list):
        raise ValueError("edit_fields.edits must be a JSON array of current field edits")
    commit_only = not raw["edits"] and allow_guarded_commit_only and "commit" in raw
    if not raw["edits"] and not commit_only:
        raise ValueError(
            "edit_fields.edits must include at least one current field. An already-correct field "
            "must still be listed to guard an explicit commit; its satisfied edit is skipped without input"
        )
    if len(raw["edits"]) > max_actions:
        raise ValueError(
            f"edit_fields.edits permits at most {max_actions} current fields within the step budget"
        )
    if not snapshot.a11y_complete:
        raise ValueError(
            "edit_fields requires a complete observation to establish current field uniqueness; "
            "inspect one current grounded recovery action instead of repeating this form stage"
        )
    completion = raw.get("completion", "stage")
    if not isinstance(completion, str) or completion not in {"stage", "goal"}:
        raise ValueError("edit_fields.completion must be stage or goal")
    steps, effects, targets, seen = [], [], [], set()
    for index, item in enumerate(raw["edits"]):
        _shape(item, {"operation", "target"}, {"value", "preconditions"}, f"edits[{index}]")
        operation = item["operation"]
        if not isinstance(operation, str) or operation not in {"fill", "check", "uncheck"}:
            raise ValueError("edit_fields operations must be fill, check or uncheck")
        if (operation == "fill") != ("value" in item):
            raise ValueError("edit_fields value is required only for fill")
        target = _target(item["target"], path=f"edits[{index}].target")
        matches = executable_controls(snapshot, operation, target)
        if not (target.app or target.window) or len(matches) != 1:
            raise ValueError(
                f"edits[{index}].target must identify one current operation-capable control with app/window; "
                "use the editable control's role/context for an anonymous field, never a nearby static label"
            )
        if matches[0].ref in seen:
            raise ValueError("edit_fields must declare each current field only once")
        seen.add(matches[0].ref)
        targets.append(target)
        effect = {
            "kind": {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}[operation],
            "target": item["target"],
        }
        if operation == "fill":
            effect["value"] = item["value"]
        effects.append(effect)
        steps.append(
            {
                "id": f"edit{index + 1}",
                "operation": operation,
                "target": item["target"],
                **({"value": item["value"]} if operation == "fill" else {}),
                "preconditions": item.get("preconditions", []),
                "postconditions": [effect],
                "recovery": "observe",
            }
        )
    edit_count = len(steps)
    commit = raw.get("commit")
    derived_dialog = derived_target = None
    if "commit" in raw:
        if allow_current_commit_refs and commit_only and isinstance(commit, dict):
            commit = _resolve_current_commit_refs(commit, snapshot)
        if (allow_dialog_commit_effect and commit_only and isinstance(commit, dict)
                and "postconditions" not in commit):
            derived_dialog, derived_target, effect = _current_dialog_effect(commit, snapshot)
            commit = {**commit, "postconditions": [effect]}
        _shape(
            commit,
            {"operation", "postconditions"},
            {
                "target",
                "ref",
                "value",
                "label",
                "preconditions",
                "button",
                "click_count",
                "recovery",
            },
            "edit_fields.commit",
        )
        if not isinstance(commit["postconditions"], list) or not commit["postconditions"]:
            raise ValueError("edit_fields.commit requires committed-result conditions")
        if not isinstance(commit["operation"], str) or commit["operation"] not in {
            "click",
            "hotkey",
            "key_press",
            "press",
        }:
            raise ValueError("edit_fields.commit must be an explicit button or guarded keyboard action")
        preconditions = commit.get("preconditions", [])
        if not isinstance(preconditions, list):
            raise ValueError("edit_fields.commit.preconditions must be an array")
        steps.append(
            {
                **commit,
                "id": "commit",
                "depends_on": [step["id"] for step in steps],
                "preconditions": effects + preconditions,
            }
        )
    goal_conditions = (
        (commit["postconditions"] if commit is not None else effects) if completion == "goal" else ()
    )
    plan = legacy(steps, completion=goal_conditions)
    if commit_only:
        step = plan.steps[0]
        persistent = {"value_equals": "fill", "checked": "check", "unchecked": "uncheck",
                      "selected": "click"}
        guards = [c for c in step.preconditions if c.kind in persistent]
        if not guards:
            raise ValueError("commit-only requires explicit persistent field value/state guards")
        readiness = []
        rebound_guards = {}
        for condition in guards:
            target = condition.target
            if target is None or not (target.app or target.window):
                raise ValueError("commit-only field guards require observed app/window context")
            matches = executable_controls(snapshot, persistent[condition.kind], target)
            if (not matches and target.match == "exact" and target.name and target.role):
                # Native field captions commonly append one colon. Rebind only
                # an otherwise exact, fully contextual unique observed field.
                def caption(name):
                    return " ".join(name.strip().removesuffix(":").strip().casefold().split())

                alternatives = {}
                for control in snapshot.controls:
                    if (len(control.label) > 240 or caption(control.label) != caption(target.name)
                            or control.label == target.name):
                        continue
                    observed_target = replace(target, name=control.label)
                    found = executable_controls(snapshot, persistent[condition.kind], observed_target)
                    for actual in found:
                        alternatives[actual.ref] = (replace(target, name=actual.label), found)
                if len(alternatives) == 1:
                    target, matches = next(iter(alternatives.values()))
                    rebound_guards[condition] = replace(condition, target=target)
                elif len(alternatives) > 1:
                    raise ValueError("commit-only caption normalization has ambiguous current fields")
            if len(matches) != 1 or condition.kind not in observable_condition_kinds(matches[0]):
                raise ValueError("commit-only field guards must uniquely bind current capable fields")
            if derived_dialog is not None and (
                not (target.app and target.window)
                or (matches[0].app, matches[0].window) != (derived_dialog.app, derived_dialog.window)
                or f"{derived_dialog.role}: {derived_dialog.label}" not in matches[0].ancestors
            ):
                raise ValueError("derived commit effect requires explicit field guards in the same dialog")
            condition = rebound_guards.get(condition, condition)
            if condition.kind == "value_equals":
                from .desktop_property import property_effect_holds

                check = AgentStep("guard", "fill", target=target, value=condition.value,
                                  postconditions=(condition,))
                if not property_effect_holds(check, snapshot, matches[0]):
                    raise ValueError("commit-only requires the committed value without a conflicting buffer")
            if condition.kind != "selected":
                ready = AgentCondition("ready_to_" + persistent[condition.kind], target, condition.value)
                if ready not in readiness:
                    readiness.append(ready)
        if derived_target is not None:
            readiness.append(AgentCondition("exists", derived_target))
        if len(step.preconditions) + len(readiness) > MAX_CONDITIONS:
            raise ValueError("commit-only guards and compiler readiness exceed the condition budget")
        step = replace(step, preconditions=(
            *(rebound_guards.get(condition, condition) for condition in step.preconditions), *readiness,
        ), derived_dialog_commit_effect=derived_dialog is not None)
        if not conditions_hold(step.preconditions, snapshot):
            raise ValueError("commit-only requires currently satisfied persistent field guards")
        if conditions_hold(step.postconditions, snapshot):
            raise ValueError("commit-only requires an unmet committed-result condition")
        plan = replace(plan, steps=(step,))
        _validate_desktop_execution_plan(plan, snapshot, 1)
        return plan
    compiled = []
    for index, step in enumerate(plan.steps):
        if index < edit_count:
            if len(step.preconditions) >= MAX_CONDITIONS:
                raise ValueError("edit_fields permits at most seven field guards plus internal readiness")
            ready = AgentCondition(f"ready_to_{step.operation}", targets[index], step.value)
            step = replace(
                step, preconditions=(*step.preconditions, ready), postconditions=(*step.postconditions, ready)
            )
        compiled.append(step)
    return replace(
        plan,
        steps=tuple(compiled),
        completion_conditions=(
            tuple(condition for step in compiled[:edit_count] for condition in step.postconditions)
            if completion == "goal" and commit is None
            else plan.completion_conditions
        ),
    )
