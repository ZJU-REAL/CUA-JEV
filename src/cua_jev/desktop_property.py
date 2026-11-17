"""Bind a declarative property stage to currently legal, verifiable GUI actions."""

from dataclasses import replace

from .agent_types import (
    AgentCondition,
    AgentControl,
    AgentPlan,
    AgentPropertyEdit,
    AgentSnapshot,
    AgentStep,
    AgentTarget,
)
from .desktop_semantics import (
    _bounded_decimal,
    conditions_hold,
    executable_controls,
    matching_controls,
    normalize_role,
    observable_condition_kinds,
    value_matches,
)


def valid_property_stage(step: AgentStep) -> bool:
    edit, scope = step.property_edit, step.target
    return bool(
        isinstance(edit, AgentPropertyEdit)
        and isinstance(edit.operation, str) and edit.operation in {"fill", "check", "uncheck", "set"}
        and isinstance(edit.value, str) and len(edit.value) <= 2000 and "\x00" not in edit.value
        and type(edit.bind_boolean_states) is bool
        and (not edit.bind_boolean_states or edit.operation == "set")
        and type(edit.prefer_verified_selection) is bool
        and (not edit.prefer_verified_selection or edit.operation == "set")
        and isinstance(edit.numeric_finalize_key, str)
        and (not edit.numeric_finalize_key or (
            edit.numeric_finalize_key in {"enter", "tab"} and edit.operation in {"fill", "set"}
            and _bounded_decimal(edit.value) is not None
        ))
        and (bool(edit.value) and edit.value.isascii()
             if edit.operation in {"fill", "set"} else not edit.value)
        and step.operation == "click" and step.navigation is True
        and step.native_navigation_gestures is True
        and isinstance(step.label, str) and 0 < len(step.label.strip()) <= 240
        and not step.ref and step.box is None and not step.value
        and not step.preconditions and not step.postconditions
        and isinstance(step.property_prepare_dependency, str)
        and ((not step.property_prepare_dependency and not step.depends_on)
             or (step.property_prepare_dependency == "prepare" and step.id == "property"
                 and step.depends_on == ("prepare",)))
        and step.recovery == "observe" and step.button == "left" and step.click_count == 1
        and isinstance(scope, AgentTarget) and not scope.name and not scope.role and scope.match == "exact"
        and all(isinstance(v, str) and len(v) <= 240 for v in (scope.app, scope.window, scope.ancestor))
        and (scope.app.strip() or scope.window.strip())
    )


def property_field_steps(
    stage: AgentStep, snapshot: AgentSnapshot, scope_controls: tuple,
) -> tuple[tuple, ...]:
    """Generate only unique actual field queries, before testing their requested values.

    A semantic match to the user property remains the decision policy's job.
    These checks establish input legality and selected-field effects only.
    """
    if not valid_property_stage(stage) or not snapshot.a11y_complete:
        return ()
    edit = stage.property_edit
    output = []
    seen = set()
    for control in scope_controls:
        if control.ref in seen or not (control.app or control.window):
            continue
        for ancestor in ("", *reversed(control.ancestors)):
            if len(ancestor) > 240:
                continue
            target = AgentTarget(
                name=control.label if len(control.label) <= 240 else "",
                role=control.role, app=control.app, window=control.window, ancestor=ancestor,
            )
            if not (target.name or target.role):
                continue
            # Matching by desired value/state would incorrectly resolve duplicate
            # fields. Prove uniqueness of executable controls first.
            operation = "fill" if edit.operation == "set" else edit.operation
            value = edit.value
            if (edit.operation == "set" and edit.bind_boolean_states
                    and edit.value.casefold() in {"on", "off"} and not control.editable):
                operation = "check" if edit.value.casefold() == "on" else "uncheck"
                value = ""
            if (edit.operation == "set" and control.selectable and not control.editable
                    and control in matching_controls(snapshot, AgentTarget(name=edit.value))):
                # The requested literal names a public option, never an inferred
                # field value. Resolve its identity before selected state or
                # executability, including disabled same-name competitors.
                if (matching_controls(snapshot, target) != (control,)
                        or executable_controls(snapshot, "click", target) != (control,)):
                    continue
                ready = AgentCondition("ready_to_select", target)
                selected = AgentCondition("selected", target)
                output.append((control, AgentStep(
                    stage.id, "click", label=stage.label, target=target,
                    preconditions=(ready,), postconditions=(selected, ready), recovery="observe",
                )))
                seen.add(control.ref)
                break
            if executable_controls(snapshot, operation, target) != (control,):
                continue
            ready = AgentCondition(f"ready_to_{operation}", target, value)
            if not conditions_hold((ready,), snapshot):
                continue
            effect = AgentCondition(
                {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}[operation],
                target, value,
            )
            output.append((control, AgentStep(
                stage.id, operation, value=value, label=stage.label, target=target,
                preconditions=(ready,), postconditions=(effect, ready), recovery="observe",
            )))
            seen.add(control.ref)
            break
    return tuple(output)


def property_effect_holds(step: AgentStep, snapshot: AgentSnapshot, control: AgentControl | None) -> bool:
    if control is None or not conditions_hold(step.postconditions, snapshot):
        return False
    if step.operation == "fill" and normalize_role(control.role) == "spinbutton" \
            and control.edit_buffer is not None:
        # A committed value alone cannot confirm a field with a conflicting
        # observed pending buffer. Repair the literal, never commit implicitly.
        return value_matches(replace(control, value=control.edit_buffer), step.value)
    return True


def compile_property_stage(
    raw: dict, snapshot: AgentSnapshot, *, allow_goal_conditions: bool = False,
    allow_preparation: bool = False, max_actions: int = 6,
    allow_value_binding: bool = False,
    allow_observed_goal: bool = False,
    allow_property_numeric_finalization: bool = False,
    require_numeric_finalize_choice: bool = False,
    require_explicit_stage_policy: bool = False,
) -> AgentPlan:
    from .agent_planner import _conditions, _navigation_scope, _text

    if type(allow_goal_conditions) is not bool:
        raise ValueError("property goal conditions require explicit compiler opt-in")
    if type(allow_preparation) is not bool or type(max_actions) is not int or not 2 <= max_actions <= 12:
        raise ValueError("property preparation requires explicit multi-action compiler opt-in")
    if type(allow_value_binding) is not bool:
        raise ValueError("property value binding requires explicit compiler opt-in")
    if type(allow_observed_goal) is not bool:
        raise ValueError("observed property completion requires explicit compiler opt-in")
    if type(allow_property_numeric_finalization) is not bool:
        raise ValueError("numeric property finalization requires explicit compiler opt-in")
    if (type(require_numeric_finalize_choice) is not bool
            or (require_numeric_finalize_choice and not allow_property_numeric_finalization)):
        raise ValueError("explicit numeric strategy choice requires numeric finalization opt-in")
    if (type(require_explicit_stage_policy) is not bool
            or (require_explicit_stage_policy and not require_numeric_finalize_choice)):
        raise ValueError("explicit property completion and confirmation require strategy opt-in")
    required = {"intent", "label", "scope", "edit"}
    if require_numeric_finalize_choice:
        required.add("numeric_finalize")
    if require_explicit_stage_policy:
        required.update({"completion", "confirm"})
    allowed = (required | ({"goal_conditions"} if allow_goal_conditions else set())
               | ({"prepare"} if allow_preparation else set())
               | ({"completion", "confirm"} if allow_observed_goal else set())
               | ({"numeric_finalize"} if allow_property_numeric_finalization else set()))
    if (not isinstance(raw, dict) or not required <= set(raw) or set(raw) - allowed
            or not isinstance(raw["edit"], dict)):
        suffix = "; optional goal_conditions" if allow_goal_conditions else ""
        raise ValueError("pursue_property requires exactly intent, label, scope and edit" + suffix)
    goal_conditions = ()
    completion = raw.get("completion", "stage")
    if not isinstance(completion, str) or completion not in {"stage", "goal"}:
        raise ValueError("pursue_property.completion must be stage or goal")
    if "goal_conditions" in raw:
        goal_conditions = _conditions(raw["goal_conditions"], path="goal_conditions")
        if not goal_conditions:
            raise ValueError("goal_conditions must contain observable whole-goal evidence or be omitted")
    if completion == "goal" and len(goal_conditions) > 6:
        raise ValueError("property goal permits six explicit guards plus two bound field conditions")
    edit = raw["edit"]
    operations = {"fill", "check", "uncheck"} | ({"set"} if allow_value_binding else set())
    if not isinstance(edit.get("operation"), str) or edit["operation"] not in operations:
        raise ValueError("pursue_property edit operation must be fill, check or uncheck"
                         + ("; set is also available for a literal value" if allow_value_binding else ""))
    if (allow_goal_conditions and edit["operation"] in {"check", "uncheck"}
            and isinstance(edit.get("value"), str) and edit["value"] == ""):
        # An explicitly empty, unused field never supplies text or a new input.
        # Keep the raw model response unchanged and the old protocol strict.
        edit = {key: value for key, value in edit.items() if key != "value"}
    required = {"operation", "value"} if edit["operation"] in {"fill", "set"} else {"operation"}
    if set(edit) != required:
        raise ValueError("pursue_property.edit allows only operation,value for a literal edit, or only "
                         "operation for check/uncheck; omit target/ref because the policy binds "
                         "the actual field")
    numeric_finalize_key = ""
    if "numeric_finalize" in raw:
        strategy = raw["numeric_finalize"]
        if (not isinstance(strategy, dict) or set(strategy) != {"key"}
                or not isinstance(strategy["key"], str)
                or strategy["key"] not in ({"enter", "tab", "none"} if require_numeric_finalize_choice
                                           else {"enter", "tab"})):
            raise ValueError("numeric_finalize requires one explicit enter/tab/none strategy key"
                             if require_numeric_finalize_choice else
                             "numeric_finalize requires one explicit enter/tab key "
                             "and a bounded numeric edit")
        if strategy["key"] != "none" and (
                edit["operation"] not in {"fill", "set"}
                or not isinstance(edit.get("value"), str) or _bounded_decimal(edit["value"]) is None):
            raise ValueError("numeric_finalize requires one explicit enter/tab key "
                             "and a bounded numeric edit")
        numeric_finalize_key = "" if strategy["key"] == "none" else strategy["key"]
    step = AgentStep(
        "property", "click", label=_text(raw["label"], "label", 240, required=True),
        target=_navigation_scope(raw["scope"]), navigation=True, native_navigation_gestures=True,
        recovery="observe", property_edit=AgentPropertyEdit(
            edit["operation"], _text(edit.get("value", ""), "edit.value", 2000),
            bind_boolean_states=allow_observed_goal and edit["operation"] == "set",
            prefer_verified_selection=allow_observed_goal and edit["operation"] == "set",
            numeric_finalize_key=numeric_finalize_key,
        ),
    )
    if not valid_property_stage(step) or not snapshot.a11y_complete:
        raise ValueError("pursue_property requires a complete observation and a legal literal property edit")
    steps = (step,)
    if "prepare" in raw:
        from .desktop_intent_compiler import compile_stage_intent

        prepared = compile_stage_intent(
            {"intent": "act_current", "label": step.label, "action": raw["prepare"]},
            snapshot, max_actions=max_actions, inherit_current_action_label=True,
        )
        if len(prepared.steps) != 1:
            raise ValueError("property preparation requires exactly one current grounded action")
        preparation = replace(prepared.steps[0], id="prepare")
        step = replace(step, depends_on=("prepare",), property_prepare_dependency="prepare")
        steps = (preparation, step)
    if "confirm" in raw and not (require_explicit_stage_policy and raw["confirm"] == "none"):
        confirmation = raw["confirm"]
        if not isinstance(confirmation, dict) or set(confirmation) != {"label"}:
            raise ValueError("pursue_property.confirm permits only an explicit confirmation label")
        steps += (AgentStep(
            "confirm", "confirm_property",
            label=_text(confirmation["label"], "confirm.label", 240, required=True),
            depends_on=("property",), recovery="observe", property_confirmation_of="property",
        ),)
    if len(steps) > max_actions:
        raise ValueError("property preparation/edit/confirmation exceed the original logical step budget")
    if numeric_finalize_key and len(steps) + 1 > max_actions:
        raise ValueError("numeric finalization requires one reserved original logical step")
    return AgentPlan(step.label, steps, reusable=True, completion_conditions=goal_conditions,
                     defer_dialog_field_completion=allow_preparation or allow_observed_goal,
                     complete_property_goal=completion == "goal")


def pending_dialog_field_completion(plan: AgentPlan, snapshot: AgentSnapshot) -> dict | None:
    """Passively defer field-only stopping in an observed dialog or modal form.

    This authorizes no input and does not assert that a particular button saves.
    The next planning stage must explicitly decide any remaining confirmation.
    """
    if (plan.defer_dialog_field_completion is not True
            or not conditions_hold(plan.completion_conditions, snapshot)):
        return None
    fields = {"value_equals", "checked", "unchecked", "selected"}
    for condition in plan.completion_conditions:
        if condition.kind not in fields or condition.target is None:
            continue
        if not snapshot.a11y_complete:
            return {"reason": "incomplete_property_completion_observation", "task_success": False}
        controls = tuple(c for c in matching_controls(snapshot, condition.target)
                         if condition.kind in observable_condition_kinds(c))
        if len(controls) != 1:
            continue  # Ordinary condition evaluation retains false/unknown evidence.
        field = controls[0]
        if field.modal_context:
            return {"reason": "field_effect_in_open_modal", "task_success": False,
                    "field_role": field.role, "app": field.app, "window": field.window,
                    "modal_context": True}
        dialogs = tuple(c for c in snapshot.controls if normalize_role(c.role) == "dialog"
                        and c.app == field.app and c.window == field.window and not c.disabled)
        if (field.app or field.window) and dialogs:
            return {"reason": "field_effect_in_open_dialog", "task_success": False,
                    "field_role": field.role, "app": field.app, "window": field.window,
                    "dialog_count": len(dialogs)}
    return None
