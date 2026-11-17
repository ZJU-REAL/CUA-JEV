"""Explicit final confirmation phases, bound only to a verified actual field."""

from .agent_types import AgentCondition, AgentSnapshot, AgentStep, AgentTarget
from .desktop_semantics import conditions_hold, executable_controls, matching_controls, normalize_role


def valid_property_confirmation(step: AgentStep) -> bool:
    """An inert compiler template; ordinary executors do not accept its operation."""
    return bool(
        step.operation == "confirm_property" and step.id == "confirm"
        and step.property_confirmation_of == "property" and step.depends_on == ("property",)
        and isinstance(step.label, str) and 0 < len(step.label.strip()) <= 240
        and not step.ref and not step.value and step.box is None and step.target is None
        and not step.preconditions and not step.postconditions and step.recovery == "observe"
        and step.navigation is False and step.native_navigation_gestures is False
        and step.property_edit is None and not step.property_prepare_dependency
        and step.derived_dialog_commit_effect is False and step.button == "left" and step.click_count == 1
    )


def property_confirmation_steps(
    template: AgentStep, field: AgentStep, current: AgentSnapshot,
) -> tuple[AgentStep, ...]:
    """Enumerate buttons in a uniquely observed dialog or actual native modal owner.

    The runner supplies ``field`` only from the selected property's verified
    receipt. This function performs no input, picks no button and claims no
    task success. Every returned action uses the ordinary desktop executor.
    """
    if (not valid_property_confirmation(template) or not current.a11y_complete
            or field.id != template.property_confirmation_of or field.target is None
            or field.navigation or not field.postconditions
            or not (field.operation in {"fill", "check", "uncheck"}
                    or (field.operation == "click"
                        and any(c.kind == "selected" for c in field.postconditions)))):
        return ()
    matches = matching_controls(current, field.target)
    if len(matches) != 1 or not conditions_hold(field.postconditions, current):
        return ()
    control = matches[0]
    if (control.app, control.window) != (field.target.app, field.target.window):
        return ()  # A substring match must not transfer a receipt to another form's owner.
    structural = bool(control.modal_owner_ref)
    dialogs = (tuple(c for c in current.controls if c.ref == control.modal_owner_ref
                     and c.modal is True and c.tag == "container"
                     and (c.app, c.window) == (control.app, control.window)) if structural else
               tuple(c for c in current.controls if normalize_role(c.role) == "dialog"
                     and (c.app, c.window) == (control.app, control.window)))
    if len(dialogs) != 1:
        return ()
    dialog = dialogs[0]
    ancestry = f"{dialog.role}: {dialog.label}"
    if (dialog.disabled or not dialog.app or not dialog.window
            or not (dialog.active or (structural and control.focused))
            or (not structural and (not dialog.label or ancestry not in control.ancestors))):
        return ()
    # Reuse the current explicit-commit ownership/identity checks. No guessed
    # dialog name, button label, keyboard key, target or value is introduced.
    from .desktop_intent_compiler import _current_dialog_effect, _current_ref_target

    output = []
    seen = set()
    scope = AgentTarget(role="button", app=dialog.app, window=dialog.window,
                        ancestor="" if structural else ancestry)
    for button in executable_controls(current, "click", scope):
        if structural and button.modal_owner_ref != dialog.ref:
            continue
        try:
            target = _current_ref_target({"ref": button.ref}, current)
            owner, dialog_target, _effect = _current_dialog_effect({
                "operation": "click", "target": target,
            }, current, allow_native_modal_owners=True)
        except ValueError:
            continue
        if owner != dialog:
            continue
        actual = AgentTarget(**target)
        if actual in seen:
            continue
        seen.add(actual)
        output.append(AgentStep(
            template.id, "click", label=template.label, target=actual, depends_on=template.depends_on,
            preconditions=field.postconditions, postconditions=(AgentCondition("absent", dialog_target),),
            recovery="observe", derived_dialog_commit_effect=True,
        ))
        if len(output) > 255:
            return ()
    return tuple(output)
