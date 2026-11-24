'Bounded read-only diagnostics for controls and navigation conditions.'

import re
from dataclasses import asdict, replace

from .agent_types import AgentSnapshot, AgentTarget
from .desktop_semantics import (
    desktop_click_modes,
    executable_control,
    matching_controls,
    normalize_role,
    observable_condition_kinds,
)

_KINDS = frozenset({
    "exists", "absent", "focused", "checked", "unchecked", "selected", "expanded",
    "value_equals", "ready_to_fill", "ready_to_check", "ready_to_uncheck",
})
_TARGET_FIELDS = frozenset(AgentTarget.__dataclass_fields__)
_GUIDANCE = (
    "These are same-name controls in the CURRENT planning observation, retrieved only to explain "
    "a historical unknown predicate. They are not equivalent targets, replacement predicates or "
    "recommended inputs. The original app/window/ancestor constraints are unchanged; only role "
    "was omitted for this diagnostic lookup. Operations and condition_kinds are individual "
    "observed capabilities, not a unique executable binding or proof of an effect. Inspect the "
    "current evidence and explicitly return a valid plan. Unknown does not prove input failure."
)
_MENU_CAPTION_GUIDANCE = (
    "These are CURRENT public controls retrieved to explain historical unknown predicates, "
    "not replacement targets or recommended inputs. A menu_caption_and_shortcut_text match "
    "compares captions while ignoring at most one terminal ... or … decoration and/or bounded "
    "chord-like suffix text. Full labels are preserved below. This is a read-only retrieval "
    "heuristic, NOT a matching alias. "
    "The original predicate remains unknown; its app/window/ancestor constraints are preserved. "
    "Inspect full observed labels and their actual window values, including an empty window "
    "for an anonymous menu. If choosing such a control, explicitly use its full observed "
    "label/ref and current capabilities; do not invent a parent-window qualifier. Exact queries "
    "and all input/verification rules remain unchanged. Unknown does not prove input failure."
)
_SHORTCUT_TEXT = re.compile(r"(?i)(?:(?:ctrl|control|alt|shift|super|meta|cmd|command)\+){1,4}[^\s+]{1,16}")


def _menu_caption(value: str) -> tuple[str, bool]:
    """Normalize one terminal dialog decoration for retrieval only, never input matching."""
    caption = " ".join(value.split()).casefold()
    for ending in ("...", "…"):
        if caption.endswith(ending):
            return caption[:-len(ending)].rstrip(), True
    return caption, False


def _menu_caption_matches(snapshot: AgentSnapshot, target: AgentTarget) -> tuple:
    """Explain a missing exact name without creating an executable alias."""
    wanted, wanted_decoration = _menu_caption(target.name)
    if not wanted:
        return ()
    result = []
    for control in snapshot.controls:
        if normalize_role(control.role) != "menuitem" or len(control.label) > 240:
            continue
        pieces = control.label.split()
        suffix = 0
        while suffix < 3 and len(pieces) > 1 and _SHORTCUT_TEXT.fullmatch(pieces[-1]):
            pieces.pop()
            suffix += 1
        caption, observed_decoration = _menu_caption(" ".join(pieces))
        if not (suffix or wanted_decoration or observed_decoration) or caption != wanted:
            continue
        # Actual name only for this diagnostic lookup. All requested ownership
        # qualifiers remain binding; disabled peers still compete.
        if control in matching_controls(snapshot, replace(target, name=control.label, role="")):
            result.append(control)
    return tuple(result)


def _target(raw: object) -> AgentTarget | None:
    if not isinstance(raw, dict) or set(raw) - _TARGET_FIELDS:
        return None
    if any(not isinstance(value, str) for value in raw.values()):
        return None
    target = AgentTarget(**raw)
    if (
        target.match != "exact" or not target.name.strip()
        or not (target.app.strip() or target.window.strip())
        or any(len(getattr(target, field)) > (80 if field == "role" else 240)
               for field in ("name", "role", "app", "window", "ancestor"))
    ):
        return None
    return target


def observed_target_diagnostics(snapshot: AgentSnapshot, history: list[dict]) -> dict | None:
    """Retrieve at most two predicates/four controls each without exposing values.

    Only the newest actual action's unverified semantic postcondition evidence
    is eligible. Later planning failures may retain that context, but a newer
    action (including a successful or nonsemantic one) ends its eligibility.
    """
    if not snapshot.a11y_complete:
        return None
    latest = next((item for item in reversed(history[-8:])
                   if isinstance(item, dict) and "action" in item and "verified" in item), None)
    if (
        latest is None or latest.get("verified") is not False
        or not isinstance(latest.get("operation"), str)
        or not isinstance(latest.get("postconditions"), list)
    ):
        return None
    predicates = []
    caption_retrieved = False
    for fact in latest["postconditions"][:8]:
        if (
            not isinstance(fact, dict) or fact.get("outcome") != "unknown"
            or fact.get("reason") not in {"target_missing", "unsupported_state"}
            or fact.get("kind") not in _KINDS
        ):
            continue
        target = _target(fact.get("target"))
        if target is None:
            continue
        matches = matching_controls(snapshot, replace(target, role=""))
        by_caption = False
        if not matches:
            matches = _menu_caption_matches(snapshot, target)
            by_caption = bool(matches)
        if not matches:
            continue
        # Do not expose sensitive controls or imply uniqueness by hiding one
        # of the matches. Omit this diagnostic entirely if any match is private.
        if any(control.input_type.lower() in {"password", "file", "hidden"} for control in matches):
            continue
        shown = []
        for control in matches[:4]:
            item = {
                "ref": control.ref, "label": control.label, "role": control.role,
                "app": control.app, "window": control.window,
                "operations": [operation for operation in ("click", "fill", "check", "uncheck")
                               if executable_control(control, operation)],
                "condition_kinds": list(observable_condition_kinds(control)),
            }
            if target.ancestor:
                # The query already preserves this qualifier. No unrelated
                # ancestor text is needed in the retrieved control summary.
                item["matches_requested_ancestor"] = True
            if control.native is not None:
                item["click_modes"] = [
                    {"button": button, "click_count": count}
                    for button, count in desktop_click_modes(control)
                ]
            shown.append(item)
        predicates.append({
            "historical_predicate": {"kind": fact["kind"], "target": asdict(target),
                                     "outcome": "unknown", "reason": fact["reason"]},
            "match_count": len(matches), "truncated": len(matches) > len(shown),
            "observed_controls": shown,
            **({"retrieval": "menu_caption_and_shortcut_text"} if by_caption else {}),
        })
        caption_retrieved = caption_retrieved or by_caption
        if len(predicates) == 2:
            break
    return ({"source": "current_planning_snapshot",
             "guidance": _MENU_CAPTION_GUIDANCE if caption_retrieved else _GUIDANCE,
             "predicates": predicates} if predicates else None)


def observed_navigation_arrival_diagnostics(snapshot, step, facts, offered) -> dict | None:
    """Explain a current delegated menu-arrival mismatch, without selecting input.

    The v60 runner must independently prove this is its current full fresh
    observation and supply the complete offers just compiled from that frame.
    This helper performs no observation, policy request, history mutation,
    candidate selection, input, condition rewrite or task-completion claim.

    Ordinary OSWorld controls do not retain NativeDesktopEvidence. Their shown
    and enabled provenance comes from the original adapter: _shown() filters
    hidden AT-SPI nodes before materialization; disabled requires positive
    enabled/sensitive; a full exact current a11y offer retains that control and
    geometry. Explicit native states, when present, must additionally be true.
    """
    from .agent_types import AgentControl, AgentStep, NativeDesktopEvidence
    from .desktop_semantics import (
        condition_evidence,
        native_navigation_click_modes,
        navigation_scope_controls,
        valid_desktop_box,
    )
    from .models import ActionCandidate

    if (not isinstance(snapshot, AgentSnapshot) or snapshot.a11y_complete is not True
            or snapshot.image is None or not isinstance(snapshot.image.jpeg, bytes)
            or not snapshot.image.jpeg or not isinstance(step, AgentStep)
            or step.operation != "click" or step.navigation is not True
            or step.property_edit is not None or step.property_confirmation_of
            or step.ref or step.box is not None or step.target is None
            or step.target.name or step.target.role or step.target.match != "exact"
            or not isinstance(facts, (list, tuple)) or not 1 <= len(facts) <= 8
            or len(facts) != len(step.postconditions)
            or not isinstance(offered, (list, tuple)) or not 1 <= len(offered) <= 255):
        return None
    if any(not isinstance(control, AgentControl) or type(control.ref) is not str
           or not control.ref for control in snapshot.controls):
        return None
    controls = {control.ref: control for control in snapshot.controls}
    if len(controls) != len(snapshot.controls) or any(not ref for ref in controls):
        return None
    fingerprint = snapshot.fingerprint()
    scoped = navigation_scope_controls(snapshot, step.target)
    # Ordinary scope only: no receipt-authorized cross-app handoff is inferred
    # by this explanation aid. The existing named-owner memory is untouched.
    targets = tuple(control for control in scoped if executable_control(
        control, "click", button=step.button, click_count=step.click_count,
    ))
    expected = set()
    for control in targets:
        modes = (desktop_click_modes(control) if step.full_legal_navigation_gestures is True
                 else native_navigation_click_modes(control)) if step.native_navigation_gestures else (
                     (step.button, step.click_count),
                 )
        expected.update((control.ref, button, count) for button, count in modes)
    if (step.native_navigation_gestures and step.full_legal_navigation_gestures is True
            and len(expected) > 255):
        expected = {
            (control.ref, button, count) for control in targets
            for button, count in native_navigation_click_modes(control)
        }
    if not 1 <= len(expected) <= 255:
        return None
    actual, candidate_ids = set(), set()
    for candidate in offered:
        if not isinstance(candidate, ActionCandidate) or candidate.id in candidate_ids:
            return None
        candidate_ids.add(candidate.id)
        args = candidate.arguments
        bound = args.get("bound_control")
        control = (controls.get(bound["ref"]) if isinstance(bound, dict)
                   and type(bound.get("ref")) is str else None)
        button, count = args.get("button", "left"), args.get("click_count", 1)
        if (candidate.channel != "gui" or candidate.capability != "agent.osworld.click"
                or candidate.intent != step.id or args.get("operation") != "click"
                or args.get("route") != "osworld-a11y" or args.get("navigation") is not True
                or args.get("navigation_app", "") or args.get("value", "")
                or args.get("fingerprint") != fingerprint or control is None
                or control.to_dict() != bound or args.get("box") != control.box
                or args.get("target") != asdict(step.target)
                or not executable_control(control, "click", button=button, click_count=count)):
            return None
        key = (control.ref, button, count)
        if key in actual:
            return None
        actual.add(key)
    if actual != expected:
        return None  # Do not treat a truncated or modified offer prefix as full.

    def positive_native_states(control):
        native = control.native
        return native is None or (
            isinstance(native, NativeDesktopEvidence)
            and native.complete is True and native.enabled is True and native.sensitive is True
            and native.visible is True and native.showing is True
            and native.parent_enabled is not False and native.parent_sensitive is not False
            and native.ancestors_enabled is not False and native.ancestors_sensitive is not False
        )

    # Facts must correspond to the original predicates evaluated on this frame;
    # an earlier historical unknown may not be substituted for a current truth.
    for condition, fact in zip(step.postconditions, facts, strict=True):
        if not isinstance(fact, dict):
            return None
        evidence = condition_evidence(condition, snapshot)
        if (fact.get("kind") != condition.kind
                or fact.get("target") != (asdict(condition.target) if condition.target else None)
                or fact.get("outcome") != evidence["outcome"]
                or fact.get("reason") != evidence["reason"]
                or type(fact.get("satisfied")) is not bool
                or fact["satisfied"] != (evidence["outcome"] == "true")):
            return None
    predicates = []
    for index, (condition, fact) in enumerate(zip(step.postconditions, facts, strict=True)):
        target = condition.target
        if (condition.kind not in _KINDS or fact["outcome"] != "unknown"
                or fact["reason"] != "target_missing" or target is None
                or normalize_role(target.role) != "menuitem"
                or _target(asdict(target)) is None or matching_controls(snapshot, target)):
            continue
        matches = _menu_caption_matches(snapshot, target)
        if (not matches or any(
                control.input_type.lower() in {"password", "file", "hidden"}
                or not control.active or control.disabled or not valid_desktop_box(control.box)
                or not executable_control(control, "click") or not positive_native_states(control)
                or not any(ref == control.ref for ref, _button, _count in actual)
                or control not in scoped
                or any(len(value) > limit for value, limit in (
                    (control.ref, 128), (control.label, 240), (control.role, 80),
                    (control.app, 240), (control.window, 240),
                ))
                for control in matches)):
            continue  # Never filter hidden/disabled/private peers into uniqueness.
        owners = {(control.app, control.window) for control in matches}
        if len(owners) != 1:
            continue
        app, window = next(iter(owners))
        roots = tuple(control for control in snapshot.controls
                      if control.app == app and control.active
                      and normalize_role(control.role) in {"frame", "window", "dialog"})
        if (len(roots) != 1 or roots[0].window != window or roots[0].disabled
                or len(roots[0].ref) > 128 or len(roots[0].role) > 80
                or not positive_native_states(roots[0])):
            continue
        # Owner identity supplies diagnostic scope, never a click target. Only
        # the actual menu controls above require executable input geometry.
        owner = roots[0]
        shown = [{
            "ref": control.ref, "label": control.label, "role": control.role,
            "app": control.app, "window": control.window,
            "operations": ["click"],
            "evidence_source": "original_OSWorld_a11y_adapter_and_exact_current_legal_offer",
            **({"matches_requested_ancestor": True} if target.ancestor else {}),
        } for control in matches[:4]]
        predicates.append({
            "fact_index": index,
            "original_predicate": {
                "kind": condition.kind, "target": asdict(target),
                "outcome": "unknown", "reason": "target_missing", "satisfied": False,
            },
            "retrieval": "menu_caption_and_shortcut_text",
            "match_count": len(matches), "truncated": len(matches) > len(shown),
            "observed_controls": shown,
            "current_menu_owner": {
                "ref": owner.ref, "role": owner.role, "app": owner.app, "window": owner.window,
            },
        })
        if len(predicates) == 2:
            break
    return ({
        "version": "observed_navigation_arrival_caption_diagnostics_v1",
        "code": "navigation_arrival_caption_mismatch",
        "source": "current_delegated_navigation_snapshot",
        "step_id": step.id, "snapshot_fingerprint": fingerprint,
        "predicates": predicates, "guidance": _MENU_CAPTION_GUIDANCE,
        "input_sent": False, "input_authority": False, "task_success": False,
        "conditions_rewritten": False,
    } if predicates else None)
