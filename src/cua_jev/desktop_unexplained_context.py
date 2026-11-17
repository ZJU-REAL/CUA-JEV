"""Pure information predicate over receipt-bound navigation observations.

This consumes only supplied full snapshots and original conditions. It does no
I/O, input, scope handoff, receipt mutation or automatic predicate correction.
"""

import hashlib
import json

from .agent_types import AgentSnapshot, AgentStep
from .desktop_semantics import condition_evidence, navigation_control_evidence, navigation_scope_controls
from .models import Verification


def _app(value):
    return " ".join(value.split()).casefold()


def _owners(controls):
    return tuple(c for c in controls if c.tag == "container"
                 and (c.role in {"frame", "window", "dialog"} or c.modal is True))


def _key(owner):
    return (_app(owner.app), owner.role, owner.window, owner.label, owner.modal)


def _children(owner, controls, owners):
    if owner.modal is True and owner.ref:
        return tuple(c for c in controls if c is not owner and c.modal_owner_ref == owner.ref)
    if not owner.window or sum(c.window == owner.window for c in owners) != 1:
        return ()
    return tuple(c for c in controls if c is not owner and c.window == owner.window)


def _active(owner, children):
    return owner.active is True or (
        owner.modal is True and any(c.focused and c.modal_owner_ref == owner.ref for c in children)
    )


def unexplained_navigation_context(
    step: AgentStep, before: AgentSnapshot, after: AgentSnapshot,
    verification: Verification, *, owner_app: str,
) -> dict | None:
    """Suggest inspection, never continuation authority or goal/effect success.

    Caller MUST establish positive receipt identity/deep data, actual preflight
    before, exact final-effect after, original step, counters and verdict first.
    This pure helper cannot prove those private relationships from its inputs.
    """
    if (
        not isinstance(step, AgentStep) or not isinstance(before, AgentSnapshot)
        or not isinstance(after, AgentSnapshot) or not isinstance(verification, Verification)
        or step.operation != "click" or step.navigation is not True or step.recovery != "observe"
        or step.property_edit is not None or step.property_confirmation_of
        or not step.postconditions or step.target is None
        or before.a11y_complete is not True or after.a11y_complete is not True
        or verification.passed is not True or verification.details.get("navigation") is not True
        or verification.details.get("accessibility_progress") is not True
        or type(owner_app) is not str or not owner_app.strip()
    ):
        return None
    evidence = [condition_evidence(condition, after) for condition in step.postconditions]
    if all(fact["outcome"] == "true" for fact in evidence):
        return None
    # Scope resolution stays the original one; exact actual app is an additional
    # information-evidence restriction, never a new execution scope permission.
    app = _app(owner_app)
    old_scope = tuple(c for c in navigation_scope_controls(before, step.target) if _app(c.app) == app)
    new_scope = tuple(c for c in navigation_scope_controls(after, step.target) if _app(c.app) == app)
    if not old_scope or not new_scope:
        return None
    old_owners, new_owners = _owners(old_scope), _owners(new_scope)
    old_keys, new_keys = {_key(c) for c in old_owners}, {_key(c) for c in new_owners}
    active_owners = tuple((owner, children) for owner in new_owners
                          if (owner.role == "dialog" or owner.modal is True)
                          and _active(owner, children := _children(owner, new_scope, new_owners)))
    # A single novel candidate is not proof of a unique active modal context.
    # Even a previously observed active dialog makes an unproven hierarchy ambiguous.
    if len(active_owners) != 1:
        return None
    candidates = []
    for owner, children in active_owners:
        # Initial narrow implementation covers observed dialogs/reported modal
        # containers; ordinary new nonmodal frames remain unsupported/unknown.
        # A container is an observed owner, not an input target. Its missing
        # enabled/sensitive flags may set disabled without invalidating its
        # explicit active/modal identity. Only actual executable descendants
        # make this an actionable context; their input gates stay unchanged.
        if not (owner.role == "dialog" or owner.modal is True):
            continue
        if _key(owner) in old_keys:
            continue
        same_kind = tuple(c for c in old_owners if (c.role, c.modal) == (owner.role, owner.modal))
        # Without stable cross-observation owner identity, replacing an equally
        # typed owner with a renamed title is not enough. A new type/modal state,
        # or an extra owner while all prior same-kind owners remain, is stronger.
        if same_kind and not all(_key(c) in new_keys for c in same_kind):
            continue
        executable = tuple(c for c in children if not c.disabled
                           and (fact := navigation_control_evidence(c)) and fact["operations"])
        if executable:
            candidates.append((owner, len(executable)))
    if len(candidates) != 1:
        return None
    owner, count = candidates[0]
    digest = hashlib.sha256(json.dumps(_key(owner), ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "reason": "unexplained_new_dialog_after_verified_navigation",
        "context_fingerprint": digest,
        "owner_role": owner.role if owner.role in {"frame", "window", "dialog"} else "reported_modal",
        "reported_modal": owner.modal,
        "executable_descendant_count": count,
        "arrival_evidence": [{"kind": condition.kind, "outcome": fact["outcome"],
                              "reason": fact["reason"]}
                             for condition, fact in zip(step.postconditions, evidence, strict=True)],
        "task_success": False,
        "input_authority": False,
    }
