"""Compile explicit desktop editing intent into an observed, reusable stage.

Only this compiler creates the internal readiness predicates. Model-authored
legacy predicates still pass through the original, narrower condition parser.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .agent_types import AgentCondition, AgentPlan, AgentSnapshot
from .desktop_semantics import normalize_role


def compile_reach_edit(raw: Any, snapshot: AgentSnapshot, max_actions: int = 6) -> AgentPlan:
    """Validate one explicit intent and compile reach -> edit, never infer intent."""
    if type(max_actions) is not int or not 2 <= max_actions <= 12:
        raise ValueError("reach_edit requires desktop multi-action planning with room for two steps")
    required = {"intent", "label", "scope", "edit"}
    if (
        not isinstance(raw, dict) or not required <= set(raw)
        or set(raw) - (required | {"completion"}) or raw["intent"] != "reach_edit"
    ):
        raise ValueError("reach_edit requires only intent, label, scope, edit, and optional completion")
    label = raw["label"]
    if not isinstance(label, str) or not label.strip() or len(label) > 240:
        raise ValueError("reach_edit.label requires a nonempty string of at most 240 characters")
    completion = raw.get("completion", "stage")
    if not isinstance(completion, str) or completion not in {"stage", "goal"}:
        raise ValueError("reach_edit.completion must be stage or goal")
    edit = raw["edit"]
    if (
        not isinstance(edit, dict) or not {"operation", "target"} <= set(edit)
        or set(edit) - {"operation", "target", "value"}
    ):
        raise ValueError("reach_edit.edit requires operation, target, and value only for fill")
    operation = edit["operation"]
    if not isinstance(operation, str) or operation not in {"fill", "check", "uncheck"}:
        raise ValueError("reach_edit.edit.operation must be fill, check, or uncheck")
    if (operation == "fill") != ("value" in edit):
        raise ValueError("reach_edit.edit.value is required only for fill")

    # Reuse the legacy parser's target, scope, text, dependency and action
    # validation. This temporary skeleton is never executed or returned. Its
    # ordinary effect predicates are replaced only after all validation passes.
    from .agent_planner import parse_plan

    effect = {"kind": {"fill": "value_equals", "check": "checked", "uncheck": "unchecked"}[operation],
              "target": edit["target"]}
    if operation == "fill":
        effect["value"] = edit["value"]
    plan = parse_plan({
        "summary": label,
        "done": False,
        "steps": [
            {"id": "reach", "operation": "click", "navigation": True, "label": label,
             "target": raw["scope"], "postconditions": [effect], "recovery": "observe"},
            {"id": "edit", **edit, "label": label, "depends_on": ["reach"],
             "preconditions": [effect], "postconditions": [effect], "recovery": "observe"},
        ],
    }, snapshot, max_actions=max_actions, profile="desktop")
    reach, mutation = plan.steps
    target = mutation.target
    if target is None or not target.name or not (target.app or target.window):
        raise ValueError("reach_edit.edit.target requires a name and explicit app or window context")
    if operation == "fill" and normalize_role(target.role) in {"checkbox", "radio"}:
        raise ValueError(
            "reach_edit.edit.target.role conflicts with fill: checkbox/radio describe checkable "
            "states, not text fields"
        )
    ready = AgentCondition(f"ready_to_{operation}", target, mutation.value)
    # Retain readiness in effect and goal checks: a truncated observation must
    # not prove uniqueness just because one matching checked/value state remains.
    verified = (*mutation.postconditions, ready)
    return replace(
        plan,
        steps=(replace(reach, postconditions=(ready,), native_navigation_gestures=True),
               replace(mutation, preconditions=(ready,), postconditions=verified)),
        completion_conditions=verified if completion == "goal" else (),
    )
