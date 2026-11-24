"""Explicit new-protocol candidate coverage; no target resolution or input."""

from dataclasses import fields, replace

from .agent_types import AgentPlan, AgentStep


class BudgetedLegalNavigationStep(AgentStep):
    """Compiler-owned view: same immutable fields and original input semantics."""

    @property
    def full_legal_navigation_gestures(self) -> bool:
        return True


def enable_full_legal_navigation_gestures(plan: AgentPlan) -> AgentPlan:
    """Opt in compiled navigation only, preserving all actions and predicates.

    Old protocols do not call this function. Current actions, preparation,
    property mutations and confirmations keep exactly their declared gesture.
    Every new gesture candidate still needs executable_control, the original
    candidate budget, Jev selection, fresh binding, receipt and effect checks.
    An expansion exceeding that budget falls back to the complete original
    navigation pool, never an arbitrary prefix. A still oversized pool is refused.
    """
    if not isinstance(plan, AgentPlan):
        raise ValueError("compiled original plan required")
    return replace(plan, steps=tuple(
        BudgetedLegalNavigationStep(**{item.name: getattr(step, item.name) for item in fields(AgentStep)})
        if step.navigation and step.native_navigation_gestures else step
        for step in plan.steps
    ))
