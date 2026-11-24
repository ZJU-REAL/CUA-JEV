"""Dispatch on complete candidates, before the model's privacy projection."""

import json

from .models import Decision


class DispatchEscalation(RuntimeError):
    def __init__(self, reason):
        if reason not in {"incomplete_observation", "uncertain_choice"}:
            raise ValueError("unsupported dispatch reason")
        self.reason = reason
        super().__init__(reason)


def identical_execution_candidates(candidates):
    """Prove exact execution aliases, never infer DOM/mouse effect equivalence."""
    if len(candidates) < 2 or len({c.id for c in candidates}) != len(candidates):
        return False
    try:
        signatures = [json.dumps({k: v for k, v in c.to_dict().items() if k != "id"},
                                 sort_keys=True, allow_nan=False) for c in candidates]
    except (TypeError, ValueError):
        return False
    return all(s == signatures[0] for s in signatures[1:])


class DispatchPolicy:
    def __init__(self, model_policy, *, emit=None, selection_kind="jev"):
        if selection_kind not in {"jev", "plan_order"}:
            raise ValueError("unsupported selection kind")
        self.selection_kind = selection_kind
        self.model_policy = model_policy
        self.emit = emit or (lambda _kind, _payload: None)
        self.singleton_decisions = 0
        self.equivalent_decisions = 0
        self._used_model = False

    @property
    def last_exchange(self):
        return getattr(self.model_policy, "last_exchange", {}) if self._used_model else {}

    def choose(self, observation, candidates):
        self._used_model = False
        confirmation = observation.state.get("explicit_property_confirmation") is True
        singleton = len(candidates) == 1 and not confirmation
        equivalent = not confirmation and identical_execution_candidates(candidates)
        if singleton or equivalent:
            reason = "singleton" if singleton else "identical_execution_aliases"
            self.singleton_decisions += singleton
            self.equivalent_decisions += equivalent
            selected = candidates[0]
            self.emit("agent_dispatch", {"route": "rule", "reason": reason,
                      "candidate_count": len(candidates), "selected_id": selected.id,
                      "observation_id": observation.observation_id, "input_sent": False})
            return Decision(observation.observation_id, selected.id,
                            {c.id: float(c.id == selected.id) for c in candidates},
                            1.0, "deterministic-" + reason, 0.0)
        if (observation.state.get("a11y_complete") is False
                and observation.state.get("candidate_scope_complete") is not True):
            raise DispatchEscalation("incomplete_observation")
        self._used_model = True
        self.emit("agent_dispatch", {"route": self.selection_kind,
                  "reason": "multiple_candidates_or_confirmation",
                  "candidate_count": len(candidates), "observation_id": observation.observation_id,
                  "input_sent": False})
        return self.model_policy.choose(observation, candidates)
