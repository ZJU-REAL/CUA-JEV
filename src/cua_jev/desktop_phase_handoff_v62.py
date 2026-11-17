"""Immutable zero-input request for the original strong current-action phase."""

import re
from dataclasses import dataclass

CONTEXT_REASONS = frozenset({
    "unsupported_high_level_action",
    "missing_future_semantics",
    "transient_or_unexplained_state",
})
CONTEXT_TOOL = "declare_current_context_needed"


@dataclass(frozen=True)
class HighLevelContextRequest:
    """No plan, target, input capability, completion or new execution budget."""

    reason: str
    declaration_sha256: str

    def __post_init__(self):
        if type(self.reason) is not str or self.reason not in CONTEXT_REASONS:
            raise ValueError("known high-level context reason required")
        if (type(self.declaration_sha256) is not str
                or re.fullmatch(r"[a-f0-9]{64}", self.declaration_sha256) is None):
            raise ValueError("high-level context declaration SHA256 required")

    def to_dict(self):
        return {"reason": self.reason, "declaration_sha256": self.declaration_sha256}
