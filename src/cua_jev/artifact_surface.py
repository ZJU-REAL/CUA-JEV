"""One scoped, create-only text artifact for open computer-use goals."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .executors.common import execute_with_receipt
from .models import ActionCandidate, ActionReceipt, Channel, Risk, Verification
from .runtime import StepResult
from .verify import VerifierRegistry


@dataclass(frozen=True)
class ArtifactState:
    ref: str
    name: str
    exists: bool
    ready: bool
    sha256: str = ""
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        # The file's contents stay in the verifier, not in model/Jev observations.
        return {
            "ref": self.ref, "name": self.name, "exists": self.exists,
            "ready": self.ready, "sha256": self.sha256,
        }


class ArtifactSurface:
    namespace = "a"
    supported_capabilities = frozenset({"artifact.write_text"})

    def __init__(self, path: Path, *, max_chars: int = 8000) -> None:
        self.path = path.resolve()
        if self.path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("artifact must be a .md or .txt file")
        if not 1 <= max_chars <= 20_000:
            raise ValueError("artifact size limit is invalid")
        self.max_chars = max_chars
        self.ready = False
        self.required_citations: tuple[str, ...] = ()

    def set_acceptance(self, *, ready: bool, citations: Sequence[str]) -> None:
        self.ready = ready
        self.required_citations = tuple(citations)

    def reset(self) -> None:
        if self.path.exists():
            raise ValueError("refusing to overwrite an existing artifact")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        pass

    def observe(self, history: Sequence[StepResult]) -> ArtifactState:
        return self.capture()

    def capture(self) -> ArtifactState:
        if not self.path.exists():
            return ArtifactState("a0", self.path.name, False, self.ready)
        if not self.path.is_file() or self.path.stat().st_size > self.max_chars * 4:
            raise ValueError("artifact changed to an unsupported target")
        text = self.path.read_text(encoding="utf-8")
        return ArtifactState(
            "a0", self.path.name, True, self.ready,
            hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
        )

    def validate(
        self, state: ArtifactState, ref: str, operation: Any, value: Any
    ) -> tuple[str, str, str]:
        if ref != state.ref or operation != "write" or state.exists or not state.ready:
            raise ValueError("artifact write is not offered until source gates pass")
        if not isinstance(value, str) or not 1 <= len(value) <= self.max_chars:
            raise ValueError("artifact text is empty or exceeds its size limit")
        if any(url not in value for url in self.required_citations):
            raise ValueError("artifact must cite each visited source URL")
        return ref, "write", value

    def compile(
        self, state: ArtifactState, ref: str, operation: str, value: str,
        subgoal: str, index: int,
    ) -> tuple[ActionCandidate, ...]:
        self.validate(state, ref, operation, value)
        return (ActionCandidate(
            f"option_{index}_artifact", Channel.API, "artifact.write_text",
            f"Create the scoped {self.path.name} artifact",
            {"path": str(self.path), "text": value}, Risk.LOCAL_WRITE,
            verifier="artifact.exact_text", intent=f"option_{index}",
        ),)

    def owns(self, candidate: ActionCandidate) -> bool:
        return candidate.capability == "artifact.write_text"

    def execute(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        def create() -> dict[str, Any]:
            if candidate.arguments["path"] != str(self.path):
                raise ValueError("artifact target changed")
            text = candidate.arguments["text"]
            if any(url not in text for url in self.required_citations):
                raise ValueError("artifact citations changed")
            with self.path.open("x", encoding="utf-8") as handle:
                handle.write(text)
            return {"path": str(self.path), "characters": len(text)}

        return execute_with_receipt(candidate, observation_id, decision_id, create)

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("artifact.exact_text", self._verify)

    def _verify(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        state = self.capture()
        passed = (
            receipt.success and state.exists
            and state.text == candidate.arguments["text"]
            and receipt.output.get("path") == str(self.path)
        )
        return Verification(bool(passed), "artifact.exact_text", {
            "exists": state.exists, "exact_text": bool(passed),
        })
