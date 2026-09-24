"""One scoped, create-only text artifact for open computer-use goals."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import time
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
    editor_open: bool = False
    can_open_editor: bool = False

    def to_dict(self) -> dict[str, Any]:
        # The file's contents stay in the verifier, not in model/Jev observations.
        return {
            "ref": self.ref, "name": self.name, "exists": self.exists,
            "ready": self.ready, "sha256": self.sha256,
            "editor_open": self.editor_open, "can_open_editor": self.can_open_editor,
        }


class ArtifactSurface:
    namespace = "a"
    supported_capabilities = frozenset({"artifact.write_text"})

    def __init__(
        self, path: Path, *, max_chars: int = 8000,
        allow_open_vscode: bool = False, allow_open_notepad: bool = False,
        editor_probe: Any | None = None,
        editor_launcher: Any | None = None,
    ) -> None:
        self.path = path.resolve()
        if self.path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("artifact must be a .md or .txt file")
        if not 1 <= max_chars <= 20_000:
            raise ValueError("artifact size limit is invalid")
        if allow_open_vscode and allow_open_notepad:
            raise ValueError("choose exactly one editor handoff target")
        self.max_chars = max_chars
        self.allow_open_vscode = allow_open_vscode
        self.allow_open_notepad = allow_open_notepad
        self.open_capability = (
            "artifact.open_vscode" if allow_open_vscode
            else "artifact.open_notepad" if allow_open_notepad else ""
        )
        self.editor_probe = editor_probe
        self.editor_launcher = editor_launcher or (
            self._launch_vscode if allow_open_vscode else self._launch_notepad
        )
        self.opened_window_handle: int | None = None
        self._opened_test_editor = False
        if self.open_capability:
            self.supported_capabilities = frozenset({
                "artifact.write_text", self.open_capability,
            })
        self.ready = False
        self.required_citations: tuple[str, ...] = ()

    def set_acceptance(self, *, ready: bool, citations: Sequence[str]) -> None:
        self.ready = ready
        self.required_citations = tuple(citations)

    def reset(self) -> None:
        if self.path.exists():
            raise ValueError("refusing to overwrite an existing artifact")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.opened_window_handle = None
        self._opened_test_editor = False

    def close(self) -> None:
        pass

    def observe(self, history: Sequence[StepResult]) -> ArtifactState:
        return self.capture()

    def capture(self) -> ArtifactState:
        if not self.path.exists():
            return ArtifactState(
                "a0", self.path.name, False, self.ready,
                can_open_editor=bool(self.open_capability),
            )
        if not self.path.is_file() or self.path.stat().st_size > self.max_chars * 4:
            raise ValueError("artifact changed to an unsupported target")
        text = self.path.read_text(encoding="utf-8")
        return ArtifactState(
            "a0", self.path.name, True, self.ready,
            hashlib.sha256(text.encode("utf-8")).hexdigest(), text,
            self._editor_is_open(),
            bool(self.open_capability),
        )

    def _editor_windows(self) -> dict[int, str]:
        try:
            from pywinauto import Desktop
        except ImportError:
            return {}
        editor_name = "Visual Studio Code" if self.allow_open_vscode else "Notepad"
        matches = Desktop(backend="uia").windows(
            title_re=rf".*{re.escape(self.path.name)}.*{editor_name}.*",
            visible_only=True,
        )
        return {int(item.handle): str(item.window_text()) for item in matches}

    def _editor_is_open(self) -> bool:
        if not self.open_capability:
            return False
        if self.editor_probe is not None:
            return self._opened_test_editor and bool(self.editor_probe())
        return (
            self.opened_window_handle is not None
            and self.opened_window_handle in self._editor_windows()
        )

    def _launch_vscode(self) -> None:
        executable = shutil.which("code") or shutil.which("code.cmd")
        if not executable:
            raise RuntimeError("VS Code CLI is not installed")
        subprocess.Popen([executable, "--new-window", str(self.path)], shell=False)

    def _launch_notepad(self) -> None:
        executable = shutil.which("notepad.exe")
        if not executable:
            raise RuntimeError("Windows Notepad is not installed")
        subprocess.Popen([executable, str(self.path)], shell=False)

    def validate(
        self, state: ArtifactState, ref: str, operation: Any, value: Any
    ) -> tuple[str, str, str]:
        if ref != state.ref:
            raise ValueError("artifact ref changed")
        if operation == "open":
            if not self.open_capability or not state.exists or state.editor_open or value:
                raise ValueError("editor open is not currently offered")
            return ref, "open", ""
        if operation != "write" or state.exists or not state.ready:
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
        if operation == "open":
            editor = "VS Code" if self.allow_open_vscode else "Notepad"
            return (ActionCandidate(
                f"option_{index}_editor", Channel.CLI, self.open_capability,
                f"Open {self.path.name} in {editor}",
                {"path": str(self.path)}, Risk.READ_ONLY,
                verifier="artifact.editor_open", intent=f"option_{index}",
            ),)
        return (ActionCandidate(
            f"option_{index}_artifact", Channel.API, "artifact.write_text",
            f"Create the scoped {self.path.name} artifact",
            {"path": str(self.path), "text": value}, Risk.LOCAL_WRITE,
            verifier="artifact.exact_text", intent=f"option_{index}",
        ),)

    def owns(self, candidate: ActionCandidate) -> bool:
        return candidate.capability in self.supported_capabilities

    def execute(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        if self.open_capability and candidate.capability == self.open_capability:
            def open_editor() -> dict[str, Any]:
                if candidate.arguments["path"] != str(self.path) or not self.path.is_file():
                    raise ValueError("artifact target changed")
                before = set(self._editor_windows()) if self.editor_probe is None else set()
                self.editor_launcher()
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if self.editor_probe is not None:
                        title = self.editor_probe()
                        if title:
                            self._opened_test_editor = True
                            return {"path": str(self.path), "window_title": title}
                    else:
                        windows = self._editor_windows()
                        fresh = [handle for handle in windows if handle not in before]
                        if fresh:
                            self.opened_window_handle = fresh[-1]
                            return {
                                "path": str(self.path),
                                "window_title": windows[self.opened_window_handle],
                            }
                    time.sleep(0.3)
                raise RuntimeError("the selected editor did not display the artifact")

            return execute_with_receipt(candidate, observation_id, decision_id, open_editor)

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
        if self.open_capability:
            registry.register("artifact.editor_open", self._verify_editor_open)

    def _verify_editor_open(
        self, candidate: ActionCandidate, receipt: ActionReceipt,
    ) -> Verification:
        state = self.capture()
        passed = (
            receipt.success and state.exists and state.editor_open
            and receipt.output.get("path") == str(self.path)
        )
        return Verification(bool(passed), "artifact.editor_open", {
            "editor_open": state.editor_open,
        })

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
