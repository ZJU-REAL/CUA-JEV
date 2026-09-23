from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from ..models import ActionCandidate, ActionReceipt
from .common import execute_with_receipt

CommandFactory = Callable[[dict], Sequence[str]]


class RegisteredCliExecutor:
    """Executes argv templates only. It never invokes a shell or accepts raw commands."""

    def __init__(self, timeout_s: float = 30) -> None:
        self.timeout_s = timeout_s
        self._commands: dict[str, CommandFactory] = {
            "cli.python_version": lambda _: ["python", "--version"],
            "cli.git_status": lambda args: [
                "git", "--no-optional-locks", "-c", "core.fsmonitor=false",
                "-C", str(Path(args["path"])), "status", "--short",
            ],
            "cli.git_diff_stat": lambda args: ["git", "-C", str(Path(args["path"])), "diff", "--stat"],
            "cli.powershell_read_text": lambda args: [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-Content -LiteralPath $args[0] -Raw",
                str(Path(args["path"])),
            ],
        }

    def register(self, capability: str, factory: CommandFactory) -> None:
        self._commands[capability] = factory

    def __call__(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        def operation() -> dict:
            factory = self._commands.get(candidate.capability)
            if factory is None:
                raise ValueError(f"unregistered CLI capability: {candidate.capability}")
            argv = list(factory(candidate.arguments))
            completed = subprocess.run(
                argv,
                cwd=candidate.arguments.get("cwd"),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                shell=False,
                check=False,
            )
            if completed.returncode:
                raise RuntimeError(f"command exited {completed.returncode}: {completed.stderr[-1000:]}")
            return {
                "argv": argv,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-20_000:],
                "stderr": completed.stderr[-5_000:],
            }

        return execute_with_receipt(candidate, observation_id, decision_id, operation)
