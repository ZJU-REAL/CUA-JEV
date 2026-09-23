"""Opt-in, read-only local action offers for open-goal browser episodes.

The planner sees opaque references and descriptions, never an executable argv or
model-chosen path. Every offer is re-created from the scoped filesystem state.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import ActionCandidate, Channel, Risk


@dataclass(frozen=True)
class ToolOffer:
    ref: str
    capability: str
    channel: Channel
    description: str
    path: str = ""

    def to_dict(self) -> dict:
        return {
            "ref": self.ref, "capability": self.capability,
            "channel": str(self.channel), "description": self.description,
        }

    def candidate(self, index: int) -> ActionCandidate:
        arguments = {"path": self.path} if self.path else {}
        if self.capability == "filesystem.list":
            arguments["max_entries"] = 100
        if self.capability == "filesystem.read_text":
            arguments["max_chars"] = 16_000
            arguments["scope_root"] = str(Path(self.path).parent)
        return ActionCandidate(
            id=f"option_{index}_tool", channel=self.channel,
            capability=self.capability, description=self.description,
            arguments=arguments, risk=Risk.READ_ONLY,
            verifier="tool.result", intent=f"option_{index}",
        )


class ScopedReadOnlyTools:
    """Enumerate bounded local offers under an explicitly authorized directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("local tool scope must be a directory")

    def offers(self) -> tuple[ToolOffer, ...]:
        offers = [
            ToolOffer("t0", "filesystem.list", Channel.API, "List the scoped directory", str(self.root)),
            ToolOffer("t1", "cli.python_version", Channel.CLI, "Read the installed Python version"),
        ]
        if (self.root / ".git").exists():
            offers.append(ToolOffer(
                f"t{len(offers)}", "cli.git_status", Channel.CLI,
                "Read Git working-tree status", str(self.root),
            ))
        for path in sorted(self.root.iterdir(), key=lambda item: item.name.casefold()):
            if len(offers) >= 16:
                break
            if (
                path.is_symlink() or not path.is_file() or path.name.startswith(".")
                or path.suffix.lower() not in {".md", ".txt"}
                or path.stat().st_size > 16_000
            ):
                continue
            offers.append(ToolOffer(
                f"t{len(offers)}", "filesystem.read_text", Channel.API,
                f"Read scoped text file {path.name}", str(path.resolve()),
            ))
        return tuple(offers)
