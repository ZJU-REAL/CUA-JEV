from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from ..models import ActionCandidate, ActionReceipt
from .common import execute_with_receipt


class FileSystemExecutor:
    """Small typed filesystem surface; no arbitrary shell expansion."""

    def __call__(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        def operation() -> dict:
            args = candidate.arguments
            if candidate.capability == "filesystem.list":
                root = Path(args["path"])
                limit = max(1, min(int(args.get("max_entries", 10_000)), 10_000))
                paths = sorted(root.iterdir(), key=lambda p: p.name.lower())
                entries = [
                    {
                        "name": path.name,
                        "is_dir": path.is_dir(),
                        "size": path.stat().st_size if path.is_file() else None,
                    }
                    for path in paths[:limit]
                ]
                return {"path": str(root.resolve()), "entries": entries,
                        "truncated": len(paths) > limit}
            if candidate.capability == "filesystem.read_text":
                path = Path(args["path"])
                if "scope_root" in args:
                    root = Path(args["scope_root"]).resolve(strict=True)
                    if path.is_symlink() or path.resolve(strict=True).parent != root:
                        raise ValueError("scoped read target moved outside its offered directory")
                    if path.stat().st_size > 16_000:
                        raise ValueError("scoped read target grew beyond the offer limit")
                limit = max(1, min(int(args.get("max_chars", 100_000)), 100_000))
                with path.open("r", encoding=args.get("encoding", "utf-8")) as handle:
                    text = handle.read(limit + 1)
                return {"path": str(path.resolve()), "text": text[:limit], "truncated": len(text) > limit}
            if candidate.capability == "filesystem.stat":
                path = Path(args["path"])
                stat = path.stat()
                digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                return {
                    "path": str(path.resolve()),
                    "is_file": path.is_file(),
                    "size": stat.st_size,
                    "sha256": digest,
                }
            if candidate.capability == "filesystem.copy":
                source = Path(args["source_path"])
                destination = Path(args["destination_path"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                return {"source": str(source.resolve()), "destination": str(destination.resolve())}
            if candidate.capability == "filesystem.write_text":
                path = Path(args["path"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(args["text"]), encoding=args.get("encoding", "utf-8"))
                return {"path": str(path.resolve()), "characters": len(str(args["text"]))}
            raise ValueError(f"unsupported filesystem capability: {candidate.capability}")

        return execute_with_receipt(candidate, observation_id, decision_id, operation)
