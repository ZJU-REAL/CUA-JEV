"""Run the bounded Python-docs model + Jev showcase in fresh Mac windows.

Credentials come from the ignored local .env. The model discovers chapter links;
this launcher supplies topic and terminal constraints, never an action sequence.
Raw evidence stays private until reviewed and explicitly rendered/published.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from cua_jev.config import load_local_env
from cua_jev.macos_bridge import MacBridge, native_binary

# Works both as `python scripts/...` and as an imported test module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.record_macos_computer import _close_all  # noqa: E402
from scripts.record_macos_computer import main as record_main  # noqa: E402

TOPICS = (
    "More Control Flow Tools", "Data Structures", "Modules", "Input and Output",
    "Errors and Exceptions", "10. Brief Tour of the Standard Library",
    "Virtual Environments and Packages",
)


def prepare_task(root: Path, output: Path, model_base_url: str, model: str,
                 *, max_steps: int = 32) -> list[str]:
    """Create a trusted reader profile and an empty, per-take file scope."""
    root.mkdir(parents=True, exist_ok=False)
    work = root / "workspace"
    work.mkdir()
    profile = root / "reader.json"
    profile.write_text(json.dumps({
        "server": {
            "name": "research", "command": str(Path(sys.executable).absolute()),
            "args": ["-m", "cua_jev.mcp_web_reader", "--origin", "https://docs.python.org"],
        },
        "tools": [{
            "name": "fetch_page",
            "description": "Read the current official Python documentation page as a bounded excerpt.",
            "arguments": {}, "parameters": {"href": {"type": "string", "maxLength": 500}},
            "required": ["href"], "expected_from_arguments": {"url": "href"},
        }],
    }, indent=2), encoding="utf-8")
    name = f"mac-python-handoff-{uuid4().hex[:8]}.txt"
    goal = (
        "Prepare a concise English Python automation onboarding guide for a teammate on this Mac. "
        "Discover seven distinct official tutorial chapter pages through currently observed browser "
        "links, covering control flow, data structures, modules, input and output, errors and "
        "exceptions, the first standard library tour, and virtual environments. Read each discovered "
        "page with the current-page MCP reader before moving on. Prefer a chapter overview URL "
        "without a fragment when offered. Use only observed excerpts and verified tool results. "
        "Include one practical takeaway and each full source URL for each topic; keep the whole "
        "guide under 3500 characters. Do not invent quotes or facts from unseen sections. "
        "After source research, query the installed Python version using its registered CLI tool "
        "and include the exact observed version near the beginning of the guide. "
        f"Write the guide as {name}. In the selected native fixture, fill Note text with exactly "
        f"{name}, click Save note, and check Reviewed. Finish by opening the completed file in "
        "TextEdit after native confirmation. Aim for about twenty useful actions, avoiding "
        "unnecessary navigation or tool calls; meeting all evidence gates matters more than count."
    )
    args = [
        "--output", str(output), "--goal", goal,
        "--url", "https://docs.python.org/3/tutorial/index.html",
        "--model-base-url", model_base_url, "--model", model, "--policy", "jev",
        "--window-title", "^CUA-JEV Local Fixture$",
        "--allow-window-text-input", "--allow-window-actions",
        "--require-window-text-contains", f"Saved: {name}\nReviewed: yes",
        "--local-tool-root", str(work), "--artifact-path", str(work / name),
        "--open-artifact-textedit", "--require-capability", "artifact.write_text",
        "--mcp-profile", str(profile), "--allow-mcp-actions", "--mcp-current-page-only",
        "--min-verified-sources", "7", "--max-steps", str(max_steps), "--final-hold", "5",
    ]
    for topic in TOPICS:
        args.extend(("--require-source-title-contains", topic))
    (root / "task.json").write_text(json.dumps({"goal": goal, "args": args}, indent=2),
                                   encoding="utf-8")
    return args



def _stop_owned(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new private recording-bundle folder")
    parser.add_argument("--model-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", type=int, default=32)
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        parser.error("this showcase requires macOS")
    if args.output.exists() or args.output.is_symlink() or not 20 <= args.max_steps <= 40:
        parser.error("use a new output directory and a step budget between 20 and 40")
    load_local_env()
    root = Path("runs/mac-python-demo") / uuid4().hex
    task_args = prepare_task(root.resolve(), args.output.resolve(), args.model_base_url,
                             args.model, max_steps=args.max_steps)
    fixture = awake = None
    probe = MacBridge()
    try:
        if probe.call("health").get("screen_locked") is True:
            raise RuntimeError("Unlock this Mac before starting the native demo")
        # Temporary assertions end with this run; they cannot unlock the session
        # or prevent an explicit user lock, and do not change system settings.
        awake = subprocess.Popen(["/usr/bin/caffeinate", "-di", "-w", str(os.getpid())],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        fixture = subprocess.Popen([str(native_binary()), "--fixture"], stderr=subprocess.DEVNULL)
        for attempt in range(30):
            try:
                probe.call("select", title_pattern="^CUA-JEV Local Fixture$")
                break
            except RuntimeError:
                if fixture.poll() is not None or attempt == 29:
                    raise
                time.sleep(0.1)
        probe.close()
        return record_main(task_args)
    finally:
        _close_all((probe.close, lambda: _stop_owned(fixture), lambda: _stop_owned(awake)))


if __name__ == "__main__":
    raise SystemExit(main())
