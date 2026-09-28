"""Real local integration smoke, with an explicitly deterministic fixture planner.

This is not a model/Jev benchmark. It launches bundled Chromium on intercepted
local fixture pages, uses the current Python CLI, and optionally controls only
CUA-JEV's own native test window. All traces and recordings remain local.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from cua_jev.cli import _open_computer_runner, _parser
from cua_jev.executors import ControlExecutor
from cua_jev.guard import ActionGuard
from cua_jev.macos_bridge import MacBridge, native_binary
from cua_jev.macos_surface import MacAccessibilitySurface
from cua_jev.models import Channel
from cua_jev.open_browser import OpenBrowserTask, PublicDecisionPolicy
from cua_jev.open_computer import ComputerPlan, OpenComputerTask
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.trace import JsonlTrace
from cua_jev.verify import VerifierRegistry


class FixturePlanner:
    """Test-only state machine; refs always come from live observations."""

    def __init__(self, text):
        self.text = text

    def plan(self, goal, state, recent):
        if "cli.python_version" not in state.requirements["completed_capabilities"]:
            ref = next(item.ref for item in state.tool_offers if item.capability == "cli.python_version")
            option = {"ref": f"t:{ref}", "operation": "invoke"}
        elif "/done" not in state.browser.url:
            ref = next(item.ref for item in state.browser.elements if item.label == "Finish")
            option = {"ref": f"b:{ref}", "operation": "click"}
        else:
            controls = state.desktop.native["controls"]
            edit = next(item for item in controls if item["name"] == "Note text")
            if edit["private_value"] != self.text:
                option = {"ref": f"d:{edit['ref']}", "operation": "fill", "value": self.text}
            else:
                button = next(item for item in controls if item["name"] == "Save note")
                option = {"ref": f"d:{button['ref']}", "operation": "click"}
        return ComputerPlan.from_dict(
            {"subgoal": "Complete local integration fixture", "options": [option]}, state
        )


class RouteTestPolicy(RulePolicy):
    def __init__(self, route):
        self.route = route

    def choose(self, observation, candidates):
        if any(candidate.capability.startswith("desktop.") for candidate in candidates):
            channel = Channel.GUI if self.route == "gui" else Channel.API
            candidates = [candidate for candidate in candidates if candidate.channel == channel]
        return super().choose(observation, candidates)


def run(output: Path, route: str, *, desktop: bool, record: bool) -> dict:
    from playwright.sync_api import sync_playwright

    token = uuid.uuid4().hex[:8]
    text = f"Mac smoke 中文 {token}"
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / f"{route}-{token}.jsonl"
    record_path = output / f"{route}-{token}.mp4" if record else None
    planner = FixturePlanner(text)
    goal = "Check Python, open the fixture result, and save the test note" if desktop else (
        "Check Python and open the local fixture result"
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page()

        def fixture(request):
            body = "<title>Fixture done</title><main>Result verified</main>" if (
                request.request.url.endswith("/done")
            ) else '<title>Fixture start</title><main><a href="/done">Finish</a></main>'
            request.fulfill(status=200, content_type="text/html", body=body)

        page.route("https://cua-jev.test/**", fixture)
        surface = MacAccessibilitySurface(
            window_title_re="^CUA-JEV Local Fixture$", allow_text_input=True, allow_button_actions=True,
            record_path=record_path,
        ) if desktop else None
        task = OpenComputerTask(
            goal=goal, planner=planner,
            browser=OpenBrowserTask(goal=goal, start_url="https://cua-jev.test/start", page=page,
                                    planner=planner, browser_channel="chromium"),
            desktop_surface=surface, local_tool_root=output,
            required_url_contains="/done", required_window_text_contains=f"Saved: {text}" if desktop else "",
            required_capabilities=("cli.python_version", "desktop.fill", "desktop.click") if desktop
            else ("cli.python_version",),
        )
        args = _parser().parse_args([
            "open-computer", "--goal", goal, "--url", "https://cua-jev.test/start",
            "--model-base-url", "https://unused.test/v1", "--model", "fixture",
            "--policy", "rule", "--trace", str(trace_path), "--max-steps", "8",
        ])
        runner = _open_computer_runner(args)

        def factory(environment):
            executors = ExecutorRegistry()
            for channel, executor in environment.executor_bindings().items():
                executors.register(channel, executor)
            executors.register(Channel.CONTROL, ControlExecutor())
            verifiers = VerifierRegistry()
            environment.register_verifiers(verifiers)
            return AgentRuntime(
                policy=PublicDecisionPolicy(RouteTestPolicy(route)),
                guard=ActionGuard(allowed_roots=[output], allow_writes=True, allow_destructive=True),
                executors=executors, verifiers=verifiers, trace=JsonlTrace(trace_path),
            )

        runner.runtime_factory = factory
        result = runner.run(task)
        browser.close()
    return {
        **result.to_dict(), "planner": "deterministic integration fixture", "policy": "rule",
        "desktop_route": route if desktop else None, "trace": str(trace_path),
        "recording": str(record_path) if record_path else None,
        "all_actions_verified": all(step.verification.passed for step in result.steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--route", choices=("ax", "gui", "both"), default="both")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/macos-smoke"))
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("this live smoke is for macOS")
    if args.record and args.browser_only:
        parser.error("--record needs the selected native window")
    fixture = None
    results = []
    try:
        if not args.browser_only:
            probe = MacBridge()
            try:
                health = probe.call("health")
            finally:
                probe.close()
            if not health["accessibility"]:
                raise RuntimeError("macOS Accessibility permission is required; run cua-jev macos-doctor")
            if args.record and not health["screen_recording"]:
                raise RuntimeError("macOS Screen Recording permission is required")
            fixture = subprocess.Popen([str(native_binary()), "--fixture"], stderr=subprocess.DEVNULL)
            probe = MacBridge()
            try:
                for attempt in range(30):
                    try:
                        probe.call("select", title_pattern="^CUA-JEV Local Fixture$")
                        break
                    except RuntimeError:
                        if fixture.poll() is not None or attempt == 29:
                            raise
                        time.sleep(0.1)
            finally:
                probe.close()
        routes = ("ax", "gui") if args.route == "both" and not args.browser_only else (args.route,)
        for route in routes:
            result = run(args.output.resolve(), route, desktop=not args.browser_only, record=args.record)
            results.append(result)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result["status"] != "success":
                break
        return 0 if all(item["status"] == "success" for item in results) else 1
    finally:
        if fixture is not None:
            fixture.terminate()
            try:
                fixture.wait(timeout=5)
            except subprocess.TimeoutExpired:
                fixture.kill()
                fixture.wait()


if __name__ == "__main__":
    raise SystemExit(main())
