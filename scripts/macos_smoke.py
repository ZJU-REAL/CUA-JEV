"""Local Mac integration: deterministic executor checks or real model planning.

By default this uses a keyless fixture planner and forced Rule routes. Supply
both --model-base-url and --model for actual grounded model proposals, then
select --policy rule or --policy jev. Model mode leaves all legal routes open.
Only intercepted browser pages, a fresh local tool directory, and CUA-JEV's own
native test window are used. This short integration is not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from cua_jev.config import load_local_env
from cua_jev.episode import EpisodeConfig, EpisodeRunner
from cua_jev.errors import CapabilityUnavailable
from cua_jev.executors import ControlExecutor
from cua_jev.guard import ActionGuard
from cua_jev.macos_bridge import MacBridge, native_binary
from cua_jev.macos_surface import MacAccessibilitySurface
from cua_jev.model_planner import ChatModelPlanner
from cua_jev.models import Channel
from cua_jev.open_browser import OpenBrowserTask, PublicDecisionPolicy
from cua_jev.open_computer import ComputerPlan, OpenComputerTask
from cua_jev.policy import JevPolicy, RulePolicy
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
            if not candidates:
                raise CapabilityUnavailable(
                    f"Forced {self.route.upper()} test route is unavailable in this observation; "
                    "no action executed and no fallback route selected"
                )
        return super().choose(observation, candidates)


def _validate_mode(model_base_url: str | None, model: str | None, policy: str, route: str) -> bool:
    if bool(model_base_url) != bool(model):
        raise ValueError("--model-base-url and --model must be supplied together")
    if policy not in {"rule", "jev"}:
        raise ValueError("--policy must be rule or jev")
    model_enabled = bool(model_base_url and model)
    if model_enabled and route != "auto":
        raise ValueError("model mode requires --route auto; AX/GUI forcing is for deterministic checks")
    if not model_enabled and policy != "rule":
        raise ValueError("--policy jev requires --model-base-url and --model")
    if route not in {"ax", "gui", "both", "auto"}:
        raise ValueError("unknown desktop route")
    return model_enabled


def _make_task(page, planner, workspace: Path, text: str, surface=None) -> OpenComputerTask:
    goal = (
        "Read the installed Python version using its registered tool "
        "and open the local browser result page."
    )
    if surface is not None:
        goal += f" Also save exactly {json.dumps(text, ensure_ascii=False)} in the selected native fixture."
    return OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url="https://cua-jev.test/start", page=page,
            planner=planner, browser_channel="chromium",
        ),
        desktop_surface=surface, local_tool_root=workspace,
        required_url_contains="/done",
        required_window_text_contains=f"Saved: {text}" if surface is not None else "",
        required_capabilities=("cli.python_version", "desktop.fill", "desktop.click") if surface is not None
        else ("cli.python_version",),
    )


def _execute_task(task, planner, trace_path: Path, *, route: str, policy: str,
                  model_enabled: bool, max_steps: int, timeout_s: float) -> dict:
    trace = JsonlTrace(trace_path)

    def factory(environment):
        executors = ExecutorRegistry()
        for channel, executor in environment.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        environment.register_verifiers(verifiers)
        inner = (JevPolicy(retries=2, fallback_on_transport=False) if policy == "jev" else
                 RouteTestPolicy(route) if not model_enabled and route in {"ax", "gui"} else RulePolicy())
        return AgentRuntime(
            policy=PublicDecisionPolicy(inner),
            guard=ActionGuard(
                allowed_roots=[task.local_tools.root], allow_writes=True, allow_destructive=True,
            ),
            executors=executors, verifiers=verifiers, trace=trace,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=max_steps, timeout_s=timeout_s)).run(task)
    terminal = [event["payload"] for event in trace.events if event["kind"] == "evaluation"]
    decisions = [event["payload"] for event in trace.events if event["kind"] == "decision"]
    policy_usage = Counter()
    for event in trace.events:
        if event["kind"] == "policy_exchange" and isinstance(event["payload"].get("usage"), dict):
            policy_usage.update({
                name: value for name, value in event["payload"]["usage"].items()
                if isinstance(name, str) and type(value) is int and value >= 0
            })
    routes = Counter(
        step.receipt.output["route"] for step in result.steps
        if step.receipt.success and step.receipt.capability.startswith("desktop.")
        and step.receipt.output.get("route") in {"ax", "gui"}
    )
    return {
        **result.to_dict(),
        "mode": "model" if model_enabled else "deterministic",
        "planner": getattr(planner, "model", "deterministic integration fixture"),
        "policy": policy,
        "forced_desktop_route": route if (
            "d" in task.providers and not model_enabled and route in {"ax", "gui"}
        ) else None,
        "actual_desktop_routes": dict(routes),
        "planner_calls": task.planner_calls,
        "planner_requests": getattr(planner, "planning_requests", 0),
        "planner_wall_ms": getattr(planner, "planning_wall_ms", 0.0),
        "planner_model_usage": getattr(planner, "usage_totals", None),
        "policy_decisions": len(decisions),
        "jev_decisions": sum(str(item.get("model", "")).startswith("jev-") for item in decisions),
        "policy_model_usage": dict(policy_usage) if policy_usage else None,
        "terminal_verified": bool(terminal and terminal[-1]["success"] and terminal[-1]["terminal"]),
        "all_actions_verified": bool(result.steps) and all(step.verification.passed for step in result.steps),
        "trace": str(trace_path),
    }


def run(output: Path, route: str, *, desktop: bool, record: bool,
        model_base_url: str | None = None, model: str | None = None, policy: str = "rule",
        max_steps: int = 8, timeout_s: float = 300, use_env_proxy: bool = False) -> dict:
    model_enabled = _validate_mode(model_base_url, model, policy, route)
    EpisodeConfig(max_steps=max_steps, timeout_s=timeout_s)
    if record and not desktop:
        raise ValueError("recording needs the selected native window")
    from playwright.sync_api import sync_playwright

    token = uuid.uuid4().hex[:8]
    text = f"Mac smoke 中文 {token}"
    output.mkdir(parents=True, exist_ok=True)
    prefix = f"{'model' if model_enabled else 'fixture'}-{route}-{token}"
    trace_path = output / f"{prefix}.jsonl"
    record_path = output / f"{prefix}.mp4" if record else None
    # Give model-selected tools a new empty directory, never earlier run output.
    workspace = output / f"{prefix}-workspace"
    workspace.mkdir()
    with ExitStack() as resources:
        planner = ChatModelPlanner(
            model_base_url, model,
            api_key=os.getenv("CUA_JEV_MODEL_API_KEY") or os.getenv("CUA_JEV_PLANNER_API_KEY"),
            use_env_proxy=use_env_proxy,
        ) if model_enabled else FixturePlanner(text)
        if model_enabled:
            resources.callback(planner.close)
        playwright = resources.enter_context(sync_playwright())
        browser = playwright.chromium.launch(headless=True)
        resources.callback(browser.close)
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
        task = _make_task(page, planner, workspace, text, surface)
        summary = _execute_task(
            task, planner, trace_path, route=route, policy=policy, model_enabled=model_enabled,
            max_steps=max_steps, timeout_s=timeout_s,
        )
        return {
            **summary, "desktop_route": route if desktop else None,
            "recording": str(record_path) if record_path else None,
            "recording_scope": "selected native fixture window only" if record else None,
        }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-only", action="store_true")
    parser.add_argument("--route", choices=("ax", "gui", "both", "auto"),
                        help="deterministic default: both; model default: auto (no route forcing)")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/macos-smoke"))
    parser.add_argument("--model-base-url", help="explicit OpenAI-compatible text-model base URL")
    parser.add_argument("--model", help="explicit text-model ID; omit both model flags for keyless checks")
    parser.add_argument("--policy", choices=("rule", "jev"), default="rule")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--timeout-s", type=float, default=300)
    parser.add_argument("--use-env-proxy", action="store_true")
    args = parser.parse_args(argv)
    if args.route is None:
        args.route = "auto" if args.model_base_url or args.browser_only else "both"
    try:
        _validate_mode(args.model_base_url, args.model, args.policy, args.route)
        EpisodeConfig(max_steps=args.max_steps, timeout_s=args.timeout_s)
    except ValueError as exc:
        parser.error(str(exc))
    if args.record and args.browser_only:
        parser.error("--record needs the selected native window")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    if sys.platform != "darwin":
        raise SystemExit("this live smoke is for macOS")
    load_local_env()
    fixture = None
    results = []
    try:
        if not args.browser_only:
            probe = MacBridge()
            try:
                health = probe.call("health")
            finally:
                probe.close()
            if health.get("screen_locked") is True:
                raise RuntimeError("macOS screen is locked; unlock the current session before the smoke test")
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
            result = run(
                args.output.resolve(), route, desktop=not args.browser_only, record=args.record,
                model_base_url=args.model_base_url, model=args.model, policy=args.policy,
                max_steps=args.max_steps, timeout_s=args.timeout_s, use_env_proxy=args.use_env_proxy,
            )
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
