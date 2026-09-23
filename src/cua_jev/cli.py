from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from .config import load_local_env
from .demo import filesystem_routing_demo
from .doctor import doctor
from .episode import EpisodeConfig, EpisodeRunner
from .executors import ControlExecutor, FileSystemExecutor
from .experiment import ExperimentRunner
from .guard import ActionGuard
from .model_planner import ChatModelPlanner
from .models import Channel
from .open_browser import HttpJsonPlanner, OpenBrowserTask, PublicDecisionPolicy
from .open_computer import OpenComputerTask
from .open_desktop import OpenDesktopTask
from .open_workspace import OpenWorkspaceTask
from .policy import JevPolicy, RulePolicy
from .registry import ExecutorRegistry
from .runtime import AgentRuntime
from .sandbox import FileOrganizationTask, sandbox_mcp_executor
from .suites import SUITE_NAMES, make_suite
from .trace import JsonlTrace
from .trace_analysis import analyze_traces
from .verify import VerifierRegistry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cua-jev", description="Typed hybrid action routing with Jev")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report optional Windows capability availability")
    demo = sub.add_parser("demo", help="Run the reproducible multi-channel routing demo")
    demo.add_argument("--policy", choices=("rule", "jev"), default="rule")
    demo.add_argument("--workspace", default="demo-workspace")
    demo.add_argument("--trace", default="runs/demo.jsonl")
    benchmark = sub.add_parser("benchmark", help="Repeat the frozen routing task and summarize outcomes")
    benchmark.add_argument("--policy", choices=("rule", "jev"), default="rule")
    benchmark.add_argument("--episodes", type=int, default=10)
    benchmark.add_argument("--workspace", default="demo-workspace")
    benchmark.add_argument("--trace", default="runs/benchmark.jsonl")
    sub.add_parser("jev-smoke", help="Make one read-only Jev decision without executing an action")
    sub.add_parser("tasks", help="List the four current workflow cases")
    analyze = sub.add_parser(
        "analyze-traces", help="Aggregate local run outcomes and Jev route probabilities"
    )
    analyze.add_argument("paths", nargs="+", help="One or more local JSONL trace files")
    episode = sub.add_parser("episode-demo", help="Run the closed-loop sandbox episode")
    episode.add_argument("--policy", choices=("rule", "jev"), default="rule")
    episode.add_argument("--workspace", default="demo-workspace/episode")
    episode.add_argument("--trace", default="runs/episode.jsonl")
    experiment = sub.add_parser("experiment", help="Run repeated closed-loop sandbox episodes")
    experiment.add_argument("--policy", choices=("rule", "jev"), default="rule")
    experiment.add_argument("--episodes", type=int, default=10)
    experiment.add_argument("--workspace", default="demo-workspace/experiment")
    experiment.add_argument("--output", default="runs/experiment-summary.json")
    experiment.add_argument("--trace", default="runs/experiment.jsonl")
    suite = sub.add_parser("suite", help="Run complete representative application tasks")
    suite.add_argument("--task", choices=(*SUITE_NAMES, "all"), default="all")
    suite.add_argument("--policy", choices=("rule", "jev"), default="rule")
    suite.add_argument("--episodes", type=int, default=1)
    suite.add_argument("--workspace", default="demo-workspace/suites")
    suite.add_argument("--output", default="runs/suite-summary.json")
    suite.add_argument("--trace", default="runs/suite.jsonl")
    suite.add_argument("--headed-edge", action="store_true")
    suite.add_argument("--open-vscode", action="store_true")
    suite.add_argument("--visible-apps", action="store_true")
    suite.add_argument("--profile", choices=("hybrid", "visible", "adaptive"), default="hybrid")
    suite.add_argument(
        "--policy-fallback",
        action="store_true",
        help="fall back to the deterministic rule policy after transient Jev transport failures",
    )
    open_browser = sub.add_parser(
        "open-browser", help="Experimental open-goal browser loop with a pluggable JSON planner"
    )
    open_browser.add_argument("--goal", required=True)
    open_browser.add_argument("--url", required=True)
    browser_planner = open_browser.add_mutually_exclusive_group(required=True)
    browser_planner.add_argument("--planner-endpoint")
    browser_planner.add_argument("--model-base-url")
    open_browser.add_argument("--model")
    open_browser.add_argument("--allow-insecure-model-http", action="store_true")
    open_browser.add_argument("--use-env-proxy", action="store_true")
    open_browser.add_argument("--policy", choices=("rule", "jev"), default="jev")
    open_browser.add_argument("--headed", action="store_true")
    open_browser.add_argument(
        "--browser-channel", choices=("msedge", "chrome", "chromium"), default="msedge"
    )
    open_browser.add_argument("--allow-form-input", action="store_true")
    open_browser.add_argument("--allow-external-actions", action="store_true")
    open_browser.add_argument("--vision-model", help="Optional VLM for viewport scene grounding")
    open_browser.add_argument("--vision-mode", choices=("fallback", "always"), default="fallback")
    open_browser.add_argument("--allow-screenshot-upload", action="store_true")
    open_browser.add_argument("--allow-visual-clicks", action="store_true")
    open_browser.add_argument(
        "--local-tool-root", type=Path,
        help="Explicitly permit bounded read-only file and registered CLI offers in this directory",
    )
    open_browser.add_argument(
        "--require-tool", default="",
        help="Acceptance gate: a registered local capability must run and verify",
    )
    open_browser.add_argument(
        "--require-url-contains", default="",
        help="Independent URL acceptance gate; overrides the planner's browser success check",
    )
    open_browser.add_argument("--max-steps", type=int, default=20)
    open_browser.add_argument("--trace", help="Opt-in local trace; may contain task data")
    open_desktop = sub.add_parser(
        "open-desktop", help="Experimental open-goal loop for one Windows UIA window"
    )
    open_desktop.add_argument("--goal", required=True)
    open_desktop.add_argument("--window-title", required=True, help="Regex matching one visible window")
    open_desktop.add_argument("--model-base-url", required=True)
    open_desktop.add_argument("--model", required=True)
    open_desktop.add_argument("--allow-insecure-model-http", action="store_true")
    open_desktop.add_argument("--use-env-proxy", action="store_true")
    open_desktop.add_argument("--policy", choices=("rule", "jev"), default="jev")
    open_desktop.add_argument("--allow-text-input", action="store_true")
    open_desktop.add_argument("--allow-button-actions", action="store_true")
    open_desktop.add_argument(
        "--vision-model", help="Optional vision-capable model ID for window screenshot grounding"
    )
    open_desktop.add_argument("--vision-mode", choices=("fallback", "always"), default="fallback")
    open_desktop.add_argument(
        "--allow-screenshot-upload", action="store_true",
        help="Explicitly permit sending the selected window image to the VLM",
    )
    open_desktop.add_argument(
        "--allow-visual-clicks", action="store_true",
        help="Offer screenshot-grounded GUI click targets (also requires --allow-button-actions)",
    )
    open_desktop.add_argument("--max-steps", type=int, default=20)
    open_desktop.add_argument("--trace", help="Opt-in local trace; may contain window data")
    computer = sub.add_parser(
        "open-computer", help="Experimental browser + selected Windows window + local-tool loop"
    )
    computer.add_argument("--goal", required=True)
    computer.add_argument("--url", required=True)
    computer.add_argument("--window-title", required=True, help="Regex matching one visible window")
    computer.add_argument("--model-base-url", required=True)
    computer.add_argument("--model", required=True)
    computer.add_argument("--allow-insecure-model-http", action="store_true")
    computer.add_argument("--use-env-proxy", action="store_true")
    computer.add_argument("--policy", choices=("rule", "jev"), default="jev")
    computer.add_argument("--browser-channel", choices=("msedge", "chrome", "chromium"), default="msedge")
    computer.add_argument("--headed-browser", action="store_true")
    computer.add_argument("--allow-form-input", action="store_true")
    computer.add_argument("--allow-browser-actions", action="store_true")
    computer.add_argument("--allow-window-text-input", action="store_true")
    computer.add_argument("--allow-window-actions", action="store_true")
    computer.add_argument("--browser-vision-model")
    computer.add_argument("--desktop-vision-model")
    computer.add_argument("--allow-screenshot-upload", action="store_true")
    computer.add_argument("--allow-browser-visual-clicks", action="store_true")
    computer.add_argument("--allow-desktop-visual-clicks", action="store_true")
    computer.add_argument("--local-tool-root", type=Path)
    computer.add_argument("--require-url-contains", default="")
    computer.add_argument("--require-window-title-contains", default="")
    computer.add_argument("--require-window-text-contains", default="")
    computer.add_argument("--require-capability", action="append", default=[])
    computer.add_argument("--max-steps", type=int, default=12)
    computer.add_argument("--trace", help="Private local trace; may contain browser/window/tool text")
    workspace = sub.add_parser(
        "open-workspace", help="Model + Jev browser-to-VS-Code note task family"
    )
    workspace.add_argument("--goal", required=True)
    workspace.add_argument("--url", required=True)
    workspace.add_argument("--note", required=True, help="New .md or .txt path; never overwrites")
    workspace.add_argument("--model-base-url", required=True)
    workspace.add_argument("--model", required=True)
    workspace.add_argument("--allow-insecure-model-http", action="store_true")
    workspace.add_argument("--use-env-proxy", action="store_true")
    workspace.add_argument("--policy", choices=("rule", "jev"), default="jev")
    workspace.add_argument("--max-steps", type=int, default=12)
    workspace.add_argument(
        "--required-source-text", action="append", default=[],
        help="Optional task acceptance clue that must appear in the live page before writing",
    )
    workspace.add_argument(
        "--research-topic", action="append", default=[],
        help="Repeat for each source-page heading to research and cite in a distinct page",
    )
    workspace.add_argument("--trace", help="Private local trace; includes page and note text")
    catalog = sub.add_parser("planner-models", help="List available IDs from a model gateway")
    catalog.add_argument("--model-base-url", required=True)
    catalog.add_argument("--allow-insecure-model-http", action="store_true")
    catalog.add_argument("--use-env-proxy", action="store_true")
    return parser


def _episode_runner(policy_name: str, workspace: Path, trace: Path) -> EpisodeRunner:
    def runtime_factory() -> AgentRuntime:
        executors = ExecutorRegistry()
        executors.register(Channel.API, FileSystemExecutor())
        executors.register(Channel.MCP, sandbox_mcp_executor())
        executors.register(Channel.CONTROL, ControlExecutor())
        policy = JevPolicy() if policy_name == "jev" else RulePolicy()
        return AgentRuntime(
            policy=policy,
            guard=ActionGuard(allowed_roots=[workspace], allow_writes=True),
            executors=executors,
            trace=JsonlTrace(trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=20))


def _suite_runner(policy_name: str, trace: Path, *, policy_fallback: bool = False) -> EpisodeRunner:
    def runtime_factory(environment) -> AgentRuntime:
        executors = ExecutorRegistry()
        for channel, executor in environment.executor_bindings().items():
            executors.register(channel, executor)
        policy = (
            JevPolicy(retries=2, fallback_on_transport=policy_fallback)
            if policy_name == "jev"
            else RulePolicy()
        )
        return AgentRuntime(
            policy=policy,
            guard=ActionGuard(allowed_roots=environment.allowed_roots, allow_writes=True),
            executors=executors,
            trace=JsonlTrace(trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=20, timeout_s=300))


def _open_browser_runner(args: argparse.Namespace) -> EpisodeRunner:
    def runtime_factory(environment: OpenBrowserTask) -> AgentRuntime:
        executors = ExecutorRegistry()
        for channel, executor in environment.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        environment.register_verifiers(verifiers)
        inner = JevPolicy(retries=2) if args.policy == "jev" else RulePolicy()
        return AgentRuntime(
            policy=PublicDecisionPolicy(inner),
            guard=ActionGuard(
                allowed_roots=[environment.local_tools.root] if environment.local_tools else (),
                allow_writes=args.allow_form_input,
                allow_destructive=args.allow_external_actions,
            ),
            executors=executors,
            verifiers=verifiers,
            trace=JsonlTrace(args.trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=args.max_steps, timeout_s=600))


def _open_desktop_runner(args: argparse.Namespace) -> EpisodeRunner:
    def runtime_factory(environment: OpenDesktopTask) -> AgentRuntime:
        executors = ExecutorRegistry()
        for channel, executor in environment.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        environment.register_verifiers(verifiers)
        inner = JevPolicy(retries=2) if args.policy == "jev" else RulePolicy()
        return AgentRuntime(
            policy=PublicDecisionPolicy(inner),
            guard=ActionGuard(
                allow_writes=args.allow_text_input,
                allow_destructive=args.allow_button_actions,
            ),
            executors=executors, verifiers=verifiers, trace=JsonlTrace(args.trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=args.max_steps, timeout_s=600))


def _open_computer_runner(args: argparse.Namespace) -> EpisodeRunner:
    def runtime_factory(environment: OpenComputerTask) -> AgentRuntime:
        executors = ExecutorRegistry()
        for channel, executor in environment.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        environment.register_verifiers(verifiers)
        inner = JevPolicy(retries=2) if args.policy == "jev" else RulePolicy()
        return AgentRuntime(
            policy=PublicDecisionPolicy(inner),
            guard=ActionGuard(
                allowed_roots=[environment.local_tools.root] if environment.local_tools else (),
                allow_writes=args.allow_form_input or args.allow_window_text_input,
                allow_destructive=args.allow_browser_actions or args.allow_window_actions,
            ),
            executors=executors, verifiers=verifiers, trace=JsonlTrace(args.trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=args.max_steps, timeout_s=900))


def _open_workspace_runner(args: argparse.Namespace) -> EpisodeRunner:
    def runtime_factory(environment: OpenWorkspaceTask) -> AgentRuntime:
        executors = ExecutorRegistry()
        for channel, executor in environment.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        environment.register_verifiers(verifiers)
        inner = JevPolicy(retries=2) if args.policy == "jev" else RulePolicy()
        return AgentRuntime(
            policy=PublicDecisionPolicy(inner),
            guard=ActionGuard(allowed_roots=[environment.note_path.parent], allow_writes=True),
            executors=executors, verifiers=verifiers, trace=JsonlTrace(args.trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=args.max_steps, timeout_s=900))


def _chat_planner(args: argparse.Namespace, *, model: str | None = None) -> ChatModelPlanner:
    model_id = model or getattr(args, "model", None)
    if not model_id:
        raise SystemExit("--model is required with --model-base-url")
    return ChatModelPlanner(
        args.model_base_url, model_id,
        api_key=os.getenv("CUA_JEV_MODEL_API_KEY") or os.getenv("CUA_JEV_PLANNER_API_KEY"),
        allow_insecure_http=args.allow_insecure_model_http,
        use_env_proxy=args.use_env_proxy,
    )


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "tasks":
        from .ui.manager import TASK_CATALOG

        print(
            json.dumps(
                [
                    {
                        "name": name,
                        "title": TASK_CATALOG[name]["title"],
                        "description": TASK_CATALOG[name]["description"],
                        "steps": TASK_CATALOG[name]["steps"],
                    }
                    for name in SUITE_NAMES
                ],
                indent=2,
            )
        )
        return 0
    if args.command == "analyze-traces":
        print(json.dumps(analyze_traces(args.paths), indent=2, ensure_ascii=False))
        return 0
    if args.command == "jev-smoke":
        from .models import ActionCandidate, Channel, Observation

        policy = JevPolicy()
        try:
            decision = policy.choose(
                Observation("Connectivity test", "Select the direct health-check action", {"test": True}),
                [
                    ActionCandidate("health_check", Channel.CONTROL, "control.noop", "Complete health check"),
                    ActionCandidate("reobserve", Channel.CONTROL, "control.reobserve", "Request more state"),
                ],
            )
        finally:
            policy.close()
        print(json.dumps(decision.to_dict(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "planner-models":
        planner = _chat_planner(args, model="catalog-only")
        try:
            print(json.dumps(planner.list_models(), indent=2, ensure_ascii=False))
        finally:
            planner.close()
        return 0
    if args.command == "open-browser":
        if args.max_steps < 1:
            raise SystemExit("--max-steps must be positive")
        if args.vision_model and not (args.allow_screenshot_upload and args.model_base_url):
            raise SystemExit("--vision-model requires --model-base-url and --allow-screenshot-upload")
        if args.allow_visual_clicks and not (args.vision_model and args.allow_external_actions):
            raise SystemExit(
                "--allow-visual-clicks requires --vision-model and --allow-external-actions"
            )
        planner = (
            HttpJsonPlanner(args.planner_endpoint, api_key=os.getenv("CUA_JEV_PLANNER_API_KEY"))
            if args.planner_endpoint else _chat_planner(args)
        )
        vision_planner = _chat_planner(args, model=args.vision_model) if args.vision_model else None
        try:
            environment = OpenBrowserTask(
                goal=args.goal,
                start_url=args.url,
                planner=planner,
                headed=args.headed,
                allow_form_input=args.allow_form_input,
                allow_external_actions=args.allow_external_actions,
                vision_grounder=vision_planner, vision_mode=args.vision_mode,
                allow_screenshot_upload=args.allow_screenshot_upload,
                allow_visual_clicks=args.allow_visual_clicks,
                browser_channel=args.browser_channel,
                local_tool_root=args.local_tool_root,
                required_tool=args.require_tool,
                required_url_contains=args.require_url_contains,
            )
            result = _open_browser_runner(args).run(environment)
        finally:
            planner.close()
            if vision_planner is not None:
                vision_planner.close()
        summary = result.to_dict()
        summary["planner_calls"] = environment.planner_calls
        summary["planner_wall_ms"] = getattr(planner, "planning_wall_ms", None)
        summary["planner_model_usage"] = getattr(planner, "usage_totals", None)
        summary["vision_calls"] = vision_planner.vision_calls if vision_planner else 0
        summary["vision_failures"] = environment.vision_failures
        summary["vision_attempts"] = summary["vision_calls"] + summary["vision_failures"]
        summary["vision_wall_ms"] = vision_planner.vision_wall_ms if vision_planner else 0
        summary["vision_model_usage"] = vision_planner.usage_totals if vision_planner else None
        summary["policy"] = args.policy
        summary["policy_decision_ms"] = sum(step.decision.latency_ms for step in result.steps)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if result.success else 1
    if args.command == "open-desktop":
        if args.max_steps < 1:
            raise SystemExit("--max-steps must be positive")
        if args.vision_model and not args.allow_screenshot_upload:
            raise SystemExit("--vision-model requires --allow-screenshot-upload")
        if args.allow_visual_clicks and not (args.vision_model and args.allow_button_actions):
            raise SystemExit(
                "--allow-visual-clicks requires --vision-model and --allow-button-actions"
            )
        planner = _chat_planner(args)
        vision_planner = _chat_planner(args, model=args.vision_model) if args.vision_model else None
        try:
            environment = OpenDesktopTask(
                goal=args.goal, window_title_re=args.window_title, planner=planner,
                allow_text_input=args.allow_text_input,
                allow_button_actions=args.allow_button_actions,
                vision_grounder=vision_planner, vision_mode=args.vision_mode,
                allow_screenshot_upload=args.allow_screenshot_upload,
                allow_visual_clicks=args.allow_visual_clicks,
            )
            result = _open_desktop_runner(args).run(environment)
        finally:
            planner.close()
            if vision_planner is not None:
                vision_planner.close()
        summary = result.to_dict()
        summary["planner_calls"] = environment.planner_calls
        summary["planner_wall_ms"] = planner.planning_wall_ms
        summary["planner_model_usage"] = planner.usage_totals
        summary["vision_calls"] = vision_planner.vision_calls if vision_planner else 0
        summary["vision_failures"] = environment.vision_failures
        summary["vision_attempts"] = summary["vision_calls"] + summary["vision_failures"]
        summary["vision_wall_ms"] = vision_planner.vision_wall_ms if vision_planner else 0
        summary["vision_model_usage"] = vision_planner.usage_totals if vision_planner else None
        summary["policy"] = args.policy
        summary["policy_decision_ms"] = sum(step.decision.latency_ms for step in result.steps)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if result.success else 1
    if args.command == "open-computer":
        if args.max_steps < 1:
            raise SystemExit("--max-steps must be positive")
        if (args.browser_vision_model or args.desktop_vision_model) and not args.allow_screenshot_upload:
            raise SystemExit("vision models require --allow-screenshot-upload")
        if args.allow_browser_visual_clicks and not (
            args.browser_vision_model and args.allow_browser_actions
        ):
            raise SystemExit("browser visual clicks require a vision model and browser-action opt-in")
        if args.allow_desktop_visual_clicks and not (
            args.desktop_vision_model and args.allow_window_actions
        ):
            raise SystemExit("desktop visual clicks require a vision model and window-action opt-in")
        planner = _chat_planner(args)
        browser_vision = (
            _chat_planner(args, model=args.browser_vision_model)
            if args.browser_vision_model else None
        )
        desktop_vision = (
            _chat_planner(args, model=args.desktop_vision_model)
            if args.desktop_vision_model else None
        )
        try:
            browser = OpenBrowserTask(
                goal=args.goal, start_url=args.url, planner=planner,
                headed=args.headed_browser, browser_channel=args.browser_channel,
                allow_form_input=args.allow_form_input,
                allow_external_actions=args.allow_browser_actions,
                vision_grounder=browser_vision,
                vision_mode="always" if browser_vision else "fallback",
                allow_screenshot_upload=args.allow_screenshot_upload,
                allow_visual_clicks=args.allow_browser_visual_clicks,
            )
            desktop = OpenDesktopTask(
                goal=args.goal, window_title_re=args.window_title, planner=planner,
                allow_text_input=args.allow_window_text_input,
                allow_button_actions=args.allow_window_actions,
                vision_grounder=desktop_vision,
                vision_mode="always" if desktop_vision else "fallback",
                allow_screenshot_upload=args.allow_screenshot_upload,
                allow_visual_clicks=args.allow_desktop_visual_clicks,
            )
            environment = OpenComputerTask(
                goal=args.goal, browser=browser, desktop=desktop, planner=planner,
                local_tool_root=args.local_tool_root,
                required_url_contains=args.require_url_contains,
                required_window_title_contains=args.require_window_title_contains,
                required_window_text_contains=args.require_window_text_contains,
                required_capabilities=args.require_capability,
            )
            result = _open_computer_runner(args).run(environment)
        finally:
            planner.close()
            if browser_vision:
                browser_vision.close()
            if desktop_vision:
                desktop_vision.close()
        summary = result.to_dict()
        summary["planner_calls"] = environment.planner_calls
        summary["planner_wall_ms"] = planner.planning_wall_ms
        summary["planner_model_usage"] = planner.usage_totals
        summary["browser_vision_failures"] = browser.vision_failures
        summary["desktop_vision_failures"] = desktop.vision_failures
        summary["browser_vision_calls"] = browser_vision.vision_calls if browser_vision else 0
        summary["desktop_vision_calls"] = desktop_vision.vision_calls if desktop_vision else 0
        summary["browser_vision_wall_ms"] = browser_vision.vision_wall_ms if browser_vision else 0
        summary["desktop_vision_wall_ms"] = desktop_vision.vision_wall_ms if desktop_vision else 0
        summary["browser_vision_model_usage"] = (
            browser_vision.usage_totals if browser_vision else None
        )
        summary["desktop_vision_model_usage"] = (
            desktop_vision.usage_totals if desktop_vision else None
        )
        summary["vision_attempts"] = sum(
            item.vision_calls for item in (browser_vision, desktop_vision) if item
        ) + browser.vision_failures + desktop.vision_failures
        summary["policy"] = args.policy
        summary["policy_decision_ms"] = sum(step.decision.latency_ms for step in result.steps)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if result.success else 1
    if args.command == "open-workspace":
        if args.max_steps < 1:
            raise SystemExit("--max-steps must be positive")
        planner = _chat_planner(args)
        try:
            environment = OpenWorkspaceTask(
                  goal=args.goal, start_url=args.url, note_path=Path(args.note), planner=planner,
                  required_source_texts=args.required_source_text,
                  research_topics=args.research_topic,
            )
            result = _open_workspace_runner(args).run(environment)
        finally:
            planner.close()
        summary = result.to_dict()
        summary["planner_calls"] = environment.planner_calls
        summary["planner_wall_ms"] = planner.planning_wall_ms
        summary["planner_model_usage"] = planner.usage_totals
        summary["policy"] = args.policy
        summary["policy_decision_ms"] = sum(step.decision.latency_ms for step in result.steps)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if result.success else 1
    if args.command in {"episode-demo", "experiment"}:
        workspace = Path(args.workspace).resolve()
        runner = _episode_runner(args.policy, workspace, Path(args.trace))
        if args.command == "episode-demo":
            result = runner.run(FileOrganizationTask(workspace))
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            return 0 if result.success else 1
        experiment = ExperimentRunner(runner).run(
            lambda: FileOrganizationTask(workspace), args.episodes, output=args.output
        )
        summary = experiment.summary()
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0 if summary["successes"] == args.episodes else 1
    if args.command == "suite":
        if args.episodes < 1:
            raise SystemExit("--episodes must be positive")
        names = SUITE_NAMES if args.task == "all" else (args.task,)
        workspace = Path(args.workspace).resolve()
        trace_base = Path(args.trace)
        summaries = []
        all_passed = True
        for name in names:
            trace = trace_base.with_name(f"{trace_base.stem}-{name}{trace_base.suffix}")
            runner = _suite_runner(args.policy, trace, policy_fallback=args.policy_fallback)
            result = ExperimentRunner(runner).run(
                lambda suite_name=name: make_suite(
                    suite_name,
                    workspace,
                    headed_edge=args.headed_edge,
                    open_vscode=args.open_vscode,
                    visible_apps=args.visible_apps,
                    profile=args.profile,
                ),
                args.episodes,
            )
            summary = result.summary()
            summaries.append(summary)
            all_passed &= summary["successes"] == args.episodes
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "policy": args.policy,
            "requested_episodes": args.episodes,
            "profile": args.profile,
            "all_passed": all_passed,
            "suites": summaries,
        }
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if all_passed else 1
    policy = JevPolicy() if args.policy == "jev" else RulePolicy()
    try:
        if args.command == "benchmark":
            if args.episodes < 1:
                raise SystemExit("--episodes must be positive")
            choices: Counter[str] = Counter()
            successes = 0
            latencies = []
            started = time.perf_counter()
            for _ in range(args.episodes):
                result = filesystem_routing_demo(policy, Path(args.workspace), Path(args.trace))
                choices[result.decision.candidate_id] += 1
                successes += int(result.verification.passed)
                latencies.append(result.decision.latency_ms)
            summary = {
                "policy": args.policy,
                "episodes": args.episodes,
                "verified": successes,
                "success_rate": successes / args.episodes,
                "choices": dict(choices),
                "mean_decision_latency_ms": sum(latencies) / len(latencies),
                "wall_ms": (time.perf_counter() - started) * 1000,
            }
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            return 0 if successes == args.episodes else 1
        result = filesystem_routing_demo(policy, Path(args.workspace), Path(args.trace))
    finally:
        close = getattr(policy, "close", None)
        if close:
            close()
    print(
        json.dumps(
            {
                "decision": result.decision.to_dict(),
                "receipt": result.receipt.to_dict(),
                "verification": result.verification.to_dict(),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result.verification.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
