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
from .frozen import builtin_task_specs
from .guard import ActionGuard
from .models import Channel
from .open_browser import HttpJsonPlanner, OpenBrowserTask, PublicDecisionPolicy
from .policy import JevPolicy, RulePolicy
from .registry import ExecutorRegistry
from .runtime import AgentRuntime
from .sandbox import FileOrganizationTask, sandbox_mcp_executor
from .suites import SUITE_NAMES, make_suite
from .trace import JsonlTrace
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
    sub.add_parser("tasks", help="List frozen representative task contracts")
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
    open_browser.add_argument("--planner-endpoint", required=True)
    open_browser.add_argument("--policy", choices=("rule", "jev"), default="jev")
    open_browser.add_argument("--headed", action="store_true")
    open_browser.add_argument("--allow-form-input", action="store_true")
    open_browser.add_argument("--allow-external-actions", action="store_true")
    open_browser.add_argument("--max-steps", type=int, default=20)
    open_browser.add_argument("--trace", help="Opt-in local trace; may contain task data")
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
                allow_writes=args.allow_form_input,
                allow_destructive=args.allow_external_actions,
            ),
            executors=executors,
            verifiers=verifiers,
            trace=JsonlTrace(args.trace),
        )

    return EpisodeRunner(runtime_factory, EpisodeConfig(max_steps=args.max_steps, timeout_s=600))


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2, ensure_ascii=False))
        return 0
    if args.command == "tasks":
        print(
            json.dumps(
                [
                    {
                        "name": task.name,
                        "application": task.application,
                        "capability_packs": task.capability_packs,
                        "max_steps": task.max_steps,
                        "digest": task.digest,
                    }
                    for task in builtin_task_specs()
                ],
                indent=2,
            )
        )
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
    if args.command == "open-browser":
        if args.max_steps < 1:
            raise SystemExit("--max-steps must be positive")
        planner = HttpJsonPlanner(
            args.planner_endpoint, api_key=os.getenv("CUA_JEV_PLANNER_API_KEY")
        )
        try:
            environment = OpenBrowserTask(
                goal=args.goal,
                start_url=args.url,
                planner=planner,
                headed=args.headed,
                allow_form_input=args.allow_form_input,
                allow_external_actions=args.allow_external_actions,
            )
            result = _open_browser_runner(args).run(environment)
        finally:
            planner.close()
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
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
