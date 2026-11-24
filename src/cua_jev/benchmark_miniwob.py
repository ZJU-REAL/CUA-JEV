"""MiniWoB pilot: upstream seeded tasks/evaluator, the CUA-JEV action backend.

This is a custom DOM+pixel harness, not BrowserGym's complete observation/action
pipeline. Private task state is read only by the evaluator, never the agent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import threading
import time
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent_browser import BrowserAgentBackend
from .agent_planner import MultimodalPlanner
from .agent_runner import AgentConfig, AgentRunner
from .config import load_service_profile
from .model_recording import ModelExchangeBuffer
from .models import Decision
from .policy import JevPolicy
from .trace import JsonlTrace
from .trace_analysis import analyze_traces

LEGACY_MODES = ("vlm", "plan_order", "hybrid")
S1S2_MODES = ("s1s2_rule", "s1s2_jev")
MODES = (*LEGACY_MODES, "adaptive_rule", "adaptive", "subgoal_rule", "subgoal_jev", *S1S2_MODES)
THINKING = {"default": None, "on": True, "off": False}
PILOT_TASKS = (
    "click-button", "click-checkboxes", "click-checkboxes-large", "click-dialog",
    "click-option", "enter-text", "enter-text-2", "choose-list",
)


class PlanOrderPolicy:
    """Ablation: same H6 planning, first legal candidate in compiler order."""

    selection_kind = "plan_order"

    def choose(self, observation, candidates):
        selected = candidates[0]
        return Decision(observation.observation_id, selected.id,
                        {c.id: float(c.id == selected.id) for c in candidates},
                        1.0, "deterministic-plan-order", 0.0)

    def close(self):
        pass


class BenchmarkTerminalFailure(RuntimeError):
    """Stop the runner after the upstream environment has ended unsuccessfully."""


class Evaluator:
    def __init__(self, task, page):
        self.task, self.page = task, page
        self.calls = 0
        self.wall_ms = 0.0
        self.last = None

    def read(self):
        began = time.perf_counter()
        try:
            reward, done, _message, info = self.task.validate(self.page, [])
            if type(reward) not in (int, float) or not math.isfinite(reward):
                raise ValueError("invalid upstream reward")
            self.last = {"reward": float(reward), "done": bool(done),
                         "raw_reward": info.get("RAW_REWARD_GLOBAL"),
                         "reason": info.get("REWARD_REASON")}
            return self.last
        finally:
            self.calls += 1
            self.wall_ms += (time.perf_counter() - began) * 1000

    def accept(self, _snapshot):
        result = self.read()
        if result["done"] and result["reward"] <= 0:
            raise BenchmarkTerminalFailure()
        return result["done"] and result["reward"] > 0


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@contextmanager
def miniwob_site(html_root: Path):
    if not (html_root / "miniwob" / "click-button.html").is_file():
        raise ValueError("expected upstream miniwob/html directory")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(_QuietHandler, directory=str(html_root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/miniwob/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def schedule(tasks, seeds, modes):
    """Rotate mode order within matched task/seed blocks; never select by outcome."""
    for index, (task, seed) in enumerate((task, seed) for task in tasks for seed in seeds):
        offset = index % len(modes)
        for mode in modes[offset:] + modes[:offset]:
            yield task, seed, mode


def aggregate(rows):
    result = {}
    for mode in MODES:
        runs = [r for r in rows if r["mode"] == mode]
        if not runs:
            continue
        durations = [r["episode_wall_ms"] for r in runs]
        ordered = sorted(durations)
        position = (len(ordered) - 1) * 0.9
        low, high = math.floor(position), math.ceil(position)
        successes = [r for r in runs if r["benchmark_success"] is True]
        success_durations = [r["episode_wall_ms"] for r in successes]
        result[mode] = {
            "attempted": len(runs), "valid_evaluations": sum(r["official"] is not None for r in runs),
            "successes": len(successes), "success_rate_all_attempts": len(successes) / len(runs),
            "episode_wall_ms_mean_all": statistics.mean(durations),
            "episode_wall_ms_median_all": statistics.median(durations),
            "episode_wall_ms_p90_all": ordered[low] + (ordered[high] - ordered[low]) * (position - low),
            "episode_wall_ms_mean_success_only": statistics.mean(success_durations) if successes else None,
            "planner_calls": sum(r.get("planner_calls", 0) for r in runs),
            "jev_calls": sum(r.get("jev_calls", 0) for r in runs),
            "vlm_reported_total_tokens": sum(r.get("model_usage", {}).get("total_tokens", 0) for r in runs),
            "runs_missing_vlm_token_usage": sum(r.get("planner_calls", 0) > 0 and
                                               "total_tokens" not in r.get("model_usage", {}) for r in runs),
        }
    return result


def matching_initial_states(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["seed"]), []).append(row)
    def present_and_equal(group, key):
        values = {r.get(key) for r in group}
        return None not in values and len(values) == 1

    return [{"task": task, "seed": seed, "modes_completed": [r["mode"] for r in group],
             "all_initial_observations_match": present_and_equal(group, "initial_observation_sha256"),
             "all_goals_match": present_and_equal(group, "goal_sha256")}
            for (task, seed), group in groups.items() if len(group) > 1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts}


def run_episode(browser, task_type, task_name, seed, mode, base_url, args, directory):
    directory.mkdir()
    began = time.perf_counter()
    row = {"task": task_name, "seed": seed, "mode": mode,
           "benchmark_success": False, "official": None, "status": "setup_error"}
    context = backend = planner = policy = task = evaluator = None
    exchanges = ModelExchangeBuffer() if getattr(args, "record_model_exchanges", False) else None
    subgoals = mode in {"subgoal_rule", "subgoal_jev"}
    structured = getattr(args, "structured_browser", False)
    if mode in S1S2_MODES and not structured:
        raise ValueError("s1s2 modes require structured_browser")
    if subgoals and not structured:
        raise ValueError("subgoal modes require structured_browser")
    adaptive = mode in {"adaptive", "adaptive_rule", "subgoal_rule", "subgoal_jev"}
    uses_jev = mode in {"hybrid", "adaptive", "subgoal_jev"}
    trace_path = directory / "trace.jsonl"
    try:
        task = task_type(seed=seed, base_url=base_url)
        context = browser.new_context(viewport=task.viewport, service_workers="block")
        page = context.new_page()
        page.set_default_timeout(task.timeout)
        from .structured_browser import StructuredBrowserBackend
        backend_type = StructuredBrowserBackend if structured else BrowserAgentBackend
        backend = backend_type(
            page, allowed_origins=[base_url], efficient_observations=True,
            extended_controls=structured or getattr(args, "extended_controls", False),
            settle_transitions=getattr(args, "settle_transitions", False),
        )
        goal, _private_setup_info = task.setup(page)
        evaluator = Evaluator(task, page)
        initial = backend.observe()
        row["initial_observation_sha256"] = hashlib.sha256(json.dumps(
            {**initial.to_dict(), "url": task_name}, sort_keys=True,
        ).encode()).hexdigest()
        row["goal_sha256"] = hashlib.sha256(goal.encode()).hexdigest()
        (directory / "goal.txt").write_text(goal)
        page.screenshot(path=str(directory / "initial.png"))
        if mode in S1S2_MODES:
            return _run_s1s2(row, mode, backend, goal, evaluator, args, directory, trace_path, began, page,
                             exchanges)
        planner = MultimodalPlanner(
            os.environ["CUA_JEV_MODEL_BASE_URL"], os.environ["CUA_JEV_VLM_MODEL"],
            enable_thinking=THINKING[getattr(args, "enable_thinking", "default")],
            thinking_budget=getattr(args, "thinking_budget", None),
            api_key=os.environ["CUA_JEV_MODEL_API_KEY"], allow_insecure_http=True,
            use_env_proxy=False, request_timeout_s=60,
            browser_edit_prefix=adaptive,
            compact_browser_controls=structured, browser_subgoals=subgoals,
        )
        if exchanges:
            exchanges.attach(planner.client)
        policy = (JevPolicy(retries=2, model=getattr(args, "jev_model", None)) if uses_jev
                  else PlanOrderPolicy() if mode in {"plan_order", "adaptive_rule", "subgoal_rule"} else None)
        row["setup_wall_ms"] = (time.perf_counter() - began) * 1000
        runner = AgentRunner(
            backend, planner, policy=policy,
            config=AgentConfig(mode="vlm" if mode == "vlm" else "hybrid", allow_actions=True,
                               max_steps=args.max_steps, max_plans=args.max_steps + 5,
                               timeout_s=args.timeout, plan_horizon=6,
                               defer_immediate_plan_refresh=True, adaptive_dispatch=adaptive,
                               reuse_verified_post_snapshot=adaptive),
            trace=JsonlTrace(trace_path),
        )
        result = runner.run(goal, success=evaluator.accept, success_uses_live_state=adaptive).to_dict()
        row.update(result)
        # Runner's legacy 'jev_calls' field meters any hybrid selection policy.
        row["selection_calls"] = result["jev_calls"]
        row["jev_calls"] = result["jev_calls"] if uses_jev else 0
        row["jev_wall_ms"] = result["jev_wall_ms"] if uses_jev else 0.0
        row["official"] = evaluator.read()
        row["benchmark_success"] = row["official"]["done"] and row["official"]["reward"] > 0
        row["observation_metrics"] = dict(backend.observation_metrics)
        row["evaluation_calls"] = evaluator.calls
        row["evaluation_wall_ms"] = evaluator.wall_ms
        row["reported_tokens_by_role"] = analyze_traces([trace_path])["reported_model_tokens"]
        page.screenshot(path=str(directory / "final.png"))
    except Exception as error:
        row["infrastructure_error_type"] = type(error).__name__
        # No exception text: transport errors may include service credentials.
        if os.environ.get("CUA_JEV_DEBUG_TRACEBACK") == "1":  # local debugging only, never in reported runs
            import traceback
            row["debug_traceback"] = traceback.format_exc()[-2000:]
    finally:
        row["episode_wall_ms"] = (time.perf_counter() - began) * 1000
        cleanup_started = time.perf_counter()
        cleanup_errors = []
        if exchanges:
            try:
                exchanges.save(directory / "model-exchanges.json")
            except Exception as error:
                row["recording_error_type"] = type(error).__name__
        for resource in (policy, planner, backend, task, context):
            closer = getattr(resource, "teardown" if resource is task else "close", None)
            if closer:
                try:
                    closer()
                except Exception as error:
                    cleanup_errors.append(type(error).__name__)
        row["cleanup_errors"] = cleanup_errors
        row["cleanup_wall_ms"] = (time.perf_counter() - cleanup_started) * 1000
        row["total_wall_ms"] = (time.perf_counter() - began) * 1000
        write_json(directory / "result.json", row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-root", type=Path, required=True)
    parser.add_argument("--upstream-revision", required=True)
    parser.add_argument("--service-profile", type=Path, required=True)
    parser.add_argument("--jev-proxy-config", type=Path,
                        help="private HTTPS CONNECT proxy URL; VLM transport ignores environment proxies")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(PILOT_TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--structured-browser", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(LEGACY_MODES))
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--record-model-exchanges", action="store_true")
    parser.add_argument("--enable-thinking", choices=sorted(THINKING), default="default",
                        help="provider enable_thinking for EVERY mode's VLM requests")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="provider thinking_budget for EVERY mode (requires --enable-thinking on)")
    parser.add_argument("--jev-model", default=None, help="Jev model for every Jev-using mode")
    parser.add_argument("--s1-choice-gate", type=float, default=0.5)
    parser.add_argument("--extended-controls", action="store_true",
                        help="same bounded ARIA/script-styled control extraction for every comparison mode")
    parser.add_argument("--settle-transitions", action="store_true",
                        help="same bounded passive transition wait before post-action capture in every mode")
    args = parser.parse_args(argv)
    if args.max_steps < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("positive budgets required")
    if any(len(v) != len(set(v)) for v in (args.tasks, args.seeds, args.modes)):
        parser.error("duplicate tasks, seeds or modes would invalidate matched blocks")
    if any(m.startswith(("subgoal_", "s1s2_")) for m in args.modes) and not args.structured_browser:
        parser.error("subgoal modes require --structured-browser for all comparison modes")
    load_service_profile(args.service_profile)
    if args.jev_proxy_config:
        proxy_url = json.loads(args.jev_proxy_config.read_text())["url"]
        os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = proxy_url
    from browsergym.miniwob import ALL_MINIWOB_TASKS
    from playwright.sync_api import sync_playwright

    tasks = {t.subdomain: t for t in ALL_MINIWOB_TASKS}
    if any(t not in tasks for t in args.tasks):
        parser.error("unknown upstream task")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, args.output / "agent-source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    manifest = {
        "kind": "miniwob_pilot_custom_cua_jev_backend", "upstream_revision": args.upstream_revision,
        "models": {"vlm": os.environ["CUA_JEV_VLM_MODEL"], "jev": os.environ["CUA_JEV_MODEL"]},
        "tasks": args.tasks, "seeds": args.seeds, "modes": args.modes,
        "max_steps": args.max_steps, "timeout_s": args.timeout, "efficient_observations": True,
        "extended_controls_all_modes": args.extended_controls or args.structured_browser,
        "settle_transitions_all_modes": args.settle_transitions,
        "structured_browser": args.structured_browser,
        "subgoal_modes": [m for m in args.modes if m.startswith("subgoal_")],
        "plan_horizon": {mode: 1 if mode == "vlm" else 6 for mode in args.modes},
        "adaptive_features": {mode: mode in {"adaptive", "adaptive_rule", "subgoal_rule", "subgoal_jev"}
                              for mode in args.modes},
        "uncertainty_calibration": "disabled; no independently calibrated artifact yet",
        "order": list(schedule(args.tasks, args.seeds, args.modes)),
        "model_thinking": args.enable_thinking, "thinking_budget": args.thinking_budget,
        "jev_model": args.jev_model or os.environ.get("CUA_JEV_MODEL"),
        "s1_choice_gate": args.s1_choice_gate, "concurrency": 1,
        "raw_model_exchanges_recorded": args.record_model_exchanges,
        "jev_transport": "ssh_reverse_https_connect" if args.jev_proxy_config else "direct",
        "action_backend": "CUA-JEV Playwright DOM and pixel actions; no benchmark oracle access",
        "evaluator": "unmodified installed BrowserGym MiniWoB task.validate; terminal and reward > 0",
        "setup": "unmodified installed task.setup; native viewport; no BrowserGym screenshot upscaling",
        "packages": {n: importlib.metadata.version(n) for n in
                     ("browsergym-core", "browsergym-miniwob", "playwright", "httpx", "Pillow", "numpy")},
        "python": platform.python_version(), "host": platform.node(),
        "agent_source_sha256": source_hashes(Path(__file__).parent),
        "task_source_sha256": source_hashes(Path(__import__("browsergym.miniwob").miniwob.__file__).parent),
        "time_definition": {
            "duration_ms": "agent loop, including evaluator polling; excludes setup and final artifacts",
            "episode_wall_ms": "per-episode setup, agent, evaluation and artifacts; excludes cleanup",
            "total_wall_ms": "episode plus cleanup; shared browser startup recorded separately",
        },
    }
    write_json(args.output / "manifest.json", manifest)
    rows = []
    experiment_start = time.perf_counter()
    with miniwob_site(args.html_root) as base_url, sync_playwright() as playwright:
        started = time.perf_counter()
        browser = playwright.chromium.launch(headless=True)
        manifest["browser_startup_wall_ms"] = (time.perf_counter() - started) * 1000
        manifest["browser_version"] = browser.version
        write_json(args.output / "manifest.json", manifest)
        try:
            for index, (name, seed, mode) in enumerate(manifest["order"]):
                print(json.dumps({"event": "start", "index": index, "task": name,
                                  "seed": seed, "mode": mode}), flush=True)
                row = run_episode(browser, tasks[name], name, seed, mode, base_url, args,
                                  args.output / f"{index:03}-{name}-{seed}-{mode}")
                rows.append(row)
                summary = {"kind": manifest["kind"], "planned": len(manifest["order"]),
                           "completed": len(rows), "aggregate": aggregate(rows), "runs": rows,
                           "matched_initial_states": matching_initial_states(rows)}
                write_json(args.output / "summary.json", summary)
                print(json.dumps({"event": "result", "index": index,
                                  **{k: row.get(k) for k in ("task", "seed", "mode", "benchmark_success",
                                      "status", "infrastructure_error_type", "planner_calls", "jev_calls",
                                      "episode_wall_ms")}}), flush=True)
        finally:
            browser.close()
    manifest["experiment_wall_ms"] = (time.perf_counter() - experiment_start) * 1000
    write_json(args.output / "manifest.json", manifest)
    return 0


def _run_s1s2(row, mode, backend, goal, evaluator, args, directory, trace_path, began, page, exchanges):
    """Figure-2 System1/System2 runner; same backend, evaluator, budgets and artifacts.

    Resources it does not own (backend, task, context) are closed by run_episode's finally.
    """
    from .jev_decisions import JevDecisionClient
    from .s1s2_planner import SubgoalPlanner
    from .s1s2_runner import S1S2Config, S1S2Runner

    planner = SubgoalPlanner(os.environ["CUA_JEV_MODEL_BASE_URL"], os.environ["CUA_JEV_VLM_MODEL"],
                             api_key=os.environ["CUA_JEV_MODEL_API_KEY"], horizon=8,
                             enable_thinking=THINKING[getattr(args, "enable_thinking", "default")],
                             thinking_budget=getattr(args, "thinking_budget", None),
                             request_timeout_s=60)
    if exchanges:
        exchanges.attach(planner.client)  # saved once by run_episode's finally
    jev = JevDecisionClient(model=getattr(args, "jev_model", None)) if mode == "s1s2_jev" else None
    try:
        row["setup_wall_ms"] = (time.perf_counter() - began) * 1000
        runner = S1S2Runner(backend, planner, jev, config=S1S2Config(
            s1="jev" if mode == "s1s2_jev" else "rule", max_steps=args.max_steps,
            timeout_s=args.timeout, choice_gate=getattr(args, "s1_choice_gate", 0.5)),
            trace=JsonlTrace(trace_path))
        result = runner.run(goal, evaluator.accept)
        row.update(result)
        row["official"] = evaluator.read()
        row["benchmark_success"] = row["official"]["done"] and row["official"]["reward"] > 0
        row["observation_metrics"] = dict(backend.observation_metrics)
        row["evaluation_calls"] = evaluator.calls
        row["evaluation_wall_ms"] = evaluator.wall_ms
        page.screenshot(path=str(directory / "final.png"))
    except Exception as error:
        row["infrastructure_error_type"] = type(error).__name__
    finally:
        if jev is not None:
            write_json(directory / "jev-exchanges.json", jev.exchanges)
        for resource in (jev, planner):
            if resource is not None:
                resource.close()
    return row


if __name__ == "__main__":
    raise SystemExit(main())
