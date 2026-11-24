"""WebArena adapter: reset per episode, evaluate only after the agent stops.

BrowserGym's WebArena port may navigate during evaluation. Its evaluator is
therefore never supplied as an AgentRunner success callback or a model tool.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

from .agent_planner import MultimodalPlanner
from .agent_runner import AgentConfig, AgentRunner
from .benchmark_miniwob import PlanOrderPolicy, write_json
from .browser_redirects import RedirectingBrowserBackend
from .model_recording import ModelExchangeBuffer
from .policy import JevPolicy
from .trace import JsonlTrace
from .trace_analysis import analyze_traces

MODES = ("vlm", "adaptive_rule", "adaptive", "subgoal_rule", "subgoal_jev", "s1s2_rule", "s1s2_jev")
S1S2_MODES = ("s1s2_rule", "s1s2_jev")
_ANSWER_INSTRUCTION = (
    "\nWhen the task is finished, return done=true with steps=[]. If the user requested an answer, "
    "put the complete final user-facing answer in summary, without a completion preamble. "
    "For an action task, summary may describe the observed completed work. "
    "A completion claim is checked separately; do not assume it is correct."
)


class AnswerPlanner:
    """Preserve a model's explicit stop answer without accessing evaluator data."""

    def __init__(self, inner):
        self.inner = inner
        self.final_answer = None

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def plan(self, goal, snapshot, history, *, max_actions=6):
        self.final_answer = None
        plan = self.inner.plan(goal + _ANSWER_INSTRUCTION, snapshot, history, max_actions=max_actions)
        if plan.done and not plan.steps:
            self.final_answer = plan.summary
        return plan


def requires_llm_judge(config):
    """Reject unsupported judge requirements before running or spending tokens."""
    def visit(value):
        if isinstance(value, dict):
            return any(key in {"fuzzy_match", "llm_fuzzy_match", "llm_ua_match"} or visit(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(visit(item) for item in value)
        return False
    return visit(config.get("eval", {}))


def validate_task_set(configs, task_ids, available_sites):
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("nonempty unique WebArena task ids required")
    by_id = {item["task_id"]: item for item in configs}
    if len(by_id) != len(configs):
        raise ValueError("duplicate official task ids")
    metadata = []
    for task_id in task_ids:
        if task_id not in by_id:
            raise ValueError("unknown WebArena task id")
        config = by_id[task_id]
        if not set(config["sites"]) <= set(available_sites):
            raise ValueError("task requires a website instance that is not configured")
        if requires_llm_judge(config):
            raise ValueError("task requires a separately configured official LLM judge")
        metadata.append({"task_id": task_id, "sites": list(config["sites"]),
                         "eval_types": list(config["eval"]["eval_types"]),
                         "intent_template_id": config.get("intent_template_id")})
    return metadata


def evaluate_final(task, page, final_answer):
    """Exactly one final official evaluation; no score, URL or answer in feedback."""
    began = time.perf_counter()
    reward, done, _message, _private_info = task.validate(
        page, [{"role": "assistant", "message": final_answer if final_answer is not None else ""}],
    )
    if type(reward) not in (int, float) or not math.isfinite(reward):
        raise ValueError("invalid official WebArena reward")
    return {"reward": float(reward), "done": bool(done),
            "evaluation_wall_ms": (time.perf_counter() - began) * 1000,
            "evaluator_calls": 1}


def _artifact_screenshot(page, directory, row):
    """The final screenshot is a record only; its failure must not abort official evaluation."""
    try:
        page.screenshot(path=str(directory / "agent-final.png"), timeout=15000)
    except Exception as error:
        row["artifact_error_type"] = type(error).__name__


def _trusted_evaluation_timeouts(page):
    """The evaluator runs after the agent stopped; a loaded site gets 60 s per page operation."""
    page.set_default_timeout(60000)
    page.set_default_navigation_timeout(60000)
    page.context.set_default_timeout(60000)
    page.context.set_default_navigation_timeout(60000)


def run_episode(browser, task_factory, *, task_id, seed, mode, reset, allowed_origins,
                directory: Path, max_steps=30, timeout_s=300, planner_factory=None,
                policy_factory=None, record_model_exchanges=True, setup_timeout_s=None,
                structured_browser=False, enable_thinking=None, thinking_budget=None, jev_model=None,
                vlm_stale_replans=0):
    if mode not in MODES:
        raise ValueError("unsupported WebArena comparison mode")
    if setup_timeout_s is not None and (
        type(setup_timeout_s) not in (int, float)
        or not math.isfinite(setup_timeout_s) or not 0 < setup_timeout_s <= 600
    ):
        raise ValueError("setup timeout must be finite and in (0, 600] seconds")
    directory.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    row = {"task": str(task_id), "seed": seed, "mode": mode, "benchmark_success": False,
           "official": None, "status": "setup_error", "reset_completed": False}
    context = backend = planner = policy = task = None
    trace_path = directory / "trace.jsonl"
    exchanges = ModelExchangeBuffer() if record_model_exchanges else None
    subgoals = mode in {"subgoal_rule", "subgoal_jev"}
    if (subgoals or mode in S1S2_MODES) and not structured_browser:
        raise ValueError("subgoal and s1s2 modes require structured_browser")
    adaptive = mode in {"adaptive_rule", "adaptive", "subgoal_rule", "subgoal_jev"}
    phase = "reset"
    try:
        reset_started = time.perf_counter()
        reset_metadata = reset()
        row["reset_wall_ms"] = (time.perf_counter() - reset_started) * 1000
        row["reset_completed"] = True
        row["reset_metadata"] = reset_metadata
        phase = "task_construction"
        task = task_factory(seed=seed, task_id=task_id)
        phase = "browser_context"
        context = browser.new_context(viewport=task.viewport, service_workers="block")
        page = context.new_page()
        page.set_default_timeout(task.timeout)
        phase = "task_setup"
        if setup_timeout_s is not None:
            # Login opens its own page, so both the context and initial page
            # need the trusted setup budget. Restore the original page/context budgets
            # even when setup fails; never extend model/action execution here.
            for target in (context, page):
                target.set_default_timeout(setup_timeout_s * 1000)
                target.set_default_navigation_timeout(setup_timeout_s * 1000)
        try:
            goal, _private_setup_info = task.setup(page)
        finally:
            if setup_timeout_s is not None:
                # A newly created Playwright context defaults to 30 seconds;
                # only the original task page was set to task.timeout above.
                for target, original_ms in ((context, 30000), (page, task.timeout)):
                    target.set_default_timeout(original_ms)
                    target.set_default_navigation_timeout(original_ms)
        # Authentication/setup is trusted environment work, outside the agent.
        # Multi-tab tasks need a tab-aware backend and are explicitly refused;
        # a setup failure stays in the denominator rather than silently closing tabs.
        if len(context.pages) != 1:
            raise ValueError("multi-tab WebArena setup needs a tab-aware backend")
        phase = "initial_observation"
        from .structured_browser import StructuredRedirectingBrowserBackend
        backend_type = (StructuredRedirectingBrowserBackend if structured_browser
                        else RedirectingBrowserBackend)
        backend = backend_type(page, allowed_origins=allowed_origins,
                                            efficient_observations=True, extended_controls=True,
                                            settle_transitions=True)
        initial = backend.observe()
        row["initial_observation_sha256"] = hashlib.sha256(
            json.dumps(initial.to_dict(), sort_keys=True).encode()).hexdigest()
        row["goal_sha256"] = hashlib.sha256(goal.encode()).hexdigest()
        (directory / "goal.txt").write_text(goal)
        phase = "initial_artifact"
        page.screenshot(path=str(directory / "initial.png"))
        phase = "agent_setup"
        if mode in S1S2_MODES:
            from .jev_decisions import JevDecisionClient
            from .s1s2_planner import SubgoalPlanner
            from .s1s2_runner import S1S2Config, S1S2Runner
            planner = SubgoalPlanner(os.environ["CUA_JEV_MODEL_BASE_URL"], os.environ["CUA_JEV_VLM_MODEL"],
                                     api_key=os.environ["CUA_JEV_MODEL_API_KEY"], horizon=8,
                                     enable_thinking=enable_thinking, thinking_budget=thinking_budget,
                                     request_timeout_s=60)
            if exchanges:
                exchanges.attach(planner.client)
            policy = JevDecisionClient(model=jev_model) if mode == "s1s2_jev" else None
            row["setup_wall_ms"] = (time.perf_counter() - began) * 1000
            phase = "agent"
            result = S1S2Runner(backend, planner, policy, trace=JsonlTrace(trace_path), config=S1S2Config(
                s1="jev" if mode == "s1s2_jev" else "rule", max_steps=max_steps, timeout_s=timeout_s,
                max_s2_calls=max_steps)).run(goal, None)
            row.update(result)
            phase = "result_recording"
            row["observation_metrics"] = dict(backend.observation_metrics)
            if policy is not None:
                write_json(directory / "jev-exchanges.json", policy.exchanges)
            phase = "final_artifact"
            _artifact_screenshot(page, directory, row)
            phase = "agent_detach"
            backend.close()
            phase = "final_evaluation"
            _trusted_evaluation_timeouts(page)
            row["official"] = evaluate_final(task, page, result["final_answer"])
            row["benchmark_success"] = row["official"]["reward"] > 0
            return row
        inner = (planner_factory(adaptive=adaptive) if planner_factory else MultimodalPlanner(
            os.environ["CUA_JEV_MODEL_BASE_URL"], os.environ["CUA_JEV_VLM_MODEL"],
            api_key=os.environ["CUA_JEV_MODEL_API_KEY"], allow_insecure_http=True,
            use_env_proxy=False, request_timeout_s=60, browser_edit_prefix=adaptive,
            enable_thinking=enable_thinking, thinking_budget=thinking_budget,
            compact_browser_controls=structured_browser, browser_subgoals=subgoals,
        ))
        if exchanges and hasattr(inner, "client"):
            exchanges.attach(inner.client)
        planner = AnswerPlanner(inner)
        policy = (policy_factory(mode) if policy_factory else
                  JevPolicy(retries=2, model=jev_model) if mode in {"adaptive", "subgoal_jev"} else
                  PlanOrderPolicy() if mode in {"adaptive_rule", "subgoal_rule"} else None)
        row["setup_wall_ms"] = (time.perf_counter() - began) * 1000
        phase = "agent"
        result = AgentRunner(
            backend, planner, policy=policy, trace=JsonlTrace(trace_path),
            config=AgentConfig(mode="vlm" if mode == "vlm" else "hybrid", allow_actions=True,
                               max_steps=max_steps, max_plans=max_steps + 5, timeout_s=timeout_s,
                               adaptive_dispatch=adaptive, reuse_verified_post_snapshot=adaptive,
                               defer_immediate_plan_refresh=True,
                               vlm_stale_replans=vlm_stale_replans if mode == "vlm" else 0),
        ).run(goal).to_dict()
        row.update(result)
        phase = "result_recording"
        row["selection_calls"] = result["jev_calls"]
        if mode not in {"adaptive", "subgoal_jev"}:
            row["jev_calls"] = 0
            row["jev_wall_ms"] = 0.0
        row["observation_metrics"] = dict(backend.observation_metrics)
        row["reported_tokens_by_role"] = analyze_traces([trace_path])["reported_model_tokens"]
        row["final_answer"] = planner.final_answer
        phase = "final_artifact"
        _artifact_screenshot(page, directory, row)
        # Detach agent route guards before the trusted evaluator follows its
        # reference URLs. No model or agent action runs after this boundary.
        phase = "agent_detach"
        backend.close()
        phase = "final_evaluation"
        _trusted_evaluation_timeouts(page)
        row["official"] = evaluate_final(task, page, planner.final_answer)
        row["benchmark_success"] = row["official"]["reward"] > 0
    except Exception as error:
        row["infrastructure_error_type"] = type(error).__name__
        row["infrastructure_error_phase"] = phase
    finally:
        row["episode_wall_ms"] = (time.perf_counter() - began) * 1000
        cleanup_started = time.perf_counter()
        cleanup_errors = []
        for resource, method in ((planner, "close"), (policy, "close"), (backend, "close"),
                                 (task, "teardown"), (context, "close")):
            if resource is not None and callable(getattr(resource, method, None)):
                try:
                    getattr(resource, method)()
                except Exception as error:
                    cleanup_errors.append(type(error).__name__)
        row["cleanup_wall_ms"] = (time.perf_counter() - cleanup_started) * 1000
        row["cleanup_errors"] = cleanup_errors
        row["total_wall_ms"] = (time.perf_counter() - began) * 1000
        if exchanges:
            exchanges.save(directory / "model-exchanges.json")
        write_json(directory / "result.json", row)
    return row
