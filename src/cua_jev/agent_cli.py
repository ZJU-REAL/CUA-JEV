"""CLI wiring for the experimental Jev + VLM browser agent."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .agent_runner import AgentConfig, AgentRunner
from .agent_types import AgentSnapshot
from .policy import JevPolicy
from .trace import JsonlTrace


def configure_parser(sub):
    parser = sub.add_parser("agent-browser", help="VLM short plans with freshly grounded Jev actions")
    parser.add_argument("--goal", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--mode", choices=("hybrid", "vlm"), default="hybrid")
    parser.add_argument("--model-base-url", default=os.getenv("CUA_JEV_MODEL_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("CUA_JEV_VLM_MODEL"))
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--allow-screenshot-upload", action="store_true")
    parser.add_argument(
        "--allow-actions",
        action="store_true",
        help="authorize input and clicks within the specified browser origin scope",
    )
    parser.add_argument("--allow-origin", action="append", default=[])
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--browser-channel", default="chromium")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-plans", type=int, default=40)
    parser.add_argument("--plan-horizon", type=int, default=6)
    parser.add_argument("--efficient-observations", action="store_true",
                        help="omit unused DOM-edit screenshots and defer immediate-plan refresh to preflight")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--require-url-contains", default="")
    parser.add_argument("--require-text", action="append", default=[])
    parser.add_argument("--expect-field", action="append", default=[], metavar="LABEL=VALUE")
    parser.add_argument("--expect-checked", action="append", default=[], metavar="LABEL")
    parser.add_argument("--trace", default="runs/agent-browser.jsonl")
    parser.add_argument("--output", default="runs/agent-browser-summary.json")
    return parser


def acceptance(args):
    fields = []
    for field in args.expect_field:
        if "=" not in field or not field.split("=", 1)[0]:
            raise ValueError("--expect-field requires LABEL=VALUE")
        fields.append(tuple(field.split("=", 1)))
    if not (args.require_url_contains or args.require_text or fields or args.expect_checked):
        return None

    def check(snapshot: AgentSnapshot) -> bool:
        if args.require_url_contains and args.require_url_contains not in snapshot.url:
            return False
        if any(text not in snapshot.text for text in args.require_text):
            return False
        for label, value in fields:
            matches = [control for control in snapshot.controls if control.label == label]
            if len(matches) != 1 or matches[0].value != value:
                return False
        for label in args.expect_checked:
            matches = [control for control in snapshot.controls if control.label == label]
            if len(matches) != 1 or not matches[0].checked:
                return False
        return True

    return check


def make_planner(args):
    from .agent_planner import MultimodalPlanner

    if not args.model_base_url or not args.model:
        raise ValueError(
            "configure CUA_JEV_MODEL_BASE_URL and CUA_JEV_VLM_MODEL, or pass --model-base-url/--model"
        )
    if not args.allow_screenshot_upload:
        raise ValueError("this multimodal agent requires --allow-screenshot-upload")
    return MultimodalPlanner(
        args.model_base_url,
        args.model,
        api_key=os.getenv("CUA_JEV_MODEL_API_KEY") or os.getenv("CUA_JEV_PLANNER_API_KEY"),
        allow_insecure_http=args.allow_insecure_model_http,
        use_env_proxy=args.use_env_proxy,
    )


def run(args) -> int:
    from playwright.sync_api import sync_playwright

    from .agent_browser import BrowserAgentBackend

    evaluator = acceptance(args)
    config = AgentConfig(
        mode=args.mode,
        max_steps=args.max_steps,
        max_plans=args.max_plans,
        plan_horizon=args.plan_horizon,
        timeout_s=args.timeout,
        allow_actions=args.allow_actions,
        defer_immediate_plan_refresh=getattr(args, "efficient_observations", False),
    )
    planner = make_planner(args)
    policy = None
    try:
        policy = JevPolicy(retries=2) if args.mode == "hybrid" else None
        with sync_playwright() as playwright:
            launch = {"headless": not args.headed}
            if args.browser_channel != "chromium":
                launch["channel"] = args.browser_channel
            browser = playwright.chromium.launch(**launch)
            try:
                context = browser.new_context(
                    viewport={"width": 1100, "height": 760}, service_workers="block"
                )
                page = context.new_page()
                # Install the origin guard before the first navigation.
                backend = BrowserAgentBackend(
                    page, allowed_origins=[args.url, *args.allow_origin],
                    efficient_observations=getattr(args, "efficient_observations", False),
                )
                try:
                    page.goto(args.url, wait_until="domcontentloaded", timeout=30000)
                    runner = AgentRunner(
                        backend,
                        planner,
                        policy=policy,
                        config=config,
                        trace=JsonlTrace(args.trace),
                        progress=lambda data: print(
                            json.dumps(data, ensure_ascii=False), file=sys.stderr, flush=True
                        ),
                    )
                    result = runner.run(args.goal, success=evaluator)
                finally:
                    backend.close()
            finally:
                browser.close()
    finally:
        planner.close()
        if policy is not None:
            policy.close()
    payload = {"mode": args.mode, "vlm_model": args.model, **result.to_dict(),
               "observation_metrics": backend.observation_metrics}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.success else 1
