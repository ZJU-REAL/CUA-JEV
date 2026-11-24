"""Live model smoke runs on disposable local tasks, not a benchmark.

Both modes use the same VLM, browser inputs, action backend and independent
acceptance checks. No scripted planner or automatic API fallback is provided.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from agent_smoke_fixture import TASKS, SmokeSite

from cua_jev.agent_browser import BrowserAgentBackend
from cua_jev.agent_cli import make_planner
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.config import load_local_env
from cua_jev.policy import JevPolicy
from cua_jev.trace import JsonlTrace


def main(argv=None):
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--modes", nargs="+", choices=("hybrid", "vlm"), default=["hybrid"])
    parser.add_argument("--model-base-url", default=os.getenv("CUA_JEV_MODEL_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("CUA_JEV_VLM_MODEL"))
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--allow-screenshot-upload", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--efficient-observations", action="store_true")
    parser.add_argument("--browser-channel", default="chromium")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/agent-smoke") / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    args = parser.parse_args(argv)
    # Check configuration before launching a browser; never silently replace a model.
    probe = make_planner(args)
    probe.close()
    from playwright.sync_api import sync_playwright

    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    with SmokeSite() as site, sync_playwright() as playwright:
        launch = {"headless": not args.headed}
        if args.browser_channel != "chromium":
            launch["channel"] = args.browser_channel
        browser = playwright.chromium.launch(**launch)
        try:
            for task in args.tasks:
                for mode in args.modes:
                    name = f"{task}-{mode}"
                    print(json.dumps({"event": "start", "task": task, "mode": mode}), flush=True)
                    with browser.new_context(
                        viewport={"width": 1100, "height": 760}, service_workers="block"
                    ) as context:
                        page = context.new_page()
                        backend = BrowserAgentBackend(
                            page, allowed_origins=[site.url(task)],
                            efficient_observations=args.efficient_observations,
                        )
                        planner = make_planner(args)
                        policy = None
                        try:
                            policy = JevPolicy(retries=2) if mode == "hybrid" else None
                            page.goto(site.url(task), wait_until="domcontentloaded")
                            runner = AgentRunner(
                                backend,
                                planner,
                                policy=policy,
                                config=AgentConfig(
                                    mode=mode,
                                    allow_actions=True,
                                    max_steps=args.max_steps,
                                    max_plans=args.max_steps + 5,
                                    timeout_s=args.timeout,
                                    defer_immediate_plan_refresh=args.efficient_observations,
                                ),
                                trace=JsonlTrace(args.output / f"{name}.jsonl"),
                                progress=lambda data: print(
                                    json.dumps(data, ensure_ascii=False), file=sys.stderr, flush=True
                                ),
                            )
                            result = runner.run(
                                site.goal(task), success=lambda _, t=task, p=page: site.verify(t, p)
                            )
                            page.screenshot(path=str(args.output / f"{name}-final.png"))
                            reports.append(
                                {
                                    "task": task,
                                    "mode": mode,
                                    "model": args.model,
                                    "live_vlm": True,
                                    "live_jev": mode == "hybrid",
                                    **result.to_dict(),
                                    "efficient_observations": args.efficient_observations,
                                    "observation_metrics": backend.observation_metrics,
                                }
                            )
                            (args.output / "summary.json").write_text(
                                json.dumps(
                                    {"kind": "local_integration_smoke", "runs": reports},
                                    ensure_ascii=False,
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            print(json.dumps(reports[-1], ensure_ascii=False), flush=True)
                        finally:
                            if policy:
                                policy.close()
                            planner.close()
                            backend.close()
        finally:
            browser.close()
    return 0 if all(report["success"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
