"""Run local VLM/Jev decisions against an owned OSWorld VM over SSH stdio.

Only observations and typed inputs cross SSH. Model API clients and credentials
remain on the local computer. The remote worker owns setup, evaluation, cleanup.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path, PurePosixPath

from run_osworld_smoke import (
    add_plan_arguments,
    experiment_settings,
    make_desktop_planner,
    planning_settings,
    run_episode,
    run_exit_code,
    use_worker_settling,
)

from cua_jev.config import load_local_env
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.osworld_observation_order import OBSERVATION_ORDERS, ORIGINAL_ORDER
from cua_jev.osworld_remote import RemoteOSWorldEnv
from cua_jev.osworld_startup_budget import BOOT_BUDGET_POLICIES, LEGACY_BOOT_BUDGET, SHARED_BOOT_BUDGET


def main(argv=None):
    load_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-user", required=True)
    parser.add_argument("--remote-project", required=True)
    parser.add_argument("--osworld-root", required=True)
    parser.add_argument("--vm-path", required=True)
    parser.add_argument("--remote-output", required=True)
    parser.add_argument("--remote-asset-cache")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=("hybrid", "vlm"), default=["hybrid"])
    parser.add_argument("--output", type=Path,
                        default=Path("runs/osworld-remote") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--model-base-url", default=os.getenv("CUA_JEV_MODEL_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("CUA_JEV_VLM_MODEL"))
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--use-env-proxy", action="store_true")
    parser.add_argument("--allow-screenshot-upload", action="store_true")
    parser.add_argument("--no-a11y", action="store_true")
    parser.add_argument("--ax-timeout", type=float, default=None,
                        help="agent AX budget on the Unix worker; fall back to vision after timeout")
    parser.add_argument("--observation-order", choices=OBSERVATION_ORDERS, default=ORIGINAL_ORDER,
                        help="explicit owned-worker acquisition order; original is the control")
    parser.add_argument(
        "--profile-observation-costs", action="store_true",
        help="use the separate diagnostic worker to record observation component costs; "
             "report separately from uninstrumented benchmark acceptance",
    )
    parser.add_argument("--vm-provider", choices=("docker", "qemu"), default="docker")
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=420)
    parser.add_argument("--owned-boot-budget", choices=BOOT_BUDGET_POLICIES, default=LEGACY_BOOT_BUDGET,
                        help="opt in to sharing the existing startup budget with one owned Docker boot")
    parser.add_argument("--startup-timeout", type=float, default=300,
                        help="SSH-to-ready budget in seconds, separate from the agent timeout (max 600)")
    add_plan_arguments(parser)
    args = parser.parse_args(argv)
    if args.ax_timeout is not None and (
        not math.isfinite(args.ax_timeout) or not 0 < args.ax_timeout <= 60 or args.no_a11y
    ):
        parser.error("--ax-timeout requires accessibility and a finite budget in (0, 60]")
    if args.no_a11y and args.image_only_fill_intermediates:
        parser.error("--image-only-fill-intermediates requires accessibility")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.remote_user):
        parser.error("--remote-user must be an explicit Unix account name")
    if not 0 <= args.pause <= 10:
        parser.error("--pause must be between 0 and 10 seconds")
    if args.max_steps <= 0 or not 0 < args.timeout <= 3600:
        parser.error("--max-steps must be positive and --timeout must be between zero and 3600 seconds")
    if not 0 < args.startup_timeout <= 600:
        parser.error("--startup-timeout must be finite and between zero and 600 seconds")
    if args.owned_boot_budget == SHARED_BOOT_BUDGET and args.vm_provider != "docker":
        parser.error("shared owned boot budget currently supports Docker only")
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*/[a-zA-Z0-9_-]+", name) for name in args.tasks):
        parser.error("tasks must use domain/task-id without a .json suffix")
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.modes)) != len(args.modes):
        parser.error("tasks and modes must be unique")
    remote_paths = [args.remote_project, args.osworld_root, args.vm_path, args.remote_output]
    if args.remote_asset_cache:
        remote_paths.append(args.remote_asset_cache)
    for value in remote_paths:
        if not PurePosixPath(value).is_absolute() or "\x00" in value or "\n" in value:
            parser.error("remote paths must be absolute POSIX paths")
    probe = make_desktop_planner(args)
    probe.close()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "decision_location": "local",
        "vm_transport": "ssh_stdio",
        "planning_by_mode": {mode: planning_settings(args, mode) for mode in args.modes},
        "experiment_settings": experiment_settings(args),
    }
    if args.observation_order != ORIGINAL_ORDER:
        metadata["requested_observation_acquisition_order"] = args.observation_order
    if args.owned_boot_budget == SHARED_BOOT_BUDGET:
        metadata["requested_owned_boot_budget"] = {
            "policy": args.owned_boot_budget, "outer_startup_timeout_s": args.startup_timeout,
            "agent_timeout_unchanged": True, "SDK_timeout_semantics_verified": False,
        }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reports = []
    for selection in args.tasks:
        for mode in args.modes:
            task = {}
            episode_dir = args.output / selection.replace("/", "-") / mode
            remote_output = str(PurePosixPath(args.remote_output) / selection.replace("/", "-") / mode)

            def create_env(*, _selection=selection, _task=task, _episode_dir=episode_dir,
                           _remote_output=remote_output, **_ignored):
                project = PurePosixPath(args.remote_project)
                worker_name = ("osworld_fill_observation_profiler.py" if args.profile_observation_costs
                               else "osworld_remote_worker.py")
                command = [str(project / ".venv/bin/python"), "-B", "-u",
                           str(project / "scripts" / worker_name),
                           "--expected-user", args.remote_user,
                           "--osworld-root", args.osworld_root, "--vm-path", args.vm_path,
                           "--vm-provider", args.vm_provider,
                           "--task", _selection, "--output", _remote_output]
                if args.observation_order != ORIGINAL_ORDER:
                    command.extend(["--observation-order", args.observation_order])
                if args.owned_boot_budget == SHARED_BOOT_BUDGET:
                    command.extend(["--owned-boot-budget", args.owned_boot_budget,
                                    "--startup-timeout", str(args.startup_timeout)])
                if args.no_a11y:
                    command.append("--no-a11y")
                if args.ax_timeout is not None:
                    command.extend(["--ax-timeout", str(args.ax_timeout)])
                if args.image_only_fill_intermediates:
                    command.append("--image-only-fill-intermediates")
                if not args.passive_settling:
                    command.append("--no-passive-settling")
                if args.remote_asset_cache:
                    command.extend(["--asset-cache", args.remote_asset_cache])
                env = RemoteOSWorldEnv(
                    host=args.host,
                    command=command,
                    log_path=_episode_dir / "worker.log",
                    startup_timeout_s=args.startup_timeout,
                )
                _task["instruction"] = env.instruction
                return env

            print(json.dumps({"event": "episode_start", "task": selection, "mode": mode,
                              "decision_location": "local", "vm_transport": "ssh_stdio"}), flush=True)
            report = run_episode(
                args, selection, task, mode,
                env_factory=create_env,
                backend_factory=OSWorldBackend,
                planner_factory=make_desktop_planner,
                initialize=lambda env, _task: env.initial_observation,
                settle=use_worker_settling,
            )
            report.update(decision_location="local", vm_transport="ssh_stdio",
                          remote_output=remote_output)
            (episode_dir / "result.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps({
                "kind": "osworld_integration_smoke", **metadata,
                "run_status": "success" if run_exit_code(reports) == 0 else "failed",
                "runs": reports,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if report.get("cleanup_status") != "confirmed":
                # A failed close/startup is not evidence that resources are gone.
                # Preserve this attempt and leave remaining suite slots unstarted.
                return run_exit_code(reports)
    return run_exit_code(reports)


if __name__ == "__main__":
    raise SystemExit(main())
