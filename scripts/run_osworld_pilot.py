"""Run three matched policies on owned local OSWorld workers, without SSH to self."""

from __future__ import annotations

import argparse
import getpass
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from run_osworld_smoke import (
    add_plan_arguments,
    experiment_settings,
    make_desktop_planner,
    run_episode,
    source_metadata,
    use_worker_settling,
)

from cua_jev.benchmark_miniwob import LEGACY_MODES as MODES
from cua_jev.benchmark_miniwob import PlanOrderPolicy, schedule, source_hashes, write_json
from cua_jev.config import load_service_profile
from cua_jev.model_recording import ModelExchangeBuffer
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.osworld_remote import RemoteOSWorldEnv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-user", required=True)
    parser.add_argument("--osworld-root", type=Path, required=True)
    parser.add_argument("--vm-path", type=Path, required=True)
    parser.add_argument("--service-profile", type=Path, required=True)
    parser.add_argument("--jev-proxy-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--model-base-url")
    parser.add_argument("--model")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--qemu-boot-timeout", type=float, default=180)
    parser.add_argument("--lazy-evaluators", action="store_true")
    parser.add_argument("--ax-timeout", type=float, default=5)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--allow-insecure-model-http", action="store_true")
    parser.add_argument("--allow-screenshot-upload", action="store_true")
    parser.add_argument("--use-env-proxy", action="store_true")
    add_plan_arguments(parser)
    args = parser.parse_args(argv)
    if getpass.getuser() != args.expected_user:
        parser.error("account does not match expected user")
    home = Path.home().resolve()
    for path in (args.osworld_root, args.vm_path, args.output):
        if not path.resolve().is_relative_to(home):
            parser.error("all environment and result paths must remain in the current user's home")
    args.output = args.output.resolve()
    project = Path(__file__).resolve().parents[1]
    if not args.output.is_relative_to(project) or args.output == project:
        parser.error("output must be inside the worker project to satisfy its ownership guard")
    args.osworld_root = args.osworld_root.resolve()
    args.vm_path = args.vm_path.resolve()
    args.vm_provider = "qemu"
    args.no_a11y = False
    args.observation_order = "original"
    load_service_profile(args.service_profile)
    os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = json.loads(
        args.jev_proxy_config.read_text())["url"]
    args.model = args.model or os.environ["CUA_JEV_VLM_MODEL"]
    args.model_base_url = args.model_base_url or os.environ["CUA_JEV_MODEL_BASE_URL"]
    probe = make_desktop_planner(args)
    probe.close()
    args.output.mkdir(parents=True, exist_ok=False)
    root_output = args.output
    shutil.copytree(project / "src/cua_jev", root_output / "agent-source",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("run_osworld_pilot.py", "run_osworld_smoke.py", "osworld_remote_worker.py"):
        shutil.copy2(project / "scripts" / name, root_output / name)
    manifest = {
        "kind": "osworld_pilot", "decision_location": "zju-18", "vm_transport": "owned_local_stdio",
        "provider": "qemu", "max_steps": args.max_steps, "agent_timeout_s": args.timeout,
        "startup_timeout_s": args.startup_timeout, "qemu_boot_timeout_s": args.qemu_boot_timeout,
        "lazy_original_evaluator_exports": args.lazy_evaluators,
        "jev_transport": "ssh_reverse_https_connect",
        "models": {"vlm": args.model, "jev": os.environ["CUA_JEV_MODEL"]},
        "python": platform.python_version(), "host": platform.node(),
        "packages": {n: importlib.metadata.version(n) for n in
                     ("gymnasium", "httpx", "Pillow", "numpy", "lxml", "borb", "easyocr")},
        "order": list(schedule(args.tasks, [0], args.modes)), "settings": experiment_settings(args),
        "agent_source_sha256": source_hashes(project / "src/cua_jev"),
        **source_metadata(args.osworld_root, args.tasks),
    }
    if manifest["osworld_tracked_files_dirty"] is not False:
        raise ValueError("pilot requires a clean evaluator checkout")
    write_json(root_output / "manifest.json", manifest)
    reports = []
    for index, (task_name, _seed, mode) in enumerate(manifest["order"]):
        args.output = root_output / f"{index:02}-{mode}"
        worker_output = args.output / "worker"
        task = {}
        exchanges = ModelExchangeBuffer()

        def create_planner(options, _exchanges=exchanges):
            planner = make_desktop_planner(options)
            _exchanges.attach(planner.client)
            return planner

        def create_env(*, _name=task_name, _output=worker_output, _task=task, **_ignored):
            command = [sys.executable, "-B", "-u", str(project / "scripts/osworld_remote_worker.py"),
                       "--expected-user", args.expected_user, "--osworld-root", str(args.osworld_root),
                       "--vm-path", str(args.vm_path), "--vm-provider", "qemu", "--task", _name,
                       "--output", str(_output), "--ax-timeout", str(args.ax_timeout),
                       "--qemu-boot-timeout", str(args.qemu_boot_timeout)]
            if args.lazy_evaluators:
                command.append("--lazy-evaluators")
            if not args.passive_settling:
                command.append("--no-passive-settling")
            # The worker and guest need neither model credentials nor the API relay.
            worker_env = {k: v for k, v in os.environ.items() if
                          not k.startswith(("CUA_JEV_", "TYPESAFE_")) and
                          k.lower() not in {"https_proxy", "http_proxy", "all_proxy"}}
            env = RemoteOSWorldEnv(
                host="owned-local-worker", command=command,
                log_path=args.output / "worker.log", startup_timeout_s=args.startup_timeout,
                timeout_s=120,
                popen_factory=lambda _argv, **kwargs: subprocess.Popen(command, env=worker_env, **kwargs),
            )
            _task["instruction"] = env.instruction
            return env

        print(json.dumps({"event": "episode_start", "index": index, "task": task_name, "mode": mode}),
              flush=True)
        runner_mode = "vlm" if mode == "vlm" else "hybrid"
        report = run_episode(
            args, task_name, task, runner_mode, env_factory=create_env,
            backend_factory=OSWorldBackend, planner_factory=create_planner,
            initialize=lambda env, _task: env.initial_observation, settle=use_worker_settling,
            policy_factory=PlanOrderPolicy if mode == "plan_order" else None,
        )
        report.update(mode=mode, live_jev=mode == "hybrid", vm_transport="owned_local_stdio",
                      decision_location="zju-18", worker_output=str(worker_output))
        exchanges.save(args.output / "model-exchanges.json")
        if report.get("runner"):
            result = report["runner"]
            result["selection_calls"] = result["jev_calls"]
            if mode != "hybrid":
                result["jev_calls"], result["jev_wall_ms"] = 0, 0.0
        write_json(args.output / "result.json", report)
        write_json(args.output / task_name.replace("/", "-") / runner_mode / "result.json", report)
        reports.append(report)
        write_json(root_output / "summary.json", {"kind": "osworld_pilot", "planned": len(manifest["order"]),
                                                  "completed": len(reports), "runs": reports})
        print(json.dumps({"event": "episode_result", "index": index, "mode": mode,
                          **{k: report.get(k) for k in ("episode_status", "official_score",
                                                       "cleanup_status", "duration_ms")}}), flush=True)
        if report.get("cleanup_status") != "confirmed":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
