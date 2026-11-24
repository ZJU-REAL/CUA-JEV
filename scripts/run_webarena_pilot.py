"""Run matched WebArena policies against separately owned, resettable websites."""

import argparse
import getpass
import hashlib
import importlib.metadata
import importlib.resources
import json
import math
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

from cua_jev.benchmark_miniwob import aggregate, matching_initial_states, schedule, source_hashes, write_json
from cua_jev.benchmark_webarena import MODES, run_episode, validate_task_set
from cua_jev.config import load_service_profile
from cua_jev.webarena_instance import OwnedWebArenaSite

SITE_NAMES = ("shopping", "shopping_admin", "reddit", "gitlab", "wikipedia", "map", "homepage")


def package_source_hashes(name):
    """Namespace distributions can have __file__ = None and several roots."""
    package = __import__(name, fromlist=["__path__"])
    roots = tuple(Path(p) for p in package.__path__)
    if not roots:
        raise ValueError("package has no installed source roots")
    return {f"root-{index}/{relative}": digest for index, root in enumerate(roots)
            for relative, digest in source_hashes(root).items()}


def finish_episode_sites(row, instances):
    """Charge container cleanup to its own episode, including the final one."""
    began = time.perf_counter()
    errors = []
    for name, instance in instances.items():
        try:
            instance.stop()
        except Exception as error:
            errors.append({"site": name, "error_type": type(error).__name__})
    elapsed = (time.perf_counter() - began) * 1000
    row["site_cleanup_errors"] = errors
    row["site_cleanup_wall_ms"] = elapsed
    row["total_wall_ms"] += elapsed
    row["cleanup_wall_ms"] += elapsed
    return not errors


def task_site_url(name, origin):
    # WebArena's SHOPPING_ADMIN variable denotes the admin application entry
    # point, while Magento's configured base URL remains the site's origin.
    return origin.rstrip("/") + ("/admin" if name == "shopping_admin" else "")


def read_site_config(path):
    config = json.loads(path.read_text())
    if set(config) != {"sites"} or not isinstance(config["sites"], dict) or not config["sites"]:
        raise ValueError("site profile requires a nonempty sites mapping")
    if set(config["sites"]) - set(SITE_NAMES):
        raise ValueError("unknown WebArena website name")
    for settings in config["sites"].values():
        if (
            not isinstance(settings, dict)
            or "image" not in settings
            or set(settings) - {"image", "container_port", "command", "after_reset", "startup_timeout_s",
                                "loopback_proxy"}
        ):
            raise ValueError("invalid owned website configuration")
        for command in [settings.get("command", []), *settings.get("after_reset", [])]:
            if not isinstance(command, list) or any(type(a) is not str for a in command):
                raise ValueError("website commands must be trusted argv lists")
    return config["sites"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-user", required=True)
    parser.add_argument("--site-profile", type=Path, required=True)
    parser.add_argument("--service-profile", type=Path, required=True)
    parser.add_argument("--jev-proxy-config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--structured-browser", action="store_true")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=["vlm", "adaptive_rule", "adaptive"])
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--enable-thinking", choices=["default", "on", "off"], default="default",
                        help="provider enable_thinking for EVERY mode")
    parser.add_argument("--thinking-budget", type=int, default=None, help="requires --enable-thinking on")
    parser.add_argument("--vlm-stale-replans", type=int, default=0,
                        help="vlm baseline: zero-input stale rejections re-grounded by VLM, not charged")
    parser.add_argument("--jev-model", default=None, help="Jev model for every Jev-using mode")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--setup-timeout", type=float, default=60,
                        help="trusted login/navigation wait per operation; restored before agent execution")
    args = parser.parse_args(argv)
    if args.max_steps < 1 or not math.isfinite(args.timeout) or not 0 < args.timeout <= 3600:
        parser.error("positive steps and finite timeout in (0, 3600] are required")
    if not math.isfinite(args.setup_timeout) or not 0 < args.setup_timeout <= 600:
        parser.error("setup timeout must be finite and in (0, 600] seconds")
    if getpass.getuser() != args.expected_user:
        parser.error("account does not match expected user")
    if not args.output.resolve().is_relative_to(Path.home().resolve()):
        parser.error("output must remain in the current user's home")
    if any(len(values) != len(set(values)) for values in (args.tasks, args.seeds, args.modes)):
        parser.error("duplicate tasks, seeds or modes")
    if any(m.startswith(("subgoal_", "s1s2_")) for m in args.modes) and not args.structured_browser:
        parser.error("subgoal modes require --structured-browser for all comparison modes")
    site_config = read_site_config(args.site_profile)
    load_service_profile(args.service_profile)
    if args.jev_proxy_config:
        proxy = json.loads(args.jev_proxy_config.read_text())["url"]
        os.environ["HTTPS_PROXY"] = os.environ["https_proxy"] = proxy
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    args.output.chmod(0o700)
    temporary = args.output / "tmp"
    temporary.mkdir(mode=0o700)
    os.environ["TMPDIR"] = tempfile.tempdir = str(temporary.resolve())
    instances = {}
    rows = []
    manifest = None
    experiment_started = time.perf_counter()
    try:
        for name, settings in site_config.items():
            instance = OwnedWebArenaSite(
                settings["image"],
                args.output / "websites" / name,
                expected_user=args.expected_user,
                container_port=settings.get("container_port", 80),
                startup_timeout_s=settings.get("startup_timeout_s", 300),
                command=tuple(settings.get("command", [])),
                loopback_proxy=settings.get("loopback_proxy", False),
            )
            instances[name] = instance
            instance.reserve_address()
        for name in SITE_NAMES:
            # BrowserGym requires all URL variables on import. Unconfigured
            # sites are unreachable placeholders and are forbidden by preflight.
            os.environ["WA_" + name.upper()] = (
                task_site_url(name, instances[name].url) if name in instances else "http://127.0.0.1:1"
            )
        os.environ["WA_FULL_RESET"] = ""
        from browsergym.webarena.task import GenericWebArenaTask
        from playwright.sync_api import sync_playwright

        raw_configs = importlib.resources.files("webarena").joinpath("test.raw.json").read_bytes()
        configs = json.loads(raw_configs)
        tasks = validate_task_set(configs, args.tasks, instances)
        task_metadata = {item["task_id"]: item for item in tasks}
        project = Path(__file__).resolve().parents[1]
        shutil.copytree(
            project / "src/cua_jev",
            args.output / "agent-source",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(__file__, args.output / Path(__file__).name)
        manifest = {
            "kind": "webarena_browsergym_port_custom_cua_jev_backend",
            "tasks": tasks,
            "seeds": args.seeds,
            "modes": args.modes,
            "max_steps": args.max_steps,
            "timeout_s": args.timeout,
            "structured_browser": args.structured_browser,
            "model_thinking": args.enable_thinking, "thinking_budget": args.thinking_budget,
            "jev_model": args.jev_model or os.environ.get("CUA_JEV_MODEL"),
            "vlm_stale_replans": args.vlm_stale_replans,
            "setup_operation_timeout_s": args.setup_timeout,
            "order": list(schedule(args.tasks, args.seeds, args.modes)),
            "models": {"vlm": os.environ["CUA_JEV_VLM_MODEL"], "jev": os.environ["CUA_JEV_MODEL"]},
            "python": platform.python_version(),
            "host": platform.node(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("browsergym-webarena", "browsergym-core", "playwright", "httpx")
            },
            "task_config_sha256": hashlib.sha256(raw_configs).hexdigest(),
            "site_profile_sha256": hashlib.sha256(args.site_profile.read_bytes()).hexdigest(),
            "agent_source_sha256": source_hashes(project / "src/cua_jev"),
            "task_source_sha256": source_hashes(
                Path(__import__("browsergym.webarena").webarena.__file__).parent
            ),
            "evaluator_source_sha256": package_source_hashes("webarena"),
            "evaluation": "one official task.validate after agent stop; no intermediate evaluator feedback",
            "llm_judge": "disabled; tasks needing a judge rejected before model calls",
            "reset": "fresh owned container/anonymous volumes from immutable image before every episode",
            "uncertainty_calibration": "disabled; no independently calibrated artifact",
            "extended_controls_all_modes": True,
            "settle_transitions_all_modes": True,
            "time_definition": {
                "total_wall_ms": "reset, setup, agent, final evaluation, browser-context and website cleanup",
                "duration_ms": "agent loop only; no evaluator polling",
                "experiment_wall_ms": "whole driver after profile load, with shared browser/network cleanup",
            },
        }
        write_json(args.output / "manifest.json", manifest)
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            manifest["browser_version"] = browser.version
            write_json(args.output / "manifest.json", manifest)
            try:
                for index, (task_id, seed, mode) in enumerate(manifest["order"]):

                    def reset(_sites=tuple(task_metadata[task_id]["sites"])):
                        for instance in instances.values():
                            instance.stop()
                        results = {}
                        for site in _sites:
                            instance = instances[site]
                            results[site] = instance.reset()
                            for command in site_config[site].get("after_reset", []):
                                instance.configure(command)
                        return results

                    print(
                        json.dumps(
                            {"event": "start", "index": index, "task": task_id, "seed": seed, "mode": mode}
                        ),
                        flush=True,
                    )
                    row = run_episode(
                        browser,
                        GenericWebArenaTask,
                        task_id=task_id,
                        seed=seed,
                        mode=mode,
                        reset=reset,
                        allowed_origins=[instance.url for instance in instances.values()],
                        directory=args.output / f"{index:04}-{task_id}-{seed}-{mode}",
                        max_steps=args.max_steps,
                        timeout_s=args.timeout,
                        setup_timeout_s=args.setup_timeout,
                        structured_browser=args.structured_browser,
                        enable_thinking={"default": None, "on": True, "off": False}[args.enable_thinking],
                        thinking_budget=args.thinking_budget, jev_model=args.jev_model,
                        vlm_stale_replans=args.vlm_stale_replans,
                    )
                    sites_clean = finish_episode_sites(row, instances)
                    write_json(args.output / f"{index:04}-{task_id}-{seed}-{mode}" / "result.json", row)
                    rows.append(row)
                    write_json(
                        args.output / "summary.json",
                        {
                            "kind": manifest["kind"],
                            "planned": len(manifest["order"]),
                            "completed": len(rows),
                            "aggregate": aggregate(rows),
                            "runs": rows,
                            "matched_initial_states": matching_initial_states(rows),
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "event": "result",
                                "index": index,
                                **{
                                    key: row.get(key)
                                    for key in (
                                        "task",
                                        "seed",
                                        "mode",
                                        "benchmark_success",
                                        "status",
                                        "infrastructure_error_type",
                                        "total_wall_ms",
                                        "planner_calls",
                                        "jev_calls",
                                    )
                                },
                            }
                        ),
                        flush=True,
                    )
                    if not sites_clean:
                        return 1
            finally:
                browser.close()
    finally:
        cleanup_errors = []
        for name, instance in instances.items():
            try:
                instance.close()
            except Exception as error:
                cleanup_errors.append({"site": name, "error_type": type(error).__name__})
        if manifest is not None:
            manifest["experiment_wall_ms"] = (time.perf_counter() - experiment_started) * 1000
            manifest["site_cleanup_errors"] = cleanup_errors
            write_json(args.output / "manifest.json", manifest)
        write_json(
            args.output / "resource-audit.json",
            {
                "cleanup_errors": cleanup_errors,
                "remaining_owned_containers": [i.container_id for i in instances.values() if i.container_id],
                "remaining_owned_networks": [i.network_id for i in instances.values() if i.network_id],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
