"""Matched GUI+MCP episodes in disposable, chenbofan-owned OSWorld guests."""

import argparse
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import httpx

from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.benchmark_miniwob import PlanOrderPolicy, schedule, source_hashes, write_json
from cua_jev.config import load_service_profile
from cua_jev.mcp_session import PersistentMcpSession
from cua_jev.model_recording import ModelExchangeBuffer
from cua_jev.osworld_backend import OSWorldBackend
from cua_jev.osworld_mcp import JointOSWorldBackend, JointOSWorldPlanner
from cua_jev.osworld_qemu import _listener_ports, _process_identity
from cua_jev.osworld_remote import RemoteOSWorldEnv
from cua_jev.policy import JevPolicy
from cua_jev.trace import JsonlTrace
from cua_jev.trace_analysis import analyze_traces

MODES = ("vlm", "adaptive_rule", "adaptive", "s1s2_rule", "s1s2_jev")
S1S2_MODES = ("s1s2_rule", "s1s2_jev")
THINKING = {"default": None, "on": True, "off": False}


def private_local_path(path):
    """The read-only VM base image may also live outside home on a faster local disk, but
    only as an owned file inside an owner-only (mode 0700) directory."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError("VM image is not a regular file")
    parent = resolved.parent.stat()
    if (parent.st_uid != os.getuid() or parent.st_mode & 0o077
            or resolved.stat().st_uid != os.getuid()):
        raise ValueError("VM image directory is not owner-only")
    return resolved


def owned_path(path, home, *, preserve_launcher=False):
    if not path.resolve().is_relative_to(home.resolve()):
        raise ValueError("resource is outside current user's home")
    # Executing the resolved target of a venv symlink loses pyvenv.cfg discovery.
    return path.absolute() if preserve_launcher else path.resolve()


def install_guest_service(worker_output, bundle, installer):
    """Operator setup only. Never exposed as an agent tool or to model prompts."""
    meta = json.loads((worker_output / "metadata.json").read_text())
    if meta.get("osworld_tracked_files_dirty") is not False:
        raise ValueError("original evaluator checkout must remain clean")
    start = meta["vm_runtime"]["starts"][-1]
    pid, identity = start["pid"], start["starttime"]
    ports = {int(k): v for k, v in start["ports"].items()}

    def owned():
        if (
            Path(f"/proc/{pid}").stat().st_uid != os.getuid()
            or _process_identity(pid) != identity
            or _listener_ports(pid, identity, set(ports.values())) != set(ports.values())
        ):
            raise ValueError("owned guest process or loopback listeners changed")

    owned()
    endpoint = f"http://127.0.0.1:{ports[5000]}"
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=180) as client:
        with bundle.open("rb") as stream:
            response = client.post(
                endpoint + "/setup/upload",
                data={"file_path": "/tmp/cua-jev-mcp-bundle.tar.gz"},
                files={"file_data": (bundle.name, stream, "application/gzip")},
            )
        response.raise_for_status()
        owned()
        code = installer.read_text()
        if len(code) > 64000:
            raise ValueError("operator setup script exceeds bounded size")
        response = client.post(
            endpoint + "/execute", json={"command": ["python3", "-c", code], "shell": False}
        )
        response.raise_for_status()
        result = response.json()
    if result.get("returncode") != 0:
        raise RuntimeError("guest service installation failed")
    setup = json.loads(result["output"])
    if setup.get("probe_returncode") != 0 or setup.get("listening") is not True:
        raise RuntimeError("guest service did not become ready")
    write_json(worker_output.parent / "mcp-install.json", setup)
    return f"http://127.0.0.1:{ports[9292]}/mcp"


def run_one(args, task, mode, directory):
    directory.mkdir(mode=0o700)
    project = Path(__file__).resolve().parents[1]
    worker_output = directory / "worker"
    declare_infeasible = False
    row = {
        "task": task,
        "seed": 0,
        "mode": mode,
        "official_score": None,
        "benchmark_success": False,
        "status": "setup_error",
    }
    began = time.perf_counter()
    env = backend = session = planner = policy = None
    exchanges, agent_started = ModelExchangeBuffer(), False
    try:
        command = [
            str(args.worker_python),
            "-B",
            "-u",
            str(project / "scripts/osworld_remote_worker.py"),
            "--expected-user",
            args.expected_user,
            "--osworld-root",
            str(args.osworld_root),
            "--vm-path",
            str(args.vm_path),
            "--vm-provider",
            "qemu",
            "--task",
            task,
            "--output",
            str(worker_output),
            "--ax-timeout",
            "5",
            *(["--qemu-scratch", str(args.qemu_scratch)] if getattr(args, "qemu_scratch", None) else []),
            "--qemu-boot-timeout",
            "600",
            "--lazy-evaluators",
            "--guest-mcp-port",
        ]
        if getattr(args, "guest_proxy", None):
            command += ["--guest-proxy", args.guest_proxy]
        if getattr(args, "lenient_startup_checks", False):
            command += ["--lenient-startup-checks"]
        worker_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("CUA_JEV_", "TYPESAFE_"))
            and k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        }
        worker_env["PYTHONPATH"] = str(project / "src")
        if getattr(args, "guest_proxy", None):
            # Host-side original setup downloads and evaluator reference files use the same proxy;
            # the worker's own loopback traffic to the guest bypasses it.
            proxy_url = "http://" + args.guest_proxy
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                worker_env[key] = proxy_url
            worker_env["NO_PROXY"] = worker_env["no_proxy"] = "localhost,127.0.0.1,::1"
        env = RemoteOSWorldEnv(
            host="owned-local-worker",
            command=command,
            log_path=directory / "worker.log",
            startup_timeout_s=getattr(args, "startup_timeout", 1500),
            timeout_s=90,
            close_timeout_s=45,
            popen_factory=lambda _argv, **kw: subprocess.Popen(command, env=worker_env, **kw),
        )
        row["guest_setup_wall_ms"] = (time.perf_counter() - began) * 1000
        mcp_at = time.perf_counter()
        url = install_guest_service(worker_output, args.mcp_bundle, args.mcp_installer)
        session = PersistentMcpSession(url=url, connect_timeout_s=120, call_timeout_s=60)
        backend = JointOSWorldBackend(OSWorldBackend(env), session)
        write_json(directory / "mcp-catalog.json", list(backend.catalog.tools.values()))
        row["mcp_catalog_count"] = len(backend.catalog.tools)
        row["mcp_setup_wall_ms"] = (time.perf_counter() - mcp_at) * 1000
        thinking = THINKING[getattr(args, "enable_thinking", "default")]
        budget = getattr(args, "thinking_budget", None)
        if mode in S1S2_MODES:
            from cua_jev.jev_decisions import JevDecisionClient
            from cua_jev.s1s2_desktop import ToolChannels
            from cua_jev.s1s2_planner import SubgoalPlanner
            from cua_jev.s1s2_runner import S1S2Config, S1S2Runner
            planner = SubgoalPlanner(os.environ["CUA_JEV_MODEL_BASE_URL"], os.environ["CUA_JEV_VLM_MODEL"],
                                     api_key=os.environ["CUA_JEV_MODEL_API_KEY"], horizon=4,
                                     enable_thinking=thinking, thinking_budget=budget, request_timeout_s=60,
                                     desktop=True, catalog=backend.catalog)
            exchanges.attach(planner.client)
            policy = JevDecisionClient(model=getattr(args, "jev_model", None)) if mode == "s1s2_jev" else None
            channels = ToolChannels(backend, s1="jev" if policy else "rule", jev=policy)
            trace = JsonlTrace(directory / "trace.jsonl")
            row["instruction_sha256"] = hashlib.sha256(env.instruction.encode()).hexdigest()
            (directory / "goal.txt").write_text(env.instruction)
            (directory / "initial.png").write_bytes(env.initial_observation["screenshot"])
            row["setup_wall_ms"] = (time.perf_counter() - began) * 1000
            agent_started = True
            agent_at = time.perf_counter()
            result = S1S2Runner(backend, planner, policy, trace=trace, channels=channels, config=S1S2Config(
                s1="jev" if policy else "rule", max_steps=args.max_steps, timeout_s=args.timeout,
                max_s2_calls=args.max_steps, max_failures=getattr(args, "max_failures", 3),
            )).run(env.instruction, None)
            row.update(result)
            row["agent_wall_ms"] = (time.perf_counter() - agent_at) * 1000
            declare_infeasible = bool(result.get("infeasible"))
            row["mcp_backend_metrics"] = dict(backend.metrics)
            row["offered_tools_last"] = [t["name"] for t in planner.offered_tools]
            if policy is not None:
                write_json(directory / "jev-exchanges.json", policy.exchanges)
            return row
        planner = JointOSWorldPlanner(
            os.environ["CUA_JEV_MODEL_BASE_URL"],
            os.environ["CUA_JEV_VLM_MODEL"],
            api_key=os.environ["CUA_JEV_MODEL_API_KEY"],
            allow_insecure_http=True,
            use_env_proxy=False,
            backend=backend,
            enable_thinking=thinking, thinking_budget=budget,
        )
        exchanges.attach(planner.client)
        policy = JevPolicy(retries=2) if mode == "adaptive" else PlanOrderPolicy()
        adaptive = mode != "vlm"
        trace = JsonlTrace(directory / "trace.jsonl")
        row["instruction_sha256"] = hashlib.sha256(env.instruction.encode()).hexdigest()
        (directory / "goal.txt").write_text(env.instruction)
        (directory / "initial.png").write_bytes(env.initial_observation["screenshot"])
        row["setup_wall_ms"] = (time.perf_counter() - began) * 1000
        agent_started = True
        agent_at = time.perf_counter()
        result = AgentRunner(
            backend,
            planner,
            policy=policy,
            trace=trace,
            config=AgentConfig(
                mode="hybrid" if adaptive else "vlm",
                allow_actions=True,
                max_steps=args.max_steps,
                max_plans=args.max_steps + 5,
                timeout_s=args.timeout,
                plan_horizon=6,
                adaptive_dispatch=adaptive,
                reuse_verified_post_snapshot=adaptive,
                max_failures=getattr(args, "max_failures", 3),
            ),
        ).run(env.instruction)
        row.update(result.to_dict())
        row["agent_wall_ms"] = (time.perf_counter() - agent_at) * 1000
        declare_infeasible = bool(getattr(planner, "declared_infeasible", False)
                                  and row.get("status") in {"unverified", "done"})
        row["selection_calls"] = row["jev_calls"]
        if mode != "adaptive":
            row["jev_calls"], row["jev_wall_ms"] = 0, 0.0
        row["reported_tokens_by_role"] = analyze_traces([directory / "trace.jsonl"])["reported_model_tokens"]
        row["observation_phases"] = backend.observation_phase_metrics
        row["mcp_backend_metrics"] = dict(backend.metrics)
    except Exception as exc:
        row["error_type"] = type(exc).__name__
        row["error_code"] = getattr(exc, "code", None)
    finally:
        # Exactly one final evaluation, outside every model/agent interface.
        if agent_started and env is not None:
            at = time.perf_counter()
            try:
                row["declared_infeasible"] = declare_infeasible
                row["official_score"] = (env.evaluate(infeasible=True) if declare_infeasible
                                         else env.evaluate())
                row["benchmark_success"] = row["official_score"] > 0
                row["evaluation_status"] = "valid"
            except Exception as exc:
                row["evaluation_status"] = "error"
                row["evaluation_error_type"] = type(exc).__name__
            row["evaluation_wall_ms"] = (time.perf_counter() - at) * 1000
        cleanup_at, errors = time.perf_counter(), []
        for resource in (policy, planner, backend if backend is not None else session, env):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    errors.append(type(exc).__name__)
        row["cleanup_status"] = (
            "confirmed"
            if not errors and env is not None and env.cleanup_status == "confirmed"
            else "unconfirmed"
        )
        row["cleanup_errors"] = errors
        row["cleanup_wall_ms"] = (time.perf_counter() - cleanup_at) * 1000
        row["total_wall_ms"] = (time.perf_counter() - began) * 1000
        if session is not None:
            row["mcp_session_metrics"] = dict(session.metrics)
        exchanges.save(directory / "model-exchanges.json")
        write_json(directory / "result.json", row)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-user", required=True)
    for name in (
        "osworld-root",
        "vm-path",
        "worker-python",
        "mcp-bundle",
        "mcp-installer",
        "service-profile",
        "jev-proxy-config",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--enable-thinking", choices=sorted(THINKING), default="default",
                        help="provider enable_thinking for EVERY mode")
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--qemu-scratch", type=Path, default=None,
                        help="private local-disk directory for QEMU overlays (harness only)")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="consecutive action failures before an episode stops (both groups)")
    parser.add_argument("--jev-model", default=None)
    parser.add_argument("--guest-proxy", default=None,
                        help="IPv4:port HTTP proxy for guest web access and downloads")
    parser.add_argument("--lenient-startup-checks", action="store_true",
                        help="harness readiness/settling failures recorded but not fatal (all modes)")
    parser.add_argument("--startup-timeout", type=float, default=1500,
                        help="fresh guest + task setup budget (recorded separately from agent time)")
    args = parser.parse_args(argv)
    if getpass.getuser() != args.expected_user:
        parser.error("account does not match expected owner")
    if not 1 <= args.max_steps <= 50 or not 0 < args.timeout <= 3600:
        parser.error("invalid episode budgets")
    home, project = Path.home().resolve(), Path(__file__).resolve().parents[1]
    for name in (
        "osworld_root",
        "vm_path",
        "worker_python",
        "mcp_bundle",
        "mcp_installer",
        "service_profile",
        "jev_proxy_config",
        "output",
    ):
        try:
            try:
                path = owned_path(getattr(args, name), home, preserve_launcher=name == "worker_python")
            except ValueError:
                if name != "vm_path":
                    raise
                path = private_local_path(args.vm_path)
        except ValueError:
            parser.error("all resources must remain in the current user's home")
        setattr(args, name, path)
    if not args.output.is_relative_to(project) or args.output == project:
        parser.error("output must be inside the owned worker project")
    if any(len(x) != len(set(x)) for x in (args.tasks, args.modes)):
        parser.error("duplicate task or mode")
    load_service_profile(args.service_profile)
    os.environ["https_proxy"] = os.environ["HTTPS_PROXY"] = json.loads(args.jev_proxy_config.read_text())[
        "url"
    ]
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    shutil.copytree(
        project / "src/cua_jev",
        args.output / "agent-source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    manifest = {
        "kind": "osworld_mcp_joint_adapter_pilot",
        "order": list(schedule(args.tasks, [0], args.modes)),
        "max_steps": args.max_steps,
        "max_failures": getattr(args, "max_failures", 3),
        "timeout_s": args.timeout,
        "concurrency": 1,
        "model_thinking": args.enable_thinking, "thinking_budget": args.thinking_budget,
        "guest_proxy": args.guest_proxy, "startup_timeout_s": args.startup_timeout,
        "lenient_startup_checks": args.lenient_startup_checks,
        "jev_model": args.jev_model or os.environ.get("CUA_JEV_MODEL"),
        "models": {"vlm": os.environ["CUA_JEV_VLM_MODEL"], "jev": os.environ["CUA_JEV_MODEL"]},
        "agent_source_sha256": source_hashes(project / "src/cua_jev"),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bundle_sha256": hashlib.sha256(args.mcp_bundle.read_bytes()).hexdigest(),
        "installer_sha256": hashlib.sha256(args.mcp_installer.read_bytes()).hexdigest(),
        "evaluation": "original final evaluator once after stop; no intermediate reward or task config",
        "tool_scope": "catalog from pinned original application server; no filesystem/git aggregator",
        "uncertainty_calibration": "disabled; no independent calibrated artifact",
        "time_definition": {
            "total_wall_ms": "fresh guest, task setup, MCP setup, agent, evaluator, cleanup",
            "agent_wall_ms": "agent only, including model and observation latency",
        },
    }
    write_json(args.output / "manifest.json", manifest)
    rows = []
    for index, (task, _seed, mode) in enumerate(manifest["order"]):
        print(json.dumps({"event": "start", "index": index, "task": task, "mode": mode}), flush=True)
        row = run_one(args, task, mode, args.output / f"{index:02}-{mode}")
        rows.append(row)
        write_json(
            args.output / "summary.json",
            {"planned": len(manifest["order"]), "completed": len(rows), "runs": rows},
        )
        print(
            json.dumps(
                {
                    "event": "result",
                    "index": index,
                    **{
                        k: row.get(k)
                        for k in (
                            "mode",
                            "status",
                            "official_score",
                            "error_type",
                            "cleanup_status",
                            "total_wall_ms",
                        )
                    },
                }
            ),
            flush=True,
        )
        if row["cleanup_status"] != "confirmed":
            leftovers = episode_leftover_processes(args.output / f"{index:02}-{mode}")
            row["cleanup_leftover_processes"] = leftovers
            write_json(args.output / f"{index:02}-{mode}" / "result.json", row)
            if leftovers:
                return 1   # a guest or worker may still be alive: stop this shard (original safety rule)
            row["cleanup_status"] = "verified_no_leftover_processes"
            write_json(args.output / "summary.json",
                       {"planned": len(manifest["order"]), "completed": len(rows), "runs": rows})
    return 0


def episode_leftover_processes(directory):
    """PIDs of this account whose command line, cwd or open files point into the episode directory."""
    root, found = str(Path(directory).resolve()), []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        base = f"/proc/{entry}"
        try:
            if os.stat(base).st_uid != os.getuid():
                continue
            hits = [open(f"{base}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace"),
                    os.readlink(f"{base}/cwd")]
            for fd in os.listdir(f"{base}/fd"):
                try:
                    hits.append(os.readlink(f"{base}/fd/{fd}"))
                except OSError:
                    pass
        except OSError:
            continue
        if any(root in item for item in hits):
            found.append(int(entry))
    return found


if __name__ == "__main__":
    raise SystemExit(main())
