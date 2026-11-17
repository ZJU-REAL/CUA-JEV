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

MODES = ("vlm", "adaptive_rule", "adaptive")


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
            "--qemu-boot-timeout",
            "450",
            "--lazy-evaluators",
            "--guest-mcp-port",
        ]
        worker_env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("CUA_JEV_", "TYPESAFE_"))
            and k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}
        }
        worker_env["PYTHONPATH"] = str(project / "src")
        env = RemoteOSWorldEnv(
            host="owned-local-worker",
            command=command,
            log_path=directory / "worker.log",
            startup_timeout_s=600,
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
        planner = JointOSWorldPlanner(
            os.environ["CUA_JEV_MODEL_BASE_URL"],
            os.environ["CUA_JEV_VLM_MODEL"],
            api_key=os.environ["CUA_JEV_MODEL_API_KEY"],
            allow_insecure_http=True,
            use_env_proxy=False,
            backend=backend,
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
            ),
        ).run(env.instruction)
        row.update(result.to_dict())
        row["agent_wall_ms"] = (time.perf_counter() - agent_at) * 1000
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
                row["official_score"] = env.evaluate()
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
    args = parser.parse_args(argv)
    if getpass.getuser() != args.expected_user:
        parser.error("account does not match expected owner")
    if not 1 <= args.max_steps <= 50 or not 0 < args.timeout <= 600:
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
            path = owned_path(getattr(args, name), home, preserve_launcher=name == "worker_python")
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
        "timeout_s": args.timeout,
        "concurrency": 1,
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
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
