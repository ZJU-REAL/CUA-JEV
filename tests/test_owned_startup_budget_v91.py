"""Inherited boot allocation never extends agent/outer budgets or starts a retry."""

import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_osworld_driver import driver as smoke
from test_osworld_remote_driver import arguments, install_episode_fakes
from test_osworld_remote_driver import driver as driver
from test_osworld_remote_worker import worker

from cua_jev.osworld_startup_budget import (
    LEGACY_BOOT_BUDGET,
    SHARED_BOOT_BUDGET,
    OwnedBootWaitBudget,
)


@pytest.fixture(autouse=True)
def no_real_process_or_network(monkeypatch):
    def blocked(*_args, **_kwargs):
        pytest.fail("startup budget tests forbid real network and subprocesses")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(subprocess, "Popen", blocked)


class Clock:
    now = 0

    def read(self):
        return self.now


@pytest.mark.parametrize("value", [None, True, 0, -1, 601, float("nan"), float("inf")])
def test_invalid_inherited_budget_is_rejected_before_a_VM_or_SDK_wait(value):
    with pytest.raises(ValueError):
        OwnedBootWaitBudget(value, worker_entry_ns=0)


def test_import_and_factory_time_are_deducted_once_before_one_original_SDK_wait():
    clock, calls, result = Clock(), [], object()
    budget = OwnedBootWaitBudget(600, worker_entry_ns=0, clock=clock.read)
    clock.now = 120_000_000_000
    budget.begin_start()
    clock.now = 150_750_000_000

    def original_wait(*, timeout):
        calls.append(timeout)
        clock.now += 3_000_000_000
        return result

    assert budget.wait_for_ready(original_wait) is result
    assert calls == [434]  # floor(600 - 150.75 - explicit 15s allowance), not a renewed600.
    assert budget.report["start_invocations"] == budget.report["SDK_wait_invocations"] == 1
    assert budget.report["SDK_wait_wall_ms"] == 3000
    assert budget.report["worker_elapsed_after_wait_ms"] == 153750
    assert budget.report["SDK_timeout_semantics_verified"] is False
    assert budget.report["cross_host_deadline_identity_proven"] is False


def test_exhausted_remaining_budget_and_second_start_never_call_the_SDK():
    clock, calls = Clock(), []
    budget = OwnedBootWaitBudget(600, worker_entry_ns=0, clock=clock.read)
    clock.now = 590_000_000_000
    with pytest.raises(TimeoutError):
        budget.begin_start()
    assert calls == []
    budget = OwnedBootWaitBudget(600, worker_entry_ns=0, clock=clock.read)
    clock.now = 0
    budget.begin_start()
    with pytest.raises(RuntimeError):
        budget.begin_start()
    assert budget.report["start_invocations"] == 2 and budget.report["SDK_wait_invocations"] == 0
    clock.now = 0
    budget = OwnedBootWaitBudget(600, worker_entry_ns=0, clock=clock.read)
    budget.begin_start()
    clock.now = 590_000_000_000
    with pytest.raises(TimeoutError):
        budget.wait_for_ready(lambda **kwargs: calls.append(kwargs))
    assert calls == [] and budget.report["wait_admission_attempts"] == 1
    assert budget.report["SDK_wait_invocations"] == 0


def test_failed_original_wait_keeps_exception_identity_and_does_not_retry():
    clock, calls = Clock(), []
    failure = TimeoutError("private original boot error")
    budget = OwnedBootWaitBudget(600, worker_entry_ns=0, clock=clock.read)
    budget.begin_start()

    def original_wait(*, timeout):
        calls.append(timeout)
        clock.now += 1_000_000_000
        raise failure

    with pytest.raises(TimeoutError) as caught:
        budget.wait_for_ready(original_wait)
    assert caught.value is failure and calls == [585]
    assert budget.report["SDK_wait_error_type"] == "TimeoutError"
    assert "private original boot error" not in str(budget.report)


@pytest.mark.parametrize("shared", [False, True])
def test_owned_factory_uses_original180_or_inherited_remainder_and_always_owns_cleanup(
    monkeypatch, tmp_path, shared,
):
    base = tmp_path / "base.qcow2"
    base.write_bytes(b"synthetic public VM fixture")
    monkeypatch.setattr(smoke, "validate_vm_path", lambda _path: base)
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda path: path.as_posix() == "/dev/kvm" or original_exists(path))
    clock, calls = Clock(), []
    clock.now = 100_000_000_000
    budget = OwnedBootWaitBudget(600, worker_entry_ns=0, clock=clock.read) if shared else None

    class Container:
        def __init__(self, options):
            self.name = options["name"]
            self.attrs = {"Config": {"Labels": dict(options["labels"])}, "NetworkSettings": {
                "Ports": {f"{port}/tcp": [{"HostIp": "127.0.0.1", "HostPort": str(port + 20000)}]
                          for port in (5000, 8006, 8080, 9222)},
            }}

        def reload(self):
            pass

        def start(self):
            calls.append("start")
            clock.now += 10_000_000_000

        def stop(self, **_kwargs):
            calls.append("stop")

        def remove(self):
            calls.append("remove")

    def create(_image, **options):
        calls.append("create")
        assert options["volumes"] == {str(base): {"bind": "/System.qcow2", "mode": "ro"}}
        return Container(options)

    class Provider:
        container = None
        client = SimpleNamespace(images=SimpleNamespace(get=lambda _name: None),
                                 containers=SimpleNamespace(create=create))

        def _wait_for_vm_ready(self, timeout):
            calls.append(("wait", timeout))
            clock.now += 1_000_000_000

        def start_emulator(self, *_args):
            pytest.fail("the shared provider factory must not start")

        def stop_emulator(self, *_args):
            pytest.fail("the shared provider factory must not stop")

    start, stop = Provider.start_emulator, Provider.stop_emulator
    options = {"boot_wait_budget": budget} if shared else {}
    with smoke.isolated_docker(Provider, base, "budget-fixture", owner="chenbofan", **options):
        Provider().start_emulator(str(base), True, "Ubuntu")
    assert calls == ["create", "start", ("wait", 475 if shared else 180), "stop", "remove"]
    assert Provider.start_emulator is start and Provider.stop_emulator is stop


@pytest.mark.parametrize("policy", [LEGACY_BOOT_BUDGET, SHARED_BOOT_BUDGET])
def test_remote_passes_only_optin_to_worker_without_extending_outer_or_agent_budget(
    driver, monkeypatch, tmp_path, policy,
):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, **{"owned-boot-budget": policy, "startup-timeout": "600"})
    assert driver.main([*argv, "--model-thinking", "on"]) == 0
    env = calls["environments"][0]
    assert env.startup_timeout_s == 600 and calls["configs"][0].timeout_s == 420
    assert "--model-thinking" not in env.command
    if policy == LEGACY_BOOT_BUDGET:
        assert "--owned-boot-budget" not in env.command and "--startup-timeout" not in env.command
    else:
        assert env.command[env.command.index("--owned-boot-budget") + 1] == policy
        assert float(env.command[env.command.index("--startup-timeout") + 1]) == 600


def test_worker_context_inherits_original_entry_clock_and_keeps_legacy_kwargs(monkeypatch, tmp_path):
    calls = []

    class Budget:
        def __init__(self, timeout, *, worker_entry_ns):
            calls.append((timeout, worker_entry_ns))
            self.report = {"version": SHARED_BOOT_BUDGET}

    @contextmanager
    def isolated(*_args, **kwargs):
        calls.append(kwargs)
        yield

    monkeypatch.setitem(sys.modules, "desktop_env.providers.docker.provider",
                        SimpleNamespace(DockerProvider=object))
    monkeypatch.setattr(worker, "isolated_docker", isolated)
    monkeypatch.setattr(worker, "OwnedBootWaitBudget", Budget)
    monkeypatch.setattr(worker, "_BOOT_BUDGET_ENTRY_NS", 123)
    args = SimpleNamespace(vm_provider="docker", vm_path=tmp_path / "base", output=tmp_path / "run")
    with worker.owned_vm_context(args, "chenbofan") as metadata:
        assert metadata == {"actual_provider": "docker", "network_mode": "docker_bridge_loopback"}
    assert calls == [{"owner": "chenbofan"}]
    args.owned_boot_budget, args.startup_timeout = SHARED_BOOT_BUDGET, 600
    with worker.owned_vm_context(args, "chenbofan") as metadata:
        assert metadata["owned_boot_wait_budget"] == {"version": SHARED_BOOT_BUDGET}
    assert calls[1] == (600, 123) and isinstance(calls[2]["boot_wait_budget"], Budget)
