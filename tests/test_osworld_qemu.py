"""No VM, Docker, SSH, QEMU executable, or real HTTP is used by these tests."""

from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cua_jev import osworld_qemu as q


@pytest.fixture(autouse=True)
def linux_signal_constants(monkeypatch):
    # All child/process calls are fakes on the Windows development host.
    monkeypatch.setattr(q.signal, "SIGKILL", 9, raising=False)
    if not hasattr(q.os, "sched_getaffinity"):
        monkeypatch.setattr(q.os, "sched_getaffinity", lambda pid: {1, 2}, raising=False)


def header(*, version=3, backing_offset=0, backing_size=0, incompatible=0, encryption=0):
    value = bytearray(104)
    value[:4] = b"QFI\xfb"
    struct.pack_into(">I", value, 4, version)
    struct.pack_into(">QI", value, 8, backing_offset, backing_size)
    struct.pack_into(">I", value, 32, encryption)
    struct.pack_into(">Q", value, 72, incompatible)
    return bytes(value)


@pytest.mark.parametrize("data", [header(version=2), header(), header(incompatible=16)])
def test_standalone_qcow_header(data):
    q._validate_qcow_header(data)


@pytest.mark.parametrize(
    "data, reason",
    [
        (b"bad", "header_invalid"),
        (header(version=4), "header_unsupported"),
        (header(encryption=1), "header_unsupported"),
        (header(backing_offset=104), "external_storage"),
        (header(backing_size=10), "external_storage"),
        (header(incompatible=4), "external_storage"),
        (header(incompatible=1), "not_clean"),
        (header(incompatible=2), "not_clean"),
    ],
)
def test_header_rejects_external_storage_before_format_open(data, reason):
    with pytest.raises(q.OwnedQemuError, match=reason):
        q._validate_qcow_header(data)


@pytest.mark.parametrize(
    "extra",
    [
        {"backing-filename": "/someone/else/image"},
        {"full-backing-filename": "/other/image"},
        {"format-specific": {"data": {"data-file": "/other/file"}}},
        {"format-specific": {"data": {"data-file-raw": True}}},
        {"backing-image": {"filename": "other"}},
        {"dirty-flag": True},
        {"format-specific": {"data": {"corrupt": True}}},
    ],
)
def test_image_metadata_rejects_chains_and_dirty_images(extra):
    with pytest.raises(q.OwnedQemuError):
        q._validate_image_info({"format": "qcow2", "virtual-size": 50 * 1024**3, **extra})


@pytest.mark.parametrize("size", [None, True, -1, 0, 129 * 1024**3, "50G"])
def test_image_metadata_size_is_bounded(size):
    with pytest.raises(q.OwnedQemuError, match="size_invalid"):
        q._validate_image_info({"format": "qcow2", "virtual-size": size})


def test_image_metadata_accepts_owned_format_only():
    assert q._validate_image_info({"format": "qcow2", "virtual-size": 50 * 1024**3}) == 50 * 1024**3
    with pytest.raises(q.OwnedQemuError, match="not_qcow2"):
        q._validate_image_info({"format": "raw", "virtual-size": 50 * 1024**3})


def test_environment_has_no_inherited_secrets_or_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", "private-test-token")
    monkeypatch.setenv("HTTP_PROXY", "http://private-proxy")
    monkeypatch.setenv("LD_PRELOAD", "unsafe")
    monkeypatch.setenv("MALLOC_ARENA_MAX", "100")
    environment = q._clean_environment(tmp_path)
    assert set(environment) == {"PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "MALLOC_ARENA_MAX"}
    assert environment["MALLOC_ARENA_MAX"] == "2" and os.environ["MALLOC_ARENA_MAX"] == "100"
    assert "private-test-token" not in json.dumps(environment)
    assert "http://private-proxy" not in json.dumps(environment)


def test_block_graph_and_network_are_explicit_and_restricted():
    fds = dict(base=10, overlay=11, code=12, vars=13, directory=14)
    ports = {5000: 14001, 9222: 14002, 8080: 14003}
    command = q._qemu_command(fds, ports)
    blocks = [json.loads(command[i + 1]) for i, arg in enumerate(command) if arg == "-blockdev"]
    by_name = {block["node-name"]: block for block in blocks}
    assert by_name["base_file"]["filename"] == "/proc/self/fd/10"
    assert by_name["base_file"]["read-only"] and by_name["base"]["read-only"]
    assert by_name["base"]["backing"] is None
    assert by_name["overlay"]["backing"] == "base"
    assert by_name["code_file"]["read-only"] and by_name["code"]["read-only"]
    net = command[command.index("-netdev") + 1]
    assert net == (
        "user,id=guest,restrict=on,hostfwd=tcp:127.0.0.1:14001-:5000,"
        "hostfwd=tcp:127.0.0.1:14002-:9222,hostfwd=tcp:127.0.0.1:14003-:8080"
    )
    assert "q35,accel=kvm" in command and command[command.index("-m") + 1] == "2048"
    assert "unix:/proc/self/fd/14/qmp.sock,server=on,wait=off" in command
    assert not set(command) & {"-daemonize", "-vnc", "-gdb", "-virtfs", "-fsdev", "-enable-kvm"}
    assert not any("tcg" in part or "hostfwd=tcp:0.0.0.0" in part for part in command)
    assert command[command.index("-monitor") + 1] == "none"
    assert command[command.index("-sandbox") + 1].endswith("spawn=deny,resourcecontrol=deny")


def test_reservations_are_distinct_loopback_and_closeable():
    sockets = q._reserve_ports()
    try:
        assert len(sockets) == 3
        addresses = [sock.getsockname() for sock in sockets]
        assert all(host == "127.0.0.1" and port > 1024 for host, port in addresses)
        assert len(set(addresses)) == 3
    finally:
        for sock in sockets:
            sock.close()


def test_mcp_port_keeps_restricted_network_and_all_forwards_loopback():
    fds = dict(base=10, overlay=11, code=12, vars=13, directory=14)
    ports = {5000: 14001, 9222: 14002, 8080: 14003, 9292: 14004}
    command = q._qemu_command(fds, ports)
    net = command[command.index("-netdev") + 1]
    assert net.startswith("user,id=guest,restrict=on,")
    assert net.endswith("hostfwd=tcp:127.0.0.1:14004-:9292")
    assert net.count("hostfwd=tcp:127.0.0.1:") == 4
    with pytest.raises(q.OwnedQemuError, match="ports_invalid"):
        q._qemu_command(fds, {**ports, 22: 14005})


def test_mcp_port_is_opt_in_and_cannot_be_added_after_start(provider):
    assert provider._guest_ports == q.GUEST_PORTS
    provider.enable_mcp_port()
    assert provider._guest_ports == (*q.GUEST_PORTS, 9292)
    provider.metadata["starts"].append({"pid": 123})
    with pytest.raises(q.OwnedQemuError, match="fresh_provider"):
        provider.enable_mcp_port()


def test_owned_path_rejects_foreign_owner_and_hardlinks(tmp_path):
    base = tmp_path / "base.qcow2"
    base.write_bytes(header())
    uid = base.stat().st_uid
    assert q._owned_path(base, uid, directory=False) == base
    with pytest.raises(q.OwnedQemuError, match="not_owned"):
        q._owned_path(base, uid + 1, directory=False)
    alias = tmp_path / "hardlink.qcow2"
    os.link(base, alias)
    with pytest.raises(q.OwnedQemuError, match="hardlinks"):
        q._owned_path(base, uid, directory=False)


def test_owned_path_rejects_symlinks(tmp_path):
    base = tmp_path / "base.qcow2"
    base.write_bytes(header())
    alias = tmp_path / "alias.qcow2"
    try:
        alias.symlink_to(base)
    except OSError:
        pytest.skip("Host does not permit symlink creation")
    with pytest.raises(q.OwnedQemuError, match="not_canonical"):
        q._owned_path(alias, base.stat().st_uid, directory=False)


def test_owner_check_rejects_root_wrong_user_and_non_linux(monkeypatch):
    monkeypatch.setattr(q.sys, "platform", "win32")
    with pytest.raises(q.OwnedQemuError, match="requires_linux"):
        q._expected_uid("owner")
    monkeypatch.setattr(q.sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=lambda user: SimpleNamespace(pw_uid=42)))
    monkeypatch.setattr(q.os, "getuid", lambda: 42, raising=False)
    monkeypatch.setattr(q.os, "geteuid", lambda: 42, raising=False)
    assert q._expected_uid("owner") == 42
    monkeypatch.setattr(q.os, "geteuid", lambda: 43, raising=False)
    with pytest.raises(q.OwnedQemuError, match="owner_mismatch"):
        q._expected_uid("owner")
    monkeypatch.setitem(sys.modules, "pwd", SimpleNamespace(getpwnam=lambda user: SimpleNamespace(pw_uid=0)))
    with pytest.raises(q.OwnedQemuError, match="owner_mismatch"):
        q._expected_uid("root")


def test_child_wrapper_sets_hard_limits_before_exec(monkeypatch):
    events = []

    class Executed(Exception):
        pass

    fake_os = SimpleNamespace(
        getppid=lambda: 321,
        environ={"PATH": "/usr/bin:/bin"},
        sched_setaffinity=lambda pid, cpus: events.append(("affinity", cpus)),
        nice=lambda amount: events.append(("nice", amount)),
        umask=lambda value: events.append(("umask", value)),
        execve=lambda *args: (_ for _ in ()).throw(Executed()),
    )
    fake_resource = SimpleNamespace(
        RLIMIT_AS=1,
        RLIMIT_FSIZE=2,
        RLIMIT_CORE=3,
        setrlimit=lambda key, value: events.append(("limit", key, value)),
    )
    monkeypatch.setitem(sys.modules, "os", fake_os)
    monkeypatch.setitem(sys.modules, "resource", fake_resource)
    monkeypatch.setitem(
        sys.modules,
        "ctypes",
        SimpleNamespace(
            CDLL=lambda *args, **kwargs: SimpleNamespace(
                prctl=lambda *args: events.append(("prctl", args)) or 0
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "sys", SimpleNamespace(argv=["wrapper", "321", "[2,4]", '["/qemu"]']))
    with pytest.raises(Executed):
        exec(q._CHILD_WRAPPER, {})
    assert ("limit", 1, (6 * 1024**3, 6 * 1024**3)) in events
    assert ("limit", 2, (2 * 1024**3, 2 * 1024**3)) in events
    assert ("limit", 3, (0, 0)) in events
    assert ("affinity", {2, 4}) in events and ("nice", 10) in events
    assert events[0][0] == "prctl" and events[1][0] == "prctl"


@pytest.fixture
def provider(tmp_path, monkeypatch):
    base = tmp_path / "base.qcow2"
    base.write_bytes(header())
    monkeypatch.setattr(q, "_expected_uid", lambda user: base.stat().st_uid)
    # Fake process PIDs below must never cause an actual /proc read, even when
    # the test suite is run on Linux. Individual sampling tests replace this.
    monkeypatch.setattr(q, "_read_process_status", lambda pid: (_ for _ in ()).throw(FileNotFoundError()))
    return q.OwnedQemuProvider(base, tmp_path, "owner")


def test_guest_memory_has_headroom_without_relaxing_host_limits(provider):
    command = q._qemu_command(
        dict(base=10, overlay=11, code=12, vars=13, directory=14), {5000: 14001, 9222: 14002, 8080: 14003}
    )
    assert command[command.index("-m") + 1] == "2048"
    assert provider.metadata["guest_ram_mib"] == 2048
    assert command[command.index("-smp") + 1] == "2"
    assert provider.metadata["vcpus"] == 2
    assert provider.metadata["rlimit_as_bytes"] == 6 * 1024**3
    assert provider.metadata["rlimit_fsize_bytes"] == 2 * 1024**3
    assert provider.metadata["rlimit_core_bytes"] == 0


class FakeProcess:
    def __init__(self, *, hang=False):
        self.pid = 54321
        self.stdout = io.BytesIO(b"fake bounded QEMU output\n")
        self.returncode = None
        self.hang = hang
        self.signals = []

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        if not self.hang:
            self.returncode = -signum


def attach_process(provider, monkeypatch, *, hang=False):
    process = FakeProcess(hang=hang)
    provider._process = process
    provider._identity = "same-start"
    provider._start_record = {"pid": process.pid, "returncode": None}
    provider.metadata["starts"].append(provider._start_record)
    provider.metadata["cleanup_confirmed"] = False
    monkeypatch.setattr(q, "_process_identity", lambda pid: "same-start")
    monkeypatch.setattr(provider, "_qmp_quit", lambda: None)
    return process


def test_stop_signals_only_owned_process_and_is_idempotent(provider, monkeypatch):
    process = attach_process(provider, monkeypatch)
    provider.owned_ports.add(12345)
    ports_identity = provider.owned_ports
    provider.stop_emulator(provider.base_path)
    assert process.signals == [q.signal.SIGTERM]
    assert provider.metadata["cleanup_confirmed"] is True
    assert provider.metadata["returncode"] == -q.signal.SIGTERM
    assert provider.metadata["starts"][-1]["returncode"] == -q.signal.SIGTERM
    assert provider.owned_ports is ports_identity and not provider.owned_ports
    provider.close()
    provider.close()
    assert provider.metadata["returncode"] == -q.signal.SIGTERM


@pytest.mark.parametrize("returncode", [0, 1, -11])
def test_stop_preserves_already_exited_child_returncode(provider, monkeypatch, returncode):
    process = attach_process(provider, monkeypatch)
    previous = {"pid": 9876, "returncode": 7}
    provider.metadata["starts"].insert(0, previous)
    process.returncode = returncode
    provider.stop_emulator(provider.base_path)
    assert provider.metadata["returncode"] == returncode
    assert provider.metadata["starts"][-1]["returncode"] == returncode
    assert previous["returncode"] == 7
    assert process.signals == [] and provider.metadata["cleanup_confirmed"]
    assert provider._process is None
    provider.close()
    assert provider.metadata["returncode"] == returncode


def test_cleanup_unconfirmed_is_not_success(provider, monkeypatch):
    process = attach_process(provider, monkeypatch, hang=True)
    with pytest.raises(q.OwnedQemuError, match="cleanup_unconfirmed"):
        provider.close()
    assert provider.metadata["cleanup_confirmed"] is False
    assert process.signals == [q.signal.SIGTERM, q.signal.SIGKILL]
    assert provider._process is process
    assert provider.metadata["returncode"] is None
    assert provider.metadata["starts"][-1]["returncode"] is None


def test_stop_never_signals_reused_pid(provider, monkeypatch):
    process = attach_process(provider, monkeypatch)
    monkeypatch.setattr(q, "_process_identity", lambda pid: "different-start")
    with pytest.raises(q.OwnedQemuError, match="identity_changed"):
        provider.close()
    assert process.signals == []
    assert not provider.metadata["cleanup_confirmed"]


def test_revert_leaves_base_and_uses_next_new_start(provider, monkeypatch):
    before = provider.base_path.read_bytes()
    process = attach_process(provider, monkeypatch)
    assert provider.revert_to_snapshot(provider.base_path, "snapshot-name") is None
    assert process.returncode is not None
    assert provider._process is None and provider.base_path.read_bytes() == before
    with pytest.raises(q.OwnedQemuError, match="save_state_unsupported"):
        provider.save_state(provider.base_path, "snapshot")


def test_wrong_base_path_never_stops_owned_process(provider, monkeypatch):
    process = attach_process(provider, monkeypatch)
    with pytest.raises(q.OwnedQemuError, match="path_changed"):
        provider.stop_emulator(provider.output_dir / "different.qcow2")
    assert process.signals == []


def test_context_patches_exact_imported_alias_and_restores(provider, monkeypatch):
    def original(*args):
        return "original"

    module = SimpleNamespace(create_vm_manager_and_provider=original)
    monkeypatch.setattr(q.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(q, "OwnedQemuProvider", lambda *args: provider)
    with q.isolated_qemu(provider.base_path, provider.output_dir, "owner") as runtime:
        assert runtime.provider is provider and runtime.metadata is provider.metadata
        manager, actual = module.create_vm_manager_and_provider("docker", "region", False)
        assert actual is provider
        assert manager.add_vm("somebody-else-file", "region") is None
        assert manager.delete_vm("somebody-else-file", "region") is None
        with pytest.raises(q.OwnedQemuError, match="explicit_base"):
            manager.get_vm_path()
        with pytest.raises(q.OwnedQemuError, match="configuration_invalid"):
            module.create_vm_manager_and_provider("qemu", "region")
        with pytest.raises(q.OwnedQemuError, match="already_active"):
            with q.isolated_qemu(provider.base_path, provider.output_dir, "owner"):
                pytest.fail("nested context must not enter")
    assert module.create_vm_manager_and_provider is original
    assert provider.metadata["cleanup_confirmed"]


def test_context_restores_factory_even_when_cleanup_fails(provider, monkeypatch):
    def original(*args):
        return "original"

    module = SimpleNamespace(create_vm_manager_and_provider=original)
    monkeypatch.setattr(q.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(q, "OwnedQemuProvider", lambda *args: provider)
    monkeypatch.setattr(provider, "close", lambda: (_ for _ in ()).throw(q.OwnedQemuError("cleanup_failed")))
    with pytest.raises(q.OwnedQemuError, match="cleanup_failed"):
        with q.isolated_qemu(provider.base_path, provider.output_dir, "owner"):
            pass
    assert module.create_vm_manager_and_provider is original
    assert not q._CONTEXT_LOCK.locked()


def fake_http(monkeypatch, calls):
    class RequestError(Exception):
        pass

    class Session:
        trust_env = True

        def get(self, url, **kwargs):
            assert not self.trust_env
            calls.append((url, kwargs))
            return SimpleNamespace(status_code=200, close=lambda: None)

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules, "requests", SimpleNamespace(Session=Session, RequestException=RequestError)
    )


def test_ready_checks_port_ownership_before_http(provider, monkeypatch):
    attach_process(provider, monkeypatch)
    provider._ports = {5000: 15000, 9222: 19222, 8080: 18080}
    calls = []
    fake_http(monkeypatch, calls)
    checks = []
    monkeypatch.setattr(q, "_listener_ports", lambda pid, identity, ports: checks.append(ports) or ports)
    provider._wait_ready()
    assert len(checks) == 2
    assert provider.owned_ports == {15000, 19222, 18080}
    assert calls == [
        ("http://127.0.0.1:15000/screenshot", {"timeout": (2, 3), "stream": True, "allow_redirects": False})
    ]
    assert provider.get_ip_address(provider.base_path) == "127.0.0.1:15000:19222:0:18080"
    provider.close()


def test_readiness_never_contacts_unowned_port(provider, monkeypatch):
    attach_process(provider, monkeypatch)
    provider._ports = {5000: 15000, 9222: 19222, 8080: 18080}
    calls = []
    fake_http(monkeypatch, calls)
    monkeypatch.setattr(q, "_listener_ports", lambda *args: set())
    ticks = iter([0, 0, 1000])
    monkeypatch.setattr(q.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(q.time, "sleep", lambda delay: None)
    with pytest.raises(q.OwnedQemuError, match="boot_timeout"):
        provider._wait_ready()
    assert calls == [] and provider.owned_ports == set()


def test_ready_address_rechecks_ownership(provider, monkeypatch):
    attach_process(provider, monkeypatch)
    provider._ports = {5000: 15000, 9222: 19222, 8080: 18080}
    provider.metadata["ready"] = True
    monkeypatch.setattr(q, "_listener_ports", lambda *args: set())
    with pytest.raises(q.OwnedQemuError, match="ownership_lost"):
        provider.get_ip_address(provider.base_path)


def test_bounded_log_drain_retains_only_limit(provider, tmp_path):
    path = tmp_path / "bounded.log"
    source = io.BytesIO(b"x" * (q.MAX_LOG_BYTES + 30000))
    provider._drain_log(source, path)
    assert path.stat().st_size == q.MAX_LOG_BYTES
    assert provider.metadata["log_discarded_bytes"] == 30000 and source.closed


def test_start_uses_fresh_directory_and_bounded_wrapper(provider, monkeypatch, tmp_path):
    launched = []
    created = []
    monkeypatch.setattr(q.os, "sched_getaffinity", lambda pid: {5, 7, 9}, raising=False)
    monkeypatch.delattr(q.os, "pidfd_open", raising=False)
    monkeypatch.setattr(q, "_process_identity", lambda pid: "same-start")

    def prepare():
        directory = tmp_path / f"start-{len(created)}"
        directory.mkdir()
        created.append(directory)
        provider._run_dir = directory
        for name in ("directory", "base", "overlay", "code", "vars"):
            provider._fds[name] = os.open(provider.base_path, os.O_RDONLY)

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return FakeProcess()

    monkeypatch.setattr(provider, "_prepare_files", prepare)
    monkeypatch.setattr(q.subprocess, "Popen", launch)
    monkeypatch.setattr(provider, "_wait_ready", lambda: provider.metadata.update(ready=True))
    monkeypatch.setattr(provider, "_qmp_quit", lambda: None)
    for os_type in ("Ubuntu", "Linux"):
        provider.start_emulator(provider.base_path, os_type=os_type)
        provider.revert_to_snapshot(provider.base_path, "unused")
    assert len(created) == 2 and created[0] != created[1]
    for command, kwargs in launched:
        assert command[1:3] == ["-I", "-c"] and command[3] == q._CHILD_WRAPPER
        assert json.loads(command[5]) == [5, 7]
        assert kwargs["shell"] is False and kwargs["close_fds"] is True
        assert len(kwargs["pass_fds"]) == 5
        assert set(kwargs["env"]) == {"PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "MALLOC_ARENA_MAX"}
        assert kwargs["env"]["MALLOC_ARENA_MAX"] == "2"
    assert provider.metadata["cleanup_confirmed"] and len(provider.metadata["starts"]) == 2
    first, second = provider.metadata["starts"]
    assert first["process_status"] is not second["process_status"]
    assert second["process_status"] is provider.metadata["process_status"]


def test_startup_error_cleans_up_and_never_relaxes(provider, monkeypatch):
    calls = []
    monkeypatch.setattr(
        provider,
        "_prepare_files",
        lambda: (_ for _ in ()).throw(q.OwnedQemuError("qemu_base_has_external_storage")),
    )
    monkeypatch.setattr(q.subprocess, "Popen", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(q.OwnedQemuError, match="external_storage"):
        provider.start_emulator(provider.base_path)
    assert calls == []
    assert provider.metadata["startup_error"] == "qemu_base_has_external_storage"
    assert provider.metadata["cleanup_confirmed"]


def test_socket_ownership_requires_owned_inode_and_loopback(monkeypatch):
    proc = "/proc/54321"
    # A foreign process owns port 15000; an owned wildcard listener must not be
    # accepted either. Only the owned loopback listener on 18080 is valid.
    tcp = (
        "header\n"
        "0: 0100007F:3A98 00000000:0000 0A 0:0 0:0 0 42 0 222\n"
        "1: 00000000:4B16 00000000:0000 0A 0:0 0:0 0 42 0 111\n"
        "2: 0100007F:46A0 00000000:0000 0A 0:0 0:0 0 42 0 333\n"
    )
    paths = {
        f"{proc}/fd": SimpleNamespace(iterdir=lambda: ["fd10", "fd11", "fd12"]),
        f"{proc}/net/tcp": SimpleNamespace(read_text=lambda: tcp),
    }
    monkeypatch.setattr(q, "Path", lambda value: paths[value])
    monkeypatch.setattr(
        q.os,
        "readlink",
        lambda value: {"fd10": "socket:[111]", "fd11": "socket:[333]", "fd12": "/private/image"}[value],
    )
    monkeypatch.setattr(q, "_process_identity", lambda pid: "start")
    assert q._listener_ports(54321, "start", {15000, 19222, 18080}) == {18080}
    with pytest.raises(q.OwnedQemuError, match="identity_changed"):
        q._listener_ports(54321, "older-start", {18080})


def test_regular_base_fd_is_read_only_and_overlay_creation_exclusive(provider, monkeypatch):
    monkeypatch.setattr(q.os, "O_NOFOLLOW", getattr(q.os, "O_NOFOLLOW", 0), raising=False)
    monkeypatch.setattr(q.os, "O_CLOEXEC", getattr(q.os, "O_CLOEXEC", 0), raising=False)
    fd = q._open_regular(provider.base_path, provider.uid)
    try:
        with pytest.raises(OSError):
            os.write(fd, b"must not change base")
    finally:
        os.close(fd)
    overlay = provider.output_dir / "overlay.qcow2"
    fd = q._open_regular(overlay, provider.uid, writable=True, create=True)
    os.close(fd)
    with pytest.raises(FileExistsError):
        q._open_regular(overlay, provider.uid, writable=True, create=True)
    assert provider.base_path.read_bytes() == header()


def test_child_limit_failure_never_executes_or_retries(monkeypatch):
    events = []

    class LimitFailed(Exception):
        pass

    def set_limit(key, value):
        events.append((key, value))
        raise LimitFailed()

    fake_os = SimpleNamespace(
        getppid=lambda: 321, environ={}, execve=lambda *args: pytest.fail("limit failure must prevent exec")
    )
    monkeypatch.setitem(sys.modules, "os", fake_os)
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RLIMIT_AS=1, RLIMIT_FSIZE=2, RLIMIT_CORE=3, setrlimit=set_limit),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctypes",
        SimpleNamespace(CDLL=lambda *args, **kwargs: SimpleNamespace(prctl=lambda *args: 0)),
    )
    monkeypatch.setitem(sys.modules, "sys", SimpleNamespace(argv=["wrapper", "321", "[2,4]", '["/qemu"]']))
    with pytest.raises(LimitFailed):
        exec(q._CHILD_WRAPPER, {})
    assert events == [(1, (6 * 1024**3, 6 * 1024**3))]


def test_image_utility_has_bounded_timeout_and_no_shell(provider, monkeypatch):
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, b"{}", b"")

    monkeypatch.setattr(q.subprocess, "run", run)
    provider._utility([q.QEMU_IMG, "info"], (10,), provider.output_dir)
    assert calls[0][1]["timeout"] == 30
    assert calls[0][1]["shell"] is False and calls[0][1]["pass_fds"] == (10,)
    assert calls[0][1]["env"]["MALLOC_ARENA_MAX"] == "2"
    assert calls[0][0][3] == q._CHILD_WRAPPER
    assert json.loads(calls[0][0][-1]) == [q.QEMU_IMG, "info"]
    monkeypatch.setattr(
        q.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("image-tool", 30)),
    )
    with pytest.raises(q.OwnedQemuError, match="utility_failed"):
        provider._utility([q.QEMU_IMG, "info"], (10,), provider.output_dir)


def test_identity_registration_failure_still_cleans_exact_popen_child(provider, monkeypatch):
    process = FakeProcess()
    provider._process = process
    provider._identity = ""
    provider.metadata["cleanup_confirmed"] = False
    monkeypatch.setattr(
        q,
        "_process_identity",
        lambda pid: (_ for _ in ()).throw(q.OwnedQemuError("qemu_process_identity_unavailable")),
    )
    provider.stop_emulator(provider.base_path)
    assert process.signals == [q.signal.SIGTERM]
    assert provider.metadata["cleanup_confirmed"] is True
    assert provider._process is None


def test_fast_child_exit_before_identity_registration_keeps_attempt_code(provider, monkeypatch, tmp_path):
    process = FakeProcess()
    process.returncode = 1
    monkeypatch.delattr(q.os, "pidfd_open", raising=False)

    def prepare():
        provider._run_dir = tmp_path
        for name in ("directory", "base", "overlay", "code", "vars"):
            provider._fds[name] = os.open(provider.base_path, os.O_RDONLY)

    monkeypatch.setattr(provider, "_prepare_files", prepare)
    monkeypatch.setattr(q.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        q,
        "_process_identity",
        lambda pid: (_ for _ in ()).throw(q.OwnedQemuError("qemu_process_identity_unavailable")),
    )
    with pytest.raises(q.OwnedQemuError, match="identity_unavailable"):
        provider.start_emulator(provider.base_path)
    assert provider.metadata["returncode"] == 1
    assert len(provider.metadata["starts"]) == 1
    assert provider.metadata["starts"][0]["returncode"] == 1
    assert provider.metadata["starts"][0]["starttime"] is None
    assert provider.metadata["cleanup_confirmed"] is True
    assert provider._process is None and process.signals == []


def test_osworld_ubuntu_default_reaches_preparation(provider, monkeypatch):
    calls = []

    def prepare():
        calls.append("prepare")
        raise q.OwnedQemuError("qemu_test_preparation_stopped")

    monkeypatch.setattr(provider, "_prepare_files", prepare)
    with pytest.raises(q.OwnedQemuError, match="test_preparation_stopped"):
        provider.start_emulator(provider.base_path)
    assert calls == ["prepare"]
    assert provider.metadata["startup_error"] == "qemu_test_preparation_stopped"
    assert provider.metadata["cleanup_confirmed"]


def test_unsupported_guest_records_pre_start_diagnostics(provider, monkeypatch):
    monkeypatch.setattr(provider, "_prepare_files", lambda: pytest.fail("must reject before preparation"))
    with pytest.raises(q.OwnedQemuError) as caught:
        provider.start_emulator(provider.base_path, os_type="Windows")
    assert caught.value.code == "qemu_guest_os_unsupported"
    assert caught.value.diagnostics == {"vm_provider": "qemu", "code": "qemu_guest_os_unsupported"}
    assert provider.metadata["startup_error"] == "qemu_guest_os_unsupported"
    assert provider.metadata["starts"] == [] and provider.metadata["cleanup_confirmed"]


def test_error_diagnostics_never_include_arbitrary_private_text():
    error = q.OwnedQemuError("PRIVATE /home/other/token something failed")
    assert str(error) == "qemu_provider_error"
    assert error.diagnostics == {"vm_provider": "qemu", "code": "qemu_provider_error"}


def status_bytes(*, peak=6000, size=5000, rss=2000, threads=8):
    return (
        "Name:\tPRIVATE PROCESS NAME\nUid:\t123 123 123 123\n"
        f"VmPeak:\t {peak} kB\nVmSize:\t{size} kB\nVmRSS:\t{rss} kB\nThreads:\t{threads}\n"
    ).encode()


def test_status_parser_keeps_only_four_numeric_fields():
    parsed = q._parse_process_status(status_bytes())
    assert parsed == {"VmPeak": 6000, "VmSize": 5000, "VmRSS": 2000, "Threads": 8}
    assert "PRIVATE" not in json.dumps(parsed)
    assert q._parse_process_status(status_bytes(peak=0, size=0, rss=0, threads=1))["VmSize"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        "not bytes",
        b"",
        status_bytes().replace(b"2000 kB", b"2000 MB"),
        status_bytes().replace(b"2000 kB", b"-1 kB"),
        status_bytes().replace(b"2000 kB", b"2000.5 kB"),
        status_bytes().replace(b"2000 kB", b"9223372036854775808 kB"),
        status_bytes().replace(b"2000 kB", b"1234567890123456789012345 kB"),
        status_bytes().replace(b"Threads:\t8\n", b""),
        status_bytes().replace(b"Threads:\t8\n", b"Threads:\t8 count\n"),
        status_bytes() + b"VmSize:\t5000 kB\n",
    ],
)
def test_status_parser_rejects_malformed_or_ambiguous_fields(raw):
    with pytest.raises(q.OwnedQemuError, match="status_invalid"):
        q._parse_process_status(raw)


def test_status_reader_is_bounded_to_exact_requested_pid(monkeypatch):
    calls = []

    class BoundedStream(io.BytesIO):
        def read(self, size):
            calls.append(("read", size))
            return super().read(size)

    stream = BoundedStream(status_bytes())

    def path(value):
        calls.append(("path", value))
        return SimpleNamespace(open=lambda mode: calls.append(("open", mode)) or stream)

    monkeypatch.setattr(q, "Path", path)
    assert q._read_process_status(54321)["Threads"] == 8
    assert calls == [("path", "/proc/54321/status"), ("open", "rb"), ("read", q.MAX_STATUS_BYTES + 1)]
    assert stream.closed
    with pytest.raises(q.OwnedQemuError, match="status_too_large"):
        q._parse_process_status(status_bytes() + b"x" * q.MAX_STATUS_BYTES)


def test_status_sampling_tracks_last_and_observed_maxima_only(provider, monkeypatch):
    process = attach_process(provider, monkeypatch)
    samples = iter(
        [
            {"VmPeak": 6000, "VmSize": 5000, "VmRSS": 2000, "Threads": 8},
            {"VmPeak": 7000, "VmSize": 4000, "VmRSS": 3000, "Threads": 4},
        ]
    )
    calls = []
    monkeypatch.setattr(q, "_read_process_status", lambda pid: calls.append(pid) or next(samples))
    provider._sample_process_status()
    provider._sample_process_status()
    summary = provider.metadata["process_status"]
    assert calls == [process.pid, process.pid]
    assert summary["samples"] == 2 and summary["read_errors"] == 0
    assert summary["last"] == {"VmPeak": 7000, "VmSize": 4000, "VmRSS": 3000, "Threads": 4}
    assert summary["maximum"] == {"VmPeak": 7000, "VmSize": 5000, "VmRSS": 3000, "Threads": 8}
    assert summary["units"] == {"VmPeak": "KiB", "VmSize": "KiB", "VmRSS": "KiB", "Threads": "count"}
    assert summary["sampling"] == "startup_poll" and "history" not in summary


@pytest.mark.parametrize(
    "error,code",
    [
        (FileNotFoundError("PRIVATE PATH"), "qemu_status_read_failed"),
        (PermissionError("PRIVATE PATH"), "qemu_status_read_failed"),
        (RuntimeError("PRIVATE DETAILS"), "qemu_status_read_failed"),
        (q.OwnedQemuError("qemu_status_invalid"), "qemu_status_invalid"),
        (q.OwnedQemuError("qemu_status_too_large"), "qemu_status_too_large"),
        (q.OwnedQemuError("unrelated_error"), "qemu_status_read_failed"),
    ],
)
def test_failed_status_sample_preserves_prior_evidence_without_interrupting(
    provider, monkeypatch, error, code
):
    attach_process(provider, monkeypatch)
    sample = {"VmPeak": 6000, "VmSize": 5000, "VmRSS": 2000, "Threads": 8}
    monkeypatch.setattr(q, "_read_process_status", lambda pid: dict(sample))
    provider._sample_process_status()
    monkeypatch.setattr(q, "_read_process_status", lambda pid: (_ for _ in ()).throw(error))
    provider._sample_process_status()
    summary = provider.metadata["process_status"]
    assert summary["samples"] == 1 and summary["read_errors"] == 1
    assert summary["last"] == summary["maximum"] == sample
    assert summary["last_error"] == code and "PRIVATE" not in json.dumps(summary)


@pytest.mark.parametrize("identity", ["", "changed"])
def test_status_sampling_requires_known_matching_child_identity(provider, monkeypatch, identity):
    attach_process(provider, monkeypatch)
    provider._identity = identity
    monkeypatch.setattr(q, "_read_process_status", lambda pid: pytest.fail("must not read unverified PID"))
    provider._sample_process_status()
    summary = provider.metadata["process_status"]
    assert summary["samples"] == 0 and summary["read_errors"] == 1
    assert summary["last_error"] == (
        "qemu_status_identity_unavailable" if not identity else "qemu_status_identity_changed"
    )


def test_status_sampling_discards_live_identity_change_during_read(provider, monkeypatch):
    attach_process(provider, monkeypatch)
    identities = iter(["same-start", "different-start"])
    monkeypatch.setattr(q, "_process_identity", lambda pid: next(identities))
    monkeypatch.setattr(q, "_read_process_status", lambda pid: q._parse_process_status(status_bytes()))
    provider._sample_process_status()
    assert provider.metadata["process_status"]["samples"] == 0
    assert provider.metadata["process_status"]["last_error"] == "qemu_status_identity_changed"


@pytest.mark.parametrize("have_process", [False, True])
def test_status_sampling_never_reads_proc_after_child_exit(provider, monkeypatch, have_process):
    if have_process:
        process = attach_process(provider, monkeypatch)
        process.returncode = -6
    monkeypatch.setattr(q, "_process_identity", lambda pid: pytest.fail("no /proc after exit"))
    monkeypatch.setattr(q, "_read_process_status", lambda pid: pytest.fail("no /proc after exit"))
    provider._sample_process_status()
    assert provider.metadata["process_status"]["samples"] == 0
    assert provider.metadata["process_status"]["read_errors"] == 0


def test_startup_poll_preserves_final_valid_sample_if_child_exits_during_read(provider, monkeypatch):
    process = attach_process(provider, monkeypatch)
    provider._ports = {5000: 15000, 9222: 19222, 8080: 18080}
    identities = []
    monkeypatch.setattr(q, "_process_identity", lambda pid: identities.append(pid) or "same-start")
    sample = {"VmPeak": 6000, "VmSize": 5000, "VmRSS": 2000, "Threads": 8}

    def read_status(pid):
        process.returncode = -6
        return dict(sample)

    monkeypatch.setattr(q, "_read_process_status", read_status)
    monkeypatch.setattr(q, "_listener_ports", lambda *args: pytest.fail("do not inspect an exited child"))
    calls = []
    fake_http(monkeypatch, calls)
    with pytest.raises(q.OwnedQemuError, match="exited_during_startup"):
        provider._wait_ready()
    assert identities == [process.pid] and calls == []
    provider.stop_emulator(provider.base_path)
    assert provider.metadata["process_status"]["last"] == sample
    assert provider.metadata["process_status"]["maximum"] == sample
    assert provider.metadata["returncode"] == -6 and provider.metadata["cleanup_confirmed"]


def test_status_read_failure_does_not_block_healthy_startup(provider, monkeypatch):
    attach_process(provider, monkeypatch)
    provider._ports = {5000: 15000, 9222: 19222, 8080: 18080}
    monkeypatch.setattr(q, "_read_process_status", lambda pid: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(q, "_listener_ports", lambda pid, identity, ports: ports)
    calls = []
    fake_http(monkeypatch, calls)
    provider._wait_ready()
    assert provider.metadata["ready"] and len(calls) == 1
    assert provider.metadata["process_status"]["last_error"] == "qemu_status_read_failed"


def test_allocator_setting_does_not_relax_existing_limits(provider):
    assert provider.metadata["malloc_arena_max"] == 2
    assert provider.metadata["guest_ram_mib"] == 2048 and provider.metadata["vcpus"] == 2
    assert provider.metadata["rlimit_as_bytes"] == 6 * 1024**3
    assert provider.metadata["rlimit_fsize_bytes"] == 2 * 1024**3
    assert provider.metadata["rlimit_core_bytes"] == 0
    assert "RLIMIT_STACK" not in q._CHILD_WRAPPER
    command = q._qemu_command(
        dict(base=10, overlay=11, code=12, vars=13, directory=14), {5000: 15000, 9222: 19222, 8080: 18080}
    )
    assert not any("thread-pool" in arg for arg in command)
