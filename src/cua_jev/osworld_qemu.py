"""Private, rootless QEMU provider for an explicitly owned OSWorld base image.

Importing this module never imports Docker or starts a process. Linux-only checks
are deferred until the context is entered; unsupported isolation fails closed.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import signal
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

QEMU = "/usr/bin/qemu-system-x86_64"
QEMU_IMG = "/usr/bin/qemu-img"
OVMF_CODE = "/usr/share/OVMF/OVMF_CODE.fd"
OVMF_VARS = "/usr/share/OVMF/OVMF_VARS.fd"
GUEST_PORTS = (5000, 9222, 8080)
GUEST_RAM_MIB = 2048
MAX_LOG_BYTES = 1_000_000
MAX_IMAGE_BYTES = 128 * 1024**3
MAX_STATUS_BYTES = 16_384
_CONTEXT_LOCK = threading.Lock()

# Run in a fresh interpreter, not preexec_fn (the worker may already have threads).
# Any failed limit, affinity, prctl, or exec is fatal; there is no relaxed retry.
_CHILD_WRAPPER = r"""
import ctypes,json,os,resource,signal,sys
parent=int(sys.argv[1]); cpus=json.loads(sys.argv[2]); command=json.loads(sys.argv[3])
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(1,signal.SIGKILL,0,0,0)!=0: raise RuntimeError('pdeathsig_failed')
if os.getppid()!=parent: raise RuntimeError('parent_changed')
if libc.prctl(38,1,0,0,0)!=0: raise RuntimeError('no_new_privs_failed')
resource.setrlimit(resource.RLIMIT_AS,(6*1024**3,6*1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**3,2*1024**3))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
os.sched_setaffinity(0,set(cpus)); os.nice(10); os.umask(0o077)
os.execve(command[0],command,dict(os.environ))
"""


class OwnedQemuError(RuntimeError):
    """A static error code, safe to include in experiment diagnostics."""

    def __init__(self, code: str):
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", code)
            else ("qemu_provider_error")
        )
        self.diagnostics = {"vm_provider": "qemu", "code": self.code}
        super().__init__(self.code)


def _expected_uid(expected_user: str) -> int:
    if not sys.platform.startswith("linux"):
        raise OwnedQemuError("qemu_requires_linux")
    import pwd

    if not isinstance(expected_user, str) or not expected_user or len(expected_user) > 64:
        raise OwnedQemuError("invalid_expected_user")
    try:
        uid = pwd.getpwnam(expected_user).pw_uid
    except KeyError:
        raise OwnedQemuError("unknown_expected_user") from None
    if uid == 0 or os.getuid() != uid or os.geteuid() != uid:
        raise OwnedQemuError("qemu_owner_mismatch")
    return uid


def _owned_path(path: str | Path, uid: int, *, directory: bool) -> Path:
    candidate = Path(path).absolute()
    try:
        resolved = candidate.resolve(strict=True)
        # Reject aliases/symlinks, including ancestors, so the caller must pass
        # its already-resolved owned location rather than an ambiguous alias.
        if candidate != resolved:
            raise OwnedQemuError("qemu_path_not_canonical")
        info = candidate.lstat()
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if not expected_type(info.st_mode) or info.st_uid != uid:
            raise OwnedQemuError("qemu_path_not_owned")
        if not directory and info.st_nlink != 1:
            raise OwnedQemuError("qemu_base_has_hardlinks")
    except OSError:
        raise OwnedQemuError("qemu_path_unavailable") from None
    return candidate


def _open_regular(path: Path, uid: int, *, writable: bool = False, create: bool = False) -> int:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | os.O_NOFOLLOW | os.O_CLOEXEC
    if create:
        flags |= os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_nlink != 1:
        os.close(fd)
        raise OwnedQemuError("qemu_file_not_owned_regular")
    return fd


def _clean_environment(directory: Path) -> dict[str, str]:
    # No credentials, proxy settings, PYTHONPATH, display or inherited QEMU config.
    return {
        "PATH": "/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "HOME": str(directory),
        "TMPDIR": str(directory),
        "MALLOC_ARENA_MAX": "2",
    }


def _limited_launch(command: list[str]) -> tuple[list[str], list[int]]:
    cpus = sorted(os.sched_getaffinity(0))[:2]
    if not cpus:
        raise OwnedQemuError("qemu_cpu_affinity_unavailable")
    return [
        sys.executable,
        "-I",
        "-c",
        _CHILD_WRAPPER,
        str(os.getpid()),
        json.dumps(cpus),
        json.dumps(command),
    ], cpus


def _validate_image_info(info: Any) -> int:
    if not isinstance(info, dict) or info.get("format") != "qcow2":
        raise OwnedQemuError("qemu_base_not_qcow2")
    size = info.get("virtual-size")
    if type(size) is not int or not 0 < size <= MAX_IMAGE_BYTES:
        raise OwnedQemuError("qemu_base_size_invalid")
    forbidden = {
        "backing-filename",
        "full-backing-filename",
        "backing-filename-format",
        "data-file",
        "data-file-raw",
        "external-data-file",
        "backing-image",
    }

    def inspect(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden and item not in (None, False, ""):
                    raise OwnedQemuError("qemu_base_has_external_storage")
                if key in {"dirty-flag", "corrupt"} and item is True:
                    raise OwnedQemuError("qemu_base_not_clean")
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(info)
    return size


def _validate_qcow_header(header: bytes) -> None:
    """Reject external files before asking qemu-img to open the image format."""
    if len(header) < 104 or header[:4] != b"QFI\xfb":
        raise OwnedQemuError("qemu_base_header_invalid")
    version = struct.unpack_from(">I", header, 4)[0]
    backing_offset, backing_size = struct.unpack_from(">QI", header, 8)
    if version not in (2, 3) or struct.unpack_from(">I", header, 32)[0] != 0:
        raise OwnedQemuError("qemu_base_header_unsupported")
    if backing_offset or backing_size:
        raise OwnedQemuError("qemu_base_has_external_storage")
    if version == 3:
        incompatible = struct.unpack_from(">Q", header, 72)[0]
        if incompatible & 4:
            raise OwnedQemuError("qemu_base_has_external_storage")
        if incompatible & 3:
            raise OwnedQemuError("qemu_base_not_clean")


def _process_identity(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19]  # field 22 (starttime), after pid and parenthesized comm
    except (OSError, IndexError):
        raise OwnedQemuError("qemu_process_identity_unavailable") from None


def _parse_process_status(raw: bytes) -> dict[str, int]:
    """Parse only bounded numeric memory/thread fields; discard all other text."""
    if not isinstance(raw, bytes):
        raise OwnedQemuError("qemu_status_invalid")
    if len(raw) > MAX_STATUS_BYTES:
        raise OwnedQemuError("qemu_status_too_large")
    result = {}
    for line in raw.splitlines():
        key = line.partition(b":")[0]
        if key not in {b"VmPeak", b"VmSize", b"VmRSS", b"Threads"}:
            continue
        pattern = (
            rb"Threads:[ \t]+([0-9]{1,19})[ \t]*"
            if key == b"Threads"
            else (rb"(?:VmPeak|VmSize|VmRSS):[ \t]+([0-9]{1,19})[ \t]+kB[ \t]*")
        )
        match = re.fullmatch(pattern, line)
        name = key.decode("ascii")
        if match is None or name in result or int(match[1]) > 2**63 - 1:
            raise OwnedQemuError("qemu_status_invalid")
        result[name] = int(match[1])
    if set(result) != {"VmPeak", "VmSize", "VmRSS", "Threads"}:
        raise OwnedQemuError("qemu_status_invalid")
    return result


def _read_process_status(pid: int) -> dict[str, int]:
    with Path(f"/proc/{pid}/status").open("rb") as stream:
        return _parse_process_status(stream.read(MAX_STATUS_BYTES + 1))


def _new_status_summary() -> dict[str, Any]:
    return {
        "sampling": "startup_poll",
        "units": {"VmPeak": "KiB", "VmSize": "KiB", "VmRSS": "KiB", "Threads": "count"},
        "samples": 0,
        "last": {},
        "maximum": {},
        "read_errors": 0,
        "last_error": None,
    }


def _listener_ports(pid: int, identity: str, ports: set[int]) -> set[int]:
    """Identify only the owned child's loopback listening socket inodes."""
    if _process_identity(pid) != identity:
        raise OwnedQemuError("qemu_process_identity_changed")
    inodes = set()
    try:
        for path in Path(f"/proc/{pid}/fd").iterdir():
            try:
                link = os.readlink(path)
            except FileNotFoundError:
                continue
            if link.startswith("socket:[") and link.endswith("]"):
                inodes.add(link[8:-1])
        owned = set()
        for line in Path(f"/proc/{pid}/net/tcp").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            address, hex_port = fields[1].split(":", 1)
            port = int(hex_port, 16)
            if address == "0100007F" and port in ports:
                owned.add(port)
    except (OSError, ValueError):
        raise OwnedQemuError("qemu_listener_ownership_unavailable") from None
    if _process_identity(pid) != identity:
        raise OwnedQemuError("qemu_process_identity_changed")
    return owned


def _reserve_ports(guest_ports=GUEST_PORTS) -> list[socket.socket]:
    reservations = []
    try:
        for _ in guest_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reservations.append(sock)
            sock.bind(("127.0.0.1", 0))
        return reservations
    except BaseException:
        for sock in reservations:
            sock.close()
        raise


def _qemu_command(fds: dict[str, int], ports: dict[int, int]) -> list[str]:
    if set(ports) not in (set(GUEST_PORTS), set(GUEST_PORTS) | {9292}):
        raise OwnedQemuError("qemu_guest_ports_invalid")

    def block(value: dict) -> list[str]:
        return ["-blockdev", json.dumps(value, separators=(",", ":"))]

    command = [
        QEMU,
        "-no-user-config",
        "-nodefaults",
        "-machine",
        "q35,accel=kvm",
        "-cpu",
        "host",
        "-smp",
        "2",
        "-m",
        str(GUEST_RAM_MIB),
        "-display",
        "none",
        "-monitor",
        "none",
        "-serial",
        "none",
        "-parallel",
        "none",
        "-nic",
        "none",
        "-no-reboot",
        "-sandbox",
        "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
        "-qmp",
        f"unix:/proc/self/fd/{fds['directory']}/qmp.sock,server=on,wait=off",
    ]
    for name in ("base", "overlay", "code", "vars"):
        command += block(
            {
                "driver": "file",
                "node-name": f"{name}_file",
                "filename": f"/proc/self/fd/{fds[name]}",
                "read-only": name in {"base", "code"},
                "locking": "on",
            }
        )
    command += block(
        {"driver": "qcow2", "node-name": "base", "file": "base_file", "read-only": True, "backing": None}
    )
    command += block(
        {
            "driver": "qcow2",
            "node-name": "overlay",
            "file": "overlay_file",
            "backing": "base",
            "read-only": False,
        }
    )
    for name in ("code", "vars"):
        command += block(
            {"driver": "raw", "node-name": name, "file": f"{name}_file", "read-only": name == "code"}
        )
    command += [
        "-machine",
        "pflash0=code,pflash1=vars",
        "-device",
        "virtio-blk-pci,drive=overlay",
        "-device",
        "VGA",
        "-netdev",
        "user,id=guest,restrict=on,"
        + ",".join(f"hostfwd=tcp:127.0.0.1:{host}-:{guest}" for guest, host in ports.items()),
        "-device",
        "e1000,netdev=guest",
    ]
    return command


class _NoopManager:
    """No registry, downloads, snapshots, or path mutations."""

    def get_vm_path(self, *args, **kwargs):
        raise OwnedQemuError("qemu_requires_explicit_base_path")

    def list_free_vms(self, *args, **kwargs):
        return []

    def initialize_registry(self, *args, **kwargs):
        return None

    add_vm = delete_vm = occupy_vm = check_and_clean = initialize_registry


class OwnedQemuProvider:
    """One private QEMU child at a time; never touches any shared VM service."""

    def __init__(self, base_path: str | Path, output_dir: str | Path, expected_user: str):
        self.uid = _expected_uid(expected_user)
        self.base_path = _owned_path(base_path, self.uid, directory=False)
        self.output_dir = _owned_path(output_dir, self.uid, directory=True)
        self.owned_ports: set[int] = set()
        self.metadata: dict[str, Any] = {
            "vm_provider": "qemu",
            "harness_provider_name": "docker",
            "machine": "q35",
            "firmware": "UEFI OVMF",
            "disk_bus": "virtio-blk-pci",
            "nic": "e1000",
            "network": "user,restrict=on",
            "bind_address": "127.0.0.1",
            "vnc": False,
            "guest_ram_mib": GUEST_RAM_MIB,
            "vcpus": 2,
            "rlimit_as_bytes": 6 * 1024**3,
            "rlimit_fsize_bytes": 2 * 1024**3,
            "rlimit_core_bytes": 0,
            "malloc_arena_max": 2,
            "sandbox": "on,obsolete=deny,elevateprivileges=deny,spawn=deny,resourcecontrol=deny",
            "cleanup_confirmed": True,
            "startup_error": None,
            "returncode": None,
            "process_status": _new_status_summary(),
            "starts": [],
        }
        self.startup_timeout_s = 180.0
        self._process: subprocess.Popen | None = None
        self._start_record: dict[str, Any] | None = None
        self._identity = ""
        self._pidfd: int | None = None
        self._fds: dict[str, int] = {}
        self._ports: dict[int, int] = {}
        self._run_dir: Path | None = None
        self._log_thread: threading.Thread | None = None
        self._closed = False
        self._guest_ports = GUEST_PORTS

    def enable_mcp_port(self) -> None:
        """Opt in before boot to one owned loopback forward for the guest MCP server."""
        if self._closed or self._process is not None or self.metadata["starts"]:
            raise OwnedQemuError("qemu_mcp_port_requires_fresh_provider")
        self._guest_ports = (*GUEST_PORTS, 9292)
        self.metadata["mcp_guest_port"] = 9292

    def _check_path(self, path_to_vm: str | Path) -> None:
        if Path(path_to_vm).absolute() != self.base_path:
            raise OwnedQemuError("qemu_base_path_changed")

    def _utility(self, args: list[str], fds: tuple[int, ...], directory: Path) -> subprocess.CompletedProcess:
        launch, _ = _limited_launch(args)
        try:
            result = subprocess.run(
                launch,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                check=False,
                shell=False,
                pass_fds=fds,
                cwd=directory,
                env=_clean_environment(directory),
            )
        except (OSError, subprocess.TimeoutExpired):
            raise OwnedQemuError("qemu_image_utility_failed") from None
        if result.returncode != 0 or len(result.stdout) > 2_000_000 or len(result.stderr) > MAX_LOG_BYTES:
            raise OwnedQemuError("qemu_image_utility_failed")
        return result

    def _prepare_files(self) -> None:
        if not os.access("/dev/kvm", os.R_OK | os.W_OK):
            raise OwnedQemuError("qemu_kvm_unavailable")
        # UUID directory is exclusive and stays inside the canonical owned root.
        directory = self.output_dir / f"qemu-{uuid.uuid4().hex}"
        directory.mkdir(mode=0o700, exist_ok=False)
        if directory.is_symlink() or directory.resolve() != directory:
            raise OwnedQemuError("qemu_run_directory_invalid")
        self._run_dir = directory
        self._fds["directory"] = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._fds["base"] = _open_regular(self.base_path, self.uid)
        _validate_qcow_header(os.pread(self._fds["base"], 104, 0))
        result = self._utility(
            [QEMU_IMG, "info", "--output=json", "-f", "qcow2", f"/proc/self/fd/{self._fds['base']}"],
            (self._fds["base"],),
            directory,
        )
        try:
            size = _validate_image_info(json.loads(result.stdout))
        except (ValueError, UnicodeError):
            raise OwnedQemuError("qemu_base_metadata_invalid") from None
        self._fds["overlay"] = _open_regular(
            directory / "overlay.qcow2", self.uid, writable=True, create=True
        )
        # The image has no filename-based backing chain. The runtime block graph
        # supplies the explicitly read-only base FD as its backing node.
        self._utility(
            [QEMU_IMG, "create", "-f", "qcow2", f"/proc/self/fd/{self._fds['overlay']}", str(size)],
            (self._fds["overlay"],),
            directory,
        )
        for name, path in (("code", OVMF_CODE), ("vars_template", OVMF_VARS)):
            firmware = Path(path)
            if firmware.resolve(strict=True) != firmware:
                raise OwnedQemuError("qemu_firmware_symlink")
            fd = _open_regular(firmware, 0)
            self._fds[name] = fd
            info = os.fstat(fd)
            if info.st_mode & 0o022 or not 0 < info.st_size <= 4 * 1024**2:
                raise OwnedQemuError("qemu_firmware_invalid")
        self._fds["vars"] = _open_regular(directory / "OVMF_VARS.fd", self.uid, writable=True, create=True)
        while chunk := os.read(self._fds["vars_template"], 65536):
            view = memoryview(chunk)
            while view:
                count = os.write(self._fds["vars"], view)
                if count <= 0:
                    raise OwnedQemuError("qemu_firmware_copy_failed")
                view = view[count:]
        os.close(self._fds.pop("vars_template"))
        self.metadata["base_virtual_size_bytes"] = size

    def _drain_log(self, stream, path: Path) -> None:
        retained = 0
        discarded = 0
        try:
            with path.open("xb") as log:
                while chunk := stream.read(8192):
                    keep = chunk[: max(0, MAX_LOG_BYTES - retained)]
                    if keep:
                        log.write(keep)
                        log.flush()
                        retained += len(keep)
                    discarded += len(chunk) - len(keep)
        except (OSError, ValueError):
            self.metadata["log_error"] = "qemu_log_unavailable"
            # Never stop draining merely because the bounded artifact failed.
            with contextlib.suppress(OSError, ValueError):
                while stream.read(8192):
                    pass
        finally:
            stream.close()
            self.metadata["log_discarded_bytes"] = discarded

    def start_emulator(self, path_to_vm: str | Path, headless: bool = True, os_type: str = "Ubuntu") -> None:
        self.metadata["startup_error"] = None
        try:
            self._check_path(path_to_vm)
            if self._closed or self._process is not None:
                raise OwnedQemuError("qemu_provider_not_idle")
            if os_type not in {"Ubuntu", "Linux"}:
                raise OwnedQemuError("qemu_guest_os_unsupported")
        except OwnedQemuError as error:
            self.metadata["startup_error"] = error.code
            raise
        reservations: list[socket.socket] = []
        self.metadata["cleanup_confirmed"] = False
        self.metadata["returncode"] = None
        self.metadata["process_status"] = _new_status_summary()
        self._start_record = None
        try:
            self._prepare_files()
            assert self._run_dir is not None
            reservations = _reserve_ports(self._guest_ports)
            self._ports = dict(
                zip(self._guest_ports, (sock.getsockname()[1] for sock in reservations), strict=True)
            )
            command = _qemu_command(self._fds, self._ports)
            launch, cpus = _limited_launch(command)
            for sock in reservations:
                sock.close()
            reservations = []
            self._process = subprocess.Popen(
                launch,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self._run_dir,
                env=_clean_environment(self._run_dir),
                shell=False,
                pass_fds=tuple(self._fds.values()),
                close_fds=True,
            )
            # Register the held child before identity probing: a fast startup
            # exit must still retain its return code in this exact attempt.
            self._start_record = {
                "directory": str(self._run_dir),
                "pid": self._process.pid,
                "starttime": None,
                "ports": dict(self._ports),
                "returncode": None,
                "process_status": self.metadata["process_status"],
            }
            self.metadata["starts"].append(self._start_record)
            if hasattr(os, "pidfd_open"):
                self._pidfd = os.pidfd_open(self._process.pid)
            self._identity = _process_identity(self._process.pid)
            self.metadata["cpu_affinity"] = cpus
            self._start_record["starttime"] = self._identity
            self._log_thread = threading.Thread(
                target=self._drain_log, args=(self._process.stdout, self._run_dir / "qemu.log"), daemon=True
            )
            self._log_thread.start()
            self._wait_ready()
        except BaseException as error:
            self.metadata["startup_error"] = (
                str(error) if isinstance(error, OwnedQemuError) else type(error).__name__
            )
            for sock in reservations:
                sock.close()
            self.stop_emulator(self.base_path)
            raise

    def _wait_ready(self) -> None:
        import requests

        deadline = time.monotonic() + self.startup_timeout_s
        session = requests.Session()
        session.trust_env = False
        try:
            while time.monotonic() < deadline:
                process = self._process
                if process is None or process.poll() is not None:
                    raise OwnedQemuError("qemu_exited_during_startup")
                self._sample_process_status()
                if process.poll() is not None:
                    raise OwnedQemuError("qemu_exited_during_startup")
                ports = set(self._ports.values())
                if _listener_ports(process.pid, self._identity, ports) == ports:
                    # Populate the same mutable set held by bounded_vm_http.
                    self.owned_ports.update(ports)
                    try:
                        response = session.get(
                            f"http://127.0.0.1:{self._ports[5000]}/screenshot",
                            timeout=(2, 3),
                            stream=True,
                            allow_redirects=False,
                        )
                        try:
                            if response.status_code == 200:
                                if (
                                    process.poll() is None
                                    and _listener_ports(process.pid, self._identity, ports) == ports
                                ):
                                    self.metadata["ready"] = True
                                    return
                        finally:
                            response.close()
                    except requests.RequestException:
                        pass
                time.sleep(0.25)
            raise OwnedQemuError("qemu_boot_timeout")
        finally:
            session.close()

    def _sample_process_status(self) -> None:
        """Best-effort evidence from this held child only; never affects limits."""
        process = self._process
        if process is None or process.poll() is not None:
            return
        summary = self.metadata["process_status"]
        try:
            if not self._identity:
                raise OwnedQemuError("qemu_status_identity_unavailable")
            if _process_identity(process.pid) != self._identity:
                raise OwnedQemuError("qemu_status_identity_changed")
            if process.poll() is not None:
                return
            sample = _read_process_status(process.pid)
            # If this unreaped child exits during the read, the sample still
            # belongs to it. Retain it without another /proc read after exit.
            if process.poll() is None and _process_identity(process.pid) != self._identity:
                raise OwnedQemuError("qemu_status_identity_changed")
        except Exception as error:
            code = error.code if isinstance(error, OwnedQemuError) else "qemu_status_read_failed"
            if code == "qemu_process_identity_unavailable":
                code = "qemu_status_identity_unavailable"
            if code not in {
                "qemu_status_too_large",
                "qemu_status_invalid",
                "qemu_status_read_failed",
                "qemu_status_identity_unavailable",
                "qemu_status_identity_changed",
            }:
                code = "qemu_status_read_failed"
            summary["read_errors"] += 1
            summary["last_error"] = code
            return
        summary["samples"] += 1
        summary["last"] = sample
        for key, value in sample.items():
            summary["maximum"][key] = max(value, summary["maximum"].get(key, value))

    def get_ip_address(self, path_to_vm: str | Path) -> str:
        self._check_path(path_to_vm)
        if self._process is None or self._process.poll() is not None or not self.metadata.get("ready"):
            raise OwnedQemuError("qemu_not_ready")
        ports = set(self._ports.values())
        if _listener_ports(self._process.pid, self._identity, ports) != ports:
            raise OwnedQemuError("qemu_listener_ownership_lost")
        return f"127.0.0.1:{self._ports[5000]}:{self._ports[9222]}:0:{self._ports[8080]}"

    def _qmp_quit(self) -> None:
        if "directory" not in self._fds:
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(1)
            connection.connect(f"/proc/self/fd/{self._fds['directory']}/qmp.sock")
            buffer = bytearray()

            def receive() -> dict:
                while b"\n" not in buffer:
                    chunk = connection.recv(4096)
                    if not chunk or len(buffer) + len(chunk) > 65536:
                        raise OwnedQemuError("qemu_qmp_invalid")
                    buffer.extend(chunk)
                line, _, rest = buffer.partition(b"\n")
                buffer[:] = rest
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise OwnedQemuError("qemu_qmp_invalid")
                return value

            if "QMP" not in receive():
                raise OwnedQemuError("qemu_qmp_invalid")
            connection.sendall(b'{"execute":"qmp_capabilities","id":"capabilities"}\n')
            for _ in range(8):
                reply = receive()
                if reply.get("id") == "capabilities":
                    if "error" in reply:
                        raise OwnedQemuError("qemu_qmp_invalid")
                    break
            else:
                raise OwnedQemuError("qemu_qmp_invalid")
            connection.sendall(b'{"execute":"quit","id":"quit"}\n')

    def _signal_owned(self, signum: int) -> None:
        process = self._process
        if process is None or process.poll() is not None:
            return
        if self._identity and _process_identity(process.pid) != self._identity:
            raise OwnedQemuError("qemu_process_identity_changed")
        if self._pidfd is not None and hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(self._pidfd, signum)
        else:
            # Before identity registration succeeds, this is still our exact
            # freshly-created, unreaped child. Popen.send_signal polls/reaps it
            # before signaling; an unreaped child PID cannot have been reused.
            # A known identity mismatch above never enters this fallback.
            process.send_signal(signum)

    def stop_emulator(self, path_to_vm: str | Path, *args, **kwargs) -> None:
        self._check_path(path_to_vm)
        process = self._process
        try:
            self._stop_held_process(process)
        finally:
            if process is not None:
                # None remains unknown/alive, never a successful exit. Keep
                # the actual Popen result even when resource cleanup raises.
                self.metadata["returncode"] = process.returncode
                if self._start_record is not None:
                    self._start_record["returncode"] = process.returncode

    def _stop_held_process(self, process: subprocess.Popen | None) -> None:
        if process is not None and process.poll() is None:
            if self._identity and _process_identity(process.pid) == self._identity:
                with contextlib.suppress(OSError, ValueError, OwnedQemuError):
                    self._qmp_quit()
            for signum, seconds in ((None, 3), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                if process.poll() is not None:
                    break
                if signum is not None:
                    self._signal_owned(signum)
                try:
                    process.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    continue
            if process.poll() is None:
                self.metadata["cleanup_confirmed"] = False
                raise OwnedQemuError("qemu_cleanup_unconfirmed")
        if self._log_thread is not None:
            self._log_thread.join(timeout=2)
            self._log_thread = None
        elif process is not None and process.stdout is not None:
            process.stdout.close()
        if self._pidfd is not None:
            os.close(self._pidfd)
            self._pidfd = None
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()
        self._process = None
        self._identity = ""
        self.owned_ports.clear()
        self._ports.clear()
        self.metadata["ready"] = False
        self.metadata["cleanup_confirmed"] = True
        # Preserve bounded logs and the private overlay for diagnosis. Never
        # remove paths outside this run or delete an image still in use.

    def revert_to_snapshot(self, path_to_vm: str | Path, snapshot_name: str) -> None:
        self.stop_emulator(path_to_vm)

    def save_state(self, path_to_vm: str | Path, snapshot_name: str):
        raise OwnedQemuError("qemu_save_state_unsupported")

    def close(self) -> None:
        self.stop_emulator(self.base_path)
        self._closed = True


@contextlib.contextmanager
def isolated_qemu(base_path: str | Path, output_dir: str | Path, expected_user: str):
    """Patch only the already-imported factory in this worker, then restore it.

    Yield the provider itself; ``runtime.provider`` is a compatibility alias.
    Its ``metadata`` dictionary and ``owned_ports`` set keep their identities.
    """
    if not _CONTEXT_LOCK.acquire(blocking=False):
        raise OwnedQemuError("qemu_context_already_active")
    provider = None
    module = None
    original = None
    try:
        provider = OwnedQemuProvider(base_path, output_dir, expected_user)
        module = importlib.import_module("desktop_env.desktop_env")
        original = module.create_vm_manager_and_provider

        def factory(provider_name, region=None, use_proxy=False):
            if provider_name != "docker" or use_proxy:
                raise OwnedQemuError("qemu_harness_configuration_invalid")
            return _NoopManager(), provider

        module.create_vm_manager_and_provider = factory
        provider.provider = provider
        yield provider
    finally:
        if module is not None and original is not None:
            module.create_vm_manager_and_provider = original
        try:
            if provider is not None:
                provider.close()
        finally:
            _CONTEXT_LOCK.release()
