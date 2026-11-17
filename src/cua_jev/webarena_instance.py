"""Resource-bounded, separately named WebArena website instances.

Only containers/networks created by this object may be removed. Image downloads
and model credentials are deliberately outside this interface.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import socket
import subprocess
import time
import uuid
from pathlib import Path

import httpx

from .loopback_forward import LoopbackForward


class WebArenaEnvironmentError(RuntimeError):
    pass


class OwnedWebArenaSite:
    def __init__(self, image: str, directory: Path, *, expected_user: str,
                 container_port: int = 80, startup_timeout_s: float = 300,
                 command: tuple[str, ...] = (), docker_run=None, loopback_proxy: bool = False):
        if type(loopback_proxy) is not bool:
            raise ValueError("loopback_proxy must be boolean")
        if getpass.getuser() != expected_user or os.getuid() == 0:
            raise WebArenaEnvironmentError("website_owner_mismatch")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./:@-]{0,240}", image)
                or type(container_port) is not int or not 1 <= container_port <= 65535
                or not 0 < startup_timeout_s <= 600):
            raise ValueError("invalid website image, port or startup budget")
        directory = Path(directory).absolute()
        if directory.is_symlink() or not directory.resolve().is_relative_to(Path.home().resolve()):
            raise WebArenaEnvironmentError("website_directory_outside_user_home")
        directory.mkdir(parents=True, exist_ok=False, mode=0o700)
        if directory.stat().st_uid != os.getuid():
            raise WebArenaEnvironmentError("website_directory_not_owned")
        directory.chmod(0o700)
        self.directory = directory.resolve()
        self.image, self.image_id = image, None
        self.container_port, self.startup_timeout_s = container_port, startup_timeout_s
        self.command = tuple(command)
        self.loopback_proxy, self._forward = loopback_proxy, None
        self.uid = os.getuid()
        self.token = uuid.uuid4().hex
        self.name = f"cua-jev-wa-{self.uid}-{self.token[:12]}"
        self.labels = {
            "cua-jev.owner-uid": str(self.uid), "cua-jev.instance": self.token,
            "cua-jev.project-sha256": hashlib.sha256(str(self.directory).encode()).hexdigest(),
        }
        self._run = docker_run or subprocess.run
        self.network_id = self.container_id = None
        self.host_port = None
        self.episodes = 0
        self.closed = False
        self.events = []
        self.pending_creates = {}

    def _docker(self, *args, timeout=30):
        try:
            result = self._run(["docker", *args], check=False, capture_output=True,
                               text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            raise WebArenaEnvironmentError("website_docker_transport_failed") from None
        if result.returncode:
            raise WebArenaEnvironmentError("website_docker_command_failed")
        return result.stdout.strip()

    def _label_args(self):
        return [item for key, value in self.labels.items() for item in ("--label", f"{key}={value}")]

    def _inspect_owned(self, kind, identifier):
        data = json.loads(self._docker(kind, "inspect", identifier))
        if len(data) != 1 or data[0].get("Id") != identifier:
            raise WebArenaEnvironmentError("website_resource_identity_mismatch")
        labels = (data[0].get("Config", {}).get("Labels", {}) if kind == "container"
                  else data[0].get("Labels", {})) or {}
        if any(labels.get(key) != value for key, value in self.labels.items()):
            raise WebArenaEnvironmentError("website_resource_not_owned")
        return data[0]

    def _recover_pending_create(self, kind):
        name = self.pending_creates[kind]
        # A timed-out docker create may have succeeded. Inspect only the exact
        # random name we submitted; absence/error is not proof of cleanup.
        data = json.loads(self._docker(kind, "inspect", name))
        if (len(data) != 1 or data[0].get("Name", "").lstrip("/") != name
                or not isinstance(data[0].get("Id"), str) or not data[0]["Id"]):
            raise WebArenaEnvironmentError("website_create_outcome_unknown")
        identifier = data[0]["Id"]
        self._inspect_owned(kind, identifier)
        setattr(self, kind + "_id", identifier)
        del self.pending_creates[kind]
        self._record("create_recovered", kind=kind)

    def _create_owned(self, kind, name, *arguments):
        self.pending_creates[kind] = name
        self._record("create_submitted", kind=kind, name=name)
        identifier = self._docker(kind, "create", *arguments)
        self._inspect_owned(kind, identifier)
        setattr(self, kind + "_id", identifier)
        del self.pending_creates[kind]
        return identifier

    @property
    def url(self):
        if self.host_port is None:
            raise WebArenaEnvironmentError("website_not_started")
        return f"http://127.0.0.1:{self.host_port}"

    def reserve_address(self):
        if self.host_port is None:
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                self.host_port = reservation.getsockname()[1]
        return self.url

    def stop(self):
        if self._forward is not None:
            self._forward.close()
            self._forward = None
        if self.container_id is not None:
            self._inspect_owned("container", self.container_id)
            self._docker("container", "rm", "--force", "--volumes", self.container_id, timeout=60)
            self.container_id = None
            self._record("container_removed")

    def configure(self, arguments):
        """Trusted operator setup only; never offered to the agent or Jev."""
        if (self.container_id is None or not isinstance(arguments, (list, tuple))
                or not arguments or any(type(a) is not str for a in arguments)):
            raise ValueError("configuration needs a running owned site and argv strings")
        self._inspect_owned("container", self.container_id)
        self._docker("container", "exec", self.container_id,
                     *(a.replace("{url}", self.url) for a in arguments), timeout=180)
        self._record("operator_configuration_completed")

    def _record(self, event, **fields):
        self.events.append({"event": event, **fields})
        temporary = self.directory / "instance.json.tmp"
        temporary.write_text(json.dumps({
            "kind": "owned_webarena_docker_site", "image": self.image,
            "image_id": self.image_id, "owner_uid": self.uid, "labels": self.labels,
            "resources": {"cpus": 2, "memory_gib": 4, "pids_limit": 256,
                          "network": "private_internal_bridge", "publish": "loopback_only",
                          "loopback_tcp_forward": self.loopback_proxy},
            "container_id": self.container_id, "network_id": self.network_id,
            "pending_creates": self.pending_creates,
            "url": self.url if self.host_port else None, "events": self.events,
        }, indent=2) + "\n")
        temporary.replace(self.directory / "instance.json")

    def reset(self):
        """Fresh image state before every episode, including read-only tasks."""
        if self.closed:
            raise WebArenaEnvironmentError("website_instance_closed")
        began = time.perf_counter()
        self.stop()
        if self.image_id is None:
            data = json.loads(self._docker("image", "inspect", self.image))
            if len(data) != 1 or not re.fullmatch(r"sha256:[0-9a-f]{64}", data[0].get("Id", "")):
                raise WebArenaEnvironmentError("website_image_identity_unavailable")
            self.image_id = data[0]["Id"]
        if self.network_id is None:
            self._create_owned("network", self.name, "--internal", "--driver", "bridge",
                               *self._label_args(), self.name)
        self.reserve_address()
        self.episodes += 1
        # A port race fails at start; it never stops another service or broadens
        # the bind address. Reset reuses this private instance's original URL.
        self._create_owned(
            "container", f"{self.name}-{self.episodes}", "--name", f"{self.name}-{self.episodes}",
            *self._label_args(), "--network", self.network_id,
            *( ["--expose", str(self.container_port)] if self.loopback_proxy else
               ["--publish", f"127.0.0.1:{self.host_port}:{self.container_port}"] ),
            "--cpus", "2", "--memory", "4g", "--memory-swap", "4g", "--pids-limit", "256",
            "--security-opt", "no-new-privileges:true", "--restart", "no",
            "--log-driver", "json-file", "--log-opt", "max-size=2m", "--log-opt", "max-file=1",
            self.image_id, *self.command,
        )
        self._inspect_owned("container", self.container_id)
        self._record("created", episode=self.episodes)
        self._docker("container", "start", self.container_id)
        if self.loopback_proxy:
            self._inspect_owned("network", self.network_id)
            own_container = self._inspect_owned("container", self.container_id)
            networks = own_container.get("NetworkSettings", {}).get("Networks", {})
            attached = [item for item in networks.values() if item.get("NetworkID") == self.network_id]
            if len(networks) != 1 or len(attached) != 1 or not attached[0].get("IPAddress"):
                raise WebArenaEnvironmentError("website_private_network_binding_invalid")
            self._forward = LoopbackForward(attached[0]["IPAddress"], self.container_port,
                                            listen_port=self.host_port)
            self._record("owned_loopback_forward_started")
        deadline = began + self.startup_timeout_s
        with httpx.Client(trust_env=False, timeout=2, follow_redirects=False) as client:
            while time.perf_counter() < deadline:
                state = self._inspect_owned("container", self.container_id).get("State", {})
                if state.get("Running") is not True:
                    raise WebArenaEnvironmentError("website_container_exited")
                try:
                    response = client.get(self.url)
                    if 200 <= response.status_code < 400:
                        elapsed = (time.perf_counter() - began) * 1000
                        self._record("ready", episode=self.episodes, reset_wall_ms=elapsed)
                        return {"url": self.url, "reset_wall_ms": elapsed,
                                "image_id": self.image_id, "episode": self.episodes,
                                "container_id": self.container_id, "network_id": self.network_id}
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        self._record("startup_timeout", episode=self.episodes)
        raise WebArenaEnvironmentError("website_startup_timeout")

    def close(self):
        if self.closed:
            return
        for kind in tuple(self.pending_creates):
            try:
                self._recover_pending_create(kind)
            except Exception:
                self._record("cleanup_unconfirmed", kind=kind, cleanup_confirmed=False)
                raise WebArenaEnvironmentError("website_create_outcome_unknown") from None
        self.stop()
        if self.network_id is not None:
            self._inspect_owned("network", self.network_id)
            self._docker("network", "rm", self.network_id)
            self.network_id = None
        self.closed = True
        self._record("closed", cleanup_confirmed=True)
