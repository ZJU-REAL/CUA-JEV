import getpass
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cua_jev.webarena_instance import OwnedWebArenaSite, WebArenaEnvironmentError


class Docker:
    def __init__(self):
        self.calls = []
        self.resources = {}
        self.site = None
        self.tag_digest = "sha256:" + "a" * 64
        self.fail_create = None

    def __call__(self, argv, **kwargs):
        assert kwargs["capture_output"] and kwargs["check"] is False
        self.calls.append(argv)
        kind, action, *args = argv[1:]
        output = ""
        if (kind, action) == ("image", "inspect"):
            output = json.dumps([{"Id": self.tag_digest}])
        elif action == "create":
            identifier = f"{kind}-{len(self.calls)}"
            labels = dict(self.site.labels)
            self.resources[identifier] = {
                "Id": identifier, "Labels": labels, "Config": {"Labels": labels},
                "State": {"Running": False},
                "Name": args[args.index("--name") + 1] if kind == "container" else args[-1],
            }
            output = identifier
            if self.fail_create == kind:
                raise subprocess.TimeoutExpired(argv, 30)
        elif action == "inspect":
            resource = self.resources.get(args[0]) or next(
                (r for r in self.resources.values() if r["Name"] == args[0]), None)
            if resource is None:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="not found")
            output = json.dumps([resource])
        elif action == "start":
            self.resources[args[0]]["State"]["Running"] = True
            self.resources[args[0]]["NetworkSettings"] = {"Networks": {
                "owned": {"NetworkID": self.site.network_id, "IPAddress": "172.29.0.2"},
            }}
        elif action == "rm":
            del self.resources[args[-1]]
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")


@pytest.fixture
def site(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    docker = Docker()
    instance = OwnedWebArenaSite("postmill-populated-exposed-withimg", tmp_path / "instance",
                                expected_user=getpass.getuser(), docker_run=docker)
    docker.site = instance
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, url):
            assert url.startswith("http://127.0.0.1:")
            return SimpleNamespace(status_code=200)
    monkeypatch.setattr("cua_jev.webarena_instance.httpx.Client", Client)
    return instance, docker


def test_reset_recreates_only_owned_container_from_same_immutable_image(site):
    instance, docker = site
    first = instance.reset()
    first_id, network_id = instance.container_id, instance.network_id
    docker.tag_digest = "sha256:" + "b" * 64
    second = instance.reset()
    assert first["url"] == second["url"]
    assert first["image_id"] == second["image_id"] == "sha256:" + "a" * 64
    assert first_id not in docker.resources and instance.container_id != first_id
    assert instance.network_id == network_id
    assert sum(c[1:3] == ["image", "inspect"] for c in docker.calls) == 1
    created = [c for c in docker.calls if c[1:3] == ["container", "create"]]
    for command in created:
        assert "--privileged" not in command and "--volume" not in command
        assert command[command.index("--cpus") + 1] == "2"
        assert command[command.index("--memory") + 1] == "4g"
        assert command[command.index("--publish") + 1].startswith("127.0.0.1:")
    instance.close()
    assert not docker.resources
    before = list(docker.calls)
    instance.close()
    assert docker.calls == before
    assert json.loads((instance.directory / "instance.json").read_text())["events"][-1] == {
        "event": "closed", "cleanup_confirmed": True,
    }


def test_owned_loopback_proxy_binds_only_inspected_network_and_closes_on_reset(site, monkeypatch):
    instance, docker = site
    forwards = []

    class Forward:
        def __init__(self, host, port, *, listen_port):
            assert host == '172.29.0.2' and port == 80 and listen_port == instance.host_port
            self.closed = False
            forwards.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setattr('cua_jev.webarena_instance.LoopbackForward', Forward)
    instance.loopback_proxy = True
    instance.reset()
    instance.reset()
    assert len(forwards) == 2 and forwards[0].closed and not forwards[1].closed
    created = [c for c in docker.calls if c[1:3] == ['container', 'create']]
    assert all('--publish' not in c and '--expose' in c for c in created)
    assert any('--internal' in c for c in docker.calls if c[1:3] == ['network', 'create'])
    instance.close()
    assert all(f.closed for f in forwards) and not docker.resources


@pytest.mark.parametrize("operation", ["reset", "close"])
def test_changed_ownership_cannot_remove_or_restart_a_resource(site, operation):
    instance, docker = site
    instance.reset()
    docker.resources[instance.container_id]["Config"]["Labels"]["cua-jev.owner-uid"] = "other-user"
    before = len(docker.calls)
    with pytest.raises(WebArenaEnvironmentError, match="not_owned"):
        getattr(instance, operation)()
    assert all(c[2] == "inspect" for c in docker.calls[before:])
    assert instance.container_id in docker.resources


def test_directory_outside_user_home_and_existing_directory_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    with pytest.raises(WebArenaEnvironmentError, match="outside_user_home"):
        OwnedWebArenaSite("image", tmp_path / "outside", expected_user=getpass.getuser())
    (tmp_path / "home").mkdir()
    with pytest.raises(FileExistsError):
        OwnedWebArenaSite("image", tmp_path / "home", expected_user=getpass.getuser())


def test_wrong_user_is_rejected_before_docker_or_directory_creation(tmp_path):
    with pytest.raises(WebArenaEnvironmentError, match="owner_mismatch"):
        OwnedWebArenaSite("image", tmp_path / "run", expected_user="not-the-current-user")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("kind", ["network", "container"])
def test_lost_create_reply_is_reconciled_before_owned_cleanup(site, kind):
    instance, docker = site
    docker.fail_create = kind
    with pytest.raises(WebArenaEnvironmentError, match="transport_failed"):
        instance.reset()
    assert instance.pending_creates and docker.resources
    instance.close()
    assert not instance.pending_creates and not docker.resources
    assert instance.events[-1] == {"event": "closed", "cleanup_confirmed": True}


def test_unresolved_create_does_not_report_cleanup_success(site):
    instance, docker = site
    docker.fail_create = "network"
    with pytest.raises(WebArenaEnvironmentError):
        instance.reset()
    docker.resources.clear()
    with pytest.raises(WebArenaEnvironmentError, match="create_outcome_unknown"):
        instance.close()
    assert not instance.closed
    assert instance.events[-1]["cleanup_confirmed"] is False
