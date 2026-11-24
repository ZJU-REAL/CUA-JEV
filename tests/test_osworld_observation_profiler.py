"""A timing diagnostic must neither add reads/inputs nor leak private observation fields."""

import base64
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "observation_profiler", Path(__file__).parents[1] / "scripts/osworld_observation_profiler.py",
)
profiler = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(profiler)


class Clock:
    now = 0.0

    def read(self):
        return self.now


class Controller:
    def __init__(self, clock):
        self.clock, self.reads = clock, []
        self.error = None

    def get_screenshot(self):
        self.reads.append("screenshot")
        self.clock.now += 0.25
        if self.error:
            raise self.error
        return b"same-image"

    def get_accessibility_tree(self):
        self.reads.append("tree")
        self.clock.now += 0.75
        return "<desktop/>"

    def get_terminal_output(self):
        raise AssertionError("the harness does not request terminal output")


class Environment:
    def __init__(self, controller):
        self.controller, self.observations = controller, 0

    def _get_obs(self):
        self.observations += 1
        return {"screenshot": self.controller.get_screenshot(),
                "accessibility_tree": self.controller.get_accessibility_tree(),
                "terminal": "PRIVATE terminal", "instruction": "PRIVATE duplicate"}

    def step(self, *_args, **_kwargs):
        pytest.fail("diagnostic may not send input")

    def evaluate(self):
        pytest.fail("diagnostic may not evaluate")


class ProtocolError(ValueError):
    pass


class BaseSession:
    def __init__(self, env):
        self.env, self.closed, self.evaluated = env, False, False

    def handle(self, request):
        assert request["method"] == "close"
        self.closed = True
        return None


def setup_session():
    clock = Clock()
    controller = Controller(clock)
    env = Environment(controller)

    def public(observation):
        return {"screenshot": base64.b64encode(observation["screenshot"]).decode(),
                "accessibility_tree": observation["accessibility_tree"]}

    worker = SimpleNamespace(WorkerSession=BaseSession, ProtocolError=ProtocolError,
                             public_observation=public)
    session = profiler.profiling_session(worker, clock=clock.read)(env)
    return session, controller


def test_exactly_one_harness_observation_and_two_original_reads_with_separate_timings():
    session, controller = setup_session()
    before = dict(vars(controller))
    packet = session.handle({"id": 1, "method": "profile_observe"})
    assert session.env.observations == 1 and controller.reads == ["screenshot", "tree"]
    assert vars(controller) == before  # The mutable reads list is the original object.
    d = packet["diagnostic"]
    assert d["worker_observation_ms"] == d["worker_total_ms"] == 1000
    assert d["controller_reads"] == {"screenshot": {"calls": 1, "wall_ms": 250},
                                      "accessibility_tree": {"calls": 1, "wall_ms": 750},
                                      "terminal": {"calls": 0, "wall_ms": 0}}
    assert base64.b64decode(packet["observation"]["screenshot"]) == b"same-image"
    assert d["screenshot_sha256"] == hashlib.sha256(b"same-image").hexdigest()
    assert d["accessibility_sha256"] == hashlib.sha256(b"<desktop/>").hexdigest()
    assert d["screenshot_bytes"] == 10 and d["accessibility_utf8_bytes"] == 10
    assert "PRIVATE" not in str(packet)


@pytest.mark.parametrize("packet", [
    {"id": 1, "method": "step"}, {"id": 1, "method": "evaluate"},
    {"id": 1, "method": "observe"}, {"id": 1, "method": "unknown"},
    {"id": True, "method": "profile_observe"}, {"id": 1, "method": []},
    {"id": 1, "method": "profile_observe", "extra": "PRIVATE"},
])
def test_only_fixed_read_only_rpc_is_accepted_without_issuing_any_observation(packet):
    session, controller = setup_session()
    with pytest.raises(ProtocolError):
        session.handle(packet)
    assert session.env.observations == 0 and not controller.reads


def test_three_read_budget_and_terminal_close_are_enforced():
    session, controller = setup_session()
    for index in range(1, 4):
        assert session.handle({"id": index, "method": "profile_observe"})["diagnostic"]["index"] == index
    with pytest.raises(ProtocolError):
        session.handle({"id": 4, "method": "profile_observe"})
    assert session.env.observations == 3 and len(controller.reads) == 6
    assert session.handle({"id": 5, "method": "close"}) is None
    with pytest.raises(ProtocolError):
        session.handle({"id": 6, "method": "profile_observe"})


def test_read_exception_is_preserved_all_patches_restored_and_failed_sample_consumes_budget():
    session, controller = setup_session()
    error = RuntimeError("PRIVATE transport exception")
    controller.error = error
    before = dict(vars(controller))
    with pytest.raises(RuntimeError) as caught:
        session.handle({"id": 1, "method": "profile_observe"})
    assert caught.value is error and vars(controller) == before
    assert session.profile_count == 1 and controller.reads == ["screenshot"]
    controller.error = None
    assert session.handle({"id": 2, "method": "profile_observe"})["diagnostic"]["index"] == 2


def test_instance_level_read_methods_are_restored_without_replacing_their_original_identity():
    clock = Clock()
    methods = {name: (lambda: None) for name in (
        "get_screenshot", "get_accessibility_tree", "get_terminal_output",
    )}
    controller = SimpleNamespace(**methods)
    timing = {name: {"calls": 0, "wall_ms": 0} for name in ("screenshot", "accessibility_tree", "terminal")}
    with profiler.timed_controller(controller, timing, clock=clock.read):
        controller.get_screenshot()
    assert all(getattr(controller, name) is method for name, method in methods.items())
