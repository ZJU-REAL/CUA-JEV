"""Owned worker timings must preserve original observations, input and exceptions."""

from types import SimpleNamespace

import pytest
from test_osworld_observation_profiler import Clock, Controller, Environment
from test_osworld_remote_worker import observation, worker


@pytest.mark.parametrize("original_fails", [False, True])
def test_existing_controller_order_result_exception_identity_and_budget_clock_unchanged(
    monkeypatch, original_fails,
):
    controller = Controller(Clock())
    env = Environment(controller)
    original = env._get_obs
    result = object()
    failure = RuntimeError("private original error")
    calls, budget_calls = [], []

    def perf_counter():
        budget_calls.append("original budget clock")
        return 123.0

    def getter(token, *, option):
        calls.append((token, option))
        worker.time.perf_counter()
        original()
        if original_fails:
            raise failure
        return result

    env._get_obs = getter
    previous_controller = dict(vars(controller))
    monkeypatch.setattr(worker.time, "perf_counter", perf_counter)
    capture = worker.ObservationCaptureTiming(env)
    try:
        with capture.scope("observe"):
            if original_fails:
                with pytest.raises(RuntimeError) as caught:
                    env._get_obs("original arg", option="original kwarg")
                assert caught.value is failure
            else:
                assert env._get_obs("original arg", option="original kwarg") is result
    finally:
        capture.restore()
    assert env._get_obs is getter
    assert calls == [("original arg", "original kwarg")]
    assert budget_calls == ["original budget clock"]
    assert env.observations == 1 and controller.reads == ["screenshot", "tree"]
    assert vars(controller) == previous_controller
    group = capture.report["groups"]["observe"]
    assert group["get_obs_calls"] == 1
    assert group["get_obs_failures"] == int(original_fails)
    assert group["controller_capture_failures"] == 0
    assert group["controller_reads"]["screenshot"]["calls"] == 1
    assert group["controller_reads"]["accessibility_tree"]["calls"] == 1
    assert group["controller_reads"]["terminal"]["calls"] == 0
    assert capture.report["clock"] == "monotonic_ns"
    assert capture.report["original_get_obs_restored"] is True
    assert "private original error" not in str(capture.report)


def test_original_controller_exception_is_not_retried_and_all_attributes_are_restored():
    controller = Controller(Clock())
    failure = RuntimeError("private failed screenshot")
    controller.error = failure
    env = Environment(controller)
    before_env, before_controller = dict(vars(env)), dict(vars(controller))
    capture = worker.ObservationCaptureTiming(env)
    try:
        with capture.scope("startup_task_setup"):
            with pytest.raises(RuntimeError) as caught:
                env._get_obs()
        assert caught.value is failure
    finally:
        capture.restore()
    assert vars(env) == {**before_env, "observations": 1}
    assert vars(controller) == before_controller
    assert controller.reads == ["screenshot"]
    group = capture.report["groups"]["startup_task_setup"]
    assert group["get_obs_failures"] == 1
    assert group["get_obs_error_types"] == {"RuntimeError": 1}
    assert group["controller_reads"]["screenshot"]["calls"] == 1
    assert group["controller_reads"]["accessibility_tree"]["calls"] == 0
    assert capture.report["component_failure_counts"] == (
        "not observed by existing timed_controller"
    )


@pytest.mark.parametrize("original_fails", [False, True])
def test_no_controller_is_unavailable_and_preserves_one_original_call(original_fails):
    result, failure, calls = object(), RuntimeError("private no-controller failure"), []

    def getter():
        calls.append("original")
        if original_fails:
            raise failure
        return result

    env = SimpleNamespace(_get_obs=getter)
    capture = worker.ObservationCaptureTiming(env)
    try:
        with capture.scope("startup_desktop_readiness"):
            if original_fails:
                with pytest.raises(RuntimeError) as caught:
                    env._get_obs()
                assert caught.value is failure
            else:
                assert env._get_obs() is result
    finally:
        capture.restore()
    assert calls == ["original"] and env._get_obs is getter
    group = capture.report["groups"]["startup_desktop_readiness"]
    assert group["get_obs_calls"] == group["controller_capture_failures"] == 1
    assert group["get_obs_failures"] == int(original_fails)
    assert group["instrumentation_errors"] == {"AttributeError": 1}


@pytest.mark.parametrize("unavailable", ["missing_unused_terminal", "slots"])
def test_controller_instrumentation_unavailable_does_not_skip_or_repeat_original(unavailable):
    events = []

    class SlotsController:
        __slots__ = ()

        def get_screenshot(self):
            events.append("screenshot")
            return b"original"

        def get_accessibility_tree(self):
            events.append("tree")
            return "<original/>"

        def get_terminal_output(self):
            pytest.fail("terminal is not part of this original observation")

    if unavailable == "slots":
        controller = SlotsController()
    else:
        controller = SimpleNamespace(
            get_screenshot=lambda: (events.append("screenshot"), b"original")[1],
            get_accessibility_tree=lambda: (events.append("tree"), "<original/>")[1],
        )
    env = Environment(controller)
    before_controller = dict(vars(controller)) if hasattr(controller, "__dict__") else None
    capture = worker.ObservationCaptureTiming(env)
    try:
        with capture.scope("startup_passive_settling"):
            assert env._get_obs()["screenshot"] == b"original"
    finally:
        capture.restore()
    assert env.observations == 1 and events == ["screenshot", "tree"]
    if before_controller is not None:
        assert vars(controller) == before_controller
    group = capture.report["groups"]["startup_passive_settling"]
    assert group["get_obs_calls"] == group["controller_capture_failures"] == 1
    assert group["get_obs_failures"] == 0


def test_unpatchable_environment_keeps_original_method_available():
    calls, result = [], object()

    class EnvironmentWithoutInstanceDictionary:
        __slots__ = ()

        def _get_obs(self):
            calls.append("original")
            return result

    env = EnvironmentWithoutInstanceDictionary()
    capture = worker.ObservationCaptureTiming(env)
    assert env._get_obs() is result
    capture.restore()
    assert calls == ["original"] and capture.report["installed"] is False
    assert capture.report["groups"] == {}  # Unobserved calls are not fabricated zeroes.
    assert capture.report["instrumentation_errors"] == {"TypeError": 1}


def test_real_worker_rpc_scopes_preserve_full_and_image_only_getter_paths(tmp_path):
    raw = observation()
    events, actions = [], []

    class ControllerForWorker:
        def get_screenshot(self):
            events.append("screenshot")
            return raw["screenshot"]

        def get_accessibility_tree(self):
            events.append("tree")
            return raw["accessibility_tree"]

        def get_terminal_output(self):
            pytest.fail("no original terminal read")

    class Env:
        require_a11y_tree = True
        controller = ControllerForWorker()

        def _get_obs(self):
            return {"screenshot": self.controller.get_screenshot(),
                    "accessibility_tree": self.controller.get_accessibility_tree()
                    if self.require_a11y_tree else None}

        def step(self, action, pause):
            actions.append((action, pause, self.require_a11y_tree))
            return self._get_obs(), 0.0, False, {}

        def close(self):
            pass

    env = Env()
    capture = worker.ObservationCaptureTiming(env)
    session = worker.WorkerSession(
        env, raw, tmp_path, image_only_fill_intermediates=True,
        guard_factory=lambda _env: SimpleNamespace(close=lambda: None),
    )
    session.capture_timing = capture
    full = {"action_type": "PRESS", "parameters": {"key": "tab"}}
    focus = {"action_type": "CLICK", "parameters": {
        "x": 10, "y": 10, "button": "left", "num_clicks": 1,
    }}
    try:
        session.handle({"id": 1, "method": "observe"})
        session.handle({"id": 2, "method": "step", "action": full, "pause": 0.5})
        partial = session.handle({"id": 3, "method": "step", "action": focus, "pause": 0.5,
                                  "observation_mode": worker.IMAGE_ONLY_VERSION})
    finally:
        capture.restore()
    assert events == ["screenshot", "tree", "screenshot", "tree", "screenshot"]
    assert actions == [(full, 0.5, True), (focus, 0.5, False)]
    assert env.require_a11y_tree is True and session.steps == 2
    assert partial["observation"]["accessibility_tree"] is None
    groups = capture.report["groups"]
    assert set(groups) == {"observe", "full_step", "image_only_step"}
    assert all(group["get_obs_calls"] == 1 for group in groups.values())
    assert groups["image_only_step"]["controller_reads"]["accessibility_tree"]["calls"] == 0
    assert all(groups[name]["controller_reads"]["accessibility_tree"]["calls"] == 1
               for name in ("observe", "full_step"))


@pytest.mark.parametrize("original_fails", [False, True])
def test_diagnostic_clock_and_exception_capture_ignore_fake_worker_io_handles(
    monkeypatch, original_fails,
):
    class OriginalDeadline(BaseException):
        pass

    result, failure, calls = object(), OriginalDeadline(), []

    def getter():
        calls.append("original")
        if original_fails:
            raise failure
        return result

    # Existing readiness tests replace these worker handles with narrow facades.
    # Diagnostic internals must not add requirements to the original facades.
    monkeypatch.setattr(worker, "sys", SimpleNamespace())
    monkeypatch.setattr(worker, "time", SimpleNamespace())
    env = SimpleNamespace(_get_obs=getter)
    capture = worker.ObservationCaptureTiming(env)
    try:
        with capture.scope("startup_desktop_readiness"):
            if original_fails:
                with pytest.raises(OriginalDeadline) as caught:
                    env._get_obs()
                assert caught.value is failure
            else:
                assert env._get_obs() is result
    finally:
        capture.restore()
    assert calls == ["original"] and env._get_obs is getter
    group = capture.report["groups"]["startup_desktop_readiness"]
    assert group["get_obs_calls"] == 1 and group["get_obs_failures"] == int(original_fails)
