"""Acquisition-order opt-in changes neither input authority nor unsupported getters."""

import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest
from test_osworld_remote_driver import arguments, install_episode_fakes
from test_osworld_remote_driver import driver as driver
from test_osworld_remote_worker import worker

from cua_jev import osworld_observation_order as order

# Independent copy of the actually captured SDK nine-line method contract.
SDK_SOURCE = """class DesktopEnv:
    def __init__(self, controller, a11y=True, terminal=False):
        self.controller = controller
        self.require_a11y_tree = a11y
        self.require_terminal = terminal
        self.instruction = "public goal"

    def _get_obs(self):
        # We provide screenshot, accessibility_tree (optional), terminal (optional), and instruction.
        # can be customized and scaled
        return {
            "screenshot": self.controller.get_screenshot(),
            "accessibility_tree": (
                self.controller.get_accessibility_tree() if self.require_a11y_tree else None
            ),
            "terminal": self.controller.get_terminal_output() if self.require_terminal else None,
            "instruction": self.instruction
        }
"""


class Controller:
    def __init__(self):
        self.calls, self.errors = [], {}
        self.screenshot, self.tree = b"stable screenshot", "<desktop/>"

    def value(self, name, result):
        self.calls.append(name)
        if name in self.errors:
            raise self.errors[name]
        return result

    def get_screenshot(self):
        return self.value("screenshot", self.screenshot)

    def get_accessibility_tree(self):
        return self.value("accessibility_tree", self.tree)

    def get_terminal_output(self):
        return self.value("terminal", "public terminal")


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    root = tmp_path / "owned_sdk"
    package = root / "desktop_env"
    package.mkdir(parents=True)
    path = package / "desktop_env.py"
    path.write_text(SDK_SOURCE, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("desktop_env.desktop_env", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return root, module.DesktopEnv, root.stat().st_uid


def install(sdk, *, mode=order.AX_FIRST_ORDER, a11y=True, terminal=False, owner=None):
    root, cls, uid = sdk
    env = cls(Controller(), a11y=a11y, terminal=terminal)
    adapter = order.OwnedObservationOrder(
        env, requested=mode, osworld_root=root, owner_uid=uid if owner is None else owner,
    )
    return env, adapter


def test_default_original_keeps_exact_result_and_public_frame_bytes_without_checking_SDK(
    sdk, monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("default control must not inspect or replace the SDK getter")

    monkeypatch.setattr(order.inspect, "getsource", forbidden)
    env, adapter = install(sdk, mode=order.ORIGINAL_ORDER)
    original = env._get_obs.__func__
    first = env._get_obs()
    expected = json.dumps(worker.public_observation(first), ensure_ascii=True, allow_nan=False)
    second = env._get_obs()
    actual = json.dumps(worker.public_observation(second), ensure_ascii=True, allow_nan=False)
    adapter.restore()
    assert actual == expected and first == second
    assert env.controller.calls == ["screenshot", "accessibility_tree"] * 2
    assert "_get_obs" not in vars(env) and env._get_obs.__func__ is original
    assert adapter.report["installed"] is False and adapter.report["support_checked"] is False


def test_supported_full_read_uses_each_original_getter_once_and_keeps_fields_and_key_order(sdk):
    env, adapter = install(sdk)
    try:
        result = env._get_obs()
        assert adapter.report["sdk_supported"] is adapter.report["installed"] is True
        assert adapter.report["method_AST_and_live_bytecode_match"] is True
        assert env.controller.calls == ["accessibility_tree", "screenshot"]
        assert list(result) == ["screenshot", "accessibility_tree", "terminal", "instruction"]
        assert result["screenshot"] is env.controller.screenshot
        assert result["accessibility_tree"] is env.controller.tree
        assert result["terminal"] is None and result["instruction"] == "public goal"
        assert adapter.report["calls"]["reordered_full"] == 1
        assert adapter.report["atomic_observation"] is False
    finally:
        adapter.restore()
    assert "_get_obs" not in vars(env) and adapter.report["restored"] is True


def test_image_only_keeps_original_getter_zero_AX_and_one_screenshot(sdk):
    env, adapter = install(sdk)
    env.require_a11y_tree = False
    try:
        result = env._get_obs()
        assert env.controller.calls == ["screenshot"] and result["accessibility_tree"] is None
        assert adapter.report["calls"]["original_image_only"] == 1
        assert adapter.report["calls"]["reordered_full"] == 0
    finally:
        adapter.restore()


@pytest.mark.parametrize("a11y", [False, True])
def test_terminal_true_refuses_install_and_preserves_original_flags_reads_and_result(sdk, a11y):
    env, adapter = install(sdk, a11y=a11y, terminal=True)
    assert env.controller.calls == [] and "_get_obs" not in vars(env)
    result = env._get_obs()
    assert env.controller.calls == (["screenshot", "accessibility_tree", "terminal"]
                                    if a11y else ["screenshot", "terminal"])
    assert result["terminal"] == "public terminal"
    assert adapter.report["installed"] is False
    assert adapter.report["reason"] == "terminal_mode_original_retained"


@pytest.mark.parametrize("stage", ["accessibility_tree", "screenshot"])
def test_reordered_original_exception_identity_propagates_without_fallback_or_retry(sdk, stage):
    env, adapter = install(sdk)
    failure = RuntimeError("private original transport error")
    env.controller.errors[stage] = failure
    try:
        with pytest.raises(RuntimeError) as caught:
            env._get_obs()
        assert caught.value is failure
        prefix = ["accessibility_tree"] if stage == "accessibility_tree" else [
            "accessibility_tree", "screenshot",
        ]
        assert env.controller.calls == prefix
        assert adapter.report["calls"]["original_fallback"] == 0
        assert adapter.report["reordered_failures"] == [{
            "attempted_getter_prefix": prefix, "exception_stage": stage,
            "exception_type": "RuntimeError", "original_path_retried": False,
        }]
        assert "private original transport error" not in str(adapter.report)
    finally:
        adapter.restore()


def test_unknown_SDK_mock_is_not_called_or_patched_during_refusal(tmp_path):
    result, calls = object(), []
    def getter():
        calls.append("original")
        return result
    env = SimpleNamespace(_get_obs=getter)
    adapter = order.OwnedObservationOrder(
        env, requested=order.AX_FIRST_ORDER, osworld_root=tmp_path,
        owner_uid=tmp_path.stat().st_uid,
    )
    assert calls == [] and env._get_obs is getter and adapter.report["installed"] is False
    assert env._get_obs() is result and calls == ["original"]


def test_owner_mismatch_and_existing_instance_override_leave_original_untouched(sdk):
    env, adapter = install(sdk, owner=sdk[2] + 1)
    assert adapter.report["installed"] is False and "_get_obs" not in vars(env)
    assert env.controller.calls == []
    result = object()
    def getter():
        return result
    env._get_obs = getter
    adapter = order.OwnedObservationOrder(
        env, requested=order.AX_FIRST_ORDER, osworld_root=sdk[0], owner_uid=sdk[2],
    )
    assert env._get_obs is getter and env._get_obs() is result
    assert adapter.report["reason"] == "instance_override_original_retained"


def test_order_installs_before_timing_and_reverse_cleanup_restores_exact_class_method(sdk, monkeypatch):
    def forbidden():
        pytest.fail("new order and diagnostics must not consume the budget perf_counter")

    monkeypatch.setattr(worker.time, "perf_counter", forbidden)
    env, adapter = install(sdk)
    original_class_method = type(env)._get_obs
    capture = worker.ObservationCaptureTiming(env)
    try:
        with capture.scope("full_step"):
            env._get_obs()
        assert env.controller.calls == ["accessibility_tree", "screenshot"]
        reads = capture.report["groups"]["full_step"]["controller_reads"]
        assert reads["screenshot"]["calls"] == reads["accessibility_tree"]["calls"] == 1
    finally:
        capture.restore()
        assert env._get_obs.__self__ is adapter
        adapter.restore()
    assert "_get_obs" not in vars(env) and env._get_obs.__func__ is original_class_method


@pytest.mark.parametrize("mode", [order.ORIGINAL_ORDER, order.AX_FIRST_ORDER])
def test_remote_CLI_transmits_order_only_when_explicitly_opted_in(driver, monkeypatch, tmp_path, mode):
    calls = install_episode_fakes(monkeypatch, driver)
    argv = arguments(tmp_path, **{"observation-order": mode})
    assert driver.main(argv) == 0
    command = calls["environments"][0].command
    metadata = json.loads((tmp_path / "run/metadata.json").read_text())
    if mode == order.ORIGINAL_ORDER:
        assert "--observation-order" not in command
        assert "requested_observation_acquisition_order" not in metadata
    else:
        assert command[command.index("--observation-order") + 1] == mode
        assert metadata["requested_observation_acquisition_order"] == mode


def test_same_method_name_and_owned_path_cannot_bypass_exact_SDK_shape_gate(sdk, monkeypatch):
    root, _cls, uid = sdk
    path = root / "desktop_env/desktop_env.py"
    changed = SDK_SOURCE.replace('"instruction": self.instruction', '"instruction": "changed"')
    assert changed != SDK_SOURCE
    path.write_text(changed, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("desktop_env.desktop_env", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    env = module.DesktopEnv(Controller())
    original = env._get_obs.__func__
    adapter = order.OwnedObservationOrder(
        env, requested=order.AX_FIRST_ORDER, osworld_root=root, owner_uid=uid,
    )
    assert env.controller.calls == [] and adapter.report["installed"] is False
    assert adapter.report["reason"] == "SDK_method_shape_original_retained"
    assert env._get_obs.__func__ is original and "_get_obs" not in vars(env)
    assert env._get_obs()["instruction"] == "changed"
    assert env.controller.calls == ["screenshot", "accessibility_tree"]
