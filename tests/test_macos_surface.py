import copy
import json
from dataclasses import replace

import httpx
import pytest
from test_open_computer import BrowserPage

from cua_jev.episode import EpisodeConfig, EpisodeRunner
from cua_jev.errors import CapabilityUnavailable
from cua_jev.executors import ControlExecutor
from cua_jev.guard import ActionGuard
from cua_jev.macos_surface import MacAccessibilitySurface, MacDesktopTask, MacSnapshot
from cua_jev.model_planner import ChatModelPlanner
from cua_jev.models import Channel
from cua_jev.open_browser import OpenBrowserTask, PublicDecisionPolicy
from cua_jev.open_computer import ComputerPlan, OpenComputerTask
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.trace import JsonlTrace
from cua_jev.verify import VerifierRegistry


class NativeFixture:
    def __init__(self):
        self.state = {
            "window_title": "Native fixture", "window_token": "pinned-window", "window_id": 42,
            "pid": 456, "bundle_id": "test.fixture", "bounds": [20, 20, 560, 320],
            "text": "No note saved", "controls": [
                {"ref": "c0", "name": "Note text", "role": "AXTextField", "control_type": "Edit",
                 "enabled": True, "gui_available": True, "rectangle": [50, 100, 300, 30], "can_invoke": False,
                 "can_set_text": True, "private_value": "old private text", "state_value": ""},
                {"ref": "c1", "name": "Save note", "role": "AXButton", "control_type": "Button",
                 "enabled": True, "gui_available": True, "rectangle": [50, 160, 100, 30], "can_invoke": True,
                 "can_set_text": False, "private_value": "", "state_value": ""},
            ],
        }
        self.commands = []
        self.closed = False
        self.foreground = True
        self.permission = True

    def call(self, command, **arguments):
        self.commands.append((command, arguments))
        if not self.permission:
            raise CapabilityUnavailable("macOS Accessibility permission is required")
        if command in {"select", "observe"}:
            return copy.deepcopy(self.state)
        if command == "act":
            if arguments["route"] == "gui" and not self.foreground:
                raise RuntimeError("Selected window is not foreground")
            if arguments["operation"] == "fill":
                self.state["controls"][0]["private_value"] = arguments["value"]
            else:
                self.state["text"] = "Saved: " + self.state["controls"][0]["private_value"]
            return {"ref": arguments["ref"], "operation": arguments["operation"], "route": arguments["route"]}
        if command in {"record_start", "record_stop"}:
            return {"recording": command == "record_start"}
        raise AssertionError(command)

    def close(self):
        self.closed = True


def surface(bridge=None, **kwargs):
    return MacAccessibilitySurface(
        window_title_re="Native fixture", bridge=bridge or NativeFixture(),
        allow_text_input=True, allow_button_actions=True, verify_timeout_s=0, **kwargs,
    )


def test_native_state_contract_has_no_windows_handle_and_redacts_values():
    state = MacSnapshot.from_native(NativeFixture().state)
    assert "old private text" in state.acceptance_text()
    assert "private text" not in json.dumps(state.to_dict())
    compact = json.dumps(state.for_model())
    for private in ("private text", "window_id", "pid", "bounds", "rectangle", "window_token"):
        assert private not in compact
    modified = copy.deepcopy(state.native)
    modified["controls"][0]["private_value"] = "changed"
    assert state.fingerprint() != MacSnapshot.from_native(modified).fingerprint()


@pytest.mark.parametrize("channel", [Channel.API, Channel.GUI])
def test_both_native_routes_fill_and_verify_actual_value(channel):
    item = surface()
    item.reset()
    state = item.observe(())
    candidates = item.compile(state, "c0", "fill", "New note 中文", "Write", 0)
    assert {candidate.channel for candidate in candidates} == {Channel.API, Channel.GUI}
    selected = next(candidate for candidate in candidates if candidate.channel == channel)
    receipt = item.execute(selected, "obs", "decision")
    assert receipt.success, receipt.error
    assert item.verify_effect(selected, receipt).passed
    assert item.capture().native["controls"][0]["private_value"] == "New note 中文"


@pytest.mark.parametrize("change", ["geometry", "value", "window", "enabled"])
def test_native_action_rejects_stale_state_before_any_input(change):
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    candidate = item.compile(item.observe(()), "c1", "click", "", "Save", 0)[0]
    if change == "geometry":
        bridge.state["bounds"][0] += 10
    elif change == "value":
        bridge.state["controls"][0]["private_value"] = "changed privately"
    elif change == "enabled":
        bridge.state["controls"][1]["enabled"] = False
    else:
        bridge.state["window_token"] = "another window"
    receipt = item.execute(candidate, "obs", "decision")
    assert not receipt.success and "changed" in receipt.error
    assert not any(command == "act" for command, _ in bridge.commands)


def test_gui_focus_failure_is_not_retried_through_ax():
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    candidate = item.compile(item.observe(()), "c1", "click", "", "Save", 0)[1]
    bridge.foreground = False
    receipt = item.execute(candidate, "obs", "decision")
    assert not receipt.success and "foreground" in receipt.error
    assert not item.verify_effect(candidate, receipt).passed
    assert [args["route"] for command, args in bridge.commands if command == "act"] == ["gui"]


def test_permission_denial_does_not_fallback_to_screenshot():
    bridge = NativeFixture()
    bridge.permission = False
    with pytest.raises(CapabilityUnavailable, match="Accessibility"):
        surface(bridge).reset()
    assert [command for command, _ in bridge.commands] == ["select"]


def test_native_candidate_cannot_forge_ref_route_or_text():
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    candidate = item.compile(item.observe(()), "c0", "fill", "Safe", "Fill", 0)[0]
    forged = replace(candidate, arguments={**candidate.arguments, "value": "Different"})
    assert not item.execute(forged, "obs", "decision").success
    assert not any(command == "act" for command, _ in bridge.commands)
    with pytest.raises(ValueError, match="unavailable"):
        item.validate(item.capture(), "c999", "click", "")
    with pytest.raises(ValueError, match="editable"):
        item.validate(item.capture(), "c1", "fill", "Value")


def test_native_candidate_cannot_mutate_trusted_arguments_in_place():
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    candidate = item.compile(item.observe(()), "c0", "fill", "Safe", "Fill", 0)[0]
    candidate.arguments["value"] = "Changed after compilation"
    assert not item.execute(candidate, "obs", "decision").success
    assert not any(command == "act" for command, _ in bridge.commands)


def test_observed_state_mutation_cannot_rewrite_candidate_freshness():
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    state = item.observe(())
    candidate = item.compile(state, "c1", "click", "", "Save", 0)[0]
    bridge.state["controls"][0]["private_value"] = "changed privately"
    state.native["controls"][0]["private_value"] = "changed privately"
    assert not item.execute(candidate, "obs", "decision").success
    assert not any(command == "act" for command, _ in bridge.commands)


def test_native_attempt_consumes_all_routes_even_without_a_visible_effect():
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    candidates = item.compile(item.observe(()), "c0", "fill", "old private text", "Fill", 0)
    assert item.execute(candidates[0], "obs", "first").success
    assert not item.execute(candidates[0], "obs", "replay").success
    assert not item.execute(candidates[1], "obs", "fallback").success
    assert sum(command == "act" for command, _ in bridge.commands) == 1


def test_reset_failure_discards_candidates_from_previous_selection():
    bridge = NativeFixture()
    item = surface(bridge)
    item.reset()
    candidate = item.compile(item.observe(()), "c1", "click", "", "Save", 0)[0]
    bridge.permission = False
    with pytest.raises(CapabilityUnavailable):
        item.reset()
    bridge.permission = True
    assert not item.execute(candidate, "obs", "decision").success
    assert not any(command == "act" for command, _ in bridge.commands)


def test_native_gui_only_control_does_not_offer_nonexistent_ax_action():
    bridge = NativeFixture()
    bridge.state["controls"][1]["can_invoke"] = False
    item = surface(bridge)
    item.reset()
    candidates = item.compile(item.observe(()), "c1", "click", "", "Save", 0)
    assert [candidate.channel for candidate in candidates] == [Channel.GUI]


def test_native_recording_lifecycle_and_create_only_path(tmp_path):
    bridge = NativeFixture()
    item = surface(bridge, record_path=tmp_path / "window.mp4")
    item.reset()
    item.close()
    assert [command for command, _ in bridge.commands] == ["select", "record_start", "record_stop"]
    assert bridge.closed
    (tmp_path / "window.mp4").touch()
    with pytest.raises(ValueError, match="new .mp4"):
        surface(record_path=tmp_path / "window.mp4").reset()


def test_active_recording_cannot_silently_reselect_the_window(tmp_path):
    bridge = NativeFixture()
    item = surface(bridge, record_path=tmp_path / "window.mp4")
    item.reset()
    with pytest.raises(RuntimeError, match="active window recording"):
        item.reset()
    item.close()
    assert [command for command, _ in bridge.commands] == ["select", "record_start", "record_stop"]


def test_mac_surface_joins_existing_computer_loop_with_independent_gates(tmp_path):
    class Planner:
        def plan(self, goal, snapshot, recent):
            if "Saved: Example note" not in snapshot.desktop.acceptance_text():
                if "Example note" not in snapshot.desktop.acceptance_text():
                    options = [{"ref": "d:c0", "operation": "fill", "value": "Example note"}]
                else:
                    options = [{"ref": "d:c1", "operation": "click"}]
            else:
                options = [{"ref": "b:e0", "operation": "click"}]
            return ComputerPlan.from_dict({"subgoal": "Complete fixture", "options": options}, snapshot)

    planner = Planner()
    item = OpenComputerTask(
        goal="Save the note and open result", planner=planner,
        browser=OpenBrowserTask(goal="Save the note and open result", start_url="https://example.test/start",
                                planner=planner, page=BrowserPage()),
        desktop_surface=surface(), required_url_contains="/done",
        required_window_text_contains="Saved: Example note",
        required_capabilities=("desktop.fill", "desktop.click"),
    )
    trace = JsonlTrace(tmp_path / "private.jsonl")

    def factory(task):
        registry = ExecutorRegistry()
        for channel, executor in task.executor_bindings().items():
            registry.register(channel, executor)
        registry.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        task.register_verifiers(verifiers)
        return AgentRuntime(policy=PublicDecisionPolicy(RulePolicy()),
                            guard=ActionGuard(allow_writes=True, allow_destructive=True),
                            executors=registry, verifiers=verifiers, trace=trace)

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=5)).run(item)
    assert result.success, result.reason
    assert [step.receipt.capability for step in result.steps] == [
        "desktop.fill", "desktop.click", "browser.click",
    ]
    assert all(step.verification.passed for step in result.steps)


def test_mac_planner_uses_native_semantics_and_private_projection():
    def handler(request):
        body = json.loads(request.content)
        assert "macOS Accessibility" in body["messages"][0]["content"]
        assert "old private text" not in str(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "subgoal": "Save note", "options": [{"ref": "c1", "operation": "click"}],
            "success": {"kind": "text_contains", "value": "Saved"},
        })}}]})
    planner = ChatModelPlanner("https://test.example/v1", "test",
                               client=httpx.Client(transport=httpx.MockTransport(handler)))
    plan = planner.plan("Save note", MacSnapshot.from_native(NativeFixture().state), [])
    assert plan.options[0].ref == "c1"


def test_standalone_mac_task_requires_caller_acceptance():
    with pytest.raises(ValueError, match="caller-supplied"):
        MacDesktopTask(goal="Finish", planner=object(), window_title_re="Fixture")


def test_background_or_obscured_control_only_offers_real_ax_route():
    bridge = NativeFixture()
    bridge.state["controls"][1]["gui_available"] = False
    item = surface(bridge)
    item.reset()
    candidates = item.compile(item.observe(()), "c1", "click", "", "Save", 0)
    assert [candidate.channel for candidate in candidates] == [Channel.API]


def test_occlusion_diagnostics_alone_do_not_verify_a_click_effect():
    bridge = NativeFixture()
    original = bridge.call

    def call(command, **arguments):
        if command == "act":
            bridge.state["controls"][1]["gui_available"] = False
            bridge.state["controls"][1]["gui_unavailable_reason"] = "GUI target is obscured"
            return {"ref": arguments["ref"], "operation": "click", "route": "ax"}
        return original(command, **arguments)

    bridge.call = call
    item = surface(bridge)
    item.reset()
    candidate = item.compile(item.observe(()), "c1", "click", "", "Save", 0)[0]
    receipt = item.execute(candidate, "observation", "decision")
    assert receipt.success
    assert not item.verify_effect(candidate, receipt).passed


def test_recording_finalize_error_still_closes_browser_and_other_providers(tmp_path):
    closed = []

    class Provider:
        def __init__(self, name, fail=False):
            self.name, self.fail = name, fail

        def close(self):
            closed.append(self.name)
            if self.fail:
                raise RuntimeError("recording did not finalize")

    item = object.__new__(OpenComputerTask)
    item.providers = {
        "b": Provider("browser"), "d": Provider("desktop", fail=True), "a": Provider("artifact"),
    }
    with pytest.raises(RuntimeError, match="recording did not finalize"):
        item.close()
    assert closed == ["artifact", "desktop", "browser"]
