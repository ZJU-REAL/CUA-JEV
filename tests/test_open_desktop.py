from types import SimpleNamespace

import pytest

from cua_jev.episode import EpisodeConfig, EpisodeRunner, EpisodeStatus
from cua_jev.executors import ControlExecutor
from cua_jev.guard import ActionGuard
from cua_jev.models import Channel
from cua_jev.open_browser import PublicDecisionPolicy
from cua_jev.open_desktop import DesktopPlan, DesktopSnapshot, OpenDesktopTask
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.verify import VerifierRegistry
from cua_jev.vision import VisualTarget, WindowImage


class FakeControl:
    def __init__(self, window, name, control_type, index):
        self.window = window
        self.name = name
        self.value = ""
        self.index = index
        self.element_info = SimpleNamespace(
            control_type=control_type, name=name, automation_id=f"id-{index}", is_password=False,
        )

    def window_text(self):
        return self.name

    def is_visible(self):
        return True

    def is_enabled(self):
        return True

    def rectangle(self):
        return SimpleNamespace(left=10, top=10 + self.index * 30, right=100, bottom=35 + self.index * 30)

    def invoke(self):
        if self.name == "Finish":
            self.window.title = "Done"

    def set_edit_text(self, value):
        self.value = value

    def get_value(self):
        return self.value


class FakeWindow:
    handle = 42

    def __init__(self):
        self.title = "Example"
        self.controls = [
            FakeControl(self, "Name", "Edit", 0),
            FakeControl(self, "Finish", "Button", 1),
        ]

    def descendants(self):
        return self.controls

    def window_text(self):
        return self.title


class FakeDesktopPlanner:
    def __init__(self, options=None):
        self.calls = 0
        self.options = options or [{"ref": "c1", "operation": "click"}]

    def plan(self, goal, snapshot, recent_actions):
        self.calls += 1
        assert goal == "Finish the window"
        return DesktopPlan.from_dict(
            {
                "subgoal": "Finish",
                "options": self.options,
                "success": {"kind": "window_title_contains", "value": "Done"},
            },
            snapshot,
        )


def test_desktop_task_runs_without_app_specific_script():
    planner = FakeDesktopPlanner()
    environment = OpenDesktopTask(
        goal="Finish the window", window_title_re="Example", planner=planner,
        window=FakeWindow(), allow_button_actions=True,
    )

    def factory(item):
        executors = ExecutorRegistry()
        executors.register(Channel.API, item)
        executors.register(Channel.GUI, item)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=4)).run(environment)
    assert result.status == EpisodeStatus.SUCCESS
    assert result.channel_counts == {"api": 1}
    assert planner.calls == 1


def test_desktop_snapshot_redacts_password_and_validates_control_refs():
    window = FakeWindow()
    secret = FakeControl(window, "Password", "Edit", 2)
    secret.element_info.is_password = True
    window.controls.append(secret)
    snapshot, handles = DesktopSnapshot.capture(window)
    assert {item.ref for item in snapshot.controls} == {"c0", "c1"}
    assert "c2" not in handles
    with pytest.raises(ValueError, match="unavailable"):
        DesktopPlan.from_dict(
            {
                "subgoal": "Try a secret",
                "options": [{"ref": "c2", "operation": "fill", "value": "x"}],
                "success": {"kind": "text_contains", "value": "x"},
            }, snapshot,
        )


def test_desktop_fill_is_verified_by_control_value():
    window = FakeWindow()
    environment = OpenDesktopTask(
        goal="Finish the window", window_title_re="Example",
        planner=FakeDesktopPlanner([{"ref": "c0", "operation": "fill", "value": "Ada"}]),
        window=window, allow_text_input=True,
    )
    environment.reset()
    observation = environment.observe(())
    candidate = environment.candidates(observation, ())[0]
    receipt = environment(candidate, observation.observation_id, "d1")
    verifiers = VerifierRegistry()
    environment.register_verifiers(verifiers)
    assert verifiers.verify(candidate, receipt).passed
    assert window.controls[0].value == "Ada"
    environment.close()


def test_changed_desktop_control_invalidates_old_plan():
    window = FakeWindow()
    planner = FakeDesktopPlanner()
    environment = OpenDesktopTask(
        goal="Finish the window", window_title_re="Example", planner=planner,
        window=window, allow_button_actions=True,
    )
    environment.reset()
    first = environment.observe(())
    environment.candidates(first, ())
    window.controls[1].name = "Finish now"
    second = environment.observe(())
    environment.candidates(second, ())
    assert planner.calls == 2
    environment.close()


class VisualFakeWindow(FakeWindow):
    def __init__(self):
        super().__init__()
        self.controls = []

    def rectangle(self):
        return SimpleNamespace(left=20, top=30, right=220, bottom=230)


class FakeVisionGrounder:
    def __init__(self):
        self.calls = 0

    def perceive_window(self, goal, window_title, ui_text, image):
        self.calls += 1
        assert goal == "Finish the window"
        assert window_title == "Example"
        assert image.region == (20, 30, 200, 200)
        return (VisualTarget("v0", "Finish", (400, 400, 600, 600)),)


class FakeVisualScreen:
    def __init__(self, window):
        self.window = window
        self.sample = bytes(64 * 64)
        self.clicks = []
        self.foreground = True

    def ensure_foreground(self, handle):
        assert handle == self.window.handle
        if not self.foreground:
            raise RuntimeError("selected window is not foreground")

    def capture_region(self, region):
        return WindowImage(b"\xff\xd8\xff\xd9", self.sample, region)

    def click_point(self, x, y):
        self.clicks.append((x, y))
        self.sample = bytes([255]) * (64 * 64)
        self.window.title = "Done"


def test_visual_fallback_routes_a_window_only_target_through_gui_and_verifies_effect():
    window = VisualFakeWindow()
    screen = FakeVisualScreen(window)
    grounder = FakeVisionGrounder()
    environment = OpenDesktopTask(
        goal="Finish the window", window_title_re="Example", window=window,
        screen=screen, planner=FakeDesktopPlanner([{"ref": "v0", "operation": "click"}]),
        allow_button_actions=True, allow_visual_clicks=True,
        vision_grounder=grounder, allow_screenshot_upload=True,
    )

    def factory(item):
        executors = ExecutorRegistry()
        executors.register(Channel.GUI, item)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=4)).run(environment)
    assert result.status == EpisodeStatus.SUCCESS
    assert result.channel_counts == {"gui": 1}
    assert grounder.calls == 1
    assert screen.clicks == [(120, 130)]


def test_visual_click_rejects_changed_screenshot_and_requires_explicit_opt_ins():
    window = VisualFakeWindow()
    screen = FakeVisualScreen(window)
    grounder = FakeVisionGrounder()
    with pytest.raises(ValueError, match="screenshot-upload consent"):
        OpenDesktopTask(
            goal="Finish the window", window_title_re="Example", planner=FakeDesktopPlanner(),
            vision_grounder=grounder,
        )
    environment = OpenDesktopTask(
        goal="Finish the window", window_title_re="Example", window=window,
        screen=screen, planner=FakeDesktopPlanner([{"ref": "v0", "operation": "click"}]),
        allow_button_actions=True, allow_visual_clicks=True,
        vision_grounder=grounder, allow_screenshot_upload=True,
    )
    environment.reset()
    observation = environment.observe(())
    assert "jpeg" not in str(observation.state)
    assert observation.state["visual_targets"][0]["ref"] == "v0"
    candidate = environment.candidates(observation, ())[0]
    screen.sample = bytes([255]) * (64 * 64)
    receipt = environment(candidate, observation.observation_id, "d1")
    assert not receipt.success
    assert "stale" in (receipt.error or "")
    assert not screen.clicks
    environment.close()


def test_visual_observation_does_not_upload_when_selected_window_loses_focus():
    window = VisualFakeWindow()
    screen = FakeVisualScreen(window)
    screen.foreground = False
    grounder = FakeVisionGrounder()
    environment = OpenDesktopTask(
        goal="Finish the window", window_title_re="Example", window=window,
        screen=screen, planner=FakeDesktopPlanner(),
        vision_grounder=grounder, allow_screenshot_upload=True,
    )
    environment.reset()
    with pytest.raises(RuntimeError, match="not foreground"):
        environment.observe(())
    assert grounder.calls == 0
    environment.close()


def test_visual_target_cannot_be_used_for_text_input():
    snapshot, _ = DesktopSnapshot.capture(VisualFakeWindow())
    from dataclasses import replace

    snapshot = replace(snapshot, visual_targets=(VisualTarget("v0", "Search", (100, 100, 300, 200)),))
    with pytest.raises(ValueError, match="click only"):
        DesktopPlan.from_dict({
            "subgoal": "Fill", "options": [{"ref": "v0", "operation": "fill", "value": "secret"}],
            "success": {"kind": "window_title_contains", "value": "Done"},
        }, snapshot)
