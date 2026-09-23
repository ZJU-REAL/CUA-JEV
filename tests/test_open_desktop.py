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
