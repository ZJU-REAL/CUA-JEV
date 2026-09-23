import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_open_desktop import FakeWindow

from cua_jev.episode import EpisodeConfig, EpisodeRunner, EpisodeStatus
from cua_jev.executors import ControlExecutor
from cua_jev.executors.common import execute_with_receipt
from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, Channel, Observation, Verification
from cua_jev.open_browser import OpenBrowserTask, PublicDecisionPolicy
from cua_jev.open_computer import ComputerPlan, OpenComputerTask
from cua_jev.open_desktop import DesktopSnapshot, OpenDesktopTask
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.verify import VerifierRegistry
from cua_jev.vision import VisualTarget


class BrowserLocator:
    def __init__(self, page):
        self.page = page

    def nth(self, _index):
        return self

    def click(self):
        self.page.url = "https://example.test/done"


class BrowserPage:
    def __init__(self):
        self.url = "https://example.test/start"
        self.main_frame = object()

    def evaluate(self, _script, _selector):
        return {
            "url": self.url, "title": "Example", "text": "private page text",
            "elements": [{
                "ref": "e0", "index": 0, "tag": "a", "role": "", "label": "Go to result",
                "input_type": "", "disabled": False, "href": "https://example.test/done",
            }],
        }

    def route(self, _pattern, _callback):
        pass

    def goto(self, url, **_kwargs):
        self.url = url

    def locator(self, _selector):
        return BrowserLocator(self)


class WindowControl:
    def __init__(self, window):
        self.window = window
        self.element_info = SimpleNamespace(
            control_type="Button", name="Finish", automation_id="finish", is_password=False,
        )

    def window_text(self):
        return "Finish"

    def is_visible(self):
        return True

    def is_enabled(self):
        return True

    def rectangle(self):
        return SimpleNamespace(left=10, top=10, right=100, bottom=40)

    def invoke(self):
        self.window.title = "Done"


class DesktopWindow:
    handle = 42

    def __init__(self):
        self.title = "Example"
        self.control = WindowControl(self)

    def descendants(self):
        return [self.control]

    def window_text(self):
        return self.title


class CrossAppPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, _goal, snapshot, _recent):
        self.calls += 1
        options = []
        if "filesystem.list" not in snapshot.requirements["completed_capabilities"]:
            options.append({"ref": "t:t0", "operation": "invoke"})
        if "/done" not in snapshot.browser.url:
            options.append({"ref": "b:e0", "operation": "click"})
        if snapshot.desktop.window_title != "Done":
            options.append({"ref": "d:c0", "operation": "click"})
        return ComputerPlan.from_dict({"subgoal": "Advance cross-app goal", "options": options}, snapshot)


def environment(tmp_path):
    goal = "Finish the browser and desktop task"
    planner = CrossAppPlanner()
    return OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url="https://example.test/start",
            planner=planner, page=BrowserPage(),
        ),
        desktop=OpenDesktopTask(
            goal=goal, window_title_re="Example", planner=planner,
            window=DesktopWindow(), allow_button_actions=True,
        ),
        local_tool_root=tmp_path,
        required_url_contains="/done", required_window_title_contains="Done",
        required_capabilities=("filesystem.list", "desktop.click"),
    )


def test_three_provider_open_goal_runs_with_independent_acceptance(tmp_path):
    task = environment(tmp_path)

    def factory(item):
        executors = ExecutorRegistry()
        for channel, executor in item.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allowed_roots=[tmp_path], allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=6)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert [step.receipt.capability for step in result.steps] == [
        "filesystem.list", "browser.click", "desktop.click",
    ]
    assert result.channel_counts == {"api": 2, "script": 1}
    assert all(step.verification.passed for step in result.steps)


def test_namespaced_refs_reject_unregistered_operations_and_paths(tmp_path):
    task = environment(tmp_path)
    task.reset()
    task.observe(())
    snapshot = task._snapshot
    assert snapshot is not None
    base = {"subgoal": "Act"}
    for option in (
        {"ref": "t:t99", "operation": "invoke"},
        {"ref": "t:t0", "operation": "invoke", "value": "rm -rf"},
        {"ref": "b:e0", "operation": "invoke"},
        {"ref": "d:c0", "operation": "fill", "value": "secret"},
        {"ref": "c0", "operation": "click"},
    ):
        with pytest.raises(ValueError):
            ComputerPlan.from_dict({**base, "options": [option]}, snapshot)
    task.close()


def test_uia_invoke_word_normalizes_to_typed_desktop_click(tmp_path):
    task = environment(tmp_path)
    task.reset()
    task.observe(())
    assert task._snapshot is not None
    plan = ComputerPlan.from_dict({
        "subgoal": "Press Finish",
        "options": [{"ref": "d:c0", "operation": "invoke"}],
    }, task._snapshot)
    assert plan.options[0].operation == "click"
    visual_snapshot = replace(
        task._snapshot,
        desktop=replace(
            task._snapshot.desktop,
            visual_targets=(VisualTarget("v0", "Seven", (100, 100, 200, 200)),),
        ),
    )
    visual_plan = ComputerPlan.from_dict({
        "subgoal": "Press Seven",
        "options": [{"ref": "d:v0", "operation": "click"}],
    }, visual_snapshot)
    assert visual_plan.options[0].ref == "v0"
    task.close()


def test_public_jev_state_omits_nested_page_window_and_tool_text():
    class CapturePolicy:
        def choose(self, observation, candidates):
            self.state = observation.state
            return RulePolicy().choose(observation, candidates)

    inner = CapturePolicy()
    policy = PublicDecisionPolicy(inner)
    policy.choose(
        Observation("goal", "subgoal", {
            "browser": {"url": "https://example.test", "title": "Public", "text": "secret page"},
            "desktop": {"window_title": "Example", "text": "secret window"},
            "tool_results": [{"result": "secret tool output"}],
        }, source="open-computer"),
        [ActionCandidate("one", Channel.CONTROL, "control.noop", "No-op")],
    )
    assert "secret" not in str(inner.state)


def test_invalid_model_plan_gets_one_safe_retry_without_execution(tmp_path):
    class FlakyPlanner(CrossAppPlanner):
        def plan(self, goal, snapshot, recent):
            if self.calls == 0:
                self.calls += 1
                raise ValueError("invalid operation")
            assert recent[-1]["planner_feedback"].startswith("Previous plan failed")
            return super().plan(goal, snapshot, recent)

    task = environment(tmp_path)
    task.planner = FlakyPlanner()
    task.reset()
    observation = task.observe(())
    candidates = task.candidates(observation, ())
    assert candidates
    assert task.planner_calls == 2
    assert task.browser.page.url == "https://example.test/start"
    assert task.desktop.window.title == "Example"
    task.close()


def test_terminal_acceptance_requires_real_state_and_registered_capability(tmp_path):
    base = environment(tmp_path)
    with pytest.raises(ValueError, match="observable state gate"):
        OpenComputerTask(
            goal=base.goal, browser=base.browser, desktop=base.desktop,
            planner=base.planner, required_capabilities=("desktop.click",),
        )
    with pytest.raises(ValueError, match="not registered"):
        OpenComputerTask(
            goal=base.goal, browser=base.browser, desktop=base.desktop,
            planner=base.planner, required_url_contains="/done",
            required_capabilities=("browser.delete_everything",),
        )


def test_system_chrome_control_is_not_a_computer_action(tmp_path):
    task = environment(tmp_path)
    task.desktop.window.control.window_text = lambda: "Close Calculator"
    task.reset()
    task.observe(())
    assert task._snapshot is not None
    with pytest.raises(ValueError, match="system chrome"):
        ComputerPlan.from_dict({
            "subgoal": "Close window",
            "options": [{"ref": "d:c0", "operation": "click"}],
        }, task._snapshot)
    task.close()


def test_replacement_accessibility_surface_uses_same_runtime_contract():
    class AlternateAccessibilitySurface:
        namespace = "d"
        supported_capabilities = frozenset({"desktop.click"})

        def __init__(self):
            self.window = DesktopWindow()

        def reset(self):
            pass

        def close(self):
            pass

        def observe(self, _history):
            return self.capture()

        def capture(self):
            return DesktopSnapshot.capture(self.window)[0]

        def validate(self, state, ref, operation, value):
            if ref != "c0" or operation != "click" or value:
                raise ValueError("unsupported accessibility ref")
            return ref, operation, ""

        def compile(self, state, ref, operation, value, subgoal, index):
            return (ActionCandidate(
                f"option_{index}_accessibility", Channel.API, "desktop.click",
                "Invoke Finish through alternate accessibility backend",
                {"ref": ref}, verifier="alternate.effect", intent=f"option_{index}",
            ),)

        def owns(self, candidate):
            return candidate.capability == "desktop.click"

        def execute(self, candidate, observation_id, decision_id):
            return execute_with_receipt(
                candidate, observation_id, decision_id,
                lambda: (self.window.control.invoke(), {"window_title": self.window.title})[1],
            )

        def register_verifiers(self, registry):
            registry.register("alternate.effect", lambda _candidate, receipt: Verification(
                receipt.success and self.window.title == "Done", "alternate.effect",
            ))

    class Planner:
        def plan(self, _goal, snapshot, _recent):
            options = []
            if "/done" not in snapshot.browser.url:
                options.append({"ref": "b:e0", "operation": "click"})
            if snapshot.desktop.window_title != "Done":
                options.append({"ref": "d:c0", "operation": "click"})
            return ComputerPlan.from_dict({"subgoal": "Finish both", "options": options}, snapshot)

    goal = "Finish both surfaces"
    planner = Planner()
    alternate = AlternateAccessibilitySurface()
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url="https://example.test/start",
            planner=planner, page=BrowserPage(),
        ),
        desktop=None, desktop_surface=alternate,
        required_url_contains="/done", required_window_title_contains="Done",
        required_capabilities=("desktop.click",),
    )

    def factory(item):
        executors = ExecutorRegistry()
        for channel, executor in item.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()), guard=ActionGuard(),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=4)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert {step.receipt.capability for step in result.steps} == {"browser.click", "desktop.click"}


def test_model_projection_keeps_grounded_refs_without_runtime_handles(tmp_path):
    task = environment(tmp_path)
    task.reset()
    task.observe(())
    assert task._snapshot is not None
    raw = task._snapshot.to_dict()
    compact = task._snapshot.for_model()
    assert raw["desktop"]["window_handle"] == 42
    assert compact["browser"]["elements"][0]["ref"] == "b:e0"
    assert compact["desktop"]["controls"][0]["ref"] == "d:c0"
    assert compact["tool_offers"][0]["ref"] == "t:t0"
    wire = json.dumps(compact)
    for runtime_only in ("window_handle", "automation_id", "rectangle", '"index"'):
        assert runtime_only not in wire
    assert len(wire) < len(json.dumps(raw))
    task.close()


def test_editor_style_text_goal_uses_private_live_edit_value():
    class Planner:
        def plan(self, _goal, snapshot, _recent):
            options = []
            if "Ada" not in snapshot.desktop.edit_values:
                options.append({"ref": "d:c0", "operation": "fill", "value": "Ada"})
            if "/done" not in snapshot.browser.url:
                options.append({"ref": "b:e0", "operation": "click"})
            return ComputerPlan.from_dict({"subgoal": "Edit and navigate", "options": options}, snapshot)

    goal = "Write Ada in the editor and open the result page"
    planner = Planner()
    window = FakeWindow()
    before_edit = DesktopSnapshot.capture(window)[0]
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url="https://example.test/start",
            planner=planner, page=BrowserPage(),
        ),
        desktop=OpenDesktopTask(
            goal=goal, window_title_re="Example", planner=planner,
            window=window, allow_text_input=True,
        ),
        required_url_contains="/done", required_window_text_contains="Ada",
        required_capabilities=("desktop.fill",),
    )

    def factory(item):
        executors = ExecutorRegistry()
        for channel, executor in item.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()), guard=ActionGuard(allow_writes=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=4)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert {step.receipt.capability for step in result.steps} == {"desktop.fill", "browser.click"}
    assert window.controls[0].value == "Ada"
    after_edit = DesktopSnapshot.capture(window)[0]
    assert after_edit.edit_values == ("Ada",)
    assert before_edit.fingerprint() != after_edit.fingerprint()
    assert "Ada" not in json.dumps(task._snapshot.to_dict()["desktop"])
