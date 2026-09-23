import io
import json
from types import SimpleNamespace

import httpx
import pytest

from cua_jev.episode import EpisodeConfig, EpisodeRunner, EpisodeStatus
from cua_jev.executors import ControlExecutor
from cua_jev.executors.cli import RegisteredCliExecutor
from cua_jev.executors.filesystem import FileSystemExecutor
from cua_jev.guard import ActionGuard
from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.open_browser import (
    BrowserPlan,
    BrowserSnapshot,
    HttpJsonPlanner,
    OpenBrowserTask,
    PublicDecisionPolicy,
)
from cua_jev.policy import JevPolicy, RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.verify import VerifierRegistry
from cua_jev.vision import VisualScene, VisualTarget


class FakeLocator:
    def __init__(self, page, index=0):
        self.page = page
        self.index = index

    def nth(self, index):
        return FakeLocator(self.page, index)

    def click(self):
        target = self.page.elements[self.index]
        self.page.url = target["href"] or "https://example.test/done"
        self.page.title_value = "Finished"
        self.page.text = "Completed"

    def fill(self, value):
        self.page.values[self.index] = value

    def input_value(self):
        return self.page.values.get(self.index, "")

    def select_option(self, value):
        self.page.values[self.index] = value


class FakePage:
    def __init__(self):
        self.url = "https://example.test/start"
        self.title_value = "Example"
        self.text = "Start"
        self.values = {}
        self.main_frame = object()
        self.elements = [
            {
                "ref": "e0", "index": 0, "tag": "a", "role": "", "label": "Finish",
                "input_type": "", "disabled": False, "href": "https://example.test/done",
            },
            {
                "ref": "e1", "index": 1, "tag": "a", "role": "", "label": "Other",
                "input_type": "", "disabled": False, "href": "https://example.test/other",
            },
            {
                "ref": "e2", "index": 2, "tag": "button", "role": "", "label": "Purchase",
                "input_type": "submit", "disabled": False, "href": "",
            },
            {
                "ref": "e3", "index": 3, "tag": "a", "role": "", "label": "Leave",
                "input_type": "", "disabled": False, "href": "https://outside.test/",
            },
        ]

    def evaluate(self, _script, _selector):
        return {
            "url": self.url, "title": self.title_value, "text": self.text,
            "elements": self.elements,
        }

    def route(self, _pattern, callback):
        self.route_callback = callback

    def goto(self, url, **_kwargs):
        self.url = url

    def locator(self, _selector):
        return FakeLocator(self)


def test_empty_transient_dom_waits_for_document_controls():
    class LoadingPage(FakePage):
        def __init__(self):
            super().__init__()
            self.ready = False
            self.waited = False

        def evaluate(self, script, selector):
            state = super().evaluate(script, selector)
            return state if self.ready else {**state, "elements": []}

        def wait_for_load_state(self, state, *, timeout):
            assert state == "domcontentloaded" and timeout <= 5000
            self.ready = True
            self.waited = True

        def wait_for_timeout(self, _delay):
            pass

    page = LoadingPage()
    task = OpenBrowserTask(
        goal="Open the result", start_url=page.url, planner=FakePlanner(), page=page,
    )
    snapshot = task._capture()
    assert page.waited
    assert snapshot.elements[0].ref == "e0"


class FakeVisualPage(FakePage):
    def __init__(self):
        super().__init__()
        self.elements = []
        self.color = "#f5f7fa"
        self.mouse = self
        self.clicks = []

    def screenshot(self, **_kwargs):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (400, 240), self.color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((250, 150, 370, 210), fill="#145ea8")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    def click(self, x, y):
        self.clicks.append((x, y))
        self.url = "https://example.test/done"
        self.title_value = "Finished"
        self.text = "Completed"


class FakeBrowserVision:
    def __init__(self):
        self.calls = 0

    def perceive_scene(self, goal, window_title, ui_text, image):
        self.calls += 1
        assert image.region == (0, 0, 400, 240)
        return VisualScene(
            "A page with a visible Finish button.", ("Finish",),
            (VisualTarget("v0", "Finish", (650, 625, 925, 875)),),
        )


class FakePlanner:
    def __init__(self, options=None):
        self.calls = 0
        self.options = options or [
            {"ref": "e0", "operation": "click"},
            {"ref": "e1", "operation": "click"},
        ]

    def plan(self, goal, snapshot, recent_actions):
        self.calls += 1
        assert goal == "Finish the example"
        assert snapshot.url == "https://example.test/start"
        assert isinstance(recent_actions, list)
        return BrowserPlan.from_dict(
            {
                "subgoal": "Choose a link",
                "options": self.options,
                "success": {"kind": "url_contains", "value": "/done"},
            },
            snapshot,
        )


def task(page=None, planner=None, **kwargs):
    return OpenBrowserTask(
        goal="Finish the example", start_url="https://example.test/start",
        planner=planner or FakePlanner(), page=page or FakePage(), **kwargs,
    )


def test_open_task_runs_with_dynamic_options_and_one_planning_call():
    environment = task()

    def factory(item):
        executors = ExecutorRegistry()
        executors.register(Channel.SCRIPT, item)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(), executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=5)).run(environment)
    assert result.status == EpisodeStatus.SUCCESS
    assert result.channel_counts == {"script": 1}
    assert result.steps[0].decision.candidate_id == "option_0_dom"
    assert result.steps[0].verification.passed
    assert environment.planner_calls == 1


def test_scoped_local_tool_and_browser_action_complete_one_open_goal(tmp_path):
    (tmp_path / "guide.md").write_text("public study notes", encoding="utf-8")
    (tmp_path / ".env").write_text("hidden", encoding="utf-8")
    environment = task(
        planner=FakePlanner([
            {"ref": "t0", "operation": "invoke"},
            {"ref": "e0", "operation": "click"},
        ]),
        local_tool_root=tmp_path, required_tool="filesystem.list",
        required_url_contains="/done",
    )

    def factory(item):
        executors = ExecutorRegistry()
        executors.register(Channel.SCRIPT, item)
        executors.register(Channel.API, FileSystemExecutor())
        executors.register(Channel.CLI, RegisteredCliExecutor())
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allowed_roots=[tmp_path]),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=5)).run(environment)
    assert result.status == EpisodeStatus.SUCCESS
    assert result.channel_counts == {"api": 1, "script": 1}
    assert [step.receipt.capability for step in result.steps] == [
        "filesystem.list", "browser.click",
    ]
    assert all(step.verification.passed for step in result.steps)


def test_tool_refs_are_scoped_and_not_model_defined(tmp_path):
    (tmp_path / "readme.md").write_text("safe", encoding="utf-8")
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    environment = task(local_tool_root=tmp_path)
    environment.reset()
    snapshot = environment._capture()
    assert {offer.capability for offer in snapshot.tool_offers} == {
        "filesystem.list", "cli.python_version", "filesystem.read_text",
    }
    assert all(".env" not in offer.description for offer in snapshot.tool_offers)
    read_offer = next(offer for offer in snapshot.tool_offers if offer.capability == "filesystem.read_text")
    candidate = read_offer.candidate(0)
    receipt = FileSystemExecutor()(candidate, "obs", "decision")
    assert receipt.success and receipt.output["text"] == "safe"
    (tmp_path / "readme.md").write_text("x" * 16_001, encoding="utf-8")
    stale = FileSystemExecutor()(candidate, "obs", "decision")
    assert not stale.success
    base = {"subgoal": "Inspect", "success": {"kind": "url_contains", "value": "/done"}}
    for option in (
        {"ref": "t99", "operation": "invoke"},
        {"ref": "t0", "operation": "click"},
        {"ref": "t0", "operation": "invoke", "value": "rm everything"},
        {"ref": "e0", "operation": "invoke"},
    ):
        with pytest.raises(ValueError):
            BrowserPlan.from_dict({**base, "options": [option]}, snapshot)
    environment.close()


def test_browser_vision_fuses_scene_and_routes_canvas_target_through_mouse_script():
    page = FakeVisualPage()
    grounder = FakeBrowserVision()
    environment = task(
        page=page, planner=FakePlanner([{"ref": "v0", "operation": "click"}]),
        vision_grounder=grounder, allow_screenshot_upload=True,
        allow_visual_clicks=True, allow_external_actions=True,
    )

    def factory(item):
        executors = ExecutorRegistry()
        executors.register(Channel.SCRIPT, item)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=3)).run(environment)
    assert result.status == EpisodeStatus.SUCCESS
    assert result.channel_counts == {"script": 1}
    assert grounder.calls == 1
    assert page.clicks == [(315.0, 180.0)]


def test_browser_vision_requires_opt_in_and_rejects_stale_viewport():
    page = FakeVisualPage()
    grounder = FakeBrowserVision()
    with pytest.raises(ValueError, match="screenshot-upload consent"):
        task(page=page, vision_grounder=grounder)
    environment = task(
        page=page, planner=FakePlanner([{"ref": "v0", "operation": "click"}]),
        vision_grounder=grounder, allow_screenshot_upload=True,
        allow_visual_clicks=True, allow_external_actions=True,
    )
    environment.reset()
    observation = environment.observe(())
    assert observation.state["visual_summary"] == "A page with a visible Finish button."
    assert observation.state["visual_text"] == ["Finish"]
    candidate = environment.candidates(observation, ())[0]
    page.color = "#000000"
    receipt = environment(candidate, observation.observation_id, "d1")
    assert not receipt.success
    assert "stale" in (receipt.error or "")
    assert not page.clicks
    environment.close()


def test_browser_vision_failure_falls_back_to_live_dom_when_available():
    class FailingVision:
        def perceive_scene(self, *_args):
            raise ValueError("malformed VLM response")

    page = FakeVisualPage()
    page.elements = FakePage().elements
    environment = task(
        page=page, vision_grounder=FailingVision(),
        vision_mode="always", allow_screenshot_upload=True,
    )
    environment.reset()
    observation = environment.observe(())
    assert observation.state["visual_summary"] == ""
    assert environment.vision_failures == 1
    assert environment.candidates(observation, ())[0].id == "option_0_dom"
    environment.close()


def test_dom_and_visual_alternatives_require_matching_label_and_box():
    page = FakeVisualPage()
    page.elements = [
        {**FakePage().elements[0], "box": [650, 625, 925, 875]},
    ]
    environment = task(
        page=page, planner=FakePlanner([
            {"ref": "e0", "operation": "click"},
            {"ref": "v0", "operation": "click"},
        ]),
        vision_grounder=FakeBrowserVision(), vision_mode="always",
        allow_screenshot_upload=True, allow_visual_clicks=True,
        allow_external_actions=True,
    )
    environment.reset()
    observation = environment.observe(())
    assert {item.id for item in environment.candidates(observation, ())} == {
        "option_0_dom", "option_0_visual",
    }
    environment.close()


def test_browser_channel_and_normalized_dom_boxes_are_bounded():
    with pytest.raises(ValueError, match="browser channel"):
        task(browser_channel="unknown")
    page = FakePage()
    page.elements[0]["box"] = [100, 200, 300, 260]
    page.elements[1]["box"] = [-1, 0, 100, 100]
    snapshot = BrowserSnapshot.capture(page)
    assert snapshot.elements[0].box == (100, 200, 300, 260)
    assert snapshot.elements[1].box is None


def test_candidate_generation_filters_external_and_button_actions():
    environment = task(
        planner=FakePlanner(
            [
                {"ref": "e0", "operation": "click"},
                {"ref": "e2", "operation": "click"},
                {"ref": "e3", "operation": "click"},
            ]
        ),
        headed=True,
    )
    environment.reset()
    observation = environment.observe(())
    candidates = environment.candidates(observation, ())
    assert {candidate.id for candidate in candidates} == {"option_0_dom", "option_0_gui"}
    environment.close()


def test_changed_browser_element_invalidates_old_plan():
    page = FakePage()
    planner = FakePlanner()
    environment = task(page=page, planner=planner)
    environment.reset()
    first = environment.observe(())
    environment.candidates(first, ())
    page.elements[0]["label"] = "New target"
    second = environment.observe(())
    environment.candidates(second, ())
    assert planner.calls == 2
    environment.close()


def test_one_plan_can_cover_multiple_form_actions():
    page = FakePage()
    page.elements.extend(
        [
            {
                "ref": "e4", "index": 4, "tag": "input", "role": "", "label": "First name",
                "input_type": "text", "disabled": False, "href": "",
            },
            {
                "ref": "e5", "index": 5, "tag": "input", "role": "", "label": "Last name",
                "input_type": "text", "disabled": False, "href": "",
            },
        ]
    )
    planner = FakePlanner(
        [
            {"ref": "e4", "operation": "fill", "value": "Ada"},
            {"ref": "e5", "operation": "fill", "value": "Lovelace"},
            {"ref": "e2", "operation": "click"},
        ]
    )
    environment = task(
        page=page, planner=planner, allow_form_input=True, allow_external_actions=True
    )

    def factory(item):
        executors = ExecutorRegistry()
        executors.register(Channel.SCRIPT, item)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allow_writes=True, allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=5)).run(environment)
    assert result.status == EpisodeStatus.SUCCESS
    assert len(result.steps) == 3
    assert planner.calls == 1
    assert page.values[4] == "Ada"
    assert page.values[5] == "Lovelace"


def test_planner_cannot_invent_element_or_use_password_control():
    snapshot = BrowserSnapshot.capture(FakePage())
    base = {"subgoal": "Continue", "success": {"kind": "url_contains", "value": "/done"}}
    with pytest.raises(ValueError, match="unavailable"):
        BrowserPlan.from_dict({**base, "options": [{"ref": "e99", "operation": "click"}]}, snapshot)
    click_with_null_value = BrowserPlan.from_dict(
        {**base, "options": [{"ref": "e0", "operation": "click", "value": None}]}, snapshot
    )
    assert click_with_null_value.options[0].value == ""
    page = FakePage()
    page.elements.append(
        {
            "ref": "e4", "index": 4, "tag": "input", "role": "", "label": "Password",
            "input_type": "password", "disabled": False, "href": "",
        }
    )
    snapshot = BrowserSnapshot.capture(page)
    with pytest.raises(ValueError, match="password"):
        BrowserPlan.from_dict(
            {**base, "options": [{"ref": "e4", "operation": "fill", "value": "private"}]},
            snapshot,
        )
    page.elements[-1]["input_type"] = "file"
    snapshot = BrowserSnapshot.capture(page)
    with pytest.raises(ValueError, match="supported text input"):
        BrowserPlan.from_dict(
            {**base, "options": [{"ref": "e4", "operation": "fill", "value": "file.txt"}]},
            snapshot,
        )


def test_select_verification_checks_value_even_when_snapshot_does_not_change():
    page = FakePage()
    page.elements.append(
        {
            "ref": "e4", "index": 4, "tag": "select", "role": "", "label": "Category",
            "input_type": "", "disabled": False, "href": "",
        }
    )
    environment = task(
        page=page,
        planner=FakePlanner([{"ref": "e4", "operation": "select", "value": "books"}]),
        allow_form_input=True,
    )
    environment.reset()
    observation = environment.observe(())
    candidate = environment.candidates(observation, ())[0]
    receipt = environment(candidate, observation.observation_id, "d1")
    verifier = VerifierRegistry()
    environment.register_verifiers(verifier)
    assert verifier.verify(candidate, receipt).passed
    assert page.values[4] == "books"
    environment.close()


def test_public_policy_does_not_send_page_text_or_action_arguments():
    def handler(request):
        body = json.loads(request.content)
        assert "private page text" not in request.content.decode()
        assert "private input" not in request.content.decode()
        assert "private OCR line" not in request.content.decode()
        assert "private tool output" not in request.content.decode()
        assert body["state"]["state"] == {"url": "https://example.test"}
        assert body["state"]["available_actions"][0]["arguments"] == {}
        return httpx.Response(
            200,
            json={
                "answers": {
                    "action": {
                        "choice": "fill", "probabilities": {"fill": 1.0}, "confidence": 1.0,
                    }
                }
            },
        )

    inner = JevPolicy(
        api_key="fake", client=httpx.Client(transport=httpx.MockTransport(handler)), retries=0
    )
    policy = PublicDecisionPolicy(inner)
    decision = policy.choose(
        Observation("test", "test", {
            "url": "https://example.test", "text": "private page text",
            "visual_text": ["private OCR line"],
            "tool_results": [{"result": "private tool output"}],
        }),
        [ActionCandidate("fill", Channel.SCRIPT, "browser.fill", "Fill a field", {"value": "private input"})],
    )
    assert decision.candidate_id == "fill"


def test_http_planner_validates_structured_response_and_authorizes():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer fake"
        assert json.loads(request.content)["goal"] == "Finish the example"
        return httpx.Response(
            200,
            json={
                "subgoal": "Choose a link",
                "options": [{"ref": "e0", "operation": "click"}],
                "success": {"kind": "url_contains", "value": "/done"},
            },
        )

    planner = HttpJsonPlanner(
        "http://127.0.0.1:9999/plan", api_key="fake",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    plan = planner.plan("Finish the example", BrowserSnapshot.capture(FakePage()), [])
    assert plan.options[0].ref == "e0"
    with pytest.raises(ValueError, match="HTTPS"):
        HttpJsonPlanner("http://example.test/plan")


def test_navigation_route_blocks_cross_origin_requests():
    environment = task()
    environment.reset()

    class Route:
        def __init__(self, url):
            self.request = SimpleNamespace(
                url=url, frame=environment.page.main_frame,
                is_navigation_request=lambda: True,
            )
            self.status = ""

        def abort(self):
            self.status = "aborted"

        def fallback(self):
            self.status = "continued"

    external = Route("https://outside.test/")
    environment._route_request(external)
    assert external.status == "aborted"
    internal = Route("https://example.test/next")
    environment._route_request(internal)
    assert internal.status == "continued"
    environment.close()


def test_real_browser_snapshot_and_dom_execution_without_external_site():
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
        except Exception as exc:
            pytest.skip(f"Edge browser unavailable: {type(exc).__name__}")
        page = browser.new_page()

        def fixture(route):
            if route.request.url.endswith("/done"):
                body = "<title>Finished</title><main>Completed</main>"
            else:
                body = (
                    "<title>Example</title><main>Start</main>"
                    '<a href="/done">Finish</a><a href="/other">Other</a>'
                )
            route.fulfill(status=200, content_type="text/html", body=body)

        page.route("https://example.test/**", fixture)
        environment = task(page=page)

        def factory(item):
            executors = ExecutorRegistry()
            executors.register(Channel.SCRIPT, item)
            executors.register(Channel.CONTROL, ControlExecutor())
            verifiers = VerifierRegistry()
            item.register_verifiers(verifiers)
            return AgentRuntime(
                policy=PublicDecisionPolicy(RulePolicy()),
                guard=ActionGuard(), executors=executors, verifiers=verifiers,
            )

        result = EpisodeRunner(factory, EpisodeConfig(max_steps=5)).run(environment)
        assert result.status == EpisodeStatus.SUCCESS
        assert result.steps[0].verification.passed
        browser.close()
