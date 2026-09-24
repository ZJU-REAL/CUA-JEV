import json
from pathlib import Path

import pytest
from test_mcp_surface import profile

from cua_jev.artifact_surface import ArtifactSurface
from cua_jev.episode import EpisodeConfig, EpisodeRunner, EpisodeStatus
from cua_jev.executors import ControlExecutor
from cua_jev.guard import ActionGuard
from cua_jev.mcp_surface import McpToolSurface
from cua_jev.models import Channel
from cua_jev.open_browser import OpenBrowserTask, PublicDecisionPolicy
from cua_jev.open_computer import ComputerPlan, OpenComputerTask, _page_key
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.verify import VerifierRegistry


class JourneyLocator:
    def __init__(self, page):
        self.page = page
        self.index = 0

    def nth(self, index):
        self.index = index
        return self

    def click(self):
        self.page.url = self.page.links()[self.index][1]


class JourneyPage:
    hub = "https://example.test/index"

    def __init__(self, count: int):
        self.count = count
        self.url = self.hub
        self.main_frame = object()

    def links(self):
        if self.url == self.hub:
            return [(f"Topic {index}", f"https://example.test/topic/{index}")
                    for index in range(self.count)]
        return [("All topics", self.hub)]

    def evaluate(self, _script, _selector):
        title = "Index" if self.url == self.hub else self.url.rsplit("/", 1)[-1]
        return {
            "url": self.url, "title": title,
            "text": f"Source material for {title}. This page has a distinct verifiable URL.",
            "elements": [{
                "ref": f"e{index}", "index": index, "tag": "a", "role": "",
                "label": label, "input_type": "", "disabled": False, "href": href,
            } for index, (label, href) in enumerate(self.links())],
        }

    def route(self, _pattern, _callback):
        pass

    def goto(self, url, **_kwargs):
        self.url = url

    def locator(self, _selector):
        return JourneyLocator(self)


def test_twenty_external_actions_across_browser_cli_mcp_and_file_api(tmp_path: Path):
    pytest.importorskip("mcp")
    count = 9
    clues = tuple(f"/topic/{index}" for index in range(count))
    artifact_path = tmp_path / "study-guide.md"

    class Planner:
        def plan(self, _goal, snapshot, _recent):
            completed = snapshot.requirements["completed_capabilities"]
            if "cli.python_version" not in completed:
                option = {"ref": "t:t1", "operation": "invoke"}
            elif "mcp.fixture.read_text" not in completed:
                option = {"ref": "m:m0", "operation": "invoke"}
            elif snapshot.requirements["pending_visits"]:
                clue = snapshot.requirements["pending_visits"][0]
                if snapshot.browser.url == JourneyPage.hub:
                    target = next(element for element in snapshot.browser.elements
                                  if clue in element.href)
                else:
                    target = snapshot.browser.elements[0]
                option = {"ref": f"b:{target.ref}", "operation": "click"}
            else:
                lines = ["# Study guide", "Source-grounded index:"]
                lines.extend(f"- {record['title']}: {record['url']}" for record in snapshot.source_records)
                option = {"ref": "a:a0", "operation": "write", "value": "\n".join(lines) + "\n"}
            return ComputerPlan.from_dict(
                {"subgoal": "Advance the study guide", "options": [option]}, snapshot
            )

    goal = "Inspect nine sources and create a cited study guide"
    planner = Planner()
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url=JourneyPage.hub, planner=planner,
            page=JourneyPage(count),
        ),
        local_tool_root=tmp_path,
        mcp_surface=McpToolSurface.from_profile(profile(tmp_path)),
        artifact_surface=ArtifactSurface(artifact_path),
        required_visits=clues,
        required_artifact_contains=("# Study guide",),
        required_capabilities=("cli.python_version", "mcp.fixture.read_text"),
    )

    def factory(item):
        executors = ExecutorRegistry()
        for channel, executor in item.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allowed_roots=[tmp_path], allow_writes=True,
                              allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=24)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert len(result.steps) == 20
    assert result.channel_counts == {"cli": 1, "mcp": 1, "script": 17, "api": 1}
    assert {step.receipt.channel for step in result.steps} == {
        Channel.CLI, Channel.MCP, Channel.SCRIPT, Channel.API,
    }
    assert all(step.verification.passed for step in result.steps)
    text = artifact_path.read_text(encoding="utf-8")
    assert all(f"https://example.test/topic/{index}" in text for index in range(count))


def test_model_selected_sources_need_verified_mcp_reads_before_artifact(
    tmp_path: Path,
):
    pytest.importorskip("mcp")
    config = profile(tmp_path)
    settings = json.loads(config.read_text(encoding="utf-8"))
    settings["tools"] = [{
        "name": "fetch_page", "description": "Read the current test page",
        "parameters": {"href": {"type": "string", "maxLength": 200}},
        "required": ["href"], "expected_from_arguments": {"url": "href"},
    }]
    config.write_text(json.dumps(settings), encoding="utf-8")
    artifact_path = tmp_path / "new-guide.md"
    editor = {"title": ""}

    def launch():
        editor["title"] = "new-guide.md - Visual Studio Code"


    class Planner:
        def plan(self, _goal, snapshot, _recent):
            requirements = snapshot.requirements
            if requirements["pending_mcp_page"]:
                option = {
                    "ref": "m:m0", "operation": "invoke",
                    "value": json.dumps({"href": snapshot.browser.url}),
                }
            elif (
                requirements["verified_source_count"] < 3
                or requirements["pending_source_titles"]
            ):
                if snapshot.browser.url == JourneyPage.hub:
                    cited = {item["url"] for item in snapshot.source_records}
                    pending = (
                        requirements["pending_source_titles"]
                        if requirements["verified_source_count"] >= 3 else []
                    )
                    target = next(
                        item for item in snapshot.browser.elements
                        if item.href not in cited
                        and (not pending or pending[0] in item.label)
                    )
                else:
                    target = snapshot.browser.elements[0]
                option = {"ref": f"b:{target.ref}", "operation": "click"}
            elif not snapshot.artifact.exists:
                option = {
                    "ref": "a:a0", "operation": "write",
                    "value": "# Newly discovered sources\n" + "\n".join(
                        item["url"] for item in snapshot.source_records
                    ),
                }
            else:
                option = {"ref": "a:a0", "operation": "open"}
            return ComputerPlan.from_dict({"subgoal": "Research", "options": [option]}, snapshot)

    goal = "Choose any three relevant source pages and cite them in a new guide"
    planner = Planner()
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url=JourneyPage.hub, planner=planner,
            page=JourneyPage(5),
        ),
        mcp_surface=McpToolSurface.from_profile(config),
        mcp_current_page_only=True,
        min_verified_sources=3,
        required_source_titles=("4",),
        artifact_surface=ArtifactSurface(
            artifact_path, allow_open_vscode=True,
            editor_probe=lambda: editor["title"],
            editor_launcher=launch,
        ),
        required_capabilities=("artifact.open_vscode",),
    )

    def factory(item):
        executors = ExecutorRegistry()
        for channel, executor in item.executor_bindings().items():
            executors.register(channel, executor)
        executors.register(Channel.CONTROL, ControlExecutor())
        verifiers = VerifierRegistry()
        item.register_verifiers(verifiers)
        return AgentRuntime(
            policy=PublicDecisionPolicy(RulePolicy()),
            guard=ActionGuard(allowed_roots=[tmp_path], allow_writes=True,
                              allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=14)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, (
        result.reason,
        [(step.receipt.capability, step.receipt.success) for step in result.steps],
        task._pending_source_titles(),
    )
    assert len(result.steps) == 13
    assert result.channel_counts == {"script": 7, "mcp": 4, "api": 1, "cli": 1}
    text = artifact_path.read_text(encoding="utf-8")
    assert "https://example.test/topic/0" in text
    assert "https://example.test/topic/1" in text
    assert "https://example.test/topic/2" in text
    assert "https://example.test/topic/4" in text


def test_model_discovered_sources_require_current_page_mcp_and_artifact(tmp_path: Path):
    goal = "Research new pages"
    planner = object()
    browser = OpenBrowserTask(
        goal=goal, start_url=JourneyPage.hub, planner=planner,
        page=JourneyPage(5),
    )
    with pytest.raises(ValueError, match="discovered sources need"):
        OpenComputerTask(
            goal=goal, planner=planner, browser=browser,
            min_verified_sources=2, artifact_surface=ArtifactSurface(tmp_path / "note.md"),
        )


def test_source_identity_ignores_in_page_fragments():
    assert _page_key("https://example.test/topic/0#intro") == (
        _page_key("https://example.test/topic/0#examples")
    )
    assert _page_key("https://example.test/topic/0#intro") != (
        _page_key("https://example.test/topic/1#intro")
    )
