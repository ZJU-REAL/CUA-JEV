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
from cua_jev.open_computer import ComputerPlan, OpenComputerTask
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
