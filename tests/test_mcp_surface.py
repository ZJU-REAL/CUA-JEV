import asyncio
import json
import sys
from pathlib import Path

import pytest
from test_open_computer import BrowserPage, DesktopWindow

from cua_jev.artifact_surface import ArtifactSurface
from cua_jev.cli import main
from cua_jev.episode import EpisodeConfig, EpisodeRunner, EpisodeStatus
from cua_jev.executors import ControlExecutor
from cua_jev.guard import ActionGuard
from cua_jev.mcp_surface import McpToolSurface, verify_mcp_structured_result
from cua_jev.models import ActionReceipt, Channel, Risk
from cua_jev.open_browser import OpenBrowserTask, PublicDecisionPolicy
from cua_jev.open_computer import ComputerPlan, OpenComputerTask
from cua_jev.open_desktop import OpenDesktopTask
from cua_jev.policy import RulePolicy
from cua_jev.registry import ExecutorRegistry
from cua_jev.runtime import AgentRuntime
from cua_jev.verify import VerifierRegistry


def profile(tmp_path: Path, *, expected=None) -> Path:
    source = tmp_path / "source.txt"
    source.write_text("verified source content", encoding="utf-8")
    file = tmp_path / "trusted-mcp.json"
    file.write_text(json.dumps({
        "server": {
            "name": "fixture", "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "mcp_echo_server.py")],
        },
        "tools": [{
            "name": "read_text", "description": "Read the caller-selected source file",
            "arguments": {"path": str(source)},
            "expected": expected if expected is not None else {"text": "verified source content"},
        }],
    }), encoding="utf-8")
    return file


def test_profile_binds_command_arguments_and_expected_result(tmp_path):
    surface = McpToolSurface.from_profile(profile(tmp_path))
    state = surface.capture()
    assert state[0].to_dict()["tool"] == "read_text"
    assert "command" not in state[0].to_dict()
    assert surface.validate(state, "m0", "invoke", "") == ("m0", "invoke", "")
    with pytest.raises(ValueError, match="without arguments"):
        surface.validate(state, "m0", "invoke", '{"path":"model override"}')
    candidate = surface.compile(state, "m0", "invoke", "", "Read source", 2)[0]
    assert candidate.risk == Risk.EXTERNAL_SIDE_EFFECT
    assert candidate.arguments["arguments"]["path"] == str(tmp_path / "source.txt")
    assert candidate.intent == "option_2"
    receipt = ActionReceipt(
        "obs", "decision", candidate.id, Channel.MCP, candidate.capability,
        True, 0, 1, {"structured_content": {"text": "wrong"}},
    )
    assert not verify_mcp_structured_result(candidate, receipt).passed


def test_schema_bounded_unseen_mcp_parameter_executes_real_protocol(tmp_path):
    pytest.importorskip("mcp")
    config = profile(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["tools"] = [{
        "name": "echo", "description": "Echo a short caller-bounded query",
        "parameters": {"text": {"type": "string", "maxLength": 40}},
        "required": ["text"],
        "expected_from_arguments": {"echo": "text"},
    }]
    config.write_text(json.dumps(data), encoding="utf-8")
    surface = McpToolSurface.from_profile(config)
    state = surface.capture()
    assert state[0].to_dict()["parameters"]["text"]["maxLength"] == 40
    with pytest.raises(ValueError, match="unregistered field"):
        surface.validate(state, "m0", "invoke", '{"text":"hello","command":"evil"}')
    with pytest.raises(ValueError, match="string parameter"):
        surface.validate(state, "m0", "invoke", json.dumps({"text": "x" * 41}))
    canonical = surface.validate(state, "m0", "invoke", '{"text":"previously unseen"}')[2]
    candidate = surface.compile(state, "m0", "invoke", canonical, "Echo", 0)[0]
    assert candidate.arguments["arguments"] == {"text": "previously unseen"}
    receipt = surface.execute(candidate, "obs", "decision")
    assert receipt.success, receipt.error
    assert receipt.output["structured_content"] == {"echo": "previously unseen"}
    assert verify_mcp_structured_result(candidate, receipt).passed


def test_real_mcp_transport_works_while_caller_has_running_event_loop(tmp_path):
    pytest.importorskip("mcp")
    surface = McpToolSurface.from_profile(profile(tmp_path))
    candidate = surface.compile(surface.capture(), "m0", "invoke", "", "Read source", 0)[0]

    async def call_from_loop():
        return surface.execute(candidate, "obs", "decision")

    receipt = asyncio.run(call_from_loop())
    assert receipt.success, receipt.error
    assert verify_mcp_structured_result(candidate, receipt).passed


def test_profile_rejects_relative_executable_and_missing_opt_in(tmp_path):
    config = profile(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["server"]["command"] = "python"
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="absolute executable"):
        McpToolSurface.from_profile(config)
    data["server"]["command"] = sys.executable
    data["tools"][0]["expected"] = {}
    config.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="expected result assertion"):
        McpToolSurface.from_profile(config)
    with pytest.raises(SystemExit, match="requires explicit --allow-mcp-actions"):
        main([
            "open-computer", "--goal", "test", "--url", "https://example.test",
            "--window-title", "^Example$", "--model-base-url", "https://model.test/v1",
            "--model", "text-model", "--mcp-profile", str(config),
        ])
    with pytest.raises(SystemExit, match="require --window-title"):
        main([
            "open-computer", "--goal", "test", "--url", "https://example.test",
            "--model-base-url", "https://model.test/v1", "--model", "text-model",
            "--allow-window-actions", "--require-url-contains", "/done",
        ])


def test_real_mcp_file_read_joins_dom_and_uia_in_one_episode(tmp_path):
    pytest.importorskip("mcp")
    surface = McpToolSurface.from_profile(profile(tmp_path))

    class Planner:
        def plan(self, _goal, snapshot, _recent):
            options = []
            if "mcp.fixture.read_text" not in snapshot.requirements["completed_capabilities"]:
                options.append({"ref": "m:m0", "operation": "invoke"})
            if "/done" not in snapshot.browser.url:
                options.append({"ref": "b:e0", "operation": "click"})
            if snapshot.desktop.window_title != "Done":
                options.append({"ref": "d:c0", "operation": "click"})
            return ComputerPlan.from_dict({"subgoal": "Finish all three", "options": options}, snapshot)

    goal = "Use a registered MCP tool, browser and desktop window"
    planner = Planner()
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url="https://example.test/start", planner=planner,
            page=BrowserPage(),
        ),
        desktop=OpenDesktopTask(
            goal=goal, window_title_re="Example", planner=planner,
            window=DesktopWindow(), allow_button_actions=True,
        ),
        mcp_surface=surface,
        required_url_contains="/done", required_window_title_contains="Done",
        required_capabilities=("mcp.fixture.read_text", "desktop.click"),
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
            guard=ActionGuard(allowed_roots=[tmp_path], allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=5)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert {step.receipt.channel for step in result.steps} == {
        Channel.MCP, Channel.SCRIPT, Channel.API,
    }
    assert all(step.verification.passed for step in result.steps)


def test_browser_and_real_mcp_run_without_windows_desktop(tmp_path):
    pytest.importorskip("mcp")
    surface = McpToolSurface.from_profile(profile(tmp_path))

    class Planner:
        def plan(self, _goal, snapshot, _recent):
            assert snapshot.desktop is None
            options = []
            if "mcp.fixture.read_text" not in snapshot.requirements["completed_capabilities"]:
                options.append({"ref": "m:m0", "operation": "invoke"})
            if "/done" not in snapshot.browser.url:
                options.append({"ref": "b:e0", "operation": "click"})
            return ComputerPlan.from_dict({"subgoal": "Read and navigate", "options": options}, snapshot)

    goal = "Read the registered source and open the result page"
    planner = Planner()
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(
            goal=goal, start_url="https://example.test/start", planner=planner,
            page=BrowserPage(),
        ),
        mcp_surface=surface, required_url_contains="/done",
        required_capabilities=("mcp.fixture.read_text",),
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
            guard=ActionGuard(allowed_roots=[tmp_path], allow_destructive=True),
            executors=executors, verifiers=verifiers,
        )

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=4)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert {step.receipt.channel for step in result.steps} == {Channel.MCP, Channel.SCRIPT}
    assert all(step.verification.passed for step in result.steps)


@pytest.mark.parametrize("per_visit", [False, True])
def test_current_page_mcp_mode_reads_each_browser_url_once(tmp_path, per_visit):
    pytest.importorskip("mcp")
    config = profile(tmp_path)
    data = json.loads(config.read_text(encoding="utf-8"))
    data["tools"] = [{
        "name": "fetch_page", "description": "Read the current page URL",
        "parameters": {"href": {"type": "string", "maxLength": 200}},
        "required": ["href"],
        "expected_from_arguments": {"url": "href"},
    }]
    config.write_text(json.dumps(data), encoding="utf-8")

    class Planner:
        def plan(self, _goal, snapshot, _recent):
            if snapshot.mcp_offers:
                option = {"ref": "m:m0", "operation": "invoke", "value": json.dumps({
                    "href": snapshot.browser.url,
                })}
            elif "/done" not in snapshot.browser.url:
                option = {"ref": "b:e0", "operation": "click"}
            else:
                option = {"ref": "a:a0", "operation": "write", "value": (
                    "# Verified note\nhttps://example.test/done\n"
                )}
            return ComputerPlan.from_dict({"subgoal": "Read and record", "options": [option]}, snapshot)

    goal = "Read both pages once and create a note"
    planner = Planner()
    task = OpenComputerTask(
        goal=goal, planner=planner,
        browser=OpenBrowserTask(goal=goal, start_url="https://example.test/start",
                                planner=planner, page=BrowserPage()),
        mcp_surface=McpToolSurface.from_profile(config),
        mcp_current_page_only=True,
        mcp_read_for_visits=per_visit,
        artifact_surface=ArtifactSurface(tmp_path / "note.md"),
        required_visits=("/done",), required_url_contains="/done",
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

    result = EpisodeRunner(factory, EpisodeConfig(max_steps=6)).run(task)
    assert result.status == EpisodeStatus.SUCCESS, result.reason
    assert [step.receipt.capability for step in result.steps] == (
        ["browser.click", "mcp.call_tool", "artifact.write_text"] if per_visit else
        ["mcp.call_tool", "browser.click", "mcp.call_tool", "artifact.write_text"]
    )
    assert task._mcp_pages_read == (
        {"https://example.test/done"} if per_visit else
        {"https://example.test/start", "https://example.test/done"}
    )
