import json
import sys
from pathlib import Path

import pytest
from test_open_computer import BrowserPage, DesktopWindow

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
    with pytest.raises(ValueError, match="bounded JSON objects"):
        McpToolSurface.from_profile(config)
    with pytest.raises(SystemExit, match="requires explicit --allow-mcp-actions"):
        main([
            "open-computer", "--goal", "test", "--url", "https://example.test",
            "--window-title", "^Example$", "--model-base-url", "https://model.test/v1",
            "--model", "text-model", "--mcp-profile", str(config),
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
