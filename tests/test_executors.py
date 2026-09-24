import subprocess
import sys
from pathlib import Path

from cua_jev.executors import (
    FileSystemExecutor,
    RegisteredCliExecutor,
    StdioMcpExecutor,
    StdioMcpServer,
    VSCodeExecutor,
)
from cua_jev.local_tools import verify_readonly_tool_result
from cua_jev.models import ActionCandidate, Channel


def test_filesystem_executor_failure_is_receipt(tmp_path):
    candidate = ActionCandidate(
        "missing", Channel.API, "filesystem.read_text", "Read missing", {"path": str(tmp_path / "missing")}
    )
    receipt = FileSystemExecutor()(candidate, "obs", "decision")
    assert not receipt.success
    assert "FileNotFoundError" in receipt.error


def test_registered_cli_does_not_use_shell(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, "Python 3", "")

    monkeypatch.setattr(subprocess, "run", run)
    candidate = ActionCandidate("version", Channel.CLI, "cli.python_version", "Read Python version")
    receipt = RegisteredCliExecutor()(candidate, "obs", "decision")
    assert receipt.success
    assert seen["kwargs"]["shell"] is False
    assert seen["argv"] == ["python", "--version"]


def test_powershell_version_is_a_fixed_readonly_command(monkeypatch):
    seen = {}

    def run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, "PowerShell 5.1.0\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    candidate = ActionCandidate(
        "ps-version", Channel.CLI, "cli.powershell_version", "Read PowerShell version",
    )
    receipt = RegisteredCliExecutor()(candidate, "obs", "decision")
    assert receipt.success
    assert seen["kwargs"]["shell"] is False
    assert seen["argv"][:4] == [
        "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
    ]
    assert verify_readonly_tool_result(candidate, receipt).passed


def test_vscode_uses_goto_argv_without_shell(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr("shutil.which", lambda name: "C:/bin/code.cmd" if name == "code" else None)

    def run(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    candidate = ActionCandidate(
        "goto",
        Channel.API,
        "vscode.goto",
        "Open source location",
        {"path": str(tmp_path / "main.py"), "line": 7, "column": 2},
    )
    receipt = VSCodeExecutor()(candidate, "obs", "decision")
    assert receipt.success
    assert seen["kwargs"]["shell"] is False
    assert seen["argv"][-1].endswith("main.py:7:2")


def test_stdio_mcp_rejects_unregistered_server_before_launch():
    candidate = ActionCandidate(
        "tool",
        Channel.MCP,
        "mcp.call_tool",
        "Call an MCP tool",
        {"server": "unknown", "tool": "read_file", "arguments": {}},
    )
    receipt = StdioMcpExecutor()(candidate, "obs", "decision")
    assert not receipt.success
    assert "not allowlisted" in receipt.error


def test_stdio_mcp_calls_real_protocol_server():
    server_path = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"
    executor = StdioMcpExecutor()
    executor.register_server(
        "echo",
        StdioMcpServer(sys.executable, (str(server_path),)),
        tools={"echo"},
    )
    candidate = ActionCandidate(
        "tool",
        Channel.MCP,
        "mcp.call_tool",
        "Call the allowlisted echo tool",
        {"server": "echo", "tool": "echo", "arguments": {"text": "hello"}},
    )
    receipt = executor(candidate, "obs", "decision")
    assert receipt.success, receipt.error
    assert receipt.output["structured_content"] == {"echo": "hello"}
