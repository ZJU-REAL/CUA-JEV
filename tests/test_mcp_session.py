import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("mcp")
pytest.importorskip("jsonschema")

from cua_jev.executors.mcp import StdioMcpServer
from cua_jev.mcp_session import McpSessionError, PersistentMcpSession


def make_session(tmp_path, **kwargs):
    events = tmp_path / "events.jsonl"
    session = PersistentMcpSession(
        server=StdioMcpServer(sys.executable, (
            str(Path(__file__).parent / "fixtures/mcp_counter_server.py"), str(events),
        )), connect_timeout_s=15, **kwargs,
    )
    return session, events


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def assert_process_gone(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def tool_data(result):
    assert result["isError"] is False
    return result["structuredContent"] or json.loads(result["content"][0]["text"])


def test_reuses_one_real_process_and_initialization_across_calls(tmp_path):
    session, events = make_session(tmp_path)
    try:
        catalog = session.list_tools()
        assert [tool["name"] for tool in catalog] == ["count_call"]
        catalog[0]["inputSchema"]["properties"].clear()
        assert "value" in session.list_tools()[0]["inputSchema"]["properties"]
        first = tool_data(session.call_tool("count_call", {"value": 4}))
        second = tool_data(session.call_tool("count_call", {"value": 5}))
        assert first == {"value": 4, "count": 1, "pid": second["pid"]}
        assert second["count"] == 2
        assert session.metrics["initializations"] == 1
        assert session.metrics["tool_calls"] == 2
        assert [event["event"] for event in read_events(events)] == ["start", "call", "call"]
    finally:
        session.close()
    session.close()
    assert session.metrics["cleanup_confirmed"]
    assert_process_gone(first["pid"])
    with pytest.raises(McpSessionError, match="closed"):
        session.call_tool("count_call", {"value": 6})


def test_invalid_arguments_and_unknown_tools_never_reach_server(tmp_path):
    session, events = make_session(tmp_path, allowed_tools={"count_call"})
    try:
        session.start()
        for name, arguments in (("unknown", {}), ("count_call", {}),
                                ("count_call", {"value": "bad"}), ("count_call", {"value": float("nan")})):
            with pytest.raises(ValueError):
                session.call_tool(name, arguments)
        assert [event["event"] for event in read_events(events)] == ["start"]
        assert session.metrics["tool_calls"] == 0
    finally:
        session.close()


def test_tool_error_is_observation_and_is_not_retried(tmp_path):
    session, events = make_session(tmp_path)
    try:
        result = session.call_tool("count_call", {"value": 1, "fail": True})
        assert result["isError"] is True
        assert tool_data(session.call_tool("count_call", {"value": 2}))["count"] == 2
        assert sum(e["event"] == "call" for e in read_events(events)) == 2
    finally:
        session.close()


def test_timeout_retires_session_without_replaying_possibly_executed_action(tmp_path):
    session, events = make_session(tmp_path, call_timeout_s=10)
    try:
        session.start()
        with pytest.raises(McpSessionError, match="outcome_unknown"):
            session.call_tool("count_call", {"value": 1, "delay": 5}, timeout_s=0.3)
        with pytest.raises(McpSessionError, match="outcome_unknown"):
            session.call_tool("count_call", {"value": 1})
    finally:
        session.close()
    log = read_events(events)
    assert [event["event"] for event in log] == ["start", "call"]
    assert_process_gone(log[0]["pid"])
    assert session.metrics["cleanup_confirmed"]


@pytest.mark.parametrize("url", ["https://example.com/mcp", "http://127.0.0.1:8000/mcp#x",
                                  "http://user:password@127.0.0.1/mcp"])
def test_rejects_non_owned_endpoints_before_connecting(url):
    with pytest.raises(ValueError, match="loopback"):
        PersistentMcpSession(url=url)


def test_missing_allowlisted_tool_aborts_startup_and_cleans_up(tmp_path):
    session, events = make_session(tmp_path, allowed_tools={"not_present"})
    try:
        with pytest.raises(McpSessionError, match="transport_failed"):
            session.start()
    finally:
        session.close()
    assert_process_gone(read_events(events)[0]["pid"])


def test_startup_cannot_push_a_tool_input_past_its_remaining_episode_budget(tmp_path):
    session, events = make_session(tmp_path)
    try:
        with pytest.raises(McpSessionError, match="budget_exhausted_before_input"):
            session.call_tool("count_call", {"value": 1}, timeout_s=0.000001)
        assert [event["event"] for event in read_events(events)] == ["start"]
        assert session.metrics["tool_calls"] == 0
        assert tool_data(session.call_tool("count_call", {"value": 2}))["count"] == 1
    finally:
        session.close()


def test_real_loopback_http_session_does_not_use_environment_proxy(tmp_path, monkeypatch):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    process = subprocess.Popen([
        sys.executable, str(Path(__file__).parent / "fixtures/mcp_counter_server.py"),
        str(tmp_path / "events.jsonl"), str(port),
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    session = PersistentMcpSession(url=f"http://127.0.0.1:{port}/mcp", connect_timeout_s=10)
    try:
        deadline = time.monotonic() + 10
        while True:
            assert process.poll() is None
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                assert time.monotonic() < deadline
                time.sleep(0.05)
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        assert tool_data(session.call_tool("count_call", {"value": 3}))["count"] == 1
        assert tool_data(session.call_tool("count_call", {"value": 4}))["count"] == 2
        assert session.metrics["initializations"] == 1
    finally:
        try:
            session.close()
        finally:
            process.terminate()
            process.wait(timeout=10)
