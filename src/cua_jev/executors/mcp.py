from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..errors import CapabilityUnavailable
from ..models import ActionCandidate, ActionReceipt
from .common import execute_with_receipt

McpTool = Callable[[dict[str, Any]], dict[str, Any]]


class InProcessMcpExecutor:
    """MCP-shaped tool boundary for deterministic tests and embedders.

    A remote MCP transport can register wrappers with the same typed call contract.
    """

    def __init__(self) -> None:
        self._tools: dict[str, McpTool] = {}

    def register_tool(self, name: str, tool: McpTool) -> None:
        if not name.startswith("mcp."):
            raise ValueError("MCP capability names must start with 'mcp.'")
        self._tools[name] = tool

    def __call__(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            tool = self._tools.get(candidate.capability)
            if tool is None:
                raise ValueError(f"unregistered MCP tool: {candidate.capability}")
            result = tool(dict(candidate.arguments))
            if not isinstance(result, dict):
                raise TypeError("MCP tool must return a JSON object")
            return result

        return execute_with_receipt(candidate, observation_id, decision_id, operation)


@dataclass(frozen=True)
class StdioMcpServer:
    """An explicitly allowlisted local MCP server process."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


class StdioMcpExecutor:
    """Real MCP stdio transport through the official Python SDK.

    Server commands are registered by trusted application code, never supplied by Jev.
    Candidate arguments may select only a registered server and tool.
    """

    def __init__(self) -> None:
        self._servers: dict[str, StdioMcpServer] = {}
        self._allowed_tools: dict[str, set[str]] = {}

    def register_server(self, name: str, server: StdioMcpServer, *, tools: set[str]) -> None:
        if not name or not tools:
            raise ValueError("MCP server name and allowlisted tools are required")
        self._servers[name] = server
        self._allowed_tools[name] = set(tools)

    async def _call(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        server = self._servers.get(server_name)
        if server is None or tool_name not in self._allowed_tools.get(server_name, set()):
            raise ValueError("MCP server or tool is not allowlisted")
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise CapabilityUnavailable("install cua-jev[mcp] for MCP stdio transport") from None
        parameters = StdioServerParameters(
            command=server.command, args=list(server.args), env=server.env or None
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                if result.isError:
                    raise RuntimeError("MCP tool returned an error")
                structured = getattr(result, "structuredContent", None)
                if structured is None:
                    structured = getattr(result, "structured_content", None)
                content = [
                    item.model_dump(mode="json", by_alias=True) if hasattr(item, "model_dump") else str(item)
                    for item in result.content
                ]
                return {"structured_content": structured, "content": content}

    def __call__(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            if candidate.capability != "mcp.call_tool":
                raise ValueError(f"unsupported MCP capability: {candidate.capability}")
            server_name = candidate.arguments["server"]
            tool_name = candidate.arguments["tool"]
            arguments = candidate.arguments.get("arguments", {})
            # Playwright's synchronous API owns a running event loop on this
            # thread. Keep the MCP SDK's async stdio session on a worker loop.
            with ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run, self._call(server_name, tool_name, arguments)
                ).result()

        return execute_with_receipt(candidate, observation_id, decision_id, operation)
