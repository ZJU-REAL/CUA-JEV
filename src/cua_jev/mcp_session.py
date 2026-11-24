"""One persistent MCP session per owned episode, with no action retries.

All async context managers enter and exit in the same worker task. This avoids
cross-task cancellation-scope errors while the synchronous agent uses Playwright.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import threading
import time
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import AsyncExitStack
from datetime import timedelta
from urllib.parse import urlsplit

import httpx

from .executors.mcp import StdioMcpServer


class McpSessionError(RuntimeError):
    """Static errors never include endpoint credentials or raw tool contents."""


def _check_schema_refs(value):
    if isinstance(value, dict):
        for key in ("$ref", "$dynamicRef", "$recursiveRef"):
            if key in value and (not isinstance(value[key], str) or not value[key].startswith("#")):
                raise ValueError("external schema references are unsupported")
        for item in value.values():
            _check_schema_refs(item)
    elif isinstance(value, list):
        for item in value:
            _check_schema_refs(item)


class PersistentMcpSession:
    def __init__(
        self,
        *,
        server: StdioMcpServer | None = None,
        url: str | None = None,
        allowed_tools: set[str] | None = None,
        connect_timeout_s: float = 30,
        call_timeout_s: float = 30,
    ):
        if (server is None) == (url is None):
            raise ValueError("provide exactly one owned stdio server or loopback URL")
        if url is not None:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username
                or parsed.password
                or parsed.fragment
            ):
                raise ValueError("MCP URL must be an explicitly owned loopback endpoint")
        for budget in (connect_timeout_s, call_timeout_s):
            if type(budget) not in (int, float) or not math.isfinite(budget) or not 0 < budget <= 120:
                raise ValueError("MCP timeout must be finite and in (0, 120]")
        self.server, self.url = server, url
        self.allowed_tools = None if allowed_tools is None else frozenset(allowed_tools)
        self.connect_timeout_s, self.call_timeout_s = connect_timeout_s, call_timeout_s
        self._lock = threading.RLock()
        self._ready, self._done = threading.Event(), threading.Event()
        self._thread = self._loop = self._queue = None
        self._catalog = {}
        self._validators = {}
        self._failed = None
        self._closed = False
        self.metrics = {
            "initializations": 0,
            "tool_calls": 0,
            "startup_wall_ms": 0.0,
            "tool_wall_ms": 0.0,
            "cleanup_confirmed": False,
        }

    @staticmethod
    def _http_client(**kwargs):
        return httpx.AsyncClient(**kwargs, trust_env=False, follow_redirects=False)

    def start(self):
        with self._lock:
            if self._closed:
                raise McpSessionError("mcp_session_closed")
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._thread_main, daemon=True, name="cua-jev-owned-mcp-session"
                )
                self._thread.start()
            if not self._ready.wait(self.connect_timeout_s + 5):
                self._failed = "mcp_startup_timeout"
                raise McpSessionError(self._failed)
            if self._failed:
                raise McpSessionError(self._failed)
            if self._done.is_set():
                raise McpSessionError("mcp_session_terminated")

    def _thread_main(self):
        try:
            asyncio.run(self._serve())
        except BaseException:
            self._failed = self._failed or "mcp_session_transport_failed"
        finally:
            self._ready.set()
            self._done.set()

    async def _serve(self):
        from jsonschema.validators import validator_for
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
        began = time.perf_counter()
        async with AsyncExitStack() as stack:
            async with asyncio.timeout(self.connect_timeout_s):
                if self.server is not None:
                    parameters = StdioServerParameters(
                        command=self.server.command, args=list(self.server.args), env=self.server.env
                    )
                    read, write = await stack.enter_async_context(stdio_client(parameters))
                else:
                    read, write, _session_id = await stack.enter_async_context(
                        streamablehttp_client(
                            self.url,
                            timeout=self.call_timeout_s,
                            sse_read_timeout=self.call_timeout_s,
                            httpx_client_factory=self._http_client,
                        )
                    )
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=self.call_timeout_s),
                    )
                )
                await session.initialize()
                self.metrics["initializations"] += 1
                cursor, seen_cursors, all_tools = None, set(), []
                while True:
                    result = await session.list_tools(cursor=cursor)
                    all_tools.extend(result.tools)
                    if len(all_tools) > 4096:
                        raise ValueError("MCP catalog exceeds bounded size")
                    cursor = result.nextCursor
                    if cursor is None:
                        break
                    if cursor in seen_cursors:
                        raise ValueError("MCP catalog repeats pagination cursor")
                    seen_cursors.add(cursor)
                for tool in all_tools:
                    if tool.name in self._catalog:
                        raise ValueError("duplicate MCP tool name")
                    if self.allowed_tools is not None and tool.name not in self.allowed_tools:
                        continue
                    data = tool.model_dump(mode="json", by_alias=True)
                    schema = data["inputSchema"]
                    _check_schema_refs(schema)
                    validator = validator_for(schema)
                    validator.check_schema(schema)
                    self._catalog[tool.name] = data
                    self._validators[tool.name] = validator(schema)
                if not self._catalog or len(json.dumps(self._catalog, allow_nan=False)) > 2_000_000:
                    raise ValueError("empty or oversized MCP catalog")
                if self.allowed_tools is not None and set(self._catalog) != self.allowed_tools:
                    raise ValueError("allowlisted MCP tool missing from server catalog")
            self.metrics["startup_wall_ms"] = (time.perf_counter() - began) * 1000
            self._ready.set()
            while True:
                request = await self._queue.get()
                if request is None:
                    break
                name, arguments, deadline, future = request
                budget = deadline - time.monotonic()
                if budget <= 0:
                    future.set_exception(McpSessionError("mcp_call_budget_exhausted_before_input"))
                    continue
                began = time.perf_counter()
                self.metrics["tool_calls"] += 1
                try:
                    async with asyncio.timeout(budget):
                        result = await session.call_tool(name, arguments=arguments)
                    # Tool-level errors are ordinary observations, not transport
                    # retries and not evidence of successful semantic effects.
                    payload = result.model_dump(mode="json", by_alias=True)
                    if len(json.dumps(payload, allow_nan=False)) > 4_000_000:
                        raise ValueError("MCP result exceeds bounded size")
                    future.set_result(payload)
                except BaseException:
                    # The server may have performed input before its response
                    # was lost. Retire the session; never replay this request.
                    self._failed = "mcp_call_failed_outcome_unknown"
                    if not future.done():
                        future.set_exception(McpSessionError(self._failed))
                    break
                finally:
                    self.metrics["tool_wall_ms"] += (time.perf_counter() - began) * 1000

    def list_tools(self):
        with self._lock:
            self.start()
            return copy.deepcopy(list(self._catalog.values()))

    def call_tool(self, name, arguments, *, timeout_s=None):
        with self._lock:
            budget = self.call_timeout_s if timeout_s is None else timeout_s
            if (type(budget) not in (int, float) or not math.isfinite(budget)
                    or not 0 < budget <= self.call_timeout_s):
                raise ValueError("MCP call budget must be positive and within the configured timeout")
            deadline = time.monotonic() + budget
            if self.allowed_tools is not None and name not in self.allowed_tools:
                raise ValueError("MCP tool is not allowlisted")
            if type(arguments) is not dict or len(json.dumps(arguments, allow_nan=False)) > 128_000:
                raise ValueError("MCP arguments must be a bounded JSON object")
            self.start()
            if name not in self._validators:
                raise ValueError("MCP tool was not in the observed catalog")
            try:
                self._validators[name].validate(arguments)
            except Exception:
                raise ValueError("MCP arguments failed the observed tool schema") from None
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpSessionError("mcp_call_budget_exhausted_before_input")
            future = Future()
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait, (name, copy.deepcopy(arguments), deadline, future),
            )
            try:
                return future.result(timeout=remaining + 5)
            except FutureTimeout:
                self._failed = "mcp_call_failed_outcome_unknown"
                raise McpSessionError(self._failed) from None

    def close(self):
        with self._lock:
            if self._closed and self.metrics["cleanup_confirmed"]:
                return
            self._closed = True
            if self._thread is None:
                self.metrics["cleanup_confirmed"] = True
                return
            if self._loop is not None and not self._done.is_set():
                try:
                    self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
                except RuntimeError:
                    pass
            self._thread.join(timeout=10)
            self.metrics["cleanup_confirmed"] = not self._thread.is_alive()
            if not self.metrics["cleanup_confirmed"]:
                raise McpSessionError("mcp_session_cleanup_unconfirmed")
