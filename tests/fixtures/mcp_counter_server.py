"""Owned real MCP process fixture; all files are confined to pytest tmp_path."""

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

events = Path(sys.argv[1])
server = FastMCP(
    "cua-jev-session-test", host="127.0.0.1", port=int(sys.argv[2]) if len(sys.argv) > 2 else 8000
)
count = 0


def record(event):
    with events.open("a") as stream:
        stream.write(json.dumps({"event": event, "pid": os.getpid()}) + "\n")


@server.tool()
async def count_call(value: int, delay: float = 0, fail: bool = False) -> dict:
    global count
    count += 1
    record("call")
    await asyncio.sleep(delay)
    if fail:
        raise ValueError("fixture tool error")
    return {"value": value, "count": count, "pid": os.getpid()}


record("start")
server.run(transport="streamable-http" if len(sys.argv) > 2 else "stdio")
