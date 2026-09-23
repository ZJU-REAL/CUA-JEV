from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("cua-jev-test")


@mcp.tool()
def echo(text: str) -> dict[str, str]:
    """Return controlled test input."""
    return {"echo": text}


@mcp.tool()
def fetch_page(href: str) -> dict[str, str]:
    """Return a test URL without network access."""
    return {"url": href}


@mcp.tool()
def read_text(path: str) -> dict[str, str]:
    """Read a short text file selected by trusted caller configuration."""
    return {"text": Path(path).read_text(encoding="utf-8")[:1000]}


if __name__ == "__main__":
    mcp.run(transport="stdio")
