"""Optional same-origin, read-only HTML MCP server for public research tasks."""

from __future__ import annotations

import argparse
import re
from html.parser import HTMLParser
from urllib.parse import urlsplit

import httpx


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    if (
        parts.scheme != "https" or not parts.hostname
        or parts.username or parts.password
    ):
        raise ValueError("MCP web reader requires an HTTPS URL without credentials")
    return parts.scheme, parts.hostname.casefold(), parts.port or 443


class _PageText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title: list[str] = []
        self.body: list[str] = []
        self._title_depth = 0
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._title_depth += 1
        if tag in {"script", "style", "nav", "footer", "header"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in {"script", "style", "nav", "footer", "header"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self.title.append(data)
        elif not self._ignored_depth:
            self.body.append(data)


def fetch_public_page(
    href: str, allowed_origin: str, *, client: httpx.Client | None = None
) -> dict[str, str]:
    if _origin(href) != _origin(allowed_origin):
        raise ValueError("MCP page is outside the configured HTTPS origin")
    if len(href) > 500:
        raise ValueError("MCP page URL exceeds the configured bound")
    own_client = client is None
    transport = client or httpx.Client(timeout=15, follow_redirects=False)
    try:
        with transport.stream("GET", href, headers={"Accept": "text/html"}) as response:
            if response.status_code != 200:
                raise ValueError(f"MCP page returned HTTP {response.status_code}")
            if "text/html" not in response.headers.get("content-type", "").casefold():
                raise ValueError("MCP page is not HTML")
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > 250_000:
                    raise ValueError("MCP page exceeds the HTML size limit")
                chunks.append(chunk)
            page = _PageText()
            page.feed(b"".join(chunks).decode(response.encoding or "utf-8", errors="replace"))
            title = re.sub(r"\s+", " ", " ".join(page.title)).strip()[:160]
            text = re.sub(r"\s+", " ", " ".join(page.body)).strip()[:2500]
            return {"url": href, "title": title, "text": text}
    finally:
        if own_client:
            transport.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-origin HTML MCP reader")
    parser.add_argument("--origin", required=True)
    args = parser.parse_args()
    _origin(args.origin)
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("cua-jev-web-reader")

    @server.tool()
    def fetch_page(href: str) -> dict[str, str]:
        """Read one public HTML page within the caller-configured HTTPS origin."""
        return fetch_public_page(href, args.origin)

    server.run(transport="stdio")


if __name__ == "__main__":
    main()
