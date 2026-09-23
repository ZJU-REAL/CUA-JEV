"""Independent acceptance check for the public Python.org → VS Code example.

This script is deliberately outside the agent's planner, candidates, and
terminal decision. It checks the created note against a fresh live source page.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def verify(note: Path, trace: Path) -> dict[str, object]:
    content = note.read_text(encoding="utf-8")
    match = re.search(r"\bSource:\s*(https://[^\s)]+)", content)
    if not match:
        raise AssertionError("note lacks a source URL")
    source_url = match.group(1)
    parsed = urlparse(source_url)
    if parsed.scheme != "https" or parsed.hostname != "www.python.org" or not re.fullmatch(
        r"/downloads/release/python-\d+/", parsed.path
    ):
        raise AssertionError("source must be an official Python release page")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page()
            page.goto(source_url, wait_until="domcontentloaded")
            text = page.locator("body").inner_text()
            page.goto("https://www.python.org/", wait_until="domcontentloaded")
            homepage = page.locator("body").inner_text()
        finally:
            browser.close()
    source = re.search(
        r"Python\s+(\d+\.\d+\.\d+)\s+Release date:\s*([A-Za-z.]+\s+\d+,\s+\d{4})",
        text,
    )
    if not source:
        raise AssertionError("could not read release number and date from fresh source")
    version, release_date = source.groups()
    if version not in content or release_date not in content:
        raise AssertionError("note's version or date does not match the fresh source")
    latest = re.search(r"Latest:\s*Python\s+(\d+\.\d+\.\d+)", homepage)
    if not latest or latest.group(1) != version:
        raise AssertionError("release version does not match the current homepage latest")

    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise AssertionError("trace is empty")
    last_run_id = rows[-1]["run_id"]
    rows = [row for row in rows if row["run_id"] == last_run_id]
    decisions = [row["payload"] for row in rows if row["kind"] == "decision"]
    receipts = [row["payload"] for row in rows if row["kind"] == "receipt"]
    episodes = [row["payload"] for row in rows if row["kind"] == "episode"]
    if not episodes or episodes[-1]["status"] != "success":
        raise AssertionError("agent episode was not successful")
    if not decisions or not all(str(item["model"]).startswith("jev-") for item in decisions):
        raise AssertionError("every decision must use Jev without a rule fallback")
    capabilities = {item["capability"] for item in receipts if item["success"]}
    if not {"workspace.browser_click", "workspace.write_note", "workspace.open_note"} <= capabilities:
        raise AssertionError("browser, note writing, and VS Code opening must all execute")
    return {
        "verified": True,
        "version": version,
        "release_date": release_date,
        "source_url": source_url,
        "actions": len(decisions),
        "channels": episodes[-1]["channel_counts"],
        "wall_time_ms": round(episodes[-1]["duration_ms"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", required=True)
    parser.add_argument("--trace", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(Path(args.note), Path(args.trace)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
