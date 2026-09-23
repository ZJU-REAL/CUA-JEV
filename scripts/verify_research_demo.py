"""Independently re-read every cited public source in an open-workspace research run.

This verifier is not called by the planner or Jev policy. It checks provenance and
execution, not the pedagogical quality of the model-written synthesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright


def verify(note: Path, trace: Path, topics: tuple[str, ...]) -> dict[str, object]:
    content = note.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise AssertionError("trace is empty")
    run_id = rows[-1]["run_id"]
    rows = [row for row in rows if row["run_id"] == run_id]
    episodes = [row["payload"] for row in rows if row["kind"] == "episode"]
    decisions = [row["payload"] for row in rows if row["kind"] == "decision"]
    receipts = [row["payload"] for row in rows if row["kind"] == "receipt"]
    observations = [row["payload"]["state"] for row in rows if row["kind"] == "observation"]
    if not episodes or episodes[-1]["status"] != "success":
        raise AssertionError("latest episode did not succeed")
    if len(decisions) < len(topics) * 2 + 2:
        raise AssertionError("episode is shorter than the requested multi-page task")
    if not all(str(row["model"]).startswith("jev-") for row in decisions):
        raise AssertionError("not every action was selected by Jev")
    if not all(row["success"] for row in receipts):
        raise AssertionError("one or more selected actions failed")
    actual = [row for row in receipts if row["capability"] == "workspace.record_evidence"]
    if len(actual) != len(topics):
        raise AssertionError("did not record one quote per requested topic")
    if not {"workspace.browser_click", "workspace.write_note", "workspace.open_note"} <= {
        row["capability"] for row in receipts
    }:
        raise AssertionError("browser navigation, note write, or editor open is missing")
    evidence = max(
        (row.get("collected_evidence", []) for row in observations), key=len, default=[]
    )
    if len(evidence) != len(topics) or {row["topic"] for row in evidence} != set(topics):
        raise AssertionError("trace evidence does not cover the requested topics")
    if len({row["url"] for row in evidence}) != len(topics):
        raise AssertionError("research evidence must come from distinct source pages")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page()
            for row in evidence:
                parsed = urlparse(row["url"])
                if parsed.scheme != "https" or parsed.hostname != "docs.python.org":
                    raise AssertionError("this demo only accepts official Python documentation")
                page.goto(row["url"], wait_until="domcontentloaded", timeout=30000)
                current = page.evaluate(
                    """() => { const root = document.querySelector('main, article, div.body') ||
                      document.body; const heading = root.querySelector('h1')?.innerText || '';
                      return {heading, text: root.innerText}; }"""
                )
                if row["topic"].casefold() not in current["heading"].casefold():
                    raise AssertionError(f"source heading changed for {row['topic']}")
                if row["quote"].casefold() not in current["text"].casefold():
                    raise AssertionError(f"source quote changed for {row['topic']}")
                if row["quote"].casefold() not in content.casefold() or row["url"] not in content:
                    raise AssertionError(f"note lacks quote or source URL for {row['topic']}")
        finally:
            browser.close()
    return {
        "verified": True,
        "topics": len(topics),
        "jev_decisions": len(decisions),
        "channels": episodes[-1]["channel_counts"],
        "wall_time_ms": round(episodes[-1]["duration_ms"]),
        "scope": "source grounding and execution; synthesis quality not evaluated",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--note", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--topic", action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.note, args.trace, tuple(args.topic)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
