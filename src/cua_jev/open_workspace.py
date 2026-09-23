"""Experimental browser-to-editor workflow with model-proposed, grounded actions.

This is a reusable *task family*, not unrestricted cross-application CUA. The
model decides which public browser link to follow and what note to write; Jev
selects a real execution route. A separate acceptance test should still check
whether the note answers the user's particular question.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .episode import Evaluation
from .errors import CapabilityUnavailable
from .executors.common import execute_with_receipt
from .executors.screen import ScreenController
from .models import ActionCandidate, ActionReceipt, Channel, Observation, Risk, Verification
from .open_browser import BrowserSnapshot, _origin
from .runtime import StepResult
from .verify import VerifierRegistry


def _same_origin(url: str, base: str) -> bool:
    try:
        return _origin(url) == _origin(base)
    except ValueError:
        return False


def _ground_quote(proposed: str, source: str) -> str | None:
    """Return literal source text for a whitespace/typographic-normalized quote.

    Models often replace a curly apostrophe with ASCII. We still require a
    contiguous source substring, then store the *original* characters rather
    than the model's rewritten text.
    """

    def normalized(value: str, *, offsets: bool = False):
        chars: list[str] = []
        spans: list[tuple[int, int]] = []
        fold = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
        for index, char in enumerate(value):
            translated = unicodedata.normalize("NFKC", fold.get(char, char)).casefold()
            for part in translated:
                if part.isspace():
                    if chars and chars[-1] == " ":
                        spans[-1] = (spans[-1][0], index + 1)
                        continue
                    part = " "
                chars.append(part)
                spans.append((index, index + 1))
        joined = "".join(chars).strip()
        if not offsets:
            return joined
        left = len(chars) - len("".join(chars).lstrip())
        right = left + len(joined)
        return joined, spans[left:right]

    target = normalized(proposed)
    haystack, spans = normalized(source, offsets=True)
    if not target:
        return None
    position = haystack.find(target)
    if position < 0:
        return None
    return source[spans[position][0]:spans[position + len(target) - 1][1]]


@dataclass(frozen=True)
class WorkspaceSnapshot:
    browser: BrowserSnapshot
    note_path: str
    note_exists: bool
    note_text: str
    editor_open: bool
    editor_title: str
    allowed_operations: tuple[str, ...] = ("browser_click", "open_note", "write_note")
    source_text: str = ""
    source_heading: str = ""
    research_topics: tuple[str, ...] = ()
    collected_evidence: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        browser = self.browser.to_dict()
        browser["elements"] = [
            {"ref": item.ref, "label": item.label, "href": item.href}
            for item in self.browser.elements
            if item.tag == "a" and item.href and _same_origin(item.href, self.browser.url)
        ][:40]
        return {
            "browser": browser, "note_name": Path(self.note_path).name,
            "note_exists": self.note_exists, "note_text": self.note_text[:1200],
            "editor_open": self.editor_open, "editor_title": self.editor_title,
            "allowed_operations": list(self.allowed_operations),
            "source_heading": self.source_heading,
            "source_text": self.source_text[:4000],
            "research_topics": list(self.research_topics),
            "collected_evidence": list(self.collected_evidence),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class WorkspaceOption:
    operation: str
    ref: str = ""
    text: str = ""
    evidence: str = ""
    topic: str = ""


@dataclass(frozen=True)
class WorkspacePlan:
    subgoal: str
    options: tuple[WorkspaceOption, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], snapshot: WorkspaceSnapshot) -> WorkspacePlan:
        if not isinstance(data, dict) or not isinstance(data.get("options"), list):
            raise ValueError("workspace planner must return an options list")
        if not 1 <= len(data["options"]) <= 8:
            raise ValueError("workspace planner must return 1-8 options")
        subgoal = data.get("subgoal")
        if not isinstance(subgoal, str) or not 1 <= len(subgoal.strip()) <= 240:
            raise ValueError("workspace planner subgoal is invalid")
        known = {item.ref: item for item in snapshot.browser.elements}
        options: list[WorkspaceOption] = []
        for item in data["options"]:
            if not isinstance(item, dict):
                raise ValueError("workspace option must be an object")
            operation = item.get("operation")
            ref = item.get("ref", "")
            text = item.get("text", "")
            evidence = item.get("evidence", "")
            topic = item.get("topic", "")
            if operation not in {
                "browser_click", "browser_back", "record_evidence", "open_note", "write_note"
            }:
                raise ValueError("unsupported workspace operation")
            if operation not in snapshot.allowed_operations:
                raise ValueError(
                    f"operation {operation} is unavailable now; choose from "
                    + ", ".join(snapshot.allowed_operations)
                )
            if not all(isinstance(value, str) for value in (ref, text, evidence, topic)):
                raise ValueError("workspace option fields must be strings")
            if operation == "browser_click":
                element = known.get(ref)
                if element is None or element.tag != "a" or element.disabled or not element.href:
                    raise ValueError("browser click must reference a live link")
                if _origin(element.href) != _origin(snapshot.browser.url):
                    raise ValueError("browser click must stay on the starting origin")
            elif operation == "browser_back":
                pass
            elif operation == "record_evidence":
                if topic not in snapshot.research_topics:
                    raise ValueError("evidence topic is not requested")
                if any(row["topic"] == topic for row in snapshot.collected_evidence):
                    raise ValueError("evidence topic was already collected")
                if any(row["url"] == snapshot.browser.url for row in snapshot.collected_evidence):
                    raise ValueError("research evidence must come from distinct pages")
                if topic.casefold() not in snapshot.source_heading.casefold():
                    raise ValueError("source heading does not match the requested topic")
                if not 8 <= len(evidence) <= 240:
                    raise ValueError("evidence must quote current main-page text")
                grounded = _ground_quote(evidence, snapshot.source_text or snapshot.browser.text)
                if grounded is None:
                    raise ValueError("evidence must quote current main-page text")
                evidence = grounded
            elif operation == "open_note":
                if snapshot.editor_open:
                    raise ValueError("editor is already open")
            else:
                if snapshot.note_exists:
                    raise ValueError("note already exists; open it in VS Code")
                if not 1 <= len(text) <= 4000:
                    raise ValueError("note text or source evidence is invalid")
                if snapshot.research_topics:
                    if len(snapshot.collected_evidence) != len(snapshot.research_topics):
                        raise ValueError("all research topics need grounded evidence before writing")
                elif not 4 <= len(evidence) <= 160 or evidence.casefold() not in (
                    snapshot.browser.text.casefold()
                ):
                    raise ValueError("evidence is not in the visible browser observation")
            options.append(WorkspaceOption(operation, ref, text, evidence, topic))
        if len({item.operation for item in options}) != 1:
            raise ValueError("one plan must describe one next intent")
        if len(set(options)) != len(options):
            raise ValueError("workspace planner returned duplicate options")
        return cls(subgoal.strip(), tuple(options))


class WorkspaceGoalPlanner(Protocol):
    def plan(
        self, goal: str, snapshot: WorkspaceSnapshot, recent_actions: Sequence[dict[str, Any]]
    ) -> WorkspacePlan: ...


class OpenWorkspaceTask:
    """One public browser origin and one scoped VS Code note, no scripted site steps."""

    name = "open-workspace"

    def __init__(
        self, *, goal: str, start_url: str, note_path: Path,
        planner: WorkspaceGoalPlanner, screen: ScreenController | None = None,
        page: Any | None = None, editor_probe: Any | None = None,
        required_source_texts: Sequence[str] = (),
        research_topics: Sequence[str] = (),
        maximize_editor: bool = False,
    ) -> None:
        if not goal.strip():
            raise ValueError("goal is required")
        self.goal = goal.strip()
        self.start_url = start_url
        self.allowed_origin = _origin(start_url)
        self.note_path = note_path.resolve()
        if self.note_path.suffix.lower() not in {".md", ".txt"}:
            raise ValueError("note must be a .md or .txt file")
        self.planner = planner
        self.required_source_texts = tuple(value.strip() for value in required_source_texts if value.strip())
        self.research_topics = tuple(value.strip() for value in research_topics if value.strip())
        if len(self.research_topics) > 10 or any(len(value) > 120 for value in self.research_topics):
            raise ValueError("research mode supports at most ten short topic headings")
        if len(self.research_topics) != len(set(self.research_topics)):
            raise ValueError("research topics must be unique")
        if self.research_topics and self.required_source_texts:
            raise ValueError("use research topics or source text clues, not both")
        self.maximize_editor = maximize_editor
        self.screen = screen or ScreenController()
        self.page = page
        self.editor_probe = editor_probe or self._editor_title
        self._playwright = None
        self._browser = None
        self._snapshot: WorkspaceSnapshot | None = None
        self._plan: WorkspacePlan | None = None
        self._plan_fingerprint = ""
        self._used: set[int] = set()
        self._before: dict[str, WorkspaceSnapshot] = {}
        self._evidence = ""
        self._source_url = ""
        self._research_evidence: list[dict[str, str]] = []
        self.planner_calls = 0

    def reset(self) -> None:
        if self.note_path.exists():
            raise ValueError("refusing to overwrite an existing note")
        self.note_path.parent.mkdir(parents=True, exist_ok=True)
        self._research_evidence.clear()
        if self.page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                raise CapabilityUnavailable("install cua-jev[browser] for workspace tasks") from None
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                channel="msedge", headless=False,
                args=[
                    "--start-maximized", "--disable-translate",
                    "--disable-features=Translate,TranslateUI", "--lang=en-US",
                ],
            )
            self.page = self._browser.new_context(no_viewport=True, locale="en-US").new_page()
        self.page.route("**/*", self._route_request)
        self.page.goto(self.start_url, wait_until="domcontentloaded")

    def close(self) -> None:
        if self._browser is not None:
            self._browser.close()
        if self._playwright is not None:
            self._playwright.stop()
        self._browser = self._playwright = None
        self.page = None

    def _route_request(self, route: Any) -> None:
        request = route.request
        if request.is_navigation_request() and request.frame == self.page.main_frame:
            try:
                if _origin(request.url) != self.allowed_origin:
                    route.abort()
                    return
            except ValueError:
                route.abort()
                return
        route.fallback()

    def _editor_title(self) -> str:
        try:
            from pywinauto import Desktop
        except ImportError:
            raise CapabilityUnavailable("install cua-jev[windows] for VS Code observation") from None
        matches = Desktop(backend="uia").windows(
            title_re=rf".*{re.escape(self.note_path.name)}.*Visual Studio Code.*", visible_only=True
        )
        return str(matches[-1].window_text()) if matches else ""

    def _capture(self) -> WorkspaceSnapshot:
        if self.page is None:
            raise RuntimeError("workspace task has not been reset")
        for attempt in range(4):
            try:
                browser = BrowserSnapshot.capture(self.page)
                page_content = self.page.evaluate(
                    """() => { const root = document.querySelector('main, article, div.body') ||
                      document.body; if (!root) throw new Error('document body not ready');
                      const heading = root.querySelector('h1') || document.querySelector('h1');
                      return {url: location.href, heading: (heading?.innerText || '').trim(),
                        text: (root.innerText || '').trim().slice(0, 6000)}; }"""
                )
                if page_content["url"] != browser.url:
                    raise RuntimeError("browser navigation changed during observation")
                break
            except Exception as exc:
                transient = any(marker in str(exc) for marker in (
                    "Execution context was destroyed", "document body not ready",
                    "browser navigation changed during observation",
                ))
                if not transient or attempt == 3:
                    raise
                self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                time.sleep(0.2)
        if _origin(browser.url) != self.allowed_origin:
            raise RuntimeError("browser left the allowed origin")
        source_heading = str(page_content["heading"])[:240]
        source_text = str(page_content["text"])[:6000]
        exists = self.note_path.is_file()
        note_text = self.note_path.read_text(encoding="utf-8") if exists else ""
        editor_title = self.editor_probe()
        if exists:
            allowed = ("open_note",) if not editor_title else ()
        elif self.research_topics:
            pending = set(self.research_topics) - {row["topic"] for row in self._research_evidence}
            if not pending:
                allowed = ("write_note",)
            elif (
                browser.url != self.start_url
                and any(topic.casefold() in source_heading.casefold() for topic in pending)
                and not any(row["url"] == browser.url for row in self._research_evidence)
            ):
                allowed = ("record_evidence",)
            else:
                allowed = ("browser_click", "browser_back") if browser.url != self.start_url else (
                    "browser_click",
                )
        elif self.required_source_texts and any(
            value.casefold() not in browser.text.casefold() for value in self.required_source_texts
        ):
            allowed = ("browser_click",)
        elif self.required_source_texts:
            allowed = ("write_note",)
        else:
            allowed = ("browser_click", "write_note")
        return WorkspaceSnapshot(
            browser, str(self.note_path), exists, note_text, bool(editor_title), editor_title,
            allowed, source_text, source_heading, self.research_topics,
            tuple(row.copy() for row in self._research_evidence),
        )

    def observe(self, history: Sequence[StepResult]) -> Observation:
        self._snapshot = self._capture()
        return Observation(
            self.goal, self._plan.subgoal if self._plan else "Find the next grounded cross-app action",
            self._snapshot.to_dict(), self.name,
        )

    def _satisfied(self, snapshot: WorkspaceSnapshot) -> bool:
        if self.research_topics:
            return (
                snapshot.editor_open and snapshot.note_exists
                and len(self._research_evidence) == len(self.research_topics)
                and all(row["quote"].casefold() in snapshot.note_text.casefold()
                        and row["url"] in snapshot.note_text
                        for row in self._research_evidence)
            )
        return (
            snapshot.editor_open and snapshot.note_exists and bool(self._evidence)
            and self._evidence.casefold() in snapshot.note_text.casefold()
            and self._source_url in snapshot.note_text
        )

    def _call_planner(self, history: Sequence[StepResult]) -> None:
        assert self._snapshot is not None
        recent = [
            {"candidate_id": item.decision.candidate_id, "verified": item.verification.passed}
            for item in history[-6:]
        ]
        for attempt in range(3):
            try:
                self.planner_calls += 1
                planner_goal = self.goal
                if self.required_source_texts:
                    planner_goal += (
                        " Before writing, the currently visible source page MUST contain: "
                        + ", ".join(self.required_source_texts)
                    )
                if self.research_topics:
                    planner_goal += (
                        " Collect one exact quote from a different official page for each "
                        "requested topic heading, then synthesize one sourced note: "
                        + ", ".join(self.research_topics)
                    )
                self._plan = self.planner.plan(planner_goal, self._snapshot, recent)
                if any(item.operation == "write_note" for item in self._plan.options):
                    missing = [
                        value for value in self.required_source_texts
                        if value.casefold() not in self._snapshot.browser.text.casefold()
                    ]
                    if missing:
                        raise ValueError(
                            "current browser page lacks required source text; navigate first: "
                            + ", ".join(missing)
                        )
                break
            except ValueError as exc:
                if attempt == 2:
                    raise
                recent.append({"planner_validation_error": str(exc)})
        self._plan_fingerprint = self._snapshot.fingerprint()
        self._used.clear()

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]:
        assert self._snapshot is not None
        if self._satisfied(self._snapshot):
            return (ActionCandidate("done", Channel.CONTROL, "control.done", "Finish verified note"),)
        if self._plan is None or self._plan_fingerprint != self._snapshot.fingerprint():
            self._call_planner(history)
        choices = self._build_candidates()
        if not choices:
            self._call_planner(history)
            choices = self._build_candidates()
        return choices

    def _build_candidates(self) -> tuple[ActionCandidate, ...]:
        assert self._snapshot is not None and self._plan is not None
        result: list[ActionCandidate] = []
        known = {item.ref: item for item in self._snapshot.browser.elements}
        for index, option in enumerate(self._plan.options):
            if index in self._used:
                continue
            if option.operation == "browser_click":
                element = known[option.ref]
                args = {"ref": element.ref, "index": element.index, "label": element.label,
                        "href": element.href}
                routes = ((Channel.SCRIPT, "dom"), (Channel.GUI, "gui"))
                label = f"Open {element.label or element.href}"
                risk = Risk.READ_ONLY
            elif option.operation == "browser_back":
                args = {}
                routes = ((Channel.SCRIPT, "history"), (Channel.GUI, "gui"))
                label = "Return to the previous source page"
                risk = Risk.READ_ONLY
            elif option.operation == "record_evidence":
                args = {"topic": option.topic, "quote": option.evidence,
                        "source_url": self._snapshot.browser.url}
                routes = ((Channel.API, "evidence"),)
                label = f"Record a grounded quote for {option.topic}"
                risk = Risk.READ_ONLY
            elif option.operation == "open_note":
                args = {"file_path": str(self.note_path)}
                routes = ((Channel.CLI, "vscode"),)
                label = "Open the scoped note in VS Code"
                risk = Risk.READ_ONLY
            else:
                note_text = option.text
                if self.research_topics:
                    source_lines = [
                        f"- {row['topic']}: \"{row['quote']}\" — {row['url']}"
                        for row in self._research_evidence
                    ]
                    note_text = note_text.rstrip() + "\n\n## Verified source excerpts\n" + (
                        "\n".join(source_lines) + "\n"
                    )
                else:
                    if option.evidence.casefold() not in note_text.casefold():
                        note_text = note_text.rstrip() + f"\n\nObserved: {option.evidence}\n"
                    if self._snapshot.browser.url not in note_text:
                        note_text = note_text.rstrip() + f"\n\nSource: {self._snapshot.browser.url}\n"
                args = {"file_path": str(self.note_path), "text": note_text,
                        "evidence": option.evidence, "source_url": self._snapshot.browser.url}
                routes = ((Channel.API, "filesystem"),)
                if self._snapshot.editor_open:
                    routes += ((Channel.GUI, "vscode"),)
                label = f"Write source-grounded note about {option.evidence[:60]}"
                risk = Risk.LOCAL_WRITE
            for channel, suffix in routes:
                result.append(ActionCandidate(
                    f"option_{index}_{suffix}", channel, f"workspace.{option.operation}",
                    f"{label} using {suffix.upper()}", args, risk,
                    verifier="workspace.effect", intent=f"option_{index}",
                ))
        return tuple(result)

    def executor_bindings(self) -> dict[Channel, Any]:
        return {channel: self for channel in (Channel.SCRIPT, Channel.GUI, Channel.CLI, Channel.API)}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("workspace.effect", self._verify_effect)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            before = self._capture()
            self._before[decision_id] = before
            action = candidate.capability.removeprefix("workspace.")
            if action == "browser_click":
                element = next(
                    (item for item in before.browser.elements
                     if item.href == candidate.arguments["href"]
                     and item.label == candidate.arguments["label"]), None
                )
                if element is None:
                    raise RuntimeError("browser link changed since observation")
                handle = self.page.evaluate_handle(
                    """target => [...document.querySelectorAll('a')].find(el => {
                      const rect = el.getBoundingClientRect();
                      const label = (el.getAttribute('aria-label') || el.innerText ||
                        el.getAttribute('title') || '').trim().slice(0, 120);
                      return el.href === target.href && label === target.label &&
                        rect.width > 0 && rect.height > 0;
                    }) || null""",
                    {"href": element.href, "label": element.label},
                ).as_element()
                if handle is None:
                    raise RuntimeError("grounded browser link no longer visible")
                if candidate.channel == Channel.SCRIPT:
                    handle.click(timeout=15000, no_wait_after=True)
                else:
                    self._physical_browser_click(handle)
                return {"url": self.page.url}
            if action == "browser_back":
                if candidate.channel == Channel.SCRIPT:
                    if self.page.go_back(wait_until="domcontentloaded", timeout=15000) is None:
                        raise RuntimeError("browser has no previous page")
                else:
                    self.screen.focus(rf".*{re.escape(self.page.title())}.*", maximize=False)
                    self.screen.hotkey("alt", "left")
                    self.page.wait_for_load_state("domcontentloaded", timeout=15000)
                return {"url": self.page.url}
            if action == "record_evidence":
                args = candidate.arguments
                if args["topic"] not in self.research_topics:
                    raise RuntimeError("unrequested research topic")
                if any(row["topic"] == args["topic"] or row["url"] == before.browser.url
                       for row in self._research_evidence):
                    raise RuntimeError("topic or source page already recorded")
                if args["topic"].casefold() not in before.source_heading.casefold():
                    raise RuntimeError("requested heading is no longer visible")
                if args["source_url"] != before.browser.url or args["quote"].casefold() not in (
                    before.source_text.casefold()
                ):
                    raise RuntimeError("research quote or source changed")
                self._research_evidence.append({
                    "topic": args["topic"], "quote": args["quote"], "url": before.browser.url,
                })
                return {"topic": args["topic"], "source_url": before.browser.url}
            if action == "open_note":
                if before.editor_open:
                    raise RuntimeError("editor already open")
                executable = shutil.which("code")
                if not executable:
                    raise CapabilityUnavailable("VS Code CLI 'code' was not found")
                subprocess.Popen([executable, "--new-window", str(self.note_path)], shell=False)
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline and not self.editor_probe():
                    time.sleep(0.3)
                if not self.editor_probe():
                    raise RuntimeError("VS Code did not display the note")
                if self.maximize_editor:
                    self.screen.focus(
                        rf".*{re.escape(self.note_path.name)}.*Visual Studio Code.*", maximize=True
                    )
                return {"editor": self.editor_probe()}
            if action == "write_note":
                args = candidate.arguments
                if args["file_path"] != str(self.note_path) or before.note_exists:
                    raise RuntimeError("note target changed or already exists")
                if self.research_topics:
                    if len(self._research_evidence) != len(self.research_topics) or not all(
                        row["quote"].casefold() in args["text"].casefold()
                        and row["url"] in args["text"] for row in self._research_evidence
                    ):
                        raise RuntimeError("note does not cite every collected source")
                else:
                    if args["evidence"].casefold() not in before.browser.text.casefold():
                        raise RuntimeError("source evidence no longer visible")
                    if args["source_url"] != before.browser.url:
                        raise RuntimeError("browser source changed since observation")
                if candidate.channel == Channel.API:
                    self.note_path.write_text(args["text"], encoding="utf-8")
                else:
                    self.screen.focus(rf".*{re.escape(self.note_path.name)}.*Visual Studio Code.*")
                    self.screen.hotkey("ctrl", "a")
                    self.screen.paste_text(args["text"])
                    self.screen.hotkey("ctrl", "s")
                self._evidence = args["evidence"]
                self._source_url = args["source_url"]
                return {"note_path": str(self.note_path), "bytes": self.note_path.stat().st_size}
            raise ValueError("unsupported workspace action")

        return execute_with_receipt(candidate, observation_id, decision_id, operation)

    def _physical_browser_click(self, locator: Any) -> None:
        title = self.page.title()
        window = self.screen.focus(rf".*{re.escape(title)}.*", maximize=False)
        locator.scroll_into_view_if_needed()
        point = locator.evaluate(
            """element => { const r = element.getBoundingClientRect(); return {
              x: (window.outerWidth-window.innerWidth)/2+r.left+r.width/2,
              y: window.outerHeight-window.innerHeight+r.top+r.height/2,
              outerWidth: window.outerWidth, outerHeight: window.outerHeight }; }"""
        )
        rect = window.rectangle()
        x = rect.left + point["x"] * rect.width() / point["outerWidth"]
        y = rect.top + point["y"] * rect.height() / point["outerHeight"]
        self.screen.click_point(x, y)

    def _verify_effect(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        if not receipt.success:
            return Verification(False, "workspace.effect", {"error": receipt.error})
        before = self._before.get(receipt.decision_id)
        if before is None:
            return Verification(False, "workspace.effect", {"error": "missing pre-action snapshot"})
        after = self._capture()
        action = candidate.capability.removeprefix("workspace.")
        if action in {"browser_click", "browser_back"}:
            passed = after.browser.fingerprint() != before.browser.fingerprint()
        elif action == "record_evidence":
            args = candidate.arguments
            passed = any(
                row["topic"] == args["topic"] and row["quote"] == args["quote"]
                and row["url"] == args["source_url"] for row in after.collected_evidence
            ) and len(after.collected_evidence) == len(before.collected_evidence) + 1
        elif action == "open_note":
            passed = after.editor_open
        else:
            passed = (
                after.note_exists and after.note_text == candidate.arguments["text"]
                and candidate.arguments["evidence"].casefold() in after.note_text.casefold()
            )
        return Verification(passed, "workspace.effect", {"state_changed": passed, "action": action})

    def evaluate(
        self, observation: Observation, candidate: ActionCandidate,
        receipt: ActionReceipt, verification: Verification,
    ) -> Evaluation:
        if candidate.capability == "control.done":
            valid = self._satisfied(self._capture())
            return Evaluation(valid, True, "note_open_and_grounded" if valid else "goal_check_failed")
        if not verification.passed:
            self._plan = None
            return Evaluation(False, False, "action_unverified_replan")
        self._used.add(int(candidate.intent.removeprefix("option_")))
        if self._satisfied(self._capture()):
            return Evaluation(True, True, "note_open_and_grounded")
        return Evaluation(False, False, "action_verified")
