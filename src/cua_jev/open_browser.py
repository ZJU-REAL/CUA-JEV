"""Experimental, task-agnostic browser loop with a replaceable goal planner.

The planner supplies grounded *options*, not code. Jev chooses among typed
options and execution channels. This module deliberately does not claim that
model-proposed completion checks prove arbitrary user goals.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from .episode import Evaluation
from .errors import CapabilityUnavailable
from .executors.common import execute_with_receipt
from .executors.screen import ScreenController
from .local_tools import ScopedReadOnlyTools, ToolOffer, verify_readonly_tool_result
from .models import ActionCandidate, ActionReceipt, Channel, Decision, Observation, Risk, Verification
from .policy import DecisionPolicy
from .runtime import StepResult
from .verify import VerifierRegistry
from .vision import VisualScene, VisualTarget, WindowImage, browser_viewport_image, changed_fraction

SELECTOR = "a[href],button,input,textarea,select,[role=button],[role=link],[role=textbox]"
MAX_ELEMENTS = 60
MAX_OPTIONS = 16
MAX_INPUT_CHARS = 1000


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError("browser tasks require an http(s) URL without embedded credentials")
    return parsed.scheme, parsed.hostname.lower(), parsed.port


def _matching_visual_target(
    element: BrowserElement, targets: tuple[VisualTarget, ...]
) -> VisualTarget | None:
    if element.box is None or not element.label:
        return None
    label = re.sub(r"\W+", "", element.label, flags=re.UNICODE).casefold()
    if len(label) < 4:
        return None
    left, top, right, bottom = element.box
    element_area = (right - left) * (bottom - top)
    for target in targets:
        target_label = re.sub(r"\W+", "", target.label, flags=re.UNICODE).casefold()
        if len(target_label) < 4 or not (
            label in target_label or target_label in label
        ):
            continue
        x0, y0, x1, y1 = target.box
        overlap = max(0, min(right, x1) - max(left, x0)) * max(
            0, min(bottom, y1) - max(top, y0)
        )
        if overlap >= 0.5 * min(element_area, (x1 - x0) * (y1 - y0)):
            return target
    return None


@dataclass(frozen=True)
class BrowserElement:
    ref: str
    index: int
    tag: str
    role: str
    label: str
    input_type: str
    disabled: bool
    href: str
    box: tuple[int, int, int, int] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BrowserElement:
        raw_box = value.get("box")
        box = None
        if (
            isinstance(raw_box, list) and len(raw_box) == 4
            and all(type(item) is int for item in raw_box)
            and 0 <= raw_box[0] < raw_box[2] <= 1000
            and 0 <= raw_box[1] < raw_box[3] <= 1000
        ):
            box = tuple(raw_box)
        return cls(
            ref=str(value["ref"]),
            index=int(value["index"]),
            tag=str(value["tag"]),
            role=str(value.get("role") or ""),
            label=str(value.get("label") or "")[:120],
            input_type=str(value.get("input_type") or ""),
            disabled=bool(value.get("disabled")),
            href=str(value.get("href") or ""),
            box=box,
        )

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class BrowserSnapshot:
    url: str
    title: str
    text: str
    elements: tuple[BrowserElement, ...]
    visual_summary: str = ""
    visual_text: tuple[str, ...] = ()
    visual_targets: tuple[VisualTarget, ...] = ()
    visual_image_hash: str = ""
    visual_viewport: tuple[int, int] | None = None
    tool_offers: tuple[ToolOffer, ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    required_tool: str = ""

    @classmethod
    def capture(cls, page: Any, *, max_elements: int = 60) -> BrowserSnapshot:
        raw = page.evaluate(
            """request => {
              const nodes = [...document.querySelectorAll(request.selector)];
              const elements = nodes.map((el, index) => {
                const rect = el.getBoundingClientRect();
                const style = getComputedStyle(el);
                if (!rect.width || !rect.height || style.visibility === 'hidden' ||
                    style.display === 'none') return null;
                const label = el.getAttribute('aria-label') ||
                  el.labels?.[0]?.innerText || el.getAttribute('placeholder') ||
                  el.innerText || el.getAttribute('title') || el.getAttribute('name') || '';
                return {
                  ref: `e${index}`, index, tag: el.tagName.toLowerCase(),
                  role: el.getAttribute('role') || '', label: label.trim().slice(0, 120),
                  input_type: el.getAttribute('type') || '', disabled: !!el.disabled,
                  href: el.href || '',
                  box: [
                    Math.max(0, Math.round(rect.left / innerWidth * 1000)),
                    Math.max(0, Math.round(rect.top / innerHeight * 1000)),
                    Math.min(1000, Math.round(rect.right / innerWidth * 1000)),
                    Math.min(1000, Math.round(rect.bottom / innerHeight * 1000))
                  ]
                };
              }).filter(Boolean).slice(0, request.limit);
              const contentRoot = document.querySelector('main, [role="main"], article') ||
                document.body;
              return {
                url: location.href, title: document.title,
                text: (contentRoot?.innerText || '').slice(0, 1600), elements
              };
            }""",
            {"selector": SELECTOR, "limit": max_elements},
        )
        return cls(
            url=str(raw["url"]),
            title=str(raw["title"])[:160],
            text=str(raw["text"])[:1600],
            elements=tuple(BrowserElement.from_dict(item) for item in raw["elements"][:max_elements]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "elements": [element.to_dict() for element in self.elements],
            "visual_summary": self.visual_summary,
            "visual_text": list(self.visual_text),
            "visual_targets": [target.to_dict() for target in self.visual_targets],
            "visual_image_hash": self.visual_image_hash,
            "visual_viewport": self.visual_viewport,
            "tool_offers": [offer.to_dict() for offer in self.tool_offers],
            "tool_results": list(self.tool_results),
            "required_tool": self.required_tool,
        }

    def fingerprint(self) -> str:
        public = self.to_dict()
        return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PlannedOption:
    ref: str
    operation: str
    value: str = ""


@dataclass(frozen=True)
class BrowserPlan:
    subgoal: str
    options: tuple[PlannedOption, ...]
    success_kind: str
    success_value: str

    @classmethod
    def from_dict(cls, data: dict[str, Any], snapshot: BrowserSnapshot) -> BrowserPlan:
        if not isinstance(data, dict) or not isinstance(data.get("options"), list):
            raise ValueError("planner must return an object with an options list")
        items = data["options"]
        if not 1 <= len(items) <= MAX_OPTIONS:
            raise ValueError("planner must return 1-16 options")
        known = {
            element.ref: element for element in
            (*snapshot.elements, *snapshot.visual_targets, *snapshot.tool_offers)
        }
        options: list[PlannedOption] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("planner option must be an object")
            ref, operation = item.get("ref"), item.get("operation")
            if (
                not isinstance(ref, str)
                or not isinstance(operation, str)
                or ref not in known
                or operation not in {"click", "fill", "select", "invoke"}
            ):
                raise ValueError("planner referenced an unavailable element or operation")
            element = known[ref]
            if isinstance(element, ToolOffer):
                if operation != "invoke" or item.get("value"):
                    raise ValueError("local tool refs support invoke only")
                options.append(PlannedOption(ref, "invoke"))
                continue
            if operation == "invoke":
                raise ValueError("invoke requires a registered local tool ref")
            if isinstance(element, VisualTarget):
                if operation != "click" or item.get("value"):
                    raise ValueError("visual targets support click only")
                options.append(PlannedOption(ref, "click"))
                continue
            if element.disabled or element.input_type.lower() == "password":
                raise ValueError("planner referenced a disabled or password control")
            value = item.get("value", "")
            if operation == "click" and value is None:
                value = ""
            if not isinstance(value, str) or len(value) > MAX_INPUT_CHARS:
                raise ValueError("planner value is not valid")
            if operation == "fill" and element.tag not in {"input", "textarea"}:
                raise ValueError("fill requires an input or textarea")
            if operation == "fill" and element.tag == "input" and element.input_type.lower() not in {
                "", "text", "search", "email", "number", "tel", "url"
            }:
                raise ValueError("fill requires a supported text input")
            if operation == "select" and element.tag != "select":
                raise ValueError("select requires a select element")
            if operation in {"fill", "select"} and not value:
                raise ValueError("fill and select require a value")
            options.append(PlannedOption(str(ref), str(operation), value))
        if len({(item.ref, item.operation, item.value) for item in options}) != len(options):
            raise ValueError("planner returned duplicate options")
        success = data.get("success")
        if not isinstance(success, dict) or success.get("kind") not in {
            "url_contains", "title_contains", "text_contains"
        }:
            raise ValueError("planner must provide a supported observable success check")
        expected = success.get("value")
        if not isinstance(expected, str) or not expected or len(expected) > 160:
            raise ValueError("planner success value is invalid")
        subgoal = data.get("subgoal")
        if not isinstance(subgoal, str) or not subgoal.strip() or len(subgoal) > 240:
            raise ValueError("planner subgoal is invalid")
        return cls(subgoal.strip(), tuple(options), str(success["kind"]), expected)


class GoalPlanner(Protocol):
    def plan(
        self, goal: str, snapshot: BrowserSnapshot, recent_actions: Sequence[dict[str, Any]]
    ) -> BrowserPlan: ...


class BrowserVisionGrounder(Protocol):
    def perceive_scene(
        self, goal: str, window_title: str, ui_text: str, image: WindowImage
    ) -> VisualScene: ...


class HttpJsonPlanner:
    """Provider-neutral HTTP contract; a model-specific adapter can expose it.

    POST {goal, snapshot, recent_actions} and return BrowserPlan's JSON shape.
    No raw scripts, selectors, or executable tool calls are accepted.
    """

    def __init__(
        self, endpoint: str, *, api_key: str | None = None, client: httpx.Client | None = None
    ) -> None:
        parsed = urlparse(endpoint)
        if not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("planner endpoint must have a host and no embedded credentials")
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"}
        ):
            raise ValueError("planner endpoint must use HTTPS or loopback HTTP")
        self.endpoint = endpoint
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=45)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def plan(
        self, goal: str, snapshot: BrowserSnapshot, recent_actions: Sequence[dict[str, Any]]
    ) -> BrowserPlan:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = self.client.post(
                self.endpoint,
                headers=headers,
                json={"goal": goal, "snapshot": snapshot.to_dict(), "recent_actions": recent_actions},
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"planner request failed ({type(exc).__name__})") from None
        return BrowserPlan.from_dict(data, snapshot)


class PublicDecisionPolicy:
    """Keep exact input text and locator arguments out of the Jev request."""

    def __init__(self, inner: DecisionPolicy) -> None:
        self.inner = inner

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close:
            close()

    @property
    def last_exchange(self) -> dict[str, Any]:
        return getattr(self.inner, "last_exchange", {})

    def choose(self, observation: Observation, candidates: Sequence[ActionCandidate]) -> Decision:
        public = tuple(replace(item, arguments={}, expected={}) for item in candidates)
        # Exact OCR lines may reproduce private screen content; Jev gets the bounded
        # scene summary and grounded target labels, but not verbatim visual text.
        public_state = {
            key: value for key, value in observation.state.items()
            if key not in {"text", "visual_text", "tool_results"}
        }
        if observation.source == "open-workspace":
            browser = public_state.get("browser", {})
            public_state = {
                "browser": {key: browser.get(key) for key in ("url", "title")},
                "note_exists": public_state.get("note_exists"),
                "editor_open": public_state.get("editor_open"),
            }
        if observation.source == "open-computer":
            browser = public_state.get("browser", {})
            desktop = public_state.get("desktop") or {}
            public_state = {
                "browser": {
                    "url": browser.get("url"), "title": browser.get("title"),
                    "elements": [{
                        key: item.get(key) for key in ("ref", "label", "tag", "disabled")
                    } for item in browser.get("elements", [])],
                    "visual_summary": browser.get("visual_summary"),
                    "visual_targets": browser.get("visual_targets", []),
                },
                "desktop": {
                    "window_title": desktop.get("window_title"),
                    "controls": [{
                        key: item.get(key) for key in ("ref", "name", "control_type", "enabled")
                    } for item in desktop.get("controls", [])],
                    "visual_summary": desktop.get("visual_summary"),
                    "visual_targets": desktop.get("visual_targets", []),
                },
                "tool_offers": public_state.get("tool_offers", []),
                "mcp_offers": public_state.get("mcp_offers", []),
                "artifact": public_state.get("artifact"),
                "requirements": public_state.get("requirements", {}),
            }
        return self.inner.choose(replace(observation, state=public_state), public)


class OpenBrowserTask:
    """An open-goal browser TaskEnvironment; no site-specific task logic."""

    name = "open-browser"

    def __init__(
        self,
        *,
        goal: str,
        start_url: str,
        planner: GoalPlanner,
        page: Any | None = None,
        headed: bool = False,
        allow_form_input: bool = False,
        allow_external_actions: bool = False,
        screen: ScreenController | None = None,
        vision_grounder: BrowserVisionGrounder | None = None,
        vision_mode: str = "fallback",
        allow_screenshot_upload: bool = False,
        allow_visual_clicks: bool = False,
        browser_channel: str = "msedge",
        local_tool_root: Path | None = None,
        required_tool: str = "",
        required_url_contains: str = "",
    ) -> None:
        if not goal.strip():
            raise ValueError("goal is required")
        if vision_mode not in {"fallback", "always"}:
            raise ValueError("vision mode must be fallback or always")
        if vision_grounder is not None and not allow_screenshot_upload:
            raise ValueError("vision requires explicit screenshot-upload consent")
        if browser_channel not in {"msedge", "chrome", "chromium"}:
            raise ValueError("browser channel must be msedge, chrome, or chromium")
        self.goal = goal.strip()
        self.start_url = start_url
        self.allowed_origin = _origin(start_url)
        self.planner = planner
        self.page = page
        self.headed = headed
        self.allow_form_input = allow_form_input
        self.allow_external_actions = allow_external_actions
        self.screen = screen or ScreenController()
        self.vision_grounder = vision_grounder
        self.vision_mode = vision_mode
        self.allow_visual_clicks = allow_visual_clicks
        self.browser_channel = browser_channel
        self.link_hints: tuple[str, ...] = ()
        self.local_tools = ScopedReadOnlyTools(local_tool_root) if local_tool_root else None
        self.required_tool = required_tool
        self.required_url_contains = required_url_contains
        if required_tool and (
            self.local_tools is None
            or required_tool not in {offer.capability for offer in self.local_tools.offers()}
        ):
            raise ValueError("required tool is not offered by the scoped capability pack")
        self._tool_results: list[dict[str, Any]] = []
        self._completed_tools: set[str] = set()
        self._visual_image: WindowImage | None = None
        self._before_images: dict[str, WindowImage] = {}
        self._playwright = None
        self._browser = None
        self._plan: BrowserPlan | None = None
        self._plan_url = ""
        self._plan_fingerprint = ""
        self._used: set[int] = set()
        self._snapshot: BrowserSnapshot | None = None
        self._before: dict[str, BrowserSnapshot] = {}
        self.vision_failures = 0
        self._tool_results.clear()
        self._completed_tools.clear()
        self.planner_calls = 0

    def reset(self) -> None:
        if self.page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                raise CapabilityUnavailable("install cua-jev[browser] for open browser tasks") from None
            self._playwright = sync_playwright().start()
            launch_options = {} if self.browser_channel == "chromium" else {
                "channel": self.browser_channel
            }
            self._browser = self._playwright.chromium.launch(
                headless=not self.headed, **launch_options
            )
            self.page = self._browser.new_page()
        self.page.route("**/*", self._route_request)
        self.page.goto(self.start_url, wait_until="domcontentloaded")
        self._plan = None
        self._used.clear()
        self._visual_image = None
        self.vision_failures = 0

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

    def _capture(self, *, vision: bool = False) -> BrowserSnapshot:
        if self.page is None:
            raise RuntimeError("browser task has not been reset")
        max_elements = 300 if self.link_hints else 60
        snapshot = BrowserSnapshot.capture(self.page, max_elements=max_elements)
        if not snapshot.elements and hasattr(self.page, "wait_for_load_state"):
            # A navigation can expose its new URL before the document's
            # interactive controls are ready. Retry a bounded number of times
            # rather than asking the planner to act on an empty transient DOM.
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                pass
            for _ in range(3):
                snapshot = BrowserSnapshot.capture(self.page, max_elements=max_elements)
                if snapshot.elements:
                    break
                self.page.wait_for_timeout(250)
        if _origin(snapshot.url) != self.allowed_origin:
            raise RuntimeError("browser left the allowed origin")
        if len(snapshot.elements) > 60:
            chosen = list(snapshot.elements[:20] if self.link_hints else snapshot.elements[:60])
            if self.link_hints:
                hints = tuple(item.casefold() for item in self.link_hints)
                chosen.extend(
                    item for item in snapshot.elements[20:]
                    if any(hint in (item.label + " " + item.href).casefold() for hint in hints)
                )
                chosen = chosen[:60]
                if len(chosen) < 60:
                    selected = {item.ref for item in chosen}
                    chosen.extend(
                        item for item in snapshot.elements
                        if item.ref not in selected
                    )
                    chosen = chosen[:60]
            snapshot = replace(snapshot, elements=tuple(sorted(chosen, key=lambda item: item.index)))
        if vision and self.vision_grounder is not None:
            actionable = any(
                not item.disabled and item.input_type.lower() != "password"
                for item in snapshot.elements
            )
            if self.vision_mode == "always" or not actionable:
                image = browser_viewport_image(self.page)
                try:
                    scene = self.vision_grounder.perceive_scene(
                        self.goal, snapshot.title, snapshot.text, image
                    )
                except (RuntimeError, ValueError):
                    self.vision_failures += 1
                    self._visual_image = None
                    if not actionable:
                        raise
                else:
                    self._visual_image = image
                    snapshot = replace(
                        snapshot, visual_summary=scene.summary, visual_text=scene.visible_text,
                        visual_targets=scene.targets if self.allow_visual_clicks else (),
                        visual_image_hash=hashlib.sha256(image.sample).hexdigest(),
                        visual_viewport=(image.region[2], image.region[3]),
                    )
            else:
                self._visual_image = None
        if self.local_tools is not None:
            snapshot = replace(
                snapshot, tool_offers=self.local_tools.offers(),
                tool_results=tuple(self._tool_results[-6:]), required_tool=self.required_tool,
            )
        return snapshot

    def observe(self, history: Sequence[StepResult]) -> Observation:
        snapshot = self._capture(vision=True)
        self._snapshot = snapshot
        return Observation(
            task=self.goal,
            subgoal=self._plan.subgoal if self._plan else "Discover the next browser subgoal",
            state=snapshot.to_dict(),
            source=self.name,
        )

    def observe_state(self, history: Sequence[StepResult]) -> BrowserSnapshot:
        self.observe(history)
        assert self._snapshot is not None
        return self._snapshot

    def capture_state(self) -> BrowserSnapshot:
        return self._capture()

    def compile_option(
        self, snapshot: BrowserSnapshot, option: PlannedOption, subgoal: str
    ) -> tuple[ActionCandidate, ...]:
        """Compile one validated option without exposing planner internals to a host."""
        previous_plan, previous_used = self._plan, self._used
        self._snapshot = snapshot
        self._plan = BrowserPlan(subgoal, (option,), "url_contains", "__not_a_terminal_check__")
        self._used = set()
        try:
            return self._build_candidates()
        finally:
            self._plan, self._used = previous_plan, previous_used

    def _satisfied(self, snapshot: BrowserSnapshot) -> bool:
        if self._plan is None:
            return False
        if self.required_tool and self.required_tool not in self._completed_tools:
            return False
        if self.required_url_contains:
            return self.required_url_contains.casefold() in snapshot.url.casefold()
        expected = self._plan.success_value.casefold()
        actual = {
            "url_contains": snapshot.url,
            "title_contains": snapshot.title,
            "text_contains": snapshot.text,
        }[self._plan.success_kind]
        return expected in actual.casefold()

    def _call_planner(self, history: Sequence[StepResult]) -> None:
        if self._snapshot is None:
            raise RuntimeError("observe before planning")
        recent = [
            {
                "candidate_id": item.decision.candidate_id,
                "verified": item.verification.passed,
                "success": item.receipt.success,
            }
            for item in history[-6:]
        ]
        self._plan = self.planner.plan(self.goal, self._snapshot, recent)
        self._plan_url = self._snapshot.url
        self._plan_fingerprint = self._snapshot.fingerprint()
        self._used.clear()
        self.planner_calls += 1

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]:
        if self._snapshot is None:
            raise RuntimeError("observe before candidate generation")
        if (
            self._plan is None
            or self._plan_url != self._snapshot.url
            or self._plan_fingerprint != self._snapshot.fingerprint()
        ):
            self._call_planner(history)
        if self._satisfied(self._snapshot):
            return (
                ActionCandidate(
                    "done", Channel.CONTROL, "control.done",
                    "Finish after the observable goal check passed.", intent="finish"
                ),
            )
        result = self._build_candidates()
        if not result:
            self._call_planner(history)
            result = self._build_candidates()
        return result

    def _build_candidates(self) -> tuple[ActionCandidate, ...]:
        assert self._snapshot is not None and self._plan is not None
        known = {element.ref: element for element in self._snapshot.elements}
        tools = {offer.ref: offer for offer in self._snapshot.tool_offers}
        result: list[ActionCandidate] = []
        for index, option in enumerate(self._plan.options):
            if index in self._used:
                continue
            tool = tools.get(option.ref)
            if tool is not None:
                if option.operation == "invoke" and tool.capability not in self._completed_tools:
                    result.append(tool.candidate(index))
                continue
            visual = next(
                (target for target in self._snapshot.visual_targets if target.ref == option.ref),
                None,
            )
            if visual is not None:
                if not (self.allow_visual_clicks and self.allow_external_actions):
                    continue
                result.append(self._visual_candidate(index, visual))
                continue
            element = known.get(option.ref)
            if element is None or element.disabled or element.input_type.lower() == "password":
                continue
            if option.operation in {"fill", "select"} and not self.allow_form_input:
                continue
            if option.operation == "click" and element.href:
                try:
                    if _origin(element.href) != self.allowed_origin:
                        continue
                except ValueError:
                    continue
            is_link = option.operation == "click" and element.tag == "a" and bool(element.href)
            if option.operation == "click" and not is_link and not self.allow_external_actions:
                continue
            risk = (
                Risk.READ_ONLY if is_link else Risk.EXTERNAL_SIDE_EFFECT
                if option.operation == "click" else Risk.LOCAL_WRITE
            )
            name = element.label or element.tag
            description = f"{option.operation.title()} {name[:70]}"
            arguments = {
                "ref": element.ref,
                "index": element.index,
                "label": element.label,
                "tag": element.tag,
                "operation": option.operation,
                "value": option.value,
            }
            for channel, suffix in ((Channel.SCRIPT, "dom"), (Channel.GUI, "gui")):
                if channel == Channel.GUI and not self.headed:
                    continue
                if option.operation == "select" and channel == Channel.GUI:
                    continue
                result.append(
                    ActionCandidate(
                        id=f"option_{index}_{suffix}", channel=channel,
                        capability=f"browser.{option.operation}",
                        description=f"{description} using {suffix.upper()}",
                        arguments=arguments, risk=risk,
                        verifier="browser.effect", intent=f"option_{index}",
                    )
                )
            if option.operation == "click" and self.allow_visual_clicks and self.allow_external_actions:
                match = _matching_visual_target(element, self._snapshot.visual_targets)
                if match is not None:
                    result.append(self._visual_candidate(index, match))
        unique: dict[tuple[str, str, str, str, str], ActionCandidate] = {}
        for candidate in result:
            args = candidate.arguments
            if candidate.capability.startswith(("filesystem.", "cli.")):
                key = (
                    str(candidate.channel), candidate.capability,
                    str(args.get("path", "")), "", "",
                )
                unique[key] = candidate
                continue
            key = (
                str(candidate.channel), candidate.capability, str(args["ref"]),
                str(args["operation"]), str(args.get("value", "")),
            )
            unique.setdefault(key, candidate)
        return tuple(unique.values())

    def _visual_candidate(self, index: int, target: VisualTarget) -> ActionCandidate:
        assert self._snapshot is not None
        return ActionCandidate(
            id=f"option_{index}_visual", channel=Channel.SCRIPT,
            capability="browser.visual_click",
            description=f"Click visually grounded {target.label} using browser mouse",
            arguments={
                "ref": target.ref, "label": target.label, "box": list(target.box),
                "visual_image_hash": self._snapshot.visual_image_hash,
                "visual_viewport": self._snapshot.visual_viewport,
                "url": self._snapshot.url, "operation": "visual_click",
            },
            risk=Risk.EXTERNAL_SIDE_EFFECT, verifier="browser.effect",
            intent=f"option_{index}",
        )

    def executor_bindings(self) -> dict[Channel, Any]:
        bindings: dict[Channel, Any] = {Channel.SCRIPT: self}
        if self.local_tools is not None:
            from .executors.cli import RegisteredCliExecutor
            from .executors.filesystem import FileSystemExecutor

            bindings[Channel.API] = FileSystemExecutor()
            bindings[Channel.CLI] = RegisteredCliExecutor()
        if self.headed:
            bindings[Channel.GUI] = self
        return bindings

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("browser.effect", self._verify_effect)
        registry.register("tool.result", verify_readonly_tool_result)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            before = self._capture()
            self._before[decision_id] = before
            if candidate.capability == "browser.visual_click":
                if self._snapshot is None or self._visual_image is None:
                    raise RuntimeError("visual browser observation is unavailable")
                target = next(
                    (item for item in self._snapshot.visual_targets
                     if item.ref == candidate.arguments["ref"]), None,
                )
                if target is None or target.label != candidate.arguments["label"] or (
                    list(target.box) != candidate.arguments["box"]
                ):
                    raise RuntimeError("visual browser target changed since observation")
                prior = replace(
                    self._snapshot, visual_summary="", visual_text=(), visual_targets=(),
                    visual_image_hash="", visual_viewport=None,
                )
                if before.fingerprint() != prior.fingerprint() or before.url != candidate.arguments["url"]:
                    raise RuntimeError("browser DOM changed since visual observation")
                current = browser_viewport_image(self.page)
                if (
                    current.region != self._visual_image.region
                    or candidate.arguments["visual_viewport"]
                    != (current.region[2], current.region[3])
                    or candidate.arguments["visual_image_hash"]
                    != hashlib.sha256(self._visual_image.sample).hexdigest()
                    or changed_fraction(self._visual_image.sample, current.sample) > 0.08
                ):
                    raise RuntimeError("visual browser screenshot became stale")
                self._before_images[decision_id] = current
                _, _, width, height = current.region
                left, top, right, bottom = target.box
                self.page.mouse.click((left + right) * width / 2000, (top + bottom) * height / 2000)
                return {"url": self.page.url, "visual_target": target.ref}
            element = next((item for item in before.elements if item.ref == candidate.arguments["ref"]), None)
            if element is None or any(
                getattr(element, name) != candidate.arguments[name]
                for name in ("label", "tag")
            ):
                raise RuntimeError("browser element changed since observation")
            locator = self.page.locator(SELECTOR).nth(element.index)
            action = candidate.arguments["operation"]
            if candidate.channel == Channel.GUI:
                self._physical_action(locator, candidate)
            elif action == "click":
                locator.click()
            elif action == "fill":
                locator.fill(candidate.arguments["value"])
            elif action == "select":
                locator.select_option(candidate.arguments["value"])
            else:
                raise ValueError("unsupported browser operation")
            return {"url": self.page.url, "element": element.ref, "operation": action}

        return execute_with_receipt(candidate, observation_id, decision_id, operation)

    def _physical_action(self, locator: Any, candidate: ActionCandidate) -> None:
        if not self.headed:
            raise RuntimeError("physical GUI route requires a headed browser")
        title = self.page.title()
        if not title:
            raise RuntimeError("cannot safely focus a browser window without a title")
        window = self.screen.focus(rf".*{re.escape(title)}.*", maximize=False)
        locator.scroll_into_view_if_needed()
        point = locator.evaluate(
            """element => {
              const rect = element.getBoundingClientRect();
              return {
                x: (window.outerWidth - window.innerWidth) / 2 + rect.left + rect.width / 2,
                y: window.outerHeight - window.innerHeight + rect.top + rect.height / 2,
                outerWidth: window.outerWidth, outerHeight: window.outerHeight
              };
            }"""
        )
        rect = window.rectangle()
        x = rect.left + float(point["x"]) * rect.width() / float(point["outerWidth"])
        y = rect.top + float(point["y"]) * rect.height() / float(point["outerHeight"])
        self.screen.click_point(x, y)
        if candidate.arguments["operation"] == "fill":
            self.screen.hotkey("ctrl", "a")
            self.screen.paste_text(candidate.arguments["value"])

    def _verify_effect(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        if not receipt.success:
            return Verification(False, "browser.effect", {"error": receipt.error})
        before = self._before.get(receipt.decision_id)
        if before is None:
            return Verification(False, "browser.effect", {"error": "missing pre-action snapshot"})
        after = self._capture()
        if candidate.capability == "browser.visual_click":
            image = self._before_images.get(receipt.decision_id)
            if image is None:
                return Verification(False, "browser.effect", {"error": "missing visual before image"})
            current = browser_viewport_image(self.page)
            fraction = changed_fraction(image.sample, current.sample)
            passed = after.fingerprint() != before.fingerprint() or fraction >= 0.01
            return Verification(passed, "browser.effect", {
                "state_changed": after.fingerprint() != before.fingerprint(),
                "visual_changed_fraction": round(fraction, 4),
            })
        if candidate.arguments["operation"] in {"fill", "select"}:
            locator = self.page.locator(SELECTOR).nth(candidate.arguments["index"])
            passed = locator.input_value() == candidate.arguments["value"]
        else:
            passed = after.fingerprint() != before.fingerprint()
        return Verification(
            passed, "browser.effect",
            {"before_url": before.url, "after_url": after.url, "state_changed": passed},
        )

    def evaluate(
        self,
        observation: Observation,
        candidate: ActionCandidate,
        receipt: ActionReceipt,
        verification: Verification,
    ) -> Evaluation:
        if candidate.capability == "control.done":
            valid = self._satisfied(self._capture())
            return Evaluation(valid, True, "goal_check_passed" if valid else "goal_check_failed")
        if not verification.passed:
            self._plan = None
            return Evaluation(False, False, "action_unverified_replan")
        index = int(candidate.intent.removeprefix("option_"))
        self._used.add(index)
        if candidate.capability.startswith(("filesystem.", "cli.")):
            self._completed_tools.add(candidate.capability)
            output = receipt.output
            summary = output.get("text", output.get("stdout", output.get("entries", "")))
            self._tool_results.append({
                "capability": candidate.capability,
                "result": str(summary)[:2000], "verified": True,
            })
        after = self._capture()
        if self._satisfied(after):
            return Evaluation(True, True, "goal_check_passed")
        if after.url != self._plan_url:
            self._plan = None
        return Evaluation(False, False, "action_verified")
