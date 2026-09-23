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
from typing import Any, Protocol
from urllib.parse import urlparse

import httpx

from .episode import Evaluation
from .errors import CapabilityUnavailable
from .executors.common import execute_with_receipt
from .executors.screen import ScreenController
from .models import ActionCandidate, ActionReceipt, Channel, Decision, Observation, Risk, Verification
from .policy import DecisionPolicy
from .runtime import StepResult
from .verify import VerifierRegistry

SELECTOR = "a[href],button,input,textarea,select,[role=button],[role=link],[role=textbox]"
MAX_ELEMENTS = 60
MAX_OPTIONS = 16
MAX_INPUT_CHARS = 1000


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        raise ValueError("browser tasks require an http(s) URL without embedded credentials")
    return parsed.scheme, parsed.hostname.lower(), parsed.port


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

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BrowserElement:
        return cls(
            ref=str(value["ref"]),
            index=int(value["index"]),
            tag=str(value["tag"]),
            role=str(value.get("role") or ""),
            label=str(value.get("label") or "")[:120],
            input_type=str(value.get("input_type") or ""),
            disabled=bool(value.get("disabled")),
            href=str(value.get("href") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


@dataclass(frozen=True)
class BrowserSnapshot:
    url: str
    title: str
    text: str
    elements: tuple[BrowserElement, ...]

    @classmethod
    def capture(cls, page: Any) -> BrowserSnapshot:
        raw = page.evaluate(
            """selector => {
              const nodes = [...document.querySelectorAll(selector)];
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
                  href: el.href || ''
                };
              }).filter(Boolean).slice(0, 60);
              return {
                url: location.href, title: document.title,
                text: (document.body?.innerText || '').slice(0, 1600), elements
              };
            }""",
            SELECTOR,
        )
        return cls(
            url=str(raw["url"]),
            title=str(raw["title"])[:160],
            text=str(raw["text"])[:1600],
            elements=tuple(BrowserElement.from_dict(item) for item in raw["elements"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "elements": [element.to_dict() for element in self.elements],
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
        known = {element.ref: element for element in snapshot.elements}
        options: list[PlannedOption] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("planner option must be an object")
            ref, operation = item.get("ref"), item.get("operation")
            if (
                not isinstance(ref, str)
                or not isinstance(operation, str)
                or ref not in known
                or operation not in {"click", "fill", "select"}
            ):
                raise ValueError("planner referenced an unavailable element or operation")
            element = known[ref]
            if element.disabled or element.input_type.lower() == "password":
                raise ValueError("planner referenced a disabled or password control")
            value = item.get("value", "")
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
        public_state = {key: value for key, value in observation.state.items() if key != "text"}
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
    ) -> None:
        if not goal.strip():
            raise ValueError("goal is required")
        self.goal = goal.strip()
        self.start_url = start_url
        self.allowed_origin = _origin(start_url)
        self.planner = planner
        self.page = page
        self.headed = headed
        self.allow_form_input = allow_form_input
        self.allow_external_actions = allow_external_actions
        self.screen = screen or ScreenController()
        self._playwright = None
        self._browser = None
        self._plan: BrowserPlan | None = None
        self._plan_url = ""
        self._used: set[int] = set()
        self._snapshot: BrowserSnapshot | None = None
        self._before: dict[str, BrowserSnapshot] = {}
        self.planner_calls = 0

    def reset(self) -> None:
        if self.page is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError:
                raise CapabilityUnavailable("install cua-jev[browser] for open browser tasks") from None
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(channel="msedge", headless=not self.headed)
            self.page = self._browser.new_page()
        self.page.route("**/*", self._route_request)
        self.page.goto(self.start_url, wait_until="domcontentloaded")
        self._plan = None
        self._used.clear()

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

    def _capture(self) -> BrowserSnapshot:
        if self.page is None:
            raise RuntimeError("browser task has not been reset")
        snapshot = BrowserSnapshot.capture(self.page)
        if _origin(snapshot.url) != self.allowed_origin:
            raise RuntimeError("browser left the allowed origin")
        return snapshot

    def observe(self, history: Sequence[StepResult]) -> Observation:
        snapshot = self._capture()
        self._snapshot = snapshot
        return Observation(
            task=self.goal,
            subgoal=self._plan.subgoal if self._plan else "Discover the next browser subgoal",
            state=snapshot.to_dict(),
            source=self.name,
        )

    def _satisfied(self, snapshot: BrowserSnapshot) -> bool:
        if self._plan is None:
            return False
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
        self._used.clear()
        self.planner_calls += 1

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]:
        if self._snapshot is None:
            raise RuntimeError("observe before candidate generation")
        if self._plan is None or self._plan_url != self._snapshot.url:
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
        result: list[ActionCandidate] = []
        for index, option in enumerate(self._plan.options):
            if index in self._used:
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
        return tuple(result)

    def executor_bindings(self) -> dict[Channel, Any]:
        bindings: dict[Channel, Any] = {Channel.SCRIPT: self}
        if self.headed:
            bindings[Channel.GUI] = self
        return bindings

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        registry.register("browser.effect", self._verify_effect)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        def operation() -> dict[str, Any]:
            before = self._capture()
            self._before[decision_id] = before
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
        after = self._capture()
        if self._satisfied(after):
            return Evaluation(True, True, "goal_check_passed")
        if after.url != self._plan_url:
            self._plan = None
        return Evaluation(False, False, "action_verified")
