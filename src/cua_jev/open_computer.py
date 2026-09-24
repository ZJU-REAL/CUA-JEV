"""Experimental, bounded browser + Windows-window + local-tool action loop.

This composes existing observers/executors instead of encoding an application
workflow. Each plan names live refs, while terminal requirements come from the
caller rather than from a model's self-assessment.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urldefrag

from .artifact_surface import ArtifactState, ArtifactSurface
from .episode import Evaluation
from .local_tools import ScopedReadOnlyTools, ToolOffer
from .mcp_surface import McpCallOffer, McpToolSurface
from .models import ActionCandidate, ActionReceipt, Channel, Observation, Verification
from .open_browser import BrowserSnapshot, OpenBrowserTask
from .open_desktop import DesktopSnapshot, OpenDesktopTask
from .runtime import StepResult
from .surface_providers import (
    ActionSurface,
    BrowserSurface,
    ReadOnlyToolSurface,
    WindowsUiaSurface,
)
from .verify import VerifierRegistry

MAX_OPTIONS = 16


def _page_key(url: str) -> str:
    """A fragment changes scroll position, not the fetched source document."""
    return urldefrag(url).url


@dataclass(frozen=True)
class ComputerSnapshot:
    browser: BrowserSnapshot
    desktop: DesktopSnapshot | None
    tool_offers: tuple[ToolOffer, ...]
    tool_results: tuple[dict[str, Any], ...]
    requirements: dict[str, Any]
    mcp_offers: tuple[McpCallOffer, ...] = ()
    artifact: ArtifactState | None = None
    source_records: tuple[dict[str, str], ...] = ()
    providers: dict[str, ActionSurface] = field(default_factory=dict, repr=False, compare=False)

    def state_for(self, namespace: str) -> Any:
        states = {"b": self.browser, "t": self.tool_offers, "m": self.mcp_offers}
        if self.desktop is not None:
            states["d"] = self.desktop
        if self.artifact is not None:
            states["a"] = self.artifact
        return states[namespace]

    def to_dict(self) -> dict[str, Any]:
        browser = self.browser.to_dict()
        desktop = self.desktop.to_dict() if self.desktop is not None else None
        for key in ("elements", "visual_targets"):
            browser[key] = [{**item, "ref": f"b:{item['ref']}"} for item in browser[key]]
        if desktop is not None:
            for key in ("controls", "visual_targets"):
                desktop[key] = [{**item, "ref": f"d:{item['ref']}"} for item in desktop[key]]
        return {
            "browser": browser, "desktop": desktop,
            "tool_offers": [
                {**offer.to_dict(), "ref": f"t:{offer.ref}"} for offer in self.tool_offers
            ],
            "mcp_offers": [
                {**offer.to_dict(), "ref": f"m:{offer.ref}"} for offer in self.mcp_offers
            ],
            "artifact": {**self.artifact.to_dict(), "ref": f"a:{self.artifact.ref}"}
            if self.artifact is not None else None,
            "source_records": list(self.source_records),
            "tool_results": list(self.tool_results),
            "requirements": self.requirements,
        }

    def for_model(self) -> dict[str, Any]:
        """Keep planning refs and semantics, omit runtime-only handles/geometry."""
        state = self.to_dict()
        browser = state["browser"]
        browser["elements"] = [{
            key: value for key, value in element.items()
            if key in {"ref", "tag", "role", "label", "input_type", "disabled", "href"}
        } for element in browser["elements"]]
        for element in browser["elements"]:
            element["href"] = element.get("href", "")[:240]
        for key in ("visual_image_hash", "visual_viewport"):
            browser.pop(key, None)
        desktop = state["desktop"]
        if desktop is not None:
            desktop["controls"] = [{
                key: value for key, value in control.items()
                if key in {"ref", "name", "control_type", "enabled", "can_invoke", "can_set_text"}
            } for control in desktop["controls"]]
            for key in ("window_handle", "visual_region", "visual_image_hash"):
                desktop.pop(key, None)
        return state

    def fingerprint(self) -> str:
        state = {**self.to_dict(), "desktop_private_fingerprint": (
            self.desktop.fingerprint() if self.desktop is not None else None
        )}
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ComputerOption:
    source: str
    ref: str
    operation: str
    value: str = ""


@dataclass(frozen=True)
class ComputerPlan:
    subgoal: str
    options: tuple[ComputerOption, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any], snapshot: ComputerSnapshot) -> ComputerPlan:
        if not isinstance(data, dict) or not isinstance(data.get("options"), list):
            raise ValueError("computer planner must return an options list")
        subgoal = data.get("subgoal")
        if not isinstance(subgoal, str) or not subgoal.strip():
            subgoal = "Advance the remaining task requirements"
        if not 1 <= len(data["options"]) <= MAX_OPTIONS:
            raise ValueError("computer planner must return 1-16 options")
        options: list[ComputerOption] = []
        rejected: list[str] = []
        for item in data["options"]:
            if not isinstance(item, dict) or not isinstance(item.get("ref"), str):
                rejected.append("option has no live target ref")
                continue
            namespace, separator, ref = item["ref"].partition(":")
            provider = snapshot.providers.get(namespace)
            if not separator or provider is None:
                rejected.append(f"{item['ref']}: unregistered namespace")
                continue
            try:
                checked = provider.validate(
                    snapshot.state_for(namespace), ref, item.get("operation"), item.get("value", "")
                )
            except ValueError as exc:
                rejected.append(f"{item['ref']} ({item.get('operation')}): {exc}")
                continue
            option = ComputerOption(namespace, *checked)
            if option not in options:
                options.append(option)
        if not options:
            detail = "; ".join(rejected[:3])
            raise ValueError(f"computer planner offered no valid action: {detail}")
        return cls(subgoal.strip()[:240], tuple(options))


class ComputerGoalPlanner(Protocol):
    def plan(
        self, goal: str, snapshot: ComputerSnapshot, recent_actions: Sequence[dict[str, Any]]
    ) -> ComputerPlan: ...


class OpenComputerTask:
    """One browser origin + one selected Windows window + optional local tools."""

    name = "open-computer"

    def __init__(
        self,
        *,
        goal: str,
        browser: OpenBrowserTask,
        desktop: OpenDesktopTask | None = None,
        desktop_surface: ActionSurface | None = None,
        planner: ComputerGoalPlanner,
        local_tool_root: Path | None = None,
        mcp_surface: McpToolSurface | None = None,
        artifact_surface: ArtifactSurface | None = None,
        mcp_current_page_only: bool = False,
        mcp_read_for_visits: bool = False,
        min_verified_sources: int = 0,
        required_source_titles: Sequence[str] = (),
        required_visits: Sequence[str] = (),
        required_artifact_contains: Sequence[str] = (),
        required_url_contains: str = "",
        required_window_title_contains: str = "",
        required_window_text_contains: str = "",
        required_capabilities: Sequence[str] = (),
    ) -> None:
        if not goal.strip() or browser.goal != goal.strip() or (
            desktop is not None and desktop.goal != goal.strip()
        ):
            raise ValueError("the composite goal must match both observers")
        if desktop_surface is not None and desktop_surface.namespace != "d":
            raise ValueError("replacement desktop surface must use the d namespace")
        if desktop_surface is None and desktop is None and (
            required_window_title_contains or required_window_text_contains
        ):
            raise ValueError("window-state gates require a desktop surface provider")
        if not any((required_url_contains, required_window_title_contains,
                    required_window_text_contains, required_visits, artifact_surface)):
            raise ValueError("at least one caller-supplied observable state gate is required")
        if required_artifact_contains and artifact_surface is None:
            raise ValueError("artifact text gates require an artifact surface")
        if mcp_current_page_only and (
            mcp_surface is None or not mcp_surface.offers
            or any("href" not in (offer.parameters or {}) for offer in mcp_surface.offers)
        ):
            raise ValueError("current-page MCP mode needs registered href parameter offers")
        visits = tuple(item.strip() for item in required_visits if item.strip())
        if len(visits) > 12 or len(visits) != len(set(visits)) or any(
            len(item) > 160 for item in visits
        ):
            raise ValueError("required visits must be at most 12 unique short URL clues")
        if mcp_read_for_visits and (not mcp_current_page_only or not visits):
            raise ValueError("per-visit MCP evidence requires current-page mode and visits")
        if type(min_verified_sources) is not int or not 0 <= min_verified_sources <= 12:
            raise ValueError("minimum verified sources must be an integer from 0 to 12")
        if min_verified_sources and (
            visits or not mcp_current_page_only or mcp_surface is None
            or artifact_surface is None
        ):
            raise ValueError(
                "discovered sources need a new artifact, current-page MCP, and no visit clues"
            )
        if any(not isinstance(item, str) for item in required_source_titles):
            raise ValueError("source-title clues must be strings")
        titles = tuple(item.strip() for item in required_source_titles if item.strip())
        if titles and (
            not min_verified_sources or len(titles) > 12
            or len({item.casefold() for item in titles}) != len(titles)
            or any(len(item) > 120 for item in titles)
        ):
            raise ValueError("source-title gates need discovered-source mode and unique short clues")
        self.goal = goal.strip()
        self.browser = browser
        if titles:
            self.browser.link_hints = titles
        self.desktop = desktop
        self.planner = planner
        self.local_tools = ScopedReadOnlyTools(local_tool_root) if local_tool_root else None
        self.providers: dict[str, ActionSurface] = {"b": BrowserSurface(browser)}
        if desktop_surface is not None:
            self.providers["d"] = desktop_surface
        elif desktop is not None:
            self.providers["d"] = WindowsUiaSurface(desktop)
        if self.local_tools is not None:
            self.providers["t"] = ReadOnlyToolSurface(self.local_tools)
        if mcp_surface is not None:
            self.providers["m"] = mcp_surface
        if artifact_surface is not None:
            self.providers["a"] = artifact_surface
        self.artifact_surface = artifact_surface
        self.mcp_current_page_only = mcp_current_page_only
        self.mcp_read_for_visits = mcp_read_for_visits
        self.min_verified_sources = min_verified_sources
        self.required_source_titles = titles
        self._mcp_pages_read: set[str] = set()
        self._discovered_sources: dict[str, dict[str, str]] = {}
        self.required_visits = visits
        self.required_artifact_contains = tuple(required_artifact_contains)
        self._visits: dict[str, dict[str, str]] = {}
        self.required_url_contains = required_url_contains
        self.required_window_title_contains = required_window_title_contains
        self.required_window_text_contains = required_window_text_contains
        self.required_capabilities = tuple(required_capabilities)
        supported_actions = set().union(
            *(provider.supported_capabilities for provider in self.providers.values())
        )
        for capability in self.required_capabilities:
            if capability not in supported_actions:
                raise ValueError("required capability is not registered in this task")
        self._snapshot: ComputerSnapshot | None = None
        self._plan: ComputerPlan | None = None
        self._plan_fingerprint = ""
        self._used: set[int] = set()
        self._completed_capabilities: set[str] = set()
        self._tool_results: list[dict[str, Any]] = []
        self.planner_calls = 0

    def reset(self) -> None:
        for provider in self.providers.values():
            provider.reset()
        self._plan = None
        self._used.clear()
        self._completed_capabilities.clear()
        self._tool_results.clear()
        self._visits.clear()
        self._mcp_pages_read.clear()
        self._discovered_sources.clear()
        if self.artifact_surface is not None:
            self.artifact_surface.set_acceptance(
                ready=not self.required_visits and not self.min_verified_sources,
                citations=(),
            )

    def _source_records(self) -> tuple[dict[str, str], ...]:
        return (*self._visits.values(), *self._discovered_sources.values())

    def _pending_source_titles(self) -> list[str]:
        return [
            clue for clue in self.required_source_titles
            if not any(
                clue.casefold() in record["title"].casefold()
                for record in self._discovered_sources.values()
            )
        ]

    def _record_visit(self, browser: BrowserSnapshot) -> None:
        for clue in self.required_visits:
            if clue not in self._visits and clue.casefold() in browser.url.casefold():
                self._visits[clue] = {
                    "clue": clue, "url": browser.url,
                    "title": browser.title, "excerpt": browser.text[:500],
                }
        if self.artifact_surface is not None:
            visits_ready = len(self._visits) == len(self.required_visits)
            sources_ready = (
                len(self._discovered_sources) >= self.min_verified_sources
                and not self._pending_source_titles()
            )
            reads_ready = not self.mcp_read_for_visits or all(
                _page_key(record["url"]) in self._mcp_pages_read
                for record in self._visits.values()
            )
            self.artifact_surface.set_acceptance(
                ready=visits_ready and sources_ready and reads_ready,
                citations=[record["url"] for record in self._source_records()],
            )

    def _requirements(self, browser: BrowserSnapshot) -> dict[str, Any]:
        pending_page = (
            browser.url if self.min_verified_sources
            and (
                len(self._discovered_sources) < self.min_verified_sources
                or self._pending_source_titles()
            )
            and _page_key(browser.url) != _page_key(self.browser.start_url)
            and _page_key(browser.url) not in self._mcp_pages_read else ""
        )
        return {
            "url_contains": self.required_url_contains,
            "window_title_contains": self.required_window_title_contains,
            "window_text_contains": self.required_window_text_contains,
            "capabilities": list(self.required_capabilities),
            "completed_capabilities": sorted(self._completed_capabilities),
            "required_visits": list(self.required_visits),
            "pending_visits": [clue for clue in self.required_visits if clue not in self._visits],
            "min_verified_sources": self.min_verified_sources,
            "verified_source_count": len(self._discovered_sources),
            "pending_source_titles": self._pending_source_titles(),
            "pending_mcp_page": pending_page,
            "artifact_contains": list(self.required_artifact_contains),
            "mcp_current_page_only": self.mcp_current_page_only,
            "mcp_pages_read": sorted(self._mcp_pages_read),
            "pending_mcp_visits": [
                clue for clue, record in self._visits.items()
                if self.mcp_read_for_visits
                and _page_key(record["url"]) not in self._mcp_pages_read
            ],
        }

    def close(self) -> None:
        for provider in reversed(tuple(self.providers.values())):
            provider.close()

    def _active_tool_offers(self) -> tuple[ToolOffer, ...]:
        if "t" not in self.providers:
            return ()
        return tuple(
            offer for offer in self.providers["t"].capture()
            if offer.capability not in self._completed_capabilities
        )

    def _active_mcp_offers(self, browser: BrowserSnapshot) -> tuple[McpCallOffer, ...]:
        if "m" not in self.providers:
            return ()
        if self.min_verified_sources and (
            _page_key(browser.url) == _page_key(self.browser.start_url)
        ):
            return ()
        if self.mcp_read_for_visits and not any(
            record["url"] == browser.url for record in self._visits.values()
        ):
            return ()
        if self.mcp_current_page_only and _page_key(browser.url) in self._mcp_pages_read:
            return ()
        return self.providers["m"].capture()

    def _capture(self) -> ComputerSnapshot:
        browser = self.providers["b"].capture()
        self._record_visit(browser)
        return ComputerSnapshot(
            browser=browser,
            desktop=self.providers["d"].capture() if "d" in self.providers else None,
            tool_offers=self._active_tool_offers(),
            mcp_offers=self._active_mcp_offers(browser),
            artifact=self.providers["a"].capture() if "a" in self.providers else None,
            source_records=self._source_records(),
            tool_results=tuple(self._tool_results[-6:]),
            requirements=self._requirements(browser),
            providers=self.providers,
        )

    def observe(self, history: Sequence[StepResult]) -> Observation:
        browser = self.providers["b"].observe(history)
        self._record_visit(browser)
        self._snapshot = ComputerSnapshot(
            browser=browser,
            desktop=self.providers["d"].observe(history) if "d" in self.providers else None,
            tool_offers=self._active_tool_offers(),
            mcp_offers=self._active_mcp_offers(browser),
            artifact=self.providers["a"].observe(history) if "a" in self.providers else None,
            source_records=self._source_records(),
            tool_results=tuple(self._tool_results[-6:]),
            requirements=self._requirements(browser),
            providers=self.providers,
        )
        return Observation(
            task=self.goal,
            subgoal=self._plan.subgoal if self._plan else "Choose the next grounded cross-app action",
            state=self._snapshot.to_dict(), source=self.name,
        )

    def _satisfied(self, snapshot: ComputerSnapshot) -> bool:
        desktop = snapshot.desktop
        requirements = (
            not self.required_url_contains or
            self.required_url_contains.casefold() in snapshot.browser.url.casefold(),
            not self.required_window_title_contains or (
                desktop is not None and
            self.required_window_title_contains.casefold()
            in desktop.window_title.casefold()),
            not self.required_window_text_contains or (
                desktop is not None and
            self.required_window_text_contains.casefold() in (
                desktop.text + "\n" + "\n".join(
                    item.name for item in desktop.controls
                ) + "\n" + "\n".join(desktop.edit_values)
            ).casefold()),
            all(item in self._completed_capabilities for item in self.required_capabilities),
            all(item in self._visits for item in self.required_visits),
            len(self._discovered_sources) >= self.min_verified_sources,
            not self._pending_source_titles(),
            not self.mcp_read_for_visits or all(
                _page_key(record["url"]) in self._mcp_pages_read
                for record in self._visits.values()
            ),
            self.artifact_surface is None or (
                snapshot.artifact is not None and snapshot.artifact.exists
                and all(item in snapshot.artifact.text for item in self.required_artifact_contains)
                and all(
                    record["url"] in snapshot.artifact.text for record in self._source_records()
                )
            ),
        )
        return all(requirements)

    def _call_planner(
        self, history: Sequence[StepResult], *, feedback: str = ""
    ) -> None:
        assert self._snapshot is not None
        recent = [
            {"candidate_id": step.decision.candidate_id, "verified": step.verification.passed}
            for step in history[-6:]
        ]
        if feedback:
            recent.append({"planner_feedback": feedback})
        for attempt in range(2):
            self.planner_calls += 1
            try:
                self._plan = self.planner.plan(self.goal, self._snapshot, recent)
                break
            except ValueError as exc:
                if attempt:
                    raise
                recent.append({
                    "planner_feedback": f"Previous plan failed validation: {exc}. "
                    "Use only refs listed in the current snapshot with their supported "
                    "operations; no action was executed."
                })
        self._plan_fingerprint = self._snapshot.fingerprint()
        self._used.clear()

    def candidates(
        self, observation: Observation, history: Sequence[StepResult]
    ) -> Sequence[ActionCandidate]:
        if self._snapshot is None:
            raise RuntimeError("observe before candidate generation")
        if self._satisfied(self._snapshot):
            return (ActionCandidate(
                "done", Channel.CONTROL, "control.done", "Finish after caller acceptance gates passed.",
            ),)
        if self._plan is None or self._plan_fingerprint != self._snapshot.fingerprint():
            self._call_planner(history)
        result = self._build_candidates()
        if not result:
            self._call_planner(
                history,
                feedback=(
                    "The last plan compiled to zero legal actions. Do not choose a completed "
                    "tool, a self-link, or an action outside the configured origin. Choose a "
                    "different live ref that advances pending_visits or other unmet requirements."
                ),
            )
            result = self._build_candidates()
        if not result:
            offered = [(option.source, option.ref, option.operation) for option in self._plan.options]
            raise ValueError(f"planner offered no executable actions: {offered}")
        return result

    def _build_candidates(self) -> tuple[ActionCandidate, ...]:
        assert self._snapshot is not None and self._plan is not None
        result: list[ActionCandidate] = []
        for index, option in enumerate(self._plan.options):
            if index in self._used:
                continue
            if option.source == "b" and self._snapshot.requirements["pending_mcp_page"]:
                continue
            if self.mcp_read_for_visits and option.source == "b" and any(
                record["url"] == self._snapshot.browser.url
                and _page_key(record["url"]) not in self._mcp_pages_read
                for record in self._visits.values()
            ):
                continue
            provider = self.providers[option.source]
            candidates = provider.compile(
                self._snapshot.state_for(option.source), option.ref,
                option.operation, option.value, self._plan.subgoal, index,
            )
            if option.source == "m" and self.mcp_current_page_only and candidates and (
                candidates[0].arguments.get("arguments", {}).get("href")
                != self._snapshot.browser.url
            ):
                continue
            if option.source == "t" and candidates and (
                candidates[0].capability in self._completed_capabilities
            ):
                continue
            result.extend(candidates)
        return tuple(result)

    def executor_bindings(self) -> dict[Channel, Any]:
        return {channel: self for channel in (
            Channel.API, Channel.CLI, Channel.SCRIPT, Channel.GUI, Channel.MCP,
        )}

    def register_verifiers(self, registry: VerifierRegistry) -> None:
        for provider in self.providers.values():
            provider.register_verifiers(registry)

    def __call__(
        self, candidate: ActionCandidate, observation_id: str, decision_id: str
    ) -> ActionReceipt:
        matches = [provider for provider in self.providers.values() if provider.owns(candidate)]
        if len(matches) != 1:
            raise ValueError("selected capability has no unique surface provider")
        return matches[0].execute(candidate, observation_id, decision_id)

    def evaluate(
        self,
        observation: Observation,
        candidate: ActionCandidate,
        receipt: ActionReceipt,
        verification: Verification,
    ) -> Evaluation:
        if candidate.capability == "control.done":
            passed = self._satisfied(self._capture())
            return Evaluation(passed, True, "acceptance_passed" if passed else "acceptance_failed")
        if not verification.passed:
            self._plan = None
            return Evaluation(False, False, "action_unverified_replan")
        index = int(candidate.intent.removeprefix("option_"))
        self._used.add(index)
        self._completed_capabilities.add(candidate.capability)
        if candidate.capability == "mcp.call_tool":
            server = candidate.arguments["server"]
            tool = candidate.arguments["tool"]
            self._completed_capabilities.add(f"mcp.{server}.{tool}")
            if self.mcp_current_page_only:
                url = candidate.arguments["arguments"]["href"]
                key = _page_key(url)
                self._mcp_pages_read.add(key)
                if self.min_verified_sources and (
                    key != _page_key(self.browser.start_url)
                ):
                    source = self._snapshot.browser if self._snapshot is not None else None
                    if source is not None and source.url == url:
                        self._discovered_sources.setdefault(key, {
                            "clue": "model_discovered", "url": url,
                            "title": source.title, "excerpt": source.text[:500],
                        })
        if self._plan is not None and self._plan.options[index].source in {"t", "m"}:
            output = receipt.output
            summary = output.get(
                "structured_content", output.get(
                    "text", output.get("stdout", output.get("entries", ""))
                )
            )
            self._tool_results.append({
                "capability": candidate.capability, "result": str(summary)[:2000],
                "verified": True,
            })
        if self._satisfied(self._capture()):
            return Evaluation(True, True, "acceptance_passed")
        return Evaluation(False, False, "action_verified")
