"""Real browser integration with test-only scripted planners, never model APIs.

These planners exercise the runner/backend contracts; they are not used by the
agent or the smoke CLI and are not evidence of model reasoning performance.
"""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from PIL import Image

from cua_jev.agent_browser import BrowserAgentBackend
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentPlan, AgentStep
from cua_jev.trace import JsonlTrace


@pytest.fixture(scope="module")
def browser():
    api = pytest.importorskip("playwright.sync_api")
    with api.sync_playwright() as playwright:
        browser = None
        for options in ({}, {"channel": "msedge"}):
            try:
                browser = playwright.chromium.launch(headless=True, **options)
                break
            except api.Error:
                continue
        if browser is None:
            pytest.skip("No local Chromium or Edge browser is installed")
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture
def page(browser):
    context = browser.new_context(viewport={"width": 1100, "height": 760})
    try:
        yield context.new_page()
    finally:
        context.close()


@pytest.fixture
def site():
    path = Path(__file__).resolve().parents[1] / "scripts" / "agent_smoke_fixture.py"
    spec = importlib.util.spec_from_file_location("agent_integration_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with module.SmokeSite() as local_site:
        yield local_site


def _control(snapshot, label):
    matches = [control for control in snapshot.controls if control.label == label]
    assert len(matches) == 1
    return matches[0]


def _events(trace, kind):
    return [event["payload"] for event in trace.events if event["kind"] == kind]


class _ScriptedPlanner:
    """Test double that records actual observations and supplies explicit test plans."""

    def __init__(self):
        self.snapshots = []
        self.histories = []
        self.horizons = []

    def plan(self, goal, snapshot, history, *, max_actions=6):
        assert goal
        assert snapshot.image is not None
        self.snapshots.append(snapshot)
        self.histories.append(list(history))
        self.horizons.append(max_actions)
        return self._plan(snapshot)


class _FormPlanner(_ScriptedPlanner):
    def _plan(self, snapshot):
        steps = []
        subject = _control(snapshot, "Subject")
        priority = _control(snapshot, "Priority")
        follow_up = _control(snapshot, "Request a follow-up")
        if subject.value != "Printer is offline":
            steps.append(AgentStep("subject", "fill", ref=subject.ref, value="Printer is offline"))
        if priority.value != "high":
            steps.append(AgentStep("priority", "select", ref=priority.ref, value="High"))
        if not follow_up.checked:
            steps.append(AgentStep("follow-up", "check", ref=follow_up.ref))
        steps.append(
            AgentStep(
                "save-ticket", "click", ref=_control(snapshot, "Save ticket").ref,
                depends_on=tuple(step.id for step in steps),
            )
        )
        return AgentPlan("Test-only form plan with independent edits and dependent save", tuple(steps))


class _SearchPlanner(_ScriptedPlanner):
    def _plan(self, snapshot):
        if urlsplit(snapshot.url).path == "/search":
            query = _control(snapshot, "Search products")
            return AgentPlan("Test-only search plan", (
                AgentStep("search-query", "fill", ref=query.ref, value="Aurora"),
                AgentStep("submit-search", "press", ref=query.ref, value="Enter",
                          depends_on=("search-query",)),
                # This old-page action must be discarded after the URL changes.
                AgentStep("obsolete-search", "click", ref=_control(snapshot, "Search").ref,
                          depends_on=("submit-search",)),
            ))
        assert urlsplit(snapshot.url).path == "/search/results"
        target = _control(snapshot, "Add Aurora Desk Lamp to shortlist")
        return AgentPlan("Test-only result selection", (
            AgentStep("shortlist-target", "click", ref=target.ref),
        ))


def _colored_box(snapshot, color):
    """Deterministic test-only pixel grounder; reads no canvas JS, DOM, or storage."""
    with Image.open(io.BytesIO(snapshot.image.jpeg)) as image:
        image = image.convert("RGB")
        width, height = image.size
        pixels = image.load()
        points = []
        for y in range(height):
            for x in range(width):
                red, green, blue = pixels[x, y]
                matches = (
                    blue > 150 and blue > red * 1.8 and blue > green * 1.5
                    if color == "blue"
                    else green > 90 and green > red * 2 and green > blue * 1.5
                )
                if matches:
                    points.append((x, y))
        assert points, f"Expected a visible {color} shape in the actual screenshot"
        xs, ys = zip(*points, strict=True)
        return (
            round(min(xs) / width * 1000), round(min(ys) / height * 1000),
            round((max(xs) + 1) / width * 1000), round((max(ys) + 1) / height * 1000),
        )


class _CanvasPlanner(_ScriptedPlanner):
    def _plan(self, snapshot):
        assert not snapshot.controls
        if len(self.snapshots) == 1:
            box = _colored_box(snapshot, "blue")
            return AgentPlan("Test-only first visual plan", (
                AgentStep("blue-circle", "visual_click", box=box, label="Blue circle"),
                # The runner must never reuse this old screenshot's box on the next board.
                AgentStep("obsolete-box", "visual_click", box=box, label="Old board target",
                          depends_on=("blue-circle",)),
            ))
        assert len(self.snapshots) == 2
        return AgentPlan("Test-only second visual plan from fresh pixels", (
            AgentStep("green-square", "visual_click", box=_colored_box(snapshot, "green"),
                      label="Green square"),
        ))


def _run(site, page, task, planner, *, max_steps, progress=None):
    url = site.url(task)
    backend = BrowserAgentBackend(page, allowed_origins=[url])
    trace = JsonlTrace()
    try:
        page.goto(url, wait_until="domcontentloaded")
        result = AgentRunner(
            backend, planner, policy=DirectPlanPolicy(), trace=trace, progress=progress,
            config=AgentConfig(mode="hybrid", allow_actions=True, max_steps=max_steps, max_plans=4),
        ).run(site.goal(task), success=lambda snapshot: site.verify(task, page))
        assert result.success, (result.to_dict(), _events(trace, "receipt"))
        assert site.verify(task, page)
        assert result.model_usage == {}
        assert all(event["passed"] for event in _events(trace, "verification"))
        return result, trace, backend.observe()
    finally:
        backend.close()


def test_one_form_plan_runs_four_actions_rebinds_refs_and_defers_save(site, page):
    planner = _FormPlanner()
    result, trace, final = _run(site, page, "form", planner, max_steps=4)
    assert result.steps == 4 and result.jev_calls == 3 and result.singleton_decisions == 1
    assert result.planner_calls == len(planner.snapshots) == 1
    assert result.plan_reuses == 3
    assert result.replan_reasons == {"initial": 1}
    assert result.routes == {"dom": 4}
    assert _control(planner.snapshots[0], "Priority").ref != _control(final, "Priority").ref
    offered = _events(trace, "candidates")
    assert all(not any(item["intent"] == "save-ticket" for item in event["items"]) for event in offered[:3])
    assert {item["intent"] for item in offered[3]["items"]} == {"save-ticket"}
    priority = next(item for item in offered[1]["items"] if item["intent"] == "priority")
    assert priority["arguments"]["ref"] == _control(final, "Priority").ref
    assert [event["candidate"] for event in _events(trace, "agent_step")] == [
        "subject:dom", "priority:dom", "follow-up:dom", "save-ticket:dom",
    ]


def test_new_visible_feedback_invalidates_remaining_form_plan(site, page):
    planner = _FormPlanner()

    def add_feedback(event):
        if event["event"] == "agent_step" and event["step"] == 1:
            page.get_by_role("status").evaluate(
                "node => node.textContent = 'Please review the remaining ticket fields.'"
            )

    result, trace, _ = _run(site, page, "form", planner, max_steps=4, progress=add_feedback)
    assert result.steps == 4
    assert result.planner_calls == len(planner.snapshots) == 2
    assert result.replan_reasons["page_or_controls_changed"] == 1
    assert "Please review the remaining ticket fields." in planner.snapshots[1].text
    assert [len(event["steps"]) for event in _events(trace, "agent_plan")] == [4, 3]


def test_search_navigation_discards_old_plan_and_uses_new_result_refs(site, page):
    planner = _SearchPlanner()
    result, trace, _ = _run(site, page, "search", planner, max_steps=3)
    assert result.steps == 3 and result.jev_calls == 2 and result.singleton_decisions == 1
    assert result.planner_calls == len(planner.snapshots) == 2
    assert result.plan_reuses == 1
    # Enter requires a new observation/plan even when navigation is delayed.
    assert result.replan_reasons == {"initial": 1, "visual_or_page_action": 1}
    assert [urlsplit(snapshot.url).path for snapshot in planner.snapshots] == ["/search", "/search/results"]
    assert [event["candidate"] for event in _events(trace, "agent_step")] == [
        "search-query:dom", "submit-search:dom", "shortlist-target:dom",
    ]
    assert "navigation" in _events(trace, "receipt")[1]["output"]["effects"]


def test_each_canvas_action_requires_new_pixels_and_independent_acceptance(site, page):
    planner = _CanvasPlanner()
    result, trace, _ = _run(site, page, "canvas", planner, max_steps=2)
    assert result.steps == result.singleton_decisions == 2 and result.jev_calls == 0
    assert result.planner_calls == len(planner.snapshots) == 2
    assert result.plan_reuses == 0
    assert result.replan_reasons == {"initial": 1, "visual_or_page_action": 1}
    assert result.routes == {"visual": 2}
    assert planner.snapshots[0].image.jpeg != planner.snapshots[1].image.jpeg
    assert planner.snapshots[0].text == planner.snapshots[1].text
    assert [event["candidate"] for event in _events(trace, "agent_step")] == [
        "blue-circle:visual", "green-square:visual",
    ]
    assert all("image" in event["output"]["effects"] for event in _events(trace, "receipt"))


def test_field_reset_replans_and_repairs_completed_dependency_before_submit(site, page):
    # A real browser runs this test-only form. Editing B resets A without adding
    # text or changing control identities, so structural invalidation cannot help.
    page.goto(site.url("form"))
    page.set_content("""
        <!doctype html><html lang="en"><title>Dependent fields</title>
        <style>body {font: 20px Arial; padding: 30px;}
          label {display: block; margin: 20px 0;} input, button {font: inherit; padding: 10px;}</style>
        <form>
          <label>Field A <input id="field-a"></label>
          <label>Field B <input id="field-b"></label>
          <button type="submit">Save both fields</button>
        </form><p role="status"></p>
        <script>
          (() => {
            const key = 'agent.integration.submissions';
            localStorage.removeItem(key);
            const a = document.querySelector('#field-a');
            const b = document.querySelector('#field-b');
            b.addEventListener('input', () => { a.value = ''; });
            document.querySelector('form').addEventListener('submit', event => {
              event.preventDefault();
              const submissions = JSON.parse(localStorage.getItem(key) || '[]');
              submissions.push({a: a.value, b: b.value});
              localStorage.setItem(key, JSON.stringify(submissions));
              document.querySelector('[role=status]').textContent = 'Saved.';
            });
          })();
        </script></html>
    """)

    class ResettingFormPlanner(_ScriptedPlanner):
        def _plan(self, snapshot):
            steps = []
            for label, step_id, value in (("Field A", "field-a", "Alpha"), ("Field B", "field-b", "Beta")):
                control = _control(snapshot, label)
                if control.value != value:
                    steps.append(AgentStep(step_id, "fill", ref=control.ref, value=value))
            steps.append(AgentStep(
                "save-fields", "click", ref=_control(snapshot, "Save both fields").ref,
                depends_on=tuple(step.id for step in steps),
            ))
            return AgentPlan("Test-only dependent field repair", tuple(steps))

    def accepted(_snapshot):
        # Check every actual submission, not the success message or only the final
        # field values. An early incorrect submit must never be hidden by a later save.
        return page.evaluate(
            "JSON.parse(localStorage.getItem('agent.integration.submissions') || '[]')"
        ) == [{"a": "Alpha", "b": "Beta"}]

    planner = ResettingFormPlanner()
    backend = BrowserAgentBackend(page)
    trace = JsonlTrace()
    try:
        result = AgentRunner(
            backend, planner, policy=DirectPlanPolicy(), trace=trace,
            config=AgentConfig(allow_actions=True, max_steps=4, max_plans=3),
        ).run('Save Field A as "Alpha" and Field B as "Beta".', success=accepted)
        assert result.success, (result.to_dict(), _events(trace, "receipt"))
        assert accepted(None)
        assert result.steps == 4 and result.planner_calls == 2
        assert result.replan_reasons == {"initial": 1, "completed_state_changed": 1}
        assert planner.snapshots[0].scope_key() == planner.snapshots[1].scope_key()
        assert _control(planner.snapshots[1], "Field A").value == ""
        assert _control(planner.snapshots[1], "Field B").value == "Beta"
        assert [event["candidate"] for event in _events(trace, "agent_step")] == [
            "field-a:dom", "field-b:dom", "field-a:dom", "save-fields:dom",
        ]
        offered = _events(trace, "candidates")
        assert all(
            not any(item["intent"] == "save-fields" for item in event["items"]) for event in offered[:3]
        )
        assert {item["intent"] for item in offered[3]["items"]} == {"save-fields"}
        assert all(event["passed"] for event in _events(trace, "verification"))
    finally:
        backend.close()
