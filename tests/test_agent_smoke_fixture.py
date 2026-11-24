from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest


def _fixture_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "agent_smoke_fixture.py"
    spec = importlib.util.spec_from_file_location("agent_smoke_fixture", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def smoke_site():
    with _fixture_module().SmokeSite() as site:
        yield site


@pytest.fixture
def page():
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
        context = browser.new_context(viewport={"width": 1100, "height": 760})
        try:
            yield context.new_page()
        finally:
            context.close()
            browser.close()


def test_site_lifecycle_routes_and_natural_language_goals():
    module = _fixture_module()
    assert module.TASKS == ("form", "search", "canvas")
    site = module.SmokeSite()
    with pytest.raises(RuntimeError):
        site.url("form")
    with site:
        thread = site._thread
        assert thread.is_alive()
        for task in module.TASKS:
            url = site.url(task)
            assert url.startswith("http://127.0.0.1:")
            assert url.endswith(f"/{task}?reset=1")
            with urlopen(url, timeout=3) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert "default-src 'none'" in response.headers["Content-Security-Policy"]
                assert '<html lang="en">' in response.read().decode()
            assert site.goal(task)
        with pytest.raises(HTTPError, match="404"):
            urlopen(site.url("form").replace("/form?reset=1", "/missing"), timeout=3)
        with pytest.raises(ValueError):
            site.url("missing")
        with pytest.raises(RuntimeError, match="already running"):
            site.__enter__()
    assert not thread.is_alive()
    with pytest.raises(RuntimeError):
        site.url("form")


def test_form_reorders_existing_controls_and_verifies_saved_values(smoke_site, page):
    page.goto(smoke_site.url("form"))
    original_ids = page.locator("button, input, select").evaluate_all("nodes => nodes.map(node => node.id)")
    assert not smoke_site.verify("form", page)
    page.get_by_label("Subject", exact=True).fill("Printer is offline")
    reordered_ids = page.locator("button, input, select").evaluate_all("nodes => nodes.map(node => node.id)")
    assert reordered_ids != original_ids
    assert sorted(reordered_ids) == sorted(original_ids)
    page.get_by_role("button", name="Save ticket", exact=True).click()
    assert page.get_by_role("status").inner_text() == "Ticket saved."
    assert not smoke_site.verify("form", page)  # A success message is insufficient.
    page.get_by_label("Priority", exact=True).select_option("high")
    page.get_by_label("Request a follow-up", exact=True).check()
    assert not smoke_site.verify("form", page)  # Edits have not been saved yet.
    page.get_by_role("button", name="Save ticket", exact=True).click()
    assert smoke_site.verify("form", page)
    assert page.locator("button, input, select").evaluate_all("nodes => nodes.map(node => node.id)") == (
        reordered_ids
    )
    page.get_by_role("status").evaluate("node => node.textContent = ''")
    assert smoke_site.verify("form", page)
    page.goto(smoke_site.url("form"))
    assert not smoke_site.verify("form", page)


def test_search_navigates_then_saves_the_exact_shortlist(smoke_site, page):
    page.goto(smoke_site.url("search"))
    page.get_by_label("Search products", exact=True).fill("Aurora")
    page.get_by_label("Search products", exact=True).press("Enter")
    page.wait_for_url("**/search/results?q=Aurora")
    assert page.locator("article").count() == 3
    assert not smoke_site.verify("search", page)
    page.get_by_role("button", name="Add Aurora Floor Lamp to shortlist", exact=True).click()
    assert page.get_by_role("status").inner_text() == "Shortlist saved."
    assert not smoke_site.verify("search", page)
    page.get_by_role("button", name="Add Aurora Desk Lamp to shortlist", exact=True).click()
    assert not smoke_site.verify("search", page)  # An unwanted product is still saved.
    page.goto(smoke_site.url("search"))
    page.get_by_label("Search products", exact=True).fill("Aurora Desk Lamp")
    page.get_by_label("Search products", exact=True).press("Enter")
    page.wait_for_url("**/search/results?**")
    page.get_by_role("button", name="Add Aurora Desk Lamp to shortlist", exact=True).click()
    assert smoke_site.verify("search", page)
    page.reload()
    assert smoke_site.verify("search", page)  # Results-page navigation preserves storage.
    page.goto(smoke_site.url("search"))
    assert not smoke_site.verify("search", page)


def _click_canvas(page, x, y):
    # Test-only pointer input checks the site; this helper is never given to the agent.
    canvas = page.locator("canvas")
    size = canvas.evaluate("node => ({width: node.clientWidth, height: node.clientHeight})")
    canvas.click(position={"x": x * size["width"] / 900, "y": y * size["height"] / 440})


def test_canvas_requires_two_real_pointer_choices_without_semantic_targets(smoke_site, page):
    page.goto(smoke_site.url("canvas"))
    assert page.locator("button, input, a, [role=button]").count() == 0
    assert page.locator("canvas").get_attribute("aria-label") == "Shape board"
    assert not smoke_site.verify("canvas", page)
    first_board = page.locator("canvas").screenshot()
    _click_canvas(page, 450, 230)  # Wrong first shape.
    assert not smoke_site.verify("canvas", page)
    _click_canvas(page, 170, 230)  # Blue circle, first board.
    assert not smoke_site.verify("canvas", page)
    assert first_board != page.locator("canvas").screenshot()
    _click_canvas(page, 170, 230)  # Same position now holds the wrong shape.
    assert not smoke_site.verify("canvas", page)
    _click_canvas(page, 730, 230)  # Green square, second board.
    assert smoke_site.verify("canvas", page)
    page.goto(smoke_site.url("canvas"))
    assert not smoke_site.verify("canvas", page)


@pytest.mark.parametrize("raw", [None, "broken json", "[]", "{}", '"Ticket saved."', "true"])
def test_verifier_rejects_absent_or_malformed_storage(smoke_site, page, raw):
    page.goto(smoke_site.url("form"))
    if raw is not None:
        page.evaluate("raw => localStorage.setItem('cua_jev.smoke.ticket', raw)", raw)
    assert not smoke_site.verify("form", page)


def test_verifier_rejects_wrong_boolean_type_and_different_origin(smoke_site, page):
    page.goto(smoke_site.url("form"))
    raw = json.dumps({"subject": "Printer is offline", "priority": "high", "followUp": 1})
    page.evaluate("raw => localStorage.setItem('cua_jev.smoke.ticket', raw)", raw)
    assert not smoke_site.verify("form", page)
    page.goto("about:blank")
    assert not smoke_site.verify("form", page)
