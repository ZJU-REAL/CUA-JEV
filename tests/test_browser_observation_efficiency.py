"""Real browser tests: typed edits save screenshots, pointer checks stay full."""

import pytest
from test_agent_browser import browser, page  # noqa: F401

from cua_jev.agent_browser import BrowserAgentBackend
from cua_jev.agent_types import AgentStep


@pytest.mark.parametrize("efficient,route,expected_full", [
    (False, "dom", 3), (True, "dom", 2), (True, "mouse", 3),
])
def test_typed_input_uses_fresh_dom_without_redundant_screenshot(page, efficient, route, expected_full):  # noqa: F811
    backend = BrowserAgentBackend(page, allowed_origins=["https://fixture.test"],
                                  efficient_observations=efficient)
    try:
        page.set_content('<label>Name<input aria-label="Name"></label>')
        before = backend.observe()
        target = next(c for c in before.controls if c.label == "Name")
        actions = backend.compile(AgentStep("name", "fill", ref=target.ref, value="Ada"), before, before)
        chosen = next(a for a in actions if a.arguments["route"] == route)
        receipt = backend.execute(chosen, "obs", "decision")
        assert receipt.success and backend.verify(chosen, receipt).passed
        assert page.locator("input").input_value() == "Ada"
        assert backend.observation_metrics["full"] == expected_full
        assert backend.observation_metrics["structured_only"] == int(efficient and route == "dom")
    finally:
        backend.close()


def test_cheap_dom_preflight_rejects_changed_value(page):  # noqa: F811
    backend = BrowserAgentBackend(page, allowed_origins=["https://fixture.test"],
                                  efficient_observations=True)
    try:
        page.set_content('<input aria-label="Name">')
        before = backend.observe()
        action = next(a for a in backend.compile(
            AgentStep("name", "fill", ref=before.controls[0].ref, value="Ada"), before, before,
        ) if a.arguments["route"] == "dom")
        page.locator("input").fill("Changed externally")
        receipt = backend.execute(action, "obs", "decision")
        assert not receipt.success
        assert page.locator("input").input_value() == "Changed externally"
        assert backend.observation_metrics["structured_only"] == 1
    finally:
        backend.close()
