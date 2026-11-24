"""Real browser reuse saves full captures while preserving fresh input checks."""

import time
from dataclasses import replace

import pytest
from test_agent_browser import browser, page  # noqa: F401
from test_agent_runner import Planner

from cua_jev.agent_browser import BrowserAgentBackend
from cua_jev.agent_runner import AgentConfig, AgentRunner, DirectPlanPolicy
from cua_jev.agent_types import AgentStep


def setup(browser_page):
    browser_page.set_content('<input aria-label="Name"><input aria-label="City">')
    return BrowserAgentBackend(browser_page, allowed_origins=["https://fixture.test"],
                               pointer_routes=False, efficient_observations=True)


def execute(backend):
    snapshot = backend.observe()
    action = backend.compile(AgentStep("name", "fill", ref=snapshot.controls[0].ref, value="Ada"),
                             snapshot, snapshot)[0]
    receipt = backend.execute(action, "obs", "decision")
    assert receipt.success and backend.verify(action, receipt).passed
    return action, receipt


def test_post_snapshot_is_once_only_and_does_not_replace_preflight(page):  # noqa: F811
    backend = setup(page)
    try:
        action, receipt = execute(backend)
        snapshot, _ = backend.take_verified_post_snapshot(action, receipt)
        assert snapshot.controls[0].value == "Ada"
        assert backend.take_verified_post_snapshot(action, receipt) is None
        next_action = backend.compile(
            AgentStep("city", "fill", ref=snapshot.controls[1].ref, value="Hangzhou"),
            snapshot, snapshot,
        )[0]
        page.locator('input[aria-label="City"]').fill("Changed externally")
        second = backend.execute(next_action, "obs2", "decision2")
        assert not second.success
        assert page.locator('input[aria-label="City"]').input_value() == "Changed externally"
        assert backend.observation_metrics["structured_only"] == 2
    finally:
        backend.close()


@pytest.mark.parametrize("mutation", ["receipt_copy", "receipt_output", "candidate", "observe",
                                      "failed_observe", "execute", "close", "expired", "future"])
def test_reuse_rejects_forgery_and_intervening_reads_or_actions(page, monkeypatch, mutation):  # noqa: F811
    backend = setup(page)
    try:
        action, receipt = execute(backend)
        if mutation == "receipt_copy":
            receipt = replace(receipt)
        elif mutation == "receipt_output":
            receipt.output["effects"].append("forged")
        elif mutation == "candidate":
            action = replace(action, arguments={**action.arguments, "value": "forged"})
        elif mutation == "observe":
            backend.observe()
        elif mutation == "failed_observe":
            def fail(**kwargs):
                raise RuntimeError("capture failed")
            monkeypatch.setattr(backend, "_capture_snapshot", fail)
            with pytest.raises(RuntimeError):
                backend.observe()
        elif mutation == "execute":
            assert not backend.execute(action, "replay", "replay").success
        elif mutation == "close":
            backend.close()
        else:
            now = time.perf_counter()
            monkeypatch.setattr("cua_jev.agent_browser.time.perf_counter",
                                lambda: now + (2 if mutation == "expired" else -2))
        assert backend.take_verified_post_snapshot(action, receipt) is None
    finally:
        backend.close()


@pytest.mark.parametrize("reuse,live,expected_reuses", [(False, True, 0), (True, False, 0),
                                                      (True, True, 2)])
def test_runner_live_evaluator_reuses_capture_and_keeps_all_input_checks(page, reuse, live, expected_reuses):  # noqa: F811
    backend = setup(page)
    checks = []
    def accepted(_snapshot):
        values = page.locator("input").evaluate_all("els => els.map(el => el.value)")
        checks.append(values)
        return values == ["Ada", "Hangzhou"]
    try:
        result = AgentRunner(backend, Planner(), policy=DirectPlanPolicy(),
                             config=AgentConfig(allow_actions=True, adaptive_dispatch=True,
                                                reuse_verified_post_snapshot=reuse)).run(
            "Fill form", success=accepted, success_uses_live_state=live,
        )
        assert result.success and result.steps == 2
        assert result.post_snapshot_reuses == expected_reuses
        assert backend.observation_metrics["structured_only"] == 2
        assert checks == [["", ""], ["Ada", ""], ["Ada", "Hangzhou"]]
        assert backend.observation_metrics["full"] == 6 - expected_reuses
    finally:
        backend.close()
