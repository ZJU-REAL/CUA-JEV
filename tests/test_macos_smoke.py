"""Offline smoke-runner contracts: no GUI, real model endpoint, or local secrets."""

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from test_macos_surface import NativeFixture, surface

from cua_jev.errors import CapabilityUnavailable
from cua_jev.model_planner import ChatModelPlanner
from cua_jev.models import ActionCandidate, Channel, Observation
from cua_jev.policy import RulePolicy


@pytest.fixture
def smoke():
    source = Path(__file__).resolve().parents[1] / "scripts" / "macos_smoke.py"
    spec = importlib.util.spec_from_file_location("macos_smoke", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixturePage:
    main_frame = object()

    def __init__(self):
        self.url = "https://cua-jev.test/start"

    def route(self, *_args):
        pass

    def goto(self, url, **_kwargs):
        self.url = url

    def evaluate(self, *_args):
        return {
            "url": self.url, "title": "Fixture", "text": "Result verified" if "/done" in self.url else "",
            "elements": [{
                "ref": "e0", "index": 0, "tag": "a", "role": "", "label": "Finish",
                "input_type": "", "disabled": False, "href": "https://cua-jev.test/done",
            }],
        }

    def locator(self, _selector):
        return self

    def nth(self, _index):
        return self

    def click(self):
        self.url = "https://cua-jev.test/done"


def model_planner(note, *, never_save=False):
    def response(request):
        # The actual adapter receives only current observations and the caller's
        # goal. A local mock supplies a model reply; it never touches the network.
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        state = payload["snapshot"]
        completed = state["requirements"]["completed_capabilities"]
        assert "old private text" not in str(payload)
        if "cli.python_version" not in completed:
            option = {"ref": next(item["ref"] for item in state["tool_offers"]
                                   if item["capability"] == "cli.python_version"), "operation": "invoke"}
        elif "/done" not in state["browser"]["url"]:
            option = {"ref": state["browser"]["elements"][0]["ref"], "operation": "click"}
        elif "desktop.fill" not in completed or never_save:
            assert note in payload["goal"]
            option = {"ref": "d:c0", "operation": "fill", "value": note}
        else:
            option = {"ref": "d:c1", "operation": "click"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "subgoal": "Complete fixture requirements", "options": [option],
                "success": {"kind": "text_contains", "value": "I am done"},
            })}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        })

    return ChatModelPlanner(
        "https://mock.invalid/v1", "offline-test-model",
        client=httpx.Client(transport=httpx.MockTransport(response)),
    )


@pytest.mark.parametrize("desktop, expected_steps", [(False, 2), (True, 4)])
def test_real_adapter_mode_reports_usage_and_independent_terminal_gates(
    smoke, tmp_path, desktop, expected_steps,
):
    note = "Synthetic note 中文"
    planner = model_planner(note)
    native = surface() if desktop else None
    task = smoke._make_task(FixturePage(), planner, tmp_path, note, native)
    trace_path = tmp_path / "trace.jsonl"
    summary = smoke._execute_task(
        task, planner, trace_path, route="auto", policy="rule", model_enabled=True,
        max_steps=8, timeout_s=10,
    )
    assert summary["status"] == "success", summary["reason"]
    assert summary["terminal_verified"] and summary["all_actions_verified"]
    assert summary["mode"] == "model" and summary["policy"] == "rule"
    assert summary["forced_desktop_route"] is None
    assert summary["planner_requests"] == expected_steps
    assert summary["planner_model_usage"]["total_tokens"] == expected_steps * 30
    assert summary["jev_decisions"] == 0
    assert summary["actual_desktop_routes"] == ({"ax": 2} if desktop else {})
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    usage = next(event["payload"] for event in events if event["kind"] == "model_usage")
    assert usage["requests"] == expected_steps


def test_model_success_claim_cannot_replace_saved_note_gate(smoke, tmp_path):
    planner = model_planner("Must save this note", never_save=True)
    bridge = NativeFixture()
    task = smoke._make_task(FixturePage(), planner, tmp_path, "Must save this note", surface(bridge))
    summary = smoke._execute_task(
        task, planner, tmp_path / "failure.jsonl", route="auto", policy="rule", model_enabled=True,
        max_steps=4, timeout_s=10,
    )
    assert summary["status"] == "max_steps"
    assert not summary["terminal_verified"]
    assert bridge.state["text"] == "No note saved"


def test_model_jev_mode_preserves_both_desktop_routes(smoke, tmp_path, monkeypatch):
    offered = []
    closed = []

    class OfflineJev(RulePolicy):
        def __init__(self, **kwargs):
            assert kwargs["fallback_on_transport"] is False
            self.last_exchange = {"usage": {"total_tokens": 5}}

        def choose(self, observation, candidates):
            desktop = [item for item in candidates if item.capability.startswith("desktop.")]
            selected = next((item for item in desktop if item.channel == Channel.GUI), candidates[0])
            if desktop:
                offered.append({item.channel for item in desktop})
            return replace(
                super().choose(observation, candidates), model="jev-offline-mock",
                candidate_id=selected.id,
                probabilities={item.id: float(item.id == selected.id) for item in candidates},
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(smoke, "JevPolicy", OfflineJev)
    planner = model_planner("Model chooses refs")
    task = smoke._make_task(FixturePage(), planner, tmp_path, "Model chooses refs", surface())
    summary = smoke._execute_task(
        task, planner, tmp_path / "trace.jsonl", route="auto", policy="jev", model_enabled=True,
        max_steps=8, timeout_s=10,
    )
    assert summary["status"] == "success", summary["reason"]
    assert summary["jev_decisions"] == 4
    assert summary["policy_model_usage"] == {"total_tokens": 20}
    assert summary["actual_desktop_routes"] == {"gui": 2}
    assert summary["forced_desktop_route"] is None
    assert offered == [{Channel.API, Channel.GUI}, {Channel.API, Channel.GUI}]
    assert closed == [True]


def test_deterministic_fixture_remains_keyless_and_is_labeled(smoke, tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "ChatModelPlanner", lambda *_args, **_kwargs: pytest.fail("unexpected model"))
    planner = smoke.FixturePlanner("Fixture note")
    task = smoke._make_task(FixturePage(), planner, tmp_path, "Fixture note", surface())
    summary = smoke._execute_task(
        task, planner, tmp_path / "trace.jsonl", route="ax", policy="rule", model_enabled=False,
        max_steps=8, timeout_s=10,
    )
    assert summary["status"] == "success"
    assert summary["mode"] == "deterministic"
    assert summary["forced_desktop_route"] == "ax"
    assert summary["planner_requests"] == 0 and summary["planner_model_usage"] is None
    assert summary["jev_decisions"] == 0


def test_unavailable_forced_route_has_an_explicit_non_jev_error(smoke):
    with pytest.raises(CapabilityUnavailable, match="Forced GUI test route is unavailable"):
        smoke.RouteTestPolicy("gui").choose(
            Observation("test", "click", {}),
            [ActionCandidate("ax", Channel.API, "desktop.click", "Invoke native button")],
        )


@pytest.mark.parametrize("arguments", [
    ["--model", "test"], ["--model-base-url", "https://mock.invalid/v1"],
    ["--model-base-url", "https://mock.invalid/v1", "--model", "test", "--route", "gui"],
    ["--policy", "jev"], ["--browser-only", "--record"], ["--max-steps", "0"], ["--timeout-s", "nan"],
])
def test_invalid_modes_fail_before_loading_env_or_starting_resources(smoke, arguments, monkeypatch):
    monkeypatch.setattr(smoke, "load_local_env", lambda: pytest.fail("must validate arguments first"))
    with pytest.raises(SystemExit):
        smoke.main(arguments)


def test_model_arguments_load_local_config_and_forward_auto_mode(smoke, monkeypatch, capsys):
    loaded = []
    calls = []
    monkeypatch.setattr(smoke.sys, "platform", "darwin")
    monkeypatch.setattr(smoke, "load_local_env", lambda: loaded.append(True))
    monkeypatch.setattr(smoke, "MacBridge", lambda: pytest.fail("headless browser mode has no native helper"))

    def run(output, route, **kwargs):
        calls.append((route, kwargs))
        return {"status": "success"}

    monkeypatch.setattr(smoke, "run", run)
    assert smoke.main([
        "--browser-only", "--model-base-url", "https://mock.invalid/v1", "--model", "test",
        "--policy", "jev", "--max-steps", "5", "--timeout-s", "90",
    ]) == 0
    assert loaded == [True]
    assert calls[0][0] == "auto"
    assert calls[0][1]["policy"] == "jev" and calls[0][1]["model"] == "test"
    assert calls[0][1]["desktop"] is False and calls[0][1]["record"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "success"


def test_locked_native_session_stops_before_fixture_or_api(smoke, monkeypatch):
    closed = []
    monkeypatch.setattr(smoke.sys, "platform", "darwin")
    monkeypatch.setattr(smoke, "load_local_env", lambda: None)
    monkeypatch.setattr(smoke, "MacBridge", lambda: SimpleNamespace(
        call=lambda _: {"screen_locked": True, "accessibility": True, "screen_recording": True},
        close=lambda: closed.append(True),
    ))
    monkeypatch.setattr(smoke, "run", lambda *a, **k: pytest.fail("must not call the model or Jev"))
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch GUI"))
    arguments = ["--model-base-url", "https://mock.invalid/v1", "--model", "test", "--policy", "jev"]
    with pytest.raises(RuntimeError, match="screen is locked"):
        smoke.main(arguments)
    assert closed == [True]


def test_headless_browser_only_never_initializes_native_helper(smoke, monkeypatch):
    monkeypatch.setattr(smoke.sys, "platform", "darwin")
    monkeypatch.setattr(smoke, "load_local_env", lambda: None)
    monkeypatch.setattr(smoke, "MacBridge", lambda: pytest.fail("headless browser mode has no native helper"))
    monkeypatch.setattr(smoke, "native_binary", lambda: pytest.fail("must not require Swift"))
    monkeypatch.setattr(smoke, "run", lambda *a, **k: {"status": "success"})
    assert smoke.main(["--browser-only"]) == 0


def test_recording_run_finalizes_native_and_closes_owned_browser(smoke, tmp_path, monkeypatch):
    bridge = NativeFixture()
    closed = []
    captured = []

    class Browser:
        def new_page(self):
            return FixturePage()

        def close(self):
            closed.append("browser")

    class Playwright:
        def __enter__(self):
            return SimpleNamespace(chromium=SimpleNamespace(launch=lambda **_kwargs: Browser()))

        def __exit__(self, *_args):
            closed.append("playwright")

    def native_surface(**kwargs):
        captured.append(kwargs)
        return surface(bridge, record_path=kwargs["record_path"])

    sync_api = ModuleType("playwright.sync_api")
    sync_api.sync_playwright = Playwright
    playwright = ModuleType("playwright")
    playwright.sync_api = sync_api
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.setattr(smoke, "MacAccessibilitySurface", native_surface)
    summary = smoke.run(tmp_path, "ax", desktop=True, record=True)
    assert summary["status"] == "success", summary["reason"]
    assert summary["recording"] == str(captured[0]["record_path"])
    assert summary["recording_scope"] == "selected native fixture window only"
    commands = [command for command, _ in bridge.commands]
    assert commands[:2] == ["select", "record_start"] and commands[-1] == "record_stop"
    assert bridge.closed and closed == ["browser", "playwright"]
    assert len(list(tmp_path.glob("*-workspace"))) == 1
