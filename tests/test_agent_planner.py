import base64
import copy
import io
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from cua_jev.agent_planner import (
    DESKTOP_HOTKEY_KEYS,
    DESKTOP_KEYS,
    DESKTOP_MODIFIERS,
    MultimodalPlanner,
    parse_plan,
)
from cua_jev.agent_runner import AgentConfig, AgentRunner
from cua_jev.agent_types import AgentCondition, AgentControl, AgentSnapshot
from cua_jev.desktop_semantics import condition_holds, executable_control, observable_condition_kinds
from cua_jev.trace import JsonlTrace
from cua_jev.vision import WindowImage


@pytest.fixture
def snapshot():
    return AgentSnapshot(
        "https://example.test/form", "Form", "Fill in the form", (
            AgentControl("e0", "Name", "input", input_type="text"),
            AgentControl("e1", "Description", "contenteditable"),
            AgentControl("e2", "Choice", "select", options=("a", "b")),
            AgentControl("e3", "Agree", "input", input_type="checkbox"),
            AgentControl("e4", "Submit", "button"),
            AgentControl("e5", "Password", "input", input_type="password"),
            AgentControl("e6", "Upload", "input", input_type="file"),
            AgentControl("e7", "Hidden", "input", input_type="hidden"),
            AgentControl("e8", "Disabled", "button", disabled=True),
        ), WindowImage(b"\xff\xd8image-only-secret", bytes(64 * 64), (0, 0, 800, 600)),
    )


@pytest.fixture
def desktop_snapshot(snapshot):
    output = io.BytesIO()
    Image.new("RGB", (1280, 720), "white").save(output, format="JPEG")
    return replace(snapshot, url="osworld://desktop", controls=(), image=WindowImage(
        output.getvalue(), bytes(64 * 64), (0, 0, 1920, 1080),
    ))


@pytest.fixture
def navigation_snapshot(desktop_snapshot):
    return replace(desktop_snapshot, controls=(AgentControl(
        "d0", "Menu", "button", role="menuitem", app="Example", window="Main",
        box=(100, 100, 300, 180),
    ),))


def plan(*steps, done=False):
    return {"summary": "Complete the form", "steps": list(steps), "done": done}


def step(**overrides):
    return {"id": "name", "operation": "fill", "ref": "e0", "value": "Ada", **overrides}


def semantic_step(**overrides):
    target = {"name": "Display name", "role": "entry", "app": "Example", "window": "Preferences"}
    return {
        "id": "s1", "operation": "fill", "target": target, "value": "Ada",
        "postconditions": [{"kind": "value_equals", "target": target, "value": "Ada"}],
        **overrides,
    }


def navigation_step(**overrides):
    return {
        "id": "navigate", "operation": "click", "navigation": True,
        "label": "Reach the preferences dialog", "target": {"app": "Example"},
        "postconditions": [{"kind": "exists", "target": {"name": "Display name"}}],
        **overrides,
    }


def response(content, **kwargs):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}], **kwargs,
    })


def planner_with(handler, **kwargs):
    return MultimodalPlanner(
        "https://model.example/v1", "vlm", api_key="private-test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_one_call_contains_screenshot_dom_and_dependency_prompt(snapshot):
    requests = []
    raw = plan(step(), step(id="description", ref="e1"), {
        "id": "submit", "operation": "click", "ref": "e4",
        "depends_on": ["name", "description"],
    })

    def handler(request):
        requests.append(request)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer private-test-key"
        body = json.loads(request.content)
        assert body["model"] == "vlm"
        assert "NOT alternative actions" in body["messages"][0]["content"]
        assert "external evaluator" in body["messages"][0]["content"]
        parts = body["messages"][1]["content"]
        state = json.loads(parts[0]["text"])
        assert state["snapshot"]["controls"][0]["ref"] == "e0"
        assert "image" not in state["snapshot"]
        assert "screenshot_geometry" not in state
        assert len(state["recent_actions"]) == 8
        prefix, encoded = parts[1]["image_url"]["url"].split(",", 1)
        assert prefix == "data:image/jpeg;base64"
        assert base64.b64decode(encoded) == snapshot.image.jpeg
        return response("```json\n" + json.dumps(raw) + "\n```", usage={
            "prompt_tokens": 40, "completion_tokens": 20, "bad": -1, "not_tokens": True,
        })

    planner = planner_with(handler)
    result = planner.plan("Submit the form", snapshot, [{"step": i} for i in range(10)])
    assert result.steps[2].depends_on == ("name", "description")
    assert not result.done
    assert len(requests) == planner.calls == 1
    assert planner.wall_ms > 0
    assert planner.wall_ms == planner.planning_wall_ms
    assert planner.usage_totals == {"prompt_tokens": 40, "completion_tokens": 20}
    assert planner.last_raw is None


@pytest.mark.parametrize("change", [
    {"ref": "missing"}, {"ref": "e5"}, {"ref": "e6"}, {"ref": "e7"}, {"ref": "e8"},
    {"selector": "#name"}, {"code": "print('bad')"}, {"url": "https://other.test"},
    {"operation": "navigate"}, {"operation": "eval"}, {"value": 1}, {"value": ""},
    {"value": "x" * 2001}, {"label": "x" * 241}, {"ref": 123}, {"id": "bad id"},
    {"depends_on": ["future"]}, {"depends_on": ["name"]}, {"depends_on": "earlier"},
    {"operation": "click", "value": "unexpected"}, {"ref": "e4"},
    {"box": [0, 0, 100, 100]}, {"operation": "select", "ref": "e0"},
    {"operation": "check", "ref": "e0", "value": ""},
    {"operation": "press", "value": "Control+L"},
    {"operation": "scroll", "ref": "", "value": "20000"},
    {"operation": "back", "value": ""},
    {"operation": "wait", "ref": "", "value": "60000"},
])
def test_parser_rejects_ungrounded_or_ill_typed_actions(snapshot, change):
    with pytest.raises(ValueError):
        parse_plan(plan(step(**change)), snapshot)


@pytest.mark.parametrize("raw", [
    None, [], {}, {"summary": "s", "steps": [], "done": 1},
    {"summary": "s", "steps": [], "done": False},
    {"summary": "", "steps": [], "done": True},
    {"summary": "x" * 1001, "steps": [], "done": True},
    {"summary": "s", "steps": [], "done": True, "code": "bad"},
])
def test_parser_rejects_invalid_root(snapshot, raw):
    with pytest.raises(ValueError):
        parse_plan(raw, snapshot)


def test_parser_accepts_editable_select_checkbox_and_page_actions(snapshot):
    raw = plan(
        step(ref="e1"), step(id="select", operation="select", ref="e2", value="a"),
        step(id="check", operation="check", ref="e3", value=""),
        step(id="uncheck", operation="uncheck", ref="e3", value="", depends_on=["check"]),
        step(id="submit", operation="press", ref="e0", value="Enter", depends_on=["name", "select"]),
        step(id="pagekey", operation="press", ref="", value="Tab"),
        step(id="scroll", operation="scroll", ref="", value="down"),
        step(id="back", operation="back", ref="", value=""),
        step(id="wait", operation="wait", ref="", value="500"),
    )
    assert len(parse_plan(raw, snapshot, max_actions=12).steps) == 9


def test_dependencies_are_acyclic_unique_and_only_backward(snapshot):
    first = step()
    with pytest.raises(ValueError, match="unique"):
        parse_plan(plan(first, copy.deepcopy(first)), snapshot)
    with pytest.raises(ValueError, match="preceding"):
        parse_plan(plan(first, step(id="second", depends_on=["name", "name"])), snapshot)
    with pytest.raises(ValueError, match="preceding"):
        parse_plan(plan(step(depends_on=["second"]), step(id="second")), snapshot)
    result = parse_plan(plan(first, step(id="second", depends_on=["name"])), snapshot)
    assert result.steps[1].depends_on == ("name",)


def test_integer_ids_normalize_without_weakening_dependencies(snapshot):
    result = parse_plan(plan(step(id=1), step(id=2, depends_on=[1])), snapshot)
    assert [item.id for item in result.steps] == ["1", "2"]
    assert result.steps[1].depends_on == ("1",)
    for raw in (
        plan(step(id=True)), plan(step(id=-1)), plan(step(id=1.5)),
        plan(step(id=1), step(id="1")),
        plan(step(id=1), step(id=2, depends_on=[True])),
        plan(step(id=1), step(id=2, depends_on=[1, "1"])),
        plan(step(id=1, depends_on=[2]), step(id=2)),
    ):
        with pytest.raises(ValueError):
            parse_plan(raw, snapshot)


def test_radio_can_be_checked_but_not_unchecked(snapshot):
    radio = AgentControl("radio", "Choice", "input", input_type="radio")
    snapshot = replace(snapshot, controls=(*snapshot.controls, radio))
    assert parse_plan(plan(step(operation="check", ref="radio", value="")), snapshot).steps
    with pytest.raises(ValueError, match="uncheck requires checkbox"):
        parse_plan(plan(step(operation="uncheck", ref="radio", value="")), snapshot)


def test_visual_actions_require_current_image_and_valid_box(snapshot):
    visual = {"id": "v1", "operation": "visual_fill", "value": "Ada",
              "box": [20, 30, 300, 200], "label": "Name"}
    assert parse_plan(plan(visual), snapshot).steps[0].box == (20, 30, 300, 200)
    with pytest.raises(ValueError, match="screenshot"):
        parse_plan(plan(visual), replace(snapshot, image=None))
    for change in (
        {"box": [0, 0, 1001, 100]}, {"box": [0, 0, True, 100]}, {"box": [0, 0, 100]},
        {"box": [5, 0, 1, 100]}, {"box": [0, 0, 1, 100]}, {"box": [0.0, 0, 10, 10]},
        {"label": ""}, {"ref": "e0"}, {"value": ""}, {"box": None},
    ):
        with pytest.raises(ValueError):
            parse_plan(plan({**visual, **change}), snapshot)


def test_null_optional_box_is_equivalent_to_absent_for_dom_actions(snapshot):
    # JSON producers may emit optional fields as null; it carries no coordinates.
    expected = parse_plan(plan(step()), snapshot)
    assert parse_plan(plan(step(box=None)), snapshot) == expected


def test_empty_done_is_advisory_and_action_limits_are_enforced(snapshot):
    assert parse_plan(plan(done=True), snapshot).done
    pending = parse_plan(plan(step(), done=True), snapshot)
    assert not pending.done
    assert pending.steps[0].operation == "fill"
    with pytest.raises(ValueError, match="current observation"):
        parse_plan(plan(step(ref="missing"), done=True), snapshot)
    with pytest.raises(ValueError, match="bounded"):
        parse_plan(plan(step(), step(id="second")), snapshot, max_actions=1)
    for limit in (0, -1, 13, True, 1.5):
        with pytest.raises(ValueError, match="max_actions"):
            parse_plan(plan(step()), snapshot, max_actions=limit)
    with pytest.raises(ValueError, match="bounded"):
        parse_plan(plan(*(step(id=f"s{i}") for i in range(13))), snapshot, max_actions=12)


def test_single_step_baseline_and_dom_only_payload(snapshot):
    def handler(request):
        body = json.loads(request.content)
        assert "at most 1 steps" in body["messages"][0]["content"]
        assert "ONE immediate next action" in body["messages"][0]["content"]
        assert len(body["messages"][1]["content"]) == 1
        return response(json.dumps(plan(step())))

    planner = planner_with(handler)
    assert len(planner.plan("Fill", replace(snapshot, image=None), [], max_actions=1).steps) == 1
    assert planner.calls == 1


def test_canvas_prompt_demands_visual_action_and_shows_complete_json(snapshot):
    def handler(request):
        instructions = json.loads(request.content)["messages"][0]["content"]
        assert "NO DOM controls" in instructions
        assert "click+box is invalid" in instructions
        assert '"id":"s1"' in instructions
        assert '"operation":"visual_click","ref":""' in instructions
        assert "ONE immediate visual action" in instructions
        assert "done MUST be false" in instructions
        return response(json.dumps(plan({
            "id": "s1", "operation": "visual_click", "ref": "", "label": "Continue",
            "box": [100, 200, 300, 280], "depends_on": [],
        })))

    planner = planner_with(handler)
    result = planner.plan("Continue", replace(snapshot, controls=()), [])
    assert result.steps[0].operation == "visual_click"


@pytest.mark.parametrize("content", [
    "Here is the plan: {}", "{} trailing", "```json\n{}\n``` trailing", "{}{}",
    '{"summary":"s","summary":"s","steps":[],"done":true}',
    "not json", [], None, pytest.param("x" * 64_001, id="oversized-json"),
])
def test_bad_model_content_is_rejected_and_attempt_is_measured(snapshot, content):
    planner = planner_with(lambda _: response(content, usage={"prompt_tokens": 3}))
    with pytest.raises(ValueError, match="invalid multimodal"):
        planner.plan("Fill", snapshot, [])
    assert planner.calls == 1
    assert planner.wall_ms > 0
    assert planner.usage_totals == {"prompt_tokens": 3}


@pytest.mark.parametrize("status", [401, 500])
def test_http_failure_does_not_expose_image_or_credentials(snapshot, status):
    planner = planner_with(lambda _: httpx.Response(status, text="private-test-key image-only-secret"))
    with pytest.raises(RuntimeError) as error:
        planner.plan("Fill", snapshot, [])
    assert "private-test-key" not in str(error.value)
    assert "image-only-secret" not in str(error.value)
    assert planner.last_validation_error == f"Model request failed (HTTPStatusError; HTTP {status})."
    assert planner.calls == 1
    assert planner.wall_ms > 0


def test_timeout_and_invalid_envelope_count_attempts(snapshot):
    def timeout(request):
        raise httpx.ReadTimeout("private-test-key", request=request)

    planner = planner_with(timeout)
    with pytest.raises(RuntimeError, match="ReadTimeout"):
        planner.plan("Fill", snapshot, [])
    assert planner.calls == 1
    assert planner.wall_ms > 0
    assert planner.last_validation_error == "Model request failed (ReadTimeout)."
    broken = planner_with(lambda _: httpx.Response(200, json={"choices": []}))
    with pytest.raises(ValueError, match="invalid multimodal"):
        broken.plan("Fill", snapshot, [])
    assert broken.calls == 1
    assert broken.wall_ms > 0


@pytest.mark.parametrize(("failure", "expected"), [
    ("http", "Model request failed (HTTPStatusError; HTTP 503)."),
    ("timeout", "Model request failed (ReadTimeout)."),
])
def test_request_failure_trace_is_specific_safe_and_cleared_on_next_call(snapshot, failure, expected):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts > 1:
            return response(json.dumps(plan(step())))
        secrets = f"private-response-body {request.url} {request.headers['Authorization']} image-only-secret"
        if failure == "timeout":
            raise httpx.ReadTimeout(secrets, request=request)
        return httpx.Response(503, text=secrets, headers={"X-Diagnostic": "private-response-header"})

    planner = planner_with(handler)
    trace = JsonlTrace()
    backend = SimpleNamespace(observe=lambda: snapshot, verify=lambda *_: pytest.fail("no action expected"))
    result = AgentRunner(
        backend, planner, config=AgentConfig(mode="vlm", max_failures=1), trace=trace,
    ).run("Fill the form")
    assert result.status == "planner_error" and result.planner_calls == 1
    recovery = next(event["payload"] for event in trace.events if event["kind"] == "agent_recovery")
    assert recovery["error_type"] == "RuntimeError"
    assert recovery["validation"] == planner.last_validation_error == expected
    serialized = json.dumps(trace.events)
    for secret in ("private-response-body", "private-response-header", "private-test-key",
                   "image-only-secret", "model.example", "Authorization", "Bearer"):
        assert secret not in serialized
    assert planner.last_raw is None
    assert planner.wall_ms > 0
    assert planner.plan("Fill the form", snapshot, []).steps
    assert planner.last_validation_error == ""
    assert planner.calls == 2


def test_usage_accumulates_and_single_step_limit_rejects_long_response(snapshot):
    planner = planner_with(lambda _: response(json.dumps(plan(step(), step(id="second"))),
                                             usage={"prompt_tokens": 5}))
    assert len(planner.plan("Fill", snapshot, []).steps) == 2
    with pytest.raises(ValueError):
        planner.plan("Fill", snapshot, [], max_actions=1)
    assert planner.calls == 2
    assert planner.usage_totals["prompt_tokens"] == 10


def test_validation_feedback_is_static_and_cleared_after_success(snapshot):
    replies = iter((
        plan(step(ref="secret-model-output")),
        plan(step()),
    ))
    planner = planner_with(lambda _: response(json.dumps(next(replies))))
    with pytest.raises(ValueError) as error:
        planner.plan("Fill", snapshot, [])
    assert planner.last_validation_error == "step ref is not present in the current observation"
    assert "secret-model-output" not in str(error.value)
    assert planner.last_raw is None
    planner.plan("Fill", snapshot, [])
    assert planner.last_validation_error == ""


@pytest.mark.parametrize("bad_response", [
    lambda: response("secret-model-output trailing JSON {}"),
    lambda: httpx.Response(200, json={"choices": []}),
    lambda: httpx.Response(200, text="secret-non-json-body"),
])
def test_protocol_feedback_never_contains_raw_reply(snapshot, bad_response):
    planner = planner_with(lambda _: bad_response())
    with pytest.raises((RuntimeError, ValueError)):
        planner.plan("Fill", snapshot, [])
    assert planner.last_validation_error in {
        "Response must contain one JSON object in message.content, optionally fenced.",
        "Response body must be valid JSON.",
    }
    assert "secret" not in planner.last_validation_error
    assert planner.last_raw is None


@pytest.mark.parametrize(("operation", "value"), [
    ("type_text", "Literal editor text\nSecond line"),
    ("hotkey", "ctrl+shift+p"),
    ("key_press", "f5"),
])
def test_desktop_actions_require_explicit_desktop_profile(snapshot, operation, value):
    raw = plan(step(operation=operation, ref="", value=value))
    with pytest.raises(ValueError, match="unsupported operation"):
        parse_plan(raw, snapshot)
    with pytest.raises(ValueError, match="unsupported operation"):
        parse_plan(raw, snapshot, profile="browser")
    result = parse_plan(raw, snapshot, profile="desktop")
    assert result.steps[0].value == value
    assert result.steps[0].operation == operation


@pytest.mark.parametrize(("operation", "value"), [
    ("type_text", ""), ("type_text", 1), ("type_text", "x" * 2001),
    ("hotkey", ""), ("hotkey", "ctrl"), ("hotkey", "ctrl++p"),
    ("hotkey", "ctrl+shift+alt+win+p"), ("hotkey", "ctrl+ctrl+p"),
    ("hotkey", "a+b"), ("hotkey", "ctrl+shift"), ("hotkey", "ctrl+UNKNOWN"),
    ("hotkey", "Ctrl+CTRL+P"), ("hotkey", "ctrl+ p"),
    ("hotkey", "ctrl+p;run()"), ("hotkey", "ctrl+\n"),
    ("key_press", "ctrl+a"), ("key_press", "Return"), ("key_press", "arbitrary_code()"),
    ("key_press", "f25"), ("key_press", True),
    ("key_press", "'"), ("key_press", "\\"),
    ("hotkey", "ctrl+'"), ("hotkey", "ctrl+\\"),
])
def test_desktop_keyboard_validation_rejects_invalid_values(snapshot, operation, value):
    with pytest.raises(ValueError):
        parse_plan(plan(step(operation=operation, ref="", value=value)), snapshot, profile="desktop")


def test_desktop_page_actions_never_accept_target_refs_boxes_or_extra_code(snapshot):
    for operation, value in (("type_text", "hello"), ("hotkey", "ctrl+a"), ("key_press", "enter")):
        for extra in ({"ref": "e0"}, {"box": [0, 0, 100, 100]}, {"code": "run()"}):
            raw = plan({**step(operation=operation, ref="", value=value), **extra})
            with pytest.raises(ValueError):
                parse_plan(raw, snapshot, profile="desktop")


def test_desktop_keyboard_whitelist_and_chords_are_shared(snapshot):
    assert {"a", "z", "0", "9", "f1", "f24", "enter", "esc", "win"} <= DESKTOP_KEYS
    assert DESKTOP_MODIFIERS <= DESKTOP_HOTKEY_KEYS == DESKTOP_KEYS
    assert not {"'", "\\"} & DESKTOP_KEYS
    for value in ("ctrl+a", "ctrl+shift+p", "ctrl+alt+shift+f12", "alt+f4", "ctrl+1", "ctrl+/"):
        result = parse_plan(plan(step(operation="hotkey", ref="", value=value)),
                            snapshot, profile="desktop")
        assert result.steps[0].value == value
    assert parse_plan(plan(step(operation="type_text", ref="", value=" " * 2000)),
                      snapshot, profile="desktop").steps[0].value == " " * 2000


@pytest.mark.parametrize("profile", ["", "unknown", "Desktop", None, True, []])
def test_unknown_profile_fails_before_any_request(snapshot, profile):
    with pytest.raises(ValueError, match="profile must"):
        parse_plan(plan(step()), snapshot, profile=profile)
    with pytest.raises(ValueError, match="profile must"):
        planner_with(lambda _: pytest.fail("must not send a request"), profile=profile)


def test_desktop_visual_targets_keep_current_screenshot_requirement(snapshot):
    visual = {"id": "s1", "operation": "visual_click", "label": "Open",
              "box": [100, 200, 300, 280]}
    assert parse_plan(plan(visual), snapshot, profile="desktop").steps[0].box == (100, 200, 300, 280)
    with pytest.raises(ValueError, match="screenshot"):
        parse_plan(plan(visual), replace(snapshot, image=None), profile="desktop")
    with pytest.raises(ValueError, match="only visual"):
        parse_plan(plan({**visual, "operation": "click", "ref": "e4"}), snapshot, profile="desktop")


def test_desktop_planner_uses_computer_prompt_and_profile_in_one_call(desktop_snapshot):
    requests = []

    def handler(request):
        requests.append(request)
        payload = json.loads(request.content)
        instructions = payload["messages"][0]["content"]
        assert "current computer screenshot" in instructions
        assert "accessibility observations" in instructions
        assert "CURRENTLY FOCUSED" in instructions
        assert "ctrl+shift+p" in instructions
        assert "ONE immediate visual action" in instructions
        assert "ENTIRE computer screenshot" in instructions
        assert "NO accessibility control refs" in instructions
        assert "NO DOM controls" not in instructions
        assert "enumerated typed UI action objects" in instructions
        assert "User-requested source code and other file contents ARE permitted" in instructions
        assert "Open/Save file dialogs may be used" in instructions
        assert "ASCII text only (no NUL)" in instructions
        assert "Do not silently translate or replace" in instructions
        assert "Literal text may include newlines, tabs, quotes, and backslashes" in instructions
        assert "password, file, or hidden" not in instructions
        assert "ONE immediate next action" in instructions
        assert len(payload["messages"][1]["content"]) == 2
        return response(json.dumps(plan(step(operation="hotkey", ref="", value="ctrl+shift+p"))))

    planner = planner_with(handler, profile="desktop")
    result = planner.plan("Open the application command palette", desktop_snapshot, [],
                          max_actions=1)
    assert result.steps[0].operation == "hotkey"
    assert len(requests) == planner.calls == 1


@pytest.mark.parametrize("operation", ["type_text", "visual_fill"])
@pytest.mark.parametrize("value", ["中文", "Résumé", "a\x00b"])
def test_desktop_rejects_text_backend_cannot_type_with_specific_feedback(snapshot, operation, value):
    action = step(operation=operation, ref="", value=value)
    if operation == "visual_fill":
        action.update(label="Editor", box=[20, 30, 300, 200])
    with pytest.raises(ValueError, match="ASCII only and cannot contain NUL"):
        parse_plan(plan(action), snapshot, profile="desktop")
    # The browser backend supports Unicode and keeps its existing schema.
    if value != "a\x00b":
        browser_action = action if operation == "visual_fill" else step(value=value)
        assert parse_plan(plan(browser_action), snapshot).steps[0].value == value


def test_desktop_literal_code_and_file_names_are_data_but_execution_payloads_are_rejected(snapshot):
    text = "from pathlib import Path\nprint('hello')\npath = 'folder\\\\file.txt'\n"
    for value in (text, "/home/user/project/test.py"):
        raw = plan(step(operation="type_text", ref="", value=value))
        assert parse_plan(raw, snapshot, profile="desktop").steps[0].value == value
    for payload in ({"code": "print('hello')"}, {"script": "echo hello"}, {"command": "ls"}):
        raw = plan({**step(operation="type_text", ref="", value=text), **payload})
        with pytest.raises(ValueError, match="unsupported fields"):
            parse_plan(raw, snapshot, profile="desktop")
    for operation in ("execute", "python", "shell"):
        with pytest.raises(ValueError, match="unsupported operation"):
            parse_plan(plan(step(operation=operation, ref="", value=text)), snapshot, profile="desktop")


def test_desktop_rejects_multiple_visual_actions_without_normalizing_them(desktop_snapshot):
    first = {"id": "s1", "operation": "visual_click", "label": "Open",
             "box": [100, 200, 300, 280]}
    second = {"id": "s2", "operation": "visual_fill", "label": "Filename",
              "value": "test.py", "box": [350, 300, 700, 400], "depends_on": ["s1"]}
    raw = plan(first, second)
    original = copy.deepcopy(raw)
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    with pytest.raises(ValueError, match="invalid multimodal"):
        planner.plan("Create the file", desktop_snapshot, [])
    assert planner.last_validation_error.startswith(
        "desktop plans allow at most one visual action; observe again before new coordinates"
    )
    assert planner.calls == 1
    assert raw == original
    assert len(parse_plan(raw, desktop_snapshot, profile="browser").steps) == 2


def test_desktop_request_uses_actual_sent_image_dimensions_and_never_guesses_box_units(desktop_snapshot):
    raw = plan({"id": "s1", "operation": "visual_click", "label": "Notifications",
                "box": [680, 0, 710, 25]})

    def handler(request):
        body = json.loads(request.content)
        instructions = body["messages"][0]["content"]
        assert "exactly 1280 pixels wide and 720 pixels high" in instructions
        assert "box_space=normalized_1000" in instructions
        assert "its center is (500,500)" in instructions
        assert "the full-image box is [0,0,1000,1000]" in instructions
        assert (
            "The pixel dimensions describe the attachment, not the preferred "
            "output coordinates" in instructions
        )
        parts = body["messages"][1]["content"]
        state = json.loads(parts[0]["text"])
        assert state["screenshot_geometry"] == {
            "width_px": 1280, "height_px": 720, "preferred_box_space": "normalized_1000",
            "supported_box_spaces": ["normalized_1000", "image_pixels"],
            "default_box_space": "normalized_1000",
        }
        with Image.open(io.BytesIO(base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]))) as image:
            assert image.size == (1280, 720)
        return response(json.dumps(raw))

    planner = planner_with(handler, profile="desktop")
    result = planner.plan("Open notifications", desktop_snapshot, [])
    # A valid 0..1000 box remains literal; apparent pixel coordinates cannot be
    # inferred reliably: a response must explicitly declare the pixel space.
    assert result.steps[0].box == (680, 0, 710, 25)


@pytest.mark.parametrize("operation", ["visual_click", "visual_fill"])
def test_desktop_declared_image_pixels_convert_saved_gimp_shape_without_guessing(
    desktop_snapshot, operation,
):
    # Regression for a 1920x1080 desktop sent as a 1280x720 JPEG. These are
    # coordinates in the attached JPEG, not normalized coordinates or VM pixels.
    raw = plan({"id": "current", "operation": operation, "label": "Preferences",
                "box_space": "image_pixels", "box": [58, 430, 120, 448],
                **({"value": "draft"} if operation == "visual_fill" else {})})
    original = copy.deepcopy(raw)
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Use the visible control", desktop_snapshot, [], max_actions=1)
    assert result.steps[0].box == (45, 597, 94, 622)
    assert result.steps[0].operation == operation and not result.reusable
    assert raw == original and desktop_snapshot.image.region == (0, 0, 1920, 1080)
    raw["steps"][0]["box_space"] = "normalized_1000"
    assert parse_plan(raw, desktop_snapshot, profile="desktop").steps[0].box == (58, 430, 120, 448)
    del raw["steps"][0]["box_space"]
    assert parse_plan(raw, desktop_snapshot, profile="desktop").steps[0].box == (58, 430, 120, 448)


@pytest.mark.parametrize("space, box, expected", [
    ("image_pixels", [0, 0, 1280, 720], (0, 0, 1000, 1000)),
    ("image_pixels", [1024, 360, 1280, 720], (800, 500, 1000, 1000)),
    ("image_pixels", [0, 0, 6, 4], (0, 0, 5, 6)),
    ("image_pixels", [1274, 716, 1280, 720], (995, 994, 1000, 1000)),
    ("normalized_1000", [0, 0, 5, 5], (0, 0, 5, 5)),
    ("normalized_1000", [995, 995, 1000, 1000], (995, 995, 1000, 1000)),
])
def test_desktop_explicit_box_spaces_share_internal_bounds_and_minimum_size(
    desktop_snapshot, space, box, expected,
):
    raw = plan({"id": "current", "operation": "visual_click", "label": "Target",
                "box_space": space, "box": box})
    assert parse_plan(raw, desktop_snapshot, profile="desktop").steps[0].box == expected


@pytest.mark.parametrize("space, box", [
    ("image_pixels", [-1, 0, 100, 100]),
    ("image_pixels", [0, 0, 1281, 720]),  # Fits the VM, but not the attached JPEG.
    ("image_pixels", [0, 0, 1280, 721]),
    ("image_pixels", [0, 0, 5, 4]),  # Width rounds to 4 internal units.
    ("image_pixels", [0, 0, 6, 3]),  # Height rounds to 4 internal units.
    ("image_pixels", [20, 20, 10, 40]),
    ("image_pixels", [20, 20, 20, 40]),
    ("image_pixels", [True, 0, 100, 100]),
    ("image_pixels", [0, 0, 100.5, 100]),
    ("image_pixels", [0, 0, 100]),
    ("normalized_1000", [0, 0, 1001, 720]),
    ("normalized_1000", [0, 0, 4, 5]),
    ("normalized_1000", [0, 0, 5, 4]),
    ("normalized_1000", [0, 0, 100, False]),
])
def test_desktop_declared_box_space_rejects_bad_or_too_small_coordinates(desktop_snapshot, space, box):
    raw = plan({"id": "current", "operation": "visual_click", "label": "Target",
                "box_space": space, "box": box})
    with pytest.raises(ValueError):
        parse_plan(raw, desktop_snapshot, profile="desktop")


@pytest.mark.parametrize("space", [None, True, 1000, "", "pixels", "image_pixels ", [], {}])
def test_desktop_visual_box_space_is_a_strict_enum(desktop_snapshot, space):
    raw = plan({"id": "current", "operation": "visual_click", "label": "Target",
                "box_space": space, "box": [100, 200, 300, 280]})
    with pytest.raises(ValueError, match="box_space must"):
        parse_plan(raw, desktop_snapshot, profile="desktop")


@pytest.mark.parametrize("space", [None, "image_pixels", "normalized_1000"])
def test_box_space_is_desktop_visual_only(snapshot, desktop_snapshot, space):
    visual = {"id": "current", "operation": "visual_click", "label": "Target",
              "box_space": space, "box": [100, 200, 300, 280]}
    with pytest.raises(ValueError, match="unsupported fields"):
        parse_plan(plan(visual), snapshot, profile="browser")
    for action in (
        step(box_space=space),
        semantic_step(box_space=space),
        {"id": "key", "operation": "key_press", "value": "esc", "box_space": space},
    ):
        with pytest.raises(ValueError, match="only visual steps may specify box_space"):
            parse_plan(plan(action), replace(desktop_snapshot, controls=snapshot.controls), profile="desktop")


def test_desktop_pixel_boxes_require_readable_image_without_exposing_bytes(snapshot):
    raw = plan({"id": "current", "operation": "visual_click", "label": "Target",
                "box_space": "image_pixels", "box": [100, 200, 300, 280]})
    with pytest.raises(ValueError, match="readable image") as error:
        parse_plan(raw, snapshot, profile="desktop")
    assert "secret" not in str(error.value)
    with pytest.raises(ValueError, match="screenshot"):
        parse_plan(raw, replace(snapshot, image=None), profile="desktop")


def test_desktop_unreadable_image_is_not_sent_or_exposed(snapshot):
    planner = planner_with(lambda _: pytest.fail("must not send an unreadable image"), profile="desktop")
    with pytest.raises(ValueError, match="desktop screenshot must be a readable image") as error:
        planner.plan("Continue", snapshot, [])
    assert "image-only-secret" not in str(error.value)
    assert planner.calls == 0


def test_model_response_cannot_switch_browser_to_desktop(snapshot):
    raw = plan(step(operation="key_press", ref="", value="enter"))
    planner = planner_with(lambda _: response(json.dumps(raw)))
    with pytest.raises(ValueError):
        planner.plan("Continue", snapshot, [])
    assert planner.profile == "browser"
    assert planner.last_validation_error == "unsupported operation"
    with pytest.raises(ValueError, match="exactly"):
        parse_plan({**raw, "profile": "desktop"}, snapshot)


def test_desktop_future_semantic_targets_are_rebound_not_required_in_current_snapshot(desktop_snapshot):
    field = semantic_step()["target"]
    raw = plan(
        semantic_step(
            operation="click", target={"name": "Preferences", "role": "menu-item"}, value="",
            postconditions=[{"kind": "exists", "target": field}], recovery="observe",
        ),
        semantic_step(id="s2", depends_on=["s1"]),
        semantic_step(
            id="s3", operation="click", value="", depends_on=["s2"],
            target={"name": "Save", "role": "push-button", "ancestor": "Preferences", "match": "exact"},
            postconditions=[{"kind": "absent", "target": {"name": "Preferences", "role": "dialog"}}],
        ),
    )
    result = parse_plan(raw, desktop_snapshot, profile="desktop")
    assert not desktop_snapshot.controls
    assert result.reusable
    assert result.steps[0].recovery == "observe"
    assert result.steps[1].target.name == "Display name"
    assert result.steps[1].target.app == "Example"
    assert result.steps[1].depends_on == ("s1",)
    assert result.steps[2].target.ancestor == "Preferences"
    assert result.steps[2].recovery == "replan"


@pytest.mark.parametrize("target", [
    None, {}, {"app": "Editor"}, {"name": " "}, {"role": ""}, {"name": 1},
    {"role": True}, {"name": "x" * 241}, {"role": "x" * 81},
    {"name": "Save", "window": "x" * 241}, {"name": "Save", "app": None},
    {"name": "Save", "ancestor": []}, {"name": "Save", "match": "regex"},
    {"name": "Save", "match": []}, {"name": "Save", "selector": "#save"},
    {"name": "Save", "box": [1, 2, 100, 200]}, {"name": "Save", "code": "private-code"},
])
def test_semantic_target_rejects_missing_ambiguous_schema_and_executable_fields(desktop_snapshot, target):
    with pytest.raises(ValueError) as error:
        parse_plan(plan(semantic_step(target=target)), desktop_snapshot, profile="desktop")
    assert "private-code" not in str(error.value)


@pytest.mark.parametrize("target", [
    {"role": "entry"}, {"name": "Save"}, {"name": " Display ", "match": "contains"},
    {"role": "check-box", "app": "Settings", "window": "Appearance", "ancestor": "Theme"},
])
def test_semantic_targets_accept_role_or_name_queries_without_current_refs(desktop_snapshot, target):
    result = parse_plan(plan(semantic_step(target=target)), desktop_snapshot, profile="desktop")
    assert result.reusable
    assert result.steps[0].target.match == target.get("match", "exact")


@pytest.mark.parametrize("condition", [
    None, "exists", {}, {"kind": True}, {"kind": "script", "value": "private-code"},
    {"kind": "exists"}, {"kind": "exists", "target": {}},
    {"kind": "exists", "target": {"name": "Save"}, "value": "unexpected"},
    {"kind": "exists", "target": {"name": "Save"}, "code": "private-code"},
    {"kind": "value_equals", "target": {"name": "Name"}},
    {"kind": "value_equals", "target": {"name": "Name"}, "value": 1},
    {"kind": "text_contains", "value": ""}, {"kind": "window_contains", "value": " "},
    {"kind": "text_contains", "value": "ready", "target": {"name": "Name"}},
    {"kind": "window_contains", "value": "x" * 2001},
])
def test_semantic_predicates_reject_unknown_or_incomplete_conditions(desktop_snapshot, condition):
    with pytest.raises(ValueError) as error:
        parse_plan(plan(semantic_step(postconditions=[condition])), desktop_snapshot, profile="desktop")
    assert "private-code" not in str(error.value)


@pytest.mark.parametrize("kind", [
    "exists", "absent", "focused", "checked", "unchecked", "selected", "expanded", "value_equals",
    "text_contains", "window_contains",
])
def test_semantic_condition_kinds_and_explicit_empty_value_are_preserved(desktop_snapshot, kind):
    predicate = {"kind": kind}
    if kind in {"text_contains", "window_contains"}:
        predicate["value"] = "Preferences"
    else:
        predicate["target"] = {"name": "Display name"}
        if kind == "value_equals":
            predicate["value"] = ""
    result = parse_plan(plan(semantic_step(postconditions=[predicate])), desktop_snapshot, profile="desktop")
    condition = result.steps[0].postconditions[0]
    assert condition.kind == kind
    assert condition.value == predicate.get("value", "")
    assert (condition.target is None) == (kind in {"text_contains", "window_contains"})


@pytest.mark.parametrize("change", [
    {"preconditions": None}, {"postconditions": {}}, {"postconditions": []},
    {"postconditions": [{"kind": "text_contains", "value": "ready"}] * 9},
    {"preconditions": [{"kind": "text_contains", "value": "ready"}] * 9},
    {"recovery": "retry_click"}, {"recovery": None}, {"recovery": {}},
    {"ref": "e0"}, {"box": [0, 0, 100, 100]}, {"value": "非 ASCII"},
])
def test_reusable_desktop_steps_have_bounded_conditions_and_recovery(desktop_snapshot, change):
    with pytest.raises(ValueError):
        parse_plan(plan(semantic_step(**change)), desktop_snapshot, profile="desktop")


@pytest.mark.parametrize("operation,value", [
    ("hotkey", "ctrl+shift+p"), ("key_press", "enter"), ("press", "Enter"),
    ("scroll", "down"), ("wait", "250"), ("type_text", "draft.txt"),
])
def test_reusable_keyboard_and_page_actions_need_guards(desktop_snapshot, operation, value):
    pre = [{"kind": "focused", "target": {"name": "Name", "role": "entry"}}]
    action = semantic_step(operation=operation, target=None, value=value, preconditions=pre)
    result = parse_plan(plan(action), desktop_snapshot, profile="desktop")
    assert result.reusable
    assert result.steps[0].preconditions[0].target.name == "Name"
    with pytest.raises(ValueError, match="require preconditions"):
        parse_plan(plan({**action, "preconditions": []}), desktop_snapshot, profile="desktop")
    with pytest.raises(ValueError, match="no direct target"):
        parse_plan(plan({**action, "target": {"name": "Name"}}), desktop_snapshot, profile="desktop")


def test_reusable_typing_requires_focus_not_just_a_visible_field(desktop_snapshot):
    action = semantic_step(operation="type_text", target=None, value="literal source code",
                           preconditions=[{"kind": "exists", "target": {"name": "Editor"}}])
    with pytest.raises(ValueError, match="focused target"):
        parse_plan(plan(action), desktop_snapshot, profile="desktop")


def test_desktop_semantic_and_legacy_fallback_plans_are_distinct(desktop_snapshot):
    visual = {"id": "v1", "operation": "visual_click", "label": "Open", "box": [100, 200, 300, 280]}
    assert not parse_plan(plan(visual), desktop_snapshot, profile="desktop").reusable
    assert not parse_plan(plan(step(operation="hotkey", ref="", value="ctrl+s")),
                          desktop_snapshot, profile="desktop").reusable
    assert not parse_plan(plan(done=True), desktop_snapshot, profile="desktop").reusable
    for actions in ([visual, semantic_step()], [semantic_step(), visual]):
        with pytest.raises(ValueError, match="cannot mix"):
            parse_plan(plan(*actions), desktop_snapshot, profile="desktop")
    with pytest.raises(ValueError, match="cannot mix"):
        parse_plan(plan({**visual, "postconditions": semantic_step()["postconditions"]}),
                   desktop_snapshot, profile="desktop")
    with pytest.raises(ValueError, match="only one current action"):
        parse_plan(plan(
            step(operation="hotkey", ref="", value="ctrl+s"),
            step(id="s2", operation="type_text", ref="", value="draft"),
        ), desktop_snapshot, profile="desktop")


def test_semantic_plans_do_not_weaken_dependency_or_profile_boundaries(snapshot, desktop_snapshot):
    for extra in (semantic_step(), {"target": None}, {"postconditions": []}, {"recovery": "replan"}):
        with pytest.raises(ValueError, match="unsupported fields"):
            parse_plan(plan({**step(), **extra}), snapshot)
    for dependencies in (["missing"], ["s1"], [True]):
        with pytest.raises(ValueError, match="preceding step ids"):
            parse_plan(plan(semantic_step(depends_on=dependencies)), desktop_snapshot, profile="desktop")
    with pytest.raises(ValueError, match="desktop select is not supported"):
        parse_plan(plan(semantic_step(operation="select")), desktop_snapshot, profile="desktop")
    with pytest.raises(ValueError, match="requires exactly"):
        parse_plan({**plan(semantic_step()), "reusable": True}, desktop_snapshot, profile="desktop")


@pytest.mark.parametrize("limit", [1, 6])
def test_desktop_semantic_prompt_supports_reuse_and_same_schema_single_step_baseline(desktop_snapshot, limit):
    desktop_snapshot = replace(desktop_snapshot, controls=(AgentControl(
        "d0", "Display name", "input", role="textbox", app="Example", window="Preferences",
        editable=True, box=(100, 100, 300, 180),
    ),))
    calls = []
    raw = plan(semantic_step())
    if limit > 1:
        raw["steps"].append(semantic_step(id="s2", depends_on=["s1"]))

    def handler(request):
        calls.append(request)
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        if limit > 1:
            assert "Target may be future; scope is current" in prompt
            assert "postconditions=observable arrival" in prompt
            assert "never proof to toggle or type again" in prompt
            assert "No target/conditions or future visual coordinates" in prompt
            example_text = prompt.split("Example schema only: ", 1)[1]
            example, _ = json.JSONDecoder().raw_decode(example_text)
            assert parse_plan(example, desktop_snapshot, max_actions=limit, profile="desktop").reusable
            return response(json.dumps(raw))
        assert "Future semantic targets need not exist in the current observation" in prompt
        assert "Every reusable step MUST have nonempty postconditions" in prompt
        assert "at most two extra fresh observations" in prompt
        assert "dialog does NOT by itself require another VLM call" in prompt
        assert "Never mix this fallback with future semantic steps or conditions" in prompt
        assert "Only plan targets known in the CURRENT observation" not in prompt
        assert "Never invent future dialogs" not in prompt
        assert "select is unsupported" in prompt
        assert "focused predicate" in prompt
        if limit == 1:
            assert "ONE immediate next action" in prompt
        examples = prompt.split("Complete JSON examples (schema examples, not targets to copy): ", 1)[1]
        decoder = json.JSONDecoder()
        count = 0
        while examples.startswith("{"):
            example, end = decoder.raw_decode(examples)
            parse_plan(example, desktop_snapshot, max_actions=limit, profile="desktop")
            examples = examples[end:].removeprefix(". ")
            count += 1
        assert count == 3
        return response(json.dumps(raw))

    planner = planner_with(handler, profile="desktop")
    result = planner.plan("Change the displayed name", desktop_snapshot, [], max_actions=limit)
    assert result.reusable
    assert len(result.steps) == (1 if limit == 1 else 2)
    assert len(calls) == planner.calls == 1
    if limit == 1:
        with pytest.raises(ValueError, match="bounded steps"):
            parse_plan(plan(semantic_step(), semantic_step(id="s2")), desktop_snapshot,
                       max_actions=1, profile="desktop")


def test_invalid_semantic_plan_feedback_exposes_static_schema_error_not_response(desktop_snapshot):
    raw = plan(semantic_step(recovery="private-model-output"))
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    with pytest.raises(ValueError) as error:
        planner.plan("Fill the name", desktop_snapshot, [])
    assert planner.last_validation_error == "recovery must be replan or observe"
    assert "private-model-output" not in str(error.value)
    assert planner.last_raw is None


def test_recovery_context_keeps_semantic_actions_legal_and_bounds_only_active_recovery(desktop_snapshot):
    current = replace(desktop_snapshot, controls=(AgentControl(
        "d0", "Display name", "input", role="textbox", app="Example", window="Preferences",
        editable=True, box=(100, 100, 300, 180),
    ),))
    feedback = {
        "reason": "no_executable_target", "recovery_required": True,
        "ready_steps": [{"step_id": "old", "operation": "click", "target": {"name": "Missing"},
                         "match_count": 0, "candidate_count": 0, "matches": [], "preconditions": []}],
    }
    requests = []
    replies = [
        plan(semantic_step(), semantic_step(id="s2")),
        plan(semantic_step()),
        plan(semantic_step(), semantic_step(id="s2")),
    ]

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        state = json.loads(body["messages"][1]["content"][0]["text"])
        assert state["snapshot"]["controls"][0]["editable"]
        assert state["recent_actions"][0]["ready_steps"][0]["candidate_count"] == 0
        return response(json.dumps(replies.pop(0)))

    planner = planner_with(handler, profile="desktop")
    # Even if a caller supplies the normal upper bound, an active recovery
    # cannot smuggle a second speculative future action into its response.
    with pytest.raises(ValueError):
        planner.plan("Fill the display name", current, [feedback], max_actions=6)
    assert planner.last_validation_error == "plan must contain bounded steps or an advisory done flag"
    recovery = planner.plan("Fill the display name", current, [feedback], max_actions=6)
    assert recovery.reusable and len(recovery.steps) == 1
    assert recovery.steps[0].operation == "fill" and recovery.steps[0].target.name == "Display name"
    resumed = planner.plan("Continue", current, [feedback, {
        "verified": True, "recovery_required": False, "feedback": "effect checked",
    }], max_actions=6)
    assert len(resumed.steps) == 2 and resumed.reusable
    assert planner.calls == len(requests) == 3


@pytest.mark.parametrize("scope", [
    {"app": "Example"}, {"window": "Preferences"},
    {"app": " Example ", "window": "Preferences", "ancestor": "Settings"},
])
def test_navigation_subgoal_scopes_fresh_controls_and_unlocks_precise_dependencies(desktop_snapshot, scope):
    result = parse_plan(plan(
        navigation_step(target=scope),
        semantic_step(depends_on=["navigate"]),
    ), desktop_snapshot, profile="desktop")
    assert result.reusable and not desktop_snapshot.controls
    navigation, precise = result.steps
    assert navigation.navigation and not precise.navigation
    assert navigation.label == "Reach the preferences dialog"
    assert navigation.target.app == scope.get("app", "").strip()
    assert navigation.target.window == scope.get("window", "")
    assert navigation.target.ancestor == scope.get("ancestor", "")
    assert navigation.target.name == navigation.target.role == ""
    assert navigation.postconditions[0].target.name == "Display name"
    assert precise.depends_on == (navigation.id,)
    assert precise.operation == "fill" and precise.value == "Ada"


@pytest.mark.parametrize("scope", [
    None, [], {}, {"ancestor": "Settings"}, {"app": " "}, {"window": ""},
    {"app": "", "window": "\t", "ancestor": "Settings"},
    {"app": 1}, {"app": "Example", "window": None},
    {"app": "x" * 241}, {"window": "x" * 241},
    {"app": "Example", "ancestor": "x" * 241},
    {"app": "Example", "name": ""}, {"app": "Example", "role": "button"},
    {"app": "Example", "match": "exact"}, {"app": "Example", "selector": "#target"},
    {"app": "Example", "code": "private-command"},
])
def test_navigation_scope_cannot_pin_controls_or_escape_application_context(desktop_snapshot, scope):
    with pytest.raises(ValueError) as error:
        parse_plan(plan(navigation_step(target=scope)), desktop_snapshot, profile="desktop")
    assert "private-command" not in str(error.value)


@pytest.mark.parametrize("change", [
    {"navigation": 1}, {"navigation": 0}, {"navigation": "true"}, {"navigation": None},
    {"navigation": []}, {"operation": "fill", "value": "Ada"}, {"operation": "check"},
    {"operation": "hotkey", "value": "ctrl+p"}, {"label": ""}, {"label": "  "},
    {"value": "private-value"}, {"ref": "e0"}, {"box": [100, 200, 300, 400]},
    {"postconditions": []}, {"postconditions": [{"kind": "exists", "target": {"app": "Example"}}]},
    {"preconditions": [{"kind": "script", "value": "private-value"}]},
])
def test_navigation_requires_safe_click_subgoal_and_observable_completion(desktop_snapshot, change):
    with pytest.raises(ValueError) as error:
        parse_plan(plan(navigation_step(**change)), desktop_snapshot, profile="desktop")
    assert "private-value" not in str(error.value)


def test_navigation_cannot_enter_browser_or_single_action_baseline(snapshot, desktop_snapshot):
    for flag in (False, True):
        with pytest.raises(ValueError, match="unsupported fields"):
            parse_plan(plan(step(navigation=flag)), snapshot)
    with pytest.raises(ValueError, match="single-action planning"):
        parse_plan(plan(navigation_step()), desktop_snapshot, profile="desktop", max_actions=1)
    # A one-subgoal fast plan is valid when the execution horizon permits
    # multiple inputs. The number of planned subgoals does not set that horizon.
    result = parse_plan(plan(navigation_step()), desktop_snapshot, profile="desktop", max_actions=6)
    assert len(result.steps) == 1 and result.steps[0].navigation
    ordinary = parse_plan(plan(semantic_step(navigation=False)), desktop_snapshot,
                          profile="desktop", max_actions=1)
    assert not ordinary.steps[0].navigation


def test_navigation_response_is_accepted_only_outside_single_action_recovery(navigation_snapshot):
    desktop_snapshot = navigation_snapshot
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(navigation_step())))

    planner = planner_with(handler, profile="desktop")
    normal = planner.plan("Change the display name", desktop_snapshot, [], max_actions=6)
    assert normal.reusable and normal.steps[0].navigation
    for limit, history in ((1, []), (6, [{"recovery_required": True}])):
        with pytest.raises(ValueError):
            planner.plan("Change the display name", desktop_snapshot, history, max_actions=limit)
        assert planner.last_validation_error == "single-action planning does not allow navigation subgoals"
        assert planner.last_raw is None
    resumed = planner.plan("Change the display name", desktop_snapshot, [
        {"recovery_required": True}, {"recovery_required": False, "verified": True},
    ], max_actions=6)
    assert resumed.steps[0].navigation
    assert planner.calls == len(requests) == 4
    # The transport examples themselves exercise the different contracts:
    # normal planning teaches a scope subgoal; baseline/recovery teach an exact
    # next action. They are all parsed through the real strict parser.
    for request, navigation_allowed in zip(requests, (True, False, False, True), strict=True):
        prompt = request["messages"][0]["content"]
        example_marker = ("Example schema only: " if navigation_allowed else
                          "Complete JSON examples (schema examples, not targets to copy): ")
        examples = prompt.split(example_marker, 1)[1]
        raw_example, _ = json.JSONDecoder().raw_decode(examples)
        example = parse_plan(raw_example, desktop_snapshot, profile="desktop",
                             max_actions=6 if navigation_allowed else 1)
        assert example.steps[0].navigation is navigation_allowed
        if navigation_allowed:
            assert example.steps[0].target.name == example.steps[0].target.role == ""
        else:
            assert example.steps[0].target.name


def test_live_hybrid_rejects_exact_navigation_chain_even_when_later_subgoal_delegates(desktop_snapshot):
    raw = plan(
        {"id": "open", "operation": "click", "target": {"name": "Edit", "app": "Example"},
         "postconditions": [{"kind": "exists", "target": {"name": "Preferences"}}]},
        navigation_step(depends_on=["open"]),
    )
    # The wire parser remains backwards compatible; only the live hybrid
    # planning contract rejects route scripting before a delegated subgoal.
    assert parse_plan(raw, desktop_snapshot, profile="desktop").reusable
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    with pytest.raises(ValueError):
        planner.plan("Change preferences", desktop_snapshot, [])
    assert planner.last_validation_error.startswith("step 1: hybrid UI navigation requires navigation=true")
    assert planner.calls == 1 and planner.last_raw is None


@pytest.mark.parametrize("postconditions,preconditions", [
    ([{"kind": "exists", "target": {"name": "Preferences"}}], []),
    ([{"kind": "expanded", "target": {"name": "Edit"}}], []),
    ([{"kind": "selected", "target": {"name": "Different target"}}], []),
    ([{"kind": "absent", "target": {"name": "Preferences"}}],
     [{"kind": "exists", "target": {"name": "Preferences"}}]),
])
def test_live_hybrid_exact_click_requires_local_effect_or_persistent_commit_guard(
    desktop_snapshot, postconditions, preconditions,
):
    raw = plan(semantic_step(), {
        "id": "route", "operation": "click", "target": {"name": "Edit"},
        "postconditions": postconditions, "preconditions": preconditions,
    })
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    with pytest.raises(ValueError):
        planner.plan("Change preferences", desktop_snapshot, [])
    assert planner.last_validation_error.startswith("step 2: hybrid UI navigation requires navigation=true")


def test_live_hybrid_navigation_then_precise_check_is_legal_without_inflated_choices(navigation_snapshot):
    desktop_snapshot = navigation_snapshot
    target = {"name": "Notifications", "role": "checkbox", "app": "Example"}
    raw = plan(
        navigation_step(postconditions=[{"kind": "exists", "target": target}]),
        {"id": "set", "operation": "check", "target": target, "depends_on": ["navigate"],
         "postconditions": [{"kind": "checked", "target": target}]},
    )
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(raw))

    planner = planner_with(handler, profile="desktop")
    result = planner.plan("Enable notifications", desktop_snapshot, [])
    assert result.steps[0].navigation and result.steps[1].operation == "check"
    assert not result.steps[1].navigation
    assert requests[0]["messages"][0]["content"].startswith("PLAN ONE MEANINGFUL DESKTOP STAGE.")


def test_live_hybrid_preserves_exact_focus_typing_and_final_guarded_commit(desktop_snapshot):
    field = {"name": "Contents", "app": "Editor"}
    value_condition = {"kind": "value_equals", "target": field, "value": "draft"}
    raw = plan(
        {"id": "focus", "operation": "click", "target": field,
         "postconditions": [{"kind": "focused", "target": field}]},
        {"id": "append", "operation": "type_text", "value": "draft", "depends_on": ["focus"],
         "preconditions": [{"kind": "focused", "target": field}],
         "postconditions": [value_condition]},
        {"id": "save", "operation": "click", "target": {"name": "Save", "app": "Editor"},
         "depends_on": ["append"], "preconditions": [value_condition],
         "postconditions": [{"kind": "absent", "target": {"name": "Unsaved"}}]},
    )
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Append text and save", desktop_snapshot, [])
    assert [s.operation for s in result.steps] == ["click", "type_text", "click"]
    assert all(not s.navigation for s in result.steps)
    # A commit cannot be used to smuggle an exact menu route before a subgoal.
    raw["steps"].append(navigation_step(depends_on=["save"]))
    with pytest.raises(ValueError):
        planner.plan("Append text and save", desktop_snapshot, [])
    assert planner.last_validation_error.startswith("step 3 (steps[2]): guarded exact click must end")


@pytest.mark.parametrize("kind", ["selected", "focused", "checked", "unchecked"])
def test_live_hybrid_accepts_exact_same_target_local_effect(desktop_snapshot, kind):
    target = {"name": "Current option", "app": "Example"}
    raw = plan({"id": "select", "operation": "click", "target": target,
                "postconditions": [{"kind": kind, "target": target}]})
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Set the local state", desktop_snapshot, [])
    assert result.reusable and not result.steps[0].navigation


def test_live_hybrid_pure_edit_plan_does_not_require_navigation(desktop_snapshot):
    planner = planner_with(lambda _: response(json.dumps(plan(semantic_step()))), profile="desktop")
    result = planner.plan("Fill the field", desktop_snapshot, [])
    assert len(result.steps) == 1 and result.steps[0].operation == "fill"
    assert not result.steps[0].navigation


@pytest.mark.parametrize("change", [{}, {"disabled": True}, {"box": None},
                                    {"tag": "text", "role": "label"}])
def test_live_immediate_click_requires_current_executable_target(desktop_snapshot, change):
    current_control = AgentControl("d0", "Edit", "button", role="menuitem", app="Example",
                                   box=(100, 100, 300, 180))
    current = replace(desktop_snapshot, controls=(replace(current_control, **change),))
    raw = plan({"id": "open", "operation": "click", "target": {"name": "Edit", "app": "Example"},
                "postconditions": [{"kind": "exists", "target": {"name": "Preferences"}}]})
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    if change:
        with pytest.raises(ValueError):
            planner.plan("Open preferences", current, [], max_actions=1)
        assert planner.last_validation_error.startswith("step 1: immediate target has no current executable")
    else:
        result = planner.plan("Open preferences", current, [], max_actions=1)
        assert result.steps[0].target.name == "Edit" and not result.steps[0].navigation


def test_immediate_validation_uses_operation_capability_and_current_preconditions(desktop_snapshot):
    target = {"name": "Display name", "role": "entry", "app": "Example", "window": "Preferences"}
    control = AgentControl("d0", "Display name", "input", role="textbox", app="Example",
                           window="Preferences", box=(100, 100, 300, 180), editable=False)
    current = replace(desktop_snapshot, controls=(control,))
    raw = plan(semantic_step())
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    with pytest.raises(ValueError):
        planner.plan("Fill the name", current, [], max_actions=1)
    assert planner.last_validation_error.startswith("step 1: immediate target has no current executable")
    raw["steps"][0]["preconditions"] = [{"kind": "focused", "target": target}]
    current = replace(current, controls=(replace(control, editable=True),))
    with pytest.raises(ValueError):
        planner.plan("Fill the name", current, [], max_actions=1)
    assert planner.last_validation_error == (
        "step 1: immediate action preconditions do not hold now; "
        "first_nontrue_precondition_index0=0, kind=focused, outcome=false, reason=state_mismatch"
    )
    current = replace(current, controls=(replace(control, editable=True, focused=True),))
    assert planner.plan("Fill the name", current, [], max_actions=1).steps[0].operation == "fill"


def test_unavailable_recovery_is_rejected_before_wait_and_visual_recovery_remains_legal(desktop_snapshot):
    feedback = {"reason": "no_executable_target", "recovery_required": True}
    raw = plan({"id": "private-step-id", "operation": "click",
                "target": {"name": "private-target-label", "app": "Example"},
                "postconditions": [{"kind": "exists", "target": {"name": "Preferences"}}],
                "recovery": "observe"})
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(raw))

    planner = planner_with(handler, profile="desktop")
    assert parse_plan(raw, desktop_snapshot, max_actions=1, profile="desktop").reusable
    with pytest.raises(ValueError) as error:
        planner.plan("Open preferences", desktop_snapshot, [feedback], max_actions=6)
    assert planner.last_validation_error.startswith("step 1: immediate target has no current executable")
    assert "private-" not in planner.last_validation_error + str(error.value)
    assert planner.last_raw is None
    raw = plan({"id": "visual", "operation": "visual_click", "label": "Preferences",
                "box": [100, 200, 300, 280]})
    result = planner.plan("Open preferences", desktop_snapshot, [feedback], max_actions=6)
    assert not result.reusable and result.steps[0].operation == "visual_click"
    assert requests[0]["messages"][0]["content"].startswith("RECOVERY PRIORITY:")
    assert planner.calls == 2


def test_desktop_goal_completion_is_explicit_and_separate_from_step_effects(desktop_snapshot):
    raw = plan(navigation_step(), semantic_step(depends_on=["navigate"]))
    assert parse_plan(raw, desktop_snapshot, profile="desktop").completion_conditions == ()
    raw["completion_conditions"] = semantic_step()["postconditions"]
    result = parse_plan(raw, desktop_snapshot, profile="desktop")
    assert result.reusable and not result.done
    assert result.completion_conditions == result.steps[-1].postconditions
    assert result.completion_conditions != result.steps[0].postconditions
    # A baseline's last action can make the same declaration without requiring
    # another model call merely to say done.
    single = {**plan(semantic_step()), "completion_conditions": raw["completion_conditions"]}
    baseline = parse_plan(single, desktop_snapshot, profile="desktop", max_actions=1)
    assert baseline.completion_conditions == result.completion_conditions


@pytest.mark.parametrize("conditions", [None, True, {}, "done", [None],
                                       [{"kind": "success"}],
                                       [{"kind": "exists", "target": {"app": "Example"}}],
                                       [{"kind": "text_contains", "value": "ready"}] * 9])
def test_goal_completion_conditions_use_the_existing_bounded_predicate_schema(desktop_snapshot, conditions):
    with pytest.raises(ValueError):
        parse_plan({**plan(semantic_step()), "completion_conditions": conditions},
                   desktop_snapshot, profile="desktop")


@pytest.mark.parametrize("raw", [
    plan({"id": "s1", "operation": "hotkey", "value": "ctrl+s"}),
    plan({"id": "s1", "operation": "visual_click", "label": "Save", "box": [100, 100, 300, 200]}),
])
def test_completion_declaration_cannot_upgrade_raw_or_visual_fallback(desktop_snapshot, raw):
    raw = {**raw, "completion_conditions": [{"kind": "text_contains", "value": "Saved"}]}
    with pytest.raises(ValueError, match="reusable semantic desktop plan or empty advisory done"):
        parse_plan(raw, desktop_snapshot, profile="desktop")


@pytest.mark.parametrize("max_actions", [1, 6])
@pytest.mark.parametrize("condition_count", [1, 8])
def test_desktop_empty_done_retains_verifiable_goal_conditions_without_noop(
    desktop_snapshot, max_actions, condition_count,
):
    raw = {**plan(done=True), "completion_conditions": [
        {"kind": "value_equals", "target": {"name": f"Field {index}", "app": "Example"},
         "value": f"Expected {index}"} for index in range(condition_count)
    ]}
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Check the entire goal", desktop_snapshot, [], max_actions=max_actions)
    assert result.done and not result.steps and not result.reusable
    assert len(result.completion_conditions) == condition_count
    assert result.completion_conditions[0].value == "Expected 0"
    assert planner.calls == 1 and not planner.last_validation_error


@pytest.mark.parametrize("conditions", [None, True, {}, [None],
                                       [{"kind": "checked", "target": {"app": "Example"}}],
                                       [{"kind": "text_contains", "value": "ready"}] * 9])
def test_empty_done_conditions_keep_strict_predicate_validation(desktop_snapshot, conditions):
    with pytest.raises(ValueError):
        parse_plan({**plan(done=True), "completion_conditions": conditions},
                   desktop_snapshot, profile="desktop")


def test_browser_rejects_desktop_completion_field_and_desktop_empty_field_preserves_legacy(snapshot,
                                                                                          desktop_snapshot):
    for predicates in ([], semantic_step()["postconditions"]):
        for raw in (plan(step()), plan(done=True)):
            with pytest.raises(ValueError):
                parse_plan({**raw, "completion_conditions": predicates}, snapshot)
    raw = {**plan(semantic_step()), "completion_conditions": []}
    assert parse_plan(raw, desktop_snapshot, profile="desktop").completion_conditions == ()


@pytest.mark.parametrize("profile", ["browser", "desktop"])
@pytest.mark.parametrize("enable_thinking", [None, False, True])
def test_thinking_extension_is_sent_only_when_explicitly_configured(snapshot, desktop_snapshot,
                                                                 profile, enable_thinking):
    requests = []
    raw = plan(step()) if profile == "browser" else plan(semantic_step())

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(raw))

    planner = planner_with(handler, profile=profile, enable_thinking=enable_thinking)
    planner.plan("Fill the name", snapshot if profile == "browser" else desktop_snapshot, [])
    body = requests[0]
    if enable_thinking is None:
        assert "enable_thinking" not in body
    else:
        assert body["enable_thinking"] is enable_thinking
    assert planner.enable_thinking is enable_thinking and planner.calls == 1


@pytest.mark.parametrize("value", [0, 1, "false", "off", [], {}])
def test_thinking_extension_rejects_nonboolean_configuration(value):
    with pytest.raises(ValueError, match="enable_thinking must be a boolean or None"):
        planner_with(lambda _: pytest.fail("invalid configuration must not send requests"),
                     enable_thinking=value)


def test_desktop_payload_preserves_controls_losslessly_and_keeps_unrepresented_text(desktop_snapshot):
    controls = (
        AgentControl("d0", "Name", "input", role="textbox", input_type="text", value="",
                     app="Example", window="Preferences", ancestors=("frame: Preferences",),
                     focused=True, editable=True, active=True, box=(100, 100, 300, 180)),
        AgentControl("d1", "Alerts", "input", role="checkbox", input_type="checkbox",
                     app="Example", window="Preferences", checked=False, box=(100, 200, 300, 280)),
        AgentControl("d2", "Theme", "button", role="listitem", app="Example", window="Preferences",
                     selected=True, expanded=True, disabled=True, options=("Light", "Dark")),
        AgentControl("d3", "Notice", "text", role="label", app="Example", window="Preferences"),
        AgentControl("d4", "Website", "a", role="link", href="https://example.test", app="Example"),
        AgentControl("d5", "Filename", "input", role="textbox", value="draft.txt", editable=True,
                     app="Example", window="Preferences", box=(100, 300, 300, 380)),
    )
    text = (
        'application: Example\n'
        'entry: Name [focused] box_1000=[100, 100, 300, 180]\n'
        'check-box: Alerts box_1000=[100, 200, 300, 280]\n'
        'entry: Filename | draft.txt box_1000=[100, 300, 300, 380]\n'
        'label: Notice | File has not been saved\n'
        'application: Other\n'
        'label: Notice\n'
        'label: Clock reads 10:42\n'
    )
    current = replace(desktop_snapshot, controls=controls, text=text, a11y_complete=False)
    before = current.fingerprint()
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(done=True)))

    planner_with(handler, profile="desktop").plan("Inspect preferences", current, [])
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    compact = state["snapshot"]
    reconstructed = [
        {**compact["control_defaults"], **{key: value for key, value in control.items()
                                         if key not in {"operations", "condition_kinds"}}}
        for control in compact["controls"]
    ]
    expected = json.loads(json.dumps([control.to_dict() for control in controls]))
    assert reconstructed == expected
    assert compact["controls"][0]["value"] == ""  # An empty editor value remains explicit.
    assert compact["control_defaults"]["checked"] is False
    assert not compact["a11y_complete"]
    assert "entry: Name" not in compact["text"] and "check-box: Alerts" not in compact["text"]
    assert "entry: Filename" not in compact["text"]
    assert "File has not been saved" in compact["text"]
    assert "application: Other\nlabel: Notice" in compact["text"]
    assert "Clock reads 10:42" in compact["text"]
    assert current.fingerprint() == before and current.text == text


@pytest.mark.parametrize("changes, expected", [
    ({}, ["click"]),
    ({"tag": "input", "role": "textbox", "input_type": "text", "editable": True}, ["click", "fill"]),
    ({"tag": "input", "role": "checkbox", "input_type": "checkbox"}, ["click", "check", "uncheck"]),
    ({"tag": "input", "role": "radio", "input_type": "radio"}, ["click", "check"]),
    ({"tag": "text", "role": "label"}, []),
    ({"disabled": True}, []),
    ({"box": None}, []),
    ({"box": (100, 100, 102, 104)}, []),
    ({"tag": "input", "role": "textbox", "input_type": "password", "editable": True}, []),
])
def test_desktop_payload_exposes_actual_supported_operations(desktop_snapshot, changes, expected):
    control = replace(AgentControl("d0", "Current control", "button", role="button", app="Example",
                                   box=(100, 100, 300, 180)), **changes)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(done=True)))

    planner_with(handler, profile="desktop").plan("Inspect", replace(
        desktop_snapshot, controls=(control,),
    ), [])
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert state["snapshot"]["controls"][0]["operations"] == expected
    assert expected == [operation for operation in ("click", "fill", "check", "uncheck")
                        if executable_control(control, operation)]
    assert "operations" not in control.to_dict()  # Derived guidance does not alter freshness data.


def test_desktop_condition_capabilities_are_distinct_from_executable_operations(desktop_snapshot):
    controls = (
        AgentControl("row", "Quiet mode", "button", role="listitem", app="Example",
                     box=(100, 100, 300, 180)),
        AgentControl("toggle", "Quiet mode", "input", role="checkbox", input_type="checkbox",
                     app="Example", checked=True, disabled=True),
        AgentControl("anonymous", "", "container", role="checkbox", input_type="checkbox",
                     app="Example", checked=False),
        AgentControl("field", "Name", "input", role="textbox", input_type="text",
                     app="Example", value="", editable=True),
        AgentControl("caption", "Status", "text", role="label", app="Example"),
    )
    current = replace(desktop_snapshot, controls=controls)
    before = current.fingerprint()
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(done=True)))

    planner_with(handler, profile="desktop").plan("Inspect observable state", current, [])
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    payload = {control["ref"]: control for control in state["snapshot"]["controls"]}
    assert payload["row"]["operations"] == ["click"]
    assert not {"checked", "unchecked"} & set(payload["row"]["condition_kinds"])
    for ref in ("toggle", "anonymous"):
        assert payload[ref]["operations"] == []
        assert {"checked", "unchecked"} <= set(payload[ref]["condition_kinds"])
    assert "value_equals" in payload["field"]["condition_kinds"]
    assert "value_equals" not in payload["caption"]["condition_kinds"]
    for control in controls:
        assert payload[control.ref]["condition_kinds"] == list(observable_condition_kinds(control))
        assert "condition_kinds" not in control.to_dict()
    assert current.fingerprint() == before


@pytest.mark.parametrize("limit", [1, 6])
def test_desktop_passes_unknown_evidence_to_recovery_without_claiming_an_opposite_state(
    desktop_snapshot, limit,
):
    current = replace(desktop_snapshot, controls=(AgentControl(
        "row", "Quiet mode", "button", role="listitem", app="Example", box=(100, 100, 300, 180),
    ),))
    history = [{
        "reason": "postcondition_unconfirmed", "recovery_required": True,
        "conditions": [
            {"kind": "checked", "target": {"name": "Quiet mode", "role": "listitem"},
             "satisfied": False, "outcome": "unknown", "reason": "unsupported_state"},
            {"kind": "checked", "target": {"name": "Other switch", "role": "checkbox"},
             "satisfied": False, "outcome": "false", "reason": "state_mismatch"},
            {"kind": "exists", "target": {"name": "Quiet mode", "role": "listitem"},
             "satisfied": True, "outcome": "true", "reason": "condition_met"},
        ],
    }]
    original = copy.deepcopy(history)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan({
            "id": "inspect", "operation": "visual_click", "label": "Visible settings control",
            "box_space": "normalized_1000", "box": [100, 200, 300, 280],
        })))

    result = planner_with(handler, profile="desktop").plan("Inspect the current setting", current, history,
                                                         max_actions=limit)
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert state["recent_actions"] == original == history
    assert "checked" not in state["snapshot"]["controls"][0]["condition_kinds"]
    assert "outcome=true/false/unknown" in requests[0]["messages"][0]["content"]
    assert not result.reusable and not result.completion_conditions
    assert result.steps[0].operation == "visual_click" and not result.steps[0].postconditions


@pytest.mark.parametrize("kind", ["checked", "unchecked"])
@pytest.mark.parametrize("role", ["button", "listitem", "tree-item", "label", "window", "private-role"])
@pytest.mark.parametrize("location", ["preconditions", "postconditions", "completion_conditions"])
def test_live_desktop_rejects_structurally_impossible_checkable_condition_roles(
    desktop_snapshot, kind, role, location,
):
    predicate = {"kind": kind, "target": {"name": "private-target", "role": role, "app": "Example"}}
    raw = plan(semantic_step())
    if location == "completion_conditions":
        raw[location] = [predicate]
    else:
        raw["steps"][0][location] = [predicate]
    original = copy.deepcopy(raw)
    # The transport parser remains compatible; execution planning applies the
    # extra contradiction check before any action is offered.
    assert parse_plan(raw, desktop_snapshot, profile="desktop").reusable
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    with pytest.raises(ValueError):
        planner.plan("Prepare the future setting", desktop_snapshot, [])
    position = "completion_conditions" if location == "completion_conditions" else "step 1"
    assert planner.last_validation_error.startswith(
        f"{position}: checked/unchecked requires a checkable role;"
    )
    assert "private-" not in planner.last_validation_error
    assert raw == original and planner.last_raw is None


@pytest.mark.parametrize("kind", ["checked", "unchecked"])
@pytest.mark.parametrize("role", [
    "", "checkbox", "radio", "check-box", "check-menu-item", "toggle-button", "toggle", "switch",
    "radio-button", "radio-menu-item",
])
def test_live_future_checkable_conditions_keep_empty_roles_and_supported_aliases(
    navigation_snapshot, kind, role,
):
    future = {"name": "Quiet mode", "app": "Example", "role": role}
    raw = plan(navigation_step(), {
        "id": "set", "operation": "check" if kind == "checked" else "uncheck",
        "depends_on": ["navigate"], "target": future,
        "postconditions": [{"kind": kind, "target": future}],
    })
    raw["completion_conditions"] = [{"kind": kind, "target": future}]
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Set the requested state", navigation_snapshot, [])
    assert result.steps[1].target.role == role
    assert result.completion_conditions[0].target.role == role


def test_desktop_payload_reduces_repeated_evidence_without_dropping_controls(desktop_snapshot):
    controls = tuple(AgentControl(f"d{i}", f"Option {i}", "button", role="button", app="Example",
                                 box=(100, 100, 300, 180)) for i in range(40))
    current = replace(desktop_snapshot, controls=controls, text='application: Example\n' + '\n'.join(
        f'button: Option {i} box_1000=[100, 100, 300, 180]' for i in range(40)
    ))
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(done=True)))

    planner_with(handler, profile="desktop").plan("Inspect options", current, [])
    compact = json.loads(requests[0]["messages"][1]["content"][0]["text"])["snapshot"]
    assert len(compact["controls"]) == len(controls) and not compact["text"]
    assert len(json.dumps(compact)) < len(json.dumps(current.to_dict())) * 0.6


def test_compaction_retains_text_only_observations_and_browser_payload(snapshot, desktop_snapshot):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(done=True)))

    text_only = replace(desktop_snapshot, text='application: Example\nlabel: Status ready')
    planner_with(handler, profile="desktop").plan("Inspect", text_only, [])
    compact = json.loads(requests[0]["messages"][1]["content"][0]["text"])["snapshot"]
    assert compact["text"] == text_only.text
    planner_with(handler).plan("Inspect", snapshot, [])
    browser = json.loads(requests[1]["messages"][1]["content"][0]["text"])["snapshot"]
    assert browser == json.loads(json.dumps(snapshot.to_dict()))


def test_active_navigation_rejects_nonexecutable_window_and_offers_app_scope(navigation_snapshot):
    current = replace(navigation_snapshot, controls=(*navigation_snapshot.controls,
        AgentControl("d1", "Window", "container", role="frame", app="Example", window="private-window"),
        AgentControl("d2", "Unavailable", "button", role="button", app="Other", disabled=True,
                     box=(100, 200, 300, 280)),
    ))
    raw = plan(navigation_step(target={"app": "Example", "window": "private-window"}))
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(raw))

    planner = planner_with(handler, profile="desktop")
    with pytest.raises(ValueError) as error:
        planner.plan("Reach preferences", current, [])
    assert planner.last_validation_error.startswith("step 1: active navigation scope has no usable")
    assert "private-window" not in planner.last_validation_error + str(error.value)
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert state["navigation_scopes"] == [{"target": {"app": "Example"}, "candidate_count": 1}]
    # One real executable control is sufficient; the planner neither invents
    # controls nor requires a large choice set just to increase candidate count.
    raw["steps"][0]["target"] = {"app": "Example"}
    assert planner.plan("Reach preferences", current, []).steps[0].navigation


def test_future_dependent_navigation_may_use_a_scope_that_has_not_appeared(navigation_snapshot):
    raw = plan(navigation_step(), navigation_step(
        id="later", depends_on=["navigate"], target={"app": "Future application"},
    ))
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Continue into the next application", navigation_snapshot, [])
    assert result.steps[1].target.app == "Future application"
    assert result.steps[1].depends_on == ("navigate",)


@pytest.mark.parametrize("already_satisfied", [False, True])
def test_navigation_waiting_on_guard_or_already_complete_does_not_require_active_candidates(
    navigation_snapshot, already_satisfied,
):
    change = ({"postconditions": [{"kind": "exists", "target": {"name": "Menu"}}]}
              if already_satisfied else
              {"preconditions": [{"kind": "exists", "target": {"name": "Not yet present"}}]})
    raw = plan(navigation_step(target={"app": "Future application"}, **change))
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    assert planner.plan("Reach a panel", navigation_snapshot, []).steps[0].navigation


def test_incomplete_observation_does_not_offer_an_active_navigation_scope(navigation_snapshot):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(plan(navigation_step())))

    planner = planner_with(handler, profile="desktop")
    with pytest.raises(ValueError):
        planner.plan("Reach preferences", replace(navigation_snapshot, a11y_complete=False), [])
    state = json.loads(requests[0]["messages"][1]["content"][0]["text"])
    assert state["navigation_scopes"] == []
    assert planner.last_validation_error.startswith("step 1: active navigation scope has no usable")


@pytest.mark.parametrize("failure_count", [1, 2])
def test_unavailable_validation_itself_prioritizes_current_visual_recovery(
    desktop_snapshot, failure_count,
):
    # Menus can expose readable names without usable bounds. A previous action
    # may already be marked verified, so recovery_required alone is insufficient.
    unavailable = AgentControl("d0", "Options", "button", role="menuitem", app="Example")
    current = replace(desktop_snapshot, controls=(unavailable,))
    raw = plan({
        "id": "open", "operation": "click", "target": {"name": "Options", "app": "Example"},
        "postconditions": [{"kind": "exists", "target": {"name": "Settings", "app": "Example"}}],
    })
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(raw))

    planner = planner_with(handler, profile="desktop")
    with pytest.raises(ValueError):
        planner.plan("Open options", current, [], max_actions=1)
    failure = {"validation": planner.last_validation_error, "error_type": "ValueError"}
    history = [{"verified": True, "recovery_required": False}, *[failure] * failure_count]
    raw = plan({"id": "visual", "operation": "visual_click", "label": "Options",
                "box": [100, 200, 300, 280]})
    result = planner.plan("Open options", current, history, max_actions=6)
    prompt = requests[-1]["messages"][0]["content"]
    state = json.loads(requests[-1]["messages"][1]["content"][0]["text"])
    assert state["snapshot"]["controls"][0]["operations"] == []
    assert prompt.startswith(f"RECOVERY PRIORITY: {failure_count} consecutive unavailable-target")
    assert "SINGLE-ACTION PLANNING" in prompt and "HYBRID PLANNING CONTRACT" not in prompt
    assert result.steps[0].operation == "visual_click" and not result.reusable
    assert result.steps[0].target is None and not result.steps[0].postconditions
    assert not result.completion_conditions and planner.last_raw is None


def test_unavailable_validation_allows_grounded_semantic_alternative_and_resets(navigation_snapshot):
    raw = plan({
        "id": "current", "operation": "click", "target": {"name": "Menu", "app": "Example"},
        "postconditions": [{"kind": "exists", "target": {"name": "Settings", "app": "Example"}}],
    })
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(json.dumps(raw))

    history = [{
        "validation": "step 1: immediate target has no current executable control; "
                      "use a currently executable target or one screenshot-grounded visual fallback",
    }]
    planner = planner_with(handler, profile="desktop")
    alternative = planner.plan("Open settings", navigation_snapshot, history)
    assert alternative.reusable and alternative.steps[0].target.name == "Menu"
    assert requests[-1]["messages"][0]["content"].startswith("RECOVERY PRIORITY:")
    history.append({"verified": True, "recovery_required": False})
    raw = plan(navigation_step())
    assert planner.plan("Continue", navigation_snapshot, history).steps[0].navigation
    assert requests[-1]["messages"][0]["content"].startswith("PLAN ONE MEANINGFUL DESKTOP STAGE.")


@pytest.mark.parametrize("observed_role, tag, input_type", [
    ("frame", "container", ""),
    ("dialog", "container", ""),
    ("checkbox", "input", "checkbox"),
    ("cell", "button", ""),
])
def test_unobserved_future_target_without_role_matches_actual_toolkit_node(
    navigation_snapshot, observed_role, tag, input_type,
):
    future_target = {"name": "Appearance", "app": "Example"}
    raw = plan(navigation_step(postconditions=[{"kind": "exists", "target": future_target}]))
    planner = planner_with(lambda _: response(json.dumps(raw)), profile="desktop")
    result = planner.plan("Reach appearance settings", navigation_snapshot, [])
    predicate = result.steps[0].postconditions[0]
    assert predicate.target.role == "" and not condition_holds(predicate, navigation_snapshot)
    future = replace(navigation_snapshot, controls=(AgentControl(
        "future", "Appearance", tag, role=observed_role, input_type=input_type, app="Example",
    ),))
    assert condition_holds(predicate, future)
    # A guessed role would prevent completion even though the named UI arrived.
    guessed = AgentCondition("exists", replace(predicate.target, role="menuitem"))
    assert not condition_holds(guessed, future)
