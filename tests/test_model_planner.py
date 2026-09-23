import json

import httpx
import pytest
from test_open_browser import FakePage
from test_open_computer import environment
from test_open_desktop import FakeWindow

from cua_jev.model_planner import ChatModelPlanner
from cua_jev.open_browser import BrowserPlan, BrowserSnapshot
from cua_jev.open_computer import ComputerPlan
from cua_jev.open_desktop import DesktopPlan, DesktopSnapshot
from cua_jev.vision import WindowImage


def test_model_catalog_and_browser_plan_use_compatible_wire_shape():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer fake"
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "text-model"}, {"id": "vision-model"}]})
        body = json.loads(request.content)
        assert body["model"] == "text-model"
        assert body["messages"][1]["role"] == "user"
        assert json.loads(body["messages"][1]["content"])["snapshot"]["elements"][0]["ref"] == "e0"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "```json\n" + json.dumps({
                    "subgoal": "Open done",
                    "options": [{"ref": "e0", "operation": "click"}],
                    "success": {"kind": "url_contains", "value": "/done"},
                }) + "\n```"}}],
                "usage": {"prompt_tokens": 51, "completion_tokens": 20},
            },
        )

    planner = ChatModelPlanner(
        "https://model.example/v1", "text-model", api_key="fake",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert planner.list_models() == ("text-model", "vision-model")
    result = planner.plan("Finish", BrowserSnapshot.capture(FakePage()), [])
    assert isinstance(result, BrowserPlan)
    assert result.options[0].ref == "e0"
    assert planner.last_usage["prompt_tokens"] == 51
    assert len(requests) == 2


def test_desktop_plan_is_grounded_and_model_cannot_return_arbitrary_code():
    snapshot, _ = DesktopSnapshot.capture(FakeWindow())

    def handler(_request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({
                "subgoal": "Finish",
                "options": [{"ref": "c1", "operation": "click"}],
                "success": {"kind": "window_title_contains", "value": "Done"},
            })}}]},
        )

    planner = ChatModelPlanner(
        "http://127.0.0.1:8010/v1", "test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = planner.plan("Finish", snapshot, [])
    assert isinstance(result, DesktopPlan)
    assert result.options[0].ref == "c1"
    with pytest.raises(ValueError, match="insecure-HTTP"):
        ChatModelPlanner("http://model.example/v1", "model")


def test_cross_app_planner_sends_compact_refs_not_runtime_handles(tmp_path):
    task = environment(tmp_path)
    task.reset()
    task.observe(())
    assert task._snapshot is not None

    def handler(request):
        body = json.loads(request.content)
        state = json.loads(body["messages"][1]["content"])["snapshot"]
        assert state["browser"]["elements"][0]["ref"] == "b:e0"
        assert state["desktop"]["controls"][0]["ref"] == "d:c0"
        assert "window_handle" not in state["desktop"]
        assert "rectangle" not in state["desktop"]["controls"][0]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "subgoal": "Finish in the browser",
                "options": [{"ref": "b:e0", "operation": "click"}],
            })}}],
        })

    planner = ChatModelPlanner(
        "https://model.example/v1", "text-model", api_key="fake",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    result = planner.plan(task.goal, task._snapshot, [])
    assert isinstance(result, ComputerPlan)
    assert result.options[0].source == "b"
    task.close()


def test_bad_model_response_fails_closed():
    def handler(_request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "run this script"}}]})

    planner = ChatModelPlanner(
        "https://model.example/v1", "test-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ValueError, match="invalid planning response"):
        planner.plan("Finish", BrowserSnapshot.capture(FakePage()), [])


def test_model_gateway_bypasses_system_proxy_unless_opted_in(monkeypatch):
    original = httpx.Client
    options = []

    def factory(**kwargs):
        options.append(kwargs["trust_env"])
        return original(transport=httpx.MockTransport(lambda _request: httpx.Response(200)), **kwargs)

    monkeypatch.setattr("cua_jev.model_planner.httpx.Client", factory)
    direct = ChatModelPlanner("http://127.0.0.1:8010/v1", "model")
    proxied = ChatModelPlanner("http://127.0.0.1:8010/v1", "model", use_env_proxy=True)
    assert options == [False, True]
    direct.close()
    proxied.close()


def test_vision_request_sends_window_jpeg_and_validates_grounded_boxes():
    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "vision-model"
        content = body["messages"][1]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps({
                "summary": "A task dialog with a Finish button.",
                "visible_text": ["Task dialog", "Finish"],
                "targets": [{"label": "Finish", "box": [400, 400, 600, 600]}],
            })}}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 40},
        })

    planner = ChatModelPlanner(
        "https://model.example/v1", "vision-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    image = WindowImage(b"\xff\xd8\xff\xd9", bytes(64 * 64), (20, 30, 200, 200))
    scene = planner.perceive_scene("Finish", "Example", "", image)
    assert scene.summary == "A task dialog with a Finish button."
    assert scene.visible_text == ("Task dialog", "Finish")
    assert scene.targets[0].ref == "v0"
    assert scene.targets[0].box == (400, 400, 600, 600)
    assert planner.vision_calls == 1
    assert planner.usage_totals["prompt_tokens"] == 300


def test_vision_model_cannot_return_out_of_window_coordinates():
    planner = ChatModelPlanner(
        "https://model.example/v1", "vision-model",
        client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps({
                "targets": [{"label": "Outside", "box": [900, 100, 1200, 200]}],
            })}}]},
        ))),
    )
    image = WindowImage(b"\xff\xd8\xff\xd9", bytes(64 * 64), (0, 0, 200, 200))
    with pytest.raises(ValueError, match="invalid targets"):
        planner.perceive_window("Finish", "Example", "", image)


def test_vision_scene_extracts_json_wrapped_in_model_commentary():
    reply = "Observed the window.\n```json\n" + json.dumps({
        "summary": "A simple settings page.",
        "visible_text": ["Save"],
        "targets": [{"label": "Save", "box": [600, 600, 900, 850]}],
    }) + "\n```"
    planner = ChatModelPlanner(
        "https://model.example/v1", "vision-model",
        client=httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(
            200, json={"choices": [{"message": {"content": reply}}]},
        ))),
    )
    scene = planner.perceive_scene(
        "Save", "Settings", "", WindowImage(b"\xff\xd8\xff\xd9", bytes(64 * 64), (0, 0, 400, 240))
    )
    assert scene.summary == "A simple settings page."
    assert scene.targets[0].ref == "v0"
