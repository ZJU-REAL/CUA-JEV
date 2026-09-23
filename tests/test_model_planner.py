import json

import httpx
import pytest
from test_open_browser import FakePage
from test_open_desktop import FakeWindow

from cua_jev.model_planner import ChatModelPlanner
from cua_jev.open_browser import BrowserPlan, BrowserSnapshot
from cua_jev.open_desktop import DesktopPlan, DesktopSnapshot


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
