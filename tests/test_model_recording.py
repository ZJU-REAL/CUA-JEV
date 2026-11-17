import json

import httpx

from cua_jev.model_recording import ModelExchangeBuffer


def test_model_exchanges_buffer_until_save_and_never_record_headers_or_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("CUA_JEV_MODEL_API_KEY", "SENSITIVE-TEST-KEY")
    recorder = ModelExchangeBuffer()
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"answer": "SENSITIVE-TEST-KEY", "usage": {"total_tokens": 7}},
    ))) as client:
        recorder.attach(client)
        response = client.post("https://fixture.test", headers={"Authorization": "Bearer PRIVATE"},
                               json={"messages": [{"role": "user", "content": "visible task"}]})
        assert response.json()["usage"]["total_tokens"] == 7
    assert not list(tmp_path.iterdir())
    path = tmp_path / "exchange.json"
    recorder.save(path)
    text = path.read_text()
    assert "SENSITIVE-TEST-KEY" not in text and "PRIVATE" not in text and "Authorization" not in text
    record = json.loads(text)["records"][0]
    assert record["status_code"] == 200 and record["response"]["answer"] == "[REDACTED]"
