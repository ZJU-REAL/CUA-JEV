"""Buffer HTTP model JSON exchanges; persist outside the timed agent loop."""

from __future__ import annotations

import json
import os
import time


class ModelExchangeBuffer:
    def __init__(self):
        self.records = []
        self.pending = {}

    def attach(self, client):
        client.event_hooks["request"].append(self.request)
        client.event_hooks["response"].append(self.response)

    def request(self, request):
        record = {"method": request.method, "request": json.loads(request.content),
                  "response": None, "status_code": None}
        self.records.append(record)
        self.pending[id(request)] = (record, time.perf_counter())

    def response(self, response):
        pending = self.pending.pop(id(response.request), None)
        if pending is None:
            return
        record, began = pending
        response.read()
        record["http_wall_ms"] = (time.perf_counter() - began) * 1000
        record["status_code"] = response.status_code
        try:
            record["response"] = response.json()
        except ValueError:
            record["response"] = {"non_json_response": True, "bytes": len(response.content)}

    def save(self, path):
        encoded = json.dumps({"kind": "model_http_exchanges", "records": self.records},
                             ensure_ascii=False, indent=2, allow_nan=False)
        for name in ("TYPESAFE_API_KEY", "CUA_JEV_MODEL_API_KEY", "CUA_JEV_PLANNER_API_KEY"):
            key = os.getenv(name)
            if key:
                encoded = encoded.replace(key, "[REDACTED]")
        path.write_text(encoded + "\n", encoding="utf-8")
