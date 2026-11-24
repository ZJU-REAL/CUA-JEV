"""Jev 1.14 System-1 decision client: choice, yes/no (noul) and box questions.

Images are sent only through the explicit TypeSafe opt-in header. Credentials,
URLs and response bodies never enter errors or metrics. Raw confidence is
recorded but is NOT a calibrated probability; callers use fixed gates only.
"""

from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass, field

import httpx

IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"
MIN_EDGE, MAX_EDGE = 256, 1568


class JevDecisionError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def prepare_image(jpeg: bytes) -> tuple[str, str]:
    """Return (format, base64) resized into Jev's accepted edge range; ratios are preserved."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as source:
        image = source.convert("RGB")
    width, height = image.size
    scale = 1.0
    if min(width, height) < MIN_EDGE:
        scale = MIN_EDGE / min(width, height)
    if max(width, height) * scale > MAX_EDGE:
        scale = MAX_EDGE / max(width, height)
    if scale != 1.0:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    out = io.BytesIO()
    image.save(out, "JPEG", quality=85)
    return "jpeg", base64.b64encode(out.getvalue()).decode("ascii")


@dataclass
class JevMeter:
    calls: int = 0
    failures: int = 0
    retries: int = 0
    wall_ms: float = 0.0
    image_calls: int = 0
    by_kind: dict = field(default_factory=dict)
    failure_codes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "retries": self.retries,
            "wall_ms": self.wall_ms,
            "image_calls": self.image_calls,
            "by_kind": dict(self.by_kind),
            "failure_codes": dict(self.failure_codes),
        }


class JevDecisionClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        api_url: str | None = None,
        timeout_s: float = 15.0,
        retries: int = 3,
        client: httpx.Client | None = None,
    ):
        self._key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self._key:
            raise JevDecisionError("missing_api_key")
        self.model = model or os.getenv("CUA_JEV_MODEL", "jev-1.14.0")
        self.api_url = api_url or os.getenv("CUA_JEV_API_URL", "https://api.typesafe.ai/v1/systemone")
        if not self.api_url.startswith("https://"):
            raise JevDecisionError("https_required")
        self.retries = retries
        # Environment proxies are honoured deliberately: Jev is reached through the
        # private proxy configured for the run. The VLM client never uses them.
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=min(5.0, timeout_s)), follow_redirects=False
        )
        self._owns = client is None
        self.meter = JevMeter()
        self.exchanges: list[dict] = []

    def close(self) -> None:
        if self._owns:
            self._client.close()

    def ask(self, kind: str, state: dict, questions: dict, *, image_jpeg: bytes | None = None) -> dict:
        body = {"model": self.model, "state": dict(state), "questions": questions}
        headers = {"Authorization": f"Bearer {self._key}"}
        if image_jpeg is not None:
            fmt, data = prepare_image(image_jpeg)
            body["state"]["screen"] = "screen-1"
            body["assets"] = {"screen-1": {"type": "image", "format": fmt, "data_base64": data}}
            headers[IMAGE_HEADER] = "true"
        began = time.perf_counter()
        code = "unknown"
        try:
            for attempt in range(self.retries + 1):
                if attempt:
                    self.meter.retries += 1
                    time.sleep(min(2.0, 0.5 * attempt))  # backoff; counted in Jev wall time
                try:
                    response = self._client.post(self.api_url, json=body, headers=headers)
                except httpx.TimeoutException:
                    code = "timeout"
                    continue
                except httpx.HTTPError:
                    code = "transport"
                    continue
                if response.status_code == 200:
                    data = response.json()
                    answers = data.get("answers")
                    if not isinstance(answers, dict) or set(answers) != set(questions):
                        code = "malformed_answers"
                        break
                    self.exchanges.append(
                        {
                            "kind": kind,
                            "questions": questions,
                            "image": image_jpeg is not None,
                            "answers": answers,
                            "usage": data.get("usage"),
                            "wall_ms": (time.perf_counter() - began) * 1000,
                        }
                    )
                    return answers
                code = f"http_{response.status_code}"
                if response.status_code not in {429, 500, 502, 503, 504}:
                    break
            self.meter.failures += 1
            self.meter.failure_codes[code] = self.meter.failure_codes.get(code, 0) + 1
            raise JevDecisionError(code)
        finally:
            self.meter.calls += 1
            self.meter.image_calls += image_jpeg is not None
            self.meter.by_kind[kind] = self.meter.by_kind.get(kind, 0) + 1
            self.meter.wall_ms += (time.perf_counter() - began) * 1000

    def choose(
        self, state: dict, instructions: str, criteria: dict[str, str], *, image_jpeg: bytes | None = None
    ) -> tuple[str, float, dict]:
        answer = self.ask(
            "choice",
            state,
            {"pick": {"type": "choice", "instructions": instructions, "criteria": criteria}},
            image_jpeg=image_jpeg,
        )["pick"]
        choice = answer.get("choice")
        if choice not in criteria:
            raise JevDecisionError("choice_outside_offer")
        return choice, float(answer.get("confidence", 0.0)), answer.get("probabilities") or {}

    def yes_no(self, state: dict, instructions: str, *, image_jpeg: bytes | None = None) -> float:
        answer = self.ask(
            "noul", state, {"q": {"type": "noul", "instructions": instructions}}, image_jpeg=image_jpeg
        )["q"]
        value = answer.get("noul")
        if type(value) not in (int, float) or not 0 <= value <= 1:
            raise JevDecisionError("malformed_noul")
        return float(value)

    def box(self, instructions: str, image_jpeg: bytes, state: dict | None = None) -> tuple[tuple, float]:
        answer = self.ask(
            "box",
            state or {},
            {"t": {"type": "box", "instructions": instructions + " in `screen`"}},
            image_jpeg=image_jpeg,
        )["t"]
        raw = answer.get("box") or answer
        try:
            edges = tuple(float(raw[k]) for k in ("left", "top", "right", "bottom"))
        except (KeyError, TypeError, ValueError):
            raise JevDecisionError("malformed_box") from None
        confidence = float(answer.get("confidence", 0.0))
        if not all(-0.05 <= v <= 1.05 for v in edges) or edges[0] > edges[2] or edges[1] > edges[3]:
            raise JevDecisionError("invalid_box")
        left, top, right, bottom = (min(1.0, max(0.0, v)) for v in edges)
        if right - left <= 0 and bottom - top <= 0 and left == 0 and top == 0:
            return (0.0, 0.0, 0.0, 0.0), 0.0  # "not found" answered as an empty box
        if right - left < 0.004:  # a point/line: widen slightly around it
            left, right = max(0.0, left - 0.002), min(1.0, right + 0.002)
        if bottom - top < 0.004:
            top, bottom = max(0.0, top - 0.002), min(1.0, bottom + 0.002)
        return (left, top, right, bottom), confidence
