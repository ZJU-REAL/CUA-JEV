"""Bounded, window-only image observations for optional VLM grounding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SAMPLE_EDGE = 64
MAX_VISUAL_TARGETS = 12
MAX_VISIBLE_TEXT = 24


@dataclass(frozen=True)
class WindowImage:
    jpeg: bytes
    sample: bytes
    region: tuple[int, int, int, int]  # left, top, width, height on the desktop

    def __post_init__(self) -> None:
        if not self.jpeg.startswith(b"\xff\xd8") or len(self.jpeg) > 2_000_000:
            raise ValueError("window screenshot must be a JPEG smaller than 2 MB")
        if len(self.sample) != SAMPLE_EDGE * SAMPLE_EDGE:
            raise ValueError("window screenshot sample must be 64x64 grayscale")
        if self.region[2] <= 0 or self.region[3] <= 0:
            raise ValueError("window screenshot region must have positive size")


@dataclass(frozen=True)
class VisualTarget:
    ref: str
    label: str
    box: tuple[int, int, int, int]  # normalized 0..1000 screenshot coordinates

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "label": self.label, "box": list(self.box)}

    @classmethod
    def from_response(cls, data: Any) -> tuple[VisualTarget, ...]:
        if not isinstance(data, dict) or not isinstance(data.get("targets"), list):
            raise ValueError("vision response must contain a targets list")
        raw = data["targets"]
        if len(raw) > MAX_VISUAL_TARGETS:
            raise ValueError("vision response contains too many targets")
        result: list[VisualTarget] = []
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError("vision target must be an object")
            label, box = item.get("label"), item.get("box")
            if not isinstance(label, str) or not 1 <= len(label.strip()) <= 100:
                raise ValueError("vision target label is invalid")
            if not isinstance(box, list) or len(box) != 4 or any(type(n) is not int for n in box):
                raise ValueError("vision target box must contain four integers")
            left, top, right, bottom = box
            if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
                raise ValueError("vision target box must fit the screenshot")
            if right - left < 5 or bottom - top < 5:
                raise ValueError("vision target box is too small to click safely")
            key = (label.strip(), (left, top, right, bottom))
            if key in seen:
                raise ValueError("vision response contains duplicate targets")
            seen.add(key)
            result.append(cls(f"v{index}", key[0], key[1]))
        return tuple(result)


@dataclass(frozen=True)
class VisualScene:
    """Bounded textual description of one screenshot; pixels never enter Jev state."""

    summary: str
    visible_text: tuple[str, ...]
    targets: tuple[VisualTarget, ...]

    @classmethod
    def from_response(cls, data: Any) -> VisualScene:
        if not isinstance(data, dict):
            raise ValueError("vision scene must be an object")
        summary = data.get("summary", "")
        lines = data.get("visible_text", [])
        if not isinstance(summary, str) or len(summary) > 1000:
            raise ValueError("vision scene summary is invalid")
        if not isinstance(lines, list) or len(lines) > MAX_VISIBLE_TEXT:
            raise ValueError("vision scene visible text is invalid")
        if any(not isinstance(line, str) or len(line) > 160 for line in lines):
            raise ValueError("vision scene visible text line is invalid")
        return cls(
            summary=summary.strip(),
            visible_text=tuple(line.strip() for line in lines if line.strip()),
            targets=VisualTarget.from_response(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "visible_text": list(self.visible_text),
            "targets": [target.to_dict() for target in self.targets],
        }


def changed_fraction(before: bytes, after: bytes, *, threshold: int = 20) -> float:
    """Fraction of coarse grayscale pixels with a meaningful change."""
    if len(before) != SAMPLE_EDGE * SAMPLE_EDGE or len(after) != len(before):
        raise ValueError("incompatible window image samples")
    return sum(abs(x - y) >= threshold for x, y in zip(before, after, strict=True)) / len(before)
