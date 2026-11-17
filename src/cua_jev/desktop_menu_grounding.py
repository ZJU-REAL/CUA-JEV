"""Pair native menu actions with text regions from the same screenshot.

OCR supplies geometry only. Names, ownership and pointer capability come from
the recorded native menu, and ordinary input preflight remains mandatory.
"""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, replace

from .agent_types import AgentControl, AgentSnapshot
from .desktop_semantics import valid_desktop_box

MENU_GROUNDING_VERSION = "native_menu_text_v1"


@dataclass(frozen=True)
class ImageWord:
    text: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageWords:
    image_sha256: str
    size: tuple[int, int]
    lines: tuple[tuple[ImageWord, ...], ...]


@dataclass(frozen=True)
class MenuItem:
    label: str
    app: str
    window: str
    ancestors: tuple[str, ...]
    window_box: tuple[int, int, int, int]
    menu_bottom: int
    order: int
    can_click: bool
    needs_box: bool


def _canonical(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _distance(label: str, text: str) -> int | None:
    label, text = _canonical(label), _canonical(text)
    if not label or len(label) != len(text):
        return None
    count = sum(a != b for a, b in zip(label, text, strict=True))
    return count if count == 0 or (len(label) >= 8 and count == 1) else None


def native_menu_items(raw: object, snapshot: AgentSnapshot) -> tuple[MenuItem, ...]:
    """Retain direct menu leaves in one focused, natively enabled active window."""
    from .osworld_backend import _component_bounds, _local_name, _native_state, _protocol_attribute

    if (not snapshot.a11y_complete or snapshot.image is None or not isinstance(raw, str)
            or not raw or len(raw) > 2_000_000 or "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper()):
        return ()
    try:
        root = ET.fromstring(raw)
    except (ET.ParseError, ValueError):
        return ()
    nodes = list(root.iter())
    if len(nodes) > 6000:
        return ()
    parents = {child: node for node in nodes for child in node}

    def path(node):
        result = [node]
        while node in parents:
            node = parents[node]
            result.append(node)
        return result

    def owner(node):
        return next((n.get("name", "") for n in path(node) if _local_name(n.tag) == "application"), "")

    active_windows = Counter(owner(n) for n in nodes
                             if _local_name(n.tag) in {"frame", "window", "dialog"}
                             and _native_state(n, "active") is True)
    menus = []
    for node in nodes:
        if _local_name(node.tag) != "menu" or _native_state(node, "focused") is not True:
            continue
        chain = path(node)
        window = next((n for n in chain if _local_name(n.tag) in {"frame", "window", "dialog"}), None)
        bar = parents.get(node)
        if (window is None or bar is None or _local_name(bar.tag) != "menu-bar"
                or _native_state(window, "active") is not True):
            continue
        app = owner(node)
        if not app or active_windows[app] != 1:
            continue
        if any(_native_state(n, state) is False for n in chain for state in ("enabled", "sensitive")):
            continue
        visible_chain = chain[:chain.index(window) + 1]
        if any(_native_state(n, state) is not True for n in visible_chain
               for state in ("enabled", "sensitive", "visible", "showing")):
            continue
        menus.append((node, chain, window, app))
    if len(menus) != 1:
        return ()
    menu, chain, window, app = menus[0]
    title = window.get("name", "")
    menu_name = menu.get("name", "")
    if (not title.strip() or len(title) > 240 or len(app) > 240
            or not menu_name.strip() or len(menu_name) > 240):
        return ()
    roots = [c for c in snapshot.controls if c.active and c.app == app
             and c.role in {"frame", "window", "dialog"}]
    if len(roots) != 1 or roots[0].window != title or roots[0].disabled:
        return ()

    def bounds(node):
        native = {_local_name(k): v for k, v in node.attrib.items() if _protocol_attribute(k, "component")}
        all_fields = {_local_name(k): v for k, v in node.attrib.items()}
        found = _component_bounds(native, snapshot.image.region[2:])
        return (found[0] if found and _component_bounds(all_fields, snapshot.image.region[2:]) == found
                else None)

    window_box, menu_box = bounds(window), bounds(menu)
    if window_box is None or menu_box is None or not (
        window_box[0] <= menu_box[0] < menu_box[2] <= window_box[2]
        and window_box[1] <= menu_box[1] < menu_box[3] <= window_box[3]
    ):
        return ()
    ancestors = tuple(f"{_local_name(n.tag)}: {n.get('name', '')}" for n in reversed(chain)
                      if _local_name(n.tag) not in {"desktop", "desktop-frame", "application"}
                      and n.get("name", "").strip())[-8:]
    parent_controls = [c for c in snapshot.controls if c.app == app and c.window == title
                       and c.role == "menuitem" and c.label == menu_name and c.ancestors == ancestors[:-1]]
    if (len(parent_controls) != 1 or not parent_controls[0].focused or not parent_controls[0].active
            or parent_controls[0].disabled or not valid_desktop_box(parent_controls[0].box)):
        return ()
    labels = Counter(_canonical(n.get("name", "")) for n in menu if _local_name(n.tag) == "menu-item")
    result = []
    for order, node in enumerate(menu):
        if _local_name(node.tag) != "menu-item" or len(node):
            continue
        label = " ".join(node.get("name", "").split())
        if not label or len(label) > 240 or labels[_canonical(label)] != 1:
            continue
        can_click = all(_native_state(node, state) is True for state in ("enabled", "sensitive", "visible"))
        # A contradictory foreign field must not be hidden by namespace flattening.
        actions = [k for k in node.attrib if _local_name(k).endswith(("_desc", "_kb"))]
        if (len(actions) != 2 or any(not _protocol_attribute(k, "action") for k in actions)
                or {_local_name(k) for k in actions} != {"click_desc", "click_kb"}):
            can_click = False
        # Disabled/already-grounded labels still compete with approximate text.
        # Removing them would turn a glyph error into a different enabled action.
        needs_box = not any(c.app == app and c.window == title and c.role == "menuitem"
                            and _canonical(c.label) == _canonical(label) and c.box is not None
                            for c in snapshot.controls)
        result.append(MenuItem(label, app, title, ancestors, window_box, menu_box[3], order,
                               can_click, needs_box))
    return tuple(result)


def _valid_words(words: ImageWords, screenshot: bytes, snapshot: AgentSnapshot) -> bool:
    if (not isinstance(words, ImageWords) or snapshot.image is None
            or words.image_sha256 != hashlib.sha256(screenshot).hexdigest()
            or words.size != snapshot.image.region[2:] or not isinstance(words.lines, tuple)
            or len(words.lines) > 2000):
        return False
    width, height = words.size
    count = 0
    for line in words.lines:
        if not isinstance(line, tuple) or len(line) > 256:
            return False
        for word in line:
            count += 1
            if (count > 2000 or not isinstance(word, ImageWord) or not isinstance(word.text, str)
                    or not word.text.strip() or len(word.text) > 240 or "\x00" in word.text
                    or not isinstance(word.box, tuple) or len(word.box) != 4
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in word.box)):
                return False
            x0, y0, x1, y1 = word.box
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                return False
    return True


def ground_menu_items(
    snapshot: AgentSnapshot, screenshot: bytes, words: ImageWords, items: tuple[MenuItem, ...],
) -> AgentSnapshot:
    """Add only uniquely paired native labels; never turn arbitrary OCR into controls."""
    if (not snapshot.a11y_complete or not items or not _valid_words(words, screenshot, snapshot)):
        return snapshot
    observed: dict[MenuItem, list[tuple[tuple[float, ...], int]]] = {item: [] for item in items}
    for line in words.lines:
        for start in range(len(line)):
            for end in range(start + 1, min(start + 8, len(line)) + 1):
                span = line[start:end]
                if any(
                    right.box[0] < left.box[2] - 2
                    or right.box[0] - left.box[2] > 2 * max(left.box[3] - left.box[1],
                                                         right.box[3] - right.box[1])
                    or min(left.box[3], right.box[3]) - max(left.box[1], right.box[1])
                    < 0.4 * min(left.box[3] - left.box[1], right.box[3] - right.box[1])
                    for left, right in zip(span, span[1:], strict=False)
                ):
                    continue
                text = " ".join(w.text for w in span)
                scores = [(item, _distance(item.label, text)) for item in items]
                scores = [(item, score) for item, score in scores if score is not None]
                if not scores:
                    continue
                best = min(score for _, score in scores)
                matches = [item for item, score in scores if score == best]
                if len(matches) != 1:
                    continue
                item = matches[0]
                box = (min(w.box[0] for w in span), min(w.box[1] for w in span),
                       max(w.box[2] for w in span), max(w.box[3] for w in span))
                a, b, c, d = item.window_box
                if a <= box[0] < box[2] <= c and max(b, item.menu_bottom) <= box[1] < box[3] <= d:
                    observed[item].append((box, best))
    controls = list(snapshot.controls)
    width, height = words.size
    paired = []
    for item, matches in observed.items():
        # Overlapping tiles may report the same text with small rounding changes.
        # Treat only almost-identical regions as one; never choose a nearer region.
        distinct = []
        for box in sorted({box for box, _ in matches}):
            if not any(all(abs(a - b) <= 2 for a, b in zip(box, old, strict=True)) for old in distinct):
                distinct.append(box)
        if len(distinct) != 1:
            continue
        paired.append((item, distinct[0], min(score for _, score in matches)))
    for item, box, score in paired:
        if not item.can_click or not item.needs_box:
            continue
        # A nearby ordered peer anchors the popup. One pair member must match
        # exactly, so an isolated approximate word cannot supply pointer input.
        if not any(
            other != item and min(score, other_score) == 0
            and abs(box[0] - peer[0]) <= 2 * max(box[3] - box[1], peer[3] - peer[1]) + 8
            and (item.order - other.order) * (box[1] - peer[1]) > 0
            and abs(box[1] - peer[1]) <= 8 * max(box[3] - box[1], peer[3] - peer[1])
            for other, peer, other_score in paired
        ):
            continue
        normalized = tuple(round(v * 1000 / (width if i % 2 == 0 else height)) for i, v in enumerate(box))
        if not valid_desktop_box(normalized):
            continue
        existing = [i for i, c in enumerate(controls) if c.app == item.app and c.window == item.window
                    and c.role == "menuitem" and _canonical(c.label) == _canonical(item.label)]
        if existing:
            if len(existing) != 1:
                continue
            index = existing[0]
            control = controls[index]
            if (control.box is not None or control.disabled or control.editable or control.native is not None
                    or control.ancestors != item.ancestors):
                continue
            controls[index] = replace(control, box=normalized)
        else:
            identity = json.dumps((item.app, item.window, item.ancestors, item.label), ensure_ascii=True)
            ref = "m" + hashlib.sha256(identity.encode()).hexdigest()[:12]
            if any(c.ref == ref for c in controls):
                continue
            controls.append(AgentControl(ref, item.label, "button", role="menuitem", box=normalized,
                                         app=item.app, window=item.window,
                                         ancestors=item.ancestors, active=True))
    if len(controls) > 512:
        return replace(snapshot, a11y_complete=False)  # Never keep a truncated new-control prefix.
    return replace(snapshot, controls=tuple(controls)) if tuple(controls) != snapshot.controls else snapshot
