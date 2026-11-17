"""Freshly grounded Playwright actions on an already-owned synchronous page.

Both DOM and mouse routes are browser scripts, not physical desktop input.
Screenshot bytes stay in memory and never enter candidates or receipts.
"""

from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .agent_types import AgentControl, AgentSnapshot, AgentStep, valid_click_parameters
from .browser_transition import SETTLE_TRANSITION
from .executors.common import execute_with_receipt
from .models import ActionCandidate, ActionReceipt, Channel, Risk, Verification
from .vision import browser_viewport_image, changed_fraction

SELECTOR = (
    'a[href],button,input,textarea,select,[role="button"],[role="link"],'
    '[role="textbox"],[role="checkbox"],[contenteditable]:not([contenteditable="false"])'
)
EXTENDED_SELECTOR = SELECTOR + (
    ',summary,[role="tab"],[role="treeitem"],[role="menuitem"],'
    '[role="menuitemcheckbox"],[role="menuitemradio"],[role="option"],'
    '[role="switch"],[role="radio"],[role="combobox"],[onclick],'
    '[tabindex]:not([tabindex="-1"])'
)
# Generic visible UI hints, independent of any benchmark's DOM/classes. One
# bounded DOM pass also finds leaf links/buttons styled by application scripts
# without semantic tags. Their clickability is a hint, never proof of an effect.
_COLLECT = r"""request => {
    if (!request.extended) return {
        elements: [...document.querySelectorAll(request.selector)], complete: true};
    const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_ELEMENT);
    const elements = [];
    let node, examined = 0;
    while ((node = walker.nextNode())) {
        if (++examined > 5000) return {elements, complete: false};
        const semantic = node.matches(request.selector);
        const pointerLeaf = !semantic && !node.children.length && node.textContent.trim() &&
            !node.parentElement?.closest(request.selector) && getComputedStyle(node).cursor === 'pointer';
        if (semantic || pointerLeaf) elements.push(node);
    }
    return {elements, complete: true};
}"""
KEYS = frozenset(
    {
        "Enter",
        "Tab",
        "Escape",
        "Backspace",
        "Delete",
        "ArrowUp",
        "ArrowDown",
        "ArrowLeft",
        "ArrowRight",
        "Home",
        "End",
        "PageUp",
        "PageDown",
        "Space",
    }
)
DOM_OPERATIONS = frozenset({"click", "fill", "select", "check", "uncheck", "press"})
PAGE_OPERATIONS = frozenset({"scroll", "back", "wait"})
VISUAL_OPERATIONS = frozenset({"visual_click", "visual_fill"})

# This extractor is shared by observation and execution. Model output is only
# ever passed as data, and never interpolated into JavaScript or CSS selectors.
_EXTRACT = r"""el => {
    const tag = el.isContentEditable ? 'contenteditable' : el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    const rect = el.getBoundingClientRect();
    const style = getComputedStyle(el);
    if (['password', 'file', 'hidden'].includes(type) || el.hidden ||
        el.closest('[hidden],[inert],[aria-hidden="true"]') || !rect.width || !rect.height ||
        style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') {
        return null;
    }
    const labelled = (el.getAttribute('aria-labelledby') || '').split(/\s+/)
        .map(id => document.getElementById(id)?.innerText || '').join(' ').trim();
    const label = el.getAttribute('aria-label') || labelled ||
        [...(el.labels || [])].map(x => x.innerText).join(' ') ||
        el.getAttribute('placeholder') || (tag === 'select' ? '' : el.innerText) ||
        el.getAttribute('title') || el.getAttribute('name') ||
        (['submit','button','reset'].includes(type) ? el.value : '') || '';
    const role = el.getAttribute('role') || (el.isContentEditable ? 'textbox' :
        tag === 'button' ? 'button' : tag === 'a' ? 'link' :
        tag === 'select' ? 'combobox' : type === 'checkbox' ? 'checkbox' : type === 'radio' ? 'radio' :
        tag === 'input' || tag === 'textarea' ? 'textbox' : '');
    const box = [Math.max(0, Math.round(rect.left / innerWidth * 1000)),
        Math.max(0, Math.round(rect.top / innerHeight * 1000)),
        Math.min(1000, Math.round(rect.right / innerWidth * 1000)),
        Math.min(1000, Math.round(rect.bottom / innerHeight * 1000))];
    return {label: label.trim().replace(/\s+/g, ' ').slice(0, 240), tag, role,
        input_type: type, href: el.href || '',
        value: String(el.isContentEditable ? el.innerText : el.value ?? ''),
        disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
        checked: !!el.checked || el.getAttribute('aria-checked') === 'true',
        options: tag === 'select' ? [...el.options].map(o => o.label) : [],
        box: box[0] < box[2] && box[1] < box[3] ? box : null};
}"""
_ENRICH = r"""(el, item) => {
    if (!item) return null;
    item.expanded = el.getAttribute('aria-expanded') === 'true' ||
        (el.tagName === 'SUMMARY' && el.parentElement?.open === true);
    item.selected = el.getAttribute('aria-selected') === 'true';
    // Many sites visually label inputs without <label for> or ARIA. Use only
    // short, visible text in a bounded ancestor containing exactly one field.
    // Never bind by an ordinal, an invented label, or hidden application state.
    const fields = 'input,textarea,select,[contenteditable]:not([contenteditable="false"])';
    if (!item.label && el.matches(fields)) {
        let container = el.parentElement;
        for (let depth = 0; depth < 4 && container && container !== document.body; depth++) {
            if (container.querySelectorAll(fields).length !== 1) break;
            const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
            let textNode;
            const parts = [];
            while ((textNode = walker.nextNode())) {
                const parent = textNode.parentElement;
                const text = textNode.nodeValue.replace(/\s+/g, ' ').trim();
                if (!text || !parent || parent.closest(fields +
                    ',script,style,noscript,button,a,[hidden],[inert],[aria-hidden="true"]')) continue;
                const box = parent.getBoundingClientRect(), style = getComputedStyle(parent);
                if (box.width && box.height && style.visibility === 'visible' && style.display !== 'none')
                    parts.push(text);
            }
            const nearby = parts.join(' ');
            if (nearby && nearby.length <= 160) { item.label = 'Nearby: ' + nearby; break; }
            container = container.parentElement;
        }
    }
    return item;
}"""
_EXTENDED_EXTRACT = "el => (" + _ENRICH + ")(el, (" + _EXTRACT + ")(el))"
_CAPTURE = (
    """request => {
    const extract = ("""
    + _EXTRACT
    + "); const enrich = (" + _ENRICH + "); const collect = (" + _COLLECT
    + r""");
    const {elements, complete} = collect(request);
    const selector = request.selector;
    const controls = elements
        .map((el, i) => {
            const item = request.extended ? enrich(el, extract(el)) : extract(el);
            if (!item) return null;
            return {...item, ref: `e${i}`};
        })
        .filter(Boolean);
    const semantic_text = [];
    if (document.body) {
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
            const parent = node.parentElement;
            const text = (node.nodeValue || '').replace(/\s+/g, ' ').trim();
            if (!text || !parent ||
                parent.closest('script,style,noscript,[hidden],[inert],[aria-hidden="true"]')) continue;
            // Controls, their descendants, labels and options are represented
            // by control identities/options. Other text retains DOM order so
            // swapping a selected account with an archived one changes scope.
            if (parent.closest(selector + ',label,option')) continue;
            const rect = parent.getBoundingClientRect();
            const style = getComputedStyle(parent);
            if (!rect.width || !rect.height || style.display === 'none' ||
                style.visibility === 'hidden' || style.visibility === 'collapse') continue;
            semantic_text.push(text);
        }
    }
    return {url: location.href, title: document.title,
        text: (document.body?.innerText || '').slice(0, 24000), controls, semantic_text,
        scroll: [Math.round(scrollX), Math.round(scrollY)], a11y_complete: complete};
}"""
)
_FIND = (
    """request => {
    const extract = ("""
    + _EXTRACT
    + "); const enrich = (" + _ENRICH + "); const collect = (" + _COLLECT
    + """);
    const collected = collect(request);
    if (!collected.complete) throw new Error('candidate scan is incomplete');
    const matches = collected.elements.filter(el => {
        const c = request.extended ? enrich(el, extract(el)) : extract(el);
        return c && JSON.stringify([c.label,c.tag,c.role,c.input_type,c.href]) ===
            JSON.stringify(request.identity);
    });
    if (matches.length !== 1) throw new Error('target identity is missing or ambiguous');
    return matches[0];
}"""
)
_FOCUS = """() => {
    const el = document.activeElement;
    return el ? [el.tagName, el.id, el.getAttribute('name'), el.getAttribute('aria-label'),
        el.getAttribute('type'), [...document.querySelectorAll('*')].indexOf(el)] : [];
}"""
_ALLOWED_TARGET = """el => {
    if (!el) return false;
    const control = el.closest('input,textarea,select,button,[contenteditable],'
        + '[role="button"],[role="textbox"],[role="checkbox"],[role="link"]');
    const labelControl = el.closest('label')?.control;
    return [el, control, labelControl].filter(Boolean).every(target => {
        const type = (target.getAttribute('type') || '').toLowerCase();
        return !['password', 'file', 'hidden'].includes(type) && !target.disabled &&
            !target.closest('[hidden],[inert],[aria-hidden="true"],:disabled,[aria-disabled="true"]');
    });
}"""


def _origin(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("scope requires an HTTP(S) origin without credentials")
    port = parsed.port
    suffix = f":{port}" if port and port != {"http": 80, "https": 443}[parsed.scheme] else ""
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return f"{parsed.scheme}://{host.lower()}{suffix}"


def _image_key(snapshot: AgentSnapshot) -> str:
    return hashlib.sha256(snapshot.image.jpeg).hexdigest() if snapshot.image else ""


def _control(raw: dict) -> AgentControl:
    return AgentControl(
        **{**raw, "options": tuple(raw["options"]), "box": tuple(raw["box"]) if raw["box"] else None}
    )


@dataclass(frozen=True)
class _Binding:
    serialized: dict
    step: AgentStep
    snapshot: AgentSnapshot
    target: AgentControl | None
    route: str


class BrowserAgentBackend:
    """A scoped action backend; the caller owns the page and its browser lifetime."""

    fresh_preflight_guaranteed = True

    def __init__(self, page: Any, *, allowed_origins=None, pointer_routes: bool = True,
                 efficient_observations: bool = False, extended_controls: bool = False,
                 settle_transitions: bool = False):
        if any(type(value) is not bool for value in
               (efficient_observations, extended_controls, settle_transitions)):
            raise ValueError("observation options must be boolean")
        self.efficient_observations = efficient_observations
        self.extended_controls = extended_controls
        self.settle_transitions = settle_transitions
        self._capture_request = {
            "selector": EXTENDED_SELECTOR if extended_controls else SELECTOR,
            "extended": extended_controls,
        }
        self.observation_metrics = {"full": 0, "structured_only": 0, "wall_ms": 0.0}
        self.page = page
        self.pointer_routes = pointer_routes
        self._blank = page.url == "about:blank"
        origins = allowed_origins if allowed_origins is not None else ([] if self._blank else [page.url])
        if isinstance(origins, str):
            origins = [origins]
        self.allowed_origins = frozenset(
            [_origin(url) for url in origins] + ([] if self._blank else [_origin(page.url)])
        )
        self._issued: dict[str, _Binding] = {}
        self._verified: dict[str, tuple[ActionReceipt, Verification]] = {}
        self._post_snapshot = None
        self._blocked = 0
        self._closed = False
        self._route_handler = self._route_request
        self._popup_handler = self._reject_popup
        self._assert_scope(page.url)
        # A page route alone misses the first request of a newly opened popup.
        # The caller supplies an exclusively owned context for this backend.
        page.context.route("**/*", self._route_handler)
        page.route("**/*", self._route_handler)
        page.on("popup", self._popup_handler)

    def _allowed(self, url: str) -> bool:
        if self._blank and url == "about:blank":
            return True
        try:
            return _origin(url) in self.allowed_origins
        except ValueError:
            return False

    def _assert_scope(self, url: str) -> None:
        if not self._allowed(url):
            raise ValueError("browser navigation is outside allowed origins")

    def _route_request(self, route: Any) -> None:
        request = route.request
        if not request.is_navigation_request():
            route.continue_()
            return
        try:
            owned_main_frame = request.frame == self.page.main_frame
        except Exception:
            # The first popup request can precede creation of its frame.
            owned_main_frame = False
        if not owned_main_frame or not self._allowed(request.url):
            self._blocked += 1
            route.abort("blockedbyclient")
            return
        # Playwright routes only the first request in a redirect chain. Inspect
        # the response without following redirects so no unapproved hop loads.
        # All document redirects are unsupported, including same-origin ones.
        try:
            response = route.fetch(max_redirects=0, timeout=5000)
        except Exception:
            self._blocked += 1
            try:
                route.abort("failed")
            except Exception:
                # Closing a page can invalidate the route while fetch fails.
                pass
            return
        try:
            if 300 <= response.status < 400:
                self._blocked += 1
                route.abort("blockedbyclient")
            else:
                route.fulfill(response=response)
        finally:
            response.dispose()

    def _reject_popup(self, popup: Any) -> None:
        self._blocked += 1
        popup.close()

    def observe(self) -> AgentSnapshot:
        return self._capture(include_image=True)

    def _capture(self, *, include_image: bool) -> AgentSnapshot:
        self._post_snapshot = None
        if self._closed:
            raise RuntimeError("browser backend is closed")
        began = time.perf_counter()
        self.observation_metrics["full" if include_image else "structured_only"] += 1
        try:
            return self._capture_snapshot(include_image=include_image)
        finally:
            self.observation_metrics["wall_ms"] += (time.perf_counter() - began) * 1000

    def _capture_snapshot(self, *, include_image: bool) -> AgentSnapshot:
        self._assert_scope(self.page.url)
        raw = self.page.evaluate(_CAPTURE, self._capture_request)
        image = browser_viewport_image(self.page) if include_image else None
        self._assert_scope(raw["url"])
        # If navigation raced with capture, do not pair old DOM with new pixels.
        if self.page.url != raw["url"]:
            raise ValueError("page changed while capturing observation")
        result = AgentSnapshot(
            url=raw["url"],
            title=raw["title"],
            text=raw["text"],
            controls=tuple(_control(item) for item in raw["controls"]),
            image=image,
            scroll=tuple(raw["scroll"]),
            semantic_text=tuple(raw["semantic_text"]),
            a11y_complete=raw["a11y_complete"],
        )
        return result

    @staticmethod
    def _unique(identity: tuple, snapshot: AgentSnapshot) -> AgentControl | None:
        matches = [control for control in snapshot.controls if control.identity() == identity]
        return matches[0] if len(matches) == 1 else None

    def compile(
        self, step: AgentStep, planned: AgentSnapshot, current: AgentSnapshot
    ) -> tuple[ActionCandidate, ...]:
        op = step.operation
        if (
            not step.id
            or op not in DOM_OPERATIONS | PAGE_OPERATIONS | VISUAL_OPERATIONS
            or not self._allowed(current.url)
            or planned.url != current.url
            or not valid_click_parameters(op, step.button, step.click_count)
            or step.button != "left"
            or step.click_count != 1
        ):
            return ()
        if len(step.value) > 10000:
            return ()
        target = None
        page_press = op == "press" and not step.ref
        if op == "press" and step.value not in KEYS:
            return ()
        if op in DOM_OPERATIONS and not page_press:
            if step.box is not None:
                return ()
            old = [control for control in planned.controls if control.ref == step.ref]
            if len(old) != 1 or self._unique(old[0].identity(), planned) is None:
                return ()
            target = self._unique(old[0].identity(), current)
            if target is None or target.disabled or (target.href and not self._allowed(target.href)):
                return ()
            if op == "fill" and not self._editable(target):
                return ()
            if op == "select" and (target.tag != "select" or step.value not in target.options):
                return ()
            if op == "check" and target.input_type not in {"checkbox", "radio"}:
                return ()
            if op == "uncheck" and target.input_type != "checkbox":
                return ()
        elif op in VISUAL_OPERATIONS:
            if step.ref or not self._valid_box(step.box) or planned.image is None or current.image is None:
                return ()
            if (
                planned.fingerprint() != current.fingerprint()
                or _image_key(planned) != _image_key(current)
                or planned.image.region != current.image.region
            ):
                return ()
        elif step.ref or step.box is not None:
            return ()
        if op == "scroll" and step.value not in {"up", "down", "left", "right"}:
            return ()
        if op == "wait" and step.value not in {"", "250", "500", "1000"}:
            return ()
        if op == "back" and step.value:
            return ()
        routes = [
            "visual" if op in VISUAL_OPERATIONS else "page" if op in PAGE_OPERATIONS or page_press else "dom"
        ]
        if self.pointer_routes and op in {"click", "fill"} and target and target.box and current.image:
            routes.append("mouse")
        result = []
        for route in routes:
            arguments = {
                "operation": op,
                "route": route,
                "value": step.value,
                "ref": target.ref if target else "",
                "url": current.url,
                "fingerprint": current.fingerprint(),
            }
            if target:
                arguments["identity"] = list(target.identity())
            if route in {"mouse", "visual"}:
                arguments.update(
                    box=step.box if op in VISUAL_OPERATIONS else target.box,
                    image_hash=_image_key(current),
                    viewport=current.image.region if current.image else None,
                )
            candidate = ActionCandidate(
                id=f"{step.id}:{route}",
                channel=Channel.SCRIPT,
                capability=f"agent.browser.{op}",
                description=f"{op} {target.label if target else step.label or 'page'} via browser {route}",
                arguments=arguments,
                risk=Risk.LOCAL_WRITE,
                # Scope, freshness and unique identity are enforced against the
                # live page in execute(), immediately before the action.
                verifier="agent.effect",
                expected={"operation": op},
                intent=step.id,
            )
            self._issued[candidate.id] = _Binding(candidate.to_dict(), step, current, target, route)
            result.append(candidate)
        return tuple(result)

    @staticmethod
    def _editable(target: AgentControl) -> bool:
        return (
            target.tag in {"input", "textarea"}
            and target.input_type
            not in {
                "checkbox",
                "radio",
                "submit",
                "button",
                "reset",
                "range",
                "color",
                "password",
                "file",
                "hidden",
            }
        ) or target.tag == "contenteditable"

    @staticmethod
    def _valid_box(box: tuple | None) -> bool:
        return (
            box is not None
            and len(box) == 4
            and all(type(n) is int for n in box)
            and 0 <= box[0] < box[2] <= 1000
            and 0 <= box[1] < box[3] <= 1000
            and box[2] - box[0] >= 5
            and box[3] - box[1] >= 5
        )

    def _handle(self, target: AgentControl):
        handle = self.page.evaluate_handle(_FIND, {**self._capture_request, "identity": target.identity()})
        element = handle.as_element()
        if element is None:
            handle.dispose()
            raise ValueError("target is no longer an element")
        raw = element.evaluate(_EXTENDED_EXTRACT if self.extended_controls else _EXTRACT)
        if raw is None or _control({**raw, "ref": target.ref}) != target:
            element.dispose()
            raise ValueError("target changed immediately before execution")
        return element

    def _mouse(self, box: tuple, snapshot: AgentSnapshot, handle=None) -> None:
        if snapshot.image is None:
            raise ValueError("mouse action requires a screenshot")
        width, height = snapshot.image.region[2:]
        x, y = (box[0] + box[2]) * width / 2000, (box[1] + box[3]) * height / 2000
        if not self.page.evaluate(
            "p => { const allowed = (" + _ALLOWED_TARGET + "); "
            "return allowed(document.elementFromPoint(p.x,p.y)); }",
            {"x": x, "y": y},
        ):
            raise ValueError("mouse target is sensitive, hidden or disabled")
        if handle is not None and not handle.evaluate(
            "(el, p) => { const hit = document.elementFromPoint(p.x,p.y); "
            "return !!hit && (hit === el || el.contains(hit)); }",
            {"x": x, "y": y},
        ):
            raise ValueError("target is obscured at its click point")
        self.page.mouse.click(x, y)

    def _preflight_matches(self, binding, before):
        return before.fingerprint() == binding.snapshot.fingerprint()

    def _verify_observed_effect(self, binding, before, after, verification):
        return verification

    def execute(self, candidate: ActionCandidate, observation_id: str, decision_id: str) -> ActionReceipt:
        self._post_snapshot = None
        post = None
        verified = Verification(False, "agent.effect", {"reason": "execution failed"})

        def operation() -> dict:
            nonlocal verified, post
            preflight_started = time.perf_counter()
            binding = self._issued.get(candidate.id)
            if binding is None or candidate.to_dict() != binding.serialized:
                raise ValueError("candidate was not issued by this backend or was altered")
            # Typed DOM value edits rely on fresh DOM values and element handles,
            # not pixels. Pointer actions retain the full screenshot preflight.
            structured_only = (self.efficient_observations and binding.route == "dom"
                               and binding.step.operation in {"fill", "select", "check", "uncheck"})
            before = self._capture(include_image=False) if structured_only else self.observe()
            if not self._preflight_matches(binding, before):
                raise ValueError("stale browser snapshot; replan before execution")
            step, target = binding.step, binding.target
            if binding.route in {"mouse", "visual"} and (
                _image_key(before) != _image_key(binding.snapshot)
                or before.image.region != binding.snapshot.image.region
            ):
                raise ValueError("stale screenshot or viewport; replan before mouse action")
            handle = self._handle(target) if target else None
            focus_before = self.page.evaluate(_FOCUS)
            blocked = self._blocked
            action_started = time.perf_counter()
            try:
                self._perform(binding, before, handle)
            finally:
                if handle is not None:
                    handle.dispose()
            effect_started = time.perf_counter()
            if self._blocked != blocked:
                raise ValueError("browser scope blocked an origin, redirect, popup or frame navigation")
            transition = None
            if self.settle_transitions and step.operation in {
                "click", "visual_click", "press", "scroll", "back"
            }:
                transition = self.page.evaluate(SETTLE_TRANSITION)
            capture_started = time.perf_counter()
            after = self.observe()
            focus_after = self.page.evaluate(_FOCUS)
            if self._blocked != blocked:
                raise ValueError("browser scope blocked an origin, redirect, popup or frame navigation")
            effects = []
            current_target = self._unique(target.identity(), after) if target else None
            if step.operation in {"fill", "visual_fill"}:
                actual_value = (
                    current_target.value
                    if current_target
                    else self.page.evaluate(
                        "() => { const el = document.activeElement; return el?.isContentEditable "
                        "? el.innerText : el?.value; }"
                    )
                )
                if actual_value == step.value:
                    effects.append("value")
            elif step.operation == "select":
                if current_target:
                    selected = self._handle(current_target)
                    try:
                        if selected.evaluate("el => [...el.selectedOptions].map(o => o.label)") == [
                            step.value
                        ]:
                            effects.append("selected")
                    finally:
                        selected.dispose()
            elif step.operation in {"check", "uncheck"}:
                if current_target and current_target.checked == (step.operation == "check"):
                    effects.append("checked")
            elif step.operation == "wait":
                if effect_started - action_started >= int(step.value or "250") / 1000:
                    effects.append("time_elapsed")
            else:
                if before.scroll != after.scroll:
                    effects.append("scroll")
                if before.url != after.url:
                    effects.append("navigation")
                if before.fingerprint() != after.fingerprint():
                    effects.append("dom")
                if focus_before != focus_after:
                    effects.append("focus")
                if (
                    before.image
                    and after.image
                    and changed_fraction(before.image.sample, after.image.sample) > 0.001
                ):
                    effects.append("image")
            verified = Verification(
                bool(effects),
                "agent.effect",
                {"effects": effects, "operation": step.operation, "task_success": False},
            )
            verified = self._verify_observed_effect(binding, before, after, verified)
            output = {
                "operation": step.operation,
                "route": binding.route,
                "effects": effects,
                "before_fingerprint": before.fingerprint(),
                "after_fingerprint": after.fingerprint(),
                "url": after.url,
            }
            if transition is not None:
                output["transition_settling"] = transition
            if (
                verified.passed
                and current_target is not None
                and step.operation in {"fill", "select", "check", "uncheck"}
            ):
                output["postcondition"] = {
                    "identity": list(current_target.identity()),
                    "value": current_target.value,
                    "checked": current_target.checked,
                }
            output["timings_ms"] = {
                "preflight": (action_started - preflight_started) * 1000,
                "action": (effect_started - action_started) * 1000,
                "effect_check": (time.perf_counter() - effect_started) * 1000,
            }
            if verified.passed:
                post = (after, capture_started)
            return output

        receipt = execute_with_receipt(candidate, observation_id, decision_id, operation)
        self._verified[receipt.receipt_id] = (receipt, verified)
        if receipt.success and verified.passed and post is not None:
            self._post_snapshot = (deepcopy(candidate.to_dict()), receipt,
                                   deepcopy(receipt.to_dict()), *post)
        # An attempted action invalidates every alternative from that view.
        self._issued.clear()
        return receipt

    def take_verified_post_snapshot(self, candidate, receipt):
        """One-shot planning evidence; the next input still runs fresh preflight."""
        cached, self._post_snapshot = self._post_snapshot, None
        if cached is None:
            return None
        serialized, original, receipt_value, snapshot, began = cached
        if (self._closed or original is not receipt or receipt.to_dict() != receipt_value
                or candidate.to_dict() != serialized or not self.verify(candidate, receipt).passed
                or (snapshot.image is None
                    and not getattr(self, "verified_structured_post_observations", False))
                or not 0 <= time.perf_counter() - began <= 1.0):
            return None
        return snapshot, began

    def _perform(self, binding: _Binding, before: AgentSnapshot, handle) -> None:
        step = binding.step
        op = step.operation
        if op in {"visual_click", "visual_fill"} or binding.route == "mouse":
            box = step.box if op in VISUAL_OPERATIONS else binding.target.box
            visual_target = None
            try:
                if op == "visual_fill":
                    width, height = before.image.region[2:]
                    visual_target = self.page.evaluate_handle(
                        "p => document.elementFromPoint(p.x, p.y)?.closest("
                        "'input,textarea,[contenteditable]:not([contenteditable=\"false\"])') || null",
                        {"x": (box[0] + box[2]) * width / 2000, "y": (box[1] + box[3]) * height / 2000},
                    )
                    handle = visual_target.as_element()
                    raw = handle.evaluate(_EXTRACT) if handle else None
                    if raw is None or not self._editable(_control({**raw, "ref": "visual"})):
                        raise ValueError("visual fill point is not an allowed editable control")
                self._mouse(box, before, handle)
                if op in {"fill", "visual_fill"}:
                    if not handle.evaluate("el => el === document.activeElement"):
                        raise ValueError("mouse click focused a different control")
                    handle.fill(step.value, timeout=1500)
            finally:
                if visual_target is not None:
                    visual_target.dispose()
        elif op == "click":
            handle.click(timeout=1500)
        elif op == "fill":
            handle.fill(step.value, timeout=1500)
        elif op == "select":
            handle.select_option(label=step.value, timeout=1500)
        elif op in {"check", "uncheck"}:
            handle.set_checked(op == "check", timeout=1500)
        elif op == "press":
            if handle is not None:
                handle.press(step.value, timeout=1500)
            else:
                if not self.page.evaluate(
                    "() => { const allowed = (" + _ALLOWED_TARGET + "); "
                    "return allowed(document.activeElement); }"
                ):
                    raise ValueError("focused control is sensitive, hidden or disabled")
                self.page.keyboard.press(step.value)
        elif op == "scroll":
            dx, dy = {"up": (0, -600), "down": (0, 600), "left": (-600, 0), "right": (600, 0)}[step.value]
            self.page.evaluate(
                "delta => window.scrollBy({left: delta[0], top: delta[1], behavior: 'instant'})", [dx, dy]
            )
        elif op == "back":
            self.page.go_back(wait_until="domcontentloaded", timeout=5000)
            self._assert_scope(self.page.url)
        elif op == "wait":
            self.page.wait_for_timeout(int(step.value or "250"))

    def verify(self, candidate: ActionCandidate, receipt: ActionReceipt) -> Verification:
        known = self._verified.get(receipt.receipt_id)
        if (
            not receipt.success
            or known is None
            or known[0] != receipt
            or receipt.candidate_id != candidate.id
            or receipt.capability != candidate.capability
        ):
            return Verification(False, "agent.effect", {"reason": receipt.error or "unknown receipt"})
        return known[1]

    def close(self) -> None:
        self._post_snapshot = None
        if not self._closed:
            self.page.unroute("**/*", self._route_handler)
            self.page.context.unroute("**/*", self._route_handler)
            self.page.remove_listener("popup", self._popup_handler)
            self._issued.clear()
            self._verified.clear()
            self._closed = True
