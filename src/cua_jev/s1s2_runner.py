"""System 1 / System 2 runner (figure 2 design).

System 2 (VLM) states short semantic subgoals. For each subgoal the router
binds a target in the current UI: a unique exact match is executed directly;
several structured candidates go to System 1 (Jev choice over DOM/a11y text,
or a lexical rule in the ablation); a target that is absent from the structure
is located on the screenshot with a Jev box question. Uncertain or failed
bindings, execution failures, violated keep-constraints and unmet expected
effects escalate to System 2 with a screenshot for grounded recovery.
The backend keeps its own preflight, freshness and effect checks; final
success is decided only by the benchmark evaluator callback.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass

from .agent_types import AgentStep
from .jev_decisions import JevDecisionError
from .models import Verification
from .s1s2_planner import Subgoal, encode_controls
from .vision import changed_fraction

EDITABLE_EXCLUDED = {
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


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", (text or "").lower()).split())


def lexical(target: str, control) -> float:
    goal = norm(target)
    if not goal:
        return 0.0
    names = [
        norm(control.label),
        norm(control.value) if control.tag in {"a", "button"} else "",
        *(norm(a) for a in control.ancestors[:1]),
    ]
    best = 0.0
    tokens = set(goal.split())
    for index, name in enumerate(names):
        if not name:
            continue
        weight = 1.0 if index < 2 else 0.6
        if name == goal:
            score = 1.0
        elif goal in name or name in goal:
            score = 0.8 * min(len(name), len(goal)) / max(len(name), len(goal)) + 0.15
        else:
            other = set(name.split())
            score = 0.7 * len(tokens & other) / max(1, len(tokens | other))
        best = max(best, weight * score)
    return best


def editable(c) -> bool:
    return (
        (c.tag in {"input", "textarea"} and c.input_type not in EDITABLE_EXCLUDED)
        or c.tag == "contenteditable"
        or c.editable
    )


def compatible(sub: Subgoal, controls) -> list:
    op = sub.op
    out = []
    for c in controls:
        if c.disabled:
            continue
        if op == "fill" and not editable(c):
            continue
        if op == "select" and not (c.tag == "select" and sub.value in c.options):
            continue
        if (
            op == "check"
            and c.input_type not in {"checkbox", "radio"}
            and c.role not in {"checkbox", "radio"}
        ):
            continue
        if op == "uncheck" and c.input_type != "checkbox" and c.role != "checkbox":
            continue
        out.append(c)
    return out


@dataclass(frozen=True)
class S1S2Config:
    s1: str = "jev"  # "jev" or "rule"
    max_steps: int = 15
    max_s2_calls: int = 12
    timeout_s: float = 120.0
    max_failures: int = 3
    choice_gate: float = 0.5  # fixed, uncalibrated: below it System 1 defers
    box_gate: float = 0.15
    expect_gate: float = 0.25  # escalate only on a clear 'not observed'
    expect_escalates: bool = False  # see check(): unconfirmed expectations are noted, not escalated
    max_choice_options: int = 60
    rule_min_score: float = 0.5
    rule_margin: float = 0.15
    alt_recovery: bool = True  # Jev picks an S2-supplied alternative before escalating
    completion_check: bool = True  # Jev checks a completion claim on the screenshot (once)
    completion_gate: float = 0.2
    max_misses: int = 12  # consecutive binding/expectation misses before giving up
    effect_gate: float = 0.6  # Jev visual confirmation of an unseen click/type effect
    visual_expect_gate: float = 0.15  # desktop: a clear visual "no" returns to System 2

    def __post_init__(self):
        if self.s1 not in {"jev", "rule"}:
            raise ValueError("s1 must be jev or rule")


_BINDING_MISSES = {"rule_no_match", "rule_ambiguous", "s1_no_structured_target", "s1_box_low_confidence"}
_COUNTED = {"execution_failed", "not_executable", "tool_failed", "keep_violated", "stale_ref"}
_ALT_REASONS = _BINDING_MISSES | {"execution_failed", "not_executable", "tool_failed", "stale_ref"}


class _Escalate(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


class S1S2Runner:
    def __init__(self, backend, planner, jev=None, *, config: S1S2Config, trace=None, channels=None):
        if config.s1 == "jev" and jev is None:
            raise ValueError("jev mode requires a Jev client")
        self.backend, self.planner, self.jev, self.cfg, self.trace = backend, planner, jev, config, trace
        self.routes = Counter()
        self.escalations = Counter()
        self.channels = channels
        self.desktop = bool(getattr(planner, "desktop", False))
        self._shortlist_key = None
        self._last_reason = None

    # ----- helpers -------------------------------------------------------------------------
    def emit(self, kind, payload):
        if self.trace is not None:
            self.trace.append(kind, payload)

    def capture(self, *, image: bool, attempts: int = 3):
        """Observation with bounded retries: pages that are still navigating or rendering
        can make one capture fail; that is transient, not an agent decision."""
        capture = getattr(self.backend, "_capture", None)
        for attempt in range(attempts):
            try:
                return capture(include_image=image) if callable(capture) else self.backend.observe()
            except Exception:
                if attempt == attempts - 1:
                    raise
                self.routes["observe_retry"] += 1
                time.sleep(0.5)

    def observe(self):
        return (
            self.capture(image=False)
            if callable(getattr(self.backend, "_capture", None))
            else self.backend.observe()
        )

    def with_image(self, snapshot):
        return snapshot if snapshot.image is not None else self.capture(image=True)

    def _jev_state(self, goal, sub, snapshot):
        return {
            "goal": goal,
            "subgoal": sub.describe() + (f" (expected: {sub.expect})" if sub.expect else ""),
            "page_title": snapshot.title,
            "page_text": " ".join(snapshot.text.split())[:1500],
        }

    # ----- System 1 binding ----------------------------------------------------------------
    def bind(self, goal, sub: Subgoal, snapshot):
        """Return (AgentStep, snapshot used for compilation)."""
        sid = uuid.uuid4().hex[:8]
        op = sub.op
        if op in {"type", "hotkey", "key"}:
            self.routes["page_action"] += 1
            return AgentStep(
                sid, {"type": "type_text", "key": "key_press"}.get(op, op), value=sub.value
            ), snapshot
        if op in {"scroll", "back", "wait"} or (op == "press" and not sub.target and not sub.ref):
            self.routes["page_action"] += 1
            return AgentStep(sid, op, value=sub.value), snapshot
        if sub.ref:
            current = [c for c in snapshot.controls if c.identity() == sub.identity]
            if len(current) != 1:
                raise _Escalate(
                    "stale_ref",
                    f"the element referenced as {sub.ref} for {sub.describe()} is no longer uniquely present",
                )
            self.routes["s2_ref_rebound"] += 1
            return AgentStep(sid, op, ref=current[0].ref, value=sub.value), snapshot
        if sub.box is not None:
            snapshot = self.with_image(snapshot)
            if self.cfg.s1 == "jev" and sub.target and sub.op in {"click", "fill"}:
                edges, confidence = self.jev.box(f"the {sub.target}", snapshot.image.jpeg)
                self.emit(
                    "s1_box",
                    {
                        "subgoal": sub.describe(),
                        "edges": edges,
                        "confidence": confidence,
                        "s2_hint": list(sub.box),
                    },
                )
                if confidence >= self.cfg.box_gate:
                    self.routes["jev_box_over_s2"] += 1
                    return self._visual(sid, sub, tuple(int(round(v * 1000)) for v in edges)), snapshot
            self.routes["s2_grounded_box"] += 1
            return self._visual(sid, sub, sub.box), snapshot
        candidates = compatible(sub, snapshot.controls)
        exact = [c for c in candidates if norm(c.label) and norm(c.label) == norm(sub.target)]
        if len(exact) == 1:
            self.routes["rule_unique_exact"] += 1
            return AgentStep(sid, op, ref=exact[0].ref, value=sub.value), snapshot
        # A sole compatible field is used only if it also matches the stated target;
        # otherwise a different field (e.g. a password box absent from the safe
        # observation) could silently receive the value.
        if (
            len(candidates) == 1
            and op in {"fill", "select"}
            and lexical(sub.target, candidates[0]) >= self.cfg.rule_min_score
        ):
            self.routes["rule_unique_compatible"] += 1
            return AgentStep(sid, op, ref=candidates[0].ref, value=sub.value), snapshot
        pool = exact if len(exact) > 1 else candidates
        if self.cfg.s1 == "rule":
            return self._rule_choice(sid, sub, pool), snapshot
        return self._jev_bind(goal, sid, sub, pool, snapshot)

    def _rule_choice(self, sid, sub, pool):
        scored = sorted(((lexical(sub.target, c), c) for c in pool), key=lambda x: -x[0])
        if not scored or scored[0][0] < self.cfg.rule_min_score:
            raise _Escalate("rule_no_match", f"no control matches {sub.target!r}")
        if len(scored) > 1 and scored[0][0] - scored[1][0] < self.cfg.rule_margin:
            raise _Escalate("rule_ambiguous", f"several controls match {sub.target!r}")
        self.routes["rule_lexical"] += 1
        return AgentStep(sid, sub.op, ref=scored[0][1].ref, value=sub.value)

    def _jev_bind(self, goal, sid, sub, pool, snapshot):
        if pool:
            ranked = sorted(pool, key=lambda c: -lexical(sub.target, c))[: self.cfg.max_choice_options]
            criteria = {
                c.ref: line
                for c, line in zip(
                    ranked,
                    encode_controls(
                        type("S", (), {"controls": tuple(ranked)})(), limit=len(ranked)
                    ).splitlines(),
                    strict=False,
                )
            }
            criteria["none"] = (
                "None of the listed elements is the target (it is missing or only visible on screen)"
            )
            instructions = (
                "Which listed UI element should receive the subgoal action? Match the subgoal target "
                "text and context against element labels; choose none if no element fits."
            )
            choice, confidence, _ = self.jev.choose(
                self._jev_state(goal, sub, snapshot), instructions, criteria
            )
            self.emit(
                "s1_choice",
                {
                    "subgoal": sub.describe(),
                    "options": len(criteria),
                    "choice": choice,
                    "confidence": confidence,
                },
            )
            if choice != "none" and confidence >= self.cfg.choice_gate:
                self.routes["jev_choice_structured"] += 1
                return AgentStep(sid, sub.op, ref=choice, value=sub.value), snapshot
        if sub.op not in {"click", "fill"}:
            raise _Escalate("s1_no_structured_target", f"no confident structured target for {sub.describe()}")
        snapshot = self.with_image(snapshot)
        edges, confidence = self.jev.box(f"the {sub.target}", snapshot.image.jpeg)
        self.emit("s1_box", {"subgoal": sub.describe(), "edges": edges, "confidence": confidence})
        if confidence < self.cfg.box_gate:
            raise _Escalate("s1_box_low_confidence", f"could not locate {sub.target!r} on screen")
        box = tuple(int(round(v * 1000)) for v in edges)
        self.routes["jev_box_visual"] += 1
        return self._visual(sid, sub, box), snapshot

    @staticmethod
    def _visual(sid, sub, box):
        left, top, right, bottom = box
        if right - left < 5:
            left, right = max(0, left - 3), min(1000, right + 3)
        if bottom - top < 5:
            top, bottom = max(0, top - 3), min(1000, bottom + 3)
        op = "visual_fill" if sub.op == "fill" else "visual_click"
        return AgentStep(
            sid,
            op,
            value=sub.value if op == "visual_fill" else "",
            box=(left, top, right, bottom),
            label=sub.target or "target",
        )

    # ----- System 1 recovery and completion ------------------------------------------------
    def _choose_alternative(self, goal, sub, escalation, snapshot):
        """Jev picks one of System 2's alternatives for a failed subgoal (or none)."""
        criteria = {
            f"a{i}": alt.describe() + (f" [tool {alt.tool[0]}]" if alt.tool else "")
            for i, alt in enumerate(sub.alts)
        }
        criteria["none"] = "None of these alternatives fits the current state"
        image = self.with_image(snapshot).image.jpeg if self.desktop else None
        state = {
            "goal": goal,
            "failed_subgoal": sub.describe(),
            "failure": escalation.detail[:300],
            "window": snapshot.title,
            "controls": encode_controls(snapshot, 80),
        }
        choice, confidence, _ = self.jev.choose(
            state,
            "The subgoal failed. Which alternative most likely reaches the same intermediate state "
            "from the current screen?",
            criteria,
            image_jpeg=image,
        )
        self.emit(
            "s1_alternative",
            {"subgoal": sub.describe(), "choice": choice, "confidence": confidence, "options": len(criteria)},
        )
        if choice == "none" or confidence < self.cfg.choice_gate:
            return None
        return sub.alts[int(choice[1:])]

    def _completion_doubted(self, goal, history, snapshot) -> bool:
        image = self.with_image(snapshot).image.jpeg if self.desktop else None
        probability = self.jev.yes_no(
            {
                "goal": goal,
                "done_so_far": history[-8:],
                "window": snapshot.title,
                "controls": encode_controls(snapshot, 80),
            },
            "Is the whole goal already accomplished in the current state?",
            image_jpeg=image,
        )
        self.emit("s1_completion", {"probability": probability})
        return probability < self.cfg.completion_gate

    # ----- state check ---------------------------------------------------------------------
    def check(self, goal, sub, step, snapshot, kept, effects=()):
        for identity, value, checked in kept:
            matches = [c for c in snapshot.controls if c.identity() == identity]
            if len(matches) == 1 and (matches[0].value, matches[0].checked) != (value, checked):
                raise _Escalate("keep_violated", f"field {identity[0]!r} changed but must stay unchanged")
        # The executor already verified a concrete effect. Ask System 1 about the stated
        # expectation only when that effect is weak (no navigation and no DOM change).
        if (
            self.cfg.s1 == "jev"
            and sub.expect
            and step.operation in {"click", "visual_click", "press"}
            and not {"navigation", "dom"} & set(effects)
            and not (self.desktop and "accessibility" in effects)
        ):
            image = self.with_image(snapshot).image.jpeg if self.desktop else None
            probability = self.jev.yes_no(
                {
                    "goal": goal,
                    "last_action": sub.describe(),
                    "page_title": snapshot.title,
                    "page_text": " ".join(snapshot.text.split())[:2000],
                    "controls": encode_controls(snapshot, 120),
                },
                f"Is this now true of the current {'screen' if image else 'page'}: {sub.expect}?",
                image_jpeg=image,
            )
            self.emit("s1_expect", {"expect": sub.expect, "probability": probability})
            if probability < self.cfg.expect_gate:
                if self.cfg.expect_escalates:
                    raise _Escalate(
                        "expect_unmet",
                        f"after {sub.describe()} the expected effect '{sub.expect}' is not observed",
                    )
                # Model-written expectations are often imprecise (a menu opens instead of a
                # navigation). Record the doubt for System 2; later binding/execution failures
                # still escalate, so a truly wrong plan cannot run on unchecked.
                self.routes["expect_unconfirmed_continue"] += 1
                return f"expected '{sub.expect}' not confirmed"
        return None

    # ----- main loop -----------------------------------------------------------------------
    def run(self, goal, success):
        cfg = self.cfg
        start = time.perf_counter()
        deadline = start + cfg.timeout_s
        status, reason = "max_steps", "step budget exhausted"
        steps = failures = misses = 0
        history: list[str] = []
        queue: list[Subgoal] = []
        kept: list[tuple] = []
        failure: str | None = None
        acceptance_ms = 0.0
        final_answer = None
        infeasible = False
        completion_checked = False
        alt_tried: set[int] = set()
        stale_retries = 0
        rebind_retried = None
        snapshot = self.observe()
        try:
            while True:
                began = time.perf_counter()
                accepted = success is not None and bool(success(snapshot))
                acceptance_ms += (time.perf_counter() - began) * 1000
                if accepted:
                    status, reason = "success", "independent acceptance checks passed"
                    break
                if steps >= cfg.max_steps:
                    break
                if time.perf_counter() >= deadline:
                    status, reason = "timeout", "time budget exhausted"
                    break
                if failures >= cfg.max_failures or misses >= cfg.max_misses:
                    status, reason = "stuck", "consecutive failures"
                    break
                if not queue:
                    if self.planner.meter.calls >= cfg.max_s2_calls:
                        status, reason = "stuck", "System 2 call budget exhausted"
                        break
                    if (
                        self.desktop
                        or failure is not None
                        or not snapshot.controls
                        or not snapshot.a11y_complete
                    ):
                        snapshot = self.with_image(snapshot)
                    key = (snapshot.title,)
                    if self.channels is not None and (
                        not self.planner.offered_tools
                        or key != self._shortlist_key
                        or self._last_reason == "tool_failed"
                    ):
                        self._shortlist_key, self._last_reason = key, None
                        try:
                            self.planner.offered_tools = self.channels.shortlist(goal, snapshot)
                        except JevDecisionError as error:
                            self.escalations[f"jev_error_{error.code}"] += 1
                            self.planner.offered_tools = []
                    try:
                        plan = self.planner.plan(goal, snapshot, history, failure=failure, deadline=deadline)
                    except TimeoutError:
                        status, reason = "timeout", "System 2 request exceeded the episode budget"
                        break
                    except Exception as error:
                        failures += 1
                        failure = (
                            f"your previous answer was rejected ({type(error).__name__}: "
                            f"{str(error)[:160]}); answer with the exact JSON format"
                        )
                        self.escalations["s2_invalid_or_failed"] += 1
                        continue
                    self.emit(
                        "s2_plan",
                        {
                            "done": plan.done,
                            "subgoals": [asdict(s) for s in plan.subgoals],
                            "keep": list(plan.keep),
                            "recovery": failure is not None,
                        },
                    )
                    failure = None
                    if (
                        plan.done
                        and success is None
                        and not plan.infeasible
                        and self.cfg.s1 == "jev"
                        and self.cfg.completion_check
                        and not completion_checked
                    ):
                        completion_checked = True
                        snapshot = self.observe()
                        if self._completion_doubted(goal, history, snapshot):
                            self.routes["jev_completion_doubt"] += 1
                            failure = (
                                "System 1 does not see the goal accomplished in the current state; "
                                "inspect it and continue, or confirm done if it truly is"
                            )
                            history.append("completion claim doubted by System 1")
                            continue
                    if plan.done:
                        infeasible = plan.infeasible
                        if success is None:
                            # No live evaluator (WebArena/OSWorld): the agent stops on its own
                            # completion claim; the benchmark evaluates once afterwards.
                            status, reason, final_answer = "done", "System 2 reported completion", plan.answer
                        else:
                            status, reason = "stuck", "System 2 reported completion; evaluator did not accept"
                        break
                    queue = list(plan.subgoals)
                    by_ref = {c.ref: c for c in snapshot.controls}
                    kept = list(
                        {(by_ref[r].identity(), by_ref[r].value, by_ref[r].checked) for r in plan.keep}
                    )
                    continue
                sub = queue.pop(0)
                try:
                    if self.channels is not None and sub.tool is not None:
                        if self.channels.choose(goal, sub, snapshot) == "mcp":
                            self.routes["mcp_tool"] += 1
                            receipt, verification, summary, post = self.channels.execute(sub, snapshot)
                            self.planner.tool_feedback.append(summary)
                            if receipt is not None:
                                steps += 1
                            snapshot = post if post is not None else self.observe()
                            self.emit("s1s2_tool", {"subgoal": sub.describe(), "summary": summary})
                            if receipt is not None and receipt.success and verification.passed:
                                history.append(sub.describe() + f" via tool -> {summary[:120]}")
                                failures = misses = 0
                                continue
                            if sub.op == "tool":
                                raise _Escalate("tool_failed", summary)
                            self.channels.stats["mcp_fallback_gui"] += 1
                        else:
                            self.routes["gui_chosen_over_tool"] += 1
                    step, planned = self.bind(goal, sub, snapshot)
                    candidates = self.backend.compile(step, planned, planned)
                    if not candidates:
                        raise _Escalate(
                            "not_executable", f"{sub.describe()} cannot be executed on the current UI"
                        )
                    candidate = next(
                        (c for c in candidates if c.arguments.get("route") == "dom"), candidates[0]
                    )
                    receipt = self.backend.execute(candidate, uuid.uuid4().hex, uuid.uuid4().hex)
                    steps += 1
                    verification = self.backend.verify(candidate, receipt)
                    post = getattr(self.backend, "take_verified_post_snapshot", lambda *_: None)(
                        candidate, receipt
                    )
                    if (
                        not receipt.success
                        and receipt.error
                        and "stale" in receipt.error.splitlines()[0]
                        and stale_retries < 2
                    ):
                        # Preflight refused input because the page changed after binding (e.g. a grid
                        # still loading). Nothing was executed: re-observe and re-bind locally.
                        stale_retries += 1
                        steps -= 1
                        self.routes["stale_rebind"] += 1
                        time.sleep(0.3)
                        snapshot = self.observe()
                        queue.insert(0, sub)
                        continue
                    stale_retries = 0
                    late = None
                    if self.desktop and receipt.success and not verification.passed:
                        # Desktop menus/dialogs often render after the step response. One bounded
                        # late look at the screen before treating the input as ineffective.
                        time.sleep(0.6)
                        late = self.observe()
                        before_image, late_image = (
                            getattr(snapshot, "image", None),
                            getattr(late, "image", None),
                        )
                        if (
                            before_image is not None
                            and late_image is not None
                            and (
                                before_image.region != late_image.region
                                or changed_fraction(before_image.sample, late_image.sample) > 0.001
                            )
                        ):
                            self.routes["late_effect_confirmed"] += 1
                            verification = Verification(True, "s1.late_screen_effect", {"effects": ["image"]})
                        elif self.cfg.s1 == "jev" and late_image is not None:
                            question = (
                                f"The agent just performed: {sub.describe()}. Is its intended result "
                                f"in place on the current screen"
                                + (f" ({sub.expect})" if sub.expect else "")
                                + "?"
                            )
                            probability = self.jev.yes_no(
                                {"goal": goal, "window": late.title}, question, image_jpeg=late_image.jpeg
                            )
                            self.emit("s1_effect", {"subgoal": sub.describe(), "probability": probability})
                            if probability >= self.cfg.effect_gate:
                                self.routes["jev_effect_confirmed"] += 1
                                verification = Verification(
                                    True, "s1.jev_screen_effect", {"effects": ["jev"]}
                                )
                    snapshot = late if late is not None else post[0] if post else self.observe()
                    self.emit(
                        "s1s2_action",
                        {
                            "subgoal": sub.describe(),
                            "step": step.to_dict(),
                            "route": candidate.arguments.get("route"),
                            "success": receipt.success,
                            "effect": verification.passed,
                            "error": receipt.error,
                        },
                    )
                    if not receipt.success or not verification.passed:
                        first = (receipt.error or "").strip().splitlines()[0][:160] if receipt.error else ""
                        raise _Escalate(
                            "execution_failed",
                            f"{sub.describe()} had no verified effect" + (f" ({first})" if first else ""),
                        )
                    effects = (
                        (receipt.output or {}).get("effects", ()) if isinstance(receipt.output, dict) else ()
                    )
                    note = self.check(goal, sub, step, snapshot, kept, effects)
                    history.append(sub.describe() + (f" -> done, but {note}" if note else " -> ok"))
                    failures = misses = 0
                except _Escalate as escalation:
                    if (
                        self.cfg.s1 == "jev"
                        and self.cfg.alt_recovery
                        and sub.alts
                        and escalation.reason in _ALT_REASONS
                        and id(sub) not in alt_tried
                    ):
                        alt_tried.add(id(sub))
                        if steps and escalation.reason == "execution_failed":
                            snapshot = self.observe()
                        alternative = self._choose_alternative(goal, sub, escalation, snapshot)
                        if alternative is not None:
                            self.routes["jev_alt_recovery"] += 1
                            history.append(
                                sub.describe() + f" -> failed ({escalation.reason}); "
                                f"trying {alternative.describe()}"
                            )
                            queue.insert(0, alternative)
                            continue
                    if escalation.reason in _BINDING_MISSES and rebind_retried is not sub and steps:
                        # The target may simply not be rendered yet after the previous step
                        # (navigation, AJAX grid, opening menu). One cheap structured retry
                        # before spending a System 2 round trip.
                        rebind_retried = sub
                        self.routes["rebind_after_wait"] += 1
                        time.sleep(0.3)
                        snapshot = self.observe()
                        queue.insert(0, sub)
                        continue
                    self.escalations[escalation.reason] += 1
                    self._last_reason = escalation.reason
                    if escalation.reason in _COUNTED:
                        failures += 1
                    else:
                        misses += 1
                    failure = escalation.detail
                    history.append(sub.describe() + f" -> FAILED ({escalation.reason})")
                    queue.clear()
                    self.emit("s1s2_escalate", {"reason": escalation.reason, "detail": escalation.detail})
                    if steps and escalation.reason == "execution_failed":
                        snapshot = self.observe()
                except JevDecisionError as error:
                    self.escalations[f"jev_error_{error.code}"] += 1
                    failures += 1
                    failure = f"fast executor unavailable for {sub.describe()}"
                    queue.clear()
                except ValueError as error:
                    # Backend preflight rejected a stale or unsafe binding before input.
                    self.escalations["preflight_rejected"] += 1
                    failures += 1
                    failure = f"{sub.describe()} was rejected before input ({str(error)[:120]})"
                    queue.clear()
                    snapshot = self.observe()
        except Exception as error:
            status, reason = "error", type(error).__name__
            if os.environ.get("CUA_JEV_DEBUG_TRACEBACK") == "1":  # local debugging only
                import traceback

                self.emit("debug_traceback", {"text": traceback.format_exc()[-2000:]})
        return {
            "status": status,
            "reason": reason,
            "steps": steps,
            "duration_ms": (time.perf_counter() - start) * 1000,
            "acceptance_ms": acceptance_ms,
            "planner_calls": self.planner.meter.calls,
            "planner_wall_ms": self.planner.meter.wall_ms,
            "model_usage": self.planner.meter.usage,
            "requests_without_usage": self.planner.meter.missing_usage,
            "planner_failures": self.planner.meter.failures,
            "planner_images": self.planner.meter.images,
            "jev_calls": self.jev.meter.calls if self.jev else 0,
            "jev_wall_ms": self.jev.meter.wall_ms if self.jev else 0.0,
            "jev_metrics": self.jev.meter.to_dict() if self.jev else None,
            "routes": dict(self.routes),
            "escalations": dict(self.escalations),
            "final_answer": final_answer,
            "infeasible": infeasible,
            "channel_stats": dict(self.channels.stats) if self.channels is not None else None,
        }
