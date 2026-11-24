"""System-1 decisions specific to the desktop (OSWorld-MCP) setting.

1. Tool shortlist: before planning, System 1 narrows the MCP catalog to the few tools
   relevant to the goal, so System 2 receives a handful of full schemas instead of all.
2. Channel choice: when System 2 offers both a GUI route and an MCP tool for one subgoal,
   System 1 chooses the channel from the current state.
Both decisions have a rule counterpart for the ablation (lexical ranking / prefer MCP).
"""

from __future__ import annotations

import re
import uuid

from .s1s2_planner import encode_controls


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if len(w) > 2}


def _tool_line(tool: dict) -> str:
    description = (tool.get("description") or "").strip()
    return f"{tool['name']}: {description.splitlines()[0][:160] if description else ''}"


class ToolChannels:
    def __init__(self, backend, *, s1: str, jev=None, shortlist_size: int = 10, choice_gate: float = 0.5):
        from .osworld_mcp import JointStep, ToolAction  # local: keeps browser runs free of MCP imports

        self._JointStep, self._ToolAction = JointStep, ToolAction
        self.backend, self.s1, self.jev = backend, s1, jev
        self.tools = list(backend.catalog.tools.values())
        self.shortlist_size, self.choice_gate = shortlist_size, choice_gate
        self.stats = {
            "shortlists": 0,
            "channel_mcp": 0,
            "channel_gui": 0,
            "mcp_calls": 0,
            "mcp_failures": 0,
            "mcp_fallback_gui": 0,
        }

    # ----- tool shortlist ------------------------------------------------------------------
    def shortlist(self, goal: str, snapshot) -> list[dict]:
        self.stats["shortlists"] += 1
        apps = " ".join(sorted({c.app for c in snapshot.controls if c.app})[:6])
        if self.s1 == "jev":
            questions = {
                f"t{i}": {
                    "type": "noul",
                    "instructions": f"Would the application tool {_tool_line(t)} directly help "
                    "accomplish the goal?",
                }
                for i, t in enumerate(self.tools)
            }
            answers = self.jev.ask(
                "tool_shortlist",
                {"goal": goal, "open_applications": apps, "window": snapshot.title},
                questions,
            )
            scores = [(answers[f"t{i}"].get("noul", 0.0), i) for i in range(len(self.tools))]
        else:
            goal_words = _words(goal + " " + apps)
            scores = [
                (len(goal_words & _words(_tool_line(t))) / (1 + len(_words(_tool_line(t)))), i)
                for i, t in enumerate(self.tools)
            ]
        ranked = sorted(scores, key=lambda x: -x[0])[: self.shortlist_size]
        return [self.tools[i] for score, i in ranked if score > 0]

    # ----- channel choice ------------------------------------------------------------------
    def choose(self, goal, sub, snapshot) -> str:
        if sub.op == "tool":
            return "mcp"
        if self.s1 != "jev":
            self.stats["channel_mcp"] += 1
            return "mcp"
        name, arguments = sub.tool
        criteria = {
            "mcp": f"call application tool {name} with {arguments[:300]}",
            "gui": f"{sub.op} {sub.target!r} {sub.value!r} in the visible GUI",
        }
        choice, confidence, _ = self.jev.choose(
            {
                "goal": goal,
                "subgoal": sub.describe(),
                "window": snapshot.title,
                "controls": encode_controls(snapshot, 80),
            },
            "Which channel completes this subgoal more reliably and directly in the current state?",
            criteria,
        )
        channel = choice if confidence >= self.choice_gate else "mcp"
        self.stats["channel_" + channel] += 1
        return channel

    # ----- MCP execution -------------------------------------------------------------------
    def execute(self, sub, snapshot):
        """Run the offered tool through the backend's catalog-validated path."""
        name, arguments = sub.tool
        step = self._JointStep(
            uuid.uuid4().hex[:8], "joint", options=(self._ToolAction(name, arguments, sub.describe()[:200]),)
        )
        candidates = self.backend.compile(step, snapshot, snapshot)
        if not candidates:
            return None, None, "tool not executable", None
        candidate = candidates[0]
        self.stats["mcp_calls"] += 1
        receipt = self.backend.execute(candidate, uuid.uuid4().hex, uuid.uuid4().hex)
        verification = self.backend.verify(candidate, receipt)
        result = (receipt.output or {}).get("tool_result") if isinstance(receipt.output, dict) else None
        texts = [c.get("text", "") for c in (result or {}).get("content", []) if isinstance(c, dict)]
        summary = f"{name} -> {'ok' if verification.passed else 'FAILED'}: {' '.join(texts)[:300]}"
        if not verification.passed:
            self.stats["mcp_failures"] += 1
        post = getattr(self.backend, "take_verified_post_snapshot", lambda *_: None)(candidate, receipt)
        return receipt, verification, summary, (post[0] if post else None)
