import json
import time

import pytest

from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.jev_decisions import JevMeter
from cua_jev.models import ActionCandidate, ActionReceipt, Channel, Verification
from cua_jev.osworld_mcp import JointStep, ToolCatalog
from cua_jev.s1s2_desktop import ToolChannels
from cua_jev.s1s2_planner import S2Meter, S2Plan, Subgoal, parse_plan
from cua_jev.s1s2_runner import S1S2Config, S1S2Runner

TOOLS = [
    {
        "name": "libreoffice_calc.export_to_pdf",
        "description": "Export the current spreadsheet to PDF",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
    {
        "name": "vlc.play",
        "description": "Start playback in VLC",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def snap(*controls, text="desk"):
    return AgentSnapshot("desktop", "Calc", text, tuple(controls))


class Backend:
    def __init__(self, tool_ok=True, after=None):
        self.catalog = ToolCatalog(TOOLS)
        self.tool_ok, self.after = tool_ok, after
        self.current = snap(AgentControl("d0", "Export", "button"))
        self.executed = []

    def observe(self):
        return self.current

    def compile(self, step, planned, current):
        if isinstance(step, JointStep):
            option = step.options[0]
            return (
                ActionCandidate(
                    "m:1",
                    Channel.MCP,
                    "agent.mcp.call",
                    "x",
                    {"tool": option.name, "arguments": json.loads(option.arguments_json)},
                ),
            )
        return (
            ActionCandidate(
                f"{step.id}:dom",
                Channel.SCRIPT,
                "agent." + step.operation,
                "x",
                {"route": "dom", "ref": step.ref},
            ),
        )

    def execute(self, candidate, observation_id, decision_id):
        self.executed.append(candidate.channel)
        ok = candidate.channel != Channel.MCP or self.tool_ok
        if ok and self.after is not None:
            self.current = self.after
        now = time.time()
        return ActionReceipt(
            observation_id,
            decision_id,
            candidate.id,
            candidate.channel,
            candidate.capability,
            ok,
            now,
            now,
            {"tool_result": {"content": [{"type": "text", "text": "done"}]}},
            None if ok else "tool error",
        )

    def verify(self, candidate, receipt):
        return Verification(receipt.success, "agent.effect")


class Planner:
    def __init__(self, plans):
        self.plans, self.meter = list(plans), S2Meter()
        self.offered_tools, self.tool_feedback, self.failures = [], [], []

    def plan(self, goal, snapshot, history, *, failure=None, deadline=None):
        self.meter.calls += 1
        self.failures.append(failure)
        return self.plans.pop(0)


class Jev:
    def __init__(self, channel="gui"):
        self.channel, self.meter = channel, JevMeter()

    def choose(self, state, instructions, criteria, image_jpeg=None):
        self.meter.calls += 1
        return self.channel, 0.9, {}

    def ask(self, kind, state, questions, image_jpeg=None):
        self.meter.calls += 1
        return {k: {"noul": 0.9 if k == "t0" else 0.1} for k in questions}


ARGS = json.dumps({"path": "/tmp/a.pdf"}, sort_keys=True)


def test_rule_prefers_tool_and_succeeds_without_gui():
    done = snap(text="exported")
    backend = Backend(after=done)
    sub = Subgoal("click", "Export", tool=("libreoffice_calc.export_to_pdf", ARGS))
    channels = ToolChannels(backend, s1="rule")
    result = S1S2Runner(
        backend, Planner([S2Plan(False, (sub,))]), None, config=S1S2Config(s1="rule"), channels=channels
    ).run("export pdf", lambda s: s.text == "exported")
    assert result["status"] == "success" and backend.executed == [Channel.MCP]
    assert result["routes"]["mcp_tool"] == 1


def test_failed_tool_falls_back_to_gui_route():
    done = snap(text="exported")
    backend = Backend(tool_ok=False, after=done)
    sub = Subgoal("click", "Export", tool=("libreoffice_calc.export_to_pdf", ARGS))
    result = S1S2Runner(
        backend,
        Planner([S2Plan(False, (sub,))]),
        None,
        config=S1S2Config(s1="rule"),
        channels=ToolChannels(backend, s1="rule"),
    ).run("g", lambda s: s.text == "exported")
    assert backend.executed == [Channel.MCP, Channel.SCRIPT] and result["status"] == "success"
    assert result["channel_stats"]["mcp_fallback_gui"] == 1


def test_jev_can_choose_gui_over_offered_tool():
    done = snap(text="exported")
    backend = Backend(after=done)
    sub = Subgoal("click", "Export", tool=("libreoffice_calc.export_to_pdf", ARGS))
    jev = Jev(channel="gui")
    result = S1S2Runner(
        backend,
        Planner([S2Plan(False, (sub,))]),
        jev,
        config=S1S2Config(s1="jev"),
        channels=ToolChannels(backend, s1="jev", jev=jev),
    ).run("g", lambda s: s.text == "exported")
    assert backend.executed == [Channel.SCRIPT] and result["routes"]["gui_chosen_over_tool"] == 1


def test_shortlists_rank_relevant_tools():
    backend = Backend()
    assert (
        ToolChannels(backend, s1="rule").shortlist("export the spreadsheet to pdf", snap())[0]["name"]
        == "libreoffice_calc.export_to_pdf"
    )
    assert (
        ToolChannels(backend, s1="jev", jev=Jev()).shortlist("x", snap())[0]["name"]
        == "libreoffice_calc.export_to_pdf"
    )


def test_desktop_parse_validates_tool_arguments():
    s = snap(AgentControl("d0", "Export", "button"))
    catalog = ToolCatalog(TOOLS)
    plan = parse_plan(
        {
            "done": False,
            "subgoals": [
                {
                    "op": "tool",
                    "tool": {"name": "libreoffice_calc.export_to_pdf", "arguments": {"path": "/tmp/a.pdf"}},
                },
                {"op": "hotkey", "value": "ctrl+s"},
                {
                    "op": "click",
                    "target": "Export",
                    "tool": {"name": "libreoffice_calc.export_to_pdf", "arguments": {}},
                },
            ],
        },
        s,
        horizon=8,
        grounded=False,
        catalog=catalog,
        desktop=True,
    )
    assert plan.subgoals[0].tool[0] == "libreoffice_calc.export_to_pdf"
    assert plan.subgoals[2].tool is None  # invalid alternative dropped, GUI route kept
    with pytest.raises(ValueError):
        parse_plan(
            {"done": False, "subgoals": [{"op": "tool", "tool": {"name": "nope", "arguments": {}}}]},
            s,
            horizon=8,
            grounded=False,
            catalog=catalog,
            desktop=True,
        )


def test_joint_baseline_parser_drops_invalid_alternative_but_keeps_valid_one():
    from cua_jev.osworld_mcp import parse_joint_plan

    s = snap(AgentControl("d0", "Export", "button"))
    catalog = ToolCatalog(TOOLS)
    raw = {
        "summary": "x",
        "done": False,
        "steps": [
            {
                "id": "1",
                "depends_on": [],
                "options": [
                    {"kind": "gui", "action": {"operation": "double_click", "ref": "free text"}},
                    {
                        "kind": "mcp",
                        "tool": "libreoffice_calc.export_to_pdf",
                        "arguments": {"path": "/tmp/a.pdf"},
                        "label": "pdf",
                    },
                ],
            }
        ],
    }
    plan = parse_joint_plan(raw, s, catalog, 6)
    assert len(plan.steps[0].options) == 1
    raw["steps"][0]["options"] = raw["steps"][0]["options"][:1]
    with pytest.raises(ValueError):
        parse_joint_plan(raw, s, catalog, 6)


def test_joint_baseline_parser_accepts_integer_step_ids():
    from cua_jev.osworld_mcp import parse_joint_plan

    s = snap(AgentControl("d0", "Export", "button"))
    raw = {
        "summary": "x",
        "done": False,
        "steps": [
            {
                "id": 1,
                "depends_on": [],
                "options": [{"kind": "mcp", "tool": "vlc.play", "arguments": {}, "label": "play"}],
            },
            {
                "id": 2,
                "depends_on": [1],
                "options": [{"kind": "mcp", "tool": "vlc.play", "arguments": {}, "label": "play"}],
            },
        ],
    }
    plan = parse_joint_plan(raw, s, ToolCatalog(TOOLS), 6)
    assert [st.id for st in plan.steps] == ["1", "2"] and plan.steps[1].depends_on == ("1",)
