from test_s1s2_desktop import TOOLS, Backend, Planner, snap

from cua_jev.agent_types import AgentControl
from cua_jev.osworld_mcp import ToolCatalog, parse_joint_plan
from cua_jev.s1s2_planner import S2Plan, Subgoal, parse_plan
from cua_jev.s1s2_runner import S1S2Config, S1S2Runner


class AltJev:
    """Choice: picks the first alternative; yes/no: fixed probability."""

    def __init__(self, completion=0.9):
        from cua_jev.jev_decisions import JevMeter

        self.meter, self.completion, self.questions = JevMeter(), completion, []

    def choose(self, state, instructions, criteria, image_jpeg=None):
        self.questions.append(("choose", sorted(criteria)))
        if "a0" in criteria:
            return "a0", 0.9, {}
        return "none", 0.9, {}

    def yes_no(self, state, instructions, image_jpeg=None):
        self.questions.append(("yes_no", instructions))
        return self.completion

    def box(self, instructions, image_jpeg, state=None):
        return (0.0, 0.0, 0.1, 0.1), 0.0


def test_parse_keeps_valid_alternatives_and_infeasible_flag():
    s = snap(AgentControl("d0", "Export", "button"))
    plan = parse_plan(
        {
            "done": False,
            "subgoals": [
                {
                    "op": "click",
                    "target": "Image",
                    "alt": [
                        {"op": "hotkey", "value": "ctrl+m"},
                        {"op": "bogus"},
                        {"op": "type", "value": "x"},
                    ],
                }
            ],
        },
        s,
        horizon=8,
        grounded=False,
        catalog=ToolCatalog(TOOLS),
        desktop=True,
    )
    assert [a.op for a in plan.subgoals[0].alts] == ["hotkey", "type"]
    done = parse_plan(
        {"done": True, "subgoals": [], "infeasible": True},
        s,
        horizon=8,
        grounded=False,
        catalog=ToolCatalog(TOOLS),
        desktop=True,
    )
    assert done.done and done.infeasible
    web = parse_plan({"done": True, "subgoals": [], "infeasible": True}, s, horizon=8, grounded=False)
    assert not web.infeasible


def test_jev_switches_to_alternative_without_system2_call():
    backend = Backend(after=snap(text="saved"))
    sub = Subgoal("check", "Missing option", alts=(Subgoal("hotkey", value="ctrl+s"),))
    planner = Planner([S2Plan(False, (sub,))])
    jev = AltJev()
    result = S1S2Runner(backend, planner, jev, config=S1S2Config(s1="jev")).run(
        "save", lambda s: s.text == "saved"
    )
    assert result["status"] == "success" and planner.meter.calls == 1
    assert result["routes"]["jev_alt_recovery"] == 1


def test_completion_claim_doubted_once_then_accepted_and_infeasible_returned():
    backend = Backend()
    planner = Planner([S2Plan(True, ()), S2Plan(True, (), infeasible=True)])
    jev = AltJev(completion=0.05)
    result = S1S2Runner(backend, planner, jev, config=S1S2Config(s1="jev")).run("goal", None)
    assert planner.meter.calls == 2 and planner.failures[1] is not None
    assert result["status"] == "done" and result["infeasible"] is True
    assert result["routes"]["jev_completion_doubt"] == 1


def test_infeasible_declaration_skips_completion_check():
    planner = Planner([S2Plan(True, (), infeasible=True)])
    jev = AltJev(completion=0.0)
    result = S1S2Runner(Backend(), planner, jev, config=S1S2Config(s1="jev")).run("goal", None)
    assert result["infeasible"] is True and planner.meter.calls == 1 and jev.questions == []


def test_joint_baseline_parser_accepts_optional_infeasible():
    s = snap(AgentControl("d0", "Export", "button"))
    plan = parse_joint_plan(
        {"summary": "cannot", "done": True, "steps": [], "infeasible": True}, s, ToolCatalog(TOOLS), 6
    )
    assert plan.done


def test_plan_with_alternatives_is_traced_as_json(tmp_path):
    from cua_jev.trace import JsonlTrace

    backend = Backend(after=snap(text="saved"))
    sub = Subgoal("check", "Missing option", alts=(Subgoal("hotkey", value="ctrl+s"),))
    trace = JsonlTrace(tmp_path / "t.jsonl")
    result = S1S2Runner(
        backend, Planner([S2Plan(False, (sub,))]), AltJev(), config=S1S2Config(s1="jev"), trace=trace
    ).run("save", lambda s: s.text == "saved")
    assert result["status"] == "success"
    assert "ctrl+s" in (tmp_path / "t.jsonl").read_text()
