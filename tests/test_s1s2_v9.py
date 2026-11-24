import dataclasses

from test_s1s2_desktop import Backend, Planner, snap
from test_s1s2_jev_role import AltJev

from cua_jev.agent_types import AgentControl
from cua_jev.s1s2_planner import S2Plan, Subgoal
from cua_jev.s1s2_runner import S1S2Config, S1S2Runner


class Img:
    def __init__(self, v):
        self.jpeg, self.region, self.sample = b"x", (0, 0, 10, 10), bytes([v]) * 16


class DeskPlanner(Planner):
    desktop = True


class ImgBackend(Backend):
    def __init__(self, after=None):
        super().__init__(after=after)
        self.current = dataclasses.replace(snap(AgentControl("d0", "Export", "button")), image=Img(0))
        if after is not None:
            self.after = dataclasses.replace(after, image=Img(1))


def test_binding_misses_replan_without_ending_episode_early():
    miss = Subgoal("check", "Missing option")
    planner = Planner([S2Plan(False, (miss,))] * 5)
    result = S1S2Runner(
        Backend(), planner, AltJev(), config=S1S2Config(s1="jev", max_failures=1, max_misses=3)
    ).run("g", lambda s: False)
    assert result["status"] == "stuck" and planner.meter.calls == 3


def test_jev_relocates_target_before_using_s2_box():
    class BoxJev(AltJev):
        def box(self, instructions, image_jpeg, state=None):
            return (0.1, 0.1, 0.2, 0.2), 0.9

    backend = ImgBackend(after=snap(text="done"))
    sub = Subgoal("click", "OK", box=(500, 500, 520, 520))
    result = S1S2Runner(
        backend, DeskPlanner([S2Plan(False, (sub,))]), BoxJev(), config=S1S2Config(s1="jev")
    ).run("g", lambda s: s.text == "done")
    assert result["status"] == "success" and result["routes"]["jev_box_over_s2"] == 1


def test_jev_box_clamps_slight_overflow_and_reads_empty_box_as_not_found():
    import pytest

    from cua_jev.jev_decisions import JevDecisionClient, JevDecisionError

    client = JevDecisionClient.__new__(JevDecisionClient)
    answers = iter(
        [
            {"t": {"box": {"left": 0.5, "top": 0.2, "right": 1.01, "bottom": 0.2}, "confidence": 0.7}},
            {"t": {"box": {"left": 0, "top": 0, "right": 0, "bottom": 0}, "confidence": 0.4}},
            {"t": {"box": {"left": 0.6, "top": 0.2, "right": 0.5, "bottom": 0.3}, "confidence": 0.9}},
        ]
    )
    client.ask = lambda *a, **k: next(answers)
    edges, conf = client.box("x", b"")
    assert edges[2] == 1.0 and edges[3] > edges[1] and conf == 0.7
    assert client.box("x", b"") == ((0.0, 0.0, 0.0, 0.0), 0.0)
    with pytest.raises(JevDecisionError):
        client.box("x", b"")
