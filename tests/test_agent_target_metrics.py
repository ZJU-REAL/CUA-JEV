"""Action alternatives must not be reported as distinct UI controls."""

from dataclasses import replace

import pytest

from cua_jev.agent_runner import _distinct_bound_target_count
from cua_jev.agent_types import AgentControl, AgentSnapshot
from cua_jev.models import ActionCandidate, Channel


def candidate(control, *, button="left", click_count=1):
    return ActionCandidate(
        id=f"choice-{control.ref}-{button}-{click_count}", channel=Channel.GUI,
        capability="agent.osworld.click", description="A currently bound action",
        arguments={"bound_control": control.to_dict(), "button": button,
                   "click_count": click_count},
    )


def test_three_gestures_are_one_target_and_identical_names_still_count_separately():
    first = AgentControl("a", "File", "button", box=(0, 0, 10, 10))
    second = replace(first, ref="b", box=(20, 0, 30, 10))
    snapshot = AgentSnapshot("osworld://desktop", "Files", "", (first, second))
    offered = (candidate(first), candidate(first, button="right"),
               candidate(first, click_count=2), candidate(second))
    assert first.identity() == second.identity()
    assert len(offered) == 4
    assert _distinct_bound_target_count(offered, snapshot) == 2


@pytest.mark.parametrize("unknown", [None, {}, {"ref": "a"}, {"ref": "missing"}])
def test_missing_or_partial_binding_makes_mixed_decision_unknown(unknown):
    control = AgentControl("a", "Option", "button")
    snapshot = AgentSnapshot("osworld://desktop", "", "", (control,))
    action = candidate(control)
    unsupported = replace(action, arguments={"bound_control": unknown})
    assert _distinct_bound_target_count((action, unsupported), snapshot) is None


def test_stale_binding_and_ambiguous_snapshot_refs_do_not_produce_target_counts():
    control = AgentControl("a", "Option", "button", box=(0, 0, 10, 10))
    changed = replace(control, box=(20, 0, 30, 10))
    action = candidate(control)
    assert _distinct_bound_target_count(
        (action,), AgentSnapshot("osworld://desktop", "", "", (changed,)),
    ) is None
    assert _distinct_bound_target_count(
        (action,), AgentSnapshot("osworld://desktop", "", "", (control, changed)),
    ) is None
