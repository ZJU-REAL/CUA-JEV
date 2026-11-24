"""Regression tests for navigation memory continuity."""

from cua_jev.desktop_navigation_memory import (
    NavigationAction,
    NavigationMemory,
    NavigationState,
    NavigationTransition,
)


def state(letter):
    # Same observed owner/scope, different exact native state.
    return NavigationState(letter * 64, "1" * 64, ("Demo", "Main", ""), (("current", "d" * 64),))


def transition(receipt, before, after):
    return NavigationTransition("reach", receipt, before, after,
                                NavigationAction("d" * 64, "left", 1), 1, True, ("unknown",))


def test_native_gap_with_same_owner_and_scope_cannot_mint_exhausted_edges_or_witness():
    a, b, c = state("a"), state("b"), state("c")
    memory = NavigationMemory(6)
    memory.reset(1)
    assert memory.record(transition("first", b, a), final_verified=True, cycle=False) is None
    witness = memory.record(transition("gap-return", c, b), final_verified=False, cycle=True)
    assert witness is None and memory.exhausted == set()
    assert memory.pending is None and memory.used is False
    assert len(memory.records) == 2  # Actual but disconnected input records remain historical.
    assert [r.final_verified for r in memory.records] == [True, False]
    assert all(e["binding"] == "historical" for e in memory.public()["edges"])


def test_contiguous_exact_return_with_trace_shape_b_to_a_to_b_still_qualifies():
    a, b = state("a"), state("b")
    memory = NavigationMemory(6)
    memory.reset(1)
    assert memory.record(transition("first", b, a), final_verified=True, cycle=False) is None
    witness = memory.record(transition("real-return", a, b), final_verified=False, cycle=True)
    assert witness is not None and witness.expected.same_observable_state(b)
    assert len(memory.exhausted) == 2 and memory.used is False
    memory.request(witness)
    assert memory.used is True and memory.pending is witness
    assert [r.final_verified for r in memory.records] == [True, False]


def test_longer_walk_with_an_unexplained_middle_gap_cannot_qualify_by_endpoint_equality():
    a, b, c, d = (state(letter) for letter in "abcd")
    memory = NavigationMemory(6)
    memory.reset(1)
    memory.record(transition("first", b, a), final_verified=True, cycle=False)
    memory.record(transition("second", a, c), final_verified=True, cycle=False)
    assert memory.record(transition("gap-return", d, b), final_verified=False, cycle=True) is None
    assert memory.exhausted == set() and not memory.used
    assert len(memory.records) == 3
