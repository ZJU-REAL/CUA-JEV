import pytest

from scripts.audit_jev_choices import audit_trace


def events():
    return [
        {"kind": "observation", "payload": {"observation_id": "obs"}},
        {"kind": "candidates", "payload": {"items": [
            {"id": "a", "intent": "fill", "arguments": {"route": "dom"}},
            {"id": "b", "intent": "fill", "arguments": {"route": "mouse"}},
        ]}},
        {"kind": "decision", "payload": {"observation_id": "obs", "candidate_id": "b",
          "decision_id": "dec", "model": "jev-test", "confidence": 0.7, "latency_ms": 400}},
    ]


def test_audit_distinguishes_first_choice_and_route_choice_without_inventing_semantic_labels():
    rows = audit_trace(events(), model="jev-test")
    assert len(rows) == 1 and rows[0]["matches_first"] is False
    assert rows[0]["selected_rank_zero_based"] == 1
    assert rows[0]["distinct_intents"] == 1 and rows[0]["selected_route"] == "mouse"
    assert "correct" not in rows[0] and "calibrated_confidence" not in rows[0]
    assert audit_trace(events(), model="other-model") == []


def test_audit_does_not_attach_a_stale_candidate_pool_to_a_later_observation():
    items = events()
    items.insert(2, {"kind": "observation", "payload": {"observation_id": "later"}})
    with pytest.raises(ValueError, match="matching"):
        audit_trace(items, model="jev-test")
