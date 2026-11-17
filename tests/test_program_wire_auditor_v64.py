"""Synthetic wire/provenance mutations exercise the unchanged complete runtime ledger."""

import copy
import hashlib
import json
import re

import pytest
from test_program_aware_reuse_auditor_v60 import payload, valid_program

from scripts.analyze_stage_program_reuse_v60 import audit_trace as audit_original
from scripts.analyze_stage_program_reuse_v64 import (
    ANCESTOR_SHA256,
    CODEC_SOURCES,
    CODEC_VERSION,
    PROGRAM_FORMAT,
    SOURCE_STATUS,
    analyze,
    audit_trace,
    canonical,
    canonical_sha,
    independently_normalize_wire,
)


def source_files():
    return {**{name: str(index + 1) * 64 for index, name in enumerate(CODEC_SOURCES)},
            "scripts/analyze_stage_program_reuse_v60.py": ANCESTOR_SHA256}


def wire_events(**runtime):
    events = valid_program(**runtime)
    declaration = payload(events, "agent_program_declared")["declaration"]
    normalized = copy.deepcopy(declaration["source_declaration"])
    wire = copy.deepcopy(normalized)
    for member in wire["stages"]:
        del member["arguments"]["completion"]
    normalized_again, mapping = independently_normalize_wire(wire)
    assert normalized_again == normalized
    metadata = {
        "wire_declaration_json": canonical(wire),
        "wire_declaration_sha256": canonical_sha(wire),
        "normalized_declaration_json": canonical(normalized),
        "normalized_declaration_sha256": canonical_sha(normalized),
        "wire_stage_mapping": mapping,
        "codec_version": CODEC_VERSION,
        "codec_source_hashes": {name: source_files()[name] for name in CODEC_SOURCES},
    }
    declaration.update(copy.deepcopy(metadata), source_declaration_status=SOURCE_STATUS)
    transport = payload(events, "agent_model_output_transport")
    transport.update(response_transport=PROGRAM_FORMAT, requested_format=PROGRAM_FORMAT)
    raw_sha = hashlib.sha256(json.dumps(wire, indent=2).encode()).hexdigest()
    transport["declaration"].update(
        copy.deepcopy(metadata), argument_sha256=raw_sha, observed_wire_argument_sha256=raw_sha,
        normalized_argument_sha256=canonical_sha(normalized),
    )
    return events


def audit(events, **options):
    return audit_trace(events, source_files=source_files(), expected_format=PROGRAM_FORMAT, **options)


def assert_unknown(report):
    assert report["program_metric_status"] == "unknown"
    assert report["positive_input_program_continuations"] is None
    assert report["acceptance_satisfied"] is False


def test_original_v60_remains_unknown_new_audit_keeps_exact_legacy_and_original_events():
    events = wire_events()
    before = copy.deepcopy(events)
    old = audit_original(events)
    result = audit(events)
    assert old["program_metric_status"] == "unknown"
    assert result["original_v60_program_audit"] == old
    assert result["legacy_fixed_original"] == old["legacy_fixed_original"]
    assert result["program_metric_status"] == result["wire_normalization_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 1
    assert result["cross_stage_positive_input_program_continuations"] == 1
    assert result["derived_view_used"] is True and result["no_agent_plan_events_injected"] is True
    assert result["agent_episode_original"]["duration_ms"] == 420123.4
    assert events == before
    actual = payload(events, "agent_program_declared")
    exported = result["program_epochs"][0]["event_evidence"][0]["payload_original"]
    assert exported == actual
    assert exported["declaration"]["source_declaration_status"] == SOURCE_STATUS
    assert result["program_requests"][0]["transport"] == payload(events, "agent_model_output_transport")
    assert result["acceptance_satisfied"] is False


def test_raw_argument_hash_is_observed_and_never_substituted_with_canonical_wire_hash():
    events = wire_events()
    transport = payload(events, "agent_model_output_transport")["declaration"]
    assert transport["argument_sha256"] != transport["wire_declaration_sha256"]
    result = audit(events)
    assert result["program_metric_status"] == "complete"
    proof = result["wire_normalization_proofs"][0]
    observed_hash = proof["observed_original_argument_sha256_not_independently_reconstructed"]
    assert observed_hash == transport["argument_sha256"]
    assert proof["canonical_hash_is_not_raw_argument_hash"] is True


def test_mixed_original60_and_verified64_epochs_keep_actual_source_status_and_qualification():
    first, second = valid_program(), wire_events()
    first.pop()  # One real terminal episode remains after both complete programs.

    def new_origin(value):
        if type(value) is str:
            return re.sub(r"\b(receipt|decision|candidate|observation)-(\d+)\b",
                          lambda match: match[1] + "-" + str(int(match[2]) + 2), value)
        if type(value) is list:
            return [new_origin(item) for item in value]
        if type(value) is dict:
            return {key: new_origin(item) for key, item in value.items()}
        return value

    second = new_origin(second)
    for event in second:
        p = event["payload"]
        for key in ("call", "origin_planner_call", "planner_calls"):
            if key in p:
                p[key] = 2
        if "program_id" in p:
            p["program_id"] = "2:" + p["program_id"].split(":", 1)[1]
        if event["kind"] == "agent_step":
            p["step"] += 2
        if event["kind"] == "agent_episode":
            p["steps"] += 2
        if event["kind"].startswith("agent_program_") and "environment_steps" in p:
            p["environment_steps"] += 4
    events = first + second
    for index, event in enumerate(events):
        event["sequence"] = index
    before = copy.deepcopy(events)
    result = audit_trace(events, source_files=source_files())
    assert result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 2
    assert result["cross_stage_positive_input_program_continuations"] == 2
    old, new = result["program_epochs"]
    assert old["source_declaration_status_original"] == "original_canonical_json"
    assert old["normalized_source_independently_verified_from_wire"] is False
    assert new["source_declaration_status_original"] == SOURCE_STATUS
    assert new["normalized_source_independently_verified_from_wire"] is True
    assert [row["origin_planner_call"] for row in (old, new)] == [1, 2]
    assert [len(row["qualified"]) for row in (old, new)] == [2, 2]
    assert old["event_evidence"][0]["payload_original"] == payload(events, "agent_program_declared")
    assert new["event_evidence"][0]["payload_original"] == payload(events, "agent_program_declared", 1)
    assert result["legacy_fixed_original"] == audit_original(events)["legacy_fixed_original"]
    assert events == before


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("key,value", [
    ("completion", "stage"), ("completion", "goal"), ("completion", None),
    ("goal_conditions", []), ("goal_conditions", None),
])
def test_wire_forbids_explicit_member_completion_even_stage_or_empty_conditions(index, key, value):
    events = wire_events()
    wire = json.loads(payload(events, "agent_model_output_transport")["declaration"]["wire_declaration_json"])
    wire["stages"][index]["arguments"][key] = value
    with pytest.raises(ValueError, match="forbidden"):
        independently_normalize_wire(wire)
    for metadata in (payload(events, "agent_model_output_transport")["declaration"],
                     payload(events, "agent_program_declared")["declaration"]):
        metadata.update(wire_declaration_json=canonical(wire), wire_declaration_sha256=canonical_sha(wire))
    assert_unknown(audit(events))


@pytest.mark.parametrize("change", [
    "wire_text", "wire_hash", "normalized_text", "normalized_hash", "wire_reordered",
    "mapping_order", "mapping_boolean_index", "mapping_extra", "mapping_id", "mapping_tool",
    "codec_version", "codec_hash", "codec_hash_missing", "normalized_argument_hash",
    "raw_argument_hash_disagreement", "transport_format", "requested_format",
    "program_source", "program_normalized_hash", "program_original_status",
    "declaration_metadata_missing", "transport_metadata_missing",
])
def test_tampered_wire_normalization_mapping_and_provenance_fail_closed(change):
    events = wire_events()
    transport = payload(events, "agent_model_output_transport")
    metadata = transport["declaration"]
    declaration = payload(events, "agent_program_declared")["declaration"]
    if change == "wire_text":
        metadata["wire_declaration_json"] += " "
    elif change == "wire_hash":
        metadata["wire_declaration_sha256"] = "f" * 64
    elif change == "normalized_text":
        value = json.loads(metadata["normalized_declaration_json"])
        value["label"] += " altered"
        metadata["normalized_declaration_json"] = canonical(value)
    elif change == "normalized_hash":
        metadata["normalized_declaration_sha256"] = "f" * 64
    elif change == "wire_reordered":
        value = json.loads(metadata["wire_declaration_json"])
        value["stages"].reverse()
        metadata.update(wire_declaration_json=canonical(value), wire_declaration_sha256=canonical_sha(value))
    elif change == "mapping_order":
        metadata["wire_stage_mapping"].reverse()
    elif change == "mapping_boolean_index":
        metadata["wire_stage_mapping"][0]["index"] = False
    elif change == "mapping_extra":
        metadata["wire_stage_mapping"][0]["new_default"] = True
    elif change in {"mapping_id", "mapping_tool"}:
        metadata["wire_stage_mapping"][0][change.removeprefix("mapping_")] = "different"
    elif change == "codec_version":
        metadata["codec_version"] = "fake_version"
    elif change == "codec_hash":
        metadata["codec_source_hashes"][CODEC_SOURCES[0]] = "f" * 64
    elif change == "codec_hash_missing":
        metadata["codec_source_hashes"].pop(CODEC_SOURCES[0])
    elif change == "normalized_argument_hash":
        metadata["normalized_argument_sha256"] = "f" * 64
    elif change == "raw_argument_hash_disagreement":
        metadata["observed_wire_argument_sha256"] = "f" * 64
    elif change == "transport_format":
        transport["response_transport"] = "semantic_tools_v60"
    elif change == "requested_format":
        transport["requested_format"] = "semantic_tools_v62"
    elif change == "program_source":
        declaration["source_declaration"]["label"] = "changed"
    elif change == "program_normalized_hash":
        declaration["declaration_sha256"] = "f" * 64
    elif change == "program_original_status":
        declaration["source_declaration_status"] = "original_canonical_json"
    elif change == "declaration_metadata_missing":
        declaration.pop("wire_declaration_json")
    else:
        metadata.pop("wire_declaration_json")
    assert_unknown(audit(events))


def test_manifest_source_binding_required_not_only_self_reported_hashes():
    assert_unknown(audit_trace(wire_events(), expected_format=PROGRAM_FORMAT))
    hashes = source_files()
    hashes[CODEC_SOURCES[0]] = "f" * 64
    assert_unknown(audit_trace(wire_events(), source_files=hashes, expected_format=PROGRAM_FORMAT))


def test_wholesale_forged_legacy_status_is_rejected_when_selected_run_format_is_v64():
    events = valid_program()
    result = audit_trace(events, source_files=source_files(), expected_format=PROGRAM_FORMAT)
    assert_unknown(result)
    assert result["derived_view_used"] is False


@pytest.mark.parametrize("change", ["final_false", "no_input", "wrong_origin", "fake_activation",
                                  "no_admission", "receipt_identity", "logical_budget"])
def test_wire_proof_does_not_replace_original_positive_input_or_fresh_activation_rules(change):
    events = wire_events()
    if change == "final_false":
        payload(events, "agent_step")["verified"] = False
        payload(events, "verification")["passed"] = False
    elif change == "no_input":
        payload(events, "receipt")["output"]["environment_steps"] = 0
    elif change == "wrong_origin":
        payload(events, "agent_step", 1)["planner_calls"] = 2
    elif change == "fake_activation":
        payload(events, "agent_program_stage_activated", 1)["http_calls"] = 1
    elif change == "no_admission":
        events = [item for item in events if item["kind"] != "agent_program_stage_admitted"]
        for index, event in enumerate(events):
            event["sequence"] = index
    elif change == "receipt_identity":
        payload(events, "receipt")["decision_id"] = "unjoined"
    else:
        payload(events, "agent_program_stage_activated", 1)["remaining_actions"] = 6
    result = audit(events)
    assert result["wire_normalization_metric_status"] == "complete"
    assert_unknown(result)


@pytest.mark.parametrize("first_n,second_n", [(None, 1), (0, 1), (None, None), (0, 0)])
def test_complete_readonly_program_has_zero_positive_continuations(first_n, second_n):
    result = audit(wire_events(first_n=first_n, second_n=second_n))
    assert result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 0


def test_numeric_false_fill_needs_actual_verified_key_under_original_rules():
    result = audit(wire_events(numeric=True))
    assert result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 1
    assert result["program_epochs"][0]["raw_environment_inputs_complete"] == 5


@pytest.mark.parametrize("text", [
    '{"label":"x","label":"y","stages":[]}',
    '{"label":"x","stages":NaN}',
])
def test_noncanonical_or_nonfinite_json_never_normalized_into_a_witness(text):
    events = wire_events()
    payload(events, "agent_model_output_transport")["declaration"]["wire_declaration_json"] = text
    assert_unknown(audit(events))


def test_no_v64_program_preserves_existing_legacy_result():
    events = valid_program()
    result = audit_trace(events)
    original = audit_original(events)
    assert result["wire_normalization_metric_status"] == "not_applicable"
    assert result["program_metric_status"] == original["program_metric_status"]
    assert result["positive_input_program_continuations"] == original["positive_input_program_continuations"]


@pytest.mark.parametrize("closed,score,missing_trace", [(True, 0.0, False), (False, 1.0, False),
                                                       (True, 0.0, True)])
def test_terminal_artifact_consumer_keeps_failures_unknowns_and_all_original_costs(
    tmp_path, closed, score, missing_trace,
):
    runs = tmp_path / "runs"
    runs.mkdir()

    def write(name, value, *, lines=False):
        path = runs / name
        text = ("\n".join(json.dumps(item) for item in value) + "\n" if lines else json.dumps(value))
        raw = text.encode()
        path.write_bytes(raw)
        return {"path": "runs/" + name, "sha256": hashlib.sha256(raw).hexdigest()}

    source = write("source.json", {"files": source_files()})
    terminal = write("terminal.json", {"terminal": True, "active_child_may_be_running": False,
                                       "source_manifest_sha256": source["sha256"]})
    summary = write("summary.json", {"runs": [{
        "task": "generic/task", "mode": "hybrid", "official_score": score, "evaluation_status": "ok",
        "experiment_settings": {"desktop_output_format": PROGRAM_FORMAT},
        "runner": {"duration_ms": 420123.4, "planner_calls": 1, "status": "timeout"},
        "episode_wall_ms": 710123.4,
    }]})
    official = write("official.json", {"closed": closed, "official_score": score, "evaluation_status": "ok"})
    trace = write("trace.jsonl", wire_events(), lines=True)
    selected = {"id": "all_original_failure_kept", "task": "generic/task", "mode": "hybrid",
                "source_manifest": source, "summary": summary, "official": official,
                "trace": None if missing_trace else trace}
    result = analyze(tmp_path, {"kind": "root_selected_positive_input_reuse_evidence_v1",
                               "terminal": True, "terminal_evidence": terminal, "episodes": [selected]})
    row = result["episodes"][0]
    assert row["original_summary_row"]["runner"]["duration_ms"] == 420123.4
    assert row["original_summary_row"]["episode_wall_ms"] == 710123.4
    assert row["official_outcome_original"]["official_score"] == score
    assert row["official_outcome_original"]["closed"] is closed
    if closed and not missing_trace:
        assert row["program_metric_status"] == "complete"
        assert row["positive_input_program_continuations"] == 1
        assert result["complete_episodes"] == 1
    else:
        assert row["program_metric_status"] == "unknown"
        assert row["positive_input_program_continuations"] is None
    assert result["acceptance_satisfied"] is False
