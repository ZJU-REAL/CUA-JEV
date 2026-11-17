"""Honest content provenance plus the unchanged complete positive-input ledger."""

import copy
import hashlib
import json
import re

import pytest
from test_program_aware_reuse_auditor_v60 import payload
from test_program_wire_auditor_v64 import source_files as source64_files
from test_program_wire_auditor_v64 import wire_events

from scripts.analyze_stage_program_reuse_v64 import audit_trace as audit64
from scripts.analyze_structured_program_reuse_v66 import (
    CONTENT_SOURCE,
    ORIGINAL_V64_SHA256,
    PROGRAM_FORMAT,
    TOOLS,
    analyze,
    audit_trace,
    canonical,
    canonical_sha,
)


def source_files():
    return {
        **source64_files(),
        CONTENT_SOURCE: "9" * 64,
        "scripts/analyze_stage_program_reuse_v64.py": ORIGINAL_V64_SHA256,
    }


def response_format():
    branches, definitions = [], {}
    for index, name in enumerate(TOOLS):
        key = f"v66_tool_{index}_arguments"
        definitions[key] = {"type": "object"}
        branches.append(
            {
                "type": "object",
                "properties": {
                    "tool": {"const": name},
                    "arguments": {"$ref": "#/$defs/" + key},
                },
                "required": ["tool", "arguments"],
                "additionalProperties": False,
            }
        )
    schema = {
        "type": "object",
        "properties": {
            "tool": {"type": "string", "enum": list(TOOLS)},
            "arguments": {"type": "object"},
        },
        "required": ["tool", "arguments"],
        "additionalProperties": False,
        "oneOf": branches,
        "$defs": definitions,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cua_jev_declaration_v66",
            "strict": True,
            "schema": schema,
        },
    }


def structured_events(**runtime):
    events = wire_events(**runtime)
    transport = payload(events, "agent_model_output_transport")
    internal = copy.deepcopy(transport["declaration"])
    arguments = json.loads(internal["wire_declaration_json"])
    internal.update(
        response_transport="semantic_tools_v64",
        requested_format="semantic_tools_v64",
        argument_sha256=canonical_sha(arguments),
        observed_wire_argument_sha256=canonical_sha(arguments),
    )
    content = {"tool": TOOLS[0], "arguments": arguments}
    raw = json.dumps(content, indent=2).encode("utf-8")
    response = response_format()
    mapped = {
        "response_transport": PROGRAM_FORMAT,
        "requested_format": PROGRAM_FORMAT,
        "phase": "normal_high_level",
        "transport_kind": "json_schema_content",
        "normal_json_schema": True,
        "provider_tool_call_count": 0,
        "tool_call_count": 0,
        "selected_tool": TOOLS[0],
        "tool_name": TOOLS[0],
        "offered_tool_names": list(TOOLS),
        "content_declaration_json": canonical(content),
        "content_declaration_sha256": canonical_sha(content),
        "observed_content_utf8_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_arguments_json": canonical(arguments),
        "canonical_arguments_sha256": canonical_sha(arguments),
        "argument_sha256": canonical_sha(arguments),
        "argument_hash_kind": "adapter_canonical_arguments",
        "content_codec_source_hashes": {CONTENT_SOURCE: source_files()[CONTENT_SOURCE]},
        "response_format_json": canonical(response),
        "response_format_sha256": canonical_sha(response),
        "schema_name": "cua_jev_declaration_v66",
        "declaration_schema_sha256": canonical_sha(response["json_schema"]["schema"]),
        "schema_namespace": "per_offered_tool_parameter_root_v66",
        "provider_schema_enforcement": "unverified",
        "parameter_schema_sha256": {
            name: canonical_sha(response["json_schema"]["schema"]["$defs"][f"v66_tool_{i}_arguments"])
            for i, name in enumerate(TOOLS)
        },
        "program_declaration": True,
        "internal_adapter": {"kind": "synthetic_legacy_mapper_payload_not_provider", "provenance": internal},
    }
    from scripts.analyze_stage_program_reuse_v64 import METADATA_KEYS

    mapped.update({key: copy.deepcopy(internal[key]) for key in METADATA_KEYS})
    transport.update(response_transport=PROGRAM_FORMAT, requested_format=PROGRAM_FORMAT, declaration=mapped)
    return events


def audit(events):
    return audit_trace(events, source_files=source_files(), expected_format=PROGRAM_FORMAT)


def assert_unknown(result):
    assert result["program_metric_status"] == "unknown"
    assert result["positive_input_program_continuations"] is None
    assert result["acceptance_satisfied"] is False


def test_full_valid_content_preserves_original64_unknown_and_original_trace_and_costs():
    events = structured_events()
    before = copy.deepcopy(events)
    result = audit(events)
    old = audit64(events, source_files=source_files(), expected_format=PROGRAM_FORMAT)
    assert old["program_metric_status"] == "unknown"
    assert result["original_v64_program_audit"] == old
    assert result["original_v60_program_audit"] == old["original_v60_program_audit"]
    assert result["structured_content_metric_status"] == result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 1
    assert result["cross_stage_positive_input_program_continuations"] == 1
    assert result["structured_content_derived_view_used"] is True
    assert result["no_provider_function_calls_invented"] is True
    assert result["agent_episode_original"]["duration_ms"] == 420123.4
    assert result["program_requests"][0]["transport"] == payload(events, "agent_model_output_transport")
    assert result["program_epochs"][0]["event_evidence"][0]["payload_original"] == payload(
        events, "agent_program_declared"
    )
    assert events == before
    assert result["acceptance_satisfied"] is False


def test_raw_content_hash_observed_only_and_adapter_argument_never_provider_raw():
    events = structured_events()
    mapped = payload(events, "agent_model_output_transport")["declaration"]
    assert mapped["observed_content_utf8_sha256"] != mapped["content_declaration_sha256"]
    result = audit(events)
    proof = result["structured_content_proofs"][0]
    assert (
        proof["observed_content_utf8_sha256_not_independently_reconstructed"]
        == mapped["observed_content_utf8_sha256"]
    )
    assert proof["provider_tool_call_count"] == 0
    wire = result["wire_normalization_proofs"][0]
    assert "observed_original_argument_sha256_not_independently_reconstructed" not in wire
    assert wire["derived_canonical_content_arguments_sha256"] == mapped["canonical_arguments_sha256"]
    epoch = result["program_epochs"][0]
    assert "observed_original_argument_sha256_not_independently_reconstructed" not in epoch
    assert epoch["argument_hash_is_canonical_content_derived_not_provider_raw"] is True


@pytest.mark.parametrize(
    "change",
    [
        "content_text",
        "content_hash",
        "content_extra",
        "content_duplicate",
        "content_nonfinite",
        "arguments_text",
        "arguments_hash",
        "arguments_value",
        "selected_tool",
        "tool_mask",
        "raw_hash_missing",
        "raw_hash_kind",
        "provider_call",
        "provider_boolean_count",
        "source_hash",
        "source_missing",
        "schema_hash",
        "schema_root_hash",
        "schema_namespace",
        "schema_choice",
        "schema_branch_ref",
        "schema_parameter_hash",
        "schema_provider_enforcement",
        "adapter_kind",
        "adapter_arguments",
        "top_legacy_provenance",
        "wire_arguments",
        "codec_normalized",
        "transport_format",
        "requested_format",
        "phase",
    ],
)
def test_content_schema_source_and_wirepart_tampering_fail_closed(change):
    events = structured_events()
    transport = payload(events, "agent_model_output_transport")
    p = transport["declaration"]
    if change == "content_text":
        p["content_declaration_json"] += " "
    elif change == "content_hash":
        p["content_declaration_sha256"] = "f" * 64
    elif change in {"content_extra", "arguments_value"}:
        value = json.loads(p["content_declaration_json"])
        if change == "content_extra":
            value["extra"] = True
        else:
            value["arguments"]["label"] += " changed"
        p.update(content_declaration_json=canonical(value), content_declaration_sha256=canonical_sha(value))
    elif change == "content_duplicate":
        p["content_declaration_json"] = '{"tool":"x","tool":"y","arguments":{}}'
    elif change == "content_nonfinite":
        p["content_declaration_json"] = '{"tool":"x","arguments":{"x":NaN}}'
    elif change == "arguments_text":
        p["canonical_arguments_json"] += " "
    elif change == "arguments_hash":
        p["canonical_arguments_sha256"] = "f" * 64
    elif change == "selected_tool":
        p["selected_tool"] = TOOLS[1]
    elif change == "tool_mask":
        p["offered_tool_names"].pop()
    elif change == "raw_hash_missing":
        p.pop("observed_content_utf8_sha256")
    elif change == "raw_hash_kind":
        p["argument_hash_kind"] = "provider_raw_arguments"
    elif change == "provider_call":
        p["provider_tool_call_count"] = p["tool_call_count"] = 1
    elif change == "provider_boolean_count":
        p["provider_tool_call_count"] = False
    elif change == "source_hash":
        p["content_codec_source_hashes"][CONTENT_SOURCE] = "f" * 64
    elif change == "source_missing":
        p["content_codec_source_hashes"] = {}
    elif change == "schema_hash":
        p["response_format_sha256"] = "f" * 64
    elif change == "schema_root_hash":
        p["declaration_schema_sha256"] = "f" * 64
    elif change == "schema_parameter_hash":
        p["parameter_schema_sha256"][TOOLS[0]] = "f" * 64
    elif change == "schema_provider_enforcement":
        p["provider_schema_enforcement"] = "verified_without_evidence"
    elif change in {"schema_namespace", "schema_choice", "schema_branch_ref"}:
        response = json.loads(p["response_format_json"])
        if change == "schema_namespace":
            response["json_schema"]["name"] = "other_protocol"
        elif change == "schema_choice":
            response["json_schema"]["schema"]["oneOf"].pop()
        else:
            response["json_schema"]["schema"]["oneOf"][0]["properties"]["arguments"] = {"$ref": "#/$defs/x"}
        p.update(
            response_format_json=canonical(response),
            response_format_sha256=canonical_sha(response),
            declaration_schema_sha256=canonical_sha(response["json_schema"]["schema"]),
        )
    elif change == "adapter_kind":
        p["internal_adapter"]["kind"] = "actual_provider_function"
    elif change == "adapter_arguments":
        p["internal_adapter"]["provenance"]["argument_sha256"] = "f" * 64
    elif change == "top_legacy_provenance":
        p["ancestor_mapping_provenance"] = {}
    elif change == "wire_arguments":
        value = json.loads(p["wire_declaration_json"])
        value["label"] += " not actual content"
        p.update(wire_declaration_json=canonical(value), wire_declaration_sha256=canonical_sha(value))
    elif change == "codec_normalized":
        p["normalized_declaration_sha256"] = "f" * 64
    elif change == "transport_format":
        transport["response_transport"] = "semantic_tools_v64"
    elif change == "requested_format":
        transport["requested_format"] = "semantic_tools_v64"
    else:
        p["phase"] = "original_current_or_review"
    assert_unknown(audit(events))


@pytest.mark.parametrize(
    "change",
    [
        "final_false",
        "no_input",
        "wrong_origin",
        "fake_activation",
        "no_admission",
        "receipt_identity",
        "logical_budget",
    ],
)
def test_content_proof_never_replaces_original_fresh_origin_input_final_or_budget_guards(change):
    events = structured_events()
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
        events = [event for event in events if event["kind"] != "agent_program_stage_admitted"]
        for index, event in enumerate(events):
            event["sequence"] = index
    elif change == "receipt_identity":
        payload(events, "receipt")["decision_id"] = "notjoined"
    else:
        payload(events, "agent_program_stage_activated", 1)["remaining_actions"] = 6
    result = audit(events)
    assert result["structured_content_metric_status"] == "complete"
    assert_unknown(result)


@pytest.mark.parametrize("first_n,second_n", [(None, 1), (0, 1), (None, None), (0, 0)])
def test_zero_or_unknown_inputs_never_manufacture_positive_continuation(first_n, second_n):
    result = audit(structured_events(first_n=first_n, second_n=second_n))
    assert result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 0


def test_no_structured_program_retains_original64_result_and_no_fake_derived_view():
    events = wire_events()
    result = audit_trace(events, source_files=source_files())
    assert result["structured_content_metric_status"] == "not_applicable"
    assert result["structured_content_derived_view_used"] is False
    assert (
        result["program_metric_status"]
        == audit64(events, source_files=source_files())["program_metric_status"]
    )


@pytest.mark.parametrize("tool", TOOLS[1:])
def test_all_nonprogram_choices_remain_selectable_without_fake_function_or_program_events(tool):
    events = structured_events()
    transport = payload(events, "agent_model_output_transport")
    mapped = transport["declaration"]
    arguments = {"label": "Public nonprogram declaration"}
    if tool == "declare_current_context_needed":
        arguments = {"reason": "unsupported_high_level_action"}
    content = {"tool": tool, "arguments": arguments}
    mapped.update(
        selected_tool=tool,
        tool_name=tool,
        program_declaration=False,
        content_declaration_json=canonical(content),
        content_declaration_sha256=canonical_sha(content),
        canonical_arguments_json=canonical(arguments),
        canonical_arguments_sha256=canonical_sha(arguments),
        argument_sha256=canonical_sha(arguments),
        observed_content_utf8_sha256=canonical_sha(content),
    )
    mapped["internal_adapter"]["provenance"].update(
        tool_name=tool, argument_sha256=canonical_sha(arguments), program_declaration=False
    )
    events = [events[0], events[-1]]
    events[1]["sequence"] = 1
    before = copy.deepcopy(events)
    result = audit(events)
    assert result["structured_content_metric_status"] == "complete"
    assert result["structured_content_proofs"][0]["selected_tool"] == tool
    assert result["structured_content_derived_view_used"] is False
    assert not result.get("program_epochs")
    assert events == before


def test_duplicate_actual_transport_call_and_missing_source_manifest_are_unknown():
    events = structured_events()
    events.insert(1, copy.deepcopy(events[0]))
    for index, event in enumerate(events):
        event["sequence"] = index
    assert_unknown(audit(events))
    assert_unknown(audit_trace(structured_events(), expected_format=PROGRAM_FORMAT))


def test_numeric_false_fill_still_requires_actual_verified_numeric_key_in_original_ledger():
    result = audit(structured_events(numeric=True))
    assert result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 1
    assert result["program_epochs"][0]["raw_environment_inputs_complete"] == 5


def test_actual_new_codec_metadata_contract_is_accepted_without_trusting_its_normalizer():
    from cua_jev import desktop_semantic_json_v66 as codec
    from scripts.analyze_stage_program_reuse_v64 import METADATA_KEYS

    events = structured_events()
    transport = payload(events, "agent_model_output_transport")
    arguments = json.loads(transport["declaration"]["canonical_arguments_json"])
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {"type": "object"},
            },
        }
        for name in TOOLS
    ]
    schema, parameters, _removed = codec.declaration_schema(tools)
    response = {
        "type": "json_schema",
        "json_schema": {
            "name": codec.SCHEMA_NAME,
            "strict": True,
            "schema": schema,
        },
    }
    request = {
        "response_transport": PROGRAM_FORMAT,
        "normal_json_schema": True,
        "offered_tool_names": list(TOOLS),
        "response_format_json": canonical(response),
        "response_format_sha256": canonical_sha(response),
        "schema_name": codec.SCHEMA_NAME,
        "declaration_schema_sha256": canonical_sha(schema),
        "parameter_schema_sha256": parameters,
        "schema_namespace": "per_offered_tool_parameter_root_v66",
        "provider_schema_enforcement": "unverified",
        "content_codec_source_hashes": codec._source_hashes(),
    }
    content = json.dumps({"tool": TOOLS[0], "arguments": arguments}, indent=2)
    mapped_object, actual = codec.decode_semantic_declaration(
        {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
        max_actions=6,
        offered_tool_names=list(TOOLS),
        request_provenance=request,
    )
    assert mapped_object["intent"] == "stage_program"
    transport["declaration"] = actual
    declaration = payload(events, "agent_program_declared")["declaration"]
    declaration.update({key: copy.deepcopy(actual[key]) for key in METADATA_KEYS})
    actual_sources = {
        **source_files(),
        **actual["codec_source_hashes"],
        **actual["content_codec_source_hashes"],
    }
    result = audit_trace(events, source_files=actual_sources, expected_format=PROGRAM_FORMAT)
    assert result["structured_content_metric_status"] == result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 1


def test_mixed_real_function_and_content_epochs_keep_distinct_provenance_labels():
    first, second = wire_events(), structured_events()
    first.pop()

    def new_origin(value):
        if type(value) is str:
            return re.sub(
                r"\b(receipt|decision|candidate|observation)-(\d+)\b",
                lambda match: match[1] + "-" + str(int(match[2]) + 2),
                value,
            )
        if type(value) is list:
            return [new_origin(item) for item in value]
        if type(value) is dict:
            return {key: new_origin(item) for key, item in value.items()}
        return value

    second = new_origin(second)
    for event in second:
        item = event["payload"]
        for key in ("call", "origin_planner_call", "planner_calls"):
            if key in item:
                item[key] = 2
        if "program_id" in item:
            item["program_id"] = "2:" + item["program_id"].split(":", 1)[1]
        if event["kind"] == "agent_step":
            item["step"] += 2
        if event["kind"] == "agent_episode":
            item["steps"] += 2
        if event["kind"].startswith("agent_program_") and "environment_steps" in item:
            item["environment_steps"] += 4
    events = first + second
    for index, event in enumerate(events):
        event["sequence"] = index
    before = copy.deepcopy(events)
    result = audit_trace(events, source_files=source_files())
    assert result["program_metric_status"] == "complete"
    assert result["positive_input_program_continuations"] == 2
    original_function, content = result["program_epochs"]
    assert original_function["normalized_source_independently_verified_from_content"] is False
    assert content["normalized_source_independently_verified_from_content"] is True
    assert "observed_original_argument_sha256_not_independently_reconstructed" in original_function
    assert "observed_original_argument_sha256_not_independently_reconstructed" not in content
    assert [row["transport"]["response_transport"] for row in result["program_requests"]] == [
        "semantic_tools_v64",
        PROGRAM_FORMAT,
    ]
    assert events == before


@pytest.mark.parametrize("closed,missing_trace", [(True, False), (False, False), (True, True)])
def test_terminal_selection_retains_failure_official_score_and_all_original_costs(
    tmp_path, closed, missing_trace
):
    runs = tmp_path / "runs"
    runs.mkdir()

    def write(name, value, *, lines=False):
        raw = ("\n".join(json.dumps(item) for item in value) + "\n" if lines else json.dumps(value)).encode()
        (runs / name).write_bytes(raw)
        return {"path": "runs/" + name, "sha256": hashlib.sha256(raw).hexdigest()}

    source = write("source.json", {"files": source_files()})
    terminal = write(
        "terminal.json",
        {"terminal": True, "active_child_may_be_running": False, "source_manifest_sha256": source["sha256"]},
    )
    summary = write(
        "summary.json",
        {
            "runs": [
                {
                    "task": "generic/task",
                    "mode": "hybrid",
                    "official_score": 0.0,
                    "evaluation_status": "ok",
                    "experiment_settings": {"desktop_output_format": PROGRAM_FORMAT},
                    "runner": {"duration_ms": 420123.4, "planner_calls": 1, "status": "timeout"},
                    "episode_wall_ms": 710123.4,
                }
            ]
        },
    )
    official = write("official.json", {"closed": closed, "official_score": 0.0, "evaluation_status": "ok"})
    trace = write("trace.jsonl", structured_events(), lines=True)
    selected = {
        "id": "original_failure",
        "task": "generic/task",
        "mode": "hybrid",
        "source_manifest": source,
        "summary": summary,
        "official": official,
        "trace": None if missing_trace else trace,
    }
    result = analyze(
        tmp_path,
        {
            "kind": "root_selected_positive_input_reuse_evidence_v1",
            "terminal": True,
            "terminal_evidence": terminal,
            "episodes": [selected],
        },
    )
    row = result["episodes"][0]
    assert row["original_summary_row"]["runner"]["duration_ms"] == 420123.4
    assert row["original_summary_row"]["episode_wall_ms"] == 710123.4
    assert row["official_outcome_original"]["official_score"] == 0.0
    assert row["official_outcome_original"]["closed"] is closed
    if closed and not missing_trace:
        assert row["program_metric_status"] == "complete"
        assert row["positive_input_program_continuations"] == 1
    else:
        assert_unknown(row)
    assert result["acceptance_satisfied"] is False
