"""Audit actual v66 structured content before reusing the unchanged v64 input ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

ORIGINAL_V64_SHA256 = "cc0a175f51b904ec98df4b0f29379bd8f37ce92473a91e91c58aeb160888b36f"
PROGRAM_FORMAT = "semantic_json_v66"
CONTENT_SOURCE = "src/cua_jev/desktop_semantic_json_v66.py"
TOOLS = (
    "declare_stage_program",
    "declare_property_stage",
    "declare_navigation_stage",
    "declare_field_edits_stage",
    "declare_completion",
    "declare_current_context_needed",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _old():
    path = Path(__file__).with_name("analyze_stage_program_reuse_v64.py")
    require(path.is_file() and not path.is_symlink(), "unchanged v64 auditor missing")
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_V64_SHA256,
        "unchanged v64 auditor bytes differ",
    )
    spec = importlib.util.spec_from_file_location("_v66_unchanged_v64_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _schema_tool_mask(mapped, old):
    response = old._canonical_object(mapped["response_format_json"])
    require(mapped["response_format_sha256"] == canonical_sha(response), "response format hash mismatch")
    require(
        set(response) == {"type", "json_schema"} and response["type"] == "json_schema",
        "actual JSON-schema response format required",
    )
    envelope = response["json_schema"]
    require(
        type(envelope) is dict
        and set(envelope) == {"name", "strict", "schema"}
        and envelope["name"] == mapped["schema_name"] == "cua_jev_declaration_v66"
        and envelope["strict"] is True,
        "v66 strict schema namespace required",
    )
    schema = envelope["schema"]
    require(
        type(schema) is dict
        and schema.get("type") == "object"
        and schema.get("additionalProperties") is False
        and set(schema.get("required", [])) == {"tool", "arguments"}
        and set(schema.get("properties", {})) == {"tool", "arguments"}
        and schema["properties"]["tool"] == {"type": "string", "enum": list(TOOLS)}
        and schema["properties"]["arguments"] == {"type": "object"}
        and type(schema.get("oneOf")) is list
        and len(schema["oneOf"]) == 6,
        "six structured declaration branches required",
    )
    require(
        mapped.get("declaration_schema_sha256") == canonical_sha(schema),
        "actual root declaration schema hash mismatch",
    )
    require(
        mapped.get("schema_namespace") == "per_offered_tool_parameter_root_v66"
        and mapped.get("provider_schema_enforcement") == "unverified",
        "schema namespace and provider enforcement boundary required",
    )
    parameters = mapped.get("parameter_schema_sha256")
    require(
        type(parameters) is dict and set(parameters) == set(TOOLS), "six parameter schema hashes required"
    )
    names = []
    for index, branch in enumerate(schema["oneOf"]):
        require(
            type(branch) is dict
            and branch.get("type") == "object"
            and branch.get("additionalProperties") is False
            and set(branch.get("required", [])) == {"tool", "arguments"}
            and set(branch.get("properties", {})) == {"tool", "arguments"},
            "closed tool/arguments schema branch required",
        )
        selector = branch["properties"]["tool"]
        require(
            type(selector) is dict and set(selector) == {"const"} and type(selector["const"]) is str,
            "literal tool selector required",
        )
        namespace = f"v66_tool_{index}_arguments"
        require(
            branch["properties"]["arguments"] == {"$ref": "#/$defs/" + namespace}
            and namespace in schema.get("$defs", {}),
            "tool argument namespace binding required",
        )
        require(
            parameters.get(selector["const"]) == canonical_sha(schema["$defs"][namespace]),
            "namespaced parameter schema hash mismatch",
        )
        names.append(selector["const"])
    require(
        names == list(TOOLS) and mapped.get("offered_tool_names") == names,
        "actual six-choice schema and tool mask differ",
    )
    return response


def _content_proof(payload, source_files, old):
    mapped = payload["declaration"]
    require(
        payload.get("response_transport") == payload.get("requested_format") == PROGRAM_FORMAT
        and mapped.get("response_transport") == mapped.get("requested_format") == PROGRAM_FORMAT,
        "actual structured transport/requested format mismatch",
    )
    require(
        mapped.get("phase") == "normal_high_level"
        and mapped.get("transport_kind") == "json_schema_content"
        and mapped.get("normal_json_schema") is True,
        "normal structured phase required",
    )
    require(
        type(mapped.get("provider_tool_call_count")) is int
        and mapped["provider_tool_call_count"] == 0
        and type(mapped.get("tool_call_count")) is int
        and mapped["tool_call_count"] == 0,
        "structured content cannot claim provider function calls",
    )
    require(
        type(source_files) is dict
        and source_files.get("scripts/analyze_stage_program_reuse_v64.py") == ORIGINAL_V64_SHA256,
        "unchanged v64 auditor source manifest binding required",
    )
    sources = mapped.get("content_codec_source_hashes")
    require(
        type(sources) is dict
        and set(sources) == {CONTENT_SOURCE}
        and old.hash64(sources[CONTENT_SOURCE])
        and sources[CONTENT_SOURCE] == source_files.get(CONTENT_SOURCE),
        "content codec differs from selected source manifest",
    )
    value = old._canonical_object(mapped["content_declaration_json"])
    require(
        set(value) == {"tool", "arguments"}
        and type(value["tool"]) is str
        and value["tool"] in TOOLS
        and type(value["arguments"]) is dict,
        "exact structured tool/arguments object required",
    )
    require(mapped["content_declaration_sha256"] == canonical_sha(value), "canonical content hash mismatch")
    arguments = old._canonical_object(mapped["canonical_arguments_json"])
    require(
        canonical(arguments) == canonical(value["arguments"])
        and mapped["canonical_arguments_sha256"] == canonical_sha(arguments),
        "content arguments canonical derivation mismatch",
    )
    require(
        mapped.get("tool_name") == mapped.get("selected_tool") == value["tool"],
        "selected tool differs from actual structured content",
    )
    require(
        mapped.get("argument_hash_kind") == "adapter_canonical_arguments"
        and mapped.get("argument_sha256") == canonical_sha(arguments),
        "derived arguments must not claim provider raw arguments",
    )
    require(old.hash64(mapped.get("observed_content_utf8_sha256")), "observed content UTF8 hash missing")
    _schema_tool_mask(mapped, old)
    adapter = mapped.get("internal_adapter")
    require(
        type(adapter) is dict
        and set(adapter) == {"kind", "provenance"}
        and adapter["kind"] == "synthetic_legacy_mapper_payload_not_provider"
        and type(adapter["provenance"]) is dict,
        "explicit private mapper provenance required",
    )
    # A nested synthetic envelope is a parser adapter, never a provider wire event.
    internal = adapter["provenance"]
    require(
        internal.get("tool_name") == value["tool"]
        and internal.get("argument_sha256") == canonical_sha(arguments),
        "internal mapper is not bound to actual canonical arguments",
    )
    require(
        not {"ancestor_mapping_provenance", "ancestor_response_transport", "ancestor_mapping_applied"}
        & set(mapped),
        "synthetic ancestor mapping must remain internal",
    )
    proof = {
        "call": payload["call"],
        "selected_tool": value["tool"],
        "content_declaration_sha256": canonical_sha(value),
        "canonical_arguments_sha256": canonical_sha(arguments),
        "observed_content_utf8_sha256_not_independently_reconstructed": mapped[
            "observed_content_utf8_sha256"
        ],
        "original_content_whitespace_not_saved_or_independently_reconstructed": True,
        "provider_tool_call_count": 0,
        "provider_function_arguments_not_present": True,
        "argument_hash_kind": "adapter_canonical_arguments",
        "response_format_sha256": mapped["response_format_sha256"],
        "offered_tool_names": list(TOOLS),
        "content_codec_source_hashes": sources,
    }
    return arguments, internal, proof


def _content_view(events, source_files, expected_format, old):
    require(type(events) is list and len(events) <= 20_000, "bounded actual original events required")
    view, proofs, issues, calls = copy.deepcopy(events), [], [], set()
    normal_programs = set()
    for index, event in enumerate(events):
        require(type(event) is dict and type(event.get("payload")) is dict, "original event shape")
        payload = event["payload"]
        if event.get("kind") != "agent_model_output_transport":
            continue
        mapped = payload.get("declaration", {})
        selected66 = expected_format == PROGRAM_FORMAT or payload.get("response_transport") == PROGRAM_FORMAT
        if not selected66:
            continue
        try:
            call = payload.get("call")
            require(
                type(call) is int and call > 0 and call not in calls, "unique actual transport call required"
            )
            calls.add(call)
            require(type(mapped) is dict, "actual decoder declaration required")
            require(
                payload.get("response_transport") == payload.get("requested_format") == PROGRAM_FORMAT,
                "selected v66 source cannot masquerade as a legacy provider transport",
            )
            if mapped.get("normal_json_schema") is False:
                require(
                    mapped.get("phase") == "original_current_or_review"
                    and mapped.get("transport_kind") == "original_function_tools"
                    and mapped.get("program_declaration") is not True,
                    "non-normal phase must retain original function protocol",
                )
                continue
            arguments, internal, proof = _content_proof(payload, source_files, old)
            proof["transport_sequence"] = index
            proofs.append(proof)
            if proof["selected_tool"] != old.PROGRAM_TOOL:
                require(
                    mapped.get("program_declaration") is not True, "nonprogram content cannot declare program"
                )
                continue
            require(mapped.get("program_declaration") is True, "actual content program identity required")
            wire, normalized, _mapping = old._validate_metadata(mapped, source_files)
            require(
                canonical(wire) == canonical(arguments),
                "actual content arguments differ from program wirepart",
            )
            old._validate_metadata(internal, source_files)
            require(
                all(canonical(mapped[key]) == canonical(internal[key]) for key in old.METADATA_KEYS),
                "internal adapter codec provenance differs from actual content wirepart",
            )
            require(
                internal.get("response_transport") == old.PROGRAM_FORMAT
                and internal.get("program_declaration") is True
                and internal.get("program_declaration_sha256") == canonical_sha(normalized),
                "internal unchanged program parser provenance required",
            )
            normal_programs.add(call)
            # This changes only a private audit view. No provider function event is created.
            projected = view[index]["payload"]
            projected["response_transport"] = projected["requested_format"] = old.PROGRAM_FORMAT
            projected["declaration"] = copy.deepcopy(internal)
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            issues.append(
                {"sequence": index, "code": str(error) if type(error) is ValueError else type(error).__name__}
            )
    for index, event in enumerate(events):
        if event.get("kind") == "agent_program_declared":
            call = event["payload"].get("origin_planner_call")
            if (expected_format == PROGRAM_FORMAT or call in calls) and call not in normal_programs:
                issues.append(
                    {"sequence": index, "code": "program lacks verified actual structured content origin"}
                )
    return view, proofs, issues, normal_programs


def _unknown(result):
    result.update(
        metric_status="unknown",
        program_metric_status="unknown",
        positive_input_program_continuations=None,
        cross_stage_positive_input_program_continuations=None,
        observed_program_continuations_lower_bound=None,
    )


def audit_trace(events, *, source_files=None, expected_format=None):
    """Producer content capture is audited; it is not an independent network replay."""
    old = _old()
    original = old.audit_trace(events, source_files=source_files, expected_format=expected_format)
    try:
        view, proofs, issues, program_calls = _content_view(events, source_files, expected_format, old)
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        view, proofs, program_calls = None, [], set()
        issues = [
            {"sequence": None, "code": str(error) if type(error) is ValueError else type(error).__name__}
        ]
    derived = bool(program_calls and not issues)
    result = (
        old.audit_trace(view, source_files=source_files, expected_format=old.PROGRAM_FORMAT)
        if derived
        else copy.deepcopy(original)
    )
    require(
        result["legacy_fixed_original"] == original["legacy_fixed_original"],
        "derived structured view changed original fixed legacy result",
    )
    if issues:
        _unknown(result)
    if derived:
        originals = {(event["sequence"], event["kind"]): event["payload"] for event in events}
        for epoch in result.get("program_epochs", []):
            call = epoch.get("origin_planner_call")
            epoch["normalized_source_independently_verified_from_content"] = call in program_calls
            for item in epoch.get("event_evidence", []):
                item["payload_original"] = copy.deepcopy(originals[(item["sequence"], item["kind"])])
            if call in program_calls:
                for key, honest_key in (
                    (
                        "observed_original_argument_sha256_not_independently_reconstructed",
                        "internal_adapter_canonical_arguments_sha256",
                    ),
                    (
                        "observed_ordered_mapped_object_sha256_not_independently_reconstructed",
                        "internal_adapter_ordered_mapped_object_sha256",
                    ),
                ):
                    if key in epoch:
                        epoch[honest_key] = epoch.pop(key)
                epoch["argument_hash_is_canonical_content_derived_not_provider_raw"] = True
        for request in result.get("program_requests", []):
            request["transport"] = copy.deepcopy(
                originals[(request["sequence"], "agent_model_output_transport")]
            )
        for proof in result.get("wire_normalization_proofs", []):
            if proof["call"] in program_calls:
                proof["derived_canonical_content_arguments_sha256"] = proof.pop(
                    "observed_original_argument_sha256_not_independently_reconstructed"
                )
                proof.pop("raw_argument_whitespace_or_insertion_order_not_reconstructed", None)
                proof.pop("canonical_hash_is_not_raw_argument_hash", None)
                proof["provider_function_arguments_not_present"] = True
        # Restore the actual original audit, not the private view's 'original'.
        result["original_v60_program_audit"] = copy.deepcopy(original["original_v60_program_audit"])
    result.update(
        kind="structured_content_and_original_positive_input_audit_v66",
        structured_content_metric_status="unknown" if issues else "complete" if proofs else "not_applicable",
        structured_content_proofs=proofs,
        structured_content_issues=issues,
        original_v64_program_audit=original,
        original_v64_analyzer_sha256=ORIGINAL_V64_SHA256,
        original_trace_events_unchanged=True,
        structured_content_derived_view_used=derived,
        structured_content_derived_view_changes=[
            "verified actual content.arguments projected into an internal v64 ledger view"
        ]
        if derived
        else [],
        no_provider_function_calls_invented=True,
        no_agent_plan_events_injected=True,
        actual_original_positive_input_and_final_verdict_rules_preserved=True,
        content_capture_is_producer_evidence_not_independent_network_replay=True,
        observed_content_hash_is_not_independently_reconstructed=True,
        acceptance_satisfied=False,
    )
    if derived:
        result["derived_view_changes"] = [
            *result.get("derived_view_changes", []),
            "verified actual v66 content projected only inside the derived input ledger",
        ]
        result["method"] = (
            "Actual selected structured content and source/schema provenance, independent canonical "
            "arguments/wire normalization, unchanged original input ledger on an explicit derived view."
        )
    return result


def analyze(root, manifest):
    """Reuse original selection/official checks and retain every selected outcome and cost."""
    old = _old()
    result = old.analyze(root, manifest)
    actual_original = copy.deepcopy(result)
    ancestor = old._ancestor()
    for row, selected in zip(result["episodes"], manifest["episodes"], strict=True):
        row["acceptance_satisfied"] = False
        if "original_summary_row" not in row or selected.get("trace") is None:
            continue
        try:
            source = ancestor.load_ref(root, selected["source_manifest"])
            events = ancestor.load_ref(root, selected["trace"], lines=True)
            summary = row["original_summary_row"]
            row.update(
                audit_trace(
                    events,
                    source_files=source["files"],
                    expected_format=(summary.get("experiment_settings", {}).get("desktop_output_format")),
                )
            )
            official = row.get("official_outcome_original")
            valid_official = (
                official is not None
                and official.get("closed") is True
                and type(official.get("official_score")) is type(summary.get("official_score"))
                and official.get("official_score") == summary.get("official_score")
                and official.get("evaluation_status") == summary.get("evaluation_status")
            )
            if not valid_official or row.get("artifact_issues") or row.get("error_type"):
                _unknown(row)
                row["positive_input_plan_continuations"] = None
        except (ValueError, KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as error:
            _unknown(row)
            row["error_type"] = type(error).__name__
    result.update(
        kind="structured_content_positive_input_reuse_offline_audit_v66",
        original_v64_selection_audit=actual_original,
        complete_episodes=sum(row["metric_status"] == "complete" for row in result["episodes"]),
        unknown_or_partial_episodes=sum(row["metric_status"] != "complete" for row in result["episodes"]),
        original_v64_analyzer_sha256=ORIGINAL_V64_SHA256,
        acceptance_satisfied=False,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, ancestor = args.root.resolve(), _old()._ancestor()
    manifest = ancestor.load_ref(root, {"path": str(args.manifest), "sha256": args.manifest_sha256})
    result = analyze(root, manifest)
    result["selection_manifest_sha256"] = args.manifest_sha256
    result["analyzer_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output if args.output.is_absolute() else root / args.output
    require(
        output.parent.is_dir()
        and output.parent.resolve().is_relative_to((root / "runs").resolve())
        and not output.exists(),
        "new output within runs required",
    )
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
