"""Independently audit v64 wire normalization and the unchanged positive-input ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path

ANCESTOR_SHA256 = "0425ef2e4ea2102943e493679e027ecd081f44f40349f4ccca9caff2640b2664"
PROGRAM_FORMAT = "semantic_tools_v64"
PROGRAM_TOOL = "declare_stage_program"
CODEC_VERSION = "semantic_tools_v64_program_members_stage_v1"
SOURCE_STATUS = "codec_normalized_canonical_json"
CODEC_SOURCES = (
    "src/cua_jev/desktop_semantic_tools_v64.py",
    "src/cua_jev/desktop_program_wire_v64.py",
    "src/cua_jev/desktop_stage_program.py",
    "src/cua_jev/desktop_semantic_tools_v62.py",
    "src/cua_jev/desktop_semantic_tools_v60.py",
)
METADATA_KEYS = (
    "wire_declaration_json", "wire_declaration_sha256",
    "normalized_declaration_json", "normalized_declaration_sha256",
    "wire_stage_mapping", "codec_version", "codec_source_hashes",
)
STAGE_ARGUMENTS = {
    "declare_navigation_stage": (
        {"label", "scope", "arrival"}, {"label", "scope", "arrival"},
    ),
    "declare_property_stage": (
        {"label", "scope", "edit_operation", "numeric_finalize_key", "confirmation_label"},
        {"label", "scope", "edit_operation", "numeric_finalize_key", "confirmation_label",
         "edit_value", "prepare"},
    ),
    "declare_field_edits_stage": (
        {"label", "edits"}, {"label", "edits", "commit"},
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def hash64(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _ancestor():
    """Load only the exact unchanged sibling, never the product codec or normalizer."""
    path = Path(__file__).with_name("analyze_stage_program_reuse_v60.py")
    require(path.is_file() and not path.is_symlink(), "original auditor sibling missing")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == ANCESTOR_SHA256,
            "original auditor sibling changed")
    spec = importlib.util.spec_from_file_location("_v64_original_v60_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate canonical declaration key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError("nonfinite canonical declaration")


def _canonical_object(text):
    require(type(text) is str and len(text.encode("utf-8")) <= 2_000_000,
            "bounded canonical declaration text required")
    value = json.loads(text, object_pairs_hook=_unique, parse_constant=_nonfinite)
    require(type(value) is dict and canonical(value) == text, "exact finite canonical object required")
    return value


def _future_without_current_inputs(value):
    if type(value) is dict:
        require(not {"ref", "box", "box_space", "prepare", "action"} & set(value),
                "wire future current input rejected")
        for item in value.values():
            _future_without_current_inputs(item)
    elif type(value) is list:
        for item in value:
            _future_without_current_inputs(item)


def independently_normalize_wire(wire):
    """Sole transform: add literal completion=stage to each existing member arguments."""
    require(type(wire) is dict and set(wire) == {"label", "stages"},
            "wire program outer fields")
    require(type(wire["label"]) is str and wire["label"].strip()
            and len(wire["label"]) <= 240, "wire program label")
    stages = wire["stages"]
    require(type(stages) is list and 2 <= len(stages) <= 12, "wire ordered stage bound")
    canonical(wire)
    normalized, mapping, seen = copy.deepcopy(wire), [], set()
    for index, member in enumerate(stages):
        require(type(member) is dict and {"id", "tool", "arguments"} <= set(member)
                <= {"id", "tool", "arguments", "entry_conditions", "recovery"},
                "wire member fields")
        identifier, tool, arguments = member["id"], member["tool"], member["arguments"]
        require(type(identifier) is str
                and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", identifier)
                and identifier not in seen, "wire ordered member identity")
        require(type(tool) is str and tool in STAGE_ARGUMENTS, "wire semantic member tool")
        require(type(arguments) is dict and "completion" not in arguments
                and "goal_conditions" not in arguments, "wire member completion fields forbidden")
        required, allowed = STAGE_ARGUMENTS[tool]
        require(required <= set(arguments) <= allowed, "wire semantic member arguments")
        if "entry_conditions" in member:
            require(type(member["entry_conditions"]) is list, "wire entry conditions shape")
        if "recovery" in member:
            require(type(member["recovery"]) is str and member["recovery"] in {"observe", "replan"},
                    "wire recovery policy")
        if index:
            _future_without_current_inputs(arguments)
            _future_without_current_inputs(member.get("entry_conditions", []))
        normalized["stages"][index]["arguments"]["completion"] = "stage"
        mapping.append({"index": index, "id": identifier, "tool": tool})
        seen.add(identifier)
    return normalized, mapping


def _validate_metadata(metadata, source_files):
    require(type(metadata) is dict and all(key in metadata for key in METADATA_KEYS),
            "wire codec provenance missing")
    require(metadata["codec_version"] == CODEC_VERSION, "wire codec version mismatch")
    sources = metadata["codec_source_hashes"]
    require(type(sources) is dict and set(sources) == set(CODEC_SOURCES),
            "exact codec source identities required")
    require(type(source_files) is dict
            and source_files.get("scripts/analyze_stage_program_reuse_v60.py") == ANCESTOR_SHA256,
            "original auditor/source manifest binding required")
    require(all(hash64(value) and source_files.get(name) == value for name, value in sources.items()),
            "codec hashes differ from selected immutable source manifest")
    wire = _canonical_object(metadata["wire_declaration_json"])
    require(metadata["wire_declaration_sha256"] == canonical_sha(wire),
            "wire canonical hash mismatch")
    normalized, mapping = independently_normalize_wire(wire)
    stated = _canonical_object(metadata["normalized_declaration_json"])
    require(canonical(stated) == canonical(normalized), "nonunique codec transform")
    require(metadata["normalized_declaration_sha256"] == canonical_sha(normalized),
            "normalized canonical hash mismatch")
    actual_mapping = metadata["wire_stage_mapping"]
    require(type(actual_mapping) is list
            and all(type(item) is dict and set(item) == {"index", "id", "tool"}
                    and type(item["index"]) is int for item in actual_mapping)
            and canonical(actual_mapping) == canonical(mapping), "wire position/id/tool mapping mismatch")
    return wire, normalized, mapping


def _program_transport(payload, expected_format):
    declaration = payload.get("declaration")
    return (type(declaration) is dict and declaration.get("program_declaration") is True
            and (expected_format == PROGRAM_FORMAT
                 or payload.get("response_transport") == PROGRAM_FORMAT
                 or declaration.get("codec_version") == CODEC_VERSION))


def _wire_view(events, source_files, expected_format):
    require(type(events) is list and len(events) <= 20_000, "bounded original events required")
    view = copy.deepcopy(events)
    requests, declarations, proofs, issues = {}, [], [], []
    for index, event in enumerate(events):
        require(type(event) is dict and type(event.get("payload")) is dict, "original event shape")
        kind, payload = event.get("kind"), event["payload"]
        if kind == "agent_model_output_transport" and _program_transport(payload, expected_format):
            call = payload.get("call")
            require(type(call) is int and call > 0, "wire transport call identity")
            requests.setdefault(call, []).append((index, payload))
        if kind == "agent_program_declared":
            declaration = payload.get("declaration", {})
            if (type(declaration) is dict
                    and (expected_format == PROGRAM_FORMAT
                         or declaration.get("source_declaration_status") == SOURCE_STATUS
                         or declaration.get("codec_version") == CODEC_VERSION)):
                declarations.append((index, payload))

    validated = {}
    for call, rows in requests.items():
        index, payload = rows[0]
        try:
            require(len(rows) == 1 and payload.get("response_transport") == PROGRAM_FORMAT,
                    "one actual v64 program transport required")
            require(payload.get("requested_format", PROGRAM_FORMAT) == PROGRAM_FORMAT,
                    "actual requested format mismatch")
            mapped = payload["declaration"]
            require(mapped.get("tool_name") == PROGRAM_TOOL, "actual program tool mismatch")
            wire, normalized, mapping = _validate_metadata(mapped, source_files)
            require(mapped.get("program_declaration_sha256") == canonical_sha(normalized),
                    "original normalized program hash changed")
            require(hash64(mapped.get("argument_sha256")), "observed raw argument hash missing")
            require(mapped.get("normalized_argument_sha256") == canonical_sha(normalized),
                    "normalized argument hash mismatch")
            if "observed_wire_argument_sha256" in mapped:
                require(mapped["observed_wire_argument_sha256"] == mapped["argument_sha256"],
                        "observed raw argument hashes disagree")
            proof = {
                "call": call, "transport_sequence": index,
                "wire_declaration_sha256": canonical_sha(wire),
                "normalized_declaration_sha256": canonical_sha(normalized),
                "wire_stage_mapping": mapping, "codec_source_hashes": mapped["codec_source_hashes"],
                "observed_original_argument_sha256_not_independently_reconstructed":
                    mapped["argument_sha256"],
                "raw_argument_whitespace_or_insertion_order_not_reconstructed": True,
                "canonical_hash_is_not_raw_argument_hash": True,
                "declaration_sequences": [],
            }
            proofs.append(proof)
            validated[call] = (mapped, normalized, proof)
            # The private derived view is only for the original v60 ledger.
            view[index]["payload"]["response_transport"] = "semantic_tools_v60"
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            issues.append({"sequence": index, "code": str(error) if type(error) is ValueError
                           else type(error).__name__})

    for index, payload in declarations:
        try:
            call, declaration = payload.get("origin_planner_call"), payload["declaration"]
            require(call in validated, "normalized declaration lacks verified original wire transport")
            mapped, normalized, proof = validated[call]
            _validate_metadata(declaration, source_files)
            require(declaration.get("source_declaration_status") == SOURCE_STATUS,
                    "normalized parser input mislabeled as original wire")
            require(all(canonical(declaration[key]) == canonical(mapped[key]) for key in METADATA_KEYS),
                    "transport/program codec provenance mismatch")
            require(canonical(declaration.get("source_declaration")) == canonical(normalized)
                    and declaration.get("declaration_sha256") == canonical_sha(normalized),
                    "normalized parser input/program identity mismatch")
            require(index > proof["transport_sequence"], "program declared before actual wire transport")
            proof["declaration_sequences"].append(index)
            view[index]["payload"]["declaration"]["source_declaration_status"] = "original_canonical_json"
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            issues.append({"sequence": index, "code": str(error) if type(error) is ValueError
                           else type(error).__name__})
    # Every v64 accepted program must have new provenance, even if it was
    # falsely labeled as an old source status or its codec key was removed.
    for index, event in enumerate(events):
        if event.get("kind") == "agent_program_declared":
            call = event["payload"].get("origin_planner_call")
            if call in requests and not any(sequence == index for sequence, _ in declarations):
                issues.append({"sequence": index, "code": "v64 declaration provenance/status missing"})
    return view, proofs, issues, bool(requests or declarations)


def audit_trace(events, *, source_files=None, expected_format=None):
    """Validate a derived normalization view before reusing every original ledger guard."""
    ancestor = _ancestor()
    original = ancestor.audit_trace(events)
    try:
        view, proofs, issues, relevant = _wire_view(events, source_files, expected_format)
    except (ValueError, KeyError, TypeError, RecursionError) as error:
        view, proofs, relevant = None, [], True
        issues = [{"sequence": None, "code": str(error) if type(error) is ValueError
                   else type(error).__name__}]
    result = copy.deepcopy(original) if issues or not relevant else ancestor.audit_trace(view)
    require(result["legacy_fixed_original"] == original["legacy_fixed_original"],
            "derived codec view changed original fixed legacy result")
    if issues:
        result.update(metric_status="unknown", program_metric_status="unknown",
                      positive_input_program_continuations=None,
                      cross_stage_positive_input_program_continuations=None,
                      observed_program_continuations_lower_bound=None)
    result.update(
        kind="program_wire_normalization_and_original_positive_input_audit_v64",
        wire_normalization_metric_status=(
            "unknown" if issues else "complete" if relevant else "not_applicable"
        ),
        wire_normalization_proofs=proofs, wire_normalization_issues=issues,
        original_v60_program_audit=original,
        original_v60_analyzer_sha256=ANCESTOR_SHA256,
        original_trace_events_unchanged=True,
        derived_view_used=bool(relevant and not issues),
        derived_view_changes=["v64 transport format projected for legacy ledger",
                              "independently verified normalized source status projected for legacy ledger"]
        if relevant and not issues else [],
        normalized_parser_input_is_not_original_model_wire=True,
        no_agent_plan_events_injected=True,
        actual_original_positive_input_and_final_verdict_rules_preserved=True,
        wire_capture_is_producer_evidence_not_independent_network_replay=True,
        acceptance_satisfied=False,
    )
    if result["derived_view_used"]:
        # Do not export projected status/format under names claiming original evidence.
        originals = {(event["sequence"], event["kind"]): event["payload"] for event in events}
        for epoch in result.get("program_epochs", []):
            for evidence in epoch.get("event_evidence", []):
                key = (evidence["sequence"], evidence["kind"])
                evidence["payload_original"] = copy.deepcopy(originals[key])
            key = (epoch["declaration_sequence"], "agent_program_declared")
            epoch["source_declaration_status_original"] = (
                originals[key]["declaration"].get("source_declaration_status")
            )
            epoch["normalized_source_independently_verified_from_wire"] = any(
                proof["call"] == epoch["origin_planner_call"] for proof in proofs
            )
        for request in result.get("program_requests", []):
            key = (request["sequence"], "agent_model_output_transport")
            request["transport"] = copy.deepcopy(originals[key])
        result["method"] = (
            "Independently verified v64 wire-to-normalized transformation plus unchanged v60 "
            "source/admission/runtime/final-verdict ledger on an explicitly derived view."
        )
    return result


def analyze(root, manifest):
    """Retain all selected original outcomes/costs; source manifest binds codec provenance."""
    ancestor = _ancestor()
    require(type(manifest) is dict and manifest.get("kind") == ancestor.SELECTION_KIND
            and manifest.get("terminal") is True, "root-selected terminal evidence required")
    reads = []
    terminal = ancestor.load_ref(root, manifest["terminal_evidence"], evidence=reads)
    require(terminal.get("terminal") is True and terminal.get("active_child_may_be_running") is not True,
            "selected batch is not terminal")
    selected_rows = manifest.get("episodes")
    require(type(selected_rows) is list and 1 <= len(selected_rows) <= 32, "bounded selected episodes")
    identities = [(row.get("id"), row.get("task"), row.get("mode")) for row in selected_rows]
    require(len(set(identities)) == len(identities), "duplicate selected identity")
    rows = []
    for selected in selected_rows:
        row = {"id": selected.get("id"), "task": selected.get("task"), "mode": selected.get("mode"),
               "input_refs_original": selected, "metric_status": "unknown",
               "program_metric_status": "unknown", "positive_input_program_continuations": None,
               "positive_input_plan_continuations": None, "official_outcome_original": None}
        try:
            summary = ancestor.load_ref(root, selected["summary"], evidence=reads)
            source = ancestor.load_ref(root, selected["source_manifest"], evidence=reads)
            require(type(source.get("files")) is dict, "source inventory missing")
            if terminal.get("source_manifest_sha256"):
                require(terminal["source_manifest_sha256"] == selected["source_manifest"]["sha256"],
                        "selected source differs from terminal evidence")
            matches = [item for item in summary.get("runs", [])
                       if item.get("task") == selected["task"] and item.get("mode") == selected["mode"]]
            require(len(matches) == 1, "summary identity ambiguous")
            original = matches[0]
            row["original_summary_row"] = original
            official = (ancestor.load_ref(root, selected["official"], evidence=reads)
                        if selected.get("official") else None)
            row["official_outcome_original"] = official
            if selected.get("copy_manifest"):
                ancestor.load_ref(root, selected["copy_manifest"], evidence=reads)
            require(selected.get("trace") is not None, "original trace missing; no zero inferred")
            events = ancestor.load_ref(root, selected["trace"], lines=True, evidence=reads)
            expected_format = original.get("experiment_settings", {}).get("desktop_output_format")
            row.update(audit_trace(events, source_files=source["files"], expected_format=expected_format))
            valid_official = (
                official is not None and official.get("closed") is True
                and type(official.get("official_score")) is type(original.get("official_score"))
                and official.get("official_score") == original.get("official_score")
                and official.get("evaluation_status") == original.get("evaluation_status")
            )
            if not valid_official:
                row.update(metric_status="unknown", program_metric_status="unknown",
                           positive_input_program_continuations=None,
                           cross_stage_positive_input_program_continuations=None,
                           positive_input_plan_continuations=None)
                row.setdefault("artifact_issues", []).append("closed official/summary agreement unavailable")
        except (ValueError, KeyError, TypeError, AttributeError, OSError, json.JSONDecodeError) as error:
            row.update(metric_status="unknown", program_metric_status="unknown",
                       positive_input_program_continuations=None,
                       cross_stage_positive_input_program_continuations=None,
                       positive_input_plan_continuations=None, error_type=type(error).__name__)
        rows.append(row)
    return {"kind": "program_wire_positive_input_reuse_offline_audit_v64", "episodes": rows,
            "complete_episodes": sum(row["metric_status"] == "complete" for row in rows),
            "unknown_or_partial_episodes": sum(row["metric_status"] != "complete" for row in rows),
            "source_artifacts_read": reads, "original_v60_analyzer_sha256": ANCESTOR_SHA256,
            "original_fixed_analyzer_unchanged_and_separately_required": True,
            "all_selected_failures_unknowns_and_original_costs_preserved": True,
            "mechanism_is_not_official_task_success": True, "acceptance_satisfied": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, ancestor = args.root.resolve(), _ancestor()
    manifest = ancestor.load_ref(root, {"path": str(args.manifest), "sha256": args.manifest_sha256})
    result = analyze(root, manifest)
    result["selection_manifest_sha256"] = args.manifest_sha256
    result["analyzer_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output if args.output.is_absolute() else root / args.output
    require(output.parent.is_dir() and output.parent.resolve().is_relative_to((root / "runs").resolve())
            and not output.exists(), "new output within runs required")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
