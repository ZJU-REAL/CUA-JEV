"""One explicit wire-to-parser normalization; no new execution authority."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .desktop_semantic_tools import _require

CODEC_VERSION = "semantic_tools_v64_program_members_stage_v1"
SOURCE_MODULES = (
    "desktop_semantic_tools_v64.py", "desktop_program_wire_v64.py", "desktop_stage_program.py",
    "desktop_semantic_tools_v62.py", "desktop_semantic_tools_v60.py",
)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def codec_source_hashes():
    root = Path(__file__).resolve().parent
    return {"src/cua_jev/" + name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in SOURCE_MODULES}


def normalize_program_wire(raw, *, max_actions):
    """Only an absent member completion field receives the fixed stage value.

    Explicit completion/goal fields are rejected, including stage/empty/null.
    All other syntax and semantic authority remain the original parser's job.
    """
    _require(type(raw) is dict and set(raw) == {"label", "stages"}, "program_outer_fields")
    items = raw["stages"]
    _require(type(items) is list and 2 <= len(items) <= max_actions, "program_stage_count")
    wire_json = canonical_json(raw)
    normalized = copy.deepcopy(raw)
    mapping = []
    for index, item in enumerate(normalized["stages"]):
        _require(type(item) is dict and {"id", "tool", "arguments"} <= set(item)
                 <= {"id", "tool", "arguments", "entry_conditions", "recovery"}, "program_stage_fields")
        arguments = item["arguments"]
        _require(type(arguments) is dict, "program_wire_arguments_must_be_object")
        _require("completion" not in arguments, "program_wire_forbids_member_completion")
        _require("goal_conditions" not in arguments, "program_wire_forbids_member_goal_conditions")
        arguments["completion"] = "stage"
        mapping.append({"index": index, "id": item["id"], "tool": item["tool"]})
    normalized_json = canonical_json(normalized)
    return normalized, {
        "wire_declaration_json": wire_json,
        "wire_declaration_sha256": hashlib.sha256(wire_json.encode("utf-8")).hexdigest(),
        "normalized_declaration_json": normalized_json,
        "normalized_declaration_sha256": hashlib.sha256(normalized_json.encode("utf-8")).hexdigest(),
        "wire_stage_mapping": mapping, "codec_version": CODEC_VERSION,
        "codec_source_hashes": codec_source_hashes(),
    }


def bind_program_provenance(program, provenance):
    """Bind only to the exact normalized declaration the original parser accepted."""
    from .desktop_stage_program import StageProgram

    _require(isinstance(program, StageProgram), "program_wire_requires_original_stage_program")
    _require(provenance.get("codec_version") == CODEC_VERSION, "program_wire_codec_version")
    wire = json.loads(provenance["wire_declaration_json"])
    normalized, expected = normalize_program_wire(wire, max_actions=program.max_actions)
    for key in expected:
        _require(provenance.get(key) == expected[key], "program_wire_provenance_mismatch")
    _require(program.declaration_json == canonical_json(normalized)
             and program.declaration_sha256 == expected["normalized_declaration_sha256"],
             "program_wire_normalized_parser_input_mismatch")
    return replace(
        program, wire_declaration_json=expected["wire_declaration_json"],
        wire_declaration_sha256=expected["wire_declaration_sha256"], codec_version=CODEC_VERSION,
        codec_source_hashes_json=canonical_json(expected["codec_source_hashes"]),
        wire_stage_mapping_json=canonical_json(expected["wire_stage_mapping"]),
    )
