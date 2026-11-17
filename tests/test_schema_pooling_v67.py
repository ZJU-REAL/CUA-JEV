"""Independent restricted pooling replay; no claim about provider generation quality."""

import copy
import hashlib
import json
import re

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_semantic_tools_v57 import current
from test_semantic_tools_v57 import observed as observed
from test_stage_program_planning_preference_v61 import requested
from test_structured_high_level_declaration_v66 import declarations, response
from test_structured_high_level_declaration_v66 import offered as offered

from cua_jev import desktop_semantic_json_v66 as codec
from cua_jev.desktop_semantic_tools_v64 import LEGACY_PROPERTY_OPERATIONS, NORMAL_PROPERTY_OPERATIONS

BASELINE66_SCHEMA_SHA256 = "60b05977ad5111f24e3305569e72d957561b4ad7264647012a7b17abe618b778"
ROOTS = tuple(f"v66_tool_{index}_arguments" for index in range(6))
ANCHOR = ROOTS[0]
POOL_KEYS = {
    "version", "anchor_root", "applied", "status", "mappings", "removed_definition_count",
    "original_schema_sha256", "pooled_schema_sha256", "original_schema_bytes", "pooled_schema_bytes",
    "provider_generation_equivalence",
}
FORBIDDEN = {
    "$id", "$anchor", "$dynamicRef", "$dynamicAnchor", "$recursiveRef", "$recursiveAnchor",
    "unevaluatedProperties", "unevaluatedItems",
}
SCHEMA_MAPS = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
SCHEMA_ARRAYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
SCHEMA_SINGLE = {"items", "additionalProperties", "propertyNames", "contains", "not", "if", "then", "else"}
DATA_KEYWORDS = {
    "type", "required", "enum", "const", "default", "examples", "title", "description", "$comment",
    "format", "pattern", "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "minItems", "maxItems", "minProperties", "maxProperties", "multipleOf",
    "uniqueItems", "dependentRequired", "readOnly", "writeOnly", "deprecated", "contentEncoding",
    "contentMediaType",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def pointer(parts):
    return "#" + "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def tokens(reference):
    assert type(reference) is str and reference.startswith("#/") and "%" not in reference
    raw = reference[2:].split("/")
    assert all(re.search(r"~(?![01])", part) is None for part in raw)
    parts = tuple(part.replace("~1", "/").replace("~0", "~") for part in raw)
    assert pointer(parts) == reference
    return parts


def resolve(document, reference):
    node = document
    for part in tokens(reference):
        if type(node) is list:
            assert part.isdecimal() and str(int(part)) == part
            node = node[int(part)]
        else:
            assert type(node) is dict and part in node
            node = node[part]
    assert type(node) in {dict, bool}
    return node


def schema_index(document):
    """Walk schema positions only; literal const/enum/default/examples stay opaque."""
    indexed, children = {}, {}

    def visit(node, parts):
        assert len(parts) <= 64 and len(indexed) < 20_000 and type(node) in {dict, bool}
        here = pointer(parts)
        indexed[here], children[here] = node, []
        if type(node) is bool:
            return
        assert not FORBIDDEN & set(node)
        assert set(node) <= SCHEMA_MAPS | SCHEMA_ARRAYS | SCHEMA_SINGLE | DATA_KEYWORDS | {"$ref", "$schema"}
        if "$schema" in node:
            assert node["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        for keyword, value in node.items():
            if keyword in SCHEMA_MAPS:
                assert type(value) is dict
                descendants = [(parts + (keyword, name), child) for name, child in value.items()]
            elif keyword in SCHEMA_ARRAYS:
                assert type(value) is list
                descendants = [(parts + (keyword, str(index)), child) for index, child in enumerate(value)]
            elif keyword in SCHEMA_SINGLE:
                descendants = [(parts + (keyword,), value)]
            else:
                continue
            for child_parts, child in descendants:
                children[here].append(pointer(child_parts))
                visit(child, child_parts)

    visit(document, ())
    return indexed, children


def closed_acyclic(document):
    indexed, edges = schema_index(document)
    for location, node in indexed.items():
        if type(node) is dict and "$ref" in node:
            reference = node["$ref"]
            parts = tokens(reference)
            assert len(parts) in {2, 4} and parts[0] == "$defs" and parts[1] in ROOTS
            assert len(parts) == 2 or parts[2] == "$defs"
            assert reference in indexed and resolve(document, reference) is indexed[reference]
            edges[location].append(reference)
    visiting, complete = set(), set()

    def inspect(location):
        assert location not in visiting
        if location in complete:
            return
        visiting.add(location)
        for child in edges[location]:
            inspect(child)
        visiting.remove(location)
        complete.add(location)

    for location in indexed:
        inspect(location)


def rewrite_schema(document, mappings):
    result = copy.deepcopy(document)
    indexed, _ = schema_index(result)
    for node in indexed.values():
        if type(node) is dict and "$ref" in node:
            node["$ref"] = mappings.get(node["$ref"], node["$ref"])
    return result


def independent_pool_proof(baseline, actual, metadata):
    """Check every edge and replay the entire edit without using the producer pool."""
    assert set(metadata) == POOL_KEYS
    assert metadata["version"] == "p0_dependency_closed_def_pool_v67"
    assert metadata["anchor_root"] == ANCHOR
    assert metadata["provider_generation_equivalence"] == "unverified"
    assert metadata["original_schema_sha256"] == digest(baseline)
    assert metadata["pooled_schema_sha256"] == digest(actual)
    assert metadata["original_schema_bytes"] == len(canonical(baseline).encode("utf-8"))
    assert metadata["pooled_schema_bytes"] == len(canonical(actual).encode("utf-8"))
    assert type(metadata["applied"]) is bool and type(metadata["removed_definition_count"]) is int
    closed_acyclic(baseline)
    closed_acyclic(actual)
    rows = metadata["mappings"]
    assert type(rows) is list
    assert all(type(row) is dict and set(row) == {"source_ref", "anchor_ref"} for row in rows)
    assert [row["source_ref"] for row in rows] == sorted({row["source_ref"] for row in rows})
    mapping = {row["source_ref"]: row["anchor_ref"] for row in rows}
    assert metadata["removed_definition_count"] == len(mapping)
    assert metadata["applied"] == bool(mapping)
    assert metadata["status"] == ("pooled" if mapping else "unchanged_no_duplicates")
    assert canonical(actual["$defs"][ANCHOR]) == canonical(baseline["$defs"][ANCHOR])
    for source, anchor in mapping.items():
        original_parts, anchor_parts = tokens(source), tokens(anchor)
        assert len(original_parts) == len(anchor_parts) == 4
        assert original_parts[:1] == anchor_parts[:1] == ("$defs",)
        assert original_parts[1] in ROOTS[1:] and anchor_parts[1] == ANCHOR
        assert original_parts[2] == anchor_parts[2] == "$defs"
        assert original_parts[3] == anchor_parts[3]
        original_node = resolve(baseline, source)
        anchor_node = resolve(baseline, anchor)
        assert canonical(rewrite_schema(original_node, mapping)) == canonical(anchor_node)
    replayed = rewrite_schema(baseline, mapping)
    for source in mapping:
        parts = tokens(source)
        del replayed["$defs"][parts[1]]["$defs"][parts[3]]
    assert canonical(replayed) == canonical(actual)
    return mapping


def original_frozen66_schema(baseline):
    # v68 corrects descriptions only. Reverse precisely that annotation before
    # checking the original observed frozen66 full-schema hash and byte count.
    # Any constraint, literal or unrelated description change still fails this pin.
    original66 = copy.deepcopy(baseline)
    indexed, _ = schema_index(original66)
    corrected = 0
    for node in indexed.values():
        if isinstance(node, dict) and isinstance(node.get("description"), str):
            text = node["description"]
            if NORMAL_PROPERTY_OPERATIONS in text:
                node["description"] = text.replace(NORMAL_PROPERTY_OPERATIONS, LEGACY_PROPERTY_OPERATIONS)
                corrected += 1
    assert corrected >= 2
    assert digest(original66) == BASELINE66_SCHEMA_SHA256
    assert len(canonical(original66).encode("utf-8")) == 31_617
    return original66


@pytest.fixture
def pool_case(offered):
    original, encoded, provenance = offered
    baseline, _, _ = codec.declaration_schema(copy.deepcopy(original["tools"]))
    original_frozen66_schema(baseline)
    actual = encoded["response_format"]["json_schema"]["schema"]
    metadata = provenance["schema_pooling"]
    return baseline, actual, metadata, provenance


def test_actual_frozen66_baseline_and_independent_full_replay_are_identical_constraints(pool_case):
    baseline, actual, metadata, provenance = pool_case
    mapping = independent_pool_proof(baseline, actual, metadata)
    assert len(mapping) == 15 and metadata["pooled_schema_bytes"] < metadata["original_schema_bytes"]
    assert len(actual["$defs"]) == len(actual["oneOf"]) == 6
    assert actual["oneOf"] == baseline["oneOf"]
    assert provenance["schema_namespace"] == "per_offered_tool_parameter_root_v66"
    assert provenance["declaration_schema_sha256"] == digest(actual)
    for index, name in enumerate(codec.DECLARATION_TOOLS):
        assert provenance["parameter_schema_sha256"][name] == digest(actual["$defs"][ROOTS[index]])
    Draft202012Validator.check_schema(baseline)
    Draft202012Validator.check_schema(actual)


@pytest.mark.parametrize("tool,arguments", declarations())
def test_all_six_real_declarations_validate_and_keep_original64_mapping_and_provenance(
    pool_case, tool, arguments
):
    baseline, actual, metadata, provenance = pool_case
    value = {"tool": tool, "arguments": copy.deepcopy(arguments)}
    Draft202012Validator(baseline).validate(value)
    Draft202012Validator(actual).validate(value)
    mapped, transport = codec.decode_semantic_declaration(
        response(value), max_actions=6, request_provenance=provenance,
        offered_tool_names=provenance["offered_tool_names"],
    )
    internal = {"choices": [{"message": {"tool_calls": [{"type": "function", "function": {
        "name": tool, "arguments": canonical(arguments),
    }}]}}]}
    previous, original_transport = codec.decode_ancestor(
        internal, max_actions=6, normal_program_eligible=True)
    assert mapped == previous and transport["internal_adapter"]["provenance"] == original_transport
    assert transport["schema_pooling"] == metadata
    assert transport["provider_tool_call_count"] == transport["tool_call_count"] == 0
    assert transport["argument_hash_kind"] == "adapter_canonical_arguments"
    assert transport["canonical_arguments_sha256"] == digest(arguments)
    assert transport["input_authority"] is False


@pytest.mark.parametrize("annotated", [False, True], ids=["original66", "annotated68"])
def test_legal_legacy_unpooled_request_without_optional_pooling_diagnostic_still_decodes(
    pool_case, annotated,
):
    baseline, _, _, pooled_provenance = pool_case
    if not annotated:
        baseline = original_frozen66_schema(baseline)
    provenance = {key: copy.deepcopy(pooled_provenance[key]) for key in codec._REQUEST_BINDING_FIELDS}
    provenance.update(response_transport="semantic_json_v66", normal_json_schema=True)
    assert "schema_pooling" not in provenance
    response_format = json.loads(provenance["response_format_json"])
    response_format["json_schema"]["schema"] = baseline
    provenance.update(
        response_format_json=canonical(response_format), response_format_sha256=digest(response_format),
        declaration_schema_sha256=digest(baseline),
        parameter_schema_sha256={name: digest(baseline["$defs"][ROOTS[index]])
                                 for index, name in enumerate(codec.DECLARATION_TOOLS)},
    )
    before = copy.deepcopy(provenance)
    tool, arguments = copy.deepcopy(declarations()[0])
    mapped, transport = codec.decode_semantic_declaration(
        response({"tool": tool, "arguments": arguments}), max_actions=6,
        request_provenance=provenance, offered_tool_names=provenance["offered_tool_names"],
    )
    assert provenance == before and "schema_pooling" not in transport
    assert mapped["intent"] == "stage_program"
    assert all(stage["arguments"]["completion"] == "stage" for stage in mapped["stages"])
    assert transport["declaration_schema_sha256"] == digest(baseline)
    assert transport["canonical_arguments_sha256"] == digest(arguments)
    assert transport["provider_tool_call_count"] == 0 and transport["input_authority"] is False


def test_normal_pooled_request_optional_metadata_is_deep_copied_in_both_directions(pool_case):
    _, _, metadata, provenance = copy.deepcopy(pool_case)
    original = copy.deepcopy(provenance)
    tool, arguments = copy.deepcopy(declarations()[4])
    _, transport = codec.decode_semantic_declaration(
        response({"tool": tool, "arguments": arguments}), max_actions=6, request_provenance=provenance,
    )
    assert transport["schema_pooling"] == metadata
    assert transport["schema_pooling"] is not provenance["schema_pooling"]
    transport["schema_pooling"]["mappings"][0]["anchor_ref"] = "changed returned diagnostic only"
    assert provenance == original
    retained_source = transport["schema_pooling"]["mappings"][0]["source_ref"]
    provenance["schema_pooling"]["mappings"][0]["source_ref"] = "changed request diagnostic only"
    assert transport["schema_pooling"]["mappings"][0]["source_ref"] == retained_source
    assert transport["provider_tool_call_count"] == 0 and transport["input_authority"] is False


@pytest.mark.parametrize("key,value", [
    ("completion", "stage"), ("completion", "goal"), ("goal_conditions", []),
])
def test_pooling_never_reintroduces_forbidden_program_member_fields(pool_case, key, value):
    baseline, actual, _, provenance = pool_case
    tool, arguments = copy.deepcopy(declarations()[0])
    arguments["stages"][1]["arguments"][key] = value
    content = {"tool": tool, "arguments": arguments}
    for schema in (baseline, actual):
        with pytest.raises(ValidationError):
            Draft202012Validator(schema).validate(content)
    with pytest.raises(ValueError):
        codec.decode_semantic_declaration(response(content), max_actions=6, request_provenance=provenance)


@pytest.mark.parametrize("change", [
    "weaker_anchor", "missing_edge", "changed_branch", "extra_definition", "bad_hash", "generation_claim",
])
def test_independent_proof_rejects_incomplete_or_weakened_producer_claims(pool_case, change):
    baseline, actual, metadata, _ = copy.deepcopy(pool_case)
    if change == "weaker_anchor":
        target = metadata["mappings"][0]["anchor_ref"]
        parts = tokens(target)
        actual["$defs"][parts[1]]["$defs"][parts[3]] = {}
    elif change == "missing_edge":
        metadata["mappings"].pop()
        metadata["removed_definition_count"] -= 1
    elif change == "changed_branch":
        actual["oneOf"][0]["properties"]["arguments"] = {"type": "object"}
    elif change == "extra_definition":
        actual["$defs"][ROOTS[1]]["$defs"]["unexpected"] = True
    elif change == "bad_hash":
        metadata["original_schema_sha256"] = "0" * 64
    else:
        metadata["provider_generation_equivalence"] = "verified"
    if change != "bad_hash":
        metadata["pooled_schema_sha256"] = digest(actual)
        metadata["pooled_schema_bytes"] = len(canonical(actual).encode("utf-8"))
    with pytest.raises((AssertionError, KeyError, ValueError)):
        independent_pool_proof(baseline, actual, metadata)


@pytest.mark.parametrize("keyword,value", [
    ("$id", "https://example.invalid/new-scope"), ("$anchor", "scope"),
    ("$dynamicRef", "#scope"), ("$dynamicAnchor", "scope"),
    ("$recursiveRef", "#"), ("$recursiveAnchor", True),
    ("unevaluatedProperties", False), ("unevaluatedItems", False),
])
def test_unsupported_resource_or_evaluated_scope_is_exact_passthrough(pool_case, keyword, value):
    baseline = copy.deepcopy(pool_case[0])
    baseline["$defs"][ROOTS[1]][keyword] = value
    actual, metadata = codec._pool_schema_definitions(baseline)
    assert canonical(actual) == canonical(baseline)
    assert metadata["status"] == "unchanged_unsupported" and metadata["applied"] is False
    assert metadata["mappings"] == [] and metadata["removed_definition_count"] == 0


def test_cycle_is_not_cut_off_and_mislabeled_as_a_successful_proof(pool_case):
    baseline = copy.deepcopy(pool_case[0])
    definitions = baseline["$defs"][ROOTS[1]]["$defs"]
    definitions["self_cycle"] = {"$ref": f"#/$defs/{ROOTS[1]}/$defs/self_cycle"}
    actual, metadata = codec._pool_schema_definitions(baseline)
    assert canonical(actual) == canonical(baseline)
    assert metadata["status"] == "unchanged_unsupported" and metadata["mappings"] == []
    with pytest.raises(AssertionError):
        closed_acyclic(baseline)


@pytest.mark.parametrize("reference", [
    "https://example.invalid/schema", "#/$defs/v66_tool_1_arguments/$defs/missing_definition",
    "#/$defs/v66_tool_1_arguments/properties/scope",
])
def test_unclosed_external_or_non_named_reference_is_exact_passthrough(pool_case, reference):
    baseline = copy.deepcopy(pool_case[0])
    baseline["$defs"][ROOTS[1]]["properties"]["reference_probe"] = {"$ref": reference}
    actual, metadata = codec._pool_schema_definitions(baseline)
    assert canonical(actual) == canonical(baseline)
    assert metadata["status"] == "unchanged_unsupported" and metadata["mappings"] == []


def test_escaped_pointer_targets_are_not_confused_with_literal_slashes_or_tildes(pool_case):
    baseline = copy.deepcopy(pool_case[0])
    name = "escaped/name~part"
    for namespace in ROOTS[:2]:
        baseline["$defs"][namespace]["$defs"][name] = {"type": "string", "pattern": "^kept$"}
    reference = pointer(("$defs", ROOTS[1], "$defs", name))
    assert reference.endswith("/escaped~1name~0part")
    baseline["$defs"][ROOTS[1]]["properties"]["escaped_probe"] = {"$ref": reference}
    assert resolve(baseline, reference)["pattern"] == "^kept$"
    actual, metadata = codec._pool_schema_definitions(baseline)
    if metadata["status"] == "unchanged_unsupported":
        assert canonical(actual) == canonical(baseline) and metadata["mappings"] == []
    else:
        independent_pool_proof(baseline, actual, metadata)
    with pytest.raises(AssertionError):
        tokens("#/$defs/v66_tool_1_arguments/$defs/bad~2token")


def test_literals_are_opaque_and_ref_siblings_are_not_merged_or_erased(pool_case):
    baseline = copy.deepcopy(pool_case[0])
    literal = {"$ref": "#/$defs/v66_tool_1_arguments/$defs/target", "$id": "literal", "contains": True}
    last = baseline["$defs"][ROOTS[-1]]
    last.update(const=copy.deepcopy(literal), enum=[copy.deepcopy(literal)], default=copy.deepcopy(literal),
                examples=[copy.deepcopy(literal)])
    actual, metadata = codec._pool_schema_definitions(baseline)
    assert metadata["status"] != "unchanged_unsupported"
    independent_pool_proof(baseline, actual, metadata)
    for key in ("const", "enum", "default", "examples"):
        assert canonical(actual["$defs"][ROOTS[-1]][key]) == canonical(last[key])
    with_sibling = copy.deepcopy(pool_case[0])
    with_sibling["$defs"][ROOTS[1]]["properties"]["sibling_probe"] = {
        "$ref": f"#/$defs/{ROOTS[1]}/$defs/target", "maxProperties": 0,
    }
    actual, metadata = codec._pool_schema_definitions(with_sibling)
    if metadata["status"] == "unchanged_unsupported":
        assert canonical(actual) == canonical(with_sibling) and metadata["mappings"] == []
    else:
        independent_pool_proof(with_sibling, actual, metadata)
        assert actual["$defs"][ROOTS[1]]["properties"]["sibling_probe"]["maxProperties"] == 0


@pytest.mark.parametrize("limit,formal", [(1, False), (1, True), (6, True)])
def test_H1_and_formal_actual_request_bytes_stay_exact58_without_calling_pool(
    observed, monkeypatch, limit, formal
):
    def forbidden_pool(_schema):
        raise AssertionError("normal schema pool must not run in original H1/formal phases")

    monkeypatch.setattr(codec, "_pool_schema_definitions", forbidden_pool)
    tool = "declare_native_current_action" if limit == 1 else "declare_completion"
    arguments = current(1) if limit == 1 else {
        "label": "Observed editor", "conditions": [{"kind": "text_contains", "value": "Editor"}],
    }
    options = {"whole_goal_recheck": True} if formal else {}
    old_plan, old_request, _ = requested(observed, arguments, tool, fmt="semantic_tools_v58",
                                         limit=limit, planning_options=options)
    new_plan, new_request, planner = requested(observed, arguments, tool, fmt="semantic_json_v66",
                                              limit=limit, planning_options=options)
    assert new_request.content == old_request.content and new_plan == old_plan
    assert planner.last_response_transport["normal_json_schema"] is False
