"""UNUSED guidance-only wrapper of55; no HTTP, parser, GUI or fallback execution."""

import copy

from .desktop_semantic_tools import _digest, _require
from .desktop_semantic_tools_v55 import decode_semantic_declaration as decode_ancestor
from .desktop_semantic_tools_v55 import encode_semantic_declaration as encode_ancestor

FORMAT = "semantic_tools_v56"
ANCESTOR_FORMAT = "semantic_tools_v55"
VISUAL_TOOL = "declare_visual_current_action"
ATOMIC_VISUAL_FILL_GUIDANCE = (
    "\nCURRENT VISUAL TEXT REPLACEMENT: When the uploaded current screenshot clearly identifies "
    "the intended editable text box, and semantic fill is unavailable or reliable focus is not "
    "established, you may choose the existing visual_fill action under its original guards. "
    "It clicks that box, selects all with Ctrl+A, and replaces its contents with exactly the "
    "literal value supplied; a separate focus-only plan is usually unnecessary. This is an "
    "option, not a forced action. Retain all visible-target, coordinate-space, fresh-image and "
    "internal execution focus/deadline guards; an unclear target does not become legal. "
    "Do not add semantic target or focus predicates to a visual declaration. "
    "A focused border or state proves neither text selection, text mutation, nor saved or "
    "persisted completion."
)
FUTURE_ROLE_GUIDANCE = (
    "\nFUTURE TARGET ROLE: For a future control or condition target whose role has not been "
    "observed for that target, OMIT role and use its expected name and app context. This "
    "also applies to postconditions: do not guess frame versus dialog, menuitem versus button, "
    "or listitem versus radio. Those extra filters can prevent a correct future UI from matching. "
    "Keep the original grounding and scope requirements; omitting role does not authorize an "
    "invented name, future field or ungrounded current input."
)
VISUAL_DESCRIPTION_SUFFIX = (
    " For the existing visual_fill operation, click the clearly visible current editable text "
    "box, use Ctrl+A and replace its contents with the supplied literal value under unchanged "
    "internal execution guards; a separate focus-only action is usually unnecessary. "
    "Focus alone does not prove text selection, mutation or persistence."
)


def _parameters(tools):
    return {tool["function"]["name"]: _digest(tool["function"]["parameters"]) for tool in tools}


def encode_semantic_declaration(body, *, max_actions, profile, contract, observation,
                                visual_box_space="normalized_1000"):
    """Preserve complete55 request; change only system suffix and one description."""
    ancestor, provenance = encode_ancestor(
        body, max_actions=max_actions, profile=profile, contract=contract, observation=observation,
        visual_box_space=visual_box_space,
    )
    _require(provenance.get("response_transport") == ANCESTOR_FORMAT, "actual55 request required")
    encoded = copy.deepcopy(ancestor)
    functions = [tool["function"] for tool in encoded["tools"]
                 if tool["function"]["name"] == VISUAL_TOOL]
    _require(len(functions) == 1, "one original visual-current declaration required")
    old_description = functions[0]["description"]
    old_system = ancestor["messages"][0]["content"]
    suffix = ATOMIC_VISUAL_FILL_GUIDANCE + (FUTURE_ROLE_GUIDANCE if max_actions > 1 else "")
    encoded["messages"][0]["content"] = old_system + suffix
    functions[0]["description"] = old_description + VISUAL_DESCRIPTION_SUFFIX
    restored = copy.deepcopy(encoded)
    restored["messages"][0]["content"] = old_system
    next(tool["function"] for tool in restored["tools"]
         if tool["function"]["name"] == VISUAL_TOOL)["description"] = old_description
    _require(restored == ancestor, "unexpected55 body or tool-parameter mutation")
    _require(_parameters(encoded["tools"]) == _parameters(ancestor["tools"]),
             "original55 parameters required")
    return encoded, {
        **provenance,
        "response_transport": FORMAT, "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_request_provenance": provenance,
        "ancestor_request_sha256": _digest(ancestor), "request_body_sha256": _digest(encoded),
        "ancestor_declaration_schema_sha256": _digest(ancestor["tools"]),
        "declaration_schema_sha256": _digest(encoded["tools"]),
        "declaration_tools_sha256": _digest(encoded["tools"]),
        "ancestor_parameter_schema_sha256": _parameters(ancestor["tools"]),
        "parameter_schema_sha256": _parameters(encoded["tools"]),
        "ancestor_system_sha256": _digest(old_system),
        "projected_system_sha256": _digest(encoded["messages"][0]["content"]),
        "generation_projection_format": FORMAT,
        "input_authority": False,
    }


def decode_semantic_declaration(payload, *, max_actions, prepare_repair_token=None):
    """Exact55 extraction/mapping; original caller must still parse and guard it."""
    raw, provenance = decode_ancestor(
        payload, max_actions=max_actions, prepare_repair_token=prepare_repair_token,
    )
    return raw, {
        **provenance,
        "response_transport": FORMAT, "requested_format": FORMAT,
        "ancestor_response_transport": ANCESTOR_FORMAT,
        "ancestor_mapping_provenance": provenance,
        "input_authority": False,
    }
