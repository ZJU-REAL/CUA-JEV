"""Lossless browser control compression. Execution observations are never changed."""
import json
from collections import Counter
from copy import deepcopy

INSTRUCTIONS = (
    ' snapshot.control_defaults contains shared control field values. For each control, '
    'inherit those defaults and then apply every explicit field override. Refs are literal '
    'observation-local targets. This is lossless encoding, not omitted UI evidence.'
)


def wire_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())


def compact_browser_snapshot(snapshot):
    """Factor repeated JSON fields, preserving values, field presence, and order."""
    original = snapshot.to_dict() if hasattr(snapshot, 'to_dict') else snapshot
    # JSON normalization matches the model's actual wire format (tuple -> list).
    data = json.loads(json.dumps(original, ensure_ascii=False, allow_nan=False))
    controls = data.get('controls', [])
    if len(controls) < 2:
        return data
    common = set.intersection(*(set(c) for c in controls)) - {'ref'}
    defaults = {}
    for key in sorted(common):
        counts = Counter(json.dumps(c[key], ensure_ascii=False, sort_keys=True) for c in controls)
        value, count = counts.most_common(1)[0]
        # Include the cost of spelling the default once and its instruction.
        if count > 1 and (count - 1) * (len(key) + len(value) + 4) > 24:
            defaults[key] = json.loads(value)
    compact = deepcopy(data)
    compact['control_defaults'] = defaults
    compact['controls'] = [{k: v for k, v in c.items()
                            if k not in defaults or json.dumps(v, sort_keys=True) !=
                            json.dumps(defaults[k], sort_keys=True)} for c in controls]
    # Refuse expansion, accounting conservatively for the explanatory prompt.
    if wire_size(compact) + len(INSTRUCTIONS.encode()) >= wire_size(data):
        return data
    assert expand_browser_snapshot(compact) == data
    return compact


def expand_browser_snapshot(snapshot):
    data = deepcopy(snapshot)
    defaults = data.pop('control_defaults', {})
    data['controls'] = [{**deepcopy(defaults), **c} for c in data.get('controls', [])]
    return data
