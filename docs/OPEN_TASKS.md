# Experimental open-task paths

The original four Windows workflows remain curated, reproducible capability packs. `open-browser` and `open-desktop` are separate experimental paths for tasks **not encoded as one of those four workflows**. Neither contains site- or app-specific selectors or scripts.

This is an integration scaffold, not a claim of general computer-use capability. The browser path covers **one Edge/Chromium page on one web origin**. The desktop path covers **accessible UI Automation controls in one explicitly selected Windows window**. Neither handles arbitrary dialogs, pure-canvas interfaces, or unrestricted cross-app workflows. The supplied campus model gateway was unreachable during implementation, so real model quality and compatibility remain untested.

## Division of labor

1. Playwright observes a browser page (URL, title, short visible text, up to 60 interactive elements with temporary `eN` refs). UI Automation observes a selected desktop window (title, visible named controls, up to 80 `cN` refs). Password controls are excluded.
2. A low-frequency planner proposes a subgoal, up to 16 grounded options, and an observable completion condition. The OpenAI-compatible chat adapter calls `/v1/chat/completions`; the original provider-neutral JSON endpoint is also supported for browsers. Neither accepts model-proposed code, selectors, coordinates, shell commands, or arbitrary tool calls.
3. The framework turns each legal browser option into DOM and (when headed) physical GUI routes. Desktop options can use UIA `invoke`/`set_edit_text` or physical GUI. Jev chooses a typed option/channel at each step. Candidate arguments and snapshot text are withheld from the Jev request, though the goal, titles, and element labels are still sent.
4. The normal `ActionGuard`, executor, and verifier run. A failed effect check or exhausted options triggers another planner call. A model-proposed completion condition is checked against live state, but **is not independent semantic proof** that the user's original goal was achieved.

The planner is intentionally called less often than Jev. This is an architectural hypothesis to test, not yet a measured speed/cost result for open tasks. The four published benchmark cases are unchanged.

## Direct model gateway

`ChatModelPlanner` uses the widely implemented OpenAI-style `GET /v1/models` and `POST /v1/chat/completions` wire shapes. It avoids model-specific parameters such as JSON mode and temperature, then validates the returned JSON against the live element snapshot. The campus gateway may differ from the official OpenAI API; successful live compatibility has **not** been established. Its key is read from `CUA_JEV_MODEL_API_KEY` (or `CUA_JEV_PLANNER_API_KEY`) and must never be committed.

```powershell
$env:CUA_JEV_MODEL_API_KEY = "..."
cua-jev planner-models --model-base-url "http://101.37.174.109:8010/v1" --allow-insecure-model-http
cua-jev open-browser --goal "Find the release page" --url "https://example.org/" --model-base-url "http://101.37.174.109:8010/v1" --model "MODEL_ID" --allow-insecure-model-http --policy jev
cua-jev open-desktop --goal "Find the requested item" --window-title "^File Explorer$" --model-base-url "http://101.37.174.109:8010/v1" --model "MODEL_ID" --allow-insecure-model-http --policy jev
```

The insecure-HTTP flag is required because this endpoint is plain HTTP. It exposes the bearer key and task content to the network unless protected by a trusted tunnel; prefer HTTPS or a local SSH tunnel. `planner-models` only lists IDs. Names alone do not prove vision support or planning quality; shortlist with controlled live probes and held-out tasks once reachable.

By default the desktop path is observation-only. `--allow-text-input` permits editable controls; `--allow-button-actions` permits clicks with possible external side effects. Fake UIA controls exercise both routes in tests, but arbitrary live desktop tasks have **not** been validated.

## Optional provider-neutral browser planner contract

`POST` JSON to an HTTPS endpoint (or local HTTP loopback):

```json
{
  "goal": "Find the project's release page",
  "snapshot": {
    "url": "https://example.org/",
    "title": "Example",
    "text": "Visible page excerpt...",
    "elements": [
      {"ref": "e0", "index": 0, "tag": "a", "role": "", "label": "Releases", "input_type": "", "disabled": false, "href": "https://example.org/releases"}
    ]
  },
  "recent_actions": []
}
```

Return:

```json
{
  "subgoal": "Open the releases page",
  "options": [{"ref": "e0", "operation": "click"}],
  "success": {"kind": "url_contains", "value": "/releases"}
}
```

Supported success kinds are `url_contains`, `title_contains`, and `text_contains`. The planner should propose **alternatives that are valid in the current snapshot**, not a multi-step script. The framework rejects unknown references, unsupported operations, disabled or password controls, and malformed values. The endpoint accepts `Authorization: Bearer <CUA_JEV_PLANNER_API_KEY>` when set. This is a CUA-JEV adapter contract, **not** a direct OpenAI/Anthropic or other provider API format. The chat adapter above is a separate route.

## Run with the provider-neutral endpoint

Install the browser extra and Edge. Set the planner key locally if the endpoint needs one; never commit it. To test routing without a Jev key, use `--policy rule`. For the intended Jev route, set `TYPESAFE_API_KEY` locally and use `--policy jev`.

```powershell
python -m pip install -e ".[browser]"
$env:CUA_JEV_PLANNER_API_KEY = "..."
cua-jev open-browser --goal "Find the release page" --url "https://example.org/" --planner-endpoint "https://your-planner.example/plan" --policy jev
```

By default only same-origin link navigation is eligible. Even a same-origin GET link can have side effects on a badly designed site, so use a controlled site. `--allow-form-input` additionally permits text-field writes and selects (which may trigger autosave); `--allow-external-actions` additionally permits button-like clicks with possible external effects. These flags are broad opt-ins: never use them as unattended authorization for purchases, deletion, or account changes. Password controls are always excluded. The browser blocks main-frame cross-origin navigation. `--headed` offers physical GUI alternatives in addition to DOM routes. `--trace PATH` opts into a local JSONL trace that **may contain page text and form values**; keep it private.

## What remains

- Restore connectivity to the campus gateway, inspect actual model IDs, test chat compatibility, and shortlist text planners versus vision-capable models with held-out goals. No model quality or generalization claim is made yet.
- Add a screenshot/VLM fallback for inaccessible UIs and multi-window navigation; single-window UIA is not desktop generality.
- Expose safe, typed CLI/MCP capabilities within the same open-task frontier; today the open paths offer DOM/UIA/GUI, while the curated workflows contain CLI/MCP.
- Make completion verification independent of the planner, add human confirmation for sensitive actions, and quantify planner calls, Jev decisions, success, latency, and cost on held-out tasks.
