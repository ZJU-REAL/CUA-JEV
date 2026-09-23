# Experimental open-task browser path

The original four Windows workflows remain curated, reproducible capability packs. `open-browser` is a separate experimental path for tasks **not encoded as one of those four workflows**. It has no store-specific or site-specific selectors, scripts, or terminal rules.

This is an integration scaffold, not a claim of general computer-use capability. An external planner still needs to interpret the goal and page state; its model API will be connected and evaluated separately. The current implementation covers **one Edge/Chromium page on one web origin**, not arbitrary Windows software.

## Division of labor

1. Playwright observes the current page: URL, title, a short visible-text excerpt, and up to 60 visible interactive elements. Each element receives a temporary reference such as `e3`.
2. A low-frequency planner proposes a subgoal, up to 16 grounded `click` / `fill` / `select` options, and an observable completion condition. It cannot return code, CSS selectors, shell commands, or arbitrary tool calls.
3. The framework turns each legal option into real DOM and (when headed) physical GUI execution routes. Jev chooses a typed option/channel at each step. Candidate arguments and page text are withheld from the Jev request, though the goal, URL, title, and element labels are still sent.
4. The normal `ActionGuard`, executor, and verifier run. A failed effect check, exhausted options, or navigation triggers another planner call. A model-proposed completion condition is checked against the live page, but **is not independent semantic proof** that the user's original goal was achieved.

The planner is intentionally called less often than Jev. This is an architectural hypothesis to test, not yet a measured speed/cost result for open tasks.

## Planner HTTP contract

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

Supported success kinds are `url_contains`, `title_contains`, and `text_contains`. The planner should propose **alternatives that are valid in the current snapshot**, not a multi-step script. The framework rejects unknown references, unsupported operations, disabled or password controls, and malformed values. The endpoint accepts `Authorization: Bearer <CUA_JEV_PLANNER_API_KEY>` when set. This is a CUA-JEV adapter contract, **not** a direct OpenAI/Anthropic or other provider API format; a provider-specific adapter will be added when a model endpoint is available.

## Run

Install the browser extra and Edge. Set the planner key locally if the endpoint needs one; never commit it. To test routing without a Jev key, use `--policy rule`. For the intended Jev route, set `TYPESAFE_API_KEY` locally and use `--policy jev`.

```powershell
python -m pip install -e ".[browser]"
$env:CUA_JEV_PLANNER_API_KEY = "..."
cua-jev open-browser --goal "Find the release page" --url "https://example.org/" --planner-endpoint "https://your-planner.example/plan" --policy jev
```

By default only same-origin link navigation is eligible. Even a same-origin GET link can have side effects on a badly designed site, so use a controlled site. `--allow-form-input` additionally permits text-field writes and selects (which may trigger autosave); `--allow-external-actions` additionally permits button-like clicks with possible external effects. These flags are broad opt-ins: never use them as unattended authorization for purchases, deletion, or account changes. Password controls are always excluded. The browser blocks main-frame cross-origin navigation. `--headed` offers physical GUI alternatives in addition to DOM routes. `--trace PATH` opts into a local JSONL trace that **may contain page text and form values**; keep it private.

## What remains

- Connect a specific planner/VLM API and test genuinely unseen websites and user goals. No model quality or generalization claim is made yet.
- Add a second observer/action adapter for Windows UI Automation and a screenshot/VLM fallback for inaccessible UIs, rather than treating DOM-only success as desktop generality.
- Make completion verification independent of the planner, add human confirmation for sensitive actions, and quantify planner calls, Jev decisions, success, latency, and cost on held-out tasks.
