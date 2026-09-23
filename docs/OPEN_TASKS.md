# Experimental open-task paths

The original four desktop workflows remain curated, reproducible capability packs, currently validated on Windows. `open-browser`, `open-desktop`, and `open-workspace` are experimental paths for tasks **not encoded as one of those four workflows**. They do not contain site- or app-specific click sequences.

This is an integration scaffold, not a claim of general computer-use capability. The browser path covers **one Edge/Chromium page on one web origin**. The desktop path covers **one explicitly selected Windows window**, using UI Automation and an optional screenshot/VLM target grounder. `open-workspace` adds one browser-to-VS-Code note-writing task family, with live model-generated navigation and content, Jev-selected routes, and a scoped file target. None handles arbitrary dialogs, pure-canvas interfaces, or unrestricted cross-app workflows. Broader model quality and task generalization remain untested.

## Division of labor

1. Playwright observes a browser page (URL, title, short visible text, up to 60 interactive elements with temporary `eN` refs). UI Automation observes a selected desktop window (title, visible named controls, up to 80 `cN` refs). Password controls are excluded from UIA observations. With explicit opt-in, `open-desktop` can additionally upload a cropped-window JPEG to a vision model, which returns up to 12 labeled `vN` boxes; visual boxes are **not** a reliable secret-field detector.
2. A low-frequency planner proposes a subgoal, up to 16 grounded options, and an observable completion condition. The OpenAI-compatible chat adapter calls `/v1/chat/completions`; the original provider-neutral JSON endpoint is also supported for browsers. Text plans cannot propose code, selectors, coordinates, shell commands, or arbitrary tool calls. The separate VLM can propose normalized visual boxes only, which are validated and converted to click-only GUI refs.
3. The framework turns each legal browser option into DOM and (when headed) physical GUI routes. Desktop options can use UIA `invoke`/`set_edit_text` or physical GUI; VLM targets offer GUI clicks only. Jev chooses a typed option/channel at each step. Candidate arguments, screenshot bytes, and snapshot text are withheld from the Jev request, though the goal, titles, and element labels are still sent.
4. The normal `ActionGuard`, executor, and verifier run. A failed effect check or exhausted options triggers another planner call. A model-proposed completion condition is checked against live state, but **is not independent semantic proof** that the user's original goal was achieved.

The long-term design aims to call the planner less often than Jev by reusing a grounded subgoal across several constrained decisions. The current pilot called both once per step, so this is still an architectural hypothesis, not a measured speed/cost result for open tasks. The four published benchmark cases are unchanged.

## Browser-to-VS-Code pilot

`open-workspace` observes a live same-origin browser page and the state of one new `.md`/`.txt` output file plus its VS Code window. The model proposes a grounded browser link, a browser-history return, an exact quote under a requested page heading, a source-backed note, or opening the note; its output cannot contain arbitrary code, selectors, coordinates, commands, or another file path. The framework gates operations using live state and optional user-specified source clues or research-topic headings. Jev chooses among browser DOM/physical GUI routes and, when available, file API/GUI routes. Quotes must occur in the live main-page text, and research sources must be distinct pages. The note is read back before the episode succeeds.

The new five-source research case starts at the official Python tutorial index with only the five desired chapter headings and a new output path. A model follows live links, records one quote per chapter, writes a guide, and opens it in VS Code. No chapter URL or click sequence is stored in the agent. One successful run completed **12 Jev decisions**: five DOM navigations, five evidence captures, one file write, and one VS Code launch. The [separate research verifier](../scripts/verify_research_demo.py) reopens every cited page in a fresh browser and checks headings, exact quotes, source URLs, and the Jev-only trace. It does **not** judge whether the synthesis is pedagogically excellent.

The recorded `dashscope/qwen3.5-plus` run used 12 model plans and 12 Jev decisions in 155.9 seconds; its video is uniformly sped up 2.5×. A separate successful `DeepSeek-V4-Flash` run took 307 seconds, including 287 seconds waiting for model plans. Neither demonstrates planner-call reduction or an end-to-end speed/cost advantage for open tasks. Campus gateway timeouts and invalid plans were also observed in failed attempts. The published recording is one independently checked take, not a reliability statistic. **That recording used text + DOM, not the new VLM path.** Broader arbitrary-task support still requires new adapters, safe executors, confirmation policies, and repeated evaluation.

```powershell
$env:CUA_JEV_MODEL_API_KEY = "..."
cua-jev open-workspace --goal "Study five official Python tutorial chapters and open a sourced guide in VS Code" `
  --url "https://docs.python.org/3/tutorial/" --note "artifacts/open-workspace/new-guide.md" `
  --model-base-url "https://YOUR_MODEL_GATEWAY/v1" --model "YOUR_MODEL_ID" `
  --research-topic "Whetting Your Appetite" `
  --research-topic "Using the Python Interpreter" `
  --research-topic "An Informal Introduction to Python" `
  --research-topic "More Control Flow Tools" `
  --research-topic "Data Structures" --max-steps 20 --policy jev
```

For an HTTP-only campus gateway, add `--allow-insecure-model-http` only on a trusted network. The local `--trace` option may record full page and note content. Do not publish raw traces or credentials.

## Direct model gateway

`ChatModelPlanner` uses the widely implemented OpenAI-style `GET /v1/models` and `POST /v1/chat/completions` wire shapes. It avoids model-specific parameters such as JSON mode and temperature, then validates the returned JSON against the live element snapshot. A small live smoke test succeeded with one campus gateway; provider-specific compatibility is not guaranteed. Its key is read from `CUA_JEV_MODEL_API_KEY` (or `CUA_JEV_PLANNER_API_KEY`) and must never be committed.

```powershell
$env:CUA_JEV_MODEL_API_KEY = "..."
cua-jev planner-models --model-base-url "http://YOUR_CAMPUS_GATEWAY:8010/v1" --allow-insecure-model-http
cua-jev open-browser --goal "Find the release page" --url "https://example.org/" --model-base-url "http://YOUR_CAMPUS_GATEWAY:8010/v1" --model "MODEL_ID" --allow-insecure-model-http --policy jev
cua-jev open-desktop --goal "Find the requested item" --window-title "^File Explorer$" --model-base-url "http://YOUR_CAMPUS_GATEWAY:8010/v1" --model "MODEL_ID" --allow-insecure-model-http --policy jev
```

The insecure-HTTP flag is required because this endpoint is plain HTTP. It exposes the bearer key and task content to the network unless protected by a trusted tunnel; prefer HTTPS or a local SSH tunnel. `planner-models` only lists IDs. Names alone do not prove vision support or planning quality; shortlist with controlled live probes and held-out tasks once reachable.

The model adapter connects directly by default, ignoring OS/environment proxies. This matters on Windows when a system proxy intercepts campus HTTP requests. Pass `--use-env-proxy` only if your model gateway actually requires that proxy.

### Optional one-window screenshot/VLM grounding

`open-desktop` accepts a separate `--vision-model` alongside the text planner. `fallback` sends a screenshot only when UIA reports no actionable control; `always` sends it every observation. The image is a resized JPEG of the selected window rectangle, held in memory and omitted from trace/Jev choices. The VLM returns visible labels and bounded `0..1000` boxes. The planner may reference these as `vN`; the framework permits click only, then checks the current window geometry, UIA state, and image freshness before a physical GUI click. After the click, UIA or pixel change checks **effect**, not semantic task completion; completion still needs a live UIA-observable condition.

```powershell
cua-jev open-desktop --goal "Complete a controlled window task" --window-title "^Your Test Window$" `
  --model-base-url "https://YOUR_MODEL_GATEWAY/v1" --model "TEXT_MODEL_ID" `
  --vision-model "VISION_MODEL_ID" --vision-mode fallback `
  --allow-screenshot-upload --allow-visual-clicks --allow-button-actions --policy jev
```

Both upload and visual clicks are deliberate opt-ins; clicking is also gated by `--allow-button-actions`. Use only a non-sensitive controlled window. An unrelated overlay inside the window rectangle could also appear in the uploaded screenshot. Unit and mocked integration tests pass, but a current 90-second campus-gateway image request timed out: **no completed live VLM task or arbitrary-task success is claimed**. The provider-neutral browser endpoint and `open-workspace` do not use this VLM route.

### Provisional model shortlist

The accessible gateway advertised these IDs, and minimal live calls established the following—not benchmark rankings:

| Model ID | Initial role | Evidence |
|---|---|---|
| `dashscope/qwen-flash` | Default structured-state planner candidate | Returned a valid grounded browser plan; a short public-site task completed with Jev. |
| `dashscope/qwen3-vl-32b-instruct` | Experimental screenshot/VLM grounder | Previously accepted a trivial image test and is now wired to the optional desktop observer; current live grounding probe timed out. |
| `dashscope/qwen-plus`, `dashscope/qwen3.5-plus` | Escalation candidates | Returned valid plans on the same tiny prompt, but were slower in one-shot calls. |

The [Qwen Flash model page](https://help.aliyun.com/en/model-studio/qwen-flash) and [visual-model documentation](https://help.aliyun.com/en/model-studio/vision-model/) describe provider capabilities, but the campus gateway's actual behavior must be evaluated separately. One prompt and one image do not establish reliability, latency distributions, or cost under real CUA workloads.

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

- Evaluate the short-listed text planner and vision-capable model on held-out goals with repeated runs, cost/latency measurement, and failure analysis. No broad model quality or generalization claim is made yet.
- Run and independently verify a live VLM-grounded desktop task; evaluate screenshot safety, box quality, success, latency, and cost on repeated held-out tasks. The current code path is single-window only, not multi-window desktop generality.
- Expose a broader registry of safe, typed CLI/MCP capabilities within the same open-task frontier; `open-workspace` currently adds only a scoped VS Code launch and note write, while the curated workflows contain richer CLI/MCP actions.
- Make completion verification independent of the planner, add human confirmation for sensitive actions, and quantify planner calls, Jev decisions, success, latency, and cost on held-out tasks.
