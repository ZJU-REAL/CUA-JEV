# Experimental open-task paths

The original four desktop workflows remain curated, reproducible capability packs, currently validated on Windows. `open-browser`, `open-desktop`, and `open-workspace` are experimental paths for tasks **not encoded as one of those four workflows**. They do not contain site- or app-specific click sequences.

This is an integration scaffold, not a claim of general computer-use capability. The browser path covers **one Playwright browser page on one web origin**, optionally fusing DOM and viewport vision and an opt-in scoped read-only local tool pack. The desktop path covers **one explicitly selected Windows window**, using UI Automation and optional screenshot/VLM grounding. `open-computer` composes the browser with an optional desktop surface and registered tools in one bounded loop; its no-window mode can use Playwright Chromium on a non-Windows host, but no macOS run has been verified. `open-workspace` adds one browser-to-VS-Code note-writing task family, with live model-generated navigation and content, Jev-selected routes, and a scoped file target. None handles arbitrary dialogs, unrestricted cross-app workflows, or general pure-canvas task completion. Broader model quality and task generalization remain untested.

## Division of labor

1. Playwright observes a browser page (URL, title, short visible text, up to 60 interactive elements with temporary `eN` refs and normalized visible boxes). UI Automation observes a selected desktop window (title, visible named controls, up to 80 `cN` refs). Password controls are excluded from structured observations. With explicit opt-in, `open-browser` uploads a viewport-only JPEG and `open-desktop` uploads a selected-window JPEG; the VLM returns a bounded scene summary, short visible-text excerpts, and up to 12 labeled `vN` boxes. Visual boxes are **not** a reliable secret-field detector.
2. A planner proposes a subgoal, up to 16 grounded options, and an observable completion condition. The OpenAI-compatible chat adapter calls `/v1/chat/completions`; the original provider-neutral JSON endpoint is also supported for browsers. Text plans cannot propose code, selectors, coordinates, shell commands, or arbitrary tool calls. With explicit `--local-tool-root`, the browser snapshot also offers bounded `tN` refs from a registered read-only pack; the planner can only invoke those refs, not change their paths or arguments. The separate VLM's bounded scene text is fused with DOM/UIA state before planning; its normalized boxes are validated and converted to click-only refs.
3. The framework turns each legal browser option into DOM and (when headed on Windows) physical GUI routes. Browser visual refs can use a Playwright mouse script; a DOM element also receives that alternative only when its label and box overlap a live VLM target. Desktop options can use UIA `invoke`/`set_edit_text` or physical GUI; desktop VLM targets offer GUI clicks only. Jev chooses a **concrete typed action and route** at each step. Candidate arguments, screenshot bytes, verbatim OCR lines, and snapshot text are withheld from the Jev request, though the bounded visual summary, goal, titles, and element labels are still sent.
4. The normal `ActionGuard`, executor, and verifier run. A failed effect check or exhausted options triggers another planner call. A model-proposed completion condition is checked against live state, but **is not independent semantic proof** that the user's original goal was achieved. Browser episodes can additionally require an independently supplied tool capability and URL substring before reporting success.

The long-term design aims to call the planner less often than Jev by reusing a grounded subgoal across several constrained decisions. The current pilot called both once per step, so this is still an architectural hypothesis, not a measured speed/cost result for open tasks. The four published benchmark cases are unchanged.

For repeated runs, keep each run's private JSONL trace locally and aggregate only numerical results with `cua-jev analyze-traces runs/*.jsonl`. The command reports completed episodes, success rate, median/p95 wall and Jev decision time, mean confidence/choice entropy, selected channels, and mean probability mass by *available* channel. It also sums provider-reported token counts where present. It deliberately leaves `cost_usd` unset until the actual model usage and a dated rate card are supplied. One synthetic-image probe or one completed case is not a benchmark.

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

### Optional scoped browser + local-tool actions

`--local-tool-root` explicitly registers a read-only pack for one directory. The planner sees opaque `tN` offers alongside live browser refs and may propose `{"ref":"tN","operation":"invoke"}`. The pack fixes the arguments; a forged ref, click operation on a tool, model-supplied path, or executable command is rejected. Offers currently include a bounded directory list, at most 16 KB from selected top-level `.md`/`.txt` files, `python --version`, and `git status --short` if the directory is a Git repository. Hidden files, symlinks, and other file types are not offered as read targets. This is **not a general MCP discovery or arbitrary CLI interface**.

```powershell
cua-jev open-browser --goal "Check local Python version and open More Control Flow Tools" `
  --url "https://docs.python.org/3/tutorial/index.html" `
  --model-base-url "https://YOUR_MODEL_GATEWAY/v1" --model "YOUR_TEXT_MODEL" `
  --local-tool-root "PATH_TO_A_CONTROLLED_DIRECTORY" `
  --require-tool cli.python_version --require-url-contains "/3/tutorial/controlflow.html" `
  --policy jev --trace "runs/private-browser-cli.jsonl"
```

Only use a non-sensitive directory: the planner receives the offered filenames and any selected tool output; the private trace can contain full receipts. Jev sees the available typed actions but **not** file text or tool result bodies. `--require-tool` verifies that a registered capability actually ran, and `--require-url-contains` checks the live URL independently of the planner's success claim. A single real run with `dashscope/qwen-flash` and Jev completed CLI then DOM in 7.0 s; Jev chose the CLI option with 0.99 probability on the first decision. This is not a reliability estimate. The browser remains locked to its starting origin; VLM was not used in that run.

A second real CLI + DOM run enabled `dashscope/qwen3-vl-32b-instruct` in `always` mode. It completed in 51.4 s, but one of two visual attempts returned an invalid scene and fell back to DOM; the other yielded a valid scene. Visual requests consumed 45.1 s. These two runs are diagnostic pilots, not paired performance evidence. The CLI now reports `vision_attempts`, `vision_calls` (valid scenes), and `vision_failures` separately.

### One browser + one selected Windows window

`open-computer` composes the existing browser and optional Windows UIA adapters with scoped local-tool, MCP, and create-only artifact surfaces. The planner sees namespaced live refs (`b:eN`, `b:vN`, `d:cN`, `d:vN`, `t:tN`, `m:mN`, `a:a0`), not selectors, handles, paths, or commands. Each accepted option is recompiled by its original adapter into real DOM, browser mouse, UIA, physical GUI, CLI, MCP, or file API candidates. Invalid options are discarded without discarding valid alternatives; a wholly invalid plan gets one feedback-informed retry. Jev selects one candidate, the normal guard executes it, and the originating adapter verifies the effect. Browser navigation remains on one origin; if selected, the desktop regex must match exactly one visible window. Common English/Chinese Close/Minimize/Maximize labels are filtered, but this is not a universal safety classifier. Terminal success uses **caller-supplied** URL/window/capability/visit/artifact gates; at least one observable state gate is mandatory.

The host now uses the `ActionSurface` contract in [`surface_providers.py`](../src/cua_jev/surface_providers.py): each provider declares its namespace and capabilities, observes/captures state, validates model refs, compiles typed executable candidates, owns and executes its candidates, and registers independent effect verifiers. `BrowserSurface`, `WindowsUiaSurface`, and `ReadOnlyToolSurface` implement it. A replacement desktop provider can be supplied under `d`; a fake accessibility backend passes an end-to-end test without altering the planner or Jev loop. This is an **extension point**, not a working macOS backend. `ComputerSnapshot.for_model()` omits runtime handles/geometry from the model request but keeps refs and semantic labels. Freshness and execution still use the full live snapshot; raw trace files may contain sensitive page/window data and remain local.

For example, on a Chinese-localized Windows system with Calculator already open:

```powershell
cua-jev open-computer --goal "Check Python version, open More Control Flow Tools, and press 七 in Calculator" `
  --url "https://docs.python.org/3/tutorial/index.html" --window-title "^计算器$" `
  --model-base-url "https://YOUR_MODEL_GATEWAY/v1" --model "YOUR_TEXT_MODEL" `
  --local-tool-root "PATH_TO_A_CONTROLLED_DIRECTORY" --allow-window-actions `
  --require-capability cli.python_version --require-capability desktop.click `
  --require-url-contains "/3/tutorial/controlflow.html" `
  --require-window-text-contains "显示为 7" --policy jev
```

Use a window-title regex matching **your system locale**. `--allow-window-actions` authorizes model-proposed clicks within that selected app; test only in a disposable, non-sensitive window. Optional browser and desktop vision models require `--allow-screenshot-upload`. A real three-action CLI → DOM → UIA pilot finished in 8.5 s with live URL and Calculator display checks. An earlier attempt stopped at an invalid desktop plan; the planner's UIA `invoke` wording is now normalized to a typed `click` only for a currently invokable control. A separate run enabled Calculator-window VLM grounding: three valid visual scenes, the same three selected channels, 48.5 s wall time, 28.2 s in VLM requests, and 14.2 s in planner requests. The planner chose the structured “七” control and Jev preferred UIA to GUI (0.89 vs 0.11); it did not select a visual-coordinate target. These are single-case integration checks, not evidence of arbitrary cross-app capability or a speed advantage.

After the provider refactor, a different live goal (Python version, the Data Structures tutorial chapter, Calculator “8”) passed three caller-supplied gates via CLI → DOM → UIA in 15.4 s. Three text-planner calls reported 15,930 tokens. The run had no VLM and was not paired with the previous goal; it is a regression smoke test, not evidence that prompt compaction saved tokens or money.

`--window-title` is now optional. Without it, the same loop accepts browser DOM/visual refs and registered `t:`/`m:` tools, but no `d:` refs or window-state gates. With `--browser-channel chromium`, this removes the Windows UIA dependency from that task path; a real MCP-protocol + mocked-browser integration test passes. A live macOS run, however, has not yet been performed.

#### Registered MCP calls

`open-computer` can add an `m:` action surface using a **trusted local** JSON profile. The profile binds one stdio server process and up to eight registered calls. Calls may have fixed arguments, bounded model-filled primitive parameters, or both. For example (replace both absolute paths with paths on your machine):

```json
{
  "server": {
    "name": "research",
    "command": "C:/absolute/path/to/python.exe",
    "args": ["C:/absolute/path/to/trusted_server.py"]
  },
  "tools": [
    {
      "name": "lookup",
      "description": "Look up the caller-selected reference",
      "arguments": {"collection": "fixed-by-caller"},
      "parameters": {"query": {"type": "string", "maxLength": 120}},
      "required": ["query"],
      "expected_from_arguments": {"query": "query"}
    }
  ]
}
```

Install `.[mcp]`, then add `--mcp-profile PATH --allow-mcp-actions` to `open-computer`. The model sees a namespaced offer such as `m:m0`, its description, tool name, and allowed parameter schema. For the example above it may propose `{"ref":"m:m0","operation":"invoke","value":"{\"query\":\"new topic\"}"}`. The runtime accepts only registered primitive fields with declared bounds or enum values; arbitrary path/file/command/secret-like parameter names are rejected. It never accepts a model-supplied server command, tool name, or override to fixed arguments. Jev chooses among the compiled MCP and other channel candidates. The candidate is guarded as `EXTERNAL_SIDE_EFFECT`, because even a nominally read-only MCP tool or its server startup might mutate state. Only use a server you trust in a controlled environment. The profile's arguments may appear in an opt-in **local** trace; never put secrets in them or commit private traces. Each profile tool must declare a nonempty static expected JSON subset and/or an output-to-argument equality check. This verifies the declared fields, **not the truth of arbitrary third-party data or overall task success**; caller-supplied terminal gates remain separate.

The end-to-end regression test starts a real MCP stdio server and performs MCP → browser DOM → desktop UIA actions in one episode. A separate real-protocol test supplies a previously unseen bounded string parameter. Neither uses the campus model/Jev APIs or establishes general MCP tool discovery, arbitrary argument schemas, or cross-app success rates. Those remain future work.

#### Cited multi-page artifact path

For a bounded research-style goal, `--require-visit-url-contains CLUE` records a live browser visit to each caller-chosen URL clue. `--artifact-path NEW.md` exposes a create-only file action only after the visit gates pass; the artifact must contain every visited full URL and is read back for exact-content verification. `--artifact-contains TEXT` adds an optional caller-defined text check. These gates establish visits, citations, and file integrity, **not factual quality of every sentence**. Source excerpts come from the page's semantic `main`/`article` content when present; a navigation exposing a new URL with no interactive DOM receives a bounded readiness retry.

The optional [`mcp_web_reader.py`](../src/cua_jev/mcp_web_reader.py) is a small, same-origin HTTPS HTML reader for a trusted local MCP profile. It caps response size, rejects redirects and non-HTML content, and returns bounded title/text fields. `--mcp-current-page-only` restricts a registered `href` parameter to the current browser URL and hides the offer after one verified read of that page. Add `--mcp-read-for-visits` to require one such read per required browser visit before the final artifact can be written. This makes 18–20-step research runs possible with genuinely distinct browser and MCP actions rather than repeated calls to the same URL. These switches are **task acceptance constraints**, not autonomous general-purpose web research or proof of source truth. The profile command remains caller-controlled and must be trusted.

An unrecorded live three-chapter run with `dashscope/qwen-plus-latest` and Jev passed the visit, CLI, MCP, citation, and file gates in **11 external actions / 147.4 s**: 4 DOM clicks, 1 CLI version check, 5 MCP calls, and 1 file write. The five MCP calls included redundant reads, prompting the current-page-once mode above. A deterministic **20-action** regression covers nine simulated browser sources, real MCP stdio, CLI, and a real file artifact; its planner and browser are fixtures. An initial eight-chapter live attempt stopped on transient empty DOM, and a retry met a campus gateway timeout. After the DOM-ready retry and an available model were used, a live eight-chapter run completed **18 actions / 251.4 s**; a visible Edge recording repeated the 18 actions in **240.7 s**. Those eight chapter clues were supplied by the caller, so this is long-horizon integration evidence, not arbitrary-task support. Private traces and raw video remain ignored under `runs/` and `artifacts/`.

#### Model-discovered sources and editor handoff

`--min-verified-sources N` is an alternative to the fixed visit-clue list. The model discovers relevant non-start pages from current DOM links, but the framework counts a source only after a successful current-page MCP read. URL fragments are one document, not separate sources. The new artifact cannot be written until the minimum is met and must cite each verified full URL. Optional `--require-source-title-contains TOPIC` clues require particular topical page titles without supplying URLs or selectors. The browser observer can retain relevant links beyond the first 60 DOM elements. `--open-artifact-vscode` then offers a CLI action to open the exact new file; verification checks a new VS Code window handle, not just a possibly stale same-name title.

One real four-topic run selected pages for control flow, modules, input/output, and errors itself, then opened its cited guide in VS Code: **12 actions / 173.8 s**, with 5 DOM, 4 MCP, 2 CLI, and 1 file-API actions. The guide and window-only recording were reviewed locally. Earlier development takes exposed three distinct problems: a source-count-only run misattributed one section; a topic-gated run found all sources but exhausted an 18-step budget after navigation detours; another run opened the artifact but a stale same-name VS Code window prevented reliable verification. The final run followed topical gates, late-link prioritization, and exact new-window verification. These are development iterations, **not independent repeated trials or an empirical success rate**. Source titles and cited URLs still do not verify every claim in a generated guide.

```powershell
cua-jev open-computer --goal "Research four topics and create a cited guide" `
  --url "https://docs.python.org/3/tutorial/index.html" `
  --model-base-url "YOUR_MODEL_GATEWAY/v1" --model "YOUR_TEXT_MODEL" `
  --mcp-profile "PATH_TO_TRUSTED_SAME_ORIGIN_READER.json" --allow-mcp-actions `
  --mcp-current-page-only --min-verified-sources 4 `
  --require-source-title-contains "Control Flow" `
  --require-source-title-contains "Modules" `
  --require-source-title-contains "Input and Output" `
  --require-source-title-contains "Errors and Exceptions" `
  --artifact-path "runs/new-guide.md" --open-artifact-vscode `
  --require-capability artifact.open_vscode --policy jev
```

An additional five-action model-discovered task enabled browser VLM `always` mode, DOM and visual-mouse candidates. It completed by DOM/MCP/file API, but **four of five VLM attempts failed validation or request**, with only one valid scene; visual requests took 115.2 s of the 187.0 s wall time and Jev selected no visual action. For DOM-rich pages, `open-computer` now defaults to `--browser-vision-mode fallback`; choose `always` explicitly only when testing the visual path. This run demonstrates fallback, not a visual advantage or reliable multimodal generality.

An additional mocked editor-style task tests UIA `fill` plus browser navigation. The terminal gate re-reads the edit control's value; observed edit values influence freshness/acceptance but are omitted from the serialized desktop snapshot sent to the planner and trace. The user-authored goal/acceptance string may of course still contain the same text. This test is not a live Notepad validation.

### Optional browser viewport vision

Install `.[browser-vision]` and `python -m playwright install chromium`, then select `--browser-channel chromium` to avoid depending on an installed Edge/Chrome binary. This enables the experimental browser path on platforms with Playwright Chromium; a live macOS run has not yet been verified. With `--vision-model MODEL_ID --vision-mode always --allow-screenshot-upload`, the VLM reads only the current viewport and gives the planner bounded scene text. Add `--allow-visual-clicks --allow-external-actions` only in a controlled task to make `vN` browser-mouse clicks executable. Even then, the runtime checks origin, DOM fingerprint, viewport size, and screenshot freshness before clicking; an image change is **not** independent semantic task verification. If an optional VLM response is malformed but live DOM controls remain, the browser continues with structured observations and reports a vision failure count.

A single real public Python-tutorial navigation completed with both DOM and visual actions offered. Jev assigned **0.99** to the DOM click and **0.01** to the visual browser-mouse click, selected DOM, and the live URL/text check passed in **9.8 s** (VLM 4.7 s, text planner 1.9 s, Jev 0.66 s). The run used `dashscope/qwen-flash` plus `dashscope/qwen3-vl-32b-instruct`. Earlier attempts failed on malformed VLM replies and a malformed plan; this is a pipeline smoke test, not evidence of a general success rate, faster end-to-end operation, or arbitrary-task support. The 12-action cross-app recording remains text + DOM only.

### Optional one-window screenshot/VLM grounding

`open-desktop` accepts a separate `--vision-model` alongside the text planner. `fallback` sends a screenshot only when UIA reports no actionable control; `always` sends it every observation. The image is a resized JPEG of the selected window rectangle, held in memory and omitted from trace/Jev choices. The VLM returns a bounded scene summary, short visible text, and labeled `0..1000` boxes. The planner may reference these as `vN`; the framework permits click only, then checks the current window geometry, UIA state, and image freshness before a physical GUI click. After the click, UIA or pixel change checks **effect**, not semantic task completion; completion still needs a live UIA-observable condition.

```powershell
cua-jev open-desktop --goal "Complete a controlled window task" --window-title "^Your Test Window$" `
  --model-base-url "https://YOUR_MODEL_GATEWAY/v1" --model "TEXT_MODEL_ID" `
  --vision-model "VISION_MODEL_ID" --vision-mode fallback `
  --allow-screenshot-upload --allow-visual-clicks --allow-button-actions --policy jev
```

Both upload and visual clicks are deliberate opt-ins; clicking is also gated by `--allow-button-actions`. Use only a non-sensitive controlled window. An unrelated overlay inside the window rectangle could also appear in the uploaded screenshot. Unit and mocked integration tests pass. A new direct, proxy-bypassing campus-gateway call with a **synthetic** Settings/Save image returned valid scene text and a bounded Save box in 7.9 s, with 351 reported tokens. This is a wire/grounding probe, **not** a completed live VLM + Jev task or arbitrary-task evaluation. The provider-neutral browser endpoint and `open-workspace` do not use this VLM route.

### Provisional model shortlist

The accessible gateway advertised these IDs, and minimal live calls established the following—not benchmark rankings:

| Model ID | Initial role | Evidence |
|---|---|---|
| `dashscope/qwen-flash` | Default structured-state planner candidate | Returned a valid grounded browser plan; a short public-site task completed with Jev. |
| `dashscope/qwen3-vl-32b-instruct` | Experimental screenshot/VLM grounder | A direct synthetic-image call returned a scene summary, OCR excerpts, and one bounded Save target in 7.9 s; this is not a desktop-task success rate. |
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

Install the browser extra and Edge, or install Playwright Chromium and select `--browser-channel chromium`. Set the planner key locally if the endpoint needs one; never commit it. To test routing without a Jev key, use `--policy rule`. For the intended Jev route, set `TYPESAFE_API_KEY` locally and use `--policy jev`.

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
