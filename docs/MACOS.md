# macOS native backend

The existing typed runtime now accepts a macOS Accessibility surface under `d:`.
Windows UIA remains a separate backend. `open-computer` and `open-desktop` choose
the native backend on macOS; the browser defaults to bundled Chromium there.

## Current evidence and boundaries

As of **2026-09-28**, local tests on macOS 26.3.2 / Python 3.12 established:

| Check | Planner / policy | Actions | Model requests | Wall time | Reported tokens |
|---|---|---:|---:|---:|---:|
| Chromium + Python CLI | Qwen3.8-Flash-Next / Rule | 2: DOM 1, CLI 1 | 2 | 5.86 s | 2,101 |
| Browser + CLI + native fixture | Qwen3.8-Flash-Next / Rule | 4: DOM 1, CLI 1, AX 2 | 4 | 38.2 s | 10,304 |
| Forced native AX route | Deterministic fixture / Rule | 4: DOM 1, CLI 1, AX 2 | 0 | — | — |
| Forced native GUI route | Deterministic fixture / Rule | 4: DOM 1, CLI 1, GUI 2 | 0 | 5.60 s | — |

The model-planned native run passed independent terminal checks and finalized
its selected-window MP4. The forced GUI run also completed its recording.
Reviewed GUI frames show the expected saved text, including Chinese characters.
A separate synthetic Qwen3.8-Flash-Next plan probe passed before the live tests;
that probe alone is not application-execution evidence. Keyless browser + CLI
checks continue to pass.

These are short, local integration checks, **not repeated reliability trials or
performance benchmarks**. Tokens are provider-reported usage, not billed cost;
different modes and attempts must not be treated as paired comparisons. Earlier
GUI attempts exposed a false obstruction from the Dock's display-sized system
plane and inherited Command modifiers on Unicode events; both were corrected.
Other attempts stopped on actual obstruction or focus loss, as required by the
guards. The successful checks are not a claim that every attempted run passed.

An independent TextEdit check created a new text file, opened it in TextEdit,
verified its exact AXDocument URL and a new window ID, rechecked that ID after
window selection, and finalized a two-second selected-window recording. This
was a standalone handoff check, not part of the model-planned four-action run.
Jev credentials are not yet configured for these Mac tests, so **no Mac model +
Jev run or roughly twenty-action demo has completed**. Portable tests additionally
cover stale state, missing permissions, exact editor-document probing, recording lifecycle, and model-usage
accounting. The three-window recording-bundle helper has offline tests only;
it is not evidence of a completed multi-window demo.

The current desktop implementation supports AX-discovered buttons, checkboxes,
radio buttons, and editable controls. It offers AXPress/AXValue where available
and physical CGEvent alternatives. It does **not** yet offer macOS desktop VLM
or visual-coordinate actions; those flags fail explicitly. Browser viewport
vision remains available through Playwright with its existing upload opt-in.
Visible Chromium on macOS offers DOM/browser-mouse routes, without advertising
the Windows physical browser driver.

## Install

Use Python 3.11 or newer. The native helper requires macOS 14+ and Apple's Command
Line Tools (`swiftc`). Native single-window MP4 recording requires macOS 15+.
No pywinauto, pywin32, AppleScript, or private Accessibility API is used by the Mac
backend.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,browser-vision,mcp,ui,recording,macos]'
python -m playwright install chromium
cua-jev doctor
cua-jev macos-doctor
```

The helper source is packaged at `src/cua_jev/native/MacBridge.swift`. It is
compiled on demand to `~/Library/Caches/cua-jev/macos/cua-jev-mac`, using a source
hash, a build lock, and atomic replacement. Nothing is downloaded or installed
system-wide by that build. Compilation can take a minute on its first run.

`macos-doctor` reports permission booleans without requesting or granting them.
If Accessibility is false, allow the launcher (for example Codex or Terminal,
or the helper if macOS lists it) in **System Settings → Privacy & Security →
Accessibility**. Capture/recording also needs **Screen & System Audio Recording**.
Restart the task after changing permissions. The backend never substitutes a
full-screen screenshot when AX or window capture is unavailable.

## Local native fixture

The fixture is a normal AppKit window with a text field, Save button, checkbox,
and observable result. It operates only on in-memory test text.

```sh
# Real browser + CLI smoke; no AX, API keys, or external website required.
python scripts/macos_smoke.py --browser-only

# Starts and cleans up its own native fixture process, then tests each route.
python scripts/macos_smoke.py --route both

# Same test with separate, single-window MP4s for AX and GUI runs.
python scripts/macos_smoke.py --route both --record
```

These are deliberately deterministic integration tests. They force each route
to validate the executors and cannot be presented as Jev's autonomous choices.
Their private JSONL traces and recordings live under ignored `runs/macos-smoke/`.
Each desktop run requires both the browser URL and the exact saved-note result,
as well as actual CLI, fill, and click capability execution.

To use actual model proposals in the same bounded fixture, configure the model
key in the local `.env`, then supply the model endpoint and ID explicitly:

```sh
# MODEL_BASE_URL and MODEL_ID refer to your local configuration.
python scripts/macos_smoke.py --browser-only \
  --model-base-url "$MODEL_BASE_URL" --model "$MODEL_ID" --policy rule

# Browser + CLI + native fixture, with a selected-window recording.
python scripts/macos_smoke.py --record \
  --model-base-url "$MODEL_BASE_URL" --model "$MODEL_ID" --policy rule
```

Model mode defaults to `--route auto` and leaves all legal routes available;
it rejects forced AX/GUI route flags. Rule still makes the final candidate
selection in these commands. After configuring `TYPESAFE_API_KEY`, replace
`--policy rule` with `--policy jev` to test actual Jev decisions. No such Mac
Jev run has been validated yet. Each invocation uses fresh private output paths.

For a real text-model + Jev task, launch the fixture separately:

```sh
cua-jev macos-fixture
```

In a second terminal, activate `.venv`, configure local `.env` credentials
(`TYPESAFE_API_KEY` and `CUA_JEV_MODEL_API_KEY`), then run:

```sh
cua-jev open-desktop \
  --goal 'Enter Mac integration verified in Note text and press Save note' \
  --window-title '^CUA-JEV Local Fixture$' \
  --require-window-text-contains 'Saved: Mac integration verified' \
  --allow-text-input --allow-button-actions \
  --model-base-url 'https://YOUR_GATEWAY/v1' --model 'YOUR_MODEL_ID' \
  --policy jev --max-steps 8 \
  --trace runs/macos-model.jsonl --record-desktop runs/macos-model.mp4
```

Use new output paths for each take. The model must choose current AX refs, then
Jev chooses AX or GUI from compiled candidates. The caller-defined terminal gate
is mandatory on standalone Mac desktop tasks; a model's proposed success claim
cannot replace it. `--app-bundle-id` optionally narrows window selection. A title
pattern must match exactly one visible window; that window remains pinned by AX
identity, process, and CGWindowID after selection.

For cross-surface model tasks, use `open-computer` with `--window-title`,
`--allow-window-text-input`, `--allow-window-actions`, and independent URL/window/
capability gates. `--record-desktop NEW.mp4` records **only that selected desktop
window**, even while browser or CLI actions run; it is not a cross-application
recording. `scripts/record_macos_computer.py` adds a separate, explicit
three-window segment bundle, but it has only offline test coverage so far.
A multi-window showcase must still verify each live segment before publication.

## Execution and verification

- A new `DesktopState` protocol separates runtime identity from semantic model
  state. Mac observations contain no Windows handle. Private edit values affect
  freshness and acceptance, but are omitted from serialized observations.
- Controls are found within the selected AX window. Secure-text subtrees,
  hidden/disabled targets, and window close/minimize/zoom controls are excluded
  or rejected. This is a bounded control filter, not a general risk classifier.
- Candidates retain the observed state. Python checks it again before execution;
  the native helper independently re-reads state before emitting the action.
  Candidates are kept as private copies and consumed after an execution attempt;
  retrying requires a new observation and newly compiled routes.
- GUI input requires the selected app and window to be foreground, the target to
  be on a display and unobscured, and an AX hit test to identify the same control.
  AX and CGEvent use desktop points; screenshot pixels are not used as click
  coordinates. This avoids multiplying Retina coordinates twice.
- The obstruction filter narrowly recognizes the real system Dock's named,
  display-sized layer-20 interaction plane. Other covering windows still block
  input, and the exception requires a target with independent AX hit identity.
- GUI text input uses Command+A and Unicode CGEvents without replacing the clipboard;
  Unicode events explicitly clear inherited modifiers. The AX route sets the
  editable control's AXValue directly.
  Verification reads the actual AX value. Click verification requires semantic
  state changes; geometry changes alone do not count. Task completion is checked
  separately against caller-supplied gates.
- The capture filter is
  [SCContentFilter(desktopIndependentWindow:)](https://developer.apple.com/documentation/screencapturekit/sccontentfilter/init(desktopindependentwindow:)),
  which selects the window itself. It is never a full-display capture followed by
  cropping. Recordings omit audio and cursor, preserve normal window sizing,
  and finalize before a run is reported successful.

`--open-artifact-textedit` adds a macOS artifact handoff route. The native probe
checks the window's exact AXDocument file URL and a newly observed window ID.
The same native document probe is used for VS Code on Mac; if that editor does
not expose AXDocument, verification fails rather than accepting a matching title.
Notepad remains Windows-only. A matching document title alone is insufficient.

## Validation and remaining work

```sh
ruff check .
pytest -q -ra
python scripts/build_pages.py --build
```

The unit suite uses fake native bridges and does not grant macOS permissions.
CI on macOS additionally compiles the native helper and exercises permission
reporting. The real browser test uses bundled Chromium; Windows-specific
Edge/Excel integrations keep their existing platform/opt-in conditions.

Remaining milestones are model + Jev tasks after Jev credentials are configured,
live multi-window recording-bundle QA, optional Mac VLM
grounding, and repeated held-out tasks including failure recovery. A roughly
twenty-action demo comes after these, with meaningful decisions and separately
reviewed macOS publication data.
Existing Windows videos and case catalogs are unchanged.

Planner/VLM usage is recorded before validating response content, so malformed
responses with provider-reported usage are counted. Composite traces contain
separate planner, browser-vision, and desktop-vision totals and request counts.
Missing gateway usage remains unknown; costs are not treated as billed charges.

The bridge applies one deadline to the complete request write and response read,
including partial pipe writes. Invalid transport responses terminate the helper.
PNG captures are checked against their actual pixel dimensions and capped at
16 million pixels. Recording prevents window reselection until finalization.
The episode budget uses a monotonic clock and rechecks after planning and Jev
selection, before dispatching another action. An already-started external call
is not forcibly interrupted; its receipt is retained if it finishes over budget.
