# macOS native backend

The existing typed runtime now accepts a macOS Accessibility surface under `d:`.
Windows UIA remains a separate backend. `open-computer` and `open-desktop` choose
the native backend on macOS; the browser defaults to bundled Chromium there.

## Current evidence and boundaries

As of 2026-09-26, development on macOS 26.3.2 / Python 3.12 established:

- The Swift native helper compiles and reports actual system permissions.
- A **real Chromium + Python CLI** integration completed two independently
  verified actions in 1.61 seconds. Its planner was a deterministic fixture and
  its policy was Rule: this is **not** a model/Jev run or performance benchmark.
- Portable tests cover native surface composition, separate AX/GUI candidates,
  text readback, stale state, missing permissions, foreground errors, exact
  editor-document probing, recording lifecycle, and model-usage accounting.
- The development host has screen recording permission but initially lacked
  Accessibility permission. Live AX/GUI actions, TextEdit handoff, and MP4
  recording therefore remain **unverified** until that system permission is
  granted and the native smoke below passes.
- Model/Jev credentials and a text-model gateway have not yet been configured on
  this host. No model + Jev macOS result, long demo, or cost claim is published.

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
recording. A future multi-window showcase must add explicit window switching
and verify each segment before publication.

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
- GUI text input uses Command+A and Unicode CGEvents without replacing the clipboard;
  the AX route sets the editable control's AXValue directly.
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

Remaining milestones are live AX/GUI failure-path testing after permission,
real model/Jev tasks after local credentials are configured, live TextEdit and
video QA, optional Mac VLM grounding, explicit recording-window switching,
and repeated held-out tasks. A roughly twenty-action demo comes after these,
with meaningful decisions and separately reviewed macOS publication data.
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
