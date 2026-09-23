# <img src="src/cua_jev/ui/static/logo-mark.svg" width="44" alt="ZJU-REAL Lab logo"> CUA-JEV: Jev for Computer Use

[Webpage](https://zjureal.com/CUA-JEV/) · [中文文档](README.zh-CN.md)

CUA-JEV is an open-source reference framework for Jev-powered computer use on Windows. A task adapter turns structured state from the browser, desktop UI, Excel, terminal, or filesystem into **legal, executable, verifiable** action candidates. [Jev](https://docs.typesafe.ai/introduction) selects an `intent × action channel`; the framework guards and executes that choice, verifies the resulting state, and observes again. The first release includes four complete workflow examples and paired Hybrid / GUI Only experiments, without training a task-specific router or requiring a VLM.

Jev's fast, typed decisions are a promising fit for downstream systems that must choose actions repeatedly under latency constraints, including computer use, embodied agents, and potentially autonomous driving. [RoboJEV](https://github.com/lykycy123/RoboJEV) already explores the embodied setting with Jev-controlled manipulation in a MuJoCo simulator. CUA-JEV explores a different setting: choosing both *what to do next* and *which computer-use channel should do it*. This is a motivation for the design, not a claim that this framework has been validated for robotics or driving.

> **Scope:** This is not a general agent that can operate any Windows application from an arbitrary instruction. Each of the four workflows currently has a task-specific adapter, candidate generator, and terminal verifier. Jev selects among constrained options; it does not generate arbitrary scripts or interpret unfamiliar screenshots.

The [experimental open-task paths](docs/OPEN_TASKS.md) separate a low-frequency model planner from Jev's typed action selection. They discover browser DOM elements or Windows UI Automation controls dynamically and offer grounded structured-tool / physical-GUI alternatives without a site- or app-specific workflow. An OpenAI-compatible model gateway adapter is available, but **the supplied campus service was unreachable during implementation; no real-model success or arbitrary-task generalization is claimed**.

![CUA-JEV architecture: task adapters, Jev selection, guarded execution, and independent verification](assets/architecture.svg)

## Why Jev × computer use?

TypeSafe AI describes Jev as a *System One* decision model for software: given state and a typed question, it returns a structured answer suitable for an application branch. Its [Choice primitive](https://docs.typesafe.ai/introduction) fits the question “which of these known actions should we take?” better than generating prose and parsing it into a tool call. TypeSafe calls its training approach RLCD; **CUA-JEV uses the existing Jev API and does not train Jev or a new CUA model**. See the [TypeSafe overview](https://typesafe.ai/) and [developer documentation](https://docs.typesafe.ai/introduction).

Computer use naturally has a hybrid action space. A subgoal may be best served by visible mouse-and-keyboard work, or by DOM, COM, CLI, MCP, or a file API. Replanning, generating commands, and parsing output with a general-purpose model at every step can add latency and model cost; a fixed router, meanwhile, may miss useful alternative routes. We narrow the open-ended planning problem to **constrained action selection**: task adapters propose real actions, Jev chooses one online, and deterministic code handles safety and factual verification.

This division of labor depends on informative structured observations and well-engineered candidates and verifiers. Jev does not replace missing perception, task decomposition, or software integration.

## How it works

1. **Observe / Generate:** A task adapter reads application state and enumerates only currently legal `intent × channel` candidates. For example, repairing a failing test can offer real VS Code GUI, MCP, restricted CLI, and file API routes.
2. **Select with Jev:** The Jev `choice` API selects a candidate ID. The chosen action and its arguments are already typed and defined; Jev does not execute arbitrary shell text.
3. **Guard / Execute:** `ActionGuard` checks observation freshness, candidate identity, path scope, write permissions, and confirmation requirements before dispatching to a registered executor.
4. **Verify / Repeat:** An executor receipt is not proof of success. An independent verifier checks the new application state; a JSONL trace records the choice, execution, timing, and evidence. The loop observes again until the terminal task condition holds.

The reusable pieces are [`ActionCandidate`](src/cua_jev/models.py), [`AgentRuntime`](src/cua_jev/runtime.py), [`ActionGuard`](src/cua_jev/guard.py), [`ExecutorRegistry`](src/cua_jev/registry.py), observer and capability interfaces, and the evaluation/trace contract. The four workflows are reference implementations of these interfaces.

### Included workflows

| Application | Goal | Competing channels in this release |
|---|---|---|
| Edge | Sign in to a public demo store, sort items, build a cart, check out, and verify the receipt | PyAutoGUI, DOM |
| Excel | Compute metrics, mark review status, create charts, and verify the workbook in a separate COM session | PyAutoGUI, COM |
| VS Code | Diagnose and repair multiple defects while repeatedly running tests | PyAutoGUI, MCP, CLI, file API |
| Explorer | Select reports from a mixed inbox and produce verifiable release artifacts | PyAutoGUI, MCP, CLI, file API |

In **GUI Only**, structured interfaces may still provide observations and element locations, but mutations use PyAutoGUI. In **Hybrid**, Jev can select among the available GUI and structured execution routes. A keyless Rule policy is also included for testing and ablations. Windows UI Automation, terminal text, and filesystem state can serve as observation channels.

## Experiments and demos

The [webpage](https://zjureal.com/CUA-JEV/) contains four workflow videos, two separate comparisons, and expandable execution records:

- **Jev Hybrid vs Jev GUI Only:** Same Jev policy and terminal verifier; different action spaces. The comparison focuses on completion time.
- **Jev Hybrid vs Codex Computer Use Hybrid:** Both may use mixed tools. The pilot compares measured wall time and estimated model cost in US dollars.

Initial v2 records from 2026-09-23 are shown below in seconds. Jev values are medians of available successful runs; Codex has one pilot run per workflow.

| Workflow | Jev Hybrid | Jev GUI Only | Codex Hybrid |
|---|---:|---:|---:|
| Edge | 79.5 | 87.4 | 77.4 |
| Excel | 84.2 | 83.2 | 39.2 |
| VS Code | 14.0 | 90.8 | 57.9 |
| Explorer | 13.2 | 211.1 | 50.9 |

VS Code and Explorer illustrate how structured routes can avoid long sequences of GUI actions. **In these Edge and Excel Hybrid runs, however, Jev still chose GUI routes; Excel was slightly slower than GUI Only.** These results do not show that Jev is faster on every task. Codex is a general-purpose tool agent, while Jev uses prebuilt task capability packs; the samples are too small for a general performance ranking. The dollar figures on the website are **model-cost estimates, not billed charges**, derived from [TypeSafe public pricing](https://docs.typesafe.ai/models) and the [OpenAI public rate card](https://help.openai.com/en/articles/20001415-chatgpt-rate-card-enterprise-token-based-pricing). Token accounting and caveats are in [`benchmarks/v2-cost-pilot-2026-09-23.json`](benchmarks/v2-cost-pilot-2026-09-23.json).

The public site is a read-only experimental snapshot. It neither calls the Jev API nor runs tasks on a visitor's computer. Video durations are not the benchmark wall times. Jev step lists come from representative run traces; Codex entries summarize recorded tool calls by phase and are not presented as equivalent atomic GUI steps. The curated publication data lives in [`website/snapshot.json`](website/snapshot.json).

## Quick start

You need Windows and Python 3.11+. Try the keyless Rule baseline first:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,all,ui]"
playwright install chromium
cua-jev doctor
cua-jev demo --policy rule
cua-jev suite --task vscode --policy rule --profile adaptive --open-vscode
```

To use Jev, put your API key in the local, Git-ignored `.env` file. **Never commit the key.**

```powershell
Copy-Item .env.example .env
# Set TYPESAFE_API_KEY=... in .env
cua-jev jev-smoke
cua-jev suite --task vscode --policy jev --profile adaptive --open-vscode
```

Use `--task edge|excel|vscode|explorer|all` to choose a workflow. `--profile adaptive` means Hybrid; `--profile visible` means GUI Only. Add `--headed-edge` for visible Edge, or `--visible-apps` for Excel/Explorer. The public store example uses a [SauceDemo](https://www.saucedemo.com/) test account and a demo order, with no real payment.

To run the local, read-only showcase:

```powershell
cua-jev-ui
# http://127.0.0.1:8768
```

To repeat paired runs and recordings:

```powershell
python -m pip install -e ".[all,ui,recording]"
python scripts/record_v2_demos.py --task all --profile both --policy jev
```

Local records are written to ignored `runs/`; raw recordings go to ignored `artifacts/demos/`. The public site uses only reviewed copies in `website/media/`. CI can run unit tests on Linux, but the four desktop workflows require Windows.

## Extend with another task

1. **Define state and success:** Implement an observer that produces compact, stable, verifiable structured state. See [`observers.py`](src/cua_jev/observers.py) and [`suites.py`](src/cua_jev/suites.py).
2. **Enumerate real alternatives:** Build an [`ActionCandidate`](src/cua_jev/models.py) for every currently legal `intent × channel`, including a stable ID, capability, arguments, risk level, and verifier. Do not inflate the candidate set with actions that cannot execute.
3. **Register capabilities and executors:** Use [`capabilities.py`](src/cua_jev/capabilities.py) and [`registry.py`](src/cua_jev/registry.py) as guides for mapping GUI/DOM/COM/CLI/MCP/API routes to actual tools. CLI actions use registered argv templates, not arbitrary `shell=True` commands.
4. **Verify and reset independently:** Assert the resulting state, define terminal success, and make the fixture repeatable. Test failed, repeated, and rejected actions. See [`verify.py`](src/cua_jev/verify.py) and [`tests/`](tests/).
5. **Evaluate before publishing:** Run Hybrid and GUI Only, recording success rate, wall time, decision time, action mix, fallback, and guard rejection—not only the best run.

The minimal JSON task in [`inspect_report.json`](src/cua_jev/predefined/inspect_report.json) introduces the candidate contract. A real new application also needs dynamic observation, executors, and terminal verification.

## Roadmap

- [x] Jev `choice`, typed candidates, shared runtime, guard, receipts, and JSONL traces.
- [x] GUI / DOM / COM / CLI / MCP / API execution interfaces and a keyless Rule baseline.
- [x] Four defined Windows workflows, independent terminal verification, Hybrid / GUI Only paired experiments, and a Codex Hybrid pilot.
- [x] Read-only website, curated experimental snapshot, real videos, and expandable execution steps.
- [ ] Move beyond four fixed cases to parameterized task families, unseen instances, adequate repeated trials, failure cases, and confidence intervals.
- [ ] Generalize task adapters across software and tasks, then explore macOS / Linux and more browser and office applications.
- [ ] Add optional VLM/visual-perception fallback for interfaces that DOM, UIA, or COM cannot describe reliably, while retaining the VLM-free path.
- [ ] Divide work with general CUA / LLM models: use them for open-ended goal interpretation and new capability construction, and Jev for frequent constrained choices; optimize jointly for success, latency, and cost.
- [ ] Improve cross-channel recovery, dynamic MCP integration, safety confirmations, and long-term regression benchmarks.

## Safety and license

`ActionGuard` fails closed on candidate identity, stale observations, allowed roots, writes, and external side effects. Excel does not run VBA; MCP can invoke only registered typed tools. A successful tool receipt still requires task-level verification. API keys are used only for local HTTPS requests and are not written to traces, site snapshots, or the Git repository.

The code is released under [Apache-2.0](LICENSE). The ZJU-REAL mark identifies the lab project and does not change ownership of third-party branding; app icon sources are listed in [`ATTRIBUTION.md`](src/cua_jev/ui/static/icons/ATTRIBUTION.md).

## Acknowledgements

We thank [TypeSafe AI](https://typesafe.ai/) for developing Jev and providing the API and documentation that make this integration possible. We also acknowledge the authors of [RoboJEV](https://github.com/lykycy123/RoboJEV) for their open embodied-agent demonstration, which helped motivate exploring Jev in another action-rich domain. CUA-JEV is an independent research and engineering project, not an official TypeSafe AI or RoboJEV product.
