# Experiment context

The current [README](../README.md) presents the latest four recorded Windows
model + Jev cases and the separate macOS validation status. This page preserves
the context for earlier results; it is not a leaderboard or a pooled benchmark.

## Earlier task-specific Windows workflows

The Edge, Excel, VS Code, and Explorer workflows used prebuilt capability packs
and terminal verifiers. Hybrid allowed both GUI and structured routes; GUI Only
restricted mutations to PyAutoGUI, while structured state could still support
observation and element location. The keyless Rule policy is a deterministic
testing baseline, not a Jev run.

Initial v2 records from 2026-09-23 are shown below in seconds. Jev values are
medians of available successful runs; Codex has one pilot per workflow.

| Workflow | Jev Hybrid | Jev GUI Only | Codex Hybrid |
|---|---:|---:|---:|
| Edge | 79.5 | 87.4 | 77.4 |
| Excel | 84.2 | 83.2 | 39.2 |
| VS Code | 14.0 | 90.8 | 57.9 |
| Explorer | 13.2 | 211.1 | 50.9 |

VS Code and Explorer benefited from structured routes in these runs. Edge and
Excel Hybrid still selected GUI; Excel was slightly slower than GUI Only.
These results do not show a universal Jev speed advantage. Codex was a general
tool agent while Jev used task-specific capabilities, and the sample is too small
for a general ranking. Cost figures are estimates, not billed charges; token
allocation and pricing assumptions are retained in the reviewed
[snapshot](../website/snapshot.json).

Videos, paired comparisons, and expandable records remain on the
[Early work page](https://zjureal.com/CUA-JEV/early-work.html). Video playback
duration differs from benchmark wall time; representative Jev steps and Codex
phase summaries are not equivalent atomic GUI actions.

## Progression to model-proposed actions

The earlier twelve-decision research pilot started at the Python tutorial index,
with five caller-supplied chapter headings but no chapter URLs or click sequence.
It performed five DOM navigations, five internal evidence records, one file API
write, and one VS Code launch. The internal records were framework state updates,
not external computer-use actions. Its separate
[source verifier](../scripts/verify_research_demo.py) checked cited page headings,
quotations, file content, and Jev decisions, rather than teaching quality.

The recorded `dashscope/qwen3.5-plus` run made twelve plans and twelve Jev choices
in 155.9 seconds; its video plays at 2.5× speed. A separate successful
`DeepSeek-V4-Flash` run took 307 seconds, with 287 seconds in planner requests.
Gateway timeouts and malformed plans also caused failed development attempts.
These are not repeated success-rate or latency measurements.

Later integration pilots added scoped CLI, Windows UIA, registered MCP calls,
model-discovered sources, and exact editor handoff. The detailed progression,
commands, verification constraints, VLM failures, and bounded model shortlist
are documented in [OPEN_TASKS.md](OPEN_TASKS.md). The four current 18–21-action
recordings have a separate reviewed [catalog](../website/windows_demos.json).
They made one planner request and one Jev decision per action, with no VLM calls.

## What to measure next

- Repeated held-out goals, successful and failed runs, and failure categories.
- Independent outcome and artifact-quality checks beyond source-count gates.
- Real selected GUI/visual routes, not just availability in a candidate list.
- End-to-end latency, planner/Jev/VLM request counts, reported token use, and cost
  assumptions, with unknown usage left unknown.
- Each platform separately. Mac's deterministic AX/GUI checks, model + Rule
  short tasks, and independent TextEdit handoff validate bounded integration;
  they are not Mac model/Jev results or Windows comparisons. See [MACOS.md](MACOS.md).

Keep raw traces and unreviewed recordings in ignored local directories. The site
builder reads reviewed repository data and media; publication remains separate
from task execution.
