# Benchmarks: VLM System 2 + Jev System 1

[中文](BENCHMARKS.zh-CN.md)

This page records how the MiniWoB++ and OSWorld-MCP results in the README were produced. Each task was run once per agent, so treat differences of a few tasks as run-to-run variation, not as a ranking.

## Agents

| | System 2 (plans) | System 1 (acts) |
|---|---|---|
| **VLM only** (`vlm`) | Qwen3.7-Plus chooses every next action from the screenshot and structured state | deterministic executor and effect checks |
| **VLM + Jev** (`s1s2_jev`) | Qwen3.7-Plus plans up to 8 (browser) or 4 (desktop) short semantic subgoals | `jev-1.14.0` and deterministic code: bind each subgoal to a live control (`choice`) or screen region (`box`), shortlist MCP tools and pick GUI vs. MCP per subgoal, confirm effects and completion on the screenshot (`noul`), pick a System-2 alternative after a failure; System 2 is called again only when System 1 cannot continue |

Both agents use the same VLM endpoint, a 512-token thinking budget and the same executors. The rule-based System 1 (`s1s2_rule`) uses the same runner with a lexical matcher in place of Jev; it is reported for MiniWoB++ as an ablation.

## MiniWoB++

- 125 MiniWoB++ tasks (upstream revision `33c3b4d`), one seed (1100), structured browser observation, at most 20 steps and 180 s per episode.
- The numbers come from the S1/S2 browser path before the OSWorld-specific changes in the same commit series.

| | Success | Agent time / episode | VLM calls / episode | VLM tokens / episode |
|---|---:|---:|---:|---:|
| VLM only | 62 / 125 | 42.5 s | 3.7 | 7.1k |
| VLM + rule-based System 1 | 60 / 125 | 23.7 s | 2.2 | 2.9k |
| **VLM + Jev** | **62 / 125** | **23.9 s** | **2.0** | **2.5k** |

On the 53 tasks that all three agents solved, the mean agent time is 27.4 s (VLM only) vs. 13.8 s (VLM + Jev), and VLM tokens per episode fall from 4.1k to 1.3k.

## OSWorld-MCP

- Task set: the 361 OSWorld-MCP tasks without the 76 Chrome tasks (our harness could not keep Chrome's DevTools connection stable): 285 tasks.
- Environment: one fresh local QEMU/KVM Ubuntu guest per episode, the OSWorld-MCP tool server (131 tools) inside the guest, the original OSWorld evaluator run once after the agent stops.
- Budgets (identical for both agents): at most 50 steps, 1800 s of agent time and 5 consecutive failed actions.
- Infeasible tasks: either agent may finish with an explicit "infeasible" declaration, which is recorded as OSWorld's own `FAIL` action before evaluation. Both agents get the same instruction.
- Agent time excludes VM boot, task setup and evaluation. Tokens are the prompt and completion tokens reported by the VLM endpoint; Jev calls are counted separately.
- Environment failures (VM boot timeouts, task asset downloads, LibreOffice tool-bridge disconnects) were rerun under the same rules up to three times. Six episodes still failed (4 VLM only, 2 VLM + Jev); their tasks are excluded, leaving 279 tasks valid for both agents.
- The runs were collected in three consecutive passes with identical agent code (only the VM image and per-episode overlays moved from network storage to a local disk between passes), plus the reruns.

| | Success | Agent time / episode | VLM calls / episode | VLM tokens / episode | Jev calls / episode |
|---|---:|---:|---:|---:|---:|
| VLM only | 103 / 279 | 223 s | 14.9 | 338k | — |
| **VLM + Jev** | **104 / 279** | **199 s** | **11.6** | **49k** | 17.2 |

- Solved only by VLM only: 28 tasks. Solved only by VLM + Jev: 29 tasks.
- Infeasible tasks (26): 15 vs. 17 correctly declared.
- On the 75 tasks both agents solved: 83 s vs. 88 s, 5.9 VLM calls each, 158k vs. 25k VLM tokens per episode.

| Application | Tasks | VLM only | VLM + Jev |
|---|---:|---:|---:|
| GIMP | 26 | 13 | 9 |
| LibreOffice Calc | 43 | 18 | 11 |
| LibreOffice Impress | 47 | 13 | 13 |
| LibreOffice Writer | 23 | 6 | 10 |
| Multi-app | 63 | 11 | 16 |
| OS | 23 | 11 | 13 |
| Thunderbird | 14 | 6 | 5 |
| VLC | 17 | 10 | 11 |
| VS Code | 23 | 15 | 16 |

What Jev did on OSWorld-MCP, summed over the 279 tasks: 2,430 MCP tool calls routed, 1,156 on-screen groundings (456 of them replacing a System-2 coordinate guess), 235 confirmations that an action had taken effect when the pixel check saw no change, and 28 completion claims sent back to System 2.

## Limits

- One run per task and agent; no held-out split. The runner's thresholds were tuned on these benchmarks.
- Chrome tasks are excluded from OSWorld-MCP; MiniWoB++ uses a single seed.
- Jev is weaker than the VLM-only agent on GIMP and Calc, where most targets are canvas regions or spreadsheet cells without useful accessibility labels.

## Reproduce

Set `TYPESAFE_API_KEY`, `CUA_JEV_MODEL_BASE_URL`, `CUA_JEV_MODEL_API_KEY` and `CUA_JEV_VLM_MODEL` (see [`.env.example`](../.env.example)), or point `--service-profile` at a private JSON profile.

```bash
# MiniWoB++
python -m cua_jev.benchmark_miniwob --html-root <miniwob-html> --upstream-revision <rev> \
  --service-profile <profile.json> --output runs/miniwob --structured-browser \
  --modes vlm s1s2_jev --seeds 1100 --max-steps 20 --timeout 180 \
  --enable-thinking on --thinking-budget 512 --jev-model jev-1.14.0 --tasks <task ...>

# OSWorld-MCP (Linux with KVM; all paths owned by the current user)
python scripts/run_osworld_mcp_pilot.py --expected-user "$USER" \
  --osworld-root <OSWorld> --vm-path <Ubuntu.qcow2> --worker-python <venv/bin/python> \
  --mcp-bundle <bundle.tar.gz> --mcp-installer <install-guest-mcp.py> \
  --service-profile <profile.json> --jev-proxy-config <proxy.json> --output runs-osworld/<name> \
  --tasks <domain/task-id ...> --modes vlm s1s2_jev \
  --max-steps 50 --timeout 1800 --max-failures 5 \
  --enable-thinking on --thinking-budget 512 --jev-model jev-1.14.0 \
  --lenient-startup-checks [--qemu-scratch <owner-only local dir>]
```

`--lenient-startup-checks` only relaxes this harness's own readiness checks (identically for both agents); `--qemu-scratch` keeps per-episode VM overlays on a local disk.
