# 基准评测：VLM 作为 System 2，Jev 作为 System 1

[English](BENCHMARKS.md)

本页说明 README 中 MiniWoB++ 和 OSWorld-MCP 结果是如何得到的。每个智能体在每道任务上只跑了一次，几道题以内的差距应看作单次运行之间的正常波动，不代表排名。

## 智能体

| | System 2（规划） | System 1（执行） |
|---|---|---|
| **仅 VLM**（`vlm`） | Qwen3.7-Plus 根据截图和结构化状态决定每一步动作 | 确定性执行器和效果检查 |
| **VLM + Jev**（`s1s2_jev`） | Qwen3.7-Plus 规划若干简短的语义子目标：浏览器上最多 8 个，桌面上最多 4 个 | 由 `jev-1.14.0` 和确定性代码完成，见下方列表 |

VLM + Jev 组中 System 1 负责：

- 把每个子目标绑定到当前界面的控件（`choice`）或屏幕区域（`box`）；
- 从 MCP 工具中筛选候选，并为每个子目标选择走 GUI 还是 MCP；
- 看截图确认执行效果和任务是否完成（`noul`）；
- 某一步失败后，从 System 2 给出的备选做法里挑一个接着做；
- 只有 System 1 无法继续时，才再次调用 System 2。

两组使用同一个 VLM 接口、512 token 的思考预算和相同的执行器。基于规则的 System 1（`s1s2_rule`）使用同一个运行器，只是用词法匹配代替 Jev；它在 MiniWoB++ 上作为消融对照列出。

## MiniWoB++

- 125 道 MiniWoB++ 任务（上游版本 `33c3b4d`），单个种子（1100），使用结构化浏览器观察，每回合最多 20 步、180 秒。
- 这组数字来自 S1/S2 浏览器路径在加入 OSWorld 相关改动之前的版本，两者属于同一系列提交。

| | 成功数 | 每回合 agent 用时 | 每回合 VLM 调用 | 每回合 VLM token |
|---|---:|---:|---:|---:|
| 仅 VLM | 62 / 125 | 42.5 秒 | 3.7 | 7.1k |
| VLM + 规则版 System 1 | 60 / 125 | 23.7 秒 | 2.2 | 2.9k |
| **VLM + Jev** | **62 / 125** | **23.9 秒** | **2.0** | **2.5k** |

在三组都做对的 53 道任务上：
- 平均 agent 用时：仅 VLM 为 27.4 秒，VLM + Jev 为 13.8 秒；
- 每回合 VLM token：从 4.1k 降到 1.3k。

## OSWorld-MCP

**实验设置**

- **任务集**：OSWorld-MCP 共 361 道任务，去掉 76 道 Chrome 任务后为 285 道。去掉的原因是我们的测评环境无法稳定保持 Chrome 的 DevTools 连接。
- **环境**：每回合启动一个全新的本地 QEMU/KVM Ubuntu 虚拟机，虚拟机内运行 OSWorld-MCP 工具服务（131 个工具）。智能体停止后，运行一次 OSWorld 原始评测器。
- **预算**（两组相同）：每回合最多 50 步、1800 秒 agent 时间，最多连续 5 次动作失败。
- **不可完成的任务**：两组智能体收到相同的说明，都可以显式声明"不可完成"来结束；评测前该声明会被记录为 OSWorld 自带的 `FAIL` 动作。
- **计量口径**：agent 用时不含虚拟机启动、任务准备和评测时间。token 指 VLM 接口返回的输入加输出 token 数，Jev 调用单独计数。
- **环境失败的处理**：因环境问题失败的回合，按同一规则最多补跑三次。这些问题包括虚拟机开机超时、任务素材下载失败、LibreOffice 工具连接断开。补跑后仍有 6 个回合失败（仅 VLM 4 个、VLM + Jev 2 个），这些回合对应的题从配对中排除，最终两组都有效的任务为 279 道。
- **数据来源**：结果由连续三轮运行加补跑合并而成。三轮的智能体代码完全相同，轮次之间只是把虚拟机镜像和每回合的临时磁盘从网络存储挪到了本地盘。

**结果**

| | 成功数 | 每回合 agent 用时 | 每回合 VLM 调用 | 每回合 VLM token | 每回合 Jev 调用 |
|---|---:|---:|---:|---:|---:|
| 仅 VLM | 103 / 279 | 223 秒 | 14.9 | 338k | — |
| **VLM + Jev** | **104 / 279** | **199 秒** | **11.6** | **49k** | 17.2 |

- 只有仅 VLM 做对的有 28 道，只有 VLM + Jev 做对的有 29 道。
- 不可完成的任务共 26 道，两组正确声明的数量分别为 15 道和 17 道。
- 在两组都做对的 75 道任务上：
  - agent 用时：83 秒对 88 秒；
  - VLM 调用：两组都是 5.9 次；
  - 每回合 VLM token：158k 对 25k。

| 应用 | 任务数 | 仅 VLM | VLM + Jev |
|---|---:|---:|---:|
| GIMP | 26 | 13 | 9 |
| LibreOffice Calc | 43 | 18 | 11 |
| LibreOffice Impress | 47 | 13 | 13 |
| LibreOffice Writer | 23 | 6 | 10 |
| 多应用 | 63 | 11 | 16 |
| OS | 23 | 11 | 13 |
| Thunderbird | 14 | 6 | 5 |
| VLC | 17 | 10 | 11 |
| VS Code | 23 | 15 | 16 |

Jev 在 OSWorld-MCP 上的工作量（279 道任务合计）：
- 选择并发起 MCP 工具调用 2430 次；
- 在屏幕上定位目标 1156 次，其中 456 次替换了 System 2 估计的坐标；
- 在像素比对没有发现变化时，确认操作实际已经生效 235 次；
- 判断任务尚未完成、把完成声明退回 System 2 共 28 次。

### 工具使用

| | 仅 VLM | VLM + Jev |
|---|---:|---:|
| 每次决策时 VLM 看到的 MCP 工具数 | 131 | 约 10 个（由 Jev 筛选） |
| 每回合 MCP 调用 / GUI 操作次数 | 7.6 / 3.3 | 8.7 / 11.0 |
| 用到 MCP 的回合占比 | 78% | 80% |
| 应用专用工具的调用占比（不含通用的 `os` 工具） | 49% | 83% |
| `os.open_shell` 的调用占比 | 43%（925 / 2132） | 10%（246 / 2430） |
| 只用 MCP 的回合中的成功率 | 44%（59/135） | 40%（42/106） |
| 同时用到 MCP 和 GUI 的回合中的成功率 | 17%（14/83） | 30%（35/117） |

按工具类别统计的调用次数：
- 仅 VLM：`os` 1096 次，`libreoffice_calc` 519 次，`libreoffice_impress` 323 次，`libreoffice_writer` 99 次，`code` 46 次，`vlc` 14 次；
- VLM + Jev：`libreoffice_impress` 799 次，`libreoffice_calc` 718 次，`os` 403 次，`libreoffice_writer` 234 次，`code` 114 次，`vlc` 78 次。

两组的工具调用失败率分别为 5% 和 8%。

解读时注意两点：
- 回合类型是按各组自己的回合分别统计的，所以最后两行中两组对应的不是同一批任务。
- 某个子目标走 MCP 还是 GUI，主要由 System 2 根据 Jev 筛出的工具决定；System 2 给 GUI 子目标附上工具时，Jev 很少改选另一个通道。

## 局限

- 每个智能体在每道任务上只跑了一次，没有留出集；运行器的各项阈值是在这些基准上调出来的。
- OSWorld-MCP 排除了 Chrome 任务；MiniWoB++ 只用了一个种子。
- 在 GIMP 和 Calc 上 Jev 不如仅 VLM。这两类应用中，大多数操作目标是画布区域或表格单元格，缺少有用的可访问性标签。

## 复现

设置 `TYPESAFE_API_KEY`、`CUA_JEV_MODEL_BASE_URL`、`CUA_JEV_MODEL_API_KEY` 和 `CUA_JEV_VLM_MODEL`（见 [`.env.example`](../.env.example)），或者用 `--service-profile` 指向一个私有的 JSON 配置文件。

```bash
# MiniWoB++
python -m cua_jev.benchmark_miniwob --html-root <miniwob-html> --upstream-revision <rev> \
  --service-profile <profile.json> --output runs/miniwob --structured-browser \
  --modes vlm s1s2_jev --seeds 1100 --max-steps 20 --timeout 180 \
  --enable-thinking on --thinking-budget 512 --jev-model jev-1.14.0 --tasks <task ...>

# OSWorld-MCP（需要带 KVM 的 Linux；所有路径须归当前用户所有）
python scripts/run_osworld_mcp_pilot.py --expected-user "$USER" \
  --osworld-root <OSWorld> --vm-path <Ubuntu.qcow2> --worker-python <venv/bin/python> \
  --mcp-bundle <bundle.tar.gz> --mcp-installer <install-guest-mcp.py> \
  --service-profile <profile.json> --jev-proxy-config <proxy.json> --output runs-osworld/<name> \
  --tasks <domain/task-id ...> --modes vlm s1s2_jev \
  --max-steps 50 --timeout 1800 --max-failures 5 \
  --enable-thinking on --thinking-budget 512 --jev-model jev-1.14.0 \
  --lenient-startup-checks [--qemu-scratch <仅本人可访问的本地目录>]
```

两个可选参数的作用：
- `--lenient-startup-checks`：只放宽本测评环境自己加的就绪检查，对两组完全相同；
- `--qemu-scratch`：把每回合虚拟机的临时磁盘放到本地盘。
