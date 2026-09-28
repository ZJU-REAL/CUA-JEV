# CUA-JEV

[![CUA-JEV：一个意图，多条执行路线，独立验证动作结果](assets/hero.svg)](https://zjureal.com/CUA-JEV/)

**将模型规划、类型化动作选择与独立验证结合起来的计算机使用参考框架。**

[项目网页](https://zjureal.com/CUA-JEV/) · [真实录像](https://zjureal.com/CUA-JEV/#windows-cases) · [English](README.md) · [macOS 指南](docs/MACOS.md) · [开放任务指南](docs/OPEN_TASKS.md)

通用模型根据实时应用状态提出下一步意图；框架把意图编译为合法的 **「意图 × 执行通道」候选**；**Jev 选择一个具体动作及其通道**。运行时检查、执行并读取真实结果，再进入下一轮决策。

同一意图可以有 GUI、DOM、可访问性、CLI、MCP 或文件 API 等路线，具体取决于当前后端。Jev 在类型化候选中作选择，不直接理解截图、生成任意脚本或代替任务验收。本项目使用现有 Jev API，不训练新的模型。

> **当前证据边界：**公开展示的是四条成功的、受约束的 Windows 模型＋Jev 运行。macOS 已有原生后端实现，并通过无密钥的真实浏览器＋CLI 冒烟测试；原生动作和 Mac 模型/Jev 任务仍待实测。这些成果尚不能证明任意任务可靠性，也不能证明普遍的速度或成本优势。

## 运行机制

![六阶段闭环：观察实时状态、模型提出意图、框架构建通道、Jev 选择、执行、独立验证](assets/architecture.svg)

| 层次 | 职责 |
|---|---|
| 观察 | 读取实时 DOM、可访问性和受限工具状态，提供控件引用与新鲜度指纹。 |
| 模型规划 | 对当前观察到的引用提出受约束操作；可选视觉模型单独提供定位信息。 |
| 动作表面 | 校验提案，仅编译当前后端能够执行的路线。 |
| Jev | 选择包含动作和执行通道的具体候选。 |
| 运行时 | 检查身份、新鲜度、范围和风险，交给已注册执行器。 |
| 独立验证 | 回读动作效果和调用者定义的终态条件，在本地记录证据、耗时与用量。 |

模型声称成功、执行器返回成功，都不足以直接判定任务完成。独立验证的强度也取决于验收条件：检查引文、文件完整性，不等于验证产物中每句话的质量。

## Windows 真实案例

四条**真实的单窗口录像**展示了：沿实时文档链接探索，通过同源 MCP 工具读取来源，生成带引文的指南，并在编辑器打开确切文件。调用者提供目标和受限验收条件，模型自行选择页面访问顺序，没有预写章节 URL 或点击序列。

| 案例 | 应用 | 动作 / Jev / 规划模型调用 | 实际执行通道 |
|---|---|---:|---|
| [Python 自动化指南](https://zjureal.com/CUA-JEV/#case-python-guide) | Edge → Terminal → VS Code | 19 / 19 / 19 | DOM 8 · MCP 8 · CLI 2 · API 1 |
| [JavaScript 学习指南](https://zjureal.com/CUA-JEV/#case-javascript-guide) | Edge → Notepad | 18 / 18 / 18 | DOM 8 · MCP 8 · CLI 1 · API 1 |
| [Git 工作流指南](https://zjureal.com/CUA-JEV/#case-git-guide) | Edge → Terminal → VS Code | 21 / 21 / 21 | DOM 10 · MCP 8 · CLI 2 · API 1 |
| [PowerShell 学习指南](https://zjureal.com/CUA-JEV/#case-powershell-guide) | Edge → Terminal → Notepad | 19 / 19 / 19 | DOM 8 · MCP 8 · CLI 2 · API 1 |

<details>
<summary>展开四条录像预览</summary>

| Python 自动化 | JavaScript 学习 |
|:---:|:---:|
| [![Python 录像](website/media/windows-python-guide.jpg)](https://zjureal.com/CUA-JEV/#case-python-guide) | [![JavaScript 录像](website/media/windows-javascript-guide.jpg)](https://zjureal.com/CUA-JEV/#case-javascript-guide) |
| Git 工作流 | PowerShell 学习 |
| [![Git 录像](website/media/windows-git-guide.jpg)](https://zjureal.com/CUA-JEV/#case-git-guide) | [![PowerShell 录像](website/media/windows-powershell-guide.jpg)](https://zjureal.com/CUA-JEV/#case-powershell-guide) |

</details>

这些是单次成功案例，不是重复试验。**四条均采用文本规划，VLM 调用为零。**浏览器曾提供 GUI 路线，但 Jev 实际选择了 DOM。每一步都调用了规划模型，因此这些录像尚未展示低频规划的收益。网页提供完整动作步骤，并区分播放时长和测得的墙钟耗时。

早期 Edge、Excel、VS Code、Explorer 工作流采用任务专用适配器。Hybrid / GUI Only 对照、Codex pilot 与早期开放任务录像保留在[早期工作页](https://zjureal.com/CUA-JEV/early-work.html)；证据及局限见[实验背景](docs/EXPERIMENTS.md)。

## 平台状态

| 路径 | 已实现 | 当前验证程度 |
|---|---|---|
| 共享运行时 | 类型化候选、guard、执行器、trace、Rule 与 Jev 策略 | 可移植单元/集成测试；Windows、Linux、macOS CI 矩阵 |
| Windows | UIA/GUI 桌面、浏览器、注册工具、编辑器交接 | 四条模型＋Jev 录像及早期任务专用工作流 |
| macOS 浏览器＋CLI | Chromium 与受限 Python CLI | 确定性规划器＋Rule 策略的真实两动作测试 |
| macOS 原生 | AX/CGEvent 候选、编辑值私有回读、确切文档探测、单窗口截图/录制 | helper 编译与离线测试；真实 AX/GUI、录像和模型＋Jev 任务待验证 |
| Linux 桌面 | 尚无原生桌面后端 | 仅可移植测试，无公开 Linux 桌面结果 |

Mac 原生 helper 需要 **macOS 14+ 和 Swift Command Line Tools**；原生 MP4 录制需要 **macOS 15+**。系统权限与本地 API 配置是后续独立验证阶段，详见 [MACOS.md](docs/MACOS.md)。

## 不需要密钥或桌面权限的上手流程

需要 **Python 3.11+**。以下命令运行本地沙箱和确定性 Rule 策略，不调用 Jev 或规划模型。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mcp]'
cua-jev doctor
cua-jev demo --policy rule
pytest -q
```

Windows 上使用 `py -m venv .venv` 创建环境，通过 `.\.venv\Scripts\Activate.ps1` 激活，后续 Python 命令相同。

macOS 上可运行真实本地 Chromium＋CLI 测试，不依赖外部网站：

```sh
python -m pip install -e '.[browser]'
python -m playwright install chromium
python scripts/macos_smoke.py --browser-only
```

从仓库中经过审核的数据构建项目网页：

```sh
python scripts/build_pages.py --build
python -m http.server 8768 --bind 127.0.0.1 --directory dist-pages
# 浏览器打开 http://127.0.0.1:8768
```

静态预览不会执行任务或访问模型 API。可选的 `cua-jev-ui` 控制台（安装 `.[ui]`）还包含本地运行管理接口，属于独立的开发工具。

### 准备运行模型＋Jev 时

原生测试窗口参见 [macOS 指南](docs/MACOS.md)；浏览器、Windows 桌面、注册 MCP 和产物流程参见[开放任务指南](docs/OPEN_TASKS.md)。参考 [.env.example](.env.example)，把密钥保存在 Git 忽略的本地 `.env`：

- `TYPESAFE_API_KEY`：Jev 决策。
- `CUA_JEV_MODEL_API_KEY`：文本/视觉模型网关。
- `CUA_JEV_PLANNER_API_KEY`：仅用于另一种提供者无关的规划端点。

补充配置时保留已有 `.env`，不要提交密钥或私有任务 trace。

运行早期 Windows 工作流可安装 `'.[all,ui,recording]'`，再执行 `cua-jev suite --task edge --policy jev --profile adaptive`；可按需把 `edge` 换成 `excel`、`vscode` 或 `explorer`。`adaptive` 对应 Hybrid，`visible` 对应 GUI Only。**macOS/Linux 不要安装面向 Windows 的 `all` 或 `windows` extras。**

## 扩展与评测

主要扩展点是 [`ActionSurface`](src/cua_jev/surface_providers.py)：observe → validate → compile → execute → verify。浏览器、可访问性和注册工具表面共用一套规划循环。

1. 定义可观察状态与调用者掌控的成功条件。
2. 构造真实的 [`ActionCandidate`](src/cua_jev/models.py) 候选，明确参数范围、风险和验证器。
3. 注册执行器；使用固定 argv 模板与可信 MCP 配置，避免任意生成 shell。
4. 独立核验效果与终态；覆盖过期状态、拒绝、失败及清理路径。
5. 对未见目标做重复实验，报告失败、通道选择、延迟、模型请求、token 与成本。

代码入口：[runtime](src/cua_jev/runtime.py)、[任务生命周期](src/cua_jev/episode.py)、[guard](src/cua_jev/guard.py)、[Mac 表面](src/cua_jev/macos_surface.py)、[trace 分析](src/cua_jev/trace_analysis.py) 和[测试](tests/)。

```sh
ruff check .
pytest -q -ra
python scripts/build_pages.py --build
```

接下来的优先项是 Mac 原生路线实测、独立验收的模型/Jev 任务、跨通道失败恢复，以及留出任务的重复评测。更广泛的窗口组合、Mac VLM 定位与减少规划调用仍待实现和验证。

## 数据边界与许可

候选动作需通过新鲜度、身份、路径范围和副作用检查；可选截图上传需要明确启用。本地 trace 可能包含页面文字、用户目标、工具结果或产物内容，请保留 `runs/`、`artifacts/` 和 `.env` 的私有属性。公开网页使用经审核的[案例目录](website/windows_demos.json)和媒体，不直接读取本地原始运行数据。早期实验中的模型费用是估算，不是账单。

代码采用 [Apache-2.0](LICENSE)。应用图标来源见 [ATTRIBUTION.md](src/cua_jev/ui/static/icons/ATTRIBUTION.md)。感谢 [TypeSafe AI](https://typesafe.ai/) 提供 Jev，以及 [RoboJEV](https://github.com/lykycy123/RoboJEV) 的具身智能实践带来的启发。CUA-JEV 是 ZJU-REAL 的独立研究与工程项目。
