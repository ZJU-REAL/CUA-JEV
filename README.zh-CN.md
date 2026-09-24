# <img src="src/cua_jev/ui/static/logo-mark.svg" width="44" alt="ZJU-REAL Lab logo"> CUA-JEV: Jev for Computer Use

[项目网页](https://zjureal.com/CUA-JEV/) · [English README](README.md)

CUA-JEV 是一个面向 Computer-Use Agent（CUA）的开源参考框架。它把浏览器、桌面 UI、办公软件、终端和文件系统的结构化状态，转换为一组**当前合法、可执行、可验证**的候选动作；[Jev](https://docs.typesafe.ai/introduction) 从中选择**具体动作及其执行路线**，框架再负责安全检查、实际执行、结果验证和下一轮观察。首版不训练专用路由模型，也不依赖 VLM，提供四个已在 Windows 验证的完整任务范例及 Hybrid / GUI Only 对照实验。

Jev 的快速、类型化决策能力，适合探索需要高频、低延迟动作选择的下游方向，例如 CUA、具身智能，以及潜在的智能驾驶场景。[RoboJEV](https://github.com/lykycy123/RoboJEV) 已在 MuJoCo 仿真中探索 Jev 控制的机器人操作；CUA-JEV 则聚焦计算机使用中的“下一步做什么、通过哪种通道执行”。

> **能力边界：**这不是“给任意指令，就能操作任意软件”的通用 Agent。目前四个案例都有任务专属适配器、候选动作与终态验证器。Jev 负责受约束的选择，不负责自由生成操作脚本或直接理解图像；实验性 VLM 路径会先把截图转换为受约束的文字场景与可点击目标。

**平台现状：**类型化决策循环、Guard 和 trace 的设计可移植，CI 已配置 Windows、Linux、macOS 单元测试。实验性浏览器路径可改用 Playwright Chromium，但 macOS 上的真实运行仍待验证；已发布的桌面案例和 `open-desktop` 仍依赖 Windows UI Automation、Excel COM、Explorer 等接口。macOS/Linux 桌面适配器**尚未实现或验证**。

![CUA-JEV：任务适配器、Jev 决策、执行器与验证器组成的闭环](assets/architecture.svg)

## 为什么是 Jev × CUA？

TypeSafe AI 将 Jev 定位为供软件直接调用的 *System One* 决策模型：输入状态和类型化问题，返回可直接用于分支的结构化答案。其 [Choice](https://docs.typesafe.ai/introduction) 原语尤其适合“从已知选项中选一个”，而不是生成一段文本再解析。TypeSafe 将模型训练方法称为 RLCD；**本项目调用现成 Jev API，并没有训练 Jev 或一个新的 CUA 模型**。[官方介绍](https://typesafe.ai/) · [开发文档](https://docs.typesafe.ai/introduction)

CUA 的动作空间天然是混合的。同一个子目标有时适合可见的鼠标键盘操作，有时更适合 DOM、COM、CLI、MCP 或文件 API。每一步都调用通用大模型重新规划、生成命令和解析输出，可能带来不必要的时间与推理费用；但固定规则也难以应对多种真实路线。我们的切入点是把**开放式规划问题收窄成受约束的动作选择问题**：适配器提供合法选项，Jev 在线选择，确定性代码负责安全与事实核验。

这个契合点也有前提：当前任务必须能提供有用的结构化观察，并且开发者需要实现可靠的候选动作和验证器。Jev 不能替代缺失的感知、任务分解或软件集成工程。

## 核心机制

1. **Observe / Generate**：任务适配器读取当前应用状态，只枚举此刻合法的 `intent × route` 候选。一个“修复失败测试”子目标可同时提供 VS Code GUI、MCP 写文件、受限 CLI、文件 API 等真实可行路线。
2. **Select with Jev**：调用 Jev 的 `choice` 接口，从候选 ID 中选出一个动作。选中的只是预定义的类型化动作及参数；Jev 不直接执行任意 shell 文本。
3. **Guard / Execute**：`ActionGuard` 检查观察版本、候选身份、路径边界、写入权限与确认要求，再分发到注册的执行器。
4. **Verify / Repeat**：执行回执不等于成功。独立验证器检查应用的新状态，JSONL trace 记录选择、执行、耗时和验证证据，然后重新观察，直到达到任务终态。

可复用的是 [`ActionCandidate`](src/cua_jev/models.py)、[`AgentRuntime`](src/cua_jev/runtime.py)、[`ActionGuard`](src/cua_jev/guard.py)、[`ExecutorRegistry`](src/cua_jev/registry.py)、观察器与能力包接口，以及评测/trace 契约；四个任务是这些接口的参考实现。

### 首版支持什么

| 任务范例 | 目标 | 首版可竞争通道 |
|---|---|---|
| Edge | 公开演示商店登录、排序、购物车、结账和收据核验 | PyAutoGUI、DOM |
| Excel | 计算指标、标记审核状态、生成图表并通过独立 COM 会话验证 | PyAutoGUI、COM |
| VS Code | 定位并逐一修复多个缺陷、反复运行测试 | PyAutoGUI、MCP、CLI、文件 API |
| Explorer | 从混合收件箱筛选报告并制作可验证的发布产物 | PyAutoGUI、MCP、CLI、文件 API |

GUI Only 中，状态读取与定位仍可使用结构化接口，但**修改操作通过 PyAutoGUI 完成**；Hybrid 则让 Jev 在真实可用的 GUI 与结构化执行路线中选择。项目同时提供无密钥的 Rule 策略，用于测试和策略消融。Windows UIA、终端文本、文件系统等也是适配器可用的观察通道。

## 实验与展示

[项目网页](https://zjureal.com/CUA-JEV/)提供四个案例视频、两组分开的比较，以及可展开的执行记录：

- **Jev Hybrid vs Jev GUI Only**：保持 Jev 策略与终态验证相同，比较动作空间对完成时间的影响。
- **Jev Hybrid vs Codex Computer Use Hybrid**：两者都允许混合工具，比较实测墙钟时间和基于公开费率的模型美元成本估算。

2026-09-23 的首批 v2 记录如下（秒；Jev 为已有成功记录的中位数，Codex 为每任务一次 pilot）：

| 案例 | Jev Hybrid | Jev GUI Only | Codex Hybrid |
|---|---:|---:|---:|
| Edge | 79.5 | 87.4 | 77.4 |
| Excel | 84.2 | 83.2 | 39.2 |
| VS Code | 14.0 | 90.8 | 57.9 |
| Explorer | 13.2 | 211.1 | 50.9 |

VS Code 和 Explorer 展示了混合通道避开大量 GUI 操作的潜力；**Edge 与 Excel 的这批 Hybrid 运行实际上仍选择了 GUI，Excel 甚至略慢**。因此不能把这四行描述为“Jev 在所有任务上更快”。Codex 是通用工具代理，Jev 则使用预先构建的任务能力包；样本数也不足以形成通用速度排名。网站展示的美元数值是按 [TypeSafe 公开价格](https://docs.typesafe.ai/models)和 [OpenAI 公开费率](https://help.openai.com/en/articles/20001415-chatgpt-rate-card-enterprise-token-based-pricing)折算的**模型成本估计，不是实际账单**。最小必要的 token 计数与限制统一收录在 [`website/snapshot.json`](website/snapshot.json)，不再单独发布过程文件。

网页是只读的公开实验快照，不连接 Jev API，也不运行用户电脑上的任务。案例视频是演示录制，表中耗时取自运行记录，而不是视频长度。Jev 展开的是代表性运行的逐步 trace；Codex 展开的是当时记录、按阶段合并的工具调用，不冒充一一对应的原子 GUI 动作。发布快照位于 [`website/snapshot.json`](website/snapshot.json)。

## 开放任务探索

[实验性的开放任务路径](docs/OPEN_TASKS.md)将模型规划与 Jev 选路分开：文本模型从实时 DOM / UIA 等观察中理解目标并提出下一步意图，框架将意图展开为合法的类型化执行路线，Jev 再选一个，随后执行、验证并重观察。

开放任务宿主支持浏览器访问验收、只创建新文件的引文产物，以及“当前页只经 MCP 读取一次／每个来源都须读取”约束。此前遇到的瞬态空 DOM 和模型网关超时已分别处理；一轮真实八章任务完成 **18 个外部动作／251.4 秒**，可见 Edge 录制重跑完成 **18 个动作／240.7 秒**（DOM 8、MCP 8、CLI 1、文件 API 1）。但这轮给出了八个章节线索，仍是受限长任务；基于模拟浏览器和脚本规划器的 20 动作回归也不是真实陌生任务成绩。

新的实验模式只给“至少 N 个不同来源”和目标主题，**不提供具体页面 URL 或点击顺序**：模型从实时链接中找页面，框架逐页以 MCP 验证、要求引用，并用来源标题条件检查主题覆盖。相关链接现在不会因为排在 DOM 前 60 项之外就消失。产物可经显式授权在 VS Code 打开，并核实是本次新窗口。一次真实 Edge → VS Code 四主题任务完成 **12 个动作／173.8 秒**，涵盖 DOM、MCP、CLI、文件 API；录制和产物留在本地。此前一次仅按来源数量验收的运行把“错误处理”误引到“数据结构”，另一次在寻找目标来源时耗尽步数，所以这些仍是探索性案例，不能宣称任意任务泛化、内容自动语义核验或已建立可靠成功率。macOS 尚未实测。

![开放任务架构：实时状态、模型提出意图、框架构建候选、Jev 选路、执行验证循环](src/cua_jev/ui/static/open-task-loop.svg)

[跨软件案例视频](https://zjureal.com/CUA-JEV/#open-task)从 Python 官方教程首页出发，根据五个章节目标自行找到五张页面、记录可回查引文、生成指南并在 VS Code 中打开，共完成 12 次 Jev 决策；没有编码章节 URL 或点击顺序。其中 5 次通过 DOM 操作 Edge，5 次在框架内部记录引文，1 次通过文件 API 写入，1 次通过 CLI 打开 VS Code；内部记录不等同于外部计算机操作。[独立验证器](scripts/verify_research_demo.py)重新访问五个来源，核对标题、引文、链接与执行记录。该路径仍是受限的“浏览器→编辑器”任务族，**不是任意任务能力**；这段公开录制只使用文本模型和 DOM，未使用 VLM，重复成功率与成本效果也尚未评测。

公开录制的单次运行实测 155.9 秒，视频为观看方便统一加速 2.5 倍；另一次成功运行耗时 307 秒，其中 287 秒等待规划模型。也出现过网关超时和模型输出不合规的失败尝试，因此目前不能以该案例宣称开放任务模式在端到端速度或可靠性上已经占优。

### 浏览器 DOM + VLM + Jev 联调（实验性）

`open-browser` 现在可在用户明确授权后，把当前网页视口截图交给 VLM，并将有界的场景文字与实时 DOM 控件一起交给文本规划模型。框架校验模型提出的 DOM/视觉目标；当同一控件的文字和位置都匹配时，可同时提供 DOM 点击和视觉定位的浏览器鼠标点击候选，由 Jev 选择具体动作。视觉点击前会检查 URL、DOM、视口尺寸和截图是否过期；页面变化只是动作效果证据，受限目标仍需用实时 URL、标题或文本确认。截图上传、视觉点击和可能有外部副作用的点击分别需要显式开关。

一次真实的公开 Python 文档单步导航用时 **9.8 秒**：VLM 生成场景信息，文本模型提出 DOM 与视觉两种候选，Jev 以 **0.99 / 0.01** 的概率选择 DOM 并完成页面验证。其中 VLM 约 4.7 秒、文本规划约 1.9 秒、Jev 约 0.66 秒。此前也出现 VLM 输出格式和规划格式导致的失败；这只是一次联调成功，**不是成功率或速度基准**。原 12 步公开视频仍然只使用文本与 DOM。

```powershell
python -m pip install -e ".[browser-vision]"
python -m playwright install chromium
cua-jev open-browser --goal "Open a chapter" --url "https://docs.python.org/3/tutorial/index.html" `
  --browser-channel chromium --model-base-url "https://YOUR_MODEL_GATEWAY/v1" `
  --model "YOUR_TEXT_MODEL" --vision-model "YOUR_VISION_MODEL" --vision-mode always `
  --allow-screenshot-upload --policy jev
```

只有在受控测试且允许视觉鼠标操作时，才追加 `--allow-visual-clicks --allow-external-actions`。校园模型网关默认直连，绕过环境代理。

### 浏览器 + 本地只读工具联调（实验性）

`open-browser` 现在也能显式接入受目录约束的只读能力包。文本模型同时看到网页控件 `eN`/`vN` 和本地工具 `tN`，Jev 从合法候选中选具体动作；命令模板、路径均由框架固定，模型不能生成任意 shell 命令或文件路径。目前提供目录列表、少量顶层 `.md`/`.txt` 文件读取、注册的 `python --version`，以及仓库目录中的 Git 状态。`--require-tool` 与 `--require-url-contains` 可提供独立的有界验收条件。

一次未录制的真实联调同时要求读取本机 Python 版本并打开官方教程的另一章节：Jev 首步以 **0.99** 概率选 CLI，第二步选 DOM 链接；工具和 URL 验收都通过，总耗时 **7.0 秒**。这是单次打通案例，**不是成功率或速度基准**；该次没有启用 VLM。启用本地目录前，请确认其中的文件名与文字可发送给规划模型，私有运行 trace 也可能含工具结果。这仍是**单浏览器源站 + 受限本地工具**，不是任意跨软件任务支持。

另一次两步联调启用了 VLM，任务也完成，但耗时 **51.4 秒**，其中视觉请求约 **45.1 秒**：一次视觉响应不合规并回退 DOM，另一次返回了有效场景。这说明回退机制可用，同时暴露视觉网关的延迟与稳定性问题；不能据此宣称 VLM 提升了速度。

### 浏览器 + 桌面窗口 + 本地工具联调（实验性）

`open-computer` 把同源浏览器、**可选的明确指定的 Windows UIA 窗口**和注册的本地/MCP 工具放入同一决策循环。模型只能引用当前的 `b:`（浏览器）、`d:`（桌面）、`t:`（本地工具）和 `m:`（MCP）引用；框架展开为实际可执行的 DOM/UIA/GUI/CLI/MCP/API 动作，Jev 选其中一个。终态由调用者给定的 URL、窗口状态和已执行能力条件验收，而不是听信模型自报完成。窗口动作需显式授权；常见中英文关闭/最小化/最大化标签会被过滤，但这并非通用安全分类器。

一次未录制的本机联调使用 Python 官方文档、本机 Python CLI 和 Windows 计算器，Jev 依次选了 **CLI → DOM → UIA** 三个动作。最终网页 URL 与计算器“显示为 7”均通过实时核对，耗时 **8.5 秒**。中间一步 Jev 对 DOM/UIA/GUI 给出 **0.90/0.09/0.01**，最后一步对 UIA/GUI 给出 **0.90/0.10**。此前一次尝试在桌面动作校验处安全停止；修正并测试了对实时可调用控件的 `invoke`→`click` 规范化后才跑通。

另一轮相同类型任务打开了**计算器窗口的 VLM 场景感知**。三次视觉响应均有效，任务耗时 **48.5 秒**，其中 VLM 请求 **28.2 秒**、文本规划 **14.2 秒**。模型最终仍提出结构化“七”控件，Jev 以 **0.89/0.11** 选择 UIA 而非 GUI，没有选择视觉坐标点击。这些只是集成联调，**不是配对速度或成功率基准**，也不说明视觉感知改善了任务。

组合任务现通过[能力提供者契约](src/cua_jev/surface_providers.py)接入浏览器、Windows UIA 和受限工具：`观察 → 校验引用 → 编译候选 → 执行 → 验证`。测试已用替代桌面提供者跑通同一决策循环，但**不代表已支持 macOS**。发送给文本规划模型的组合状态保留实时引用和语义标签，去掉执行时才需要的窗口句柄与坐标；完整状态仍留在运行时校验。另一次不同目标的真实 CLI → DOM → UIA 回归（Python 版本、另一文档章节、计算器“8”）完成三次 Jev 决策，终态全部通过，耗时 **15.4 秒**；文本规划三次调用共报告 **15,930 tokens**。这只是功能烟测，不是降本或提速结论。

可选的 [MCP 工具提供者](src/cua_jev/mcp_surface.py)也已进入同一候选空间，以 `m:` 引用与 DOM、UIA、CLI 等路线并列。由用户信任的本地配置固定 MCP stdio 程序、工具白名单及基础参数；模型只能选择已有引用，并填写配置声明的、有严格类型与长度限制的简单参数，不能生成启动命令或改写固定参数。必须同时提供 `--mcp-profile` 与 `--allow-mcp-actions` 才能执行，且默认按可能有外部副作用的动作处理。真实 MCP 文件读取已在 DOM＋UIA 组合的端到端测试中通过；另有一条陌生参数的真实协议测试。这**不代表任意第三方 MCP 服务安全可靠**。[配置格式与限制](docs/OPEN_TASKS.md#registered-mcp-calls)见详细说明。

另一个模拟的编辑器式跨软件任务覆盖 UIA 填写文本＋浏览器跳转。终态检查会重新读取编辑控件的实时值，但不把该观察值放进发给模型的桌面快照。**这不是在真实 Notepad 上的成功结果。**

`open-computer` 现在可以不指定 Windows 窗口，只运行浏览器＋注册的 CLI/MCP 工具；真实 MCP 协议与浏览器 DOM 的无桌面窗口集成测试已通过。用 Playwright Chromium 时，这条代码路径不依赖 Windows UIA，但**尚未在 macOS 上实际验证**。MCP 还支持模型填写配置声明的简单参数，已用未预设的字符串做真实协议测试；这离任意参数、任意任务仍有距离。

### 可选窗口截图 / VLM 路径（实验性）

`open-desktop` 现在可以截取**明确指定的单个 Windows 窗口**，将缩放后的 JPEG 发给单独指定的视觉模型。VLM 返回有长度限制的场景摘要、可见文字和归一化目标框；文本规划模型把这些与 UIA 控件一起理解并提出下一步选项，框架校验后展开为可执行候选，Jev 再选择**具体动作与通道**。Jev 只接收文字摘要和候选，不接收像素或逐行 OCR。点击前检查窗口位置与截图是否过期，点击后读取 UIA 或像素变化；**画面变化不等于任务成功**，终态仍需 UIA 中可观察的证据。默认 `fallback` 仅在 UIA 没有可操作控件时调用 VLM；`always` 同时使用两类观察。

上传截图和视觉点击都需要显式开关。请只在非敏感、可控的窗口中使用：截图可能包含私人内容，`--allow-button-actions` 也可能允许具有外部副作用的点击。该路径通过了模拟网关及窗口的离线集成测试；校园网关直连的**合成图像**探针已返回正确的场景、文字和 Save 按钮框，耗时约 7.9 秒、报告 351 tokens。之后的真实组合任务中 VLM 场景融合也已跑通，但最终使用 UIA 而非视觉坐标动作；**不能宣称任意软件/任务已泛化或视觉动作有收益**。上述 12 步公开视频依旧是文本 + DOM 案例。

```powershell
cua-jev open-desktop --goal "完成可控窗口内的任务" --window-title "^Your Test Window$" `
  --model-base-url "https://YOUR_MODEL_GATEWAY/v1" --model "TEXT_MODEL_ID" `
  --vision-model "VISION_MODEL_ID" --vision-mode fallback `
  --allow-screenshot-upload --allow-visual-clicks --allow-button-actions --policy jev
```

模型密钥从本机 `CUA_JEV_MODEL_API_KEY` 读取；HTTP 校园网关需要额外加 `--allow-insecure-model-http`，应优先使用 HTTPS 或可信隧道。具体范围与限制见[开放任务说明](docs/OPEN_TASKS.md)。

## 快速开始

需要 Windows 和 Python 3.11+。先体验不需要 Jev 密钥的 Rule 基线：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,all,ui]"
playwright install chromium
cua-jev doctor
cua-jev demo --policy rule
cua-jev suite --task vscode --policy rule --profile adaptive --open-vscode
```

调用 Jev 时，把密钥写入本机被 Git 忽略的 `.env`，**不要提交密钥**：

```powershell
Copy-Item .env.example .env
# 在 .env 中设置 TYPESAFE_API_KEY=...
cua-jev jev-smoke
cua-jev suite --task vscode --policy jev --profile adaptive --open-vscode
```

完整任务可用 `--task edge|excel|vscode|explorer|all`。`--profile adaptive` 是 Hybrid；`--profile visible` 为 GUI Only。Edge 可加 `--headed-edge`，Excel/Explorer 可加 `--visible-apps`。公开商店案例使用 [SauceDemo](https://www.saucedemo.com/) 测试账号与演示订单，不涉及真实付款。

本地只读展示页：

```powershell
cua-jev-ui
# http://127.0.0.1:8768
```

可重复的配对实验和录制：

```powershell
python -m pip install -e ".[all,ui,recording]"
python scripts/record_v2_demos.py --task all --profile both --policy jev
```

本地运行记录保存在被忽略的 `runs/`，原始视频保存在被忽略的 `artifacts/demos/`。公开网页只使用经过审核的 `website/media/` 副本。CI 可在 Linux 上执行单元测试，但四个真实桌面案例需要 Windows 环境。

## 扩展一个新任务

1. **定义任务状态与终态条件**：实现观察器，输出尽量简洁、稳定、可验证的结构化状态；参考 [`observers.py`](src/cua_jev/observers.py) 与 [`suites.py`](src/cua_jev/suites.py)。
2. **枚举真实候选动作**：为每个当前合法的 `意图 × 通道` 构建 [`ActionCandidate`](src/cua_jev/models.py)，提供稳定 ID、能力名、参数、风险级别与验证器。不要把不可执行的路线当作候选来“凑多样性”。
3. **注册能力与执行器**：参考 [`capabilities.py`](src/cua_jev/capabilities.py) 和 [`registry.py`](src/cua_jev/registry.py)，让 GUI/DOM/COM/CLI/MCP/API 等通道映射到真正的工具调用。CLI 只允许注册的 argv 模板，不使用任意 `shell=True` 命令。
4. **实现独立验证与复位**：验证实际状态变化，定义任务成功谓词和可重复的 reset；为失败、重复动作、拒绝执行等情况补测试。参考 [`verify.py`](src/cua_jev/verify.py) 与 [`tests/`](tests/)。
5. **评测再发布**：同时跑 Hybrid 与 GUI Only，记录成功率、墙钟时间、决策时间、动作通道分布、回退和拒绝；不要只展示最佳一次运行。

最小的 JSON 任务示例在 [`inspect_report.json`](src/cua_jev/predefined/inspect_report.json)。它适合学习候选动作契约；真正的应用扩展还需要动态观察、执行器和终态验证器。

## Roadmap

- [x] Jev `choice` 接入、类型化候选、统一运行时、guard、回执和 JSONL trace。
- [x] GUI / DOM / COM / CLI / MCP / API 多通道执行接口；无密钥 Rule 基线。
- [x] 四个已在 Windows 验证的桌面任务、独立终态验证、Hybrid / GUI Only 配对实验与 Codex Hybrid pilot。
- [x] 只读项目网页、脱敏实验快照、真实视频与可展开执行步骤。
- [x] 实验性模型 + Jev 跨软件任务族，五来源、12 动作案例及独立来源核验。
- [ ] 用未见目标和网站进行足量重复试验；公布失败案例、置信区间、端到端延迟与美元成本，不能用单次成功案例代替基准。
- [ ] 把任务适配器做得更通用：跨软件、跨任务，逐步扩展到 macOS / Linux 与更多浏览器/办公应用。
- [x] 在实验性单窗口桌面路径中加入可选的截图/VLM 目标锚定；保留无 VLM 路径，并通过离线集成测试。
- [x] 将受限的视觉场景文本与 UIA 观察融合，并完成一次直连校园视觉模型的合成图像探针；加入 trace 级别的成功率、耗时和 Jev 通道概率分析。
- [x] 将浏览器 DOM 与视口视觉信息融合，加入可验证的视觉鼠标候选，并完成一次真实的 VLM＋文本模型＋Jev 单步浏览器联调；失败尝试也保留为本地实验记录。
- [x] 将浏览器、一个指定的 Windows UIA 窗口和受限 CLI/API 工具组合为开放任务循环，并完成一条真实三通道任务的独立终态验收。
- [x] 抽取初版能力提供者接口，以替代桌面后端做契约测试；裁剪模型输入，并在不同目标上真实回归 CLI → DOM → UIA 三通道任务。
- [x] 将需显式授权的固定配置 MCP 调用加入组合动作空间，并在同一轮任务中完成真实 MCP 协议＋DOM＋UIA 集成测试。
- [x] 浏览器＋CLI/MCP 组合任务不再强制依赖 Windows 窗口；加入受 schema 约束的模型填写参数并完成真实协议测试。
- [x] 加入调用者定义的来源访问、引文产物验收、同源 MCP 网页读取与逐页去重约束，以及确定性 20 动作跨通道回归测试。
- [x] 真实跑通并本地录制 18 动作受限长任务；加入模型自主发现来源、主题标题验收、深层链接优先显示，并完成 12 动作 Edge → VS Code 真实联调。
- [ ] 真正跑通并录制模型＋Jev 的约 20 步陌生目标任务，随后重复测量产物质量、失败类型、动作分布、耗时、token 与成本。
- [x] 增加编辑器式跨软件回归；编辑框实时值用于新鲜度和终态核验，不进入模型可见的桌面观察。
- [x] 在组合任务中完成一次真实的 VLM 场景融合联调；最终桌面动作仍是 UIA，视觉定位动作的收益尚未验证。
- [ ] 扩展现有提供者接口，接入 macOS Accessibility、动态 MCP 和更多软件；突破单窗口与任务族限制，并在两种系统上的陌生跨软件任务验证。
- [ ] 完成真实 VLM 任务及未见桌面任务的重复评测，报告定位、安全性、成功率、延迟和成本。
- [ ] 缓存或复用较慢的规划模型调用；让通用 CUA / LLM 模型处理陌生目标，Jev 承担高频受约束选择，以成功率、延迟和成本共同优化。
- [ ] 从可信 MCP 服务发现工具 schema 并核对注册声明；完善跨通道失败恢复、逐动作安全确认与长期回归基准。

## 安全与许可

`ActionGuard` 对候选身份、过期观察、允许根目录、写入和外部副作用执行 fail-closed 检查。Excel 不运行 VBA；MCP 只能调用已注册的类型化工具。成功回执不是终态成功，必须由任务验证器确认。API 密钥不写入 trace、网页快照或 Git 仓库；模型请求默认要求 HTTPS，只有显式允许时才会使用不安全的 HTTP。

代码以 [Apache-2.0](LICENSE) 发布。ZJU-REAL 标识用于表明实验室项目身份，不改变第三方品牌素材的权利归属；应用图标的来源见 [`ATTRIBUTION.md`](src/cua_jev/ui/static/icons/ATTRIBUTION.md)。

## 致谢

感谢 [TypeSafe AI](https://typesafe.ai/) 开发 Jev 并提供 API 与文档，使本项目的集成实验成为可能。也感谢 [RoboJEV](https://github.com/lykycy123/RoboJEV) 作者公开具身智能方向的 Jev 实践，启发我们进一步探索计算机使用这一多动作通道场景。CUA-JEV 是独立的研究与工程项目，并非 TypeSafe AI 或 RoboJEV 的官方产品。
