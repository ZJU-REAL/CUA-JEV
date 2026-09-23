const $ = (selector) => document.querySelector(selector);
const escapeHTML = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
const formatMs = (value) => value == null ? "—" : value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
const agentLabel = { jev: "Jev", rule: "Rule baseline", codex_computer_use: "Codex Computer Use", jev_with_fallback: "Jev + fallback" };
let tasks = {};
let demos = {};
let selectedTask = "edge";
let benchmarkRequest = 0;
const staticSite = document.documentElement.dataset.siteMode === "static";

function dataURL(url) {
  if (!staticSite) return url;
  if (url === "/api/bootstrap") return "data/bootstrap.json";
  const benchmark = url.match(/^\/api\/benchmarks\?task=([a-z]+)$/);
  if (benchmark) return `data/benchmarks/${benchmark[1]}.json`;
  const steps = url.match(/^\/api\/runs\/([a-zA-Z0-9-]+)\/steps$/);
  if (steps) return `data/steps/${steps[1]}.json`;
  throw new Error("This request is not part of the published read-only snapshot.");
}

const caseMeta = {
  edge: { app: "EDGE", className: "edge", summary: "Public store checkout", stages: "Sign in · sort · cart · checkout · receipt" },
  excel: { app: "EXCEL", className: "excel", summary: "Analysis report delivery", stages: "7 metrics · 2 charts · COM verification" },
  vscode: { app: "VS CODE", className: "vscode", summary: "Multi-defect repair", stages: "8 defects · 9 test runs · source verification" },
  explorer: { app: "EXPLORER", className: "explorer", summary: "Quarterly release pipeline", stages: "10 reports · 5 artifacts · byte verification" }
};

const loopDetails = {
  observe: {
    kicker: "01 · OBSERVE",
    title: "Frame a typed decision",
    summary: "A task-specific adapter turns structured app state into the next set of legal choices for Jev.",
    jev: "Receives the current subgoal, structured state, and only the legal actions available now.",
    runtime: "Reads DOM, UI Automation, COM, terminal, or filesystem state through the task adapter."
  },
  select: {
    kicker: "02 · SELECT",
    title: "Choose intent and action space",
    summary: "Jev compares typed candidates across GUI and structured channels, then commits to one executable action.",
    jev: "Selects the next intent × channel pair from the constrained candidate set; it can reselect after new evidence.",
    runtime: "The task adapter offers typed candidates; the runtime validates Jev’s response."
  },
  execute: {
    kicker: "03 · EXECUTE",
    title: "Guard and execute the choice",
    summary: "The runtime checks scope and arguments before dispatching the selected action to the real Windows tool.",
    jev: "Selects an offered action with prebuilt arguments; it does not directly control the operating system.",
    runtime: "Applies safety guards, invokes PyAutoGUI, DOM, COM, CLI, MCP, script, or API, and records the receipt."
  },
  verify: {
    kicker: "04 · VERIFY",
    title: "Return independent evidence",
    summary: "A verifier checks the resulting application state, not merely whether the executor reported success.",
    jev: "Consumes the verified result on the next turn and continues, retries, or changes action space.",
    runtime: "Runs task assertions, records wall time and action mix, and terminates only when the task contract is satisfied."
  }
};

async function getJSON(url, retries = 0) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetch(dataURL(url), { cache: staticSite ? "default" : "no-store" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      return data;
    } catch (error) {
      lastError = error;
      if (attempt < retries) await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
  throw lastError;
}

function appIcon(id) {
  return `<img src="static/icons/${escapeHTML(id)}.svg" width="28" height="28" alt="">`;
}

function renderLoopStep(id) {
  const detail = loopDetails[id];
  if (!detail) return;
  $("#loop-kicker").textContent = detail.kicker;
  $("#loop-title").textContent = detail.title;
  $("#loop-summary").textContent = detail.summary;
  $("#loop-jev").textContent = detail.jev;
  $("#loop-runtime").textContent = detail.runtime;
  document.querySelectorAll("[data-loop-step]").forEach((button) => button.setAttribute("aria-selected", String(button.dataset.loopStep === id)));
}

function appPreview(meta, routes) {
  return `<div class="case-window ${meta.className}"><div class="window-bar"><span></span><span></span><span></span><b>${meta.app}</b></div><div class="window-content"><div class="window-sidebar"></div><div class="window-canvas"><i></i><i></i><i></i><i></i></div></div><div class="route-overlay">${routes.slice(0, 4).map((route) => `<span>${escapeHTML(route.split(" · ")[0])}</span>`).join("")}<b>Jev selects ↗</b></div></div>`;
}

function casePreview(id, meta, routes) {
  const sources = demos[id] || {};
  const initial = sources.hybrid || sources.gui_only;
  if (!initial) return appPreview(meta, routes);
  return `<div class="case-video"><video controls muted playsinline preload="metadata" src="${escapeHTML(initial)}" aria-label="${escapeHTML(meta.summary)} demonstration"></video><div class="video-switch" aria-label="Select demonstration mode">${sources.hybrid ? `<button data-video="${escapeHTML(sources.hybrid)}" aria-pressed="${initial === sources.hybrid}">Hybrid</button>` : ""}${sources.gui_only ? `<button data-video="${escapeHTML(sources.gui_only)}" aria-pressed="${initial === sources.gui_only}">GUI Only</button>` : ""}</div></div>`;
}

function renderCases() {
  $("#case-grid").innerHTML = Object.entries(tasks).map(([id, task]) => {
    const meta = caseMeta[id];
    return `<article class="case-card"><div class="case-preview">${casePreview(id, meta, task.hybrid_routes || [])}<span class="case-app-badge">${appIcon(id)}<b>${meta.app}</b></span></div><div class="case-copy"><div class="case-meta"><span>VERIFIED WORKFLOW</span><b>${task.steps} STEPS</b></div><h3>${meta.summary}</h3><p>${escapeHTML(task.description)}</p><small>${meta.stages}</small></div></article>`;
  }).join("");
  document.querySelectorAll(".video-switch button").forEach((button) => button.addEventListener("click", () => {
    const wrapper = button.closest(".case-video");
    const video = wrapper.querySelector("video");
    video.pause();
    video.src = button.dataset.video;
    video.load();
    wrapper.querySelectorAll("button").forEach((item) => item.setAttribute("aria-pressed", String(item === button)));
  }));
}

function renderTabs() {
  $("#task-tabs").innerHTML = Object.keys(tasks).map((id) => `<button role="tab" aria-selected="${id === selectedTask}" data-task="${id}">${appIcon(id)}<span>${escapeHTML(caseMeta[id].app)}</span></button>`).join("");
  document.querySelectorAll("[data-task]").forEach((button) => button.addEventListener("click", async () => {
    selectedTask = button.dataset.task;
    renderTabs();
    await renderBenchmark();
  }));
}

function benchmarkRow(row, maxDuration, comparison) {
  const duration = row.median_wall_time_ms;
  const width = duration && maxDuration ? Math.max(6, duration / maxDuration * 100) : 0;
  const kind = row.action_space === "hybrid" ? "hybrid" : "gui";
  const mode = kind === "hybrid" ? "Hybrid" : "GUI Only";
  const label = comparison === "agent" ? (agentLabel[row.agent] || row.agent) : `${agentLabel[row.agent] || row.agent} · ${mode}`;
  const usd = row.agent === "codex_computer_use" ? row.median_reference_cost_usd : row.median_model_cost_usd;
  const cost = usd == null ? "—" : `$${usd < 0.01 ? usd.toFixed(4) : usd.toFixed(3)}`;
  const costCell = comparison === "agent" ? `<div class="result-cost" aria-label="Estimated model cost"><strong>${cost}</strong></div>` : "";
  const runId = escapeHTML(row.representative_run_id || "");
  return `<details class="result-detail" data-run-id="${runId}"><summary class="result-row ${comparison}-row"><span class="result-name"><b>${escapeHTML(label)}</b></span><span class="result-track"><i class="${kind}" style="--width:${width}%"></i><strong>${formatMs(duration)}</strong></span>${costCell}<span class="row-open">View steps</span></summary><div class="step-panel"><p>Open to load recorded steps.</p></div></details>`;
}

function renderRecordedSteps(data) {
  const steps = data.steps || [];
  if (!steps.length) return `<div class="step-empty">This run did not capture a step-level trace.</div>`;
  const source = data.source === "jev_run_trace" ? "Representative Jev run" : "Recorded Codex tool calls, grouped by stage";
  const verdict = data.terminal_verified ? "Terminal verified" : "Terminal result unavailable";
  const items = steps.map((step) => {
    const channel = step.channel ? escapeHTML(String(step.channel).toUpperCase()) : "ACTION";
    const tool = step.capability ? ` · ${escapeHTML(step.capability)}` : "";
    const status = step.verified === true ? "Verified" : data.source !== "jev_run_trace" ? "Recorded" : step.verified === false ? "Check failed" : "Check unavailable";
    const statusClass = step.verified === true ? "is-verified" : step.verified === false && data.source === "jev_run_trace" ? "is-failed" : "";
    return `<li><span class="step-number">${String(step.number).padStart(2, "0")}</span><div class="step-copy"><b>${escapeHTML(step.title)}</b><p>${channel}${tool}</p></div><span class="step-status ${statusClass}">${status}</span></li>`;
  }).join("");
  return `<div class="step-panel-head"><span>${source}</span><b>${verdict}</b></div><ol class="step-list">${items}</ol>`;
}

function bindStepDetails() {
  document.querySelectorAll(".result-detail").forEach((detail) => detail.addEventListener("toggle", async () => {
    if (!detail.open || detail.dataset.loaded) return;
    detail.dataset.loaded = "true";
    const panel = detail.querySelector(".step-panel");
    panel.innerHTML = `<p>Loading recorded steps…</p>`;
    if (!detail.dataset.runId) {
      panel.innerHTML = `<div class="step-empty">No successful run is available for this result.</div>`;
      return;
    }
    try {
      const data = await getJSON(`/api/runs/${encodeURIComponent(detail.dataset.runId)}/steps`, 1);
      panel.innerHTML = renderRecordedSteps(data);
    } catch (error) {
      detail.dataset.loaded = "";
      panel.innerHTML = `<div class="step-empty">Steps unavailable: ${escapeHTML(error.message)}. Close and reopen to retry.</div>`;
    }
  }));
}

function pairedRows(rows) {
  const hybrid = rows.find((row) => row.agent === "jev" && row.action_space === "hybrid" && row.median_wall_time_ms != null);
  const gui = rows.find((row) => row.agent === "jev" && row.action_space === "gui_only" && row.median_wall_time_ms != null);
  return hybrid && gui ? ["jev", { hybrid, gui_only: gui }] : null;
}

async function renderHeadlineMetrics() {
  const results = await Promise.all(Object.keys(tasks).map(async (task) => {
    try {
      const { rows = [] } = await getJSON(`/api/benchmarks?task=${encodeURIComponent(task)}`);
      const pair = pairedRows(rows);
      if (!pair) return null;
      const [, values] = pair;
      return values.gui_only.median_wall_time_ms / values.hybrid.median_wall_time_ms;
    } catch (_) {
      return null;
    }
  }));
  const speedups = results.filter((value) => Number.isFinite(value));
  $("#best-speedup").textContent = speedups.length ? `${Math.max(...speedups).toFixed(1)}×` : "—";
}

async function renderBenchmark() {
  const request = ++benchmarkRequest;
  const taskId = selectedTask;
  const task = tasks[taskId];
  $("#benchmark-name").textContent = task.title;
  $("#benchmark-description").textContent = task.description;
  ["#action-rows", "#agent-rows"].forEach((selector) => {
    $(selector).setAttribute("aria-busy", "true");
    $(selector).innerHTML = `<div class="empty-results"><b>Loading measured runs</b><span>Reading the local benchmark store…</span></div>`;
  });
  $("#speedup-callout").innerHTML = `<strong>—</strong><span>Hybrid speedup</span>`;
  try {
    const { rows = [] } = await getJSON(`/api/benchmarks?task=${encodeURIComponent(taskId)}`, 1);
    if (request !== benchmarkRequest) return;
    const pair = pairedRows(rows);
    if (pair) {
      const [, values] = pair;
      const actionRows = [values.hybrid, values.gui_only];
      const maxActionTime = Math.max(...actionRows.map((row) => row.median_wall_time_ms));
      $("#action-rows").innerHTML = actionRows.map((row) => benchmarkRow(row, maxActionTime, "action")).join("");
      const ratio = values.gui_only.median_wall_time_ms / values.hybrid.median_wall_time_ms;
      const faster = ratio >= 1 ? "Hybrid" : "GUI Only";
      $("#speedup-callout").innerHTML = `<strong>${Math.max(ratio, 1 / ratio).toFixed(1)}×</strong><span>${faster} faster</span>`;
    } else {
      $("#action-rows").innerHTML = `<div class="empty-results"><b>Jev pair unavailable</b><span>Run both Hybrid and GUI Only to compare action spaces.</span></div>`;
    }
    const jevHybrid = rows.find((row) => row.agent === "jev" && row.action_space === "hybrid" && row.median_wall_time_ms != null);
    const codexHybrid = rows.find((row) => row.agent === "codex_computer_use" && row.action_space === "hybrid" && row.median_wall_time_ms != null);
    if (jevHybrid && codexHybrid) {
      const agentRows = [jevHybrid, codexHybrid];
      const maxAgentTime = Math.max(...agentRows.map((row) => row.median_wall_time_ms));
      $("#agent-rows").innerHTML = agentRows.map((row) => benchmarkRow(row, maxAgentTime, "agent")).join("");
    } else {
      $("#agent-rows").innerHTML = `<div class="empty-results"><b>Hybrid baseline unavailable</b><span>This comparison needs successful Jev and Codex Hybrid runs.</span></div>`;
    }
    bindStepDetails();
  } catch (error) {
    if (request !== benchmarkRequest) return;
    ["#action-rows", "#agent-rows"].forEach((selector) => {
      $(selector).innerHTML = `<div class="empty-results"><b>Results unavailable</b><span>${escapeHTML(error.message)}</span><button class="retry-button" type="button">Retry</button></div>`;
    });
    document.querySelectorAll(".retry-button").forEach((button) => button.addEventListener("click", renderBenchmark, { once: true }));
  } finally {
    if (request === benchmarkRequest) ["#action-rows", "#agent-rows"].forEach((selector) => $(selector).setAttribute("aria-busy", "false"));
  }
}

async function boot() {
  try {
    const bootstrap = await getJSON("/api/bootstrap");
    tasks = bootstrap.tasks || {};
    demos = bootstrap.demos || {};
    renderCases();
    renderTabs();
    document.querySelectorAll("[data-loop-step]").forEach((button) => button.addEventListener("click", () => renderLoopStep(button.dataset.loopStep)));
    await Promise.all([renderBenchmark(), renderHeadlineMetrics()]);
  } catch (error) {
    $("#case-grid").innerHTML = `<div class="empty-results"><b>Project data unavailable</b><span>${escapeHTML(error.message)}</span></div>`;
  } finally {
    const id = location.hash.slice(1);
    const target = id && document.getElementById(id);
    if (target) requestAnimationFrame(() => {
      const root = document.documentElement;
      const previous = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      target.scrollIntoView({ block: "start" });
      root.style.scrollBehavior = previous;
    });
  }
}

boot();
