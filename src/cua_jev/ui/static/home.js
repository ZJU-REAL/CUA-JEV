"use strict";

const grid = document.querySelector("#windows-grid");
const status = document.querySelector("#case-status");
const macGrid = document.querySelector("#mac-grid");
const macStatus = document.querySelector("#mac-case-status");
const escapeHTML = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
const channelLabels = {gui:"GUI",script:"DOM",uia:"UI Automation",cli:"CLI",mcp:"MCP",api:"File API",com:"Excel COM"};
const channelColors = {gui:"#aa6e49",script:"#326983",uia:"#7f719c",cli:"#347b65",mcp:"#c09846",api:"#66a49a",com:"#88935c"};
const isText = (value) => typeof value === "string" && value.trim().length > 0;
const isCount = (value) => Number.isSafeInteger(value) && value >= 0;
const isChannel = (value) => Object.hasOwn(channelLabels, value);
const macChannelLabels = {script:"DOM",api:"API (file / AX)",gui:"GUI",cli:"CLI",mcp:"MCP"};

function validatedCases(data) {
  if (!data || data.schema_version !== 2 || !Array.isArray(data.cases) || data.cases.length > 4) {
    throw new Error("Invalid case catalog");
  }
  const ids = new Set();
  for (const item of data.cases) {
    if (!item || item.verified !== true || typeof item.id !== "string" || !/^[a-z0-9-]+$/.test(item.id) || ids.has(item.id) ||
        !isText(item.title) || !isText(item.summary) ||
        !/^media\/windows-[a-z0-9-]+\.mp4$/.test(item.video || "") ||
        (item.poster !== undefined && !/^media\/windows-[a-z0-9-]+\.(?:jpg|jpeg|png|webp)$/.test(item.poster)) ||
        !Array.isArray(item.apps) || !item.apps.length || !item.apps.every(isText) ||
        !isCount(item.actions) || item.actions < 17 || item.actions > 23 ||
        !isCount(item.jev_calls) || item.jev_calls !== item.actions ||
        !isCount(item.model_calls) || item.model_calls < 1 || !isCount(item.vlm_calls) ||
        !Number.isFinite(item.wall_time_s) || item.wall_time_s <= 0 ||
        !item.channels || typeof item.channels !== "object" || Array.isArray(item.channels) ||
        !Object.entries(item.channels).every(([id, count]) => isChannel(id) && isCount(count)) ||
        Object.values(item.channels).reduce((sum, count) => sum + count, 0) !== item.actions ||
        !Array.isArray(item.steps) || item.steps.length !== item.actions ||
        !item.steps.every((step) => step && isText(step.title) && isChannel(step.channel) &&
          (step.app === undefined || isText(step.app)))) {
      throw new Error("Invalid case record");
    }
    const actualChannels = {};
    for (const step of item.steps) actualChannels[step.channel] = (actualChannels[step.channel] || 0) + 1;
    if (Object.keys(channelLabels).some((id) => (actualChannels[id] || 0) !== (item.channels[id] || 0))) {
      throw new Error("Inconsistent case action mix");
    }
    ids.add(item.id);
  }
  return data.cases;
}

function validatedMacCases(data) {
  const object = (value) => value && typeof value === "object" && !Array.isArray(value);
  const keysMatch = (value, keys) => object(value) && Object.keys(value).length === keys.length && keys.every((key) => Object.hasOwn(value, key));
  const privateText = /[A-Za-z]:\\|\/Users\/|\/home\/|apikey_|TYPESAFE_API_KEY|sk-[A-Za-z0-9]/i;
  const publicText = (value, limit = 2000) => isText(value) && value.length <= limit && !privateText.test(value);
  const count = (value) => isCount(value) && value <= 1000000;
  if (!keysMatch(data, ["schema_version", "cases"]) || data.schema_version !== 1 || !Array.isArray(data.cases) || data.cases.length > 1) {
    throw new Error("Invalid macOS case catalog");
  }
  const fields = ["id", "verified", "title", "summary", "apps", "video", "poster", "actions", "jev_calls", "model_calls", "vlm_calls", "wall_time_s", "channels", "routes", "planner_model", "jev_model", "recording_scope", "timing_note", "steps"];
  for (const item of data.cases) {
    if (!keysMatch(item, fields) || item.id !== "python-onboarding" || item.verified !== true ||
        !["title", "summary", "planner_model", "jev_model", "recording_scope", "timing_note"].every((key) => publicText(item[key])) ||
        !item.jev_model.startsWith("jev-") || item.jev_model.toLowerCase().includes("fallback") ||
        !Array.isArray(item.apps) || item.apps.length < 1 || item.apps.length > 10 || !item.apps.every((app) => publicText(app, 100)) ||
        item.video !== "media/macos-python-onboarding.mp4" || item.poster !== "media/macos-python-onboarding.jpg" ||
        !["actions", "jev_calls", "model_calls", "vlm_calls"].every((key) => count(item[key])) ||
        item.actions < 1 || item.actions > 100 || item.jev_calls < item.actions || item.model_calls < 1 ||
        !Number.isFinite(item.wall_time_s) || item.wall_time_s <= 0 ||
        !object(item.channels) || !Object.entries(item.channels).every(([id, value]) => Object.hasOwn(macChannelLabels, id) && count(value)) ||
        Object.values(item.channels).reduce((sum, value) => sum + value, 0) !== item.actions ||
        !object(item.routes) || !Object.entries(item.routes).every(([id, value]) => ["ax", "gui"].includes(id) && count(value)) ||
        !Array.isArray(item.steps) || item.steps.length !== item.actions) {
      throw new Error("Invalid macOS case record");
    }
    const actualChannels = {script:0,api:0,gui:0,cli:0,mcp:0};
    const actualRoutes = {ax:0,gui:0};
    for (const step of item.steps) {
      if (!object(step) || !Object.keys(step).every((key) => ["title", "channel", "app", "route"].includes(key)) ||
          !publicText(step.title) || !Object.hasOwn(macChannelLabels, step.channel) ||
          (Object.hasOwn(step, "app") && !publicText(step.app, 100)) ||
          (Object.hasOwn(step, "route") && (!["ax", "gui"].includes(step.route) || step.channel !== {ax:"api",gui:"gui"}[step.route])) ||
          (step.channel === "gui" && step.route !== "gui")) {
        throw new Error("Invalid macOS action step");
      }
      actualChannels[step.channel] += 1;
      if (step.route) actualRoutes[step.route] += 1;
    }
    if (Object.keys(actualChannels).some((key) => actualChannels[key] !== (item.channels[key] || 0)) ||
        Object.keys(actualRoutes).some((key) => actualRoutes[key] !== (item.routes[key] || 0))) {
      throw new Error("Inconsistent macOS action counts");
    }
  }
  return data.cases;
}

function channelMix(channels, actions, labels = channelLabels) {
  const values = Object.entries(channels).filter(([, count]) => count > 0);
  return `<div class="mix-head"><b>Executed action routes</b><span>${actions} committed actions</span></div>
    <div class="mix-track" role="img" aria-label="${escapeHTML(values.map(([id,count]) => `${labels[id]}: ${count}`).join(", "))}">
      ${values.map(([id,count]) => `<span style="width:${(count / actions * 100).toFixed(2)}%;background:${channelColors[id]}"></span>`).join("")}
    </div><div class="mix-legend">${values.map(([id,count]) => `<span><i aria-hidden="true" style="background:${channelColors[id]}"></i>${escapeHTML(labels[id])} ${count}</span>`).join("")}</div>`;
}

function caseCard(item, index) {
  const steps = item.steps.map((step) => `<li><b>${escapeHTML(channelLabels[step.channel])}${step.app ? ` · ${escapeHTML(step.app)}` : ""}</b><span>${escapeHTML(step.title)}</span></li>`).join("");
  const poster = item.poster ? ` poster="${escapeHTML(item.poster)}"` : "";
  return `<article class="window-case" id="case-${item.id}" aria-labelledby="title-${item.id}">
    <div class="case-media"><div class="video-frame"><video controls playsinline preload="none"${poster} aria-label="${escapeHTML(item.title)}: recorded Windows workflow"><source src="${item.video}" type="video/mp4">Your browser cannot play this video. <a href="${item.video}">Open the recording</a>.</video></div><p class="video-status" hidden role="status">The video could not be loaded. Use the recording link below to open it directly.</p><div class="media-caption"><span>Task-window recording · ${escapeHTML(item.recording_speed || "1× playback")}</span><a href="${item.video}" aria-label="Open the recording for ${escapeHTML(item.title)}">Open recording ↗</a></div></div>
    <div class="case-detail"><p class="case-id"><span>CASE ${String(index + 1).padStart(2,"0")} / VERIFIED WINDOWS RUN</span><span>${item.wall_time_s.toFixed(1)} s wall time</span></p><h3 id="title-${item.id}">${escapeHTML(item.title)}</h3><p class="case-summary">${escapeHTML(item.summary)}</p><div class="app-list" aria-label="Applications">${item.apps.map((app) => `<span>${escapeHTML(app)}</span>`).join("")}</div>
      <dl class="case-metrics"><div><dt>Actions</dt><dd>${item.actions}</dd></div><div><dt>Jev calls</dt><dd>${item.jev_calls}</dd></div><div><dt>Model calls</dt><dd>${item.model_calls}</dd></div><div><dt>VLM calls</dt><dd>${item.vlm_calls}</dd></div></dl>
      ${channelMix(item.channels, item.actions)}
    </div><details class="step-drawer"><summary>Inspect all ${item.actions} verified actions</summary><ol>${steps}</ol></details>
  </article>`;
}

function macCaseCard(item) {
  const steps = item.steps.map((step) => `<li><b>${escapeHTML(step.route === "ax" ? "AX · API" : macChannelLabels[step.channel])}${step.app ? ` · ${escapeHTML(step.app)}` : ""}</b><span>${escapeHTML(step.title)}</span></li>`).join("");
  return `<article class="window-case mac-case" id="mac-case-${item.id}" aria-labelledby="mac-title-${item.id}">
    <div class="case-media"><div class="video-frame"><video controls playsinline preload="none" poster="${item.poster}" aria-label="${escapeHTML(item.title)}: recorded macOS workflow"><source src="${item.video}" type="video/mp4">Your browser cannot play this video. <a href="${item.video}">Open the recording</a>.</video></div><p class="video-status" hidden role="status">The video could not be loaded. Use the recording link below to open it directly.</p><div class="media-caption"><span>Reviewed macOS recording</span><a href="${item.video}" aria-label="Open the recording for ${escapeHTML(item.title)}">Open recording ↗</a></div><div class="mac-capture-notes"><p><strong>CAPTURE SCOPE</strong>${escapeHTML(item.recording_scope)}</p><p><strong>PLAYBACK &amp; TIMING</strong>${escapeHTML(item.timing_note)}</p></div></div>
    <div class="case-detail"><p class="case-id"><span>macOS / VERIFIED MODEL + JEV RUN</span><span>${item.wall_time_s.toFixed(1)} s wall time</span></p><h3 id="mac-title-${item.id}">${escapeHTML(item.title)}</h3><p class="case-summary">${escapeHTML(item.summary)}</p><div class="app-list" aria-label="Applications">${item.apps.map((app) => `<span>${escapeHTML(app)}</span>`).join("")}</div>
      <dl class="case-metrics"><div><dt>Actions</dt><dd>${item.actions}</dd></div><div><dt>Jev calls</dt><dd>${item.jev_calls}</dd></div><div><dt>Model requests</dt><dd>${item.model_calls}</dd></div><div><dt>VLM requests</dt><dd>${item.vlm_calls}</dd></div></dl>
      ${channelMix(item.channels, item.actions, macChannelLabels)}
      <p class="mac-native-routes"><strong>Native subset:</strong> AX ${item.routes.ax || 0} · GUI ${item.routes.gui || 0} (included above)</p><p class="mac-models">Planner: ${escapeHTML(item.planner_model)}<br>Decision policy: ${escapeHTML(item.jev_model)}</p>
    </div><details class="step-drawer"><summary>Inspect all ${item.actions} verified actions</summary><ol>${steps}</ol></details>
  </article>`;
}

function setupVideos(container = grid) {
  const videos = [...container.querySelectorAll("video")];
  for (const video of videos) {
    video.addEventListener("play", () => {
      for (const other of document.querySelectorAll(".case-media video")) if (other !== video) other.pause();
    });
    const showError = () => { video.closest(".case-media").querySelector(".video-status").hidden = false; };
    video.addEventListener("error", showError);
    video.querySelector("source").addEventListener("error", showError);
  }
}

async function loadMacCases() {
  macGrid.setAttribute("aria-busy", "true");
  macStatus.textContent = "Loading the macOS recording catalog…";
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("data/mac_demos.json", {signal: controller.signal});
    if (!response.ok) throw new Error("macOS case catalog unavailable");
    const cases = validatedMacCases(await response.json());
    macGrid.innerHTML = cases.map(macCaseCard).join("");
    macStatus.innerHTML = cases.length ? "" : '<p>No reviewed macOS recording is published here yet. <a href="https://github.com/ZJU-REAL/CUA-JEV/blob/main/docs/MACOS.md">The macOS guide</a> tracks completed integration checks and remaining work.</p>';
    setupVideos(macGrid);
  } catch {
    macGrid.innerHTML = "";
    macStatus.innerHTML = '<p>The macOS recording catalog could not be loaded. You can retry or <a href="data/mac_demos.json">open the separate macOS catalog</a>.</p><button class="retry-button" type="button">Try again</button>';
    macStatus.querySelector("button").addEventListener("click", loadMacCases, {once: true});
  } finally {
    window.clearTimeout(timeout);
    macGrid.setAttribute("aria-busy", "false");
  }
}

async function loadCases() {
  grid.setAttribute("aria-busy", "true");
  status.textContent = "Loading the recorded cases…";
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 12000);
  try {
    const response = await fetch("data/windows_demos.json", {signal: controller.signal});
    if (!response.ok) throw new Error("Case catalog unavailable");
    const cases = validatedCases(await response.json());
    grid.innerHTML = cases.map(caseCard).join("");
    if (cases.length === 4) status.textContent = "";
    else if (cases.length) status.textContent = `${cases.length} of 4 planned Windows recordings are available. Additional cases are still being evaluated.`;
    else status.innerHTML = '<p>No verified recordings have been published yet. <a href="https://github.com/ZJU-REAL/CUA-JEV">Explore the project on GitHub</a>.</p>';
    setupVideos();
  } catch {
    grid.innerHTML = "";
    status.innerHTML = '<p>The recorded cases could not be loaded. You can retry or <a href="data/windows_demos.json">open the published case data</a>.</p><button class="retry-button" type="button">Try again</button>';
    status.querySelector("button").addEventListener("click", loadCases, {once: true});
  } finally {
    window.clearTimeout(timeout);
    grid.setAttribute("aria-busy", "false");
  }
}

function setupNavigation() {
  const toggle = document.querySelector(".nav-toggle");
  const nav = document.querySelector("#main-nav");
  if (!toggle || !nav) return;
  toggle.hidden = false;
  const close = () => {
    toggle.setAttribute("aria-expanded", "false");
    toggle.querySelector("span").textContent = "+";
  };
  toggle.addEventListener("click", () => {
    const expanded = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(expanded));
    toggle.querySelector("span").textContent = expanded ? "−" : "+";
  });
  nav.addEventListener("click", (event) => { if (event.target.closest("a")) close(); });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && toggle.getAttribute("aria-expanded") === "true") {
      close();
      toggle.focus();
    }
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".masthead")) close();
  });
  window.matchMedia("(min-width: 801px)").addEventListener("change", close);
}

setupNavigation();
if (grid && status) loadCases();
if (macGrid && macStatus) loadMacCases();
