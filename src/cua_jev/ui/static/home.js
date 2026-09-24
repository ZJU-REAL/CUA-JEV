const grid = document.querySelector("#windows-grid");
const status = document.querySelector("#case-status");
const escapeHTML = (value) => String(value ?? "").replace(/[&<>'"]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[char]);
const channelLabels = {gui:"GUI",script:"DOM / script",uia:"UI Automation",cli:"CLI",mcp:"MCP",api:"File API",com:"Excel COM"};
const channelColors = {gui:"#b96843",script:"#376c94",uia:"#8265a5",cli:"#44735e",mcp:"#c39546",api:"#699ca1",com:"#7d9b59"};

function validatedCases(data) {
  if (!data || !Array.isArray(data.cases)) throw new Error("Case data is invalid.");
  return data.cases.filter((item) =>
    item && item.verified === true && typeof item.id === "string" &&
    typeof item.title === "string" && typeof item.video === "string" &&
    Array.isArray(item.apps) && Array.isArray(item.steps) &&
    item.steps.length === item.actions && Number.isInteger(item.actions) &&
    Number.isInteger(item.jev_calls) && Number.isInteger(item.model_calls) &&
    Number.isInteger(item.vlm_calls) && item.jev_calls > 0 && item.model_calls > 0 &&
    item.channels && Object.values(item.channels).reduce((sum, value) => sum + value, 0) === item.actions
  );
}

function channelMix(channels, actions) {
  const values = Object.entries(channels).filter(([, count]) => count > 0);
  return `<div class="mix-head"><b>Executed action space</b><span>${actions} committed actions</span></div>
    <div class="mix-track" role="img" aria-label="${escapeHTML(values.map(([id,count]) => `${channelLabels[id] || id}: ${count}`).join(", "))}">
      ${values.map(([id,count]) => `<span style="width:${(count / actions * 100).toFixed(2)}%;background:${channelColors[id] || "#7c8b96"}"></span>`).join("")}
    </div><div class="mix-legend">${values.map(([id,count]) => `<span><i style="background:${channelColors[id] || "#7c8b96"}"></i>${escapeHTML(channelLabels[id] || id)} ${count}</span>`).join("")}</div>`;
}

function caseCard(item, index) {
  const steps = item.steps.map((step) => `<li><b>${escapeHTML(channelLabels[step.channel] || step.channel)}${step.app ? ` · ${escapeHTML(step.app)}` : ""}</b><br><span>${escapeHTML(step.title)}</span></li>`).join("");
  const poster = item.poster ? ` poster="${escapeHTML(item.poster)}"` : "";
  return `<article class="window-case" id="case-${escapeHTML(item.id)}">
    <div class="case-media"><video controls playsinline preload="metadata"${poster} aria-label="${escapeHTML(item.title)} recorded Windows workflow"><source src="${escapeHTML(item.video)}" type="video/mp4">Your browser does not support the video.</video><div class="media-caption"><span>Real window recording · ${escapeHTML(item.recording_speed || "1× playback")}</span><span>${Number(item.wall_time_s).toFixed(1)} s measured wall time</span></div></div>
    <div class="case-detail"><p class="case-id">CASE ${String(index + 1).padStart(2,"0")} · VERIFIED WINDOWS RUN</p><h3>${escapeHTML(item.title)}</h3><p class="case-summary">${escapeHTML(item.summary)}</p><div class="app-list" aria-label="Applications">${item.apps.map((app) => `<span>${escapeHTML(app)}</span>`).join("")}</div>
      <dl class="case-metrics"><div><dt>Actions</dt><dd>${item.actions}</dd></div><div><dt>Jev calls</dt><dd>${item.jev_calls}</dd></div><div><dt>Model calls</dt><dd>${item.model_calls}</dd></div><div><dt>VLM calls</dt><dd>${item.vlm_calls}</dd></div></dl>
      ${channelMix(item.channels, item.actions)}
    </div><details class="step-drawer"><summary>Explore all ${item.actions} verified actions</summary><ol>${steps}</ol></details>
  </article>`;
}

async function loadCases() {
  try {
    const response = await fetch("data/windows_demos.json", {cache:"no-store"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const cases = validatedCases(await response.json());
    grid.innerHTML = cases.map(caseCard).join("");
    status.textContent = cases.length === 4 ? "" : `${cases.length} of 4 planned cases are verified and published. Remaining cases are still being evaluated.`;
    if (!cases.length) grid.innerHTML = '<p>No recorded Windows case is ready for publication yet.</p>';
  } catch (error) {
    status.textContent = `Verified case data is temporarily unavailable: ${error.message}`;
  }
}

loadCases();
