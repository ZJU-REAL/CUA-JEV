/* CUA-JEV public homepage: method figure, recorded Windows cases, copy buttons. */
(() => {
  "use strict";

  const esc = (value) => String(value ?? "").replace(/[&<>'"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"})[c]);
  const CH = {
    gui: "GUI", script: "DOM", uia: "UIA", com: "COM", cli: "CLI", mcp: "MCP", api: "File API",
  };
  const chLabel = (id) => CH[id] || String(id || "").toUpperCase();
  const chip = (id, text) => `<span class="cj-ch" data-ch="${esc(id)}">${esc(text || chLabel(id))}</span>`;
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Neutral glyphs only: no product or brand logos on this page.
  const GLOBE = '<svg viewBox="0 0 16 16" width="10" height="10" aria-hidden="true"><circle cx="8" cy="8" r="6.2" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M1.8 8h12.4M8 1.8c-3 3.4-3 9 0 12.4M8 1.8c3 3.4 3 9 0 12.4" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>';
  const APP_GLYPH = {"Edge": GLOBE, "Terminal": "&gt;_", "VS Code": "{}", "Notepad": "N", "MCP reader": "M", "Files": "F"};
  const appMark = (app) => `<i class="cj-glyph" aria-hidden="true">${APP_GLYPH[app] || esc(String(app || "?").charAt(0))}</i>`;

  /* ------------------------------------------------------------------
   * Hero: light replay of step 3 (subgoal -> candidates -> Jev -> guard -> click -> verify)
   * ------------------------------------------------------------------ */
  function initHero() {
    const fig = document.getElementById("cj-hero-fig");
    if (!fig) return;
    const url = fig.querySelector('[data-hk="url"]');
    const title = fig.querySelector('[data-hk="title"]');
    const PAGES = [
      ["https://docs.python.org/3/tutorial/introduction.html", "3. An Informal Introduction to Python"],
      ["https://docs.python.org/3/tutorial/controlflow.html", "4. More Control Flow Tools"],
    ];
    // 0 subgoal · 1 candidates · 2 Jev deciding · 3 chosen · 4 guard · 5 pointer to "next"
    // 6 hover · 7 click + load · 8 chapter 4 shown · 9 verify
    const HS = [500, 800, 1000, 1000, 800, 700, 1100, 500, 900, 700, 3600];
    const setStep = (n) => {
      fig.querySelectorAll("[data-hs]").forEach((el) => el.classList.toggle("is-shown", Number(el.dataset.hs) <= n));
      fig.classList.toggle("is-ptr", n >= 5 && n < 8);
      fig.classList.toggle("is-hover", n >= 6 && n < 8);
      fig.classList.toggle("is-click", n === 7);
      fig.classList.toggle("is-load", n >= 7);
      fig.classList.toggle("is-nav", n >= 8);
      const [u, t] = PAGES[n >= 8 ? 1 : 0];
      url.textContent = u;
      title.textContent = t;
    };
    // data-hs: 0 subgoal, 1 candidates, 2 Jev line, 3 chosen, 4 guard, 8 verify -> map to timeline
    fig.querySelectorAll('[data-hs="8"]').forEach((el) => { el.dataset.hs = "9"; });
    if (reduced) { setStep(9); return; }
    let token = 0;
    let visible = false;
    const sleep = (ms, t) => new Promise((r) => setTimeout(() => r(t === token), ms));
    async function loop() {
      const t = ++token;
      while (t === token) {
        setStep(-1);
        for (let n = 0; n < HS.length; n++) {
          if (!(await sleep(HS[n], t))) return;
          if (n < HS.length - 1) setStep(n);
        }
      }
    }
    setStep(-1);
    const toggle = (vis) => { visible = vis; if (vis) loop(); else token++; };
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((e) => { if (e[0].isIntersecting !== visible) toggle(e[0].isIntersecting); }, {threshold: 0.25}).observe(fig);
    } else {
      toggle(true);
    }
    window.cjHero = {show(n) { token++; setStep(n); }};
  }

  /* ------------------------------------------------------------------
   * Method figure: a desktop scene replaying five recorded steps of the
   * python-guide case. Step number, title, app and chosen channel are from
   * the run; screen frames are real; competing candidates are illustrative.
   * ------------------------------------------------------------------ */
  const DOC_LINES = [
    "# Python Automation Engineer Field Guide",
    "## 1. Introduction to Python",
    "Python is an interpreted, high-level programming language ideal for automation tasks…",
    "**Source:** https://docs.python.org/3/tutorial/introduction.html",
    "## 2. Control Flow Tools",
    "**Source:** https://docs.python.org/3/tutorial/controlflow.html",
    "## 3. Data Structures",
  ];
  const TERM_HEAD = '<span class="cj-sc-dim"># scoped CLI · registered commands only</span>\n';
  const PROMPT = "PS C:\\guide&gt; ";
  const NEXT = {x: 86, y: 6.2};

  const FIG = [
    {
      n: 1, ch: "script", app: "Edge", title: "Click 3. An Informal Introduction to Python using DOM",
      img: "fig-index", url: "https://docs.python.org/3/tutorial/index.html", tab: "The Python Tutorial",
      goal: "Open the next unread tutorial chapter",
      wins: ["br", "term"], main: ["br"], term: "",
      cands: [
        {ch: "script", host: "view", x: 30, y: 97, kind: "pin", label: "e19 ↓ “3. An Informal Introduction…”", dx: -20, dy: -40, pick: true},
        {ch: "gui", host: "view", x: 33, y: 95, kind: "cursor", label: "scroll, then click its box", dx: 34, dy: -76},
      ],
      guard: ["fresh observation", "same origin", "no write"],
      exec: {type: "swap", img: "fig-chapter3", url: "https://docs.python.org/3/tutorial/introduction.html", tab: "3. An Informal Introduction to Python"},
      verify: {host: "view", x: 63, y: 27, text: "chapter 3 loaded"},
    },
    {
      n: 2, ch: "mcp", app: "MCP reader", title: "Read one official Python documentation HTML page via scoped MCP",
      img: "fig-chapter3", url: "https://docs.python.org/3/tutorial/introduction.html", tab: "3. An Informal Introduction to Python",
      goal: "Record source evidence from this chapter",
      wins: ["br", "term"], main: ["br"], term: "",
      cands: [
        {ch: "mcp", host: "view", x: 98.5, y: 50, kind: "sheet", label: "read_page · this URL", dx: -190, dy: -14, pick: true},
        {ch: "script", host: "view", x: NEXT.x, y: NEXT.y, kind: "pin", label: "click “next”", dx: -150, dy: 18},
        {ch: "gui", host: "view", x: NEXT.x + 1.2, y: NEXT.y + 2, kind: "cursor", label: "click “next”", dx: -126, dy: 46},
      ],
      guard: ["fresh observation", "same origin", "read-only"],
      exec: {type: "sheet"},
      verify: {host: "view", x: 62, y: 88, text: "evidence 1 / 8 recorded"},
    },
    {
      n: 17, ch: "cli", app: "Terminal", title: "Read the installed Python version",
      img: "fig-stdlib", url: "https://docs.python.org/3/tutorial/stdlib.html", tab: "10. Brief tour of the standard library",
      goal: "Check the local Python runtime",
      wins: ["br", "term"], main: ["term"], term: "",
      cands: [
        {ch: "cli", host: "term", x: 34, y: 30, kind: "pin", label: "python --version", dx: -40, dy: -66, pick: true},
        {ch: "api", host: "scene", x: 80, y: 38, kind: "pin", label: "write the guide", dx: -40, dy: 14},
        {ch: "script", host: "view", x: NEXT.x, y: NEXT.y, kind: "pin", label: "click “next”", dx: -150, dy: 18},
      ],
      guard: ["fresh observation", "registered argv", "read-only"],
      exec: {type: "type", cmd: "python --version", out: "→ version string captured"},
      verify: {host: "term", x: 40, y: 92, text: "version in state"},
    },
    {
      n: 18, ch: "api", app: "Files", title: "Create the scoped python-automation-guide.md artifact",
      img: "fig-stdlib", url: "https://docs.python.org/3/tutorial/stdlib.html", tab: "10. Brief tour of the standard library",
      goal: "Write a cited guide from the evidence",
      wins: ["br", "term", "doc"], main: ["doc"], term: "python --version",
      cands: [
        {ch: "api", host: "doc", x: 50, y: 42, kind: "pin", label: "write python-automation-guide.md", dx: -120, dy: 14, pick: true},
        {ch: "gui", host: "doc", x: 30, y: 72, kind: "cursor", label: "type it in an editor", dx: -24, dy: 22},
      ],
      guard: ["fresh observation", "scoped path", "new file only"],
      exec: {type: "fill"},
      verify: {host: "doc", x: 50, y: 97, text: "file exists · cites sources"},
    },
    {
      n: 19, ch: "cli", app: "VS Code", title: "Open python-automation-guide.md in VS Code",
      img: "fig-stdlib", url: "https://docs.python.org/3/tutorial/stdlib.html", tab: "10. Brief tour of the standard library",
      goal: "Show the guide in an editor",
      wins: ["br", "term", "doc"], main: ["term"], term: "python --version", docFull: true,
      cands: [
        {ch: "cli", host: "term", x: 34, y: 70, kind: "pin", label: "open guide in editor", dx: -250, dy: -14, pick: true},
        {ch: "gui", host: "doc", x: 40, y: 18, kind: "cursor", label: "open via GUI", dx: 20, dy: 16},
      ],
      guard: ["fresh observation", "registered launch", "file in scope"],
      exec: {type: "type", cmd: "code python-automation-guide.md", editor: true},
      verify: {host: "ed", x: 66, y: 22, text: "guide open · task complete"},
    },
  ];

  const PHASE_MS = [1100, 1500, 1700, 1400, 1300, 1900, 1300, 1100];
  const HOLD_MS = 1300;
  const sceneMs = PHASE_MS.reduce((a, b) => a + b, 0) + HOLD_MS;
  const CURSOR = '<svg class="cj-sc-cur" viewBox="0 0 16 20" width="16" height="20"><path d="M1.5 1.5v15l4-3.6 2.7 6 2.4-1.1-2.7-5.8 5.4-.3z" fill="#fff" stroke="#111" stroke-width="1.3" stroke-linejoin="round"/></svg>';
  const SHIELD = '<svg viewBox="0 0 20 22" width="18" height="20"><path d="M10 1.5 2.5 4.5v6c0 4.8 3.2 8.4 7.5 10 4.3-1.6 7.5-5.2 7.5-10v-6z" fill="#e6f4ec" stroke="#1c7a46" stroke-width="1.5"/><path d="m6.5 11 2.4 2.4 4.6-5" fill="none" stroke="#1c7a46" stroke-width="1.7"/></svg>';
  const jsonl = (s) => `{"step":${s.n},"channel":"${s.ch === "script" ? "dom" : s.ch}","app":"${s.app}","verified":true}`;

  function initFigure() {
    const fig = document.getElementById("cj-fig");
    if (!fig) return;
    const sc = fig.querySelector(".cj-sc");
    const q = (k) => fig.querySelector(`[data-k="${k}"]`);
    const win = (w) => sc.querySelector(`[data-w="${w}"]`);
    const hosts = {
      view: sc.querySelector(".cj-sc-view"), term: sc.querySelector(".cj-sc-term-body"),
      doc: sc.querySelector(".cj-sc-doc-body"), ed: sc.querySelector(".cj-sc-ed img"), scene: sc,
    };
    const over = q("over");
    const tabsBox = fig.querySelector(".cj-fig-tabs");
    const playBtn = fig.querySelector(".cj-fig-play");
    const rail = [...fig.querySelectorAll(".cj-fig-rail li")];
    tabsBox.innerHTML = FIG.map((s, i) =>
      `<button type="button" role="tab" class="cj-fig-tab" aria-selected="${i === 0}" data-i="${i}"><span class="cj-fig-tab-t">Step ${s.n}</span>${chip(s.ch)}<i class="cj-fig-tab-bar" aria-hidden="true"></i></button>`).join("");
    const tabs = [...tabsBox.querySelectorAll(".cj-fig-tab")];
    const st = {scene: 0, phase: -1, playing: !reduced, visible: false, token: 0};
    let placed = [];

    function place(el, spec) {
      const h = hosts[spec.host];
      if (!h || !h.offsetParent) { el.hidden = true; return; }
      el.hidden = false;
      const s = sc.getBoundingClientRect();
      const r = h.getBoundingClientRect();
      el.style.left = `${r.left - s.left + r.width * spec.x / 100}px`;
      el.style.top = `${r.top - s.top + r.height * spec.y / 100}px`;
    }
    function add(html, spec, cls) {
      const el = document.createElement("div");
      el.className = cls;
      el.innerHTML = html;
      over.append(el);
      placed.push([el, spec]);
      place(el, spec);
      return el;
    }
    const relayout = () => placed.forEach(([el, spec]) => place(el, spec));
    if ("ResizeObserver" in window) new ResizeObserver(relayout).observe(sc);

    function termText(s, extra) {
      let t = TERM_HEAD;
      if (s.term) t += `${PROMPT}${esc(s.term)}\n<span class="cj-sc-dim">→ version string captured</span>\n`;
      return t + (extra || `${PROMPT}<span class="cj-sc-caret">▌</span>`);
    }

    function reset(i) {
      const s = FIG[i];
      st.scene = i;
      placed = [];
      over.innerHTML = "";
      sc.classList.remove("is-picked", "is-exec");
      ["br", "term", "doc", "ed"].forEach((w) => {
        const el = win(w);
        el.classList.toggle("is-shown", s.wins.includes(w));
        el.classList.toggle("is-main", s.main.includes(w));
        el.classList.remove("cj-ping");
      });
      win("ed").classList.remove("is-shown");
      q("imgA").src = `static/${s.img}.jpg?v=20261114n`;
      q("imgB").classList.remove("is-on");
      q("url").textContent = s.url;
      q("title").textContent = s.tab;
      q("sheet").className = "cj-sc-sheet";
      q("term").innerHTML = termText(s);
      q("doc").innerHTML = s.docFull ? DOC_LINES.map((l) => `<li class="is-on">${esc(l)}</li>`).join("") : "";
      win("doc").classList.toggle("is-empty", !s.docFull);
      q("bubble").classList.remove("is-on");
      q("goal").textContent = s.goal;
      const lines = FIG.slice(0, i).map((x) => `<li><code>${esc(jsonl(x))}</code></li>`).join("");
      q("trace").innerHTML = lines;
      tabs.forEach((t, j) => {
        t.setAttribute("aria-selected", String(j === i));
        const bar = t.querySelector(".cj-fig-tab-bar");
        bar.style.transition = "none";
        bar.style.width = j < i ? "100%" : "0%";
      });
      st.phase = -1;
      rail.forEach((li) => li.classList.remove("is-on", "is-done"));
    }

    function ping(el) {
      if (reduced || sc.classList.contains("cj-noanim")) return;
      el.classList.remove("cj-ping");
      void el.offsetWidth;
      el.classList.add("cj-ping");
    }

    function apply(ph) {
      const s = FIG[st.scene];
      st.phase = ph;
      rail.forEach((li, k) => { li.classList.toggle("is-on", k === ph); li.classList.toggle("is-done", k < ph); });
      const pick = s.cands.find((c) => c.pick);
      if (ph === 0) {
        s.main.forEach((w) => ping(win(w)));
      } else if (ph === 1) {
        q("bubble").classList.add("is-on");
      } else if (ph === 2) {
        s.cands.forEach((c, k) => {
          if (c.kind === "sheet") q("sheet").classList.add("is-peek");
          const mark = c.kind === "cursor" ? CURSOR : '<i class="cj-sc-pin"></i>';
          const el = add(`${mark}<span class="cj-sc-lab" style="left:${c.dx}px;top:${c.dy}px">${chip(c.ch)}<span>${esc(c.label)}</span></span>`,
            c, `cj-sc-chip cj-sc-k-${c.kind}${c.pick ? " is-pick" : ""}`);
          el.style.setProperty("--i", k);
          el.dataset.ch = c.ch;
        });
      } else if (ph === 3) {
        sc.classList.add("is-picked");
        const el = over.querySelector(".cj-sc-chip.is-pick .cj-sc-lab");
        if (el) el.insertAdjacentHTML("beforeend", '<b class="cj-sc-jev">Jev</b>');
      } else if (ph === 4) {
        add(`${SHIELD}<span>${s.guard.map((g) => `<i>✓ ${esc(g)}</i>`).join("")}</span>`,
          {host: pick.host, x: pick.x, y: pick.y, dx: 0, dy: 0}, "cj-sc-guard").firstElementChild;
        const g = over.lastElementChild;
        g.style.marginLeft = `${pick.dx}px`;
        g.style.marginTop = `${pick.dy + (pick.dy < 0 ? -46 : 34)}px`;
      } else if (ph === 5) {
        sc.classList.add("is-exec");
        const e = s.exec;
        if (e.type === "swap") {
          add("", pick, "cj-sc-rip");
          const b = q("imgB");
          b.src = `static/${e.img}.jpg?v=20261114n`;
          setTimeout(() => { b.classList.add("is-on"); q("url").textContent = e.url; q("title").textContent = e.tab; }, sc.classList.contains("cj-noanim") ? 0 : 600);
        } else if (e.type === "sheet") {
          q("sheet").classList.add("is-open");
        } else if (e.type === "type") {
          const n = e.cmd.length;
          q("term").innerHTML = termText(s, `${PROMPT}<span class="cj-sc-typed" style="--n:${n}">${esc(e.cmd)}</span>\n${e.out ? `<span class="cj-sc-dim cj-sc-out">${esc(e.out)}</span>` : ""}`);
          if (e.editor) {
            const show = () => win("ed").classList.add("is-shown", "is-main");
            if (sc.classList.contains("cj-noanim")) show(); else setTimeout(show, 1000);
          }
        } else if (e.type === "fill") {
          win("doc").classList.remove("is-empty");
          q("doc").innerHTML = DOC_LINES.map((l, k) => `<li style="--i:${k}">${esc(l)}</li>`).join("");
          requestAnimationFrame(() => q("doc").querySelectorAll("li").forEach((li) => li.classList.add("is-on")));
        }
      } else if (ph === 6) {
        add(`<b>✓</b> ${esc(s.verify.text)}`, s.verify, "cj-sc-ok");
      } else if (ph === 7) {
        q("trace").insertAdjacentHTML("beforeend", `<li class="is-new"><code>${esc(jsonl(s))}</code></li>`);
      }
    }

    function jump(i, ph) {
      sc.classList.add("cj-noanim");
      reset(i);
      for (let p = 0; p <= ph; p++) apply(p);
      requestAnimationFrame(() => requestAnimationFrame(() => sc.classList.remove("cj-noanim")));
    }

    function startBar(remainingMs, fromPct) {
      const bar = tabs[st.scene].querySelector(".cj-fig-tab-bar");
      bar.style.transition = "none";
      bar.style.width = `${fromPct}%`;
      void bar.offsetWidth;
      bar.style.transition = `width ${remainingMs}ms linear`;
      bar.style.width = "100%";
    }
    function freezeBar() {
      const bar = tabs[st.scene].querySelector(".cj-fig-tab-bar");
      const w = bar.getBoundingClientRect().width / Math.max(1, bar.parentElement.getBoundingClientRect().width) * 100;
      bar.style.transition = "none";
      bar.style.width = `${w}%`;
    }
    const wait = (ms, token) => new Promise((resolve) => setTimeout(() => resolve(token === st.token), ms));

    async function run() {
      const token = ++st.token;
      while (token === st.token) {
        const elapsed = PHASE_MS.slice(0, st.phase + 1).reduce((a, b) => a + b, 0);
        startBar(sceneMs - elapsed, elapsed / sceneMs * 100);
        for (let ph = st.phase + 1; ph < PHASE_MS.length; ph++) {
          apply(ph);
          if (!(await wait(PHASE_MS[ph], token))) return;
        }
        if (!(await wait(HOLD_MS, token))) return;
        reset((st.scene + 1) % FIG.length);
      }
    }

    function sync() {
      if (st.playing && st.visible) run();
      else { st.token++; freezeBar(); }
      playBtn.setAttribute("aria-pressed", String(st.playing));
      playBtn.setAttribute("aria-label", st.playing ? "Pause animation" : "Play animation");
      fig.classList.toggle("is-paused", !st.playing);
    }

    tabs.forEach((t) => t.addEventListener("click", () => {
      st.token++;
      const i = Number(t.dataset.i);
      if (reduced) { jump(i, PHASE_MS.length - 1); return; }
      reset(i);
      st.playing = true;
      sync();
    }));
    playBtn.addEventListener("click", () => {
      if (reduced) return;
      st.playing = !st.playing;
      if (st.playing && st.phase >= PHASE_MS.length - 1) reset((st.scene + 1) % FIG.length);
      sync();
    });

    if (reduced) {
      fig.classList.add("is-static");
      playBtn.hidden = true;
      jump(0, PHASE_MS.length - 1);
      return;
    }
    reset(0);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver((entries) => {
        const vis = entries[0].isIntersecting;
        if (vis !== st.visible) { st.visible = vis; sync(); }
      }, {threshold: 0.3}).observe(fig);
    } else {
      st.visible = true;
      sync();
    }
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState !== "visible") { st.token++; freezeBar(); } else if (st.playing && st.visible) run();
    });
    // For verification and debugging only.
    window.cjFigure = {show(i, ph) { st.token++; st.playing = false; jump(i, ph); sync(); }};
  }

  /* ------------------------------------------------------------------
   * Recorded Windows cases
   * ------------------------------------------------------------------ */
  function validCases(data) {
    if (!data || !Array.isArray(data.cases)) throw new Error("case data is invalid");
    return data.cases.filter((c) =>
      c && c.verified === true && typeof c.id === "string" && typeof c.title === "string" &&
      typeof c.video === "string" && Array.isArray(c.apps) && Array.isArray(c.steps) &&
      Number.isInteger(c.actions) && c.steps.length === c.actions &&
      Number.isInteger(c.jev_calls) && Number.isInteger(c.model_calls) && Number.isInteger(c.vlm_calls) &&
      c.channels && Object.values(c.channels).reduce((a, b) => a + b, 0) === c.actions);
  }

  const CH_ORDER = ["script", "uia", "gui", "com", "mcp", "cli", "api"];

  function mixBar(channels, total) {
    const rows = Object.entries(channels).filter(([, n]) => n > 0)
      .sort((a, b) => (CH_ORDER.indexOf(a[0]) + 99) % 99 - (CH_ORDER.indexOf(b[0]) + 99) % 99);
    const aria = rows.map(([id, n]) => `${chLabel(id)} ${n}`).join(", ");
    return `<div class="cj-mix"><div class="cj-mix-head"><span>Channel mix</span><span>${total} actions</span></div>
      <div class="cj-mix-bar" role="img" aria-label="Channel mix: ${esc(aria)}">${rows.map(([id, n]) =>
        `<span data-ch="${esc(id)}" style="flex:${n}"></span>`).join("")}</div>
      <div class="cj-mix-legend">${rows.map(([id, n]) => `<span><i data-ch="${esc(id)}"></i>${esc(chLabel(id))} <b>${n}</b></span>`).join("")}</div></div>`;
  }

  function caseCard(c, idx) {
    const num = String(idx + 1).padStart(2, "0");
    const poster = c.video.replace(/\.mp4$/, ".jpg");
    const wall = Number(c.wall_time_s);
    const wallTxt = Number.isFinite(wall) ? `${wall.toFixed(1)} s` : "—";
    const steps = c.steps.map((s, i) =>
      `<li><span class="cj-tr-n">${i + 1}</span>${chip(s.channel)}<span class="cj-tr-app">${appMark(s.app)}${esc(s.app || "")}</span><span class="cj-tr-t">${esc(s.title)}</span></li>`).join("");
    const models = [c.planner_model && `Planner <code>${esc(c.planner_model)}</code>`, c.jev_model && `Jev <code>${esc(c.jev_model)}</code>`].filter(Boolean).join(" · ");
    return `<article class="cj-case" id="case-${esc(c.id)}">
      <div class="cj-case-media">
        <div class="cj-win cj-case-win">
          <video class="cj-video" controls playsinline preload="none" poster="${esc(poster)}" width="640" height="360" aria-label="${esc(c.title)}: recorded Windows run">
            <source src="${esc(c.video)}" type="video/mp4">Your browser does not support embedded video.
          </video>
        </div>
        <p class="cj-case-cap">Real window-only recording · ${esc(c.recording_speed || "real-time playback")} · wall time ${wallTxt}</p>
      </div>
      <div class="cj-case-info">
        <p class="cj-case-k">Case ${num} · verified run</p>
        <h3>${esc(c.title)}</h3>
        <p class="cj-case-sum">${esc(c.summary)}</p>
        <div class="cj-apps" aria-label="Applications">${c.apps.map((a) => `<span>${appMark(a)}${esc(a)}</span>`).join("")}</div>
        <dl class="cj-stats">
          <div><dt>Actions</dt><dd>${c.actions}</dd></div>
          <div><dt>Jev calls</dt><dd>${c.jev_calls}</dd></div>
          <div><dt>Model calls</dt><dd>${c.model_calls}</dd></div>
          <div><dt>VLM calls</dt><dd>${c.vlm_calls}</dd></div>
          <div><dt>Wall time</dt><dd>${wallTxt}</dd></div>
        </dl>
        ${mixBar(c.channels, c.actions)}
        ${models ? `<p class="cj-models">${models}</p>` : ""}
      </div>
      <details class="cj-trace"><summary>Step trace · all ${c.actions} actions</summary><ol class="cj-tr-list">${steps}</ol></details>
    </article>`;
  }

  async function initCases() {
    const box = document.getElementById("cj-cases");
    const status = document.getElementById("cj-cases-status");
    if (!box) return;
    try {
      const res = await fetch("data/windows_demos.json", {cache: "no-store"});
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const cases = validCases(await res.json());
      box.innerHTML = cases.map(caseCard).join("");
      status.textContent = cases.length === 4 ? "" : `${cases.length} of 4 planned cases are published.`;
      status.hidden = cases.length === 4;
      if (!cases.length) box.innerHTML = "<p>No recorded Windows case is published yet.</p>";
      if (location.hash.startsWith("#case-")) {
        const target = document.getElementById(location.hash.slice(1));
        if (target) target.scrollIntoView();
      }
    } catch (err) {
      status.textContent = `Recorded case data is unavailable right now (${err.message}).`;
    }
  }

  /* ------------------------------------------------------------------
   * Copy buttons
   * ------------------------------------------------------------------ */
  function initCopy() {
    document.querySelectorAll(".cj-copy").forEach((btn) => btn.addEventListener("click", async () => {
      const code = btn.parentElement.querySelector("code");
      try {
        await navigator.clipboard.writeText(code.textContent);
        btn.textContent = "Copied";
      } catch {
        btn.textContent = "Select & copy";
      }
      setTimeout(() => { btn.textContent = "Copy"; }, 1600);
    }));
  }

  initHero();
  initFigure();
  initCases();
  initCopy();
})();
