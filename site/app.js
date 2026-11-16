// Agent Replay project page. Every number comes from data.js, which scripts/site_data.py builds
// from the committed recordings in evals/.
(() => {
  const D = window.REPLAY;
  if (!D) return;
  const $ = (s, el = document) => el.querySelector(s);
  const fmt = (n) => n.toLocaleString("en-US");
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const short = (text, n) => (text.length > n ? `${text.slice(0, n).replace(/\s+\S*$/, "")}…` : text);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CHECKS = { completed: "Finished", expected_text: "Right number", judge: "Judge agrees", max_steps: "Within 5 steps", must_call_tools: "Used the tools" };
  const tag = (ok, yes = "pass", no = "fail") => `<span class="tag ${ok ? "ok" : "no"}">${ok ? yes : no}</span>`;
  const B = D.baseline, R = D.regressed;

  /* ---------- the player ---------- */
  const steps = $("#p-steps"), answer = $("#p-answer"), checks = $("#p-checks"), verdict = $("#p-verdict");
  let variant = "baseline", timers = [];

  function stepHTML(s) {
    if (s.kind === "model") {
      const what = s.asks_for.length ? `Model asks for <code>${esc(s.asks_for.join(", "))}</code>` : `Model writes the answer (${fmt(s.tokens_out)} tokens)`;
      return `<span class="kind">model</span><span>${what}</span><span class="ms">${fmt(s.ms)} ms</span>`;
    }
    const r = s.result || {};
    const got = s.tool === "get_order" ? `total ${r.total} ${r.currency}` : s.tool === "convert_currency" ? `${r.converted} ${r.to}` : "a result";
    return `<span class="kind tool">tool</span><span><code>${esc(s.tool)}</code> returns ${esc(got)}</span><span class="ms">${fmt(s.ms)} ms</span>`;
  }

  function play() {
    timers.forEach(clearTimeout);
    timers = [];
    const run = D[variant];
    steps.innerHTML = run.steps.map((s) => `<li class="step">${stepHTML(s)}</li>`).join("");
    answer.classList.remove("is-in");
    answer.innerHTML = `<span>Final answer</span>${esc(short(run.answer, 170))}`;
    checks.innerHTML = run.checks.map((c) => `<li class="check" data-ok="${c.passed}">${CHECKS[c.name] || c.name}</li>`).join("");
    verdict.innerHTML = "";
    const items = [...steps.children];
    const finish = () => {
      answer.classList.add("is-in");
      checks.querySelectorAll(".check").forEach((el) => el.classList.add(el.dataset.ok === "true" ? "ok" : "no"));
      const failed = run.checks.filter((c) => !c.passed).length;
      const text = failed ? `Fails ${failed} of ${run.checks.length} checks` : `Passes all ${run.checks.length} checks`;
      verdict.innerHTML = `${text}<small>${fmt(run.ms)} ms</small>`;
    };
    if (reduce) { items.forEach((el) => el.classList.add("is-in")); finish(); return; }
    // Played at four times the recorded speed. A step shows as running until its time is up.
    let t = 200;
    run.steps.forEach((s, i) => {
      const el = items[i];
      timers.push(setTimeout(() => el.classList.add("is-in", "is-running"), t));
      t += Math.max(s.ms / 4, 200);
      timers.push(setTimeout(() => el.classList.remove("is-running"), t));
    });
    timers.push(setTimeout(finish, t + 250));
  }
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
    variant = b.dataset.variant;
    document.querySelectorAll(".seg button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    play();
  }));
  $("#p-play").addEventListener("click", play);
  play();

  /* ---------- how it works ---------- */
  const firstModel = B.steps.find((s) => s.kind === "model");
  const hash = (h) => `${h.slice(0, 16)}…`;
  const STAGES = [
    {
      name: "Record",
      desc: "Every model call and tool call is saved with its tokens and time.",
      html: () => `<div class="p-head"><b>Run T11</b><span class="dim">baseline prompt, recorded live</span></div>
        <table><thead><tr><th>Step</th><th class="num">Time</th><th class="num">Tokens</th></tr></thead><tbody>
        ${B.steps.map((s) => `<tr><td>${s.kind === "model" ? "Model call" : `Tool <code>${esc(s.tool)}</code>`}</td><td class="num">${fmt(s.ms)} ms</td><td class="num">${s.kind === "model" ? fmt(s.tokens_in + s.tokens_out) : "0"}</td></tr>`).join("")}
        </tbody></table>
        <p class="p-note">Stored in Postgres as one run of ${B.steps.length} spans: ${fmt(B.ms)} ms and ${fmt(B.tokens_in + B.tokens_out)} tokens.</p>`,
    },
    {
      name: "Replay",
      desc: "Requests are hashed. A match is served from the recording with no model call.",
      html: () => `<div class="p-head"><b>Cache lookup</b><span class="dim">SHA-256 of the first model request</span></div>
        <table><thead><tr><th>Prompt</th><th>Request hash</th><th class="num">Lookup</th></tr></thead><tbody>
        <tr><td>Baseline</td><td><code>${hash(firstModel.request_hash)}</code></td><td class="num"><span class="tag hit">hit</span></td></tr>
        <tr><td>Changed</td><td><code>${hash(R.steps[0].request_hash)}</code></td><td class="num"><span class="tag no">miss</span></td></tr>
        </tbody></table>
        <p class="p-note">A hit replays in milliseconds and costs nothing. A changed prompt changes the hash, and in CI a miss is never sent to a model.</p>`,
    },
    {
      name: "Score",
      desc: "Four deterministic checks and an LLM judge. A task passes only if all five do.",
      html: () => `<div class="p-head"><b>Checks for T11</b><span class="dim">judged by a separate model</span></div>
        <table><thead><tr><th>Check</th><th class="num">Baseline</th><th class="num">Changed</th></tr></thead><tbody>
        ${B.checks.map((c, i) => `<tr><td>${CHECKS[c.name]}</td><td class="num">${tag(c.passed)}</td><td class="num">${tag(R.checks[i].passed)}</td></tr>`).join("")}
        </tbody></table>
        <p class="p-note">Judge on the changed prompt: "${esc(R.checks.find((c) => c.name === "judge").reason)}"</p>`,
    },
    {
      name: "Gate",
      desc: "The suite is compared with the baseline. Past a threshold, the check fails.",
      html: () => `<div class="p-head"><b>Regression gate</b><span class="tag no">FAIL</span></div>
        <table><thead><tr><th>Metric</th><th class="num">Baseline</th><th class="num">Changed</th><th class="num">Status</th></tr></thead><tbody>
        ${D.gate.map((g) => `<tr><td>${esc(g.metric)}<br><span class="dim">${esc(g.threshold)}</span></td><td class="num">${esc(g.baseline)}</td><td class="num">${esc(g.candidate)}</td><td class="num">${tag(g.status === "PASS")}</td></tr>`).join("")}
        </tbody></table>
        <p class="p-note"><code>gate-check</code> exits 1, so the required check fails and the pull request cannot merge.</p>`,
    },
  ];
  const tabs = $("#tabs"), panel = $("#stage-panel");
  tabs.innerHTML = STAGES.map((s, i) => `<button type="button" role="tab" id="tab-${i}" aria-controls="stage-panel" aria-selected="${i === 0}" tabindex="${i === 0 ? 0 : -1}"><span class="t-name">${s.name}</span><span class="t-desc">${s.desc}</span></button>`).join("");
  const tabButtons = [...tabs.children];
  function select(i, animate) {
    tabButtons.forEach((b, j) => { b.setAttribute("aria-selected", String(i === j)); b.tabIndex = i === j ? 0 : -1; });
    panel.setAttribute("aria-labelledby", `tab-${i}`);
    const swap = () => { panel.innerHTML = STAGES[i].html(); panel.classList.remove("is-swapping"); };
    if (!animate || reduce) { swap(); return; }
    panel.classList.add("is-swapping");
    setTimeout(swap, 160);
  }
  tabButtons.forEach((b, i) => {
    b.addEventListener("click", () => select(i, true));
    b.addEventListener("keydown", (e) => {
      const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!d) return;
      e.preventDefault();
      const n = (i + d + STAGES.length) % STAGES.length;
      tabButtons[n].focus();
      select(n, true);
    });
  });
  select(0, false);

  /* ---------- results ---------- */
  const g = Object.fromEntries(D.gate.map((r) => [r.metric, r]));
  const broke = D.board.filter((t) => t.baseline && !t.regressed).map((t) => t.id);
  const sec = (ms) => `${(Number(ms) / 1000).toFixed(1)} s`;
  const pass = g["pass rate"], p95 = g["p95 latency (ms)"], tok = g["mean total tokens"];
  $("#metrics").innerHTML = [
    ["Pass rate", `${pass.candidate} <span>from ${pass.baseline}</span>`, `${tag(false, "", "FAIL")} ${esc(pass.delta)}, limit 5 pp`],
    ["p95 latency", `${sec(p95.candidate)} <span>from ${sec(p95.baseline)}</span>`, `${tag(false, "", "FAIL")} ${esc(p95.delta)}, limit 20 %`],
    ["Mean total tokens", `${fmt(Math.round(tok.candidate))} <span>from ${fmt(Math.round(tok.baseline))}</span>`, `${tag(true, "PASS")} ${esc(tok.delta)}, limit 15 %`],
    ["Tasks that broke", `${broke.length} <span>of ${D.board.length}</span>`, esc(broke.join(", "))],
  ].map(([k, v, m]) => `<div class="metric"><dt>${k}</dt><dd class="val">${v}</dd><dd class="meta">${m}</dd></div>`).join("");

  const SHOW = new Set(["T08", "T09", "T10", "T11", "T12", "T13", "T14", "T15", "T16", "T17"]);
  const table = $(".tasks"), body = $("#tasks"), more = $("#tasks-more");
  const result = (t) => t.baseline === t.regressed
    ? `<span class="tag">${t.baseline ? "Still passes" : "Still fails"}</span>`
    : t.regressed ? `<span class="tag ok">Now passes</span>` : `<span class="tag no">Now fails</span>`;
  let open = "T11";
  function drawTasks() {
    body.innerHTML = D.board.map((t) => {
      const extra = SHOW.has(t.id) ? "" : " extra";
      const isOpen = t.id === open;
      const row = `<tr class="row${extra}${isOpen ? " is-open" : ""}" data-id="${t.id}" tabindex="0" aria-expanded="${isOpen}">
        <td class="id">${t.id}</td><td class="q">${esc(t.question)}</td>
        <td class="num ms-cell hide-sm">${fmt(t.baseline_ms)} ms</td><td class="num ms-cell hide-sm">${fmt(t.regressed_ms)} ms</td>
        <td class="num">${result(t)}</td></tr>`;
      const detail = isOpen ? `<tr class="detail${extra}"><td></td><td colspan="4"><div class="answers">
        <div><b>Baseline ${tag(t.baseline)}</b>${esc(short(t.baseline_answer, 240))}</div>
        <div><b>Changed prompt ${tag(t.regressed)}</b>${esc(short(t.regressed_answer, 240))}</div></div></td></tr>` : "";
      return row + detail;
    }).join("");
  }
  const toggle = (row) => { open = row.dataset.id === open ? null : row.dataset.id; drawTasks(); body.querySelector(`[data-id="${row.dataset.id}"]`).focus({ preventScroll: true }); };
  body.addEventListener("click", (e) => { const row = e.target.closest("tr.row"); if (row) toggle(row); });
  body.addEventListener("keydown", (e) => { const row = e.target.closest("tr.row"); if (row && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggle(row); } });
  table.dataset.collapsed = "";
  more.addEventListener("click", () => {
    const collapsed = table.hasAttribute("data-collapsed");
    table.toggleAttribute("data-collapsed", !collapsed);
    more.textContent = collapsed ? "Show fewer tasks" : "Show all 30 tasks";
    more.setAttribute("aria-expanded", String(collapsed));
  });
  drawTasks();

  /* ---------- prompt diff ---------- */
  $("#prompt-diff").innerHTML = D.prompt_change.map((c) => `<span class="${c.op === "removed" ? "rm" : "add"}">${esc(c.text)}</span>`).join("");

  /* ---------- copy ---------- */
  const copy = $("#copy");
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#cmd").textContent); copy.textContent = "Copied"; }
    catch { copy.textContent = "Select to copy"; }
    setTimeout(() => { copy.textContent = "Copy"; }, 1600);
  });

  /* ---------- sticky nav ---------- */
  const nav = $(".nav");
  const stick = () => nav.classList.toggle("is-stuck", scrollY > 4);
  addEventListener("scroll", stick, { passive: true });
  stick();

  /* ---------- reveal on scroll ---------- */
  const targets = [...document.querySelectorAll("[data-reveal]")];
  targets.forEach((el) => el.style.setProperty("--d", String([...el.parentElement.children].filter((c) => c.hasAttribute("data-reveal")).indexOf(el))));
  if (!("IntersectionObserver" in window)) { targets.forEach((el) => el.classList.add("is-in")); return; }
  const io = new IntersectionObserver((entries) => {
    for (const en of entries) if (en.isIntersecting) { en.target.classList.add("is-in"); io.unobserve(en.target); }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
  targets.forEach((el) => io.observe(el));
  addEventListener("load", () => setTimeout(() => targets.forEach((el) => { const r = el.getBoundingClientRect(); if (r.top < innerHeight && r.bottom > 0) el.classList.add("is-in"); }), 2000));
})();
