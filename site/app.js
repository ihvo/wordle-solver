/* wordle-guesser — interactive story.
   Everything runs on the real 2315-word answer list (words.js → ANSWERS).
   The "agent" reproduces the strategy the RL net learned: the hybrid rail
   (best candidate when survivors fit the budget, entropy probe when stuck). */

"use strict";
const OPENER = "slate";
const MAX_GUESSES = 6;
const POOL = ANSWERS;                       // action space = the answer list
const A = "a".charCodeAt(0);

/* ---------- core Wordle rules ---------- */
// Returns a 5-char string of 'g'/'y'/'x' with correct duplicate handling.
function feedback(guess, answer) {
  const res = [0, 0, 0, 0, 0];              // 2=g, 1=y, 0=x
  const used = [false, false, false, false, false];
  for (let i = 0; i < 5; i++) if (guess[i] === answer[i]) { res[i] = 2; used[i] = true; }
  for (let i = 0; i < 5; i++) {
    if (res[i]) continue;
    for (let j = 0; j < 5; j++) {
      if (!used[j] && guess[i] === answer[j]) { res[i] = 1; used[j] = true; break; }
    }
  }
  return res.map(c => (c === 2 ? "g" : c === 1 ? "y" : "x")).join("");
}

function filterCandidates(cands, guess, colors) {
  return cands.filter(c => feedback(guess, c) === colors);
}

// Shannon entropy (bits) of a guess's feedback distribution over candidates.
function entropy(guess, cands) {
  const counts = new Map();
  for (let k = 0; k < cands.length; k++) {
    const p = feedback(guess, cands[k]);
    counts.set(p, (counts.get(p) || 0) + 1);
  }
  const n = cands.length;
  let h = 0;
  for (const c of counts.values()) { const pr = c / n; h -= pr * Math.log2(pr); }
  return h;
}

// The agent's move = the hybrid rail.
function agentMove(cands, guessesMade) {
  if (cands.length === 1) return { word: cands[0], probe: false, reason: "only one word left" };
  const remaining = MAX_GUESSES - guessesMade;
  const stuck = cands.length > remaining;          // can't just enumerate -> probe
  const pool = stuck ? POOL : cands;
  let best = pool[0], bestScore = -1;
  const candSet = new Set(cands);
  for (let i = 0; i < pool.length; i++) {
    const g = pool[i];
    // tiny tie-break toward a word that is itself still possible (can win outright)
    const score = entropy(g, cands) + (candSet.has(g) ? 1e-9 : 0);
    if (score > bestScore) { bestScore = score; best = g; }
  }
  return { word: best, probe: stuck && !candSet.has(best), reason: stuck ? "stuck" : "narrowing" };
}

/* ---------- tile rendering ---------- */
function tileEl(letter, color, sm, flipDelay) {
  const d = document.createElement("div");
  d.className = "tile" + (sm ? " sm" : "") + (letter ? " filled" : "") + (color ? " " + color : "");
  if (flipDelay != null) { d.classList.add("flip"); d.style.animationDelay = flipDelay + "ms"; }
  d.textContent = letter || "";
  return d;
}
function renderTiles(container, word, colors, { sm = false, flip = false } = {}) {
  container.innerHTML = "";
  const row = document.createElement("div");
  row.className = "tiles";
  for (let i = 0; i < 5; i++) {
    row.appendChild(tileEl(word ? word[i] : "", colors ? colors[i] : "", sm, flip ? i * 90 : null));
  }
  container.appendChild(row);
  return row;
}
const $ = id => document.getElementById(id);

/* ===================================================================== */
/* HERO — cycle a few flipping words                                     */
/* ===================================================================== */
(function hero() {
  const board = $("heroBoard");
  if (!board) return;
  const frames = [
    ["slate", "xxgxx"], ["pound", "ggxgg"], ["bound", "ggggg"],
  ];
  let i = 0;
  function show() {
    const [w, c] = frames[i % frames.length];
    renderTiles(board, w, c, { flip: true });
    i++;
  }
  show();
  setInterval(show, 2600);
})();

/* ===================================================================== */
/* STAGE 0 — feedback engine                                             */
/* ===================================================================== */
(function feedbackDemo() {
  const g = $("fbGuess"), a = $("fbAnswer"), out = $("fbTiles");
  if (!out) return;
  function run() {
    const gw = (g.value || "").toLowerCase().padEnd(5, " ").slice(0, 5);
    const aw = (a.value || "").toLowerCase().padEnd(5, " ").slice(0, 5);
    if (!/^[a-z]{5}$/.test(gw) || !/^[a-z]{5}$/.test(aw)) {
      renderTiles(out, gw.toUpperCase(), null); return;
    }
    renderTiles(out, gw, feedback(gw, aw), { flip: true });
  }
  $("fbGo").addEventListener("click", run);
  [g, a].forEach(el => el.addEventListener("keydown", e => { if (e.key === "Enter") run(); }));
  run();
})();

/* ===================================================================== */
/* STAGE 1 — candidate counter (chained filtering)                       */
/* ===================================================================== */
(function candidateCounter() {
  const tiles = $("candTiles"), countEl = $("candCount"), chips = $("candChips"), hint = $("candHint");
  if (!countEl) return;
  let cands = ANSWERS.slice();
  function draw(lastGuess, lastColors) {
    if (lastGuess) renderTiles(tiles, lastGuess, lastColors, { flip: true });
    countEl.textContent = cands.length;
    chips.innerHTML = "";
    cands.slice(0, 12).forEach(w => {
      const c = document.createElement("span"); c.className = "chip"; c.textContent = w; chips.appendChild(c);
    });
    if (cands.length > 12) { const m = document.createElement("span"); m.className = "chip"; m.textContent = "+" + (cands.length - 12) + " more"; chips.appendChild(m); }
    hint.textContent = cands.length === 0 ? "No consistent words — that feedback combination is unreachable. Reset to start over."
      : cands.length === 1 ? "One candidate left — that's the answer."
      : "A high-information guess splits these into the smallest expected groups.";
  }
  function apply() {
    const gw = ($("candGuess").value || "").toLowerCase();
    const cc = ($("candColors").value || "").toLowerCase();
    if (!/^[a-z]{5}$/.test(gw) || !/^[gyx]{5}$/.test(cc)) { hint.textContent = "Enter a 5-letter guess and 5 colors (g/y/x)."; return; }
    cands = filterCandidates(cands, gw, cc);
    draw(gw, cc);
  }
  $("candGo").addEventListener("click", apply);
  $("candReset").addEventListener("click", () => { cands = ANSWERS.slice(); draw(); });
  draw();
})();

/* ===================================================================== */
/* STAGE 2a — sufficient-statistic toggle                                */
/* ===================================================================== */
(function ssToggle() {
  const a = $("inHistory"), b = $("inCands"), box = $("ssExplain");
  if (!box) return;
  const texts = {
    hist: "<p style='margin:0'><b>Why it fails:</b> from raw history the policy has to reconstruct the candidate set <i>and</i> re-derive entropy maximization end-to-end, from one argmax label per state. It doesn't — it stays at the random-consistent baseline (~4.1 avg).</p>",
    cand: "<p style='margin:0'><b>Why it works:</b> the candidate set is the sufficient statistic, so the policy only has to rank guesses given its summary — a tractable readout. Win rate goes from random to <b>99.4%</b>.</p>",
  };
  function set(which) {
    a.classList.toggle("on", which === "hist"); b.classList.toggle("on", which === "cand");
    box.innerHTML = which === "hist" ? texts.hist : texts.cand;
  }
  a.addEventListener("click", () => set("hist"));
  b.addEventListener("click", () => set("cand"));
  set("hist");
})();

/* ===================================================================== */
/* STAGE 2b — per-position letter frequency bars                         */
/* ===================================================================== */
(function freqBars() {
  const grid = $("freqGrid"), countEl = $("freqCount");
  if (!grid) return;
  function compute(cands) {
    const cols = [];
    for (let i = 0; i < 5; i++) {
      const counts = new Array(26).fill(0);
      for (const w of cands) counts[w.charCodeAt(i) - A]++;
      const ranked = counts.map((c, k) => [String.fromCharCode(A + k), c]).filter(x => x[1] > 0)
        .sort((x, y) => y[1] - x[1]).slice(0, 4);
      cols.push(ranked);
    }
    return cols;
  }
  function draw(cands) {
    countEl.textContent = cands.length + " candidates — per-position letter marginals over the survivors:";
    const cols = compute(cands);
    grid.innerHTML = "";
    cols.forEach((ranked, i) => {
      const col = document.createElement("div"); col.className = "freqcol";
      col.innerHTML = `<h4>slot ${i + 1}</h4>`;
      const max = ranked.length ? ranked[0][1] : 1;
      ranked.forEach(([ltr, c]) => {
        const bar = document.createElement("div"); bar.className = "bar";
        bar.innerHTML = `<span class="ltr">${ltr}</span><span class="track"><span class="fill" style="width:${Math.round(100 * c / max)}%"></span></span>`;
        col.appendChild(bar);
      });
      grid.appendChild(col);
    });
  }
  function update() {
    const gw = ($("freqGuess").value || "").toLowerCase();
    const cc = ($("freqColors").value || "").toLowerCase();
    let cands = ANSWERS.slice();
    if (/^[a-z]{5}$/.test(gw) && /^[gyx]{5}$/.test(cc)) cands = filterCandidates(cands, gw, cc);
    draw(cands.length ? cands : ANSWERS.slice());
  }
  $("freqGo").addEventListener("click", update);
  $("freqReset").addEventListener("click", () => draw(ANSWERS.slice()));
  update();
})();

/* ===================================================================== */
/* STAGE 4 — the trap                                                    */
/* ===================================================================== */
(function trap() {
  const board = $("trapBoard"), msg = $("trapMsg"), chips = $("trapChips");
  if (!board) return;
  const family = ANSWERS.filter(w => w.endsWith("ound") && "bfhmprsw".includes(w[0]));
  const answer = "pound";            // the one we're stuck on
  const guessesLeft = 3;
  chips.innerHTML = "";
  family.forEach(w => { const c = document.createElement("span"); c.className = "chip"; c.textContent = w; chips.appendChild(c); });

  function showOneByOne() {
    board.innerHTML = "";
    // worst-case: try them in order, each wrong one only rules itself out
    let order = family.filter(w => w !== answer).slice(0, guessesLeft);
    order.forEach((w, i) => {
      const row = document.createElement("div"); row.style.marginBottom = "6px";
      renderTiles(row, w, feedback(w, answer)); board.appendChild(row);
    });
    msg.innerHTML = `<b>${guessesLeft} guesses</b>, <b>${family.length} look-alikes</b>. Enumerating one at a time, each miss only rules itself out — you can exhaust the budget before reaching <b>${answer.toUpperCase()}</b>. <span style="color:#b23">This is the masked-play ceiling (~99.7%).</span>`;
  }
  function showProbe() {
    board.innerHTML = "";
    // find a single non-family word whose colors split the family best
    let best = null, bestH = -1;
    for (const g of POOL) {
      if (family.includes(g)) continue;
      const h = entropy(g, family);
      if (h > bestH) { bestH = h; best = g; }
    }
    const row1 = document.createElement("div"); row1.style.marginBottom = "8px";
    renderTiles(row1, best, feedback(best, answer), { flip: true }); board.appendChild(row1);
    const after = filterCandidates(family, best, feedback(best, answer));
    const row2 = document.createElement("div");
    renderTiles(row2, answer, feedback(answer, answer), { flip: true }); board.appendChild(row2);
    msg.innerHTML = `One probe — <b>${best.toUpperCase()}</b> — isn't a candidate, but its feedback partitions the cluster. After it, only <b>${after.length === 1 ? after[0].toUpperCase() : after.length + " candidates"}</b> remain, so the next guess wins. <span style="color:var(--green-d)">This is the move masked cloning can't make.</span>`;
  }
  const b1 = $("trapGuessOne"), b2 = $("trapProbe");
  b1.addEventListener("click", () => { b1.classList.add("on"); b2.classList.remove("on"); showOneByOne(); });
  b2.addEventListener("click", () => { b2.classList.add("on"); b1.classList.remove("on"); showProbe(); });
  showOneByOne();
})();

/* ===================================================================== */
/* CHARTS (inline SVG)                                                    */
/* ===================================================================== */
function lineChart(el, pts, { ymin, ymax, yticks, xlabel, ylabel, flatLine }) {
  if (!el) return;
  const W = 720, H = 300, m = { l: 52, r: 16, t: 16, b: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xmax = pts[pts.length - 1][0], xmin = pts[0][0];
  const sx = x => m.l + (x - xmin) / (xmax - xmin) * iw;
  const sy = y => m.t + (1 - (y - ymin) / (ymax - ymin)) * ih;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${ylabel} over ${xlabel}">`;
  // grid + y ticks
  yticks.forEach(t => {
    svg += `<line class="grid" x1="${m.l}" y1="${sy(t)}" x2="${W - m.r}" y2="${sy(t)}"/>`;
    svg += `<text class="lbl" x="${m.l - 8}" y="${sy(t) + 4}" text-anchor="end">${t}</text>`;
  });
  // axes
  svg += `<line class="axis" x1="${m.l}" y1="${m.t}" x2="${m.l}" y2="${H - m.b}"/>`;
  svg += `<line class="axis" x1="${m.l}" y1="${H - m.b}" x2="${W - m.r}" y2="${H - m.b}"/>`;
  if (flatLine != null) {
    svg += `<line class="ln faint" x1="${m.l}" y1="${sy(flatLine)}" x2="${W - m.r}" y2="${sy(flatLine)}"/>`;
    svg += `<text class="lbl" x="${W - m.r}" y="${sy(flatLine) - 6}" text-anchor="end">${flatLine}% = perfect</text>`;
  }
  // line
  const d = pts.map((p, i) => (i ? "L" : "M") + sx(p[0]).toFixed(1) + " " + sy(p[1]).toFixed(1)).join(" ");
  svg += `<path class="ln" d="${d}"/>`;
  pts.forEach(p => { svg += `<circle class="pt" cx="${sx(p[0]).toFixed(1)}" cy="${sy(p[1]).toFixed(1)}" r="3"/>`; });
  // x labels (ends + middle)
  [xmin, Math.round((xmin + xmax) / 2), xmax].forEach(x => {
    svg += `<text class="lbl" x="${sx(x)}" y="${H - m.b + 22}" text-anchor="middle">${x}</text>`;
  });
  svg += `<text class="lbl" x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">${xlabel}</text>`;
  svg += `</svg>`;
  el.innerHTML = svg;
}

// RL training trajectory (hybrid win %, Run 3). Real eval points.
lineChart($("rlChart"),
  [[0,99.65],[25,99.44],[50,99.57],[75,99.52],[100,99.61],[125,99.70],[150,99.70],[175,99.74],
   [200,99.65],[225,99.70],[250,99.78],[275,99.83],[300,99.74],[325,99.74],[350,99.83],[375,99.91],
   [400,99.87],[425,99.78],[450,99.83],[475,99.91],[500,99.83],[525,99.83],[550,99.96],[575,100],
   [600,100],[625,99.78],[650,99.96],[675,100],[700,100]],
  { ymin: 99.3, ymax: 100.05, yticks: [99.4, 99.6, 99.8, 100], xlabel: "training rounds", ylabel: "win rate (%)", flatLine: 100 });

// Stage progression — grouped bars (avg guesses) + win-rate line.
(function progressionChart() {
  const el = $("progChart"); if (!el) return;
  const data = [
    { name: "Clone", avg: 3.62, win: 99.4 },
    { name: "+ opener", avg: 3.55, win: 99.7 },
    { name: "+ RL", avg: 3.48, win: 100 },
    { name: "Expert", avg: 3.50, win: 100 },
  ];
  const W = 720, H = 300, m = { l: 48, r: 48, t: 18, b: 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const aMin = 3.4, aMax = 3.7;                    // avg axis (left)
  const wMin = 99.0, wMax = 100.1;                 // win axis (right)
  const syA = v => m.t + (1 - (v - aMin) / (aMax - aMin)) * ih;
  const syW = v => m.t + (1 - (v - wMin) / (wMax - wMin)) * ih;
  const bw = iw / data.length, barW = 42;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="metrics by stage">`;
  [3.4, 3.5, 3.6, 3.7].forEach(t => {
    svg += `<line class="grid" x1="${m.l}" y1="${syA(t)}" x2="${W - m.r}" y2="${syA(t)}"/>`;
    svg += `<text class="lbl" x="${m.l - 8}" y="${syA(t) + 4}" text-anchor="end">${t}</text>`;
  });
  data.forEach((d, i) => {
    const cx = m.l + bw * i + bw / 2;
    const y = syA(d.avg);
    svg += `<rect x="${cx - barW / 2}" y="${y}" width="${barW}" height="${H - m.b - y}" rx="5" fill="var(--gray)" opacity="0.85"/>`;
    svg += `<text class="lbl" x="${cx}" y="${y - 6}" text-anchor="middle">${d.avg.toFixed(2)}</text>`;
    svg += `<text class="lbl" x="${cx}" y="${H - m.b + 20}" text-anchor="middle" style="font-weight:700;fill:var(--ink)">${d.name}</text>`;
  });
  // win-rate line on right axis
  const lp = data.map((d, i) => [m.l + bw * i + bw / 2, syW(d.win)]);
  svg += `<path class="ln" d="${lp.map((p, i) => (i ? "L" : "M") + p[0] + " " + p[1]).join(" ")}"/>`;
  lp.forEach((p, i) => {
    svg += `<circle class="pt" cx="${p[0]}" cy="${p[1]}" r="3.5"/>`;
    svg += `<text class="lbl" x="${p[0]}" y="${p[1] - 9}" text-anchor="middle" style="fill:var(--green-d);font-weight:700">${data[i].win}%</text>`;
  });
  [99, 99.5, 100].forEach(t => { svg += `<text class="lbl" x="${W - m.r + 8}" y="${syW(t) + 4}" text-anchor="start">${t}%</text>`; });
  svg += `</svg>`;
  el.innerHTML = svg;
})();

/* ===================================================================== */
/* ARCHITECTURE DIAGRAMS (inline SVG)                                     */
/* ===================================================================== */
(function diagrams() {
  const FILL = { in: "#eef1f4", param: "#eaf5e6", op: "#ffffff", dec: "#fff7e6", out: "#6aaa64" };
  const STROKE = { in: "#c3c7cc", param: "#6aaa64", op: "#c3c7cc", dec: "#c9b458", out: "#6aaa64" };
  const TXT = { in: "#1a1a1b", param: "#1a1a1b", op: "#1a1a1b", dec: "#1a1a1b", out: "#ffffff" };
  const MONO = "ui-monospace,SFMono-Regular,Menlo,monospace";
  function box(x, y, w, h, title, sub, kind = "op") {
    let s = `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="9" fill="${FILL[kind]}" stroke="${STROKE[kind]}" stroke-width="1.5"/>`;
    const ty = sub ? y + h / 2 - 2 : y + h / 2 + 4;
    s += `<text x="${x + w / 2}" y="${ty}" text-anchor="middle" style="font-weight:700;font-size:13px;fill:${TXT[kind]}">${title}</text>`;
    if (sub) s += `<text x="${x + w / 2}" y="${y + h / 2 + 14}" text-anchor="middle" style="font-weight:600;font-size:10.5px;font-family:${MONO};fill:${kind === "out" ? "#eafae6" : "#6b7075"}">${sub}</text>`;
    return s;
  }
  const arrow = (x1, y1, x2, y2, m) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#9aa0a6" stroke-width="1.5" marker-end="url(#${m})"/>`;
  const tag = (x, y, t, c) => `<text x="${x}" y="${y}" text-anchor="middle" style="font-weight:700;font-size:11px;font-family:${MONO};fill:${c || "#6b7075"}">${t}</text>`;
  const svg = (w, h, m, inner) => `<svg viewBox="0 0 ${w} ${h}" role="img"><defs><marker id="${m}" markerWidth="9" markerHeight="9" refX="6.5" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#9aa0a6"/></marker></defs>${inner}</svg>`;
  const set = (id, html) => { const el = document.getElementById(id); if (el) el.innerHTML = html; };

  // Entropy teacher — linear pipeline
  (function () {
    const m = "ahE", y = 34, h = 56, w = 152, gap = 176, x0 = 12; let s = "";
    s += box(x0 + 0 * gap, y, w, h, "Candidate set", "consistent answers", "in");
    s += box(x0 + 1 * gap, y, w, h, "Pattern matrix", "P[guess, answer]", "op");
    s += box(x0 + 2 * gap, y, w, h, "Feedback hist.", "per guess", "op");
    s += box(x0 + 3 * gap, y, w, h, "Entropy → argmax", "max info gain", "op");
    s += box(x0 + 3 * gap + w + 24, y, 116, h, "guess", "", "out");
    for (let i = 0; i < 3; i++) s += arrow(x0 + i * gap + w, y + h / 2, x0 + (i + 1) * gap, y + h / 2, m);
    s += arrow(x0 + 3 * gap + w, y + h / 2, x0 + 3 * gap + w + 24, y + h / 2, m);
    set("archTeacher", svg(x0 + 3 * gap + w + 24 + 116 + 12, 126, m, s));
  })();

  // Candidate-only MLP — single row
  (function () {
    const m = "ahM", y = 34, h = 56; let s = "";
    const it = [[12, 152, "Candidate features", "156-d", "in"], [200, 150, "Linear + GELU", "156→128", "param"],
      [382, 128, "Linear", "128→128", "param"], [542, 152, "Linear head", "128→2315", "param"],
      [726, 116, "logits", "2,315 words", "out"]];
    it.forEach(b => { s += box(b[0], y, b[1], h, b[2], b[3], b[4]); });
    for (let i = 0; i < it.length - 1; i++) s += arrow(it[i][0] + it[i][1], y + h / 2, it[i + 1][0], y + h / 2, m);
    set("archMLP", svg(726 + 116 + 12, 126, m, s));
  })();

  // Transformer policy (BC + RL) — two branches merging
  (function () {
    const m = "ahT", h = 56, yT = 40, yB = 200; let s = "";
    s += box(14, yT, 156, h, "History tokens", "(g,fb)×≤5  + START", "in");
    s += box(196, yT, 196, h, "Transformer encoder", "embed+pos · 3× · 4 heads", "param");
    s += box(414, yT, 120, h, "[START]", "128-d read-out", "op");
    s += box(14, yB, 156, h, "Candidate features", "156-d", "in");
    s += box(196, yB, 196, h, "MLP", "156→128→128, GELU", "param");
    s += box(566, 118, 108, h, "concat", "256-d", "op");
    s += box(700, 118, 166, h, "Linear head", "256→2315", "param");
    s += box(700, yB, 166, 46, "logits → guess", "argmax / mask", "out");
    s += arrow(170, yT + h / 2, 196, yT + h / 2, m);
    s += arrow(392, yT + h / 2, 414, yT + h / 2, m);
    s += arrow(170, yB + h / 2, 196, yB + h / 2, m);
    s += arrow(534, yT + h / 2, 566, 134, m);
    s += arrow(392, yB + h / 2, 566, 152, m);
    s += arrow(674, 118 + h / 2, 700, 118 + h / 2, m);
    s += arrow(783, 118 + h, 783, yB, m);
    set("archTransformer", svg(880, 300, m, s));
  })();

  // Hybrid rail — decision flow
  (function () {
    const m = "ahH", h = 54; let s = "";
    s += box(12, 86, 168, h, "State", "candidates C, left R", "in");
    s += box(214, 86, 150, h, "C ≤ R ?", "the rail", "dec");
    s += box(420, 24, 212, h, "mask → best cand.", "argmax over C", "op");
    s += box(420, 150, 212, h, "unmask → probe", "argmax over full pool", "param");
    s += box(674, 86, 118, h, "next guess", "", "out");
    s += arrow(180, 86 + h / 2, 214, 86 + h / 2, m);
    s += arrow(364, 102, 420, 51, m); s += tag(398, 82, "yes", "#5d9657");
    s += arrow(364, 126, 420, 177, m); s += tag(398, 150, "no", "#b23b3b");
    s += arrow(632, 51, 674, 100, m);
    s += arrow(632, 177, 674, 126, m);
    s += `<text x="12" y="212" style="font-weight:600;font-size:11px;font-family:${MONO};fill:#6b7075">turn 1 is forced to the fixed opener (SLATE), bypassing the net</text>`;
    set("archHybrid", svg(804, 230, m, s));
  })();

  // History-only cross-attention — no candidate features
  (function () {
    const m = "ahX", h = 52; let s = "";
    s += box(12, 24, 150, h, "History tokens", "(g,fb)×≤5", "in");
    s += box(190, 24, 180, h, "Transformer encoder", "embed+pos · 3×", "param");
    s += box(12, 132, 180, h, "Word ← letters", "(n_words, d)", "param");
    s += box(424, 78, 184, h, "cross-attention", "word attends history", "param");
    s += box(636, 78, 150, h, "logits", "per word", "out");
    s += arrow(162, 50, 190, 50, m);
    s += arrow(370, 50, 424, 95, m); s += tag(400, 64, "K,V", "#6b7075");
    s += arrow(192, 158, 424, 117, m); s += tag(300, 132, "query", "#6b7075");
    s += arrow(608, 104, 636, 104, m);
    set("archXattn", svg(800, 208, m, s));
  })();

  // Generative decoder — FULL pipeline: encode (history + candidates → state z) then
  // decode the word letter-by-letter (no vocabulary).
  (function () {
    const m = "ahG", g = "#6b7075"; let s = "";
    // --- ENCODE (top): two branches merge into state z ---
    s += box(14, 20, 148, 44, "History tokens", "(g,fb)×≤5 +START", "in");
    s += box(176, 20, 188, 44, "Transformer enc.", "embed+pos · 3× · 4 heads", "param");
    s += box(378, 20, 92, 44, "[START]", "128-d", "op");
    s += box(14, 84, 148, 44, "Candidate feats", "170-d (+C/R)", "in");
    s += box(176, 84, 188, 44, "MLP", "170→128, GELU", "param");
    s += box(540, 52, 116, 46, "state z", "concat · 256-d", "op");
    s += arrow(162, 42, 176, 42, m);
    s += arrow(364, 42, 378, 42, m);
    s += arrow(162, 106, 176, 106, m);
    s += arrow(470, 50, 540, 68, m); s += tag(508, 40, "128", g);
    s += arrow(364, 100, 540, 86, m); s += tag(452, 92, "128", g);
    // --- DECODE (bottom): state z → GRU writes 5 letters → word ---
    s += box(110, 246, 130, 50, "state_proj", "256→128 · h₀", "param");
    s += box(300, 246, 140, 50, "GRUCell ×5", "128→128", "param");
    s += box(480, 246, 110, 50, "out", "128→26", "param");
    s += box(632, 246, 110, 50, "word", "5 letters", "out");
    s += arrow(240, 271, 300, 271, m); s += tag(270, 265, "h₀", g);
    s += arrow(440, 271, 480, 271, m);
    s += arrow(590, 271, 632, 271, m);
    // bridge: state z → state_proj, routed through the clear corridor (y=134) then down the left
    s += `<path d="M598,98 L598,134 L64,134 L64,271 L110,271" fill="none" stroke="#9aa0a6" stroke-width="1.5" marker-end="url(#${m})"/>`;
    s += tag(80, 200, "z", g);
    // per-step inputs feeding the GRU
    s += box(300, 156, 96, 36, "prev letter", "from t−1", "in");
    s += box(412, 156, 96, 36, "letter_emb", "27→128", "param");
    s += box(300, 200, 96, 36, "marginal", "26-d / slot", "in");
    s += box(412, 200, 96, 36, "marg_proj", "26→128", "param");
    s += arrow(396, 174, 412, 174, m);
    s += arrow(396, 218, 412, 218, m);
    s += arrow(460, 192, 384, 246, m);
    s += arrow(460, 236, 402, 246, m);
    // autoregressive feedback: out → prev letter (routed above the input band, y=150)
    s += `<path d="M535,246 L535,150 L348,150 L348,156" fill="none" stroke="#9aa0a6" stroke-width="1.5" marker-end="url(#${m})"/>`;
    s += tag(560, 146, "feedback", g);
    set("archDecoder", svg(770, 312, m, s));
  })();
})();

/* ===================================================================== */
/* Scroll reveal + active nav                                            */
/* ===================================================================== */
(function reveal() {
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.12 });
  document.querySelectorAll(".reveal").forEach(el => io.observe(el));

  const links = [...document.querySelectorAll(".nav a.step")];
  const secs = links.map(l => document.querySelector(l.getAttribute("href")));
  const spy = new IntersectionObserver(es => {
    es.forEach(e => {
      if (e.isIntersecting) {
        const id = "#" + e.target.id;
        links.forEach(l => l.classList.toggle("active", l.getAttribute("href") === id));
      }
    });
  }, { rootMargin: "-45% 0px -50% 0px" });
  secs.forEach(s => s && spy.observe(s));
})();
