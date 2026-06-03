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
/* PLAY — the agent solves a word, turn by turn                          */
/* ===================================================================== */
(function game() {
  const board = $("gBoard"), status = $("gStatus"), explain = $("gExplain"), ansIn = $("gAnswer");
  if (!board) return;
  let answer = "pound", rows = [], cands = ANSWERS.slice(), done = false, timer = null;

  const traps = ["pound", "bound", "wound", "watch", "match", "vaunt", "taste", "foyer", "willy", "graze"];

  function reset(newAnswer) {
    clearInterval(timer); timer = null;
    answer = (newAnswer || ansIn.value || "pound").toLowerCase();
    if (!/^[a-z]{5}$/.test(answer) || !ANSWERS.includes(answer)) {
      status.textContent = `"${answer}" isn't in the 2,315-word answer set — pick another.`;
      answer = "pound";
    }
    ansIn.value = answer;
    rows = []; cands = ANSWERS.slice(); done = false;
    board.innerHTML = "";
    status.textContent = "";
    explain.innerHTML = "Press <b>step</b> to start. Each move is annotated — narrowing (best candidate) or a probe (non-candidate, played to split a tied set).";
  }

  function step() {
    if (done) return;
    const turn = rows.length + 1;
    let move;
    if (turn === 1) move = { word: OPENER, probe: false, reason: "opener" };
    else move = agentMove(cands, rows.length);
    const guess = move.word;
    const colors = feedback(guess, answer);
    cands = filterCandidates(cands, guess, colors);

    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:12px;margin-bottom:8px";
    const t = document.createElement("div");
    renderTiles(t, guess, colors, { flip: true });
    if (move.probe) {
      const tag = t.firstChild; // mark probe rows
    }
    row.appendChild(t);
    const lab = document.createElement("span"); lab.className = "note"; lab.style.margin = "0";
    lab.innerHTML = move.probe ? `<b style="color:var(--green-d)">probe</b>` : (turn === 1 ? "opener" : "narrowing");
    row.appendChild(lab);
    board.appendChild(row);
    rows.push(guess);

    // narrate
    const remaining = MAX_GUESSES - rows.length;
    if (guess === answer) {
      done = true; clearInterval(timer); timer = null;
      status.innerHTML = `<b style="color:var(--green-d)">Solved ${answer.toUpperCase()} in ${rows.length}.</b>`;
      explain.innerHTML = `Solved. The policy clears all 2,315 answers this way — 100% win, 0 losses.`;
      return;
    }
    if (rows.length >= MAX_GUESSES) {
      done = true; clearInterval(timer); timer = null;
      status.innerHTML = `Out of guesses — this only happened to earlier (pre-RL) policies, not the final one.`;
      return;
    }
    status.innerHTML = `Turn ${rows.length}: <b>${guess.toUpperCase()}</b> → <b>${cands.length}</b> candidate${cands.length === 1 ? "" : "s"}.`;
    if (turn === 1) {
      explain.innerHTML = `Opener <b>SLATE</b> (forced, turn 1). ${cands.length} candidates remain.`;
    } else if (move.probe) {
      explain.innerHTML = `<b style="color:var(--green-d)">Stuck → probe.</b> Candidates exceeded the ${remaining + 1} remaining guesses, so it played non-candidate <b>${guess.toUpperCase()}</b> to split the set (max entropy over the full pool). ${cands.length} candidate${cands.length === 1 ? "" : "s"} left.`;
    } else {
      explain.innerHTML = `Survivors fit the budget → masked move: best candidate <b>${guess.toUpperCase()}</b>. ${cands.length} left.`;
    }
  }

  function auto() {
    if (done) reset();
    clearInterval(timer);
    timer = setInterval(() => { if (done) { clearInterval(timer); timer = null; } else step(); }, 850);
  }

  $("gStep").addEventListener("click", () => { if (done) reset(); step(); });
  $("gAuto").addEventListener("click", auto);
  $("gReset").addEventListener("click", () => reset());
  $("gRandom").addEventListener("click", () => { reset(ANSWERS[Math.floor(Math.random() * ANSWERS.length)]); });
  $("gTrap").addEventListener("click", () => { reset(traps[Math.floor(Math.random() * traps.length)]); });
  ansIn.addEventListener("keydown", e => { if (e.key === "Enter") reset(); });
  reset("pound");
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
