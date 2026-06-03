# Research notebook — probing vs. openers (2026-06-02)

How we tried to close the behavior-cloned model's gap to the entropy solver, what
failed, and the one lever that worked. Numbers are measured over all 2315 answers unless
noted.

## TL;DR

- Drove the **live NYT Wordle** with the solver via Chrome DevTools; it solved #1809
  (BASIS) in **4**, the BC model in **5** — a clean demo of the model's one real weakness.
- That weakness is **probing**: the model never plays a non-candidate word to split a
  tied set. We decomposed the 3.62→3.50 gap and confirmed probing is the bulk of it.
- **Relabeling the BC data with probe targets failed** at both model sizes — a small net
  can't *rank* a probe whose value is a multi-step setup.
- **Switching the opener RAISE→SLATE worked**: free, Pareto, transfers to the net.
  Shipped models went **3.62→3.549 (transformer)** and **3.614→3.566 (MLP)**.
- The opener must be **forced at inference**, not learned — the net learns a one-example
  opening unreliably (left alone the SLATE net opens "crypt" and collapses to 3.88).

## 1. Setup — driving live NYT Wordle

Chrome DevTools MCP, two primitives: type a guess by dispatching synthetic `keydown`
events on `window` (React accepts them — one call per word + Enter); read feedback from
each tile's `data-state` (`correct/present/absent` → `g/y/x`). Next guess comes from a
stateless helper, `scripts/next_guess.py word:gyx …`, which rebuilds the candidate set
from history and prints the policy's pick.

| Policy | #1809 (BASIS) path | Guesses |
|---|---|---|
| Entropy teacher | raise → **antic** → basil → basis | 4 |
| BC model (RAISE) | raise → basic → basil → basin → basis | 5 |

At turn 2 both faced `{basic, basil, basin, basis, satin}`. The teacher played **ANTIC** —
a *non-candidate* — purely to split them. The masked model can't; it ranks survivors and
peels them off one per turn. That single difference is the thesis of the repo, live.

## 2. The gap, decomposed

| Policy | Avg | Losses | What it adds |
|---|---:|---:|---|
| BC model (masked) | 3.62 | ~14 | — *(where we were)* |
| Candidate-only teacher | 3.582 | 11 | the model's *actual* BC target |
| Full-pool teacher (3.50 benchmark) | **3.4955** | **0** | probing within the 2315 answers |
| Greedy entropy, allowed pool (12 972) | 3.4635 | 0 | obscure probes |
| Optimal (search, Selby 2022) | 3.4212 | 0 | lookahead |

Two findings here. (a) The model was distilled from a **candidate-only** teacher
(`dataset.py` labels used `pool=candidates`), *not* the 3.50 benchmark teacher
(`pool=all`) — so it was a faithful clone of a weaker policy. (b) Greedy entropy is
myopic: even with the full 12 972-word action space it tops out at 3.4635; the last
~0.04 to the 3.42 optimum needs **search**, not vocabulary. 3.42 is a search result, not
a learning one — Wordle is finite and deterministic, so the optimum is a static tree.

## 3. Experiment A — relabel with probe targets (failed)

Added `dataset.py --probe` to label with the full-pool teacher (spikes probes like
`antic` at 0.97 instead of committing to `basic`). Retrained both nets.

| Probe-trained net | masked | unmasked (win) | hybrid (win) |
|---|---:|---:|---:|
| MLP 0.34M | 3.97 | 30% | 51% |
| Transformer 1.04M | 4.07 | 59% | 94% / 4.39 |

**Worse everywhere**, and 3× capacity barely helped. Why: ranking *candidates* is an easy
readout (≈ most-frequent-letter consistent word) from the candidate-set features; ranking
a *probe* means computing information-gain for each of 2315 words against the live set —
structured `O(pool×candidates)` entropy the solver does explicitly, which BC can't
amortize into a small head from one argmax label per state. **Conclusion: probing is not a
labeling problem you can relabel away.** (Reverted; the `--probe` flag did not ship.)

## 4. Experiment B — the opener (worked)

The opening is *one* deterministic state (no feedback yet), so it isn't learned from data —
it seeds the whole self-play distribution and is otherwise a chosen constant. Greedy picks
RAISE by immediate entropy, but that's myopic. Sweeping openers under the model's own
policy (candidate-only greedy continuation):

| opener | avg | losses |
|---|---:|---:|
| **slate** | **3.518** | 7 |
| least | 3.523 | 9 |
| trace | 3.528 | 7 |
| crate | 3.534 | 8 |
| raise *(old default)* | 3.582 | 11 |

RAISE has the **highest** opening entropy and one of the **worst** averages. SLATE leads
to easier endgames.

**Combining openers (fixed turn-1 + turn-2) is worse**, not better — I hoped a fixed
second word would act as a baked-in probe; it doesn't:

| strategy | avg | losses |
|---|---:|---:|
| slate → crone (best fixed pair) | 3.777 | 11 |
| **slate, adaptive turn-2** | **3.518** | 7 |

Forcing a fixed turn-2 costs ~0.26 guesses and no fewer losses — for the ~half of games
where adaptive turn-2 already corners the answer, the fixed word burns a turn. Reacting to
feedback beats any pre-committed sequence.

## 5. Shipped — forced opener

Reseed the data from SLATE **and** force it at inference (the net can't reliably learn a
one-example opening). `config.opener` rides in the checkpoint; `ModelPolicy`/`evaluate.py`
play it on turn 1 and bypass the net. Retrained both at 24 epochs (12 undertrained the
post-SLATE states — a 12-epoch run misread as "barely transfers" at 3.606, fixed to 3.566
at 24).

| model | before | after | losses |
|---|---:|---:|---:|
| Transformer `policy.pt` | 3.62 | **3.549** | 14 → **7** |
| MLP `policy_mlp.pt` | 3.614 | **3.566** | 13 → 11 |

~0.05–0.07 better at **zero parameter cost** — the genuine "best small model" win of the
session. The transformer's 7 losses match the candidate-only teacher's loss count.

**Validation.** New SLATE model on live #1809: **SLATE → GASSY → BASIS, solved in 3** (vs
the old model's 5). Random self-play: COUPE 3, PLUMB 4, OUTDO 3 — right around 3.55.

## 6. Takeaways

- **Probing is an RL problem, not a labeling one.** No supervised label can teach a move
  whose entire value is a multi-step setup.
- **The opener is a chosen constant.** Seed the data from it and force it at play time;
  pick it by expected guesses (SLATE), not greedy entropy (RAISE).
- **Measure, don't assume.** Two of my mid-stream calls were wrong and the data corrected
  them: the opener "barely transfers" (undertrained checkpoint) and the gap is "just a
  labeling artifact" (probing is genuinely hard to learn).

## Follow-up (separate effort, now on `main`)

The "probing can't be distilled" result is exactly *why* RL works: a probe's payoff is a
return, not a per-state label. `rl.py` post-trains the BC transformer with GRPO +
teacher-demo exploration, played under a **hybrid rail** (mask while candidates fit the
budget, probe when they outnumber it). `models/policy_rl.pt` reaches **100% win, 3.481
avg, 0 losses** — it's the new CLI/evaluate default. See `AGENTS.md` Findings log.
