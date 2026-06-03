# A solver-free net: playing from the tokens alone

*Research log — 2026-06-03. Part 5 of 5. Previously:
[entropy-teacher-and-cloned-policies](2026-06-01-entropy-teacher-and-cloned-policies.md) →
[probing-and-openers](2026-06-02-probing-and-openers.md) →
[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md) →
[deprecating-the-rail](2026-06-03-deprecating-the-rail.md).*

## TL;DR

Every policy so far is fed a **candidate-set summary** the solver computes each turn. But the
candidate set is a deterministic function of the `(guess, feedback)` history — the information
is already in the tokens. So: can a net drop the solver's tracking and play from the raw
history alone? **Yes — and perfectly.** A 0.5M-param history-only net reaches **100% / 0 losses
playing token-only** (no candidate features, proven by random-feature invariance), avg 3.464.
The original history-only net sat at random (~4.06); three things got it to 100%: a
**cross-attention readout** (the structural unlock), a **duplicate-letter feature** derived from
the guess, and a **raw-GRPO** polish that optimised the win objective directly.

## 1. Why "the info is in the tokens" wasn't enough originally

A history-only transformer `history → one summary vector → Linear(→2315)` learned nothing
(Part 1). Two walls:

- **Readout bottleneck.** A single 128-d summary read out linearly can't represent an
  arbitrary subset of 2,315 words. "Which words are still consistent" is up to ~2,315 bits
  depending on the *specific* feedback; it doesn't fit. Bigger `d_model` barely helped — the
  *structure* was the limit.
- **Brittle filtering.** Candidacy is `feedback(gᵢ, w) == cᵢ` for every turn — a discrete
  function with duplicate-letter handling, where one letter flips the answer.

## 2. The fix: a per-word × history readout

Give each word a way to check itself against the *specific* history. `CrossAttnWordHead`:
embed each vocabulary word from its letters; let it **cross-attend the encoded history tokens**;
score it against its own attended context → a per-word logit. Now a word's logit depends on
how its letters relate to each past guess+color — the consistency check, learned implicitly.
We drop `candidate_features` entirely and feed **only tokens**, then train BC + DAgger.

## 3. Result

`models/policy_xattn.pt` (0.5M params, history-only), over all 2,315 — **raw == hybrid**:

| Mode | Win | Avg | Losses |
|---|---|---|---|
| **raw (token-only)** | **100.0%** | **3.464** | **0** |
| masked | ~99.4% | 3.55 | — |

Token-only is **proven**: feeding random vs. zero features (even different dims) gives identical
logits — the head ignores them; the decision is the history alone. From *random* to *perfect*, at
half the parameters of the feature net, with **no candidate set ever computed for the policy**.

## 4. Closing the last few — and the trap on the way

A first cross-attention net (DAgger'd on the hybrid expert's actions) reached **99.87% / 3 losses**
token-only — already a strong result. Two more steps closed it:

- **Duplicate-letter feature (`--letter-count`).** The brittle losses were rare/double-letter
  words (`fjord, patty, sushi`); duplicate handling is the fiddliest part of the filter to
  approximate. We add a per-token feature = *how many times this letter appears in its own guess*
  — a function of the guess (the net already sees it), so still solver-free. It cleared the
  double-letter cases.
- **Raw-GRPO (`rl_raw.py`).** Imitation and val-accuracy stop helping at the margin, so optimise
  the win objective directly: sample raw at every turn, reward = `solved ? (7-turns) : 0`, GRPO
  baseline, the full-pool expert seeded into each group on the hard answers, KL-anchored to the
  BC net. It closed the residual (`rajah, witch`) → **100% / 0 losses**.

`99.87 → (feature) → ~99.9 → (raw-GRPO) → 100`.

## 5. Honest notes

- **I bet against this twice, and was wrong twice.** First that brittle filtering would wreck a
  history-only net (it hit 99.87%); then that the last losses wouldn't close without big cost (the
  feature + a short RL polish closed them to 0).
- **One real trap on the way:** a naive **DAgger round regressed** the net — raw 99.87 → 99.22
  while *val-accuracy rose* (0.636 → 0.725). The standing gotcha (val-acc ≠ game performance). The
  fix that mattered was **selecting on raw win rate** (`train --select-play`), not imitation loss.
- **Margins are device-noisy.** At exactly this level, MPS vs CPU argmax can flip a borderline
  game; we verify on CPU (deterministic) — 0 losses there.
- **The environment still gives feedback.** We removed the solver's *candidate tracking*, not the
  game; Wordle still colors each guess. The solver-free net (`--xattn`) is a research variant; the
  shipped default stays `policy_raw.pt`.

## Lessons

1. **A capability "the net can't learn" is often a readout-structure problem.** Same tokens, same
   budget — only the head changed, and history-only went from random to near-perfect.
2. **Then it's a representation problem, then an objective problem.** The duplicate feature handed
   the net the fiddly sub-computation; raw-GRPO optimised the thing we actually cared about. Each
   layer of the gap had a different fix.
3. **Val-accuracy lies, every time we forget it.** It rose while play fell; rank on games.
