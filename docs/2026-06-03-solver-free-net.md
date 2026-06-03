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
history alone? With the right readout, **yes** — a 0.5M-param history-only net reaches
**99.87% / 3 losses playing token-only** (no candidate features, proven by random-feature
invariance). The original history-only net sat at random (~4.06); the unlock was an
architecture change, not size.

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

`models/policy_xattn.pt` (0.5M params, history-only), over all 2,315:

| Mode | Win | Avg | Losses |
|---|---|---|---|
| **raw (token-only)** | **99.87%** | **3.478** | **3** (`fjord, patty, sushi`) |
| hybrid (rail on top) | 100.0% | 3.465 | 0 |
| masked | 99.27% | 3.59 | 17 |

Token-only is **proven**: feeding random vs. zero features (even different dims) gives identical
logits — the head ignores them; the decision is the history alone. From *random* to *99.87%*,
at half the parameters of the feature net.

## 4. Honest notes

- **I bet against this.** Going in I expected brittle filtering to wreck it; it reached 99.87%.
  The cross-attention readout was the whole story.
- **The DAgger round to close the last 3 *regressed* it** — raw 99.87 → 99.22 while val-accuracy
  *rose* (0.636 → 0.725). The project's standing gotcha (val-acc ≠ game performance) plus a
  capacity-limited net overfitting the imitation target. The first net was better; we kept it.
- **The residual 3 are the brittle bits:** `fjord` (rare letters), `patty`/`sushi` (double
  letters — duplicate handling is the fiddliest part of the filter to approximate). The exact
  solver has zero such error. Closing them needs **game-performance-based selection** (RL on the
  net, or eval-selected checkpoints), not more imitation — that's open.
- **The environment still gives feedback.** We removed the solver's *candidate tracking*, not
  the game; Wordle still colors each guess. And 99.87% ≠ 100% — this is a *near*-solver-free net,
  a research result, not the shipped default (`policy_raw.pt`, 100%, still uses solver features).

## Lessons

1. **A capability "the net can't learn" is often a readout-structure problem.** Same tokens, same
   budget — only the head changed, and history-only went from random to near-perfect.
2. **Val-accuracy lies, every time we forget it.** It rose while play fell; rank on games.
3. **You can move the solver's exact work into a net, but it stays an approximation** — fine for
   99.9%, costly for the last 0.1% where exactness matters (duplicate letters).
