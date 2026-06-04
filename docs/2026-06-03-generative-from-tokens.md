# Generation from tokens alone: mapping the ceiling

*Research log — 2026-06-03. Author: Ihar + Clawd. Part 7 of 8. Previously:
[entropy-teacher-and-cloned-policies](2026-06-01-entropy-teacher-and-cloned-policies.md) →
[probing-and-openers](2026-06-02-probing-and-openers.md) →
[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md) →
[deprecating-the-rail](2026-06-03-deprecating-the-rail.md) →
[solver-free-net](2026-06-03-solver-free-net.md) →
[generative-decoder](2026-06-03-generative-decoder.md). Next:
[word-seed-decoder](2026-06-04-word-seed-decoder.md).*

## TL;DR

Two earlier results, combined into one hard question. The **solver-free net** (Part 5) plays from
*tokens only*. The **generative decoder** (Part 6) plays with *no vocabulary*, but it's fed the
solver's candidate features. What if we demand **both at once** — a letter-by-letter generator that
sees *only* the `(guess, color)` token history, no candidate features, no marginals, no trie? How
close to the classifier's 100% can it get?

Answer: **~43%, and that's near the ceiling.** We climbed there deliberately, naming and removing
one wall at a time — bare **11%** → cross-attention **30%** → learned marginal **36%** → a learned
word-LM prior **43%** (all RL'd). Every lever did what theory predicted. The last wall is
**structural and does not move**: a left-to-right generator conditioned on per-slot statistics
cannot do **per-word candidacy**, which is exactly what a classifier does — and the cross-attention
*classifier* already plays **100% from the same tokens** (Part 5). So "99% from tokens" was never
the open question; "99% from tokens *with a vocabulary-free generator*" is, and the answer is no.

## The method: measure-first, one wall at a time

The discipline that made this clean was refusing to guess at the bottleneck. For each rung we read
three numbers, not one:

- **trainCE vs. teacher-forced val CE** — is there a generalization gap? (capacity vs. structure)
- **teacher-forced exact-word accuracy** — given the *true* prefix, can the read-out even identify
  the target? (isolates the read-out from exposure bias)
- **free-running win / invalid-word rate** — the actual game.

`decoder.py --history-only [--xattn] [--learned-marginal [--marg-attn]] [--word-lm]`, all with
`--val-frac 0.05`, reports all three.

## The ladder

**Behavior cloning (the diagnosis):**

| BC variant | TF-exact | marg-acc | free-run win | invalid |
|---|---|---|---|---|
| bare (pooled seed) | 18% | — | 11% | 48% |
| + cross-attention (A) | 25% | — | 7% | 73% |
| + learned marginal, pooled (B) | 28% | 72% | 12% | 61% |
| + learned marginal, attention (B2) | 29% | 73% | 14% | 52% |
| + word-LM prior (#2) | 31% | 73% | **20%** | ~50% |

**After RL (the same checkpoints, GRPO + valid-word bonus):**

| RL'd policy | win | invalid |
|---|---|---|
| A — cross-attention | 29.8% | 30% |
| B — + learned marginal | 35.7% | ~25% |
| **#2 — + word-LM** | **42.7%** | 22% |

## What each rung taught us

1. **Not capacity.** The bare baseline showed `trainCE ≈ val CE` at every epoch — no
   generalization gap — while teacher-forced exact-word accuracy flat-lined at **~18%**. A model
   that fits train and val equally but still can't name the word *given the true prefix* isn't
   short on parameters; its **read-out** is the limit. (More layers / `d_model` were never tried in
   anger because Part 5 already showed they barely move this — same finding, generative side.)

2. **The read-out was a real wall, and cross-attention widens it.** Letting each decoder step
   cross-attend the full encoded history (instead of decoding from one pooled 128-d seed) lifted
   TF-exact 18 → 25%. The generative twin of Part 5's structural unlock.

3. **The candidate marginal is learnable from tokens — ~73%.** A small head, supervised at *train
   time* by the solver's per-slot frequencies, re-derives "which letters are alive in each slot"
   from the history alone, and feeds its *own prediction* to the decoder at inference (solver-free).
   This recovered the conditioning lever (B+RL 36%). Predicting the marginal with per-slot
   cross-attention queries (B2) instead of from the pooled seed nudged accuracy 72 → 73% and
   little else — **the marginal head was not the bottleneck**; ~73% is the encoder's ceiling for
   deriving candidacy from tokens.

4. **A learned word-LM is the biggest single lever, and the honest one.** A separate 5-letter
   word-LM (a GRU trained on the answer manifold — parametric, *no trie*) mixed in as a
   **product of experts** (`logit = decoder + α·wordLM`) pushed BC free-run win to 20% and
   RL to **42.7%**. The decoder says "which letters keep us in-set," the LM says "which letters
   continue a real word." It is the soft, learned version of the vocabulary constraint we banned.

## The wall that doesn't move

42.7%, still creeping, with **~22% invalid**. Two structural reasons, neither a tuning problem:

- **Soft validity has a floor.** A word-LM *lowers* the probability of fakes; only a hard trie
  *zeroes* them, and that's the constraint we removed. Sharper LM + higher α might recover another
  ~15pp (→ ~55–60%), but not the rest.
- **Per-slot conditioning ≠ per-word candidacy.** The marginal is *the same* for many different
  candidate sets that require *different* guesses. A left-to-right generator conditioned on
  marginals cannot resolve the endgame neighbour-traps — which needs joint, per-word reasoning.
  That is a **classifier**.

And we have the classifier: the cross-attention word head (Part 5) reads **only these tokens** and
plays **100%**. So the metric "99% from tokens" is already met — just not by a generator. A
generative head reaches 99% only by being *conditioned on* word-level scores, i.e., by containing a
classifier, at which point the generation is cosmetic.

## Lessons

1. **Measure the bottleneck before you optimise it.** The train-vs-val gap killed the "add more
   layers" instinct in ten minutes; teacher-forced exact-word accuracy separated read-out limits
   from exposure bias. We never tuned the wrong thing.
2. **Solver knowledge can be amortised into weights.** The candidate marginal is ~73% recoverable
   from tokens; distilling the solver at train time gives a solver-free policy at inference — the
   same move that internalised the rail (Part 4) and dropped the candidate features (Part 5).
3. **Match the head to the problem.** For a *closed* answer set, ranking beats generating: the
   classifier is valid-by-construction and does per-word candidacy for free. Generation only earns
   its keep with an *open* vocabulary — and there you'd constrain it with a trie anyway.

This is a research variant; nothing here ships. The infrastructure
(`use_candidates`/`decoder_xattn`/`decoder_learned_marginal`/`decoder_marg_attn`/`decoder_word_lm`
flags + the `decoder.py` diagnostics) is in the tree; the checkpoints are regenerable from the
commands above.
