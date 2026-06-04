# Building the base: an entropy teacher and the nets that cloned it

*Research log — 2026-06-01 to 06-02. Author: Ihar + Clawd. The origin of the project. Part 1 of 6; continues in
[probing-and-openers](2026-06-02-probing-and-openers.md) →
[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md) →
[deprecating-the-rail](2026-06-03-deprecating-the-rail.md) →
[solver-free-net](2026-06-03-solver-free-net.md) →
[generative-decoder](2026-06-03-generative-decoder.md).*

## TL;DR

We built a strong non-neural baseline (an entropy-maximizing solver: **100% win, 3.50
avg**), then trained a small PyTorch net to play by **cloning** it. The one finding that
made the net work: **the sufficient statistic for Wordle is the remaining candidate set,
not the move history.** Feed the net that, with a blended distillation target, and a 1.04M
transformer reaches **99.4% / 3.62** — both masked and unaided. An ablation then showed the
transformer's history encoder is *redundant*: a plain **0.34M-param MLP plays identically**.

| Piece | Params | Win | Avg | Source |
|---|---:|---|---|---|
| Entropy teacher (no net) | — | 100% | 3.50 | `solver.py`, commit `ed62ea4` |
| Behavior-cloned transformer | 1.04M | 99.4% | 3.62 | `model.py`, commit `ed62ea4` |
| Candidate-only MLP (perceptron) | 0.34M | 99.4% | 3.61 | `--no-history`, commit `c2f5192` |

---

## 1. The entropy teacher (the baseline that everything rides on)

Wordle is an information-theory problem, not a language one. The classic strong baseline
maximizes expected information gain:

- Precompute a full **pattern matrix** `P[guess, answer]` — the feedback code (base-3 over
  gray/yellow/green, 0–242) of every guess against every answer. 2315×2315, built once and
  cached. It's the hot path; everything downstream is a vectorized histogram over it.
- Each turn, score every guess by the **Shannon entropy** of its feedback distribution over
  the remaining candidates, and play the max — the guess expected to shrink the candidate
  set the most. Tie-break toward a guess that's itself still a candidate (so it can win
  outright).

Over all 2315 answers: **100% win, 3.4955 avg.** This solver is three things at once for
the rest of the project — the **teacher** to imitate, the **baseline** to beat, and the
**exact constraint engine** (candidate filtering) the net always rides on.

## 2. The behavior-cloned transformer — and the finding that made it work

The goal was a PyTorch net that predicts the next guess. The first instinct — a transformer
over the `(guess, feedback)` move history — **failed outright**:

> A transformer fed only the token history learned *nothing useful*. It sat exactly at the
> "random consistent guess" baseline (masked avg ~4.08 vs random 4.06).

Why: from raw history the net would have to re-derive the candidate set **and** re-emulate
entropy maximization from scratch. That's the whole solver, learned implicitly, from one
argmax label per state. No.

**The fix — feed the net its actual state.** The sufficient statistic for Wordle is the set
of remaining candidate words, which the solver already computes exactly each turn. We
summarize it as a 156-d feature vector (`encoding.candidate_features`):

- **per-position letter frequencies** (5×26) — drives green / position information;
- **letter-present-anywhere frequencies** (26) — drives yellow / presence information.

The presence half matters: per-position marginals alone are lossy for entropy (they ignore
letter co-occurrence and yellow info). Adding presence pushed masked validation accuracy
**65% → 85%**. With candidate features, ranking became learnable (masked play went from the
random baseline to ~3.88, then 3.62 with the richer features).

**The blended distillation target** (the other half, `dataset._soft_target`). The label for
each state is a *distribution* over candidates:

> weight = `(1 − α)` spike on the single best (max-entropy) candidate
>        + `α · softmax(entropy / T)` tail spread over the *other* candidates.   (α=0.4, T=0.3)

This was the key trade-off:

| Target | Masked play | Solo (unaided) play |
|---|---|---|
| Hard one-hot best candidate (α→0) | sharp ✅ | dumps leftover mass on **invalid** words → ~52% ❌ |
| Pure soft entropy (α→1) | dull → regresses to ~4.13 ❌ | valid ✅ (~73%) |
| **Blend: spike + soft tail over candidates** | sharp ✅ | valid ✅ |

The spike keeps masked ranking decisive; the soft tail keeps leftover probability on *other
candidates* rather than invalid words, so unaided play matches masked play. Result: **3.62
avg, 99.4% win, both masked and solo** — at the candidate-entropy ceiling (~3.58). Training
data is teacher self-play with a DAgger-style branch (with probability ε, play a random
remaining candidate, but always *label* the teacher's optimal move) so the net handles
off-policy states the CLI hits when the user guesses something else.

## 3. The candidate-only MLP — attention turned out to be unnecessary

If the candidate set is the sufficient statistic and the solver hands it to us exactly, what
is the transformer's history encoder *for*? Ablation answered: **nothing measurable.** A
`use_history=False` flag drops the encoder entirely, leaving a candidate-features MLP into
the word head.

- The **0.34M-param MLP plays identically** to the 1.04M transformer (3.61 / 99.4%).
- Other ablations: attention heads and encoder depth barely move the needle; halving
  `d_model` *does* hurt; a letter-factored output head roughly halves params for ~+0.06 avg.
- ~90% of the MLP's params are the output head (d → 2315 words) — the obvious shrink target.

So the model is best understood as **a learned ranker over the candidate set**; the solver
does the exact constraint propagation. The "sequence model" framing was a red herring.

## 4. A methodology trap worth flagging

**Don't rank models by validation top-1 accuracy.** It diverges from game performance — a
0.07M variant once had the *highest* val acc and the *worst* actual play. The only honest
ranking is to play all 2315 games (`evaluate.py`). This bit us early and is now a standing
convention.

## 5. Where this left us

A solid, tiny, honest player: **99.4% / 3.62**, matching the candidate-entropy ceiling, with
a 0.34M-param net that needs no attention. But a stubborn ~0.12 gap to the teacher's 3.50
remained, and ~14 answers were unwinnable while masked-to-candidates (the SLATE opener in
part 2 cut that to 7). Two follow-ups chased that gap:

- **[probing-and-openers](2026-06-02-probing-and-openers.md)** — diagnosed the gap as
  *probing* (the teacher plays non-candidate words to split tied sets), showed it can't be
  distilled by supervised relabeling, and banked the free win instead: a forced **SLATE**
  opener (3.62 → 3.55).
- **[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md)** — taught the net to
  probe with **RL on the terminal win**, reaching **100%**.

## Lessons

1. **Give a search/combinatorial policy its actual state; don't make the net reconstruct it
   from action history.** Candidate features turned a dead model into a working one.
2. **When distilling, a blended hard-spike + soft-tail-over-the-valid-actions target beats
   either pure hard or pure soft** — it sharpens the top choice without leaking probability
   onto invalid actions.
3. **Rank on the real task, not a proxy metric.** Val accuracy lied; full-game evaluation
   didn't.
4. **Ablate aggressively.** The headline architecture (a transformer) wasn't earning its
   parameters; a perceptron matched it.
