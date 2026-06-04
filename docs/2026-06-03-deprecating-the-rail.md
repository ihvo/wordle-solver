# Folding the rail into the net: raw play to 100%

*Research log — 2026-06-03. Part 4 of 8. Previously:
[entropy-teacher-and-cloned-policies](2026-06-01-entropy-teacher-and-cloned-policies.md) →
[probing-and-openers](2026-06-02-probing-and-openers.md) →
[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md). Next:
[solver-free-net](2026-06-03-solver-free-net.md).*

## TL;DR

The 100% policy from Part 3 leaned on the **hybrid rail** — an inference rule (mask to
candidates while they fit the budget, probe when stuck). Could the network internalize that
rule and play **raw** (pure `argmax`, no mask, no probe branch) at 100%? Yes. Two ingredients:
**(1)** add the candidate *count* and *remaining guesses* to the features (the rail's inputs,
which the normalized 156-d vector throws away), and **(2)** **DAgger** the raw net on the
hybrid expert. Raw win rate went `99.27% → 99.70% → 99.87% → 100.0%` over three rounds — and
DAgger (imitation) alone did it; the planned RL polish wasn't needed.

| Raw (unmasked) policy | Win | Avg | Losses |
|---|---|---|---|
| BC, 156-d (Part 2) | 99.27% | 3.56 | 17 |
| + C/R features, DAgger r0 | 99.70% | 3.48 | 7 |
| DAgger r1 | 99.87% | 3.48 | 3 |
| **DAgger r2 (shipped)** | **100.0%** | **3.461** | **0** |

On the final checkpoint **raw == hybrid == 100% / 3.461**: with the rail switched off it plays
identically and never loses. `models/policy_raw.pt` is the new default.

## 1. What "deprecate the rail" means

The rail does two jobs the raw policy must absorb: stay valid in normal states, and probe only
when stuck. The starting raw number was **99.31% / 16 losses** (Part 3) — the rail's two jobs,
unlearned.

## 2. The blocker was representation, not optimization

The rail tests `candidates ≤ guesses_left` (`C ≤ R`). The network could see **neither**:

- **`C`:** `candidate_features` is normalized (`pos /= n`, `present = mean`), so it's
  scale-invariant — `C = 2` and `C = 200` look identical. The count is discarded.
- **`R`:** only weakly present for the Transformer (via positions); the candidate-only MLP has
  **no** notion of the budget at all.

You can't learn a rule whose inputs aren't in the observation. So step 0 was a feature
augmentation: append `log C`, a small-count bucket (0,1,…,6+), and a remaining-guesses one-hot
→ 156-d **→ 170-d** (`CAND_DIM_AUG`), backward-compatible (old checkpoints still load). On
plain BC these extra dims do nothing (the BC target is candidate ranking) — they're the
*enabler* for the next step. This is exactly why the Part-2 relabel failed: the decision it
was being asked to learn wasn't observable.

## 3. DAgger from the hybrid expert

The hybrid policy is a perfect (100%) expert, so we imitate its **action** at every state, with
the new features, and play the student raw:

- **Round 0** — collect expert self-play (+ exploration over the full pool to cover the states a
  raw policy might wander into); hard one-hot targets; train (warm-started from the augmented BC
  net). Raw → 99.70%.
- **Rounds 1–2** — the DAgger step that matters: roll the *current student* out **raw**, record
  the states it actually visits, label them with the expert, aggregate, retrain. This fixes the
  policy's own mistakes under its own distribution. Raw → 99.87% → **100.0%**.

No reward, no RL — supervised imitation was enough once the decision was observable and the
student's own state distribution was covered.

## 4. Honest notes

- **I expected it to plateau short of 0.** Going in, the bet was that a learned threshold would
  stay fuzzy near `C ≈ R` and leave a few stubborn losses — "a theorem (the rail) vs. a function
  approximator." It hit exactly 0/2315. Pleasant to be wrong; worth flagging that it's still an
  *empirical* 0 over the closed universe, not a proof.
- **It's complete for this checkpoint.** Wordle's 2,315 answers are the whole input space and we
  evaluated all of them deterministically, so raw 0-losses is a full guarantee *for this
  checkpoint* — but a retrain isn't guaranteed to reproduce it. We keep the rail available as a
  free, exact safety net (`--safe`); it's just no longer required.
- **The opener stays forced.** Turn 1 is one deterministic state; the net learns a one-example
  opening unreliably (see `opener-is-a-forced-constant`). That's the only inference-time special
  case left.
- **Masked-only play regressed** (≈99.5%) — the net now prefers a probe even where masking would
  enumerate. Irrelevant: raw is the deploy mode.
- **The solver doesn't go away.** The 170-d input is still computed from the candidate set each
  turn. Deprecating the rail removes the `if C ≤ R` branch, not the candidate tracking.

## 5. What shipped

- `encoding.py`: optional `C`/`R` augmentation (`CAND_DIM_AUG = 170`), backward-compatible.
- `dagger.py`: imitate the hybrid expert with raw targets; `--student` for DAgger rounds.
- `train.py --init` (warm start), `dataset.py --state-aug`, aug threaded through `evaluate`/`policy`.
- `models/policy_raw.pt` — raw 100% / 0 losses / 3.461; the new CLI + `evaluate` default
  (play raw; `--safe` re-enables the rail, `--rl` is the Part-3 hybrid policy).

## Lessons

1. **Check the observation before blaming the optimizer.** A rule the net "can't learn" is often
   a rule whose inputs you never gave it. Normalizing away the count was the whole problem.
2. **DAgger beats reward when you have a perfect expert.** We reached for RL; imitation of the
   100% policy — over the student's own state distribution — was simpler and sufficient.
3. **Move provably-correct logic into the model only when the payoff is real.** Here it was a
   research result (can the net internalize the rail?) plus a cleaner deploy path — not a saved
   dependency, since the solver still featurizes the state.
