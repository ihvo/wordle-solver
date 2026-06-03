# AGENTS.md — wordle-guesser

Guidance for agents working in this repo. Keep this file current as the project moves.

## Goal

Build a Wordle helper whose core is a **PyTorch-trained model that identifies the most
probable next guess**. Training a neural net with PyTorch is a hard requirement (not an
optional add-on). An entropy-maximizing solver serves as the teacher, baseline, and
exact constraint engine the model rides on.

## Status (2026-06-03)

- **Done:** vectorized feedback engine; entropy teacher (100% win, 3.50 avg over all
  2315 answers); transformer policy behavior-cloned from it; candidate-only **MLP** (0.34M);
  all forced to open **SLATE** (see Conventions); interactive CLI; **29 passing tests**.
  Checkpoints committed under `models/`.
- **RL post-training → 100% (hybrid).** `rl.py` post-trains the BC transformer with GRPO +
  teacher-demo exploration to learn *probing*. Played **hybrid** (mask while candidates fit
  the budget, probe when stuck) `models/policy_rl.pt` is 100% / 3.481 / 0 losses. See Part 3.
- **The rail is now internalized → raw 100% (the current headline).** Adding the candidate
  *count* + *remaining guesses* to the features (156-d **→ 170-d**, `CAND_DIM_AUG`) and
  **DAgger**-ing the raw net on the hybrid expert (`dagger.py`) gives `models/policy_raw.pt`:
  **100% win, 3.461 avg, 0 losses playing pure raw `argmax`** — no mask, no probe branch
  (forced opener only). It internalized the rail; raw == hybrid. New CLI/evaluate default.
  See Part 4 / memory `rail-internalized-via-dagger`.
- **Why earlier supervised relabel failed but DAgger worked:** the rail tests `C ≤ R`, and the
  *normalized* 156-d features hide the candidate count (and the MLP the budget) — the decision
  wasn't observable. With C/R in the input, imitation of the 100% expert suffices (no RL).
- **Not done (see Next steps):** trim raw avg 3.461 → ~3.45; full ~10.6k allowed-guess action
  space; shrink the head at iso-quality.

## Setup

```bash
uv sync          # creates .venv on Python 3.12, installs torch/numpy/tqdm
```
Device auto-selects MPS → CUDA → CPU. Python 3.14 has no torch wheels — stay on 3.12.

## Workflows

### Rebuild a model from scratch
The shipped checkpoints open **SLATE** — pass the *same* `--opener` to both steps (the
dataset seeds from it, the model bakes it into `config.opener` and forces it at turn 1).
1. `uv run python -m wordle_guesser.dataset --opener slate` → self-play to `data/bc_dataset.npz`
2. `uv run python -m wordle_guesser.train --opener slate --epochs 24` → BC transformer to `models/policy.pt`
   - MLP variant: add `--no-history --out models/policy_mlp.pt`
   - the opener needs ~24 epochs to land cleanly; 12 undertrains the post-SLATE states
3. **RL post-train → 100% hybrid:** `uv run python -m wordle_guesser.rl` warm-starts from
   `models/policy.pt`, plays unmasked under the safety rail → `models/policy_rl.pt`. Selects
   on the hybrid win-rate; watch the printed `raw %` to confirm the BC anchor holds.
4. **Deprecate the rail → raw 100%:** with `dataset --state-aug` + `train`, get a 170-d BC
   net (`models/policy_aug.pt`); then **DAgger** it on the hybrid expert:
   `dagger ... | train --init ... | evaluate`, iterating with `dagger --student <ckpt>` to
   cover the student's own raw states. ~3 rounds take raw 99.3 → 100. Ships `models/policy_raw.pt`.
5. `uv run python -m wordle_guesser.evaluate [--model PATH]` → teacher / **raw** / hybrid / masked
6. `uv run pytest` before committing

### Play / debug a single game
`uv run wordle-guesser [--rl|--bc|--mlp|--teacher|--safe|--no-mask|--model PATH]`. Default is
the raw policy (`policy_raw.pt`) playing pure argmax. The pattern matrix rebuilds on first run.

## Key decisions

**Which policy to run:**

| Need | Use |
|---|---|
| Best play (100%, default) | raw — `models/policy_raw.pt` (no flag; pure argmax, no rail) |
| The RL hybrid policy (rail on) | `--rl` (`models/policy_rl.pt`) |
| Guaranteed-valid suggestions | `--safe` (overlay the rail on any model) |
| The behavior-cloned transformer | `--bc` (`models/policy.pt`, masked) |
| Smallest model | MLP — `--mlp` (`use_history=False`, masked) |
| No model, pure baseline | `--teacher` |

**Distillation target** (`dataset._soft_target`, the crux of model quality):

| Target | Masked play | Solo (raw) play |
|---|---|---|
| Hard one-hot best candidate (`alpha→0`) | sharp ✅ | dumps mass on invalid words ❌ |
| Soft entropy over candidates (`alpha→1`) | dull ❌ | valid ✅ |
| **Blended: `(1-alpha)`·best + `alpha`·softmax(entropy/T)** (`alpha=0.4, T=0.3`) | sharp ✅ | valid ✅ |

## Conventions & gotchas

- **Feedback codes are base-3.** color: `gray=0, yellow=1, green=2`; slot 0 is the least
  significant digit; a pattern is `sum(color_i * 3**i)`, range 0–242. Strings use
  `g/y/x` (`feedback.pattern_to_code` / `code_to_pattern`).
- **Don't make the model reconstruct state from history.** The sufficient statistic is
  the *candidate set*. **Do** pass `encoding.candidate_features` (156-d). A history-only
  model sits at the random baseline.
- **Don't let the net learn the opening.** Turn 1 is one deterministic state; trained on a
  single example it's learned unreliably (the SLATE net, left to itself, opens "crypt" and
  collapses to 3.88). **Do** set `config.opener` — `ModelPolicy`/`evaluate.py` force it at
  turn 1 and bypass the net. Greedy max-entropy (RAISE) is *myopic*: it wins turn-1 info
  but loses overall. SLATE/TRACE/LEAST/CRATE (~3.52 by expected guesses) beat RAISE (3.58);
  RAISE has the *highest* opening entropy and one of the *worst* averages.
- **The default `policy_raw.pt` plays *raw* (pure argmax, no rail) and is 100%.** It uses the
  **170-d** features (`CAND_DIM_AUG`: 156-d marginals + `log C` + count-bucket + remaining-
  guesses one-hot) and was DAgger'd on the hybrid expert, so it internalized `C ≤ R` and probes
  on its own. `ModelPolicy`/`evaluate_model` auto-detect aug from `config.cand_dim` and feed
  `remaining`. The **hybrid rail still exists** as a free safety net (`--safe`, or `--rl` for the
  Part-3 policy) — probing and masking are mutually exclusive (a probe is a non-candidate, so a
  candidate mask forbids it), so the rail keeps the mask while `len(candidates) ≤ guesses_left`
  and lifts it when stuck. It's just no longer *required*.
- **Don't normalize away a feature a downstream rule needs.** The 156-d marginals are fractions,
  so they hid the candidate *count* — the very thing the rail tests. That's why supervised
  probing failed for ages; the fix was adding C/R to the input, not a bigger net. When a net
  "can't learn" a rule, check its inputs first.
- **RL trains only the stuck decision; the BC anchor protects the rest.** All policy
  gradient lands on stuck states; a frozen KL anchor on a sample of *normal* (non-stuck)
  states keeps unmasked play from rotting. Drop the anchor and raw play collapses (94.6%)
  while hybrid still looks fine — so **watch the printed `raw %`**, not just hybrid.
- **Don't judge a model by val top-1 accuracy.** It diverges from game performance (an
  0.07M variant had the *highest* val acc and the *worst* play). **Do** rank models with
  `evaluate.py`, which plays all 2315 games.
- **Don't change `CAND_DIM`, the token scheme, or `MAX_LEN` without retraining.**
  Checkpoints store `PolicyConfig` and won't load against a changed encoding. **Do**
  retrain and write a new checkpoint.
- **Don't re-add `models/` to `.gitignore`.** Checkpoints are tracked so clones play
  immediately. **Do** keep `data/*.npy` and `data/*.npz` ignored — they're reproducible.
- **Pattern matrix is the hot path.** It's precomputed once and cached; entropy is a
  vectorized histogram over it (`solver.guess_entropies`). Don't loop per-pair in Python.
- **Pushing** goes to the `ihvo` GitHub account (not the default active `gh` account):
  `gh auth switch --user ihvo`, push, then switch back.

## Findings log (how we got here)

> Full narrative with numbers: the **[docs/](docs/)** research log (4 parts).

1. Transformer over `(guess, feedback)` history alone → **failed**, stuck at the
   random-consistent baseline (~4.06 avg). The candidate set, not the history, is the
   state. Adding `candidate_features` fixed it (masked 3.88).
2. Hard vs. soft distillation targets **trade** masked play against solo play. The
   blended target (above) plus richer features (per-position **and** letter-present
   frequencies) reached 3.62 on both.
3. Ablations: attention heads and encoder depth barely matter; **dropping the encoder
   entirely** (candidate-only MLP) costs nothing (1.04M → 0.34M). Halving `d_model`
   *does* hurt. The factored-letter head roughly halves params for ~+0.06 avg.
4. **Probing is an RL problem, not a labeling one.** BC of full-pool probe labels failed
   (masked ceiling ~99.7%; raw collapses). Post-training the *same net* with GRPO on the
   terminal win reward — teacher trajectories seed exploration on the trap states, a BC
   anchor pins normal play — closed it to **100%**. The decisive moves: fold the safety
   rail into the loop so gradient hits only the stuck decision; select on the hybrid metric
   you deploy; anchor the rest. The 99.7→100 gain *is* the ~7 neighbour-trap answers
   (`-ound`, `-atch`, `-aunt`, `-aste`, `-ight`).
5. **The rail can be internalized — raw play to 100% (no RL).** The rail tests `C ≤ R`, but
   the normalized 156-d features hide the count and the MLP the budget — the decision wasn't
   observable. Adding C/R (→170-d) and **DAgger**-ing the raw net on the hybrid expert took raw
   99.3→99.7→99.9→**100** over three rounds. Imitation sufficed once the decision was in the
   input; no reward needed. The forced opener stays.

The model is essentially a learned ranker over the candidate set; the solver does the exact
constraint propagation, and the net has learned to *probe* — under the rail (`policy_rl.pt`) or,
with C/R features, on its own (`policy_raw.pt`). See memory notes
`candidate-set-is-sufficient-statistic`, `rl-probing-reaches-100`, `rail-internalized-via-dagger`.

## Module map

| Module | Role |
|---|---|
| `words.py` | Word lists + compact vocabulary (letter-index arrays). |
| `feedback.py` | Green/yellow/gray with correct duplicate handling; scalar + vectorized. |
| `solver.py` | Pattern matrix, vectorized entropy selection, candidate filtering, game env, teacher. |
| `encoding.py` | `encode_state` + `candidate_features` (156-d; +C/R → 170-d `CAND_DIM_AUG`). |
| `model.py` | `WordlePolicy` (`use_history`, `factored_head` flags) + save/load. |
| `dataset.py` | Self-play → blended soft-target examples; `--state-aug` for 170-d features. |
| `train.py` | Soft-cross-entropy (BC) loop; `--init` warm-start; infers `cand_dim` from data. |
| `rl.py` | **RL post-training:** GRPO + teacher demos + BC anchor → `policy_rl.pt` (hybrid 100%). |
| `dagger.py` | **DAgger:** imitate the hybrid expert with C/R features → `policy_raw.pt` (raw 100%). |
| `evaluate.py` | Batched full-vocab play; `probe_when_stuck` gives the hybrid; default `policy_raw.pt`. |
| `policy.py` | Checkpoint → `GameState → guess` wrapper; auto-aug; `probe_when_stuck` rail. |
| `cli.py` | Interactive solver (default = raw `policy_raw.pt`; `--rl`/`--safe`/`--bc`/`--mlp`). |

## Next steps

- **Raw avg 3.461 → 3.45:** win rate is maxed; trim guesses (a short RL/DAgger tail, or a
  speed-weighted target).
- **Full allowed-guess action space (~10.6k):** richer probes than the answer list allows;
  needs a new output head (breaks current checkpoints) and more exploration.
- **Shrink further:** the candidate-only MLP's params are ~90% output head — a factored
  head gets toward ~0.1M, but stacking cuts past that broke play; measure each step.
