# AGENTS.md — wordle-guesser

Guidance for agents working in this repo. Keep this file current as the project moves.

## Goal

Build a Wordle helper whose core is a **PyTorch-trained model that identifies the most
probable next guess**. Training a neural net with PyTorch is a hard requirement (not an
optional add-on). An entropy-maximizing solver serves as the teacher, baseline, and
exact constraint engine the model rides on.

## Status (2026-06-02)

- **Done:** vectorized feedback engine; entropy teacher (100% win, 3.50 avg over all
  2315 answers); transformer policy behavior-cloned from it (99.7% win, 3.55 avg masked);
  candidate-only **MLP** alternative (0.34M params); all seeded from and forced to open
  **SLATE** (see Conventions); interactive CLI; **27 passing tests**. Checkpoints committed
  under `models/`.
- **RL post-training → 100% (the headline).** `rl.py` post-trains the BC transformer with
  GRPO + teacher-demo exploration to learn *probing*. Played **hybrid** (mask while
  candidates fit the budget, probe when stuck — see Conventions) the RL checkpoint
  `models/policy_rl.pt` is **100% win, 3.481 avg, 0 losses** over all 2315 — matching the
  full-pool teacher's win rate. This is the new CLI/evaluate default. Raw (unmasked) play
  stays healthy at 99.3% thanks to a BC anchor; masked-only is 99.6%.
- **The earlier "probing can't be distilled" result still holds — and is *why* RL works.**
  Behavior cloning of full-pool probe labels failed (the net can't *rank* a probe whose
  value is a multi-step setup). RL succeeds because it optimises the terminal win directly:
  a probe's payoff is a return, not a per-state label. See Findings log.
- **Not done (see Next steps):** push hybrid avg 3.481 → teacher's 3.45; the full ~10.6k
  allowed-guess action space; shrink the head at iso-quality.

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
3. **RL post-train → 100%:** `uv run python -m wordle_guesser.rl` warm-starts from
   `models/policy.pt`, plays unmasked under the safety rail, and saves the best hybrid
   checkpoint to `models/policy_rl.pt` (~700 updates, a few minutes on MPS). Selects on the
   hybrid win-rate it deploys with; watch the printed `raw %` to confirm the anchor holds.
4. `uv run python -m wordle_guesser.evaluate [--model PATH]` → teacher / **hybrid** / masked / raw
5. `uv run pytest` before committing

### Play / debug a single game
`uv run wordle-guesser [--bc|--mlp|--teacher|--no-mask|--model PATH]`. Default is the RL
policy played hybrid. The pattern matrix (`data/patterns_answers.npy`) rebuilds on first run.

## Key decisions

**Which policy to run:**

| Need | Use |
|---|---|
| Best play (100%, default) | RL + hybrid — `models/policy_rl.pt` (no flag) |
| The behavior-cloned transformer | `--bc` (`models/policy.pt`, masked) |
| Smallest model | MLP — `--mlp` (`use_history=False`, masked) |
| No model, pure baseline | `--teacher` |
| See the model's unaided picks | `--no-mask` |

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
- **The hybrid rail is the deploy policy — probing and masking are mutually exclusive.**
  A probe is a *non-candidate* word, so the candidate mask forbids it. The rule
  (`ModelPolicy(probe_when_stuck=True)`, `evaluate_model(probe_when_stuck=True)`): keep the
  mask while `len(candidates) ≤ guesses_left` (enumerating can't lose), lift it once they
  outnumber the budget so the net can probe. This is exact and free, and only the RL net
  was *trained* to probe — `--bc`/`--mlp` play masked (their best mode).
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

> Full narrative with numbers: the **[docs/](docs/)** research log (3 parts).

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

The model is essentially a learned ranker over the candidate set; the solver does the exact
constraint propagation, and on the stuck states the RL net has learned to *probe*. See the
memory notes `candidate-set-is-sufficient-statistic` and `rl-probing-reaches-100`.

## Module map

| Module | Role |
|---|---|
| `words.py` | Word lists + compact vocabulary (letter-index arrays). |
| `feedback.py` | Green/yellow/gray with correct duplicate handling; scalar + vectorized. |
| `solver.py` | Pattern matrix, vectorized entropy selection, candidate filtering, game env, teacher. |
| `encoding.py` | History tokenization (`encode_state`) + candidate features (`candidate_features`). |
| `model.py` | `WordlePolicy` (`use_history`, `factored_head` flags) + save/load. |
| `dataset.py` | Self-play + DAgger exploration → blended soft-target examples. |
| `train.py` | Soft-cross-entropy (BC) training loop. |
| `rl.py` | **RL post-training:** GRPO + teacher demos + BC anchor → `policy_rl.pt` (probing → 100%). |
| `evaluate.py` | Batched full-vocab play; `probe_when_stuck` gives the hybrid policy. |
| `policy.py` | Checkpoint → `GameState → guess` wrapper; `probe_when_stuck` for the hybrid rail. |
| `cli.py` | Interactive solver (default = RL + hybrid). |

## Next steps

- **Hybrid avg 3.481 → 3.45:** the win rate is maxed; trim guesses. Reward already favours
  speed — try a longer/lower-LR RL tail, or shape lightly once 100% is locked.
- **Full allowed-guess action space (~10.6k):** richer probes than the answer list allows;
  needs a new output head (breaks current checkpoints) and more exploration.
- **Shrink further:** the candidate-only MLP's params are ~90% output head — a factored
  head gets toward ~0.1M, but stacking cuts past that broke play; measure each step.
