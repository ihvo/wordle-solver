# AGENTS.md — wordle-guesser

Guidance for agents working in this repo. Keep this file current as the project moves.

## Goal

Build a Wordle helper whose core is a **PyTorch-trained model that identifies the most
probable next guess**. Training a neural net with PyTorch is a hard requirement (not an
optional add-on). An entropy-maximizing solver serves as the teacher, baseline, and
exact constraint engine the model rides on.

## Status (2026-06-02)

- **Done:** vectorized feedback engine; entropy teacher (100% win, 3.50 avg over all
  2315 answers); transformer policy distilled from it (**99.7% win, 3.55 avg, 7 losses**);
  candidate-only **MLP** alternative (0.34M params, **3.57 avg**); both seeded from and
  forced to open **SLATE** (see Conventions); interactive CLI; 23 passing tests. Pushed to
  `github.com/ihvo/wordle-solver` (private). Checkpoints are committed under `models/`.
- **Tried and rejected:** non-candidate *probing* distillation (full-pool labels) — a
  small net **can't learn to pick probes**; both the 0.34M and 1.04M nets got *worse*
  (masked ~4.0, unmasked 30–59% win). The 3.50→3.42 gap needs lookahead/search, not a
  relabel. The opener change (RAISE→SLATE) banked ~0.06 instead.
- **Not done (see Next steps):** the full ~10.6k allowed-guess action space; shrinking
  the head (factored / width sweep) at iso-quality.

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
2. `uv run python -m wordle_guesser.train --opener slate --epochs 24` → transformer to `models/policy.pt`
   - MLP variant: add `--no-history --out models/policy_mlp.pt`
   - the opener needs ~24 epochs to land cleanly; 12 undertrains the post-SLATE states
3. `uv run python -m wordle_guesser.evaluate [--model PATH]` → win-rate / avg-guesses vs teacher
4. `uv run pytest` before committing

### Play / debug a single game
`uv run wordle-guesser [--mlp|--teacher|--no-mask|--model PATH]`. The pattern matrix
(`data/patterns_answers.npy`) rebuilds automatically on first run (~1s).

## Key decisions

**Which policy to run:**

| Need | Use |
|---|---|
| Smallest model, same quality | MLP — `--mlp` (`use_history=False`) |
| The transformer / "sequence model" | default `models/policy.pt` |
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

1. Transformer over `(guess, feedback)` history alone → **failed**, stuck at the
   random-consistent baseline (~4.06 avg). The candidate set, not the history, is the
   state. Adding `candidate_features` fixed it (masked 3.88).
2. Hard vs. soft distillation targets **trade** masked play against solo play. The
   blended target (above) plus richer features (per-position **and** letter-present
   frequencies) reached 3.62 on both.
3. Ablations: attention heads and encoder depth barely matter; **dropping the encoder
   entirely** (candidate-only MLP) costs nothing (1.04M → 0.34M). Halving `d_model`
   *does* hurt. The factored-letter head roughly halves params for ~+0.06 avg.

The model is essentially a learned ranker over the candidate set; the solver does the
exact constraint propagation. See the project memory note `candidate-set-is-sufficient-statistic`.

## Module map

| Module | Role |
|---|---|
| `words.py` | Word lists + compact vocabulary (letter-index arrays). |
| `feedback.py` | Green/yellow/gray with correct duplicate handling; scalar + vectorized. |
| `solver.py` | Pattern matrix, vectorized entropy selection, candidate filtering, game env, teacher. |
| `encoding.py` | History tokenization (`encode_state`) + candidate features (`candidate_features`). |
| `model.py` | `WordlePolicy` (`use_history`, `factored_head` flags) + save/load. |
| `dataset.py` | Self-play + DAgger exploration → blended soft-target examples. |
| `train.py` | Soft-cross-entropy training loop. |
| `evaluate.py` | Batched full-vocab play, model vs teacher. |
| `policy.py` | Checkpoint → `GameState → guess` inference wrapper. |
| `cli.py` | Interactive solver. |

## Next steps

- **Probing toward 3.50:** let the action space include non-candidate words to extract
  information. Drops the candidate mask; this was the original failure mode, so expect to
  need explicit validity learning. Bump output vocab to the allowed-guess list.
- **Shrink further:** the candidate-only MLP's params are ~90% output head — a factored
  head gets toward ~0.1M, but stacking cuts past that broke play; measure each step.
