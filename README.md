# wordle-guesser

A Wordle solver with two cooperating brains:

1. An **entropy-maximizing teacher** — at each turn it picks the guess that
   maximizes the expected information (Shannon entropy) over the words still
   consistent with the feedback so far. This is the strong, classic,
   information-theoretic baseline. No training involved.
2. A **PyTorch transformer policy** — a small sequence model that reads the game
   state and predicts the next guess. It is trained (behavior cloning /
   distillation) to imitate the teacher, and is measured against it.

## An honest note on the design

Wordle is fundamentally an *information-theory* problem, not a language problem:
the optimal strategy is to maximize expected information gain over the candidate
set. A neural net can, at best, *approximate* that. So the teacher is the strong
solver; the transformer is the "learn to play it" exercise the project is built
around.

Three findings, in the order they bit us:

1. **The sufficient statistic for Wordle is the set of remaining candidate
   words, not the move history.** A transformer fed only the `(guess, feedback)`
   history learned nothing — it sat exactly at the "guess a random still-valid
   word" baseline, because it would have had to re-derive the candidate set *and*
   emulate entropy maximization from scratch. Feeding it a compact summary of the
   candidate set (per-position letter frequencies + letter-present frequencies,
   156-d) made the decision learnable.
2. **Hard vs. soft targets trade masked play against solo play.** A one-hot
   "best candidate" target sharpens ranking (good when masked to candidates) but
   lets the model dump leftover probability on invalid words (bad when playing
   solo). A soft entropy distribution does the opposite.
3. **A blended target gets both.** Putting most of the mass as a spike on the
   single best candidate, and the rest as a soft tail over *other candidates*,
   keeps ranking sharp **and** keeps the model's probability on valid words. With
   that, solo play essentially matches masked play.

## Results (full 2315-word answer set)

| Policy | Win rate | Avg guesses |
|---|---:|---:|
| Teacher (entropy, full pool) | 100.0% | 3.50 |
| Entropy over candidates (masked-play ceiling) | 99.5% | 3.58 |
| **Transformer, masked to candidates** | **99.4%** | **3.62** |
| **Transformer, raw (no solver help, plays solo)** | **99.3%** | **3.62** |
| *(baseline: random consistent guess)* | 97.9% | 4.06 |

"Masked" means the model only ranks words still consistent with the feedback —
it never suggests an impossible word. "Raw" lets the model pick any word in the
vocabulary entirely on its own. The trained model reaches the candidate-entropy
ceiling either way: it has effectively learned to *be* the entropy solver, and to
know which words are still valid without being told.

## Setup

```bash
uv sync          # Python 3.12 venv + torch, numpy, tqdm
```

The official word lists live in `data/` (answers + allowed guesses).

## Two trained models

Ablations showed Wordle doesn't need a sequence model — the candidate set is the
sufficient statistic, and the solver already computes it. So there are two
interchangeable policies (same `WordlePolicy` class, the `use_history` flag
selects between them), and you can use either:

| Model | Architecture | Params | File | Masked avg / win |
|---|---|---:|---|---:|
| **Transformer** (default) | history encoder + candidate-feature MLP | 1.04M | `models/policy.pt` | 3.61 / 99.4% |
| **MLP** (`--mlp`) | candidate-feature MLP only (no attention) | 0.34M | `models/policy_mlp.pt` | 3.61 / 99.4% |

They play identically; the MLP is 3× smaller and is the honest "minimum that
fits the problem." See `references` in the commit history for the full ablation.

## Usage

Play interactively (the app). It suggests a guess; you type the feedback Wordle
gave you (`g`=green, `y`=yellow, `x`=gray), and it suggests the next one:

```bash
uv run wordle-guesser                 # transformer (default)
uv run wordle-guesser --mlp           # the smaller candidate-only MLP
uv run wordle-guesser --teacher       # the entropy solver, no model
uv run wordle-guesser --no-mask       # let the model rank ALL words, not just valid ones
uv run wordle-guesser --model PATH    # any checkpoint
```

Reproduce the models from scratch:

```bash
uv run python -m wordle_guesser.dataset                                    # self-play -> data/bc_dataset.npz
uv run python -m wordle_guesser.train                                      # transformer -> models/policy.pt
uv run python -m wordle_guesser.train --no-history --out models/policy_mlp.pt   # MLP alternative
uv run python -m wordle_guesser.evaluate --model models/policy_mlp.pt      # model vs teacher over all answers
```

Run the tests:

```bash
uv run pytest
```

## How it fits together

| Module | Role |
|---|---|
| `words.py` | Word-list loading and the compact vocabulary (letter-index arrays). |
| `feedback.py` | Wordle feedback (green/yellow/gray) with correct duplicate handling; scalar + vectorized. |
| `solver.py` | Pattern-matrix precompute, vectorized entropy guess selection, candidate filtering, the game environment, and the entropy teacher. |
| `encoding.py` | Tokenizes the `(guess, feedback)` history and builds the candidate-set feature vector. |
| `model.py` | The transformer policy: history encoder + candidate-feature path → softmax over words. |
| `dataset.py` | Self-play with the teacher (plus DAgger-style exploration) to produce `(state → best candidate)` examples. |
| `train.py` | Behavior-cloning training loop. |
| `evaluate.py` | Batched full-vocabulary evaluation, model vs teacher. |
| `policy.py` | Inference wrapper turning a checkpoint into a `GameState → guess` policy. |
| `cli.py` | The interactive solver. |

## Scope / limitations

- v1 uses the official answer list (~2315 words) as both the candidate set and
  the model's output vocabulary. The ~10.6k allowed-guess list is bundled for a
  future "full-probe" extension.
- The model is at the masked-play ceiling (~3.58, entropy over candidates only).
  Closing the last 0.12 to the teacher's 3.50 requires *probing* with
  non-candidate words — a deliberate next step, not a v1 goal.
