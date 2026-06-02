# wordle-guesser

A Wordle solver that tells you what to guess next — backed by a small PyTorch model,
trained to play the game, that reaches the information-theoretic optimum.

Two brains cooperate:
- An **entropy solver** picks the guess that reveals the most information (the classic, strong baseline).
- A **trained neural net** learns to imitate it and identify the most probable answer.

## Quick start

```bash
uv sync                 # Python 3.12 venv + torch/numpy/tqdm
uv run wordle-guesser   # it suggests a word; you type the colors Wordle gave back
```

Feedback is 5 letters — `g` = green, `y` = yellow, `x` = gray (e.g. `xxgxx`):

```
Turn 1 — suggested: RAISE
feedback: xxgxx
Turn 2 — 51 candidate(s) — suggested: GLINT
feedback: xygxx
Turn 3 — 6 candidate(s) — suggested: CHILD
...
```

Handy flags: `--mlp` (the smaller model), `--teacher` (pure solver, no model), `--no-mask`.

## How well does it play?

Over all 2315 answers: **99.4% solved, 3.62 guesses on average** — essentially matching
the entropy solver (3.50 avg), and it plays just as well with *no* solver help at all.

There are two interchangeable trained models, identical in play:

| Model | Size | File |
|---|---:|---|
| Transformer (default) | 1.04M params | `models/policy.pt` |
| MLP (`--mlp`) | 0.34M params | `models/policy_mlp.pt` |

## An honest note

Wordle is really an information-theory problem, not a language one — so the model can
at best *approximate* the entropy solver, and it does. The interesting result is **how
small** a model reaches the optimum: the transformer's attention turned out to be
unnecessary, and a plain 3-layer MLP plays identically.

## More

Play / train / test commands, the design story, and how to extend it live in
**[AGENTS.md](AGENTS.md)**.
