# wordle-guesser

A Wordle solver that tells you what to guess next — backed by a small PyTorch model,
post-trained with RL to **solve every one of the 2315 answers (100%)**.

Two brains cooperate:
- An **entropy solver** picks the guess that reveals the most information (the classic, strong baseline).
- A **trained neural net** learns to play — first by imitating the solver, then sharpened
  with reinforcement learning until it can *probe* its way out of the hardest endgames.

## Quick start

```bash
uv sync                 # Python 3.12 venv + torch/numpy/tqdm
uv run wordle-guesser   # it suggests a word; you type the colors Wordle gave back
```

Feedback is 5 letters — `g` = green, `y` = yellow, `x` = gray (e.g. `xxgxx`):

```
Turn 1 — suggested: SLATE
feedback: xxgxx
Turn 2 — 51 candidate(s) — suggested: GLINT
feedback: xygxx
Turn 3 — 6 candidate(s) — suggested: CHILD
...
```

Handy flags: `--bc` (the behavior-cloned net), `--mlp` (smaller model), `--teacher` (pure
solver, no model), `--no-mask`.

## How well does it play?

Over all 2315 answers:

| Policy | Win | Avg guesses | Losses |
|---|---|---|---|
| **RL (default)** | **100.0%** | **3.48** | **0** |
| Behavior-cloned (`--bc`) | 99.7% | 3.55 | 7 |
| Entropy solver (`--teacher`) | 100.0% | 3.50 | 0 |

The default is a **1.04M-param net post-trained with RL** to match the solver's perfect
win rate. It plays a *hybrid* policy: while the remaining candidates fit the guess budget
it just picks one (it can't lose by enumerating), but once they outnumber the budget it
**probes** — spends a guess on a non-candidate word that splits the survivors. That probe
is the only way to crack the neighbour-traps (`bound/found/hound/…`, `watch/match/…`), and
it's what behavior cloning alone could never learn.

All policies open **SLATE** — a fixed first move forced at play time, *not* learned by the
net (the opening is one deterministic state, so it's a chosen constant).

## An honest note

Wordle is an information-theory problem, not a language one. The interesting results are
**how small** a net reaches the optimum (attention turned out unnecessary — a 3-layer MLP
matches the transformer), and that the last 0.3% — the probing the solver does for free —
had to be earned with **RL on the terminal win**, because no supervised label can teach a
move whose entire value is a multi-step setup. The 100% leans on the solver's exact
candidate tracking (the rail); the net's own unaided raw play is ~99.3%.

## More

Play / train / test commands, the design story, and how to extend it live in
**[AGENTS.md](AGENTS.md)**. The full build is a three-part research log under `docs/`:

1. **[Entropy teacher & cloned policies](docs/2026-06-01-entropy-teacher-and-cloned-policies.md)** — the baseline, the BC transformer, and the perceptron that matched it.
2. **[Probing vs. openers](docs/2026-06-02-probing-and-openers.md)** — diagnosing the gap, the failed relabel, the SLATE opener.
3. **[RL post-training to 100%](docs/2026-06-02-rl-post-training-to-100.md)** — teaching the net to probe with RL.
