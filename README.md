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

Handy flags: `--rl` (the hybrid-rail policy), `--safe` (rail overlay, guaranteed-valid
suggestions), `--bc` (behavior-cloned net), `--mlp` (smaller model), `--teacher` (pure solver).

## How well does it play?

Over all 2315 answers:

| Policy | Win | Avg guesses | Losses |
|---|---|---|---|
| **Default (raw)** | **100.0%** | **3.46** | **0** |
| RL + hybrid rail (`--rl`) | 100.0% | 3.48 | 0 |
| Behavior-cloned (`--bc`) | 99.7% | 3.55 | 7 |
| Entropy solver (`--teacher`) | 100.0% | 3.50 | 0 |

The default is a **1.04M-param net** that **probes on its own** — playing pure `argmax` over
the whole word list (no candidate mask, no rules), it solves every answer. Probing means
spending a guess on a *non-candidate* word that splits a cluster of look-alikes
(`bound/found/hound/…`, `watch/match/…`) — the only way to crack those traps, and something
behavior cloning could never learn. We first taught it with a safety rail (still available
via `--rl`/`--safe`), then **folded that rail into the network** by giving it the two features
the rule needs (how many candidates remain, how many guesses are left) and imitating the
expert with DAgger.

All policies open **SLATE** — a fixed first move forced at play time, *not* learned by the
net (the opening is one deterministic state, so it's a chosen constant).

## An honest note

Wordle is an information-theory problem, not a language one. The interesting results are
**how small** a net reaches the optimum (attention turned out unnecessary — a 3-layer MLP
matches the transformer), and that the last 0.3% — *probing* — couldn't be learned from
supervised labels (a probe's value is a multi-step setup). It took reinforcement learning to
discover it, then a representation fix (exposing the candidate count + remaining guesses) to
let the network internalize the rule and play raw, with no inference-time rail. The solver
still tracks the candidate set to build the net's input — that dependency stays; the rail is
just no longer required.

## More

Play / train / test commands, the design story, and how to extend it live in
**[AGENTS.md](AGENTS.md)**. The full build is a three-part research log under `docs/`:

1. **[Entropy teacher & cloned policies](docs/2026-06-01-entropy-teacher-and-cloned-policies.md)** — the baseline, the BC transformer, and the perceptron that matched it.
2. **[Probing vs. openers](docs/2026-06-02-probing-and-openers.md)** — diagnosing the gap, the failed relabel, the SLATE opener.
3. **[RL post-training to 100%](docs/2026-06-02-rl-post-training-to-100.md)** — teaching the net to probe with RL.
4. **[Deprecating the rail](docs/2026-06-03-deprecating-the-rail.md)** — C/R features + DAgger fold the rail into the net; raw play hits 100%.

Prefer it interactive? Open **[`site/index.html`](site/index.html)** in any browser — a
beginner-friendly walkthrough of all five stages with live demos (the feedback engine, the
shrinking candidate list, the probing trap) and a playable agent that solves any word in
front of you. No build step; the real answer list runs in the page.
