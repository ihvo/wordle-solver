# Research log

How wordle-guesser went from an entropy solver to a 100%-win neural policy, in three parts
(read in order):

1. **[Entropy teacher & cloned policies](2026-06-01-entropy-teacher-and-cloned-policies.md)**
   — the origin: a 3.50 solver, behavior-cloned into a 1.04M transformer and a 0.34M
   perceptron (99.4% / 3.62). The candidate set is the sufficient statistic; attention is
   unnecessary.
2. **[Probing vs. openers](2026-06-02-probing-and-openers.md)** — diagnosing the gap to the
   solver: probing can't be distilled by relabeling, but a forced **SLATE** opener banks
   3.62 → 3.55 for free.
3. **[RL post-training to 100%](2026-06-02-rl-post-training-to-100.md)** — teaching the net
   to *probe* with RL on the terminal win → **100% / 0 losses / 3.481 avg**.

Operational guide (how to build/train/run): [`../AGENTS.md`](../AGENTS.md).
