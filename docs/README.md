# Research log

How wordle-guesser went from an entropy solver to a 100%-win neural policy — and then dropped the
solver and the vocabulary entirely. Eight parts, read in order:

1. **[Entropy teacher & cloned policies](2026-06-01-entropy-teacher-and-cloned-policies.md)**
   — the origin: a 3.50 solver, behavior-cloned into a 1.04M transformer and a 0.34M
   perceptron (99.4% / 3.62). The candidate set is the sufficient statistic; attention is
   unnecessary.
2. **[Probing vs. openers](2026-06-02-probing-and-openers.md)** — diagnosing the gap to the
   solver: probing can't be distilled by relabeling, but a forced **SLATE** opener banks
   3.62 → 3.55 for free.
3. **[RL post-training to 100%](2026-06-02-rl-post-training-to-100.md)** — teaching the net
   to *probe* with RL on the terminal win → **100% / 0 losses / 3.481 avg** (under the hybrid rail).
4. **[Deprecating the rail](2026-06-03-deprecating-the-rail.md)** — fold the inference-time rail
   into the weights with count/remaining features + DAgger; the net plays raw at **100%**.
5. **[A solver-free net](2026-06-03-solver-free-net.md)** — drop the solver's candidate tracking:
   a 0.5M cross-attention net plays **100% from the tokens alone** and even learns its own opener.
6. **[The generative decoder](2026-06-03-generative-decoder.md)** — the next generation: drop the
   2,315-way classifier and *write* the word letter-by-letter with no vocabulary. Marginal
   conditioning + RL takes it from 33% to **99.44% / 0.08% non-words**.
7. **[Generation from tokens alone](2026-06-03-generative-from-tokens.md)** — the hard combination:
   no candidate features *and* no vocabulary. Cross-attn → learned marginal → word-LM climbs
   11% → **43%** and *looks* capped — a generator can't do per-word candidacy via a per-slot marginal.
8. **[The word-seed decoder](2026-06-04-word-seed-decoder.md)** — break that ceiling: give the
   generator per-word candidacy (a cross-attention word head) and have it *spell* the selected word
   as characters. A straight-through hard seed takes a generative, history-in/characters-out
   transformer to **95.08% / 3.54** — within five points of the selection classifier.

Operational guide (how to build/train/run): [`../AGENTS.md`](../AGENTS.md).
