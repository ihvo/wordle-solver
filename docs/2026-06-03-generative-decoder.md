# The next generation: a generative decoder that *writes* the word

*Research log — 2026-06-03. Author: Ihar + Clawd. Part 6 of 7. Previously:
[entropy-teacher-and-cloned-policies](2026-06-01-entropy-teacher-and-cloned-policies.md) →
[probing-and-openers](2026-06-02-probing-and-openers.md) →
[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md) →
[deprecating-the-rail](2026-06-03-deprecating-the-rail.md) →
[solver-free-net](2026-06-03-solver-free-net.md). Next:
[generative-from-tokens](2026-06-03-generative-from-tokens.md).*

## TL;DR

Every policy so far is a **classifier**: it scores all 2,315 answer words and takes the argmax.
The action set is closed, so a wrong "word" is impossible by construction. This part asks the
opposite question — can a net **generate** the guess letter-by-letter, with *no vocabulary at
all*? It can emit any of 26⁵ ≈ 12M strings, so nothing structurally stops it from outputting a
non-word.

First answer (teacher-forced behavior cloning): **barely — 33% win, 41% non-words**. It looked
like the wrong tool. It wasn't. The failure was **exposure bias plus missing conditioning**, not
the generative head. Three changes turned it around:

1. **RL on its own rollouts** — the decoder finally sees its own prefixes → **71% / 12%**.
2. **Per-slot candidate-marginal conditioning** (the big lever) — feed each letter step the live
   per-position candidate frequencies, so it generates *toward* real in-set words.
3. **A valid-word reward bonus + more RL** → **99.44% win / 0.08% non-words / avg 3.55** (CPU
   greedy, 13 losses), a pure generative, no-constraint head **competitive with the classifiers**.

The catch is instructive: the conditioning that fixes validity also **suppresses probing**, so the
decoder caps right at the masked-play ceiling — the last ~0.5% is the neighbour-traps, not
non-words. Same wall as a masked classifier, reached from the opposite direction.

## 1. The architecture (0.59M params)

Same shared front-end as the classifiers, new head.

- **Front-end → state z (256-d).** History tokens (`token = letter×3 + color`, ≤26 positions)
  through learned token + positional embeddings and a **3-layer Transformer encoder** (d=128,
  4 heads, FF 256, ~397K params); the 156-d candidate features (130-d per-slot letter frequencies +
  26-d letter-present frequencies) through an MLP to 128-d. Concatenate → a 256-d state `z`. (Base
  156-d, *not* the 170-d `+C/R` augmentation the raw classifier uses.)
- **Decoder head (`LetterDecoder`, ~142K params).** A GRU that writes 5 letters autoregressively:
  - `state_proj` (256→128) turns `z` into the GRU's initial hidden state `h₀`.
  - At each slot `t`, a `GRUCell(128→128)` consumes two inputs, summed: the **previous letter** via
    `letter_emb` (`Embedding(27→128)`; 26 letters + a START symbol) and the **per-slot candidate
    marginal** via `marg_proj` (`Linear(26→128)`).
  - `out` (`Linear(128→26)`) emits that slot's letter distribution. Greedy/sampled letters feed
    back in as the next `prev letter`.

No trie, no masking, no allowed-word list anywhere in the loop — the only "vocabulary" pressure is
the marginal conditioning and the training signal.

## 2. Why BC alone fails: exposure bias

Teacher-forced BC trains each step on the *ground-truth* previous letters, so the decoder never
practises recovering from its own mistakes. At play time it conditions on its own (sometimes
wrong) prefixes and drifts off the word manifold — emitting plausible fakes like `birty`. Result:
**33% win, 41% non-words**. Letting it roll out under RL (reward the win, score an invalid word as
a loss) closes most of that gap to **71% / 12%** — confirming the head was fine; the *training
regime* was the problem.

## 3. The big lever: per-slot marginal conditioning

The decoder knows the history through `z`, but at the moment it writes slot `t` it has no direct
view of *which letters are still alive in that slot*. So we compute the **per-position candidate
marginals** — for each of the 5 slots, the frequency of each letter among the remaining candidates
— and feed slot `t`'s marginal into the GRU step via `marg_proj`. Now generation is biased toward
letters that actually occur in surviving words, slot by slot. This is the single change that moves
the needle most.

We ran a short-budget matrix to pick the strategy, best → worst:

| Variant | Short-budget win |
|---|---|
| **#4 marginal conditioning** | **~53%** |
| #5 valid-word reward bonus | ~33% |
| baseline (RL only) | ~30% |
| #2 LM-prior pretrain | ~26% (*hurt*) |
| #3 unlikelihood penalty @0.5 | destabilised RL, no gain |

Conditioning beat both the soft fix (an LM prior over real words) and the explicit penalty
(unlikelihood on invalid letters). Telling the generator *about the action set's structure* worked;
penalising it *after the fact* did not.

## 4. Closing it out with RL

Starting from the marginal-conditioned checkpoint (~97%), continue word-level GRPO with the
valid-word bonus (`--valid-bonus 0.3`, lr 1.5e-5):

```
upd  50  win 97.41%  invalid 0.88%
upd 200  win 98.27%  invalid 0.54%
upd 350  win 99.05%  invalid 0.24%
upd 500  win 99.27%  invalid 0.15%
upd 750  win 99.44%  invalid 0.08%   <- best
FINAL greedy  win 99.44%  avg 3.551  invalid 0.08%  losses 13  (CPU)
```

Win rate climbs and invalids fall together — exactly what you want: the policy isn't trading
validity for wins, it's learning both at once.

## 5. Beam search: a crutch the confident model doesn't need

Decoding greedily takes the argmax letter at each step; **beam search** keeps the top-W partial
words by cumulative log-prob and picks the best complete one (`generate_beam`, batched, W=8). It
**rescued the weak BC decoder** (greedy 35% → beam-8 45%) by escaping locally-greedy traps. But on
the trained decoder it was **redundant — even slightly worse** (greedy 97.15 vs beam 97.06): beam
maximises *word probability*, which isn't the same as *guess value*, and a confident model already
puts the good word first. So beam is a fix for an under-trained decoder, not a free win.

## 6. The ceiling, and the tension

The decoder plateaus at ~99.4% (13 losses), and almost all of those are **genuine
neighbour-traps** — clusters like `_IGHT` where several answers differ by one letter and you can't
afford to enumerate them in the turns left. The catch: the marginal conditioning that fixed
validity also **steers away from non-candidate words** — but a *probe* is exactly a non-candidate
word played to split a trap. So the same lever that solved validity **suppresses probing**, and the
decoder caps right at the masked-play ceiling — the same wall the masked classifier hits. A clean
100% from here needs probing back (or a valid-by-construction generator that can still probe).

## Honest notes

- **"Generation can't do closed-vocab" was wrong.** I said as much earlier in the project; it was
  exposure bias + missing conditioning, not the head. Condition a generator on the structure of the
  valid action set and it nearly matches selection.
- **Validity was never the hard part once conditioned** — 0.08% non-words. The hard part is the
  same as for every other policy: probing the neighbour-traps.
- **Margins are device-noisy**; all final numbers are CPU greedy (deterministic).
- **This is a research variant.** Evaluated via `decoder.eval_decoder`; not wired into the CLI. The
  shipped policies remain the classifiers / the solver-free `--xattn` net. Checkpoint:
  `models/policy_decoder.pt`.

## Lessons

1. **Blame the training regime before the architecture.** A head that scores 33% under teacher
   forcing and 99.4% under RL + conditioning was never the problem — the data distribution it
   trained on was.
2. **Condition on the structure of the action set, don't penalise violations.** Feeding the
   per-slot marginals beat both an LM prior and an explicit unlikelihood penalty.
3. **Different sub-problems, different fixes — and the same final wall.** Exposure bias → RL;
   validity → conditioning; the irreducible last 0.5% → probing, which conditioning fights. Every
   policy in this project, classifier or generator, ends up staring at the neighbour-traps.
