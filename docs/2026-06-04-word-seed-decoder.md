# The word-seed decoder: a generative transformer to 97% from tokens (and a 99.4% decision head)

*Research log — 2026-06-04. Author: Ihar + Clawd. Part 8 of 8. Previously:
[entropy-teacher-and-cloned-policies](2026-06-01-entropy-teacher-and-cloned-policies.md) →
[probing-and-openers](2026-06-02-probing-and-openers.md) →
[rl-post-training-to-100](2026-06-02-rl-post-training-to-100.md) →
[deprecating-the-rail](2026-06-03-deprecating-the-rail.md) →
[solver-free-net](2026-06-03-solver-free-net.md) →
[generative-decoder](2026-06-03-generative-decoder.md) →
[generative-from-tokens](2026-06-03-generative-from-tokens.md).*

## TL;DR

The brief: a **transformer** that takes the **game history** as input and **emits the next word
as output characters** — target 99% win / <3.6 avg. Part 7 left a generative, tokens-only head
stuck at **51%** with a conclusion that looked structural ("per-slot conditioning can't do per-word
candidacy"). That conclusion was wrong about the ceiling. The fix: give the generator **per-word
candidacy** the same way the 100% classifier does — a **cross-attention word head** scores all
2,315 words — then **spell the selected word, letter by letter, as output characters.** The unlock
was a **straight-through hard selection**: seed the decoder with *one* word's embedding (clean), not
a softmax blend. That took it **51% → 93.9% at BC → 97.28% / avg 3.55** (history-in, characters-out).

The sharpest finding came from probing *why* it wasn't higher: the **word head already decides at
99.44%** — its argmax-word play is essentially the 100% classifier. The whole remaining gap is
**character transcription**, not candidacy. A hard consistency-mask we'd added (zero feedback-
inconsistent letters) was doing double duty: it *blocked spelling the probe words that win the
neighbour-traps* **and** became a crutch the decoder leaned on. Removing it and retraining an
**unaided speller** (dense BC) lifted the spelled win 95.08 → **97.28%**, with the *decision* head at
**99.44%** — so the last ~2 points are pure spelling fidelity, not strategy.

## The architecture

```
history tokens ─▶ Transformer encoder ─▶ cross-attention word head ─▶ word logits (2,315)
                          │                                              │  argmax (straight-through)
                          │                                              ▼
                          │                                     selected word embedding  (the "seed")
                          ▼                                              │
                   (decoder cross-attn)  ───────────────────────────────┤
                                                                         ▼
                                          GRU decoder ×5  ─▶  out (26) ─▶ hard consistency-mask ─▶ characters
```

- **Encoder:** the proven history front-end — `token = letter×3 + color`, learned token+positional
  embeddings, 3-layer Transformer (d=128, 4 heads). Reads *only* the tokens.
- **Word head (`CrossAttnWordHead`):** each of the 2,315 words (embedded from its letters)
  cross-attends the encoded history → a per-word logit. This is the classifier's candidacy
  mechanism — what a per-slot marginal could never do.
- **Straight-through seed:** `seed = embedding[argmax(word_logits)]` on the forward pass (a *single*
  clean word), with the softmax gradient on the backward pass. The decoder spells one real word, not
  a blend of candidates.
- **Decoder:** a GRU seeded by `state_proj(seed)`, optionally cross-attending the history, emitting
  5 letters; a token-derived **hard consistency-mask** zeroes feedback-inconsistent letters.
- **Training:** spelling CE (the decoder) + a word-classification aux (the head learns candidacy),
  then gentle KL-anchored RL. 0.69M params, history-only (candidate features provably ignored).

## The ladder

| stage | spelled win | avg | invalid | head (decision) |
|---|---|---|---|---|
| Part-7 generative-from-tokens ceiling (hard-mask) | 51% | 3.56 | 20% | — |
| word-seed, **soft** blend (BC → RL) | 70% → 87% | 3.56 | ~5% | — |
| word-seed, **hard** straight-through (BC) | 93.9% | 3.54 | 2.3% | — |
| word-seed, hard + gentle RL (mask on) | 95.08% | 3.54 | 1.8% | **99.44%** |
| word-seed, **unaided speller, no mask** (shipped) | **97.28%** | **3.55** | 0.9% | **99.44%** |

`models/policy_word_seed.pt`. Two jumps: the **soft → hard** seed (70 → 94 at BC — spelling one clean
word beats decoding a blend), then **dropping the consistency-mask** and retraining an unaided speller
(95.08 → 97.28 — the mask had been blocking the probe words that win the traps).

## The diagnostic that reframed it: the head already decides at 99.44%

Trying to RL the win directly (route B: sample a *word* from the head, reward the outcome — a clean
categorical policy gradient, no straight-through) the very first eval printed the answer: the head's
**argmax-word play wins 99.44%**, while the spelled output won only 95.08%. So the model's *decision*
was already at target — the gap was entirely **character transcription**. And the spell loss exploded
(~10⁶): the hard consistency-mask zeroes the letters of *probe* words (non-candidates), but the head
*wins the neighbour-traps by probing* — so the mask made exactly those games unspellable, and was also
a crutch the decoder leaned on (turn it off cold and spelling collapses to 19%). The fix is to retrain
the speller **without** the mask so it can spell any pick — dense teacher-forced BC did this fast
(83% → 97% in ~12 epochs, far quicker than the rollout-sparse RL recovery).

## What broke the 51% conclusion

Part 7 was right that a *per-slot marginal* can't resolve which specific word to play, and that soft
generation leaks non-words. It was wrong that this caps a generator at ~51%. The miss: a generator
doesn't have to reconstruct candidacy from a per-slot statistic — it can **attend the vocabulary
directly** (the word head) and pick a word, exactly like the classifier, then merely *transcribe*
it. The character-output constraint never required throwing away per-word reasoning; it only
required the final emission to be letters.

## Honest notes

- **97.28% spelled, not 99% — but the *decision* head is 99.44%.** The avg target (<3.6) is met; the
  spelled win is ~2 points under the head because the decoder still mis-spells a fraction of picks
  (0.9% non-words + a few spelled-to-a-different-word). The strategy is at target; transcription isn't.
- **The remaining 2 points are spelling fidelity, not candidacy.** To close it: a stronger/larger
  speller, or a decodable-by-construction word embedding — a bounded transcription problem, since the
  head already plays at 99.44%.
- **What didn't close it (dead-ends worth recording):** naive RL (regressed the confident policy);
  RL through the straight-through seed (noisy, oscillates 93.9 → 95.1 → 92.6); a frozen 100%-classifier
  head + fresh decoder (the frozen encoder starves the decoder's cross-attention); KD from the
  classifier (the head↔decoder embedding coupling destabilizes it). The thing that *did* work was
  prosaic: take the mask off and let dense BC retrain the speller.
- **The honest comparison:** a plain cross-attention **classifier hits 100% / 3.46** from the same
  tokens — it *selects* rather than *generates*. The generative/character-output constraint costs the
  last ~2.7 points, and they live entirely in spelling.

## Lessons

1. **A "structural ceiling" is a hypothesis, not a result** — retest it when the framing changes.
   The 51% wall fell the moment the generator was allowed per-word attention.
2. **Make the hard thing easy to spell.** Straight-through hard selection (one clean word) beat a
   soft blend by 24 points at BC — the decoder's job became transcription, not disambiguation.
3. **Measure the sub-components.** Reading the head's argmax-word win (99.44%) separately from the
   spelled win (95→97%) is what revealed the real bottleneck — and that a "consistency aid" (the mask)
   was actively sabotaging the probe spelling it was meant to help. A crutch can cap you below where
   you'd be without it.

Shipped as a research variant (`models/policy_word_seed.pt`, eval via `decoder.eval_decoder`); the
classifier / `--xattn` nets remain the production policies. Infra: `model.py`
`decoder_word_seed`/`decoder_word_seed_hard`/`decoder_hard_mask`/`decoder_constraints` +
`head_logits`/`spell_logits` (detached speller path); `decoder.py`
(`--word-seed[-hard]`/`--hard-mask`/`--constraints`/`--init`/`--freeze-encoder`/`--init-lr-scale`/
`--kd-teacher`); `rl_word_seed.py` (route B — clean categorical RL over the word head + head-eval
diagnostic). The shipped 97.28% came from a clean BC re-run **without** the consistency-mask
(`--word-seed-hard`, no `--hard-mask`, warm-started head + differential lr).
