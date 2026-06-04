# The word-seed decoder: a generative transformer to 95% from tokens

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
a softmax blend. That took it **51% → 93.9% at BC**, and **95.08% with gentle RL** (avg 3.54). Not
the 99% goal — but a generative transformer (history-in, characters-out) within five points of the
selection-based classifier, up from a line that looked capped at 51%.

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

| stage | win | avg | invalid |
|---|---|---|---|
| Part-7 generative-from-tokens ceiling (hard-mask) | 51% | 3.56 | 20% |
| word-seed, **soft** blend (BC → RL) | 70% → 87% | 3.56 | ~5% |
| word-seed, **hard** straight-through (BC) | 93.9% | 3.54 | 2.3% |
| word-seed, hard **+ gentle RL** (shipped) | **95.08%** | **3.54** | 1.8% |

`models/policy_word_seed.pt`. The **soft → hard** seed is the single biggest jump (70 → 94 at BC):
spelling one clean word is far easier than decoding a blended embedding of several candidates.

## What broke the 51% conclusion

Part 7 was right that a *per-slot marginal* can't resolve which specific word to play, and that soft
generation leaks non-words. It was wrong that this caps a generator at ~51%. The miss: a generator
doesn't have to reconstruct candidacy from a per-slot statistic — it can **attend the vocabulary
directly** (the word head) and pick a word, exactly like the classifier, then merely *transcribe*
it. The character-output constraint never required throwing away per-word reasoning; it only
required the final emission to be letters.

## Honest notes

- **95.08%, not 99%.** The avg target (<3.6) is met; the win target is not.
- **The remaining gap (95→99) is candidacy + probing, not structure.** The word head plateaus ~95%
  (vs the dedicated cross-attn classifier's 100%) because its encoder is shared with a spelling
  decoder; and RL through the **straight-through** seed gives noisy gradients, so it *oscillates*
  (93.9 → 95.1 → 92.6) rather than converging.
- **What didn't close it:** naive RL (regressed the confident policy), a frozen 100%-classifier head
  + fresh decoder (the frozen encoder starves the decoder's cross-attention), warm-start + joint
  train (fresh decoder relearns spelling too slowly), and KD from the classifier (the head↔decoder
  coupling through the embeddings destabilizes it — improving the head moves the seeds and breaks the
  speller). Each is a real lever mis-tuned, not a wall.
- **The honest comparison:** a plain cross-attention **classifier hits 100% / 3.46** from the same
  tokens — it *selects* rather than *generates*. The generative/character-output constraint is what
  costs the last few points.

## Lessons

1. **A "structural ceiling" is a hypothesis, not a result** — retest it when the framing changes.
   The 51% wall fell the moment the generator was allowed per-word attention.
2. **Make the hard thing easy to spell.** Straight-through hard selection (one clean word) beat a
   soft blend by 24 points at BC — the decoder's job became transcription, not disambiguation.
3. **Reaching 99% generatively is an infrastructure problem, not an architecture one.** It needs RL
   built for the straight-through path, or a clean two-stage distillation that decouples
   head-improvement from the speller — a deliberate effort, not more blind runs.

Shipped as a research variant (`models/policy_word_seed.pt`, eval via `decoder.eval_decoder`); the
classifier / `--xattn` nets remain the production policies. Infra: `model.py`
`decoder_word_seed`/`decoder_word_seed_hard`/`decoder_hard_mask`/`decoder_constraints` +
`decoder.py` (`--word-seed[-hard]`/`--hard-mask`/`--constraints`/`--init`/`--freeze-encoder`/
`--init-lr-scale`/`--kd-teacher`).
