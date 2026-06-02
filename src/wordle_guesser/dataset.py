"""Generate behavior-cloning data by self-play with the entropy teacher.

For every game we record, at each turn, ``(encoded history -> teacher's optimal
guess)``. With probability ``epsilon`` (after the opening) the guess we actually
*play* is a random remaining candidate rather than the teacher's pick — a
DAgger-style branch. The recorded label is always the teacher's optimal action,
so the model learns the right move even from histories it wouldn't have chosen.
This is what lets the CLI cope when you guess a different word than it suggests.

Identical ``(state, target)`` pairs are deduplicated: the teacher is
deterministic, so each distinct state has one correct action and we only need to
cover states, not reweight by frequency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .encoding import MAX_LEN, candidate_features, encode_state
from .solver import MAX_GUESSES, best_guess, guess_entropies, load_pattern_matrix
from .words import DATA_DIR, load_vocabulary

TOPK = 32  # how many candidates to keep in each soft target


def _soft_target(p, candidates, temperature, alpha, k=TOPK):
    """Top-k blended label over candidates.

    weight = ``(1-alpha)`` on the single best (max-entropy) candidate
           + ``alpha`` * softmax(entropy / T) spread over candidates.

    The spike keeps masked-play ranking sharp (the model must clearly prefer the
    best candidate); the soft tail keeps the model's leftover probability on
    *other candidates* rather than invalid words, which is what makes unaided
    (raw) play work. Returns ``(global_indices, weights)`` sorted descending.
    """
    if len(candidates) == 1:
        return np.asarray([int(candidates[0])]), np.asarray([1.0], dtype=np.float32)
    ent = guess_entropies(p, candidates, pool=candidates)
    logits = ent / temperature
    soft = np.exp(logits - logits.max())
    soft /= soft.sum()
    w = alpha * soft
    w[int(np.argmax(ent))] += 1.0 - alpha
    if len(candidates) > k:
        sel = np.argpartition(w, -k)[-k:]
    else:
        sel = np.arange(len(candidates))
    idx = candidates[sel]
    weights = w[sel]
    weights = weights / weights.sum()
    order = np.argsort(-weights)
    return idx[order].astype(np.int64), weights[order].astype(np.float32)


def generate(
    vocab,
    p: np.ndarray,
    passes: int = 8,
    epsilon: float = 0.3,
    temperature: float = 0.3,
    alpha: float = 0.4,
    seed: int = 0,
    max_guesses: int = MAX_GUESSES,
):
    """Self-play data distilling the candidate-entropy ranking (soft targets).

    Each example is ``(history tokens, candidate features) -> soft distribution``
    over the top candidates, weighted by ``softmax(entropy / temperature)``.
    Deduped by state (the history tokens determine everything downstream).
    """
    rng = np.random.default_rng(seed)
    letters = vocab.letters
    n = len(vocab)
    all_words = np.arange(n)
    seen: set[tuple[int, ...]] = set()
    tokens: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    idx_rows: list[np.ndarray] = []
    w_rows: list[np.ndarray] = []

    def record(toks, candidates):
        key = tuple(toks.tolist())
        if key in seen:
            return
        seen.add(key)
        idx, w = _soft_target(p, candidates, temperature, alpha)
        row_i = np.full(TOPK, -1, dtype=np.int64)
        row_w = np.zeros(TOPK, dtype=np.float32)
        row_i[: len(idx)] = idx
        row_w[: len(w)] = w
        tokens.append(toks)
        feats.append(candidate_features(candidates, letters))
        idx_rows.append(row_i)
        w_rows.append(row_w)

    # The opening is constant across all games; record it once.
    opening = best_guess(p, all_words, pool=all_words)
    record(encode_state([], [], letters), all_words)

    for _ in range(passes):
        for answer in rng.permutation(n):
            answer = int(answer)
            code = int(p[opening, answer])
            guesses: list[int] = [opening]
            codes: list[int] = [code]
            candidates = all_words[p[opening, all_words] == code]

            for _turn in range(1, max_guesses):
                if len(candidates) == 0 or guesses[-1] == answer:
                    break
                record(encode_state(guesses, codes, letters), candidates)

                target = best_guess(p, candidates, pool=candidates)
                explore = len(candidates) > 1 and rng.random() < epsilon
                played = int(rng.choice(candidates)) if explore else target

                code = int(p[played, answer])
                guesses.append(played)
                codes.append(code)
                candidates = candidates[p[played, candidates] == code]
                if played == answer:
                    break

    padded = np.full((len(tokens), MAX_LEN), 79, dtype=np.uint8)  # 79 = TOK_PAD
    for i, t in enumerate(tokens):
        padded[i, : len(t)] = t
    return padded, np.stack(feats), np.stack(idx_rows), np.stack(w_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate behavior-cloning dataset.")
    ap.add_argument("--passes", type=int, default=16)
    ap.add_argument("--epsilon", type=float, default=0.35)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.4, help="soft-tail mass; 1-alpha spikes the best candidate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "bc_dataset.npz")
    args = ap.parse_args()

    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)

    print(f"generating: {args.passes} passes, eps={args.epsilon}, T={args.temperature}, alpha={args.alpha} ...")
    tokens, feats, tgt_idx, tgt_w = generate(
        vocab,
        p,
        passes=args.passes,
        epsilon=args.epsilon,
        temperature=args.temperature,
        alpha=args.alpha,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=tokens, feats=feats, tgt_idx=tgt_idx, tgt_w=tgt_w)
    print(f"wrote {len(tokens)} unique states to {args.out}")


if __name__ == "__main__":
    main()
