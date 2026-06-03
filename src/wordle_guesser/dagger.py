"""DAgger: train a RAW (unmasked) policy to imitate the hybrid expert.

The hybrid rail — mask to candidates while they fit the budget, probe when stuck —
reaches 100%, but it's an inference-time rule. To fold it *into* the network and
deprecate the rail, we imitate the expert's hard action at every state, using the
augmented candidate features (count + remaining guesses) so the net can actually
see the ``C <= R`` distinction the rail tests. The student then plays raw argmax.

Round 0 collects expert self-play (+ exploration). Later rounds pass ``--student``
to roll the current student out *raw* and label the states it actually visits with
the expert action (the DAgger aggregation that fixes its own mistakes). Output is a
train.py-compatible npz with hard one-hot targets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .encoding import MAX_LEN, TOK_PAD, candidate_features, encode_state, pad_batch
from .model import load_checkpoint
from .solver import MAX_GUESSES, best_guess, load_pattern_matrix
from .train import pick_device
from .words import DATA_DIR, load_vocabulary


def expert_action(p, candidates, remaining, all_words) -> int:
    """The hybrid rail over the entropy teacher: best candidate while they fit the
    budget, otherwise the max-entropy probe over the full pool."""
    if len(candidates) == 1:
        return int(candidates[0])
    if len(candidates) <= remaining:
        return best_guess(p, candidates, pool=candidates)
    return best_guess(p, candidates, pool=all_words)


@torch.no_grad()
def _student_raw(model, device, guesses, codes, candidates, letters, remaining) -> int:
    toks = encode_state(guesses, codes, letters)
    tokens, kpm = pad_batch([toks])
    feats = candidate_features(candidates, letters, remaining)[None]
    lg = model(
        torch.from_numpy(tokens).to(device),
        torch.from_numpy(kpm).to(device),
        torch.from_numpy(feats).to(device),
    )[0].float().cpu().numpy()
    return int(np.argmax(lg))  # raw / unmasked


def collect(vocab, p, opener_idx, passes, epsilon, seed, student=None, device="cpu",
            full_pool_explore=0.4, max_guesses=MAX_GUESSES):
    """Roll out games; record (history, aug features) -> expert hard action."""
    rng = np.random.default_rng(seed)
    letters = vocab.letters
    n = len(vocab)
    all_words = np.arange(n)
    seen: set[tuple[int, ...]] = set()
    tokens, feats, tgt = [], [], []

    def record(toks, candidates, n_prior):
        key = tuple(toks.tolist())
        if key in seen:
            return
        seen.add(key)
        remaining = max_guesses - n_prior
        action = opener_idx if n_prior == 0 else expert_action(p, candidates, remaining, all_words)
        tokens.append(toks)
        feats.append(candidate_features(candidates, letters, remaining))
        tgt.append(action)

    record(encode_state([], [], letters), all_words, 0)
    for _ in range(passes):
        for answer in rng.permutation(n):
            answer = int(answer)
            code = int(p[opener_idx, answer])
            guesses, codes = [opener_idx], [code]
            candidates = all_words[p[opener_idx, all_words] == code]
            for _turn in range(1, max_guesses):
                if len(candidates) == 0 or guesses[-1] == answer:
                    break
                record(encode_state(guesses, codes, letters), candidates, len(guesses))
                remaining = max_guesses - len(guesses)
                if student is not None:  # DAgger: behave as the raw student, label by expert
                    played = _student_raw(student, device, guesses, codes, candidates, letters, remaining)
                else:
                    played = expert_action(p, candidates, remaining, all_words)
                if len(candidates) > 1 and rng.random() < epsilon:  # exploration for coverage
                    played = int(rng.integers(n)) if rng.random() < full_pool_explore else int(rng.choice(candidates))
                code = int(p[played, answer])
                guesses.append(played)
                codes.append(code)
                candidates = candidates[p[played, candidates] == code]
                if played == answer:
                    break

    m = len(tokens)
    padded = np.full((m, MAX_LEN), TOK_PAD, dtype=np.uint8)
    for i, t in enumerate(tokens):
        padded[i, : len(t)] = t
    return padded, np.stack(feats).astype(np.float32), np.array(tgt, dtype=np.int64)[:, None], np.ones((m, 1), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="DAgger dataset: imitate the hybrid expert with raw targets.")
    ap.add_argument("--opener", default="slate")
    ap.add_argument("--passes", type=int, default=8)
    ap.add_argument("--epsilon", type=float, default=0.35)
    ap.add_argument("--student", type=Path, default=None, help="roll out this checkpoint RAW to cover its own states")
    ap.add_argument("--append", type=Path, default=None, help="aggregate (+dedup) with an existing dataset npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "dagger_dataset.npz")
    args = ap.parse_args()

    device = pick_device(args.device)
    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)
    opener_idx = vocab.index[args.opener]
    student = None
    if args.student:
        student, _ = load_checkpoint(args.student, map_location=str(device))
        student.to(device).eval()
        print(f"rolling out student {args.student} (raw) for DAgger coverage")

    print(f"collecting: {args.passes} passes, eps={args.epsilon}, opener={args.opener} ...")
    tok, ft, ti, tw = collect(vocab, p, opener_idx, args.passes, args.epsilon, args.seed, student, str(device))

    if args.append and args.append.exists():
        d = np.load(args.append)
        tok = np.concatenate([d["tokens"], tok])
        ft = np.concatenate([d["feats"], ft])
        ti = np.concatenate([d["tgt_idx"], ti])
        tw = np.concatenate([d["tgt_w"], tw])
        _, uniq = np.unique(tok, axis=0, return_index=True)  # expert is deterministic, so dedup is safe
        uniq = np.sort(uniq)
        tok, ft, ti, tw = tok[uniq], ft[uniq], ti[uniq], tw[uniq]
        print(f"aggregated with {args.append}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, tokens=tok, feats=ft, tgt_idx=ti, tgt_w=tw)
    print(f"wrote {len(tok)} states to {args.out}  (feat dim {ft.shape[1]})")


if __name__ == "__main__":
    main()
