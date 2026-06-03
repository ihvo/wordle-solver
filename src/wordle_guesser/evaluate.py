"""Evaluate the trained policy: play every answer and compare to the teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .encoding import candidate_features, encode_state, pad_batch
from .model import load_checkpoint
from .solver import MAX_GUESSES, EntropyTeacher, load_pattern_matrix, play_game
from .train import MODELS_DIR, pick_device
from .words import load_vocabulary


def summarize(solved_turn: np.ndarray) -> dict:
    """solved_turn: per-game #guesses, or 0 for a loss. Returns metrics."""
    won = solved_turn > 0
    n = len(solved_turn)
    dist = {int(k): int(v) for k, v in zip(*np.unique(solved_turn[won], return_counts=True))}
    return {
        "games": n,
        "win_rate": float(won.mean()),
        "avg_guesses": float(solved_turn[won].mean()) if won.any() else float("nan"),
        "losses": int((~won).sum()),
        "distribution": dist,
    }


@torch.no_grad()
def evaluate_model(
    model, vocab, p, device, mask_to_candidates=True, probe_when_stuck=False,
    max_guesses=MAX_GUESSES,
) -> np.ndarray:
    """Play all answers in lockstep; return per-game guess count (0 = loss).

    ``probe_when_stuck`` plays the *hybrid* (deploy) policy: keep the candidate
    mask while the survivors fit the remaining budget, but lift it once they
    outnumber it so the model can probe with a non-candidate. This is how the RL
    checkpoint reaches 100%.
    """
    n = len(vocab)
    letters = vocab.letters
    opener = getattr(model.config, "opener", None)
    opener_idx = vocab.index[opener] if opener else None
    guesses = [[] for _ in range(n)]
    codes = [[] for _ in range(n)]
    candidates = [np.arange(n) for _ in range(n)]
    result = np.zeros(n, dtype=np.int32)
    active = list(range(n))

    for turn in range(max_guesses):
        if not active:
            break
        tok_arrays = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(tok_arrays)
        feats = np.stack([candidate_features(candidates[i], letters) for i in active])
        logits = (
            model(
                torch.from_numpy(tokens).to(device),
                torch.from_numpy(kpm).to(device),
                torch.from_numpy(feats).to(device),
            )
            .float()
            .cpu()
            .numpy()
        )
        remaining = max_guesses - turn  # guesses left including this one
        still_active = []
        for row, i in enumerate(active):
            lg = logits[row]
            cand = candidates[i]
            if opener_idx is not None and turn == 0:
                guess = opener_idx
            elif mask_to_candidates and not (probe_when_stuck and len(cand) > remaining):
                guess = int(cand[np.argmax(lg[cand])])
            else:  # raw, or stuck under the hybrid rail -> probe over all words
                guess = int(np.argmax(lg))
            code = int(p[guess, i])  # answer for game i is word i
            guesses[i].append(guess)
            codes[i].append(code)
            candidates[i] = candidates[i][p[guess, candidates[i]] == code]
            if guess == i:
                result[i] = turn + 1
            else:
                still_active.append(i)
        active = still_active
    return result


def evaluate_teacher(vocab, p, max_guesses=MAX_GUESSES) -> np.ndarray:
    teacher = EntropyTeacher(p)
    result = np.zeros(len(vocab), dtype=np.int32)
    for a in range(len(vocab)):
        g, _, solved = play_game(a, teacher, p, len(vocab), max_guesses)
        result[a] = len(g) if solved else 0
    return result


def _print_row(label: str, m: dict) -> None:
    dist = " ".join(f"{k}:{m['distribution'].get(k, 0)}" for k in range(1, MAX_GUESSES + 1))
    print(
        f"{label:<22} win {m['win_rate']*100:6.2f}%   "
        f"avg {m['avg_guesses']:.4f}   losses {m['losses']:>3}   [{dist}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the Wordle policy vs the teacher.")
    ap.add_argument("--model", type=Path, default=MODELS_DIR / "policy_rl.pt")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)

    model, ckpt_words = load_checkpoint(args.model, map_location=str(device))
    model.to(device)
    if ckpt_words != vocab.words:
        raise SystemExit("checkpoint vocabulary does not match data/answers.txt")

    print(f"device: {device}   model: {args.model.name}\n")
    _print_row("teacher (entropy)", summarize(evaluate_teacher(vocab, p)))
    _print_row("model (hybrid)", summarize(evaluate_model(model, vocab, p, device, True, probe_when_stuck=True)))
    _print_row("model (masked)", summarize(evaluate_model(model, vocab, p, device, True)))
    _print_row("model (raw, unmasked)", summarize(evaluate_model(model, vocab, p, device, False)))


if __name__ == "__main__":
    main()
