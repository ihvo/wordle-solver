"""Stateless next-guess helper for driving a live game.

Given the guesses played so far and the g/y/x feedback each received, rebuild the
consistent candidate set from scratch and print the next guess. By default the
entropy teacher picks; ``--model`` / ``--mlp`` use the trained policy instead.

Usage:
    python scripts/next_guess.py                       # opening guess (teacher)
    python scripts/next_guess.py --model raise:xgyyx    # model, after one guess
    python scripts/next_guess.py --mlp raise:xgyyx antic:yxxgx

Output (one JSON object): {"guess", "n_candidates", "candidates_sample", "solved", ...}.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from wordle_guesser.feedback import SOLVED_CODE, pattern_to_code
from wordle_guesser.solver import (
    EntropyTeacher,
    GameState,
    best_guess,
    filter_candidates,
    load_pattern_matrix,
)
from wordle_guesser.words import load_vocabulary


def main() -> None:
    ap = argparse.ArgumentParser(description="Print the solver's next Wordle guess.")
    ap.add_argument("history", nargs="*", help="played guesses as word:gyx (e.g. raise:xgyyx)")
    ap.add_argument("--model", default=None, help="checkpoint path (default models/policy.pt)")
    ap.add_argument("--mlp", action="store_true", help="use the candidate-only MLP checkpoint")
    ap.add_argument("--no-mask", action="store_true", help="let the model rank all words")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)

    # Replay history → candidate set + GameState.
    guesses: list[int] = []
    codes: list[int] = []
    candidates = np.arange(len(vocab))
    solved = False
    for tok in args.history:
        word, _, pat = tok.partition(":")
        idx = vocab.get(word.lower())
        if idx is None:
            print(json.dumps({"error": f"guess {word!r} not in vocabulary"}))
            return
        code = pattern_to_code(pat)
        guesses.append(idx)
        codes.append(code)
        if code == SOLVED_CODE:
            solved = True
        candidates = filter_candidates(p, candidates, idx, code)

    if solved:
        print(json.dumps({"guess": None, "n_candidates": int(len(candidates)),
                          "candidates_sample": [], "solved": True}))
        return
    if len(candidates) == 0:
        print(json.dumps({"error": "no candidates left (answer not in our 2315-word list)",
                          "n_candidates": 0}))
        return

    use_model = args.model is not None or args.mlp
    state = GameState(guesses=guesses, codes=codes, candidates=candidates)
    result = {
        "n_candidates": int(len(candidates)),
        "candidates_sample": [vocab.words[i] for i in candidates[:12]],
        "solved": False,
    }

    if use_model:
        from pathlib import Path

        from wordle_guesser.model import load_checkpoint
        from wordle_guesser.policy import ModelPolicy
        from wordle_guesser.train import MODELS_DIR, pick_device

        path = Path(args.model) if args.model else MODELS_DIR / ("policy_mlp.pt" if args.mlp else "policy.pt")
        device = pick_device(args.device)
        model, _ = load_checkpoint(path, map_location=str(device))
        policy = ModelPolicy(model, vocab, device=str(device), mask_to_candidates=not args.no_mask)
        guess = int(policy(state))
        result["guess"] = vocab.words[guess]
        result["source"] = f"model:{path.name}"
        result["top_picks"] = [[w, round(q, 3)] for w, q in policy.topk(state, k=5)]
    else:
        teacher = EntropyTeacher(p)
        guess = teacher.opening() if not args.history else best_guess(p, candidates)
        result["guess"] = vocab.words[guess]
        result["source"] = "teacher"

    print(json.dumps(result))


if __name__ == "__main__":
    main()
