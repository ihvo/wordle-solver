"""Interactive Wordle solver.

It suggests a guess; you play it (or any other answer-list word), type the
feedback Wordle gave you, and it suggests the next guess. By default the
suggestion comes from the trained transformer policy; ``--teacher`` uses the
entropy solver instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .feedback import SOLVED_CODE, code_to_pattern, pattern_to_code
from .model import load_checkpoint
from .policy import ModelPolicy
from .solver import (
    MAX_GUESSES,
    EntropyTeacher,
    GameState,
    filter_candidates,
    load_pattern_matrix,
)
from .train import MODELS_DIR, pick_device
from .words import load_vocabulary

HELP = (
    "Feedback legend:  g = green (right spot)   y = yellow (wrong spot)   "
    "x = gray (absent)\nExample:  gyxxg     (type 'q' to quit)\n"
)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)


def _sample(words, candidates, k=8) -> str:
    picks = [words[i] for i in candidates[:k]]
    more = "" if len(candidates) <= k else f", … (+{len(candidates) - k})"
    return ", ".join(picks) + more


def run(policy, vocab, p, *, is_model: bool, max_guesses: int = MAX_GUESSES) -> None:
    print(HELP)
    state = GameState(candidates=np.arange(len(vocab)))

    for turn in range(1, max_guesses + 1):
        n_cand = len(state.candidates)
        if n_cand == 0:
            print("No candidates left — the feedback was inconsistent, or the answer")
            print("isn't in the official list. Restart and re-check your colors.")
            return

        suggestion = vocab.words[policy(state)]
        print(f"\nTurn {turn}/{max_guesses} — {n_cand} candidate(s): {_sample(vocab.words, state.candidates)}")
        print(f"  suggested guess:  {suggestion.upper()}")
        if is_model:
            tops = policy.topk(state, k=5)
            print("  model top picks:  " + ", ".join(f"{w} {q*100:.0f}%" for w, q in tops))

        # Which word did you actually play?
        while True:
            played = _ask(f"  guess played [{suggestion}]: ").lower() or suggestion
            if played == "q":
                return
            idx = vocab.get(played)
            if idx is None:
                print("    (v1 works with official answer-list words — try the suggestion)")
                continue
            break

        # What feedback did Wordle give?
        while True:
            raw = _ask("  feedback (g/y/x): ")
            if raw == "q":
                return
            try:
                code = pattern_to_code(raw)
                break
            except ValueError as exc:
                print(f"    {exc}")

        if code == SOLVED_CODE:
            print(f"\nSolved — the word is {played.upper()} in {turn} guess(es). ")
            return

        state = GameState(
            guesses=[*state.guesses, idx],
            codes=[*state.codes, code],
            candidates=filter_candidates(p, state.candidates, idx, code),
        )
        print(f"    recorded {played.upper()} -> {code_to_pattern(code)}")

    print("\nOut of guesses. Remaining candidates:", _sample(vocab.words, state.candidates, 12))


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive Wordle solver.")
    ap.add_argument("--model", type=Path, default=None, help="checkpoint path (overrides the flags below)")
    ap.add_argument("--rl", action="store_true", help="RL policy played hybrid (models/policy_rl.pt)")
    ap.add_argument("--xattn", action="store_true", help="history-only net, no candidate features (models/policy_xattn.pt)")
    ap.add_argument("--bc", action="store_true", help="behavior-cloned transformer (models/policy.pt)")
    ap.add_argument("--mlp", action="store_true", help="candidate-only MLP (models/policy_mlp.pt)")
    ap.add_argument("--teacher", action="store_true", help="use the entropy solver, not the model")
    ap.add_argument("--safe", action="store_true", help="overlay the hybrid rail (guarantees valid endgame suggestions)")
    ap.add_argument("--no-mask", action="store_true", help="force raw ranking over all words")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)

    if args.teacher:
        run(EntropyTeacher(p), vocab, p, is_model=False)
        return

    # Default = the raw policy (models/policy_raw.pt): it internalized the rail, so it
    # plays pure argmax (forced opener only) and still solves all 2315. --rl is the older
    # hybrid-rail policy; --bc/--mlp play masked. mask/probe defaults per mode below.
    if args.model is not None:
        model_path, mask, probe = args.model, not args.no_mask, False
    elif args.rl:
        model_path, mask, probe = MODELS_DIR / "policy_rl.pt", True, True
    elif args.xattn:        # history-only: reads tokens, ignores candidate features
        model_path, mask, probe = MODELS_DIR / "policy_xattn.pt", False, False
    elif args.bc:
        model_path, mask, probe = MODELS_DIR / "policy.pt", True, False
    elif args.mlp:
        model_path, mask, probe = MODELS_DIR / "policy_mlp.pt", True, False
    else:
        model_path, mask, probe = MODELS_DIR / "policy_raw.pt", False, False

    if args.safe:        # opt back into the rail for guaranteed-valid suggestions
        mask, probe = True, True
    if args.no_mask:     # force raw
        mask, probe = False, False

    if not model_path.exists():
        raise SystemExit(
            f"no model at {model_path}. Train one first, or run with --teacher.\n"
            "  uv run python -m wordle_guesser.dataset --opener slate --state-aug\n"
            "  uv run python -m wordle_guesser.train --opener slate --epochs 24\n"
            "  uv run python -m wordle_guesser.dagger ...   # -> models/policy_raw.pt"
        )
    device = pick_device(args.device)
    model, _ = load_checkpoint(model_path, map_location=str(device))
    policy = ModelPolicy(
        model, vocab, device=str(device),
        mask_to_candidates=mask, probe_when_stuck=probe,
    )
    run(policy, vocab, p, is_model=True)


if __name__ == "__main__":
    main()
