"""Inference wrapper: turn a trained model into a GameState -> guess policy."""

from __future__ import annotations

import numpy as np
import torch

from .encoding import candidate_features, encode_state, pad_batch
from .solver import GameState


class ModelPolicy:
    """Single-game policy used by the CLI and as a callable for play_game.

    When ``mask_to_candidates`` is set (the default), the model only ranks words
    still consistent with the feedback so far — it never suggests an impossible
    word, and the solver's exact filtering guarantees a win once one candidate
    remains. Disable it to see the model's raw, unconstrained preference.
    """

    def __init__(self, model, vocab, device="cpu", mask_to_candidates: bool = True):
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.device = device
        self.mask_to_candidates = mask_to_candidates
        # The opening is a fixed, known move — play it directly rather than asking
        # the net to recall a single training example (which it does unreliably).
        op = getattr(model.config, "opener", None)
        self.opener_idx = vocab.index[op] if op else None

    @torch.no_grad()
    def logits(self, state: GameState) -> torch.Tensor:
        tok = encode_state(state.guesses, state.codes, self.vocab.letters)
        tokens, mask = pad_batch([tok])
        feats = candidate_features(state.candidates, self.vocab.letters)[None, :]
        out = self.model(
            torch.from_numpy(tokens).to(self.device),
            torch.from_numpy(mask).to(self.device),
            torch.from_numpy(feats).to(self.device),
        )[0].float().cpu()
        if self.mask_to_candidates and state.candidates is not None and len(state.candidates):
            masked = torch.full_like(out, float("-inf"))
            cand = torch.as_tensor(np.asarray(state.candidates), dtype=torch.long)
            masked[cand] = out[cand]
            return masked
        return out

    def __call__(self, state: GameState) -> int:
        if self.opener_idx is not None and state.turn == 0:
            return self.opener_idx
        return int(self.logits(state).argmax())

    def topk(self, state: GameState, k: int = 5) -> list[tuple[str, float]]:
        if self.opener_idx is not None and state.turn == 0:
            return [(self.vocab.words[self.opener_idx], 1.0)]
        probs = torch.softmax(self.logits(state), dim=0)
        k = min(k, int((probs > 0).sum()))
        vals, idx = torch.topk(probs, k)
        return [(self.vocab.words[int(i)], float(v)) for v, i in zip(vals, idx)]
