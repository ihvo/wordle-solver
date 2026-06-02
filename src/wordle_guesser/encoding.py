"""Tokenization of a game state for the transformer policy.

The model reads the history of prior turns. Each prior guess contributes 5
tokens, one per slot, encoding *(letter, color)* jointly::

    token = letter_index * 3 + color        # letter 0..25, color 0..2  -> 0..77

A leading ``START`` token serves as the sequence summary that the output head
reads from. ``PAD`` fills unused positions. Learned positional embeddings over
``MAX_LEN`` positions implicitly encode both the turn and the slot within a word.
"""

from __future__ import annotations

import numpy as np

WORD_LEN = 5
MAX_PRIOR_GUESSES = 5  # we predict guesses 1..6 from 0..5 prior turns
MAX_LEN = 1 + WORD_LEN * MAX_PRIOR_GUESSES  # 26 token positions

TOK_START = 78
TOK_PAD = 79
N_TOKENS = 80

CAND_DIM = WORD_LEN * 26 + 26  # 156: per-position freqs (130) + letter-present freqs (26)

_POW3 = (1, 3, 9, 27, 81)


def candidate_features(candidates: np.ndarray, letters: np.ndarray) -> np.ndarray:
    """Summarize the remaining candidate set as a (156,) float32 vector.

    Two views, because entropy depends on more than per-slot marginals:

    * per-slot letter frequencies (5×26): fraction of candidates with each
      letter in each slot — drives green/position information.
    * letter-present frequencies (26): fraction of candidates containing each
      letter anywhere — drives yellow/presence information.

    Together these are (close to) the sufficient statistic the entropy ranking
    depends on, which is what makes the next-guess decision learnable.
    """
    pos = np.zeros((WORD_LEN, 26), dtype=np.float32)
    present = np.zeros(26, dtype=np.float32)
    n = len(candidates)
    if n == 0:
        return np.concatenate([pos.ravel(), present])
    sub = letters[candidates]  # (n, 5)
    has = np.zeros((n, 26), dtype=bool)
    rows = np.arange(n)
    for i in range(WORD_LEN):
        np.add.at(pos[i], sub[:, i], 1.0)
        has[rows, sub[:, i]] = True
    pos /= n
    present = has.mean(axis=0).astype(np.float32)
    return np.concatenate([pos.ravel(), present])


def encode_state(
    guesses: list[int], codes: list[int], letters: np.ndarray
) -> np.ndarray:
    """Encode prior turns into a 1-D int64 token array (unpadded).

    ``letters`` is the vocabulary's (N, 5) letter-index array.
    """
    toks = [TOK_START]
    for g, code in zip(guesses, codes):
        row = letters[g]
        for i in range(WORD_LEN):
            color = (code // _POW3[i]) % 3
            toks.append(int(row[i]) * 3 + color)
    return np.array(toks, dtype=np.int64)


def pad_to(tokens: np.ndarray, length: int = MAX_LEN) -> np.ndarray:
    """Right-pad a token array to ``length`` with ``TOK_PAD``."""
    out = np.full(length, TOK_PAD, dtype=tokens.dtype)
    out[: len(tokens)] = tokens
    return out


def pad_batch(token_arrays: list[np.ndarray], length: int = MAX_LEN):
    """Stack variable-length token arrays into ``(B, length)`` plus a pad mask.

    Returns ``(tokens, key_padding_mask)`` where the mask is ``True`` at padded
    positions (the convention ``nn.Transformer`` expects).
    """
    b = len(token_arrays)
    tokens = np.full((b, length), TOK_PAD, dtype=np.int64)
    for i, arr in enumerate(token_arrays):
        tokens[i, : len(arr)] = arr
    mask = tokens == TOK_PAD
    return tokens, mask
