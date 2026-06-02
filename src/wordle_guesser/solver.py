"""Entropy-maximizing teacher solver and the game environment.

The teacher precomputes a full pattern matrix ``P[guess, answer]`` over the
vocabulary, then at each turn picks the guess that maximizes the Shannon
entropy of the feedback distribution over the remaining candidates — i.e. the
guess expected to shrink the candidate set the most. Ties prefer a guess that is
itself still a candidate (so it can win outright).

This is the strong, classic baseline the transformer policy is trained to
imitate and is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .feedback import N_PATTERNS, feedback_codes_for_guess
from .words import DATA_DIR, Vocabulary

MAX_GUESSES = 6


# --------------------------------------------------------------------------- #
# Pattern matrix
# --------------------------------------------------------------------------- #
def build_pattern_matrix(vocab: Vocabulary) -> np.ndarray:
    """Compute P[g, a] = feedback code of guess g against answer a. (N, N) uint8."""
    n = len(vocab)
    p = np.empty((n, n), dtype=np.uint8)
    letters = vocab.letters
    for g in range(n):
        p[g] = feedback_codes_for_guess(letters[g], letters).astype(np.uint8)
    return p


def load_pattern_matrix(
    vocab: Vocabulary, cache_path: str | Path = DATA_DIR / "patterns_answers.npy"
) -> np.ndarray:
    """Load the pattern matrix, building and caching it on first use."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        p = np.load(cache_path)
        if p.shape == (len(vocab), len(vocab)):
            return p
    p = build_pattern_matrix(vocab)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, p)
    return p


# --------------------------------------------------------------------------- #
# Entropy guess selection
# --------------------------------------------------------------------------- #
def _row_histograms(codes: np.ndarray) -> np.ndarray:
    """Per-row counts over the 243 pattern bins for a (G, C) code matrix."""
    g = codes.shape[0]
    flat = (np.arange(g)[:, None] * N_PATTERNS + codes.astype(np.int64)).ravel()
    counts = np.bincount(flat, minlength=g * N_PATTERNS)
    return counts.reshape(g, N_PATTERNS)


def guess_entropies(
    p: np.ndarray, candidates: np.ndarray, pool: np.ndarray
) -> np.ndarray:
    """Shannon entropy (bits) of each guess in ``pool`` over ``candidates``."""
    codes = p[np.ix_(pool, candidates)]  # (G, C)
    counts = _row_histograms(codes).astype(np.float64)
    total = counts.sum(axis=1, keepdims=True)
    probs = counts / total
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(probs > 0, probs * np.log2(probs), 0.0)
    return -plogp.sum(axis=1)


def best_guess(
    p: np.ndarray, candidates: np.ndarray, pool: np.ndarray | None = None
) -> int:
    """Pick the maximum-entropy guess from ``pool`` against ``candidates``."""
    if len(candidates) == 1:
        return int(candidates[0])
    if pool is None:
        pool = np.arange(p.shape[0])
    entropy = guess_entropies(p, candidates, pool)
    # Tie-break: prefer a guess that is itself a candidate (could win now).
    score = entropy + 1e-9 * np.isin(pool, candidates)
    return int(pool[int(np.argmax(score))])


def filter_candidates(
    p: np.ndarray, candidates: np.ndarray, guess: int, code: int
) -> np.ndarray:
    """Keep candidates consistent with observing ``code`` for ``guess``."""
    return candidates[p[guess, candidates] == code]


# --------------------------------------------------------------------------- #
# Game environment
# --------------------------------------------------------------------------- #
@dataclass
class GameState:
    """What a policy sees on its turn."""

    guesses: list[int] = field(default_factory=list)
    codes: list[int] = field(default_factory=list)
    candidates: np.ndarray | None = None  # current consistent answer set

    @property
    def turn(self) -> int:
        return len(self.guesses)


def play_game(
    answer: int,
    policy,
    p: np.ndarray,
    vocab_size: int,
    max_guesses: int = MAX_GUESSES,
) -> tuple[list[int], list[int], bool]:
    """Play one game with ``policy`` (a callable GameState -> guess index)."""
    candidates = np.arange(vocab_size)
    state = GameState(candidates=candidates)
    for _ in range(max_guesses):
        guess = int(policy(state))
        code = int(p[guess, answer])
        state.guesses.append(guess)
        state.codes.append(code)
        candidates = filter_candidates(p, candidates, guess, code)
        state.candidates = candidates
        if guess == answer:
            return state.guesses, state.codes, True
    return state.guesses, state.codes, False


# --------------------------------------------------------------------------- #
# Teacher policy
# --------------------------------------------------------------------------- #
class EntropyTeacher:
    """Greedy maximum-entropy policy over the vocabulary."""

    def __init__(self, p: np.ndarray, pool: np.ndarray | None = None):
        self.p = p
        self.n = p.shape[0]
        self.pool = np.arange(self.n) if pool is None else np.asarray(pool)
        self._opening: int | None = None

    def opening(self) -> int:
        """The (deterministic) first guess, computed once and cached."""
        if self._opening is None:
            self._opening = best_guess(self.p, np.arange(self.n), self.pool)
        return self._opening

    def act(self, state: GameState) -> int:
        candidates = state.candidates
        if candidates is None or len(candidates) == self.n:
            return self.opening()
        return best_guess(self.p, candidates, self.pool)

    __call__ = act
