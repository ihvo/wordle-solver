"""Wordle feedback: green / yellow / gray with correct duplicate-letter handling.

A feedback *pattern* over 5 slots is encoded as a base-3 integer in [0, 243):
each slot contributes ``color * 3**slot`` where gray=0, yellow=1, green=2.
Slot 0 is the least-significant digit.

Two implementations are provided:

* :func:`feedback_code` — scalar, readable reference (used in tests / parsing).
* :func:`feedback_codes_for_guess` — vectorized over many answers at once; this
  is what builds the pattern matrix the entropy solver runs on.
"""

from __future__ import annotations

import numpy as np

WORD_LEN = 5
N_PATTERNS = 3**WORD_LEN  # 243

GRAY, YELLOW, GREEN = 0, 1, 2
_CHAR_TO_COLOR = {"x": GRAY, "b": GRAY, ".": GRAY, "y": YELLOW, "g": GREEN}
_COLOR_TO_CHAR = "xyg"
_POWERS = np.array([3**i for i in range(WORD_LEN)], dtype=np.int32)  # 1,3,9,27,81

#: The pattern code for an all-green (solved) row.
SOLVED_CODE = int((np.full(WORD_LEN, GREEN) * _POWERS).sum())  # 242


def feedback_code(guess: str, answer: str) -> int:
    """Feedback of ``guess`` against ``answer`` as a base-3 pattern code."""
    colors = [GRAY] * WORD_LEN
    counts: dict[str, int] = {}
    for ch in answer:
        counts[ch] = counts.get(ch, 0) + 1
    # First pass: greens consume a letter from the pool.
    for i in range(WORD_LEN):
        if guess[i] == answer[i]:
            colors[i] = GREEN
            counts[guess[i]] -= 1
    # Second pass: yellows, left to right, only while the letter remains.
    for i in range(WORD_LEN):
        if colors[i] == GREEN:
            continue
        c = guess[i]
        if counts.get(c, 0) > 0:
            colors[i] = YELLOW
            counts[c] -= 1
    return sum(color * 3**i for i, color in enumerate(colors))


def feedback_codes_for_guess(guess: np.ndarray, answers: np.ndarray) -> np.ndarray:
    """Vectorized feedback of one ``guess`` (5,) against ``answers`` (M, 5).

    Letters are integer indices 0..25. Returns (M,) int array of pattern codes.
    Mirrors :func:`feedback_code` exactly, including duplicate handling.
    """
    guess = np.asarray(guess, dtype=np.int64)
    answers = np.asarray(answers, dtype=np.int64)
    m = answers.shape[0]
    rows = np.arange(m)

    colors = np.zeros((m, WORD_LEN), dtype=np.int8)
    greens = answers == guess[None, :]  # (M, 5)
    colors[greens] = GREEN

    # Per-answer remaining letter counts, after removing greens.
    avail = np.zeros((m, 26), dtype=np.int16)
    np.add.at(avail, (rows[:, None], answers), 1)
    for i in range(WORD_LEN):
        g = greens[:, i]
        avail[g, guess[i]] -= 1

    # Second pass: assign yellows left to right while the letter remains.
    for i in range(WORD_LEN):
        c = int(guess[i])
        mark = (~greens[:, i]) & (avail[:, c] > 0)
        colors[mark, i] = YELLOW
        avail[mark, c] -= 1

    return (colors.astype(np.int32) * _POWERS[None, :]).sum(axis=1)


def pattern_to_code(pattern: str) -> int:
    """Parse a feedback string like ``"gyxxg"`` into a pattern code."""
    pattern = pattern.strip().lower().replace(" ", "")
    if len(pattern) != WORD_LEN:
        raise ValueError(f"feedback must be {WORD_LEN} chars, got {pattern!r}")
    code = 0
    for i, ch in enumerate(pattern):
        if ch not in _CHAR_TO_COLOR:
            raise ValueError(f"bad feedback char {ch!r}; use g/y/x")
        code += _CHAR_TO_COLOR[ch] * 3**i
    return code


def code_to_pattern(code: int) -> str:
    """Render a pattern code back to a ``g/y/x`` string."""
    return "".join(_COLOR_TO_CHAR[(code // 3**i) % 3] for i in range(WORD_LEN))
