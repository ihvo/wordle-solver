"""Tests for the pattern matrix, candidate filtering, and the teacher."""

import numpy as np
import pytest

from wordle_guesser.feedback import SOLVED_CODE, feedback_code
from wordle_guesser.solver import (
    EntropyTeacher,
    build_pattern_matrix,
    filter_candidates,
    play_game,
)
from wordle_guesser.words import Vocabulary, load_vocabulary


@pytest.fixture(scope="module")
def vocab() -> Vocabulary:
    return load_vocabulary()


@pytest.fixture(scope="module")
def pmatrix(vocab):
    # A small deterministic slice keeps the test fast but exercises real words.
    sub = Vocabulary.from_words(vocab.words[:300])
    return sub, build_pattern_matrix(sub)


def test_diagonal_is_solved(pmatrix):
    sub, p = pmatrix
    assert np.all(np.diag(p) == SOLVED_CODE)


def test_matrix_matches_scalar(pmatrix):
    sub, p = pmatrix
    rng = np.random.default_rng(1)
    for _ in range(200):
        g = int(rng.integers(len(sub)))
        a = int(rng.integers(len(sub)))
        assert p[g, a] == feedback_code(sub.words[g], sub.words[a])


def test_filter_keeps_the_answer(pmatrix):
    sub, p = pmatrix
    candidates = np.arange(len(sub))
    answer = 42
    for guess in (0, 100, 250):
        code = int(p[guess, answer])
        candidates = filter_candidates(p, candidates, guess, code)
        assert answer in candidates


def test_teacher_solves_within_six(vocab):
    p = build_pattern_matrix(vocab)
    teacher = EntropyTeacher(p)
    rng = np.random.default_rng(7)
    for answer in rng.choice(len(vocab), size=25, replace=False):
        guesses, _, solved = play_game(int(answer), teacher, p, len(vocab))
        assert solved, f"teacher failed on {vocab.words[answer]!r}"
        assert len(guesses) <= 6
