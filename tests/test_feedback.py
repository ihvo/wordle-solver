"""Tests for the feedback engine, with emphasis on duplicate-letter handling."""

import numpy as np
import pytest

from wordle_guesser.feedback import (
    SOLVED_CODE,
    code_to_pattern,
    feedback_code,
    feedback_codes_for_guess,
    pattern_to_code,
)
from wordle_guesser.words import load_vocabulary, words_to_letters


def _code(guess: str, answer: str) -> str:
    return code_to_pattern(feedback_code(guess, answer))


@pytest.mark.parametrize(
    "guess,answer,expected",
    [
        ("crane", "crane", "ggggg"),  # exact match
        ("crane", "boost", "xxxxx"),  # nothing shared
        ("sense", "abide", "xxxxg"),  # 2nd 'e' stays gray: only 'e' consumed by green
        ("three", "eerie", "xxgyg"),  # one yellow 'e' survives, the other is gray
        ("llama", "lille", "gyxxx"),  # 1st 'l' green, 2nd 'l' yellow (answer has more l's)
    ],
)
def test_known_patterns(guess, answer, expected):
    assert _code(guess, answer) == expected


def test_duplicate_in_guess_capped_by_answer_count():
    # guess 'geese' has three e's; answer 'abide' has one, matched green at slot 4.
    # The other two e's must therefore be gray, not yellow.
    assert _code("geese", "abide") == "xxxxg"


def test_solved_code():
    assert SOLVED_CODE == feedback_code("crane", "crane")
    assert code_to_pattern(SOLVED_CODE) == "ggggg"


def test_pattern_roundtrip():
    for code in range(243):
        assert pattern_to_code(code_to_pattern(code)) == code


def test_pattern_parsing_is_lenient():
    assert pattern_to_code("g y x x g") == pattern_to_code("gyxxg")
    assert pattern_to_code("GYXXG") == pattern_to_code("gyxxg")
    assert pattern_to_code("gybbg") == pattern_to_code("gyxxg")  # 'b' == gray
    with pytest.raises(ValueError):
        pattern_to_code("gyxx")  # too short
    with pytest.raises(ValueError):
        pattern_to_code("gyxxz")  # bad char


def test_vectorized_matches_scalar_on_full_vocab():
    """The vectorized path must equal the scalar reference everywhere."""
    vocab = load_vocabulary()
    answers = vocab.letters  # (N, 5)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(vocab), size=40, replace=False)
    for gi in sample:
        guess_word = vocab.words[gi]
        vec = feedback_codes_for_guess(vocab.letters[gi], answers)
        scalar = np.array(
            [feedback_code(guess_word, a) for a in vocab.words], dtype=vec.dtype
        )
        assert np.array_equal(vec, scalar), f"mismatch for guess {guess_word!r}"


def test_words_to_letters_roundtrip():
    arr = words_to_letters(["crane", "abbey"])
    assert arr.shape == (2, 5)
    assert arr[0].tolist() == [ord(c) - 97 for c in "crane"]
