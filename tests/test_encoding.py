"""Tests for game-state tokenization."""

import numpy as np

from wordle_guesser.encoding import (
    CAND_DIM,
    CAND_DIM_AUG,
    MAX_LEN,
    TOK_PAD,
    TOK_START,
    candidate_features,
    encode_state,
    pad_batch,
)
from wordle_guesser.feedback import feedback_code
from wordle_guesser.words import load_vocabulary


def test_opening_state_is_just_start():
    vocab = load_vocabulary()
    tok = encode_state([], [], vocab.letters)
    assert tok.tolist() == [TOK_START]


def test_tokens_encode_letter_and_color():
    vocab = load_vocabulary()
    g = vocab.encode("crane")
    code = feedback_code("crane", "trace")
    tok = encode_state([g], [code], vocab.letters)
    assert tok[0] == TOK_START
    assert len(tok) == 1 + 5
    # Each history token packs (letter, color): token // 3 is the letter index.
    letters = [(t // 3) for t in tok[1:]]
    assert letters == [ord(c) - 97 for c in "crane"]
    colors = [(t % 3) for t in tok[1:]]
    assert all(0 <= c <= 2 for c in colors)


def test_candidate_features():
    vocab = load_vocabulary()
    # A two-word set with known letters: "crane", "crate".
    cands = np.array([vocab.encode("crane"), vocab.encode("crate")])
    feat = candidate_features(cands, vocab.letters)
    assert feat.shape == (CAND_DIM,)
    pos, present = feat[:130].reshape(5, 26), feat[130:]
    # Both words start "cra": those slots are fully determined (freq 1.0).
    assert pos[0, ord("c") - 97] == 1.0
    assert pos[1, ord("r") - 97] == 1.0
    assert pos[2, ord("a") - 97] == 1.0
    # Slot 3 splits n/t at 0.5 each; slot 4 is all 'e'.
    assert pos[3, ord("n") - 97] == 0.5 and pos[3, ord("t") - 97] == 0.5
    assert pos[4, ord("e") - 97] == 1.0
    # 'c','r','a','e' present in all; 'n' and 't' in half.
    for ch in "crae":
        assert present[ord(ch) - 97] == 1.0
    assert present[ord("n") - 97] == 0.5 and present[ord("t") - 97] == 0.5


def test_candidate_features_empty_set():
    vocab = load_vocabulary()
    feat = candidate_features(np.array([], dtype=int), vocab.letters)
    assert feat.shape == (CAND_DIM,) and not feat.any()


def test_candidate_features_state_aug():
    """With ``remaining`` set, append count + budget features (170-d)."""
    vocab = load_vocabulary()
    cands = np.array([vocab.encode("pound"), vocab.encode("bound")])
    feat = candidate_features(cands, vocab.letters, remaining=2)
    assert feat.shape == (CAND_DIM_AUG,)
    extra = feat[CAND_DIM:]
    assert 0.0 < extra[0] < 1.0           # log-count, normalized
    assert extra[1 + 2] == 1.0            # count bucket = 2 candidates
    assert extra[8 + (2 - 1)] == 1.0      # remaining-guesses one-hot at R=2
    assert extra.sum() == extra[0] + 2.0  # logC + two one-hots
    # the 156-d prefix is unchanged from the un-augmented call
    assert np.array_equal(feat[:CAND_DIM], candidate_features(cands, vocab.letters))


def test_pad_batch_shapes_and_mask():
    vocab = load_vocabulary()
    a = encode_state([], [], vocab.letters)  # len 1
    g = vocab.encode("crane")
    b = encode_state([g], [feedback_code("crane", "trace")], vocab.letters)  # len 6
    tokens, mask = pad_batch([a, b])
    assert tokens.shape == (2, MAX_LEN)
    assert mask.dtype == np.bool_
    assert mask[0].sum() == MAX_LEN - 1  # only START is real
    assert mask[1].sum() == MAX_LEN - 6
    assert tokens[0, 1] == TOK_PAD
