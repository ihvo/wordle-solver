"""Tests for the RL post-training pieces (pure functions + teacher demos)."""

import numpy as np
import pytest

from wordle_guesser.encoding import CAND_DIM_AUG
from wordle_guesser.model import load_checkpoint
from wordle_guesser.policy import ModelPolicy
from wordle_guesser.rl import answer_weights, precompute_teacher_demos, reward_for
from wordle_guesser.solver import MAX_GUESSES, build_pattern_matrix, play_game
from wordle_guesser.train import MODELS_DIR
from wordle_guesser.words import load_vocabulary


@pytest.fixture(scope="module")
def vocab():
    return load_vocabulary()


@pytest.fixture(scope="module")
def pmatrix(vocab):
    return build_pattern_matrix(vocab)


def test_reward_orders_wins_above_losses_and_favours_speed():
    assert reward_for(0) == 0.0  # a loss
    # every win beats every loss, and fewer turns scores strictly higher
    rewards = [reward_for(t) for t in range(1, MAX_GUESSES + 1)]
    assert all(r > 0 for r in rewards)
    assert rewards == sorted(rewards, reverse=True)
    assert reward_for(2) == MAX_GUESSES - 1  # win-in-2 -> 5


def test_answer_weights_upweight_hard_games(vocab):
    n = len(vocab)
    turns = np.full(n, 3, dtype=np.int32)  # everyone wins in 3 ...
    turns[0] = 0   # ... except a loss
    turns[1] = 6   # ... and a slow win
    w = answer_weights(turns, tau=2.0, uniform_mix=0.3)
    assert np.isclose(w.sum(), 1.0)
    assert (w > 0).all()
    assert w[0] > w[1] > w[2]  # loss > slow win > fast win


def test_teacher_demos_win_and_replay_to_the_answer(vocab, pmatrix):
    p = pmatrix
    opener_idx = vocab.encode("slate")
    all_words = np.arange(len(vocab))
    # a known neighbour-trap plus an easy word
    answers = [vocab.encode(w) for w in ("pound", "watch", "vaunt", "crane")]
    demos = precompute_teacher_demos(p, vocab, opener_idx, answers=answers)

    for a in answers:
        steps, reward = demos[a]
        assert reward > 0, f"teacher failed on {vocab.words[a]!r}"
        # reward is consistent with trajectory length (opener is turn 1)
        assert reward == reward_for(1 + len(steps))
        # the last probe is the winning guess; replaying the actions reaches a
        cands = all_words[p[opener_idx, all_words] == p[opener_idx, a]]
        last_action = None
        for _toks, _feats, action, _stuck in steps:
            code = int(p[action, a])
            cands = cands[p[action, cands] == code]
            last_action = action
        assert last_action == a
        assert len(steps) <= MAX_GUESSES - 1  # at most turns 2..6
        # a neighbour-trap forces at least one stuck (probe) step
        if vocab.words[a] in ("pound", "vaunt"):
            assert any(stuck for *_x, stuck in steps)


def test_rl_hybrid_policy_wins_neighbour_traps(vocab, pmatrix):
    """The shipped RL policy, played hybrid, solves the traps masked play can't."""
    ckpt = MODELS_DIR / "policy_rl.pt"
    if not ckpt.exists():
        pytest.skip("RL checkpoint not built yet (run `python -m wordle_guesser.rl`)")
    model, _ = load_checkpoint(ckpt)
    hybrid = ModelPolicy(model, vocab, mask_to_candidates=True, probe_when_stuck=True)
    for w in ("pound", "bound", "vaunt", "watch", "foyer"):
        _g, _c, solved = play_game(vocab.encode(w), hybrid, pmatrix, len(vocab))
        assert solved, f"hybrid policy lost {w!r}"


def test_raw_policy_probes_without_the_rail(vocab, pmatrix):
    """The shipped raw policy (aug features) solves the traps playing pure argmax —
    no candidate mask, no probe branch — i.e. it internalized the rail."""
    ckpt = MODELS_DIR / "policy_raw.pt"
    if not ckpt.exists():
        pytest.skip("raw checkpoint not built yet (run the DAgger pipeline)")
    model, _ = load_checkpoint(ckpt)
    assert model.config.cand_dim == CAND_DIM_AUG  # uses the count/budget features
    raw = ModelPolicy(model, vocab, mask_to_candidates=False, probe_when_stuck=False)
    for w in ("pound", "bound", "vaunt", "watch", "foyer", "taste", "wound"):
        _g, _c, solved = play_game(vocab.encode(w), raw, pmatrix, len(vocab))
        assert solved, f"raw policy lost {w!r}"


def test_xattn_policy_plays_from_tokens_only(vocab, pmatrix):
    """The history-only net solves traps playing raw, with NO candidate set fed in."""
    ckpt = MODELS_DIR / "policy_xattn.pt"
    if not ckpt.exists():
        pytest.skip("xattn checkpoint not built yet (train with --xattn)")
    model, _ = load_checkpoint(ckpt)
    assert model.config.xattn
    raw = ModelPolicy(model, vocab, mask_to_candidates=False, probe_when_stuck=False)
    # incl. rajah/witch — the brittle double-letter / family traps the feature + raw-GRPO closed
    for w in ("pound", "bound", "wound", "watch", "vaunt", "patty", "sushi", "rajah", "witch"):
        _g, _c, solved = play_game(vocab.encode(w), raw, pmatrix, len(vocab))
        assert solved, f"xattn policy lost {w!r}"
