"""Shape and save/load tests for the policy and its ablation variants."""

import torch

from wordle_guesser.encoding import CAND_DIM, MAX_LEN, WORD_LEN
from wordle_guesser.model import (
    LetterDecoder,
    PolicyConfig,
    WordlePolicy,
    load_checkpoint,
    save_checkpoint,
)
from wordle_guesser.words import load_vocabulary

N = 40  # tiny vocab slice for fast tests


def _batch(b=4, length=MAX_LEN):
    tokens = torch.randint(0, 78, (b, length))
    mask = torch.zeros(b, length, dtype=torch.bool)
    feats = torch.rand(b, CAND_DIM)
    return tokens, mask, feats


def test_transformer_forward_shape():
    m = WordlePolicy(PolicyConfig(n_words=N))
    out = m(*_batch())
    assert out.shape == (4, N)


def test_candidate_only_is_smaller_and_works():
    full = sum(p.numel() for p in WordlePolicy(PolicyConfig(n_words=N)).parameters())
    mlp = WordlePolicy(PolicyConfig(n_words=N, use_history=False))
    # No attention/embeddings: just the candidate MLP + head.
    names = {n.split(".")[0] for n, _ in mlp.named_parameters()}
    assert names == {"cand_proj", "head"}
    assert sum(p.numel() for p in mlp.parameters()) < full
    assert mlp(*_batch()).shape == (4, N)


def test_xattn_history_only_ignores_candidate_features():
    """The cross-attention head reads only the history — candidate features must
    not affect its output."""
    words = load_vocabulary().letters[:N]
    m = WordlePolicy(PolicyConfig(n_words=N, xattn=True), word_letters=words).eval()
    tokens = torch.randint(0, 78, (4, MAX_LEN))
    mask = torch.zeros(4, MAX_LEN, dtype=torch.bool)
    mask[:, 4:] = True
    out = m(tokens, mask, torch.zeros(4, CAND_DIM))
    assert out.shape == (4, N)
    # two very different "feature" inputs -> identical logits (features are ignored)
    with torch.no_grad():
        a = m(tokens, mask, torch.zeros(4, CAND_DIM))
        b = m(tokens, mask, torch.rand(4, CAND_DIM) * 9)
    assert torch.allclose(a, b, atol=1e-6)


def test_decoder_shapes_teacher_forced_and_generated():
    """The generative head: teacher-forced logits (B,5,26), and autoregressive
    decoding / sampling / beam all return (B,5) letters."""
    m = WordlePolicy(PolicyConfig(n_words=N, decoder=True, decoder_marginal=True)).eval()
    tokens, mask, feats = _batch()
    target = torch.randint(0, 26, (4, WORD_LEN))
    logits = m(tokens, mask, feats, target_letters=target)
    assert logits.shape == (4, WORD_LEN, 26)

    gen = m.generate(tokens, mask, feats)
    assert gen.shape == (4, WORD_LEN) and gen.min() >= 0 and gen.max() < 26
    letters, logps = m.decode_sample(tokens, mask, feats)
    assert letters.shape == (4, WORD_LEN) and logps.shape == (4, WORD_LEN)
    beam = m.generate_beam(tokens, mask, feats, beam=5)
    assert beam.shape == (4, WORD_LEN) and beam.min() >= 0 and beam.max() < 26


def test_decoder_marginal_conditioning_changes_output():
    """The marginal lever: with conditioning on, the per-slot candidate marginals
    must move the logits (same state z); with it off, they're ignored entirely."""
    z = torch.rand(4, 32)
    target = torch.randint(0, 26, (4, WORD_LEN))
    m1 = torch.rand(4, WORD_LEN, 26)
    m2 = torch.rand(4, WORD_LEN, 26)

    on = LetterDecoder(d=16, fusion_dim=32, use_marginal=True).eval()
    with torch.no_grad():
        assert not torch.allclose(
            on(z, target=target, marg=m1), on(z, target=target, marg=m2), atol=1e-5
        )

    off = LetterDecoder(d=16, fusion_dim=32, use_marginal=False).eval()
    with torch.no_grad():
        # marg is ignored when the lever is off — even a non-None marg has no effect.
        assert torch.allclose(
            off(z, target=target, marg=m1), off(z, target=target, marg=None), atol=1e-6
        )


def test_decoder_roundtrip(tmp_path):
    cfg = PolicyConfig(n_words=N, decoder=True, decoder_marginal=True)
    m = WordlePolicy(cfg).eval()
    tokens, mask, feats = _batch()
    target = torch.randint(0, 26, (4, WORD_LEN))
    path = tmp_path / "decoder.pt"
    save_checkpoint(path, m, [f"w{i}" for i in range(N)])
    m2, _ = load_checkpoint(path)
    m2.eval()
    with torch.no_grad():
        assert torch.allclose(
            m(tokens, mask, feats, target_letters=target),
            m2(tokens, mask, feats, target_letters=target),
            atol=1e-5,
        )


def test_history_only_decoder_ignores_candidate_features():
    """Solver-free generative decoder: no candidate MLP, candidate features ignored."""
    m = WordlePolicy(PolicyConfig(n_words=N, use_history=True, use_candidates=False, decoder=True)).eval()
    assert not hasattr(m, "cand_proj")
    tokens, mask, _ = _batch()
    with torch.no_grad():
        a = m.generate(tokens, mask, torch.zeros(4, CAND_DIM))
        b = m.generate(tokens, mask, torch.rand(4, CAND_DIM) * 9)
    assert torch.equal(a, b)


def test_xattn_decoder_reads_the_history():
    """The cross-attention decoder's output depends on the token sequence it attends."""
    m = WordlePolicy(PolicyConfig(n_words=N, use_history=True, use_candidates=False,
                                  decoder=True, decoder_xattn=True)).eval()
    mask = torch.zeros(4, MAX_LEN, dtype=torch.bool)
    feats = torch.zeros(4, CAND_DIM)
    tgt = torch.randint(0, 26, (4, WORD_LEN))
    t1, t2 = torch.randint(1, 78, (4, MAX_LEN)), torch.randint(1, 78, (4, MAX_LEN))
    with torch.no_grad():
        assert not torch.allclose(m(t1, mask, feats, target_letters=tgt),
                                  m(t2, mask, feats, target_letters=tgt))


def test_learned_marginal_decoder_is_solver_free():
    """B/B2: the marginal is predicted from the encoder (return_aux), so play ignores
    candidate features at inference — pooled and attention heads alike."""
    for marg_attn in (False, True):
        m = WordlePolicy(PolicyConfig(n_words=N, use_history=True, use_candidates=False, decoder=True,
                                      decoder_xattn=True, decoder_learned_marginal=True,
                                      decoder_marg_attn=marg_attn)).eval()
        tokens, mask, _ = _batch()
        tgt = torch.randint(0, 26, (4, WORD_LEN))
        logits, marg_logits = m(tokens, mask, torch.zeros(4, CAND_DIM), target_letters=tgt, return_aux=True)
        assert logits.shape == (4, WORD_LEN, 26) and marg_logits.shape == (4, WORD_LEN, 26)
        with torch.no_grad():
            a = m.generate(tokens, mask, torch.zeros(4, CAND_DIM))
            b = m.generate(tokens, mask, torch.rand(4, CAND_DIM) * 9)
        assert torch.equal(a, b)


def test_word_lm_decoder():
    """#2: the word-LM branch exists, scores real words, and play still produces 5 letters."""
    m = WordlePolicy(PolicyConfig(n_words=N, use_history=True, use_candidates=False,
                                  decoder=True, decoder_word_lm=True)).eval()
    assert hasattr(m.dec, "lm_gru")
    tgt = torch.randint(0, 26, (4, WORD_LEN))
    assert m.dec.word_lm_logits(tgt).shape == (4, WORD_LEN, 26)
    tokens, mask, feats = _batch()
    assert m.generate(tokens, mask, feats).shape == (4, WORD_LEN)


def test_factored_head_forward_and_roundtrip(tmp_path):
    words = load_vocabulary().letters[:N]
    cfg = PolicyConfig(n_words=N, factored_head=True)
    m = WordlePolicy(cfg, word_letters=words)
    out = m(*_batch())
    assert out.shape == (4, N)

    # Save/load must restore the letter-index buffer and reproduce outputs.
    path = tmp_path / "factored.pt"
    save_checkpoint(path, m, [f"w{i}" for i in range(N)])
    m2, _ = load_checkpoint(path)
    m.eval(); m2.eval()
    batch = _batch()
    with torch.no_grad():
        assert torch.allclose(m(*batch), m2(*batch), atol=1e-5)
