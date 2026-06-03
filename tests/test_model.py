"""Shape and save/load tests for the policy and its ablation variants."""

import torch

from wordle_guesser.encoding import CAND_DIM, MAX_LEN
from wordle_guesser.model import PolicyConfig, WordlePolicy, load_checkpoint, save_checkpoint
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
