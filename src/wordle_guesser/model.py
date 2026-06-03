"""The transformer policy: reads encoded game history, predicts the next guess.

Two ablation switches live in :class:`PolicyConfig`:

* ``use_history`` — when False, the transformer encoder is dropped entirely and
  the model predicts from the candidate-set features alone (tests whether the
  history path earns its parameters).
* ``factored_head`` — when True, the big ``d → n_words`` output matrix is replaced
  by per-word embeddings *built from each word's letters*, scoring words as a dot
  product. This swaps ~595K free parameters for a ~16K letter table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .encoding import CAND_DIM, MAX_LEN, N_TOKENS, TOK_PAD, WORD_LEN


@dataclass
class PolicyConfig:
    n_words: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_len: int = MAX_LEN
    n_tokens: int = N_TOKENS
    cand_dim: int = CAND_DIM
    use_history: bool = True
    factored_head: bool = False
    xattn: bool = False  # history-only: words cross-attend the encoded history (no candidate features)
    letter_count: bool = False  # add a per-token "how many of this letter in its guess" embedding (helps duplicates)
    opener: str | None = None  # fixed turn-1 word; the net can't learn a 1-example opening


class CrossAttnWordHead(nn.Module):
    """History-only readout. Each vocabulary word (embedded from its letters)
    cross-attends the encoded ``(guess, feedback)`` history and is scored against
    its own attended context. The per-word × history interaction is what lets the
    net check candidacy itself — no solver-computed candidate set is fed in.
    """

    def __init__(self, d: int, n_words: int, nhead: int, word_letters: np.ndarray | None):
        super().__init__()
        self.nhead = nhead
        self.dh = d // nhead
        self.letter_emb = nn.Embedding(WORD_LEN * 26, d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.word_bias = nn.Parameter(torch.zeros(n_words))
        idx = np.zeros((n_words, WORD_LEN), dtype=np.int64)
        if word_letters is not None:
            idx = (np.arange(WORD_LEN) * 26 + word_letters).astype(np.int64)
        self.register_buffer("word_letter_idx", torch.from_numpy(idx))

    def word_embeddings(self) -> torch.Tensor:
        return self.letter_emb(self.word_letter_idx).sum(dim=1)  # (n_words, d)

    def forward(self, hist, key_padding_mask) -> torch.Tensor:
        """hist (B,L,d) encoded history; mask (B,L) True=pad -> logits (B,n_words)."""
        b, length, _ = hist.shape
        w = self.word_embeddings()                                # (n, d)
        n, h, dh = w.shape[0], self.nhead, self.dh
        q = self.q(w).view(n, h, dh)
        k = self.k(hist).view(b, length, h, dh)
        v = self.v(hist).view(b, length, h, dh)
        scores = torch.einsum("nhe,blhe->bhnl", q, k) / (dh ** 0.5)   # (B,h,n,L)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        ctx = torch.einsum("bhnl,blhe->bnhe", scores.softmax(-1), v)  # (B,n,h,dh)
        ctx = self.out(ctx.reshape(b, n, h * dh))                    # (B,n,d)
        return torch.einsum("nd,bnd->bn", w, ctx) + self.word_bias    # (B, n_words)


class WordlePolicy(nn.Module):
    """Candidate-feature MLP + optional Transformer-over-history, into a word head."""

    def __init__(self, config: PolicyConfig, word_letters: np.ndarray | None = None):
        super().__init__()
        self.config = config
        d = config.d_model

        # --- history path (built when used directly or by the cross-attn head) ---
        if config.use_history or config.xattn:
            self.tok_emb = nn.Embedding(config.n_tokens, d, padding_idx=TOK_PAD)
            self.pos_emb = nn.Embedding(config.max_len, d)
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config.num_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            self.register_buffer("_positions", torch.arange(config.max_len), persistent=False)
            if config.letter_count:  # 0 = none/pad; 1..5 = times this letter appears in its guess
                self.count_emb = nn.Embedding(6, d)

        # --- history-only cross-attention head: no candidate path at all ---
        if config.xattn:
            self.word_head = CrossAttnWordHead(d, config.n_words, config.nhead, word_letters)
            return

        # --- candidate-set path (always, unless xattn) ---
        self.cand_proj = nn.Sequential(
            nn.Linear(config.cand_dim, d), nn.GELU(), nn.Linear(d, d)
        )

        # --- output head ---
        fusion_dim = d * (2 if config.use_history else 1)
        if config.factored_head:
            # Each word is scored by <state, sum of its (position, letter) embeddings>.
            self.letter_emb = nn.Embedding(WORD_LEN * 26, d)
            self.head_proj = nn.Linear(fusion_dim, d)
            self.word_bias = nn.Parameter(torch.zeros(config.n_words))
            idx = np.zeros((config.n_words, WORD_LEN), dtype=np.int64)
            if word_letters is not None:
                idx = (np.arange(WORD_LEN) * 26 + word_letters).astype(np.int64)
            self.register_buffer("word_letter_idx", torch.from_numpy(idx))
        else:
            self.head = nn.Linear(fusion_dim, config.n_words)

    def _word_embeddings(self) -> torch.Tensor:
        return self.letter_emb(self.word_letter_idx).sum(dim=1)  # (n_words, d)

    def _letter_counts(self, tokens: torch.Tensor) -> torch.Tensor:
        """For each history token, how many times its letter appears in its own guess
        (1..5; 0 for START/PAD). A function of the guess only — solver-free; it hands
        the net the duplicate-letter bookkeeping that exact filtering hinges on."""
        length = tokens.size(1)
        lett = tokens // 3                                   # (B,L); 26 = START/PAD
        real = lett < 26
        grp = (self._positions[:length] - 1) // 5            # guess index per slot (-1 for START)
        same_grp = grp[:, None] == grp[None, :]              # (L,L)
        same_let = lett[:, :, None] == lett[:, None, :]       # (B,L,L)
        cnt = (same_grp[None] & same_let & real[:, None, :]).sum(-1)  # (B,L)
        return (cnt.clamp(max=5) * real).long()

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        pos = self._positions[: tokens.size(1)]
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        if self.config.letter_count:
            x = x + self.count_emb(self._letter_counts(tokens))
        return x

    def forward(self, tokens, key_padding_mask, cand_feats) -> torch.Tensor:
        """tokens (B,L) long; mask (B,L) bool True=pad; cand_feats (B,156) -> (B,n_words).

        In ``xattn`` mode ``cand_feats`` is ignored — the policy reads only the history.
        """
        if self.config.xattn:
            hist = self.encoder(self._embed(tokens), src_key_padding_mask=key_padding_mask)
            return self.word_head(hist, key_padding_mask)

        parts = []
        if self.config.use_history:
            x = self.encoder(self._embed(tokens), src_key_padding_mask=key_padding_mask)
            parts.append(self.norm(x[:, 0]))  # START summary
        parts.append(self.cand_proj(cand_feats))
        z = torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

        if self.config.factored_head:
            state = self.head_proj(z)
            return state @ self._word_embeddings().t() + self.word_bias
        return self.head(z)


def save_checkpoint(path, model: WordlePolicy, vocab_words: list[str]) -> None:
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "vocab": vocab_words,
        },
        path,
    )


def load_checkpoint(path, map_location="cpu") -> tuple[WordlePolicy, list[str]]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = WordlePolicy(PolicyConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["vocab"]
