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


class WordlePolicy(nn.Module):
    """Candidate-feature MLP + optional Transformer-over-history, into a word head."""

    def __init__(self, config: PolicyConfig, word_letters: np.ndarray | None = None):
        super().__init__()
        self.config = config
        d = config.d_model

        # --- history path (optional) ---
        if config.use_history:
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

        # --- candidate-set path (always) ---
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

    def forward(self, tokens, key_padding_mask, cand_feats) -> torch.Tensor:
        """tokens (B,L) long; mask (B,L) bool True=pad; cand_feats (B,156) -> (B,n_words)."""
        parts = []
        if self.config.use_history:
            pos = self._positions[: tokens.size(1)]
            x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
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
