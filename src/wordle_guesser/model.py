"""The transformer policy: reads encoded game history, predicts the next guess."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .encoding import CAND_DIM, MAX_LEN, N_TOKENS, TOK_PAD


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


class WordlePolicy(nn.Module):
    """Small Transformer encoder over (letter, color) history tokens.

    The ``START`` token at position 0 attends over the whole history and its
    final hidden state is projected to a distribution over the word vocabulary.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.n_tokens, config.d_model, padding_idx=TOK_PAD)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, config.num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        # Candidate-set summary path: project the 130-d frequency vector and fuse
        # it with the history summary before the word head.
        self.cand_proj = nn.Sequential(
            nn.Linear(config.cand_dim, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.head = nn.Linear(2 * config.d_model, config.n_words)
        self.register_buffer("_positions", torch.arange(config.max_len), persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
        cand_feats: torch.Tensor,
    ) -> torch.Tensor:
        """tokens: (B, L) long; mask: (B, L) bool (True=pad); cand_feats: (B, 130).

        Returns (B, n_words) logits.
        """
        pos = self._positions[: tokens.size(1)]
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        history = self.norm(x[:, 0])  # START position summary
        candidates = self.cand_proj(cand_feats)
        return self.head(torch.cat([history, candidates], dim=-1))


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
