"""Generative-head experiment: emit the guess letter-by-letter instead of a
classifier over the word vocabulary.

Trains a candidate-feature net whose head is a `LetterDecoder` (a GRU seeded by
the fused state, 5 autoregressive letter steps). Teacher-forced cross-entropy on
the best-candidate target's letters. Evaluated by *generating* the guess token by
token — UNCONSTRAINED, so it can emit non-words; we report the invalid-word rate
alongside win rate. The question: can a generative head learn to write valid words
and play well, vs. the classifier that can only ever emit real words?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .encoding import candidate_features, encode_state, pad_batch
from .model import PolicyConfig, WordlePolicy, save_checkpoint
from .solver import MAX_GUESSES, load_pattern_matrix
from .train import MODELS_DIR, load_dataset, pick_device
from .words import DATA_DIR, load_vocabulary


@torch.no_grad()
def eval_decoder(model, vocab, p, device, sample=False, max_guesses=MAX_GUESSES):
    """Play all answers, generating each guess letter-by-letter (unconstrained).

    A generated string not in the vocabulary is an *invalid* (illegal) guess → the
    game is lost. Returns win/avg/losses plus invalid-generation stats.
    """
    n = len(vocab)
    letters = vocab.letters
    opener = getattr(model.config, "opener", None)
    opener_idx = vocab.index[opener] if opener else None
    guesses = [[] for _ in range(n)]
    codes = [[] for _ in range(n)]
    cands = [np.arange(n) for _ in range(n)]
    turns = np.zeros(n, dtype=np.int32)
    active = list(range(n))
    gen_total = gen_invalid = 0
    games_invalid = set()

    for turn in range(max_guesses):
        if not active:
            break
        tok_arrays = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(tok_arrays)
        feats = np.stack([candidate_features(cands[i], letters) for i in active])
        if opener_idx is not None and turn == 0:
            chosen = [opener_idx] * len(active)  # opener forced; decoder writes turns 2..6
        else:
            gen = model.generate(torch.from_numpy(tokens).to(device), torch.from_numpy(kpm).to(device),
                                 torch.from_numpy(feats).to(device), sample=sample).cpu().numpy()
            chosen = []
            for row in gen:
                gen_total += 1
                word = "".join(chr(97 + int(c)) for c in row)
                idx = vocab.get(word)
                chosen.append(idx)
                if idx is None:
                    gen_invalid += 1
        still = []
        for row, i in enumerate(active):
            g = chosen[row]
            if g is None:                       # invalid (non-word) → illegal → lost
                games_invalid.add(i)
                continue
            code = int(p[g, i])
            guesses[i].append(g)
            codes[i].append(code)
            cands[i] = cands[i][p[g, cands[i]] == code]
            if g == i:
                turns[i] = turn + 1
            else:
                still.append(i)
        active = still

    won = turns > 0
    summary = {
        "win_rate": float(won.mean()),
        "avg_guesses": float(turns[won].mean()) if won.any() else float("nan"),
        "losses": int((~won).sum()),
        "invalid_rate": gen_invalid / max(gen_total, 1),
        "games_with_invalid": len(games_invalid),
    }
    return summary, turns


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + eval the generative (letter-decoder) head.")
    ap.add_argument("--data", type=Path, default=DATA_DIR / "bc_dataset.npz")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy_decoder.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--opener", default="slate")
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)

    tokens, mask, feats, tgt_idx, _tw = load_dataset(args.data)
    target_word = tgt_idx[:, 0]                                   # best-candidate target
    target_letters = torch.from_numpy(vocab.letters[target_word.numpy()].astype(np.int64))  # (M,5)
    print(f"dataset: {tokens.size(0)} states, feat dim {feats.size(1)}")

    ds = TensorDataset(tokens, mask, feats, target_letters)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    cfg = PolicyConfig(n_words=len(vocab), cand_dim=feats.size(1), use_history=True,
                       decoder=True, opener=args.opener.lower())
    model = WordlePolicy(cfg, word_letters=vocab.letters).to(device)
    print(f"model: {sum(pp.numel() for pp in model.parameters())/1e6:.3f}M params (decoder head)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_key = (-1.0, 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for tok, msk, ft, tl in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            tok, msk, ft, tl = tok.to(device), msk.to(device), ft.to(device), tl.to(device)
            logits = model(tok, msk, ft, target_letters=tl)         # (B,5,26) teacher-forced
            loss = F.cross_entropy(logits.reshape(-1, 26), tl.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * tok.size(0)
        sched.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            m, _turns = eval_decoder(model, vocab, p, device)
            key = (m["win_rate"], -m["avg_guesses"])
            flag = ""
            if key > best_key:
                best_key = key
                save_checkpoint(args.out, model, vocab.words)
                flag = "  <- saved"
            print(f"epoch {epoch:2d}  loss {running/tokens.size(0):.4f}  "
                  f"win {m['win_rate']*100:6.2f}%  avg {m['avg_guesses']:.3f}  losses {m['losses']:>3}  "
                  f"invalid {m['invalid_rate']*100:.2f}% ({m['games_with_invalid']} games){flag}")

    print(f"best win {best_key[0]*100:.2f}% avg {-best_key[1]:.3f}; saved to {args.out}")


if __name__ == "__main__":
    main()
