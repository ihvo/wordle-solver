"""Behavior-cloning training loop for the transformer policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .encoding import TOK_PAD
from .model import PolicyConfig, WordlePolicy, save_checkpoint
from .words import DATA_DIR, load_vocabulary

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def pick_device(requested: str | None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_dataset(path: Path):
    data = np.load(path)
    tokens = torch.from_numpy(data["tokens"].astype(np.int64))
    feats = torch.from_numpy(data["feats"].astype(np.float32))
    tgt_idx = torch.from_numpy(data["tgt_idx"].astype(np.int64))  # (M, K), -1 = pad
    tgt_w = torch.from_numpy(data["tgt_w"].astype(np.float32))  # (M, K)
    mask = tokens == TOK_PAD
    return tokens, mask, feats, tgt_idx, tgt_w


def soft_ce(logits: torch.Tensor, tgt_idx: torch.Tensor, tgt_w: torch.Tensor) -> torch.Tensor:
    """KL/cross-entropy against a sparse soft label (top-K indices + weights).

    Pad slots carry weight 0, so the clamped gather is harmless for them.
    """
    logp = F.log_softmax(logits, dim=1)
    gathered = logp.gather(1, tgt_idx.clamp(min=0))  # (B, K)
    return -(tgt_w * gathered).sum(dim=1).mean()


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Top-1 agreement with the single best candidate (tgt_idx[:, 0])."""
    model.eval()
    correct = total = 0
    for tokens, mask, feats, tgt_idx, _ in loader:
        logits = model(tokens.to(device), mask.to(device), feats.to(device))
        correct += (logits.argmax(1).cpu() == tgt_idx[:, 0]).sum().item()
        total += tgt_idx.size(0)
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Wordle transformer policy.")
    ap.add_argument("--data", type=Path, default=DATA_DIR / "bc_dataset.npz")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ff", type=int, default=256, help="feed-forward dim")
    ap.add_argument("--no-history", action="store_true", help="drop the transformer; candidate features only")
    ap.add_argument("--factored-head", action="store_true", help="letter-factored word head")
    ap.add_argument("--xattn", action="store_true", help="history-only: words cross-attend the history, no candidate features")
    ap.add_argument("--letter-count", action="store_true", help="add a per-token letter-count-in-guess embedding (helps duplicates)")
    ap.add_argument("--select-play", action="store_true", help="checkpoint on raw game win-rate, not val top-1 accuracy")
    ap.add_argument("--opener", default=None, help="fixed turn-1 word, forced at inference (match the dataset's --opener)")
    ap.add_argument("--init", type=Path, default=None, help="warm-start from this checkpoint (same architecture)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    vocab = load_vocabulary()
    print(f"device: {device}")
    opener = args.opener.lower() if args.opener else None
    if opener and opener not in vocab.index:
        raise SystemExit(f"opener {opener!r} is not in the answer vocabulary")

    tokens, mask, feats, tgt_idx, tgt_w = load_dataset(args.data)
    n = tokens.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"dataset: {n} pairs  ->  train {len(train_idx)}, val {len(val_idx)}")

    def subset(idx):
        return TensorDataset(tokens[idx], mask[idx], feats[idx], tgt_idx[idx], tgt_w[idx])

    train_loader = DataLoader(subset(train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(subset(val_idx), batch_size=max(args.batch_size, 64))

    config = PolicyConfig(
        n_words=len(vocab),
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        cand_dim=feats.size(1),  # 156, or 170 with --state-aug features
        use_history=not args.no_history,
        factored_head=args.factored_head,
        xattn=args.xattn,
        letter_count=args.letter_count,
        opener=opener,
    )
    model = WordlePolicy(config, word_letters=vocab.letters).to(device)
    if args.init:  # warm-start (e.g. DAgger from the augmented BC checkpoint)
        ckpt = torch.load(args.init, map_location=str(device), weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"warm-started from {args.init}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # --select-play ranks checkpoints by actual raw game win-rate, not val top-1
    # accuracy (which diverges from play — the recurring gotcha). Lazy import to
    # avoid the train<->evaluate import cycle.
    play_p = None
    if args.select_play:
        from .evaluate import evaluate_model, summarize
        from .solver import load_pattern_matrix
        play_p = load_pattern_matrix(vocab)

    best_key = (-1.0, 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for tok, msk, ft, t_idx, t_w in bar:
            tok, msk, ft = tok.to(device), msk.to(device), ft.to(device)
            t_idx, t_w = t_idx.to(device), t_w.to(device)
            opt.zero_grad()
            loss = soft_ce(model(tok, msk, ft), t_idx, t_w)
            loss.backward()
            opt.step()
            running += loss.item() * tok.size(0)
            bar.set_postfix(loss=f"{loss.item():.3f}")
        sched.step()

        train_loss = running / len(train_idx)
        if args.select_play:  # select on raw win-rate, then fewer guesses, then losses
            m = summarize(evaluate_model(model, vocab, play_p, device, mask_to_candidates=False))
            key, metric = (m["win_rate"], -m["avg_guesses"]), f"raw {m['win_rate']*100:.2f}% / {m['losses']} loss"
        else:
            va = evaluate(model, val_loader, device)
            key, metric = (va, 0.0), f"val_acc {va:.4f}"
        flag = ""
        if key > best_key:
            best_key = key
            save_checkpoint(args.out, model, vocab.words)
            flag = "  <- saved"
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  {metric}{flag}")

    print(f"best {best_key}; checkpoint at {args.out}")


if __name__ == "__main__":
    main()
