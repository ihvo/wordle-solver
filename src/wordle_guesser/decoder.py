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

from .encoding import WORD_LEN, candidate_features, encode_state, pad_batch
from .model import PolicyConfig, WordlePolicy, save_checkpoint
from .solver import MAX_GUESSES, load_pattern_matrix
from .train import MODELS_DIR, load_dataset, pick_device
from .words import DATA_DIR, load_vocabulary


@torch.no_grad()
def eval_decoder(model, vocab, p, device, sample=False, beam=1, max_guesses=MAX_GUESSES):
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
            tt, mm, ff = (torch.from_numpy(tokens).to(device), torch.from_numpy(kpm).to(device),
                          torch.from_numpy(feats).to(device))
            gen = (model.generate_beam(tt, mm, ff, beam=beam) if beam > 1
                   else model.generate(tt, mm, ff, sample=sample)).cpu().numpy()
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


def lm_pretrain(model, vocab, device, epochs, lr=1e-3, bs=512):
    """#2: pretrain the decoder as a 5-letter-word language model — generate every real
    word from a single neutral state — so its prior is 'real words only' before conditioning."""
    n = len(vocab)
    letters = vocab.letters
    tok, kpm = pad_batch([encode_state([], [], letters)])         # neutral: empty history
    feats = candidate_features(np.arange(n), letters)[None]        # all candidates
    tok = torch.from_numpy(tok).to(device); kpm = torch.from_numpy(kpm).to(device)
    feats = torch.from_numpy(feats).to(device)
    target_all = torch.from_numpy(letters.astype(np.int64)).to(device)  # (N,5)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot = 0.0
        for s in range(0, n, bs):
            b = idx[s:s + bs]; k = len(b)
            tl = target_all[b]
            z = model._state(tok.expand(k, -1), kpm.expand(k, -1), feats.expand(k, -1))
            logits = model.dec(z, target=tl, marg=model._dec_marg(feats.expand(k, -1)))
            loss = F.cross_entropy(logits.reshape(-1, 26), tl.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * k
        print(f"  lm-pretrain epoch {ep+1}/{epochs}  loss {tot/n:.4f}")


def word_lm_pretrain(model, vocab, device, epochs, lr=1e-3, bs=512):
    """#2: pretrain the decoder's word-LM branch on the real-word manifold (all answers,
    uniform) so it's a good validity prior before it's mixed into generation."""
    letters = torch.from_numpy(vocab.letters.astype(np.int64)).to(device)  # (N,5)
    n = letters.size(0)
    lm_params = [p for nm, p in model.dec.named_parameters() if nm.startswith("lm_")]
    opt = torch.optim.AdamW(lm_params, lr=lr)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        tot = 0.0
        for s in range(0, n, bs):
            b = letters[idx[s:s + bs]]
            logits = model.dec.word_lm_logits(b)
            loss = F.cross_entropy(logits.reshape(-1, 26), b.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        print(f"  word-LM pretrain {ep+1}/{epochs}  loss {tot/n:.4f}")


@torch.no_grad()
def teacher_forced_eval(model, tok, msk, ft, tl, device, bs=2048):
    """Teacher-forced metrics on a held-out split: mean per-letter CE and the
    fraction of states whose 5 argmax letters all match the target word. This
    isolates the *readout* (can the state identify the target, given the true
    prefix) from exposure bias (the free-running eval below). With a learned
    marginal, also reports per-slot top-1 accuracy of the predicted marginal vs
    the solver's (does the encoder re-derive candidacy from tokens?)."""
    model.eval()
    learned = model.config.decoder_learned_marginal
    n = tok.size(0)
    tot_loss = 0.0
    exact = 0
    marg_hit = 0
    for s in range(0, n, bs):
        e = s + bs
        a = (tok[s:e].to(device), msk[s:e].to(device), ft[s:e].to(device))
        tgt = tl[s:e].to(device)
        if learned:
            logits, marg_logits = model(*a, target_letters=tgt, return_aux=True)
            true_marg = ft[s:e, : WORD_LEN * 26].reshape(-1, WORD_LEN, 26).to(device)
            marg_hit += (marg_logits.argmax(-1) == true_marg.argmax(-1)).sum().item()
        else:
            logits = model(*a, target_letters=tgt)                   # (b,5,26)
        tot_loss += F.cross_entropy(logits.reshape(-1, 26), tgt.reshape(-1), reduction="sum").item()
        exact += (logits.argmax(-1) == tgt).all(1).sum().item()
    marg_acc = marg_hit / (n * WORD_LEN) if learned else None
    return tot_loss / (n * 5), exact / n, marg_acc


def main() -> None:
    ap = argparse.ArgumentParser(description="Train + eval the generative (letter-decoder) head.")
    ap.add_argument("--data", type=Path, default=DATA_DIR / "bc_dataset.npz")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy_decoder.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--opener", default="slate")
    ap.add_argument("--marginal", action="store_true", help="#4: feed per-position candidate marginals into the decoder")
    ap.add_argument("--history-only", action="store_true", help="drop the candidate-feature MLP: state = encoder read-out only (solver-free)")
    ap.add_argument("--xattn", action="store_true", help="decoder cross-attends the encoded history each step (reads full sequence, not a pooled seed)")
    ap.add_argument("--learned-marginal", action="store_true", help="B: predict the candidate marginal from the encoder (aux-supervised) and feed it to the decoder (solver-free)")
    ap.add_argument("--marg-attn", action="store_true", help="B2: predict the marginal with per-slot queries cross-attending the history (not the pooled seed)")
    ap.add_argument("--aux-coef", type=float, default=1.0, help="weight of the marginal-prediction auxiliary loss")
    ap.add_argument("--word-lm", action="store_true", help="#2: mix a learned 5-letter word-LM prior into generation (product of experts; parametric, no trie)")
    ap.add_argument("--word-lm-alpha", type=float, default=1.0, help="weight of the word-LM logits in the product of experts")
    ap.add_argument("--word-lm-coef", type=float, default=1.0, help="weight of the word-LM manifold aux loss during BC")
    ap.add_argument("--word-lm-pretrain", type=int, default=8, help="epochs to pretrain the word-LM branch on the answer manifold before BC")
    ap.add_argument("--val-frac", type=float, default=0.0, help="held-out fraction for teacher-forced val metrics (0=off)")
    ap.add_argument("--lm-pretrain", type=int, default=0, help="#2: epochs of word-LM pretraining before BC (0=off)")
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)

    if args.history_only and args.marginal:
        ap.error("--history-only is incompatible with --marginal (marginals are candidate-derived)")
    if args.learned_marginal and args.marginal:
        ap.error("--learned-marginal is incompatible with --marginal (pick predicted or solver marginal)")
    if args.marg_attn and not args.learned_marginal:
        ap.error("--marg-attn only applies with --learned-marginal")

    tokens, mask, feats, tgt_idx, _tw = load_dataset(args.data)
    target_word = tgt_idx[:, 0]                                   # best-candidate target
    target_letters = torch.from_numpy(vocab.letters[target_word.numpy()].astype(np.int64))  # (M,5)
    print(f"dataset: {tokens.size(0)} states, feat dim {feats.size(1)}")

    # held-out split for teacher-forced val metrics (readout vs exposure-bias diagnosis)
    val = None
    if args.val_frac > 0:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(tokens.size(0), generator=g)
        n_val = int(tokens.size(0) * args.val_frac)
        vi, ti = perm[:n_val], perm[n_val:]
        val = (tokens[vi], mask[vi], feats[vi], target_letters[vi])
        tokens, mask, feats, target_letters = tokens[ti], mask[ti], feats[ti], target_letters[ti]
        print(f"split: {tokens.size(0)} train / {n_val} val")

    ds = TensorDataset(tokens, mask, feats, target_letters)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    cfg = PolicyConfig(n_words=len(vocab), cand_dim=feats.size(1), use_history=True,
                       use_candidates=not args.history_only,
                       decoder=True, decoder_marginal=args.marginal, decoder_xattn=args.xattn,
                       decoder_learned_marginal=args.learned_marginal, decoder_marg_attn=args.marg_attn,
                       decoder_word_lm=args.word_lm, decoder_word_lm_alpha=args.word_lm_alpha,
                       opener=args.opener.lower())
    model = WordlePolicy(cfg, word_letters=vocab.letters).to(device)
    parts = ["decoder"]
    if args.history_only:
        parts.append("history-only")
    if args.xattn:
        parts.append("xattn")
    if args.learned_marginal:
        parts.append("learned-marg" + ("(attn)" if args.marg_attn else ""))
    if args.marginal:
        parts.append("marginal")
    if args.word_lm:
        parts.append(f"word-LM(α={args.word_lm_alpha})")
    print(f"model: {sum(pp.numel() for pp in model.parameters())/1e6:.3f}M params ({', '.join(parts)} head)")
    if args.lm_pretrain:
        print(f"word-LM pretraining ({args.lm_pretrain} epochs) ...")
        lm_pretrain(model, vocab, device, args.lm_pretrain)
    if args.word_lm and args.word_lm_pretrain:
        print(f"word-LM branch pretraining ({args.word_lm_pretrain} epochs) ...")
        word_lm_pretrain(model, vocab, device, args.word_lm_pretrain)
    all_letters = torch.from_numpy(vocab.letters.astype(np.int64)).to(device) if args.word_lm else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best_key = (-1.0, 0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for tok, msk, ft, tl in tqdm(loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            tok, msk, ft, tl = tok.to(device), msk.to(device), ft.to(device), tl.to(device)
            if args.learned_marginal:
                logits, marg_logits = model(tok, msk, ft, target_letters=tl, return_aux=True)
                dec_loss = F.cross_entropy(logits.reshape(-1, 26), tl.reshape(-1))
                true_marg = ft[:, : WORD_LEN * 26].reshape(-1, WORD_LEN, 26)  # solver marginal target
                marg_loss = -(true_marg * F.log_softmax(marg_logits, dim=-1)).sum(-1).mean()
                loss = dec_loss + args.aux_coef * marg_loss
            else:
                logits = model(tok, msk, ft, target_letters=tl)     # (B,5,26) teacher-forced
                loss = F.cross_entropy(logits.reshape(-1, 26), tl.reshape(-1))
            if args.word_lm:  # keep the word-LM a sharp real-word manifold prior (uniform sample)
                wb = all_letters[torch.randint(0, all_letters.size(0), (512,), device=device)]
                lm_logits = model.dec.word_lm_logits(wb)
                loss = loss + args.word_lm_coef * F.cross_entropy(lm_logits.reshape(-1, 26), wb.reshape(-1))
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
            tf = ""
            if val is not None:
                vce, vacc, vmarg = teacher_forced_eval(model, *val, device)
                mstr = f" marg {vmarg*100:5.2f}%" if vmarg is not None else ""
                tf = f"| TF-val CE {vce:.3f} exact {vacc*100:5.2f}%{mstr}  "
            print(f"epoch {epoch:2d}  trainCE {running/tokens.size(0):.4f}  {tf}"
                  f"| free-run win {m['win_rate']*100:6.2f}%  avg {m['avg_guesses']:.3f}  "
                  f"invalid {m['invalid_rate']*100:.2f}%{flag}")

    print(f"best win {best_key[0]*100:.2f}% avg {-best_key[1]:.3f}; saved to {args.out}")


if __name__ == "__main__":
    main()
