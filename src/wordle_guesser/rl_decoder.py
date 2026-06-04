"""RL the generative decoder — the fair test of exposure bias.

Teacher-forced BC leaves the letter-decoder at ~33% win / ~41% invalid words,
because it never trains on its own (free-running) prefixes. Here we optimise the
free-running policy directly: roll out by *sampling* the guess letter-by-letter,
reward = ``solved ? (7-turns) : 0`` with an **invalid (non-word) generation = loss
and terminate**. The action is the 5-letter word; its log-prob is the sum of the
five letter log-probs, so it's word-level GRPO over a generative head. KL-anchored
to the BC decoder for stability. Question: how high can validity / win rate go once
the decoder is trained on what it actually emits?
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .decoder import eval_decoder
from .encoding import candidate_features, encode_state, pad_batch
from .model import load_checkpoint, save_checkpoint
from .rl import answer_weights, reward_for
from .solver import MAX_GUESSES, load_pattern_matrix
from .train import MODELS_DIR, pick_device
from .words import load_vocabulary


def _word(letters_row) -> str:
    return "".join(chr(97 + int(c)) for c in letters_row)


@torch.no_grad()
def rollouts(model, p, vocab, answers, opener_idx, device, temp=1.0):
    """Sample one game per answer; turn 1 forced opener, turns 2+ generated letter-by-letter.
    A non-word generation ends the game as a loss. Records (state, letters, word-logp) per step."""
    letters = vocab.letters
    n = len(vocab)
    all_words = np.arange(n)
    ng = len(answers)
    guesses = [[opener_idx] for _ in range(ng)]
    codes = [[int(p[opener_idx, answers[i]])] for i in range(ng)]
    cands = [all_words[p[opener_idx, all_words] == codes[i][0]] for i in range(ng)]
    solved = [1 if answers[i] == opener_idx else 0 for i in range(ng)]
    done = [s > 0 for s in solved]
    rec_tok, rec_mask, rec_feat, rec_lett, rec_logp, rec_game = [], [], [], [], [], []

    for turn in range(2, MAX_GUESSES + 1):
        active = [i for i in range(ng) if not done[i]]
        if not active:
            break
        tok_arrays = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(tok_arrays)
        feats = np.stack([candidate_features(cands[i], letters) for i in active])
        z = model._state(torch.from_numpy(tokens).to(device), torch.from_numpy(kpm).to(device),
                         torch.from_numpy(feats).to(device))
        gen, lps = model.dec.sample(z, temp=temp)            # (A,5), (A,5)
        gen = gen.cpu().numpy(); wlogp = lps.sum(1).cpu().numpy()
        for row, i in enumerate(active):
            rec_tok.append(tokens[row].copy()); rec_mask.append(kpm[row].copy()); rec_feat.append(feats[row].copy())
            rec_lett.append(gen[row].astype(np.int64)); rec_logp.append(float(wlogp[row])); rec_game.append(i)
            idx = vocab.get(_word(gen[row]))
            if idx is None:                                  # invalid word -> illegal -> lost
                done[i] = True
                continue
            code = int(p[idx, answers[i]])
            guesses[i].append(idx); codes[i].append(code)
            cands[i] = cands[i][p[idx, cands[i]] == code]
            if idx == answers[i]:
                solved[i] = turn; done[i] = True
    reward = np.array([reward_for(s) for s in solved], dtype=np.float32)
    return (rec_tok, rec_mask, rec_feat, rec_lett, rec_logp, rec_game), reward


def build_batch(model, p, vocab, chosen, group, opener_idx, device):
    n_sampled = group
    answers = np.repeat(chosen, n_sampled)
    slot = np.repeat(np.arange(len(chosen)), n_sampled)
    (tok, mask, feat, lett, logp, game), reward = rollouts(model, p, vocab, answers, opener_idx, device)
    baseline = np.zeros(len(chosen), np.float32)
    for g in range(len(chosen)):
        rs = reward[slot == g]
        baseline[g] = rs.mean() if len(rs) else 0.0
    adv = [float(reward[gi] - baseline[slot[gi]]) for gi in game]
    if not tok:
        return None, reward
    return {
        "tokens": torch.from_numpy(np.stack(tok)).to(device),
        "mask": torch.from_numpy(np.stack(mask)).to(device),
        "feats": torch.from_numpy(np.stack(feat)).to(device),
        "letters": torch.from_numpy(np.stack(lett)).to(device),
        "old_logp": torch.tensor(logp, dtype=torch.float32, device=device),
        "adv": torch.tensor(adv, dtype=torch.float32, device=device),
    }, reward


def update(model, ref, batch, opt, clip, kl_coef, ent_coef, epochs, mb_size, max_norm):
    tokens, mask, feats = batch["tokens"], batch["mask"], batch["feats"]
    lett, old_logp, adv = batch["letters"], batch["old_logp"], batch["adv"]
    adv = adv / (adv.std() + 1e-8)
    with torch.no_grad():
        ref_logp = F.log_softmax(ref(tokens, mask, feats, target_letters=lett).float(), -1)  # (N,5,26)
    n = lett.shape[0]
    st = {"loss": 0.0, "kl": 0.0, "ent": 0.0, "clipfrac": 0.0, "nb": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=tokens.device)
        for s in range(0, n, mb_size):
            mb = perm[s:s + mb_size]
            logp_full = F.log_softmax(model(tokens[mb], mask[mb], feats[mb], target_letters=lett[mb]).float(), -1)
            word_logp = logp_full.gather(-1, lett[mb][..., None]).squeeze(-1).sum(1)  # (b,)
            ratio = torch.exp(word_logp - old_logp[mb])
            a = adv[mb]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            probs = logp_full.exp()
            ent = -(probs * logp_full).sum(-1).mean()
            kl = (probs * (logp_full - ref_logp[mb])).sum(-1).mean()
            loss = -surr + kl_coef * kl - ent_coef * ent
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm); opt.step()
            st["loss"] += loss.item(); st["kl"] += kl.item(); st["ent"] += ent.item()
            st["clipfrac"] += ((ratio - 1).abs() > clip).float().mean().item(); st["nb"] += 1
    nb = max(st.pop("nb"), 1)
    return {k: v / nb for k, v in st.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="RL the generative decoder (word-level GRPO over letters).")
    ap.add_argument("--init", type=Path, default=MODELS_DIR / "policy_decoder.pt")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy_decoder.pt")
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--answers-per-batch", type=int, default=96)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--mb-size", type=int, default=1024)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=3e-3)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--prio-tau", type=float, default=2.0)
    ap.add_argument("--uniform-mix", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)
    n = len(vocab)

    model, _ = load_checkpoint(args.init, map_location=str(device))
    opener_idx = vocab.index[model.config.opener]
    model.to(device).eval()
    ref = copy.deepcopy(model).to(device).eval()
    for pm in ref.parameters():
        pm.requires_grad_(False)

    print(f"device: {device}  (generative-decoder GRPO)")
    m0, turns = eval_decoder(model, vocab, p, device)
    print(f"init  win {m0['win_rate']*100:.2f}%  avg {m0['avg_guesses']:.3f}  invalid {m0['invalid_rate']*100:.2f}%")
    best_key = (m0["win_rate"], -m0["avg_guesses"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
    last = {"loss": 0.0, "kl": 0.0, "ent": 0.0, "clipfrac": 0.0}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for u in range(1, args.updates + 1):
        chosen = rng.choice(n, size=args.answers_per_batch, p=weights, replace=True)
        batch, s_reward = build_batch(model, p, vocab, chosen, args.group, opener_idx, device)
        if batch is not None:
            last = update(model, ref, batch, opt, clip=args.clip, kl_coef=args.kl_coef,
                          ent_coef=args.ent_coef, epochs=args.ppo_epochs, mb_size=args.mb_size,
                          max_norm=args.max_norm)
        if u % args.eval_every == 0 or u == args.updates:
            m, turns = eval_decoder(model, vocab, p, device)
            weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
            key = (m["win_rate"], -m["avg_guesses"])
            flag = ""
            if key > best_key:
                best_key = key; save_checkpoint(args.out, model, vocab.words); flag = "  <- saved"
            print(f"upd {u:4d}  R {s_reward.mean():.2f}  kl {last['kl']:.3f} ent {last['ent']:.2f}  | "
                  f"win {m['win_rate']*100:.2f}%  avg {m['avg_guesses']:.3f}  invalid {m['invalid_rate']*100:.2f}%{flag}")

    print(f"best win {best_key[0]*100:.2f}% avg {-best_key[1]:.3f}; saved to {args.out}")


if __name__ == "__main__":
    main()
