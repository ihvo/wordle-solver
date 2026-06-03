"""Raw-GRPO polish for the history-only (xattn) net.

DAgger + the duplicate-letter feature take the solver-free net to ~99.9% raw, but
the last 1-2 losses are brittle endgames where imitation (and val-acc) stop
helping. This optimises the *win objective* directly: the policy samples raw at
every turn (turn 1 forced opener), reward = ``solved ? (7 - turns) : 0``, GRPO
group baseline. The full-pool hybrid expert is seeded into each group so the hard
answers have a winning trajectory (non-zero advantage where every sample loses);
a KL anchor to the BC checkpoint keeps the ~2300 easy games from drifting. Selects
on raw win rate. Reuses ``rl.ppo_update`` / anchor / demos.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import TOK_PAD, candidate_features, encode_state, pad_batch, pad_to
from .evaluate import evaluate_model, summarize
from .model import load_checkpoint, save_checkpoint
from .rl import answer_weights, collect_anchor_states, ppo_update, precompute_teacher_demos, reward_for
from .solver import MAX_GUESSES, load_pattern_matrix
from .train import MODELS_DIR, pick_device
from .words import load_vocabulary


@torch.no_grad()
def raw_rollouts(model, p, letters, n, answers, opener_idx, device):
    """One raw (unmasked, sampled-every-turn) game per answer; record every step."""
    all_words = np.arange(n)
    ng = len(answers)
    guesses = [[opener_idx] for _ in range(ng)]
    codes = [[int(p[opener_idx, answers[i]])] for i in range(ng)]
    cands = [all_words[p[opener_idx, all_words] == codes[i][0]] for i in range(ng)]
    solved = [1 if answers[i] == opener_idx else 0 for i in range(ng)]
    done = [s > 0 for s in solved]
    tok, mask, feat, act, logp, game = [], [], [], [], [], []

    for turn in range(2, MAX_GUESSES + 1):
        active = [i for i in range(ng) if not done[i]]
        if not active:
            break
        toks = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(toks)
        feats = np.stack([candidate_features(cands[i], letters) for i in active])  # ignored by xattn
        lp = F.log_softmax(model(torch.from_numpy(tokens).to(device), torch.from_numpy(kpm).to(device),
                                 torch.from_numpy(feats).to(device)).float(), dim=1)
        sampled = torch.multinomial(lp.exp(), 1).squeeze(1)
        chosen = lp.gather(1, sampled[:, None]).squeeze(1).cpu().numpy()
        sampled = sampled.cpu().numpy()
        for row, i in enumerate(active):
            a = int(sampled[row])
            tok.append(tokens[row].copy()); mask.append(kpm[row].copy()); feat.append(feats[row].copy())
            act.append(a); logp.append(float(chosen[row])); game.append(i)
            code = int(p[a, answers[i]])
            guesses[i].append(a); codes[i].append(code)
            cands[i] = cands[i][p[a, cands[i]] == code]
            if a == answers[i]:
                solved[i] = turn; done[i] = True
    reward = np.array([reward_for(s) for s in solved], dtype=np.float32)
    return (tok, mask, feat, act, logp, game), reward


def build_batch(model, p, vocab, demos, chosen, group, opener_idx, device):
    """Raw rollouts (group-1 per answer) + the full expert demo trajectory, GRPO advantages."""
    letters = vocab.letters
    n = len(vocab)
    n_sampled = group - 1
    answers = np.repeat(chosen, n_sampled)
    slot = np.repeat(np.arange(len(chosen)), n_sampled)
    (s_tok, s_mask, s_feat, s_act, s_logp, s_game), s_reward = raw_rollouts(
        model, p, letters, n, answers, opener_idx, device)

    baseline = np.zeros(len(chosen), np.float32)
    for g in range(len(chosen)):
        rs = s_reward[slot == g]
        baseline[g] = rs.mean() if len(rs) else 0.0

    tok, mask, feat = list(s_tok), list(s_mask), list(s_feat)
    act, old_logp = list(s_act), list(s_logp)
    adv = [float(s_reward[gi] - baseline[slot[gi]]) for gi in s_game]

    t_tok, t_mask, t_feat, t_act, t_adv = [], [], [], [], []
    for g, answer in enumerate(chosen):
        steps, t_reward = demos[int(answer)]
        a_demo = t_reward - baseline[g]
        for toks, feats, action, _stuck in steps:  # raw: every expert step, not just stuck ones
            padded = pad_to(toks)
            t_tok.append(padded); t_mask.append(padded == TOK_PAD); t_feat.append(feats)
            t_act.append(int(action)); t_adv.append(float(a_demo))

    if t_tok:
        tt = torch.from_numpy(np.stack(t_tok)).to(device)
        tm = torch.from_numpy(np.stack(t_mask)).to(device)
        tf = torch.from_numpy(np.stack(t_feat)).to(device)
        with torch.no_grad():
            tlp = F.log_softmax(model(tt, tm, tf).float(), dim=1)
            t_old = tlp.gather(1, torch.tensor(t_act, device=device)[:, None]).squeeze(1).cpu().numpy()
        tok += t_tok; mask += t_mask; feat += t_feat; act += t_act; old_logp += list(t_old); adv += t_adv

    if not tok:
        return None, s_reward
    batch = {
        "tokens": torch.from_numpy(np.stack(tok)).to(device),
        "mask": torch.from_numpy(np.stack(mask)).to(device),
        "feats": torch.from_numpy(np.stack(feat)).to(device),
        "actions": torch.tensor(act, dtype=torch.long, device=device),
        "old_logp": torch.tensor(old_logp, dtype=torch.float32, device=device),
        "adv": torch.tensor(adv, dtype=torch.float32, device=device),
    }
    return batch, s_reward


def eval_raw(model, vocab, p, device):
    turns = evaluate_model(model, vocab, p, device, mask_to_candidates=False)
    return summarize(turns), turns


def main() -> None:
    ap = argparse.ArgumentParser(description="Raw-GRPO polish for the history-only net.")
    ap.add_argument("--init", type=Path, default=MODELS_DIR / "policy_xattn.pt")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy_xattn.pt")
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--answers-per-batch", type=int, default=64)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--mb-size", type=int, default=512)
    ap.add_argument("--kl-coef", type=float, default=0.02)
    ap.add_argument("--anchor-kl-coef", type=float, default=0.5)
    ap.add_argument("--anchor-cap", type=int, default=6000)
    ap.add_argument("--ent-coef", type=float, default=3e-3)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--prio-tau", type=float, default=1.5)
    ap.add_argument("--uniform-mix", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    vocab = load_vocabulary()
    p = load_pattern_matrix(vocab)
    n = len(vocab)

    model, words = load_checkpoint(args.init, map_location=str(device))
    opener = getattr(model.config, "opener", None)
    opener_idx = vocab.index[opener]
    model.to(device).eval()
    ref = copy.deepcopy(model).to(device).eval()
    for pm in ref.parameters():
        pm.requires_grad_(False)

    print(f"device: {device}  opener: {opener.upper()}  (history-only raw-GRPO)")
    demos = precompute_teacher_demos(p, vocab, opener_idx)
    anchor = collect_anchor_states(ref, p, vocab, opener_idx, device, args.anchor_cap, rng)
    print(f"  anchored on {anchor['tokens'].shape[0]} normal states")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    m0, turns = eval_raw(model, vocab, p, device)
    print(f"init  raw win {m0['win_rate']*100:.2f}%  avg {m0['avg_guesses']:.4f}  losses {m0['losses']}")
    best_key = (m0["win_rate"], -m0["avg_guesses"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
    last = {"loss": 0.0, "kl": 0.0, "ent": 0.0, "anchor_kl": 0.0, "clipfrac": 0.0}

    for update in range(1, args.updates + 1):
        chosen = rng.choice(n, size=args.answers_per_batch, p=weights, replace=True)
        batch, s_reward = build_batch(model, p, vocab, demos, chosen, args.group, opener_idx, device)
        if batch is not None:
            last = ppo_update(model, ref, batch, opt, clip=args.clip, kl_coef=args.kl_coef,
                              ent_coef=args.ent_coef, epochs=args.ppo_epochs, mb_size=args.mb_size,
                              max_norm=args.max_norm, anchor=anchor, anchor_kl_coef=args.anchor_kl_coef)
        if update % args.eval_every == 0 or update == args.updates:
            m, turns = eval_raw(model, vocab, p, device)
            weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
            key = (m["win_rate"], -m["avg_guesses"])
            flag = ""
            if key > best_key:
                best_key = key
                save_checkpoint(args.out, model, vocab.words)
                flag = "  <- saved"
            print(f"upd {update:4d}  R {s_reward.mean():.3f}  kl {last['kl']:.3f} aKL {last['anchor_kl']:.4f} "
                  f"ent {last['ent']:.2f}  | raw {m['win_rate']*100:.2f}%  avg {m['avg_guesses']:.4f}  losses {m['losses']}{flag}")

    print(f"best raw win {best_key[0]*100:.2f}%  avg {-best_key[1]:.4f}; saved to {args.out}")


if __name__ == "__main__":
    main()
