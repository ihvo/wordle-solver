"""RL post-training: teach the policy to *probe* by playing for the win.

The behavior-cloned policy is already at the candidate-masked ceiling (~99.7%):
when it stays inside the candidate set it cannot win the neighbour-trap answers
(``-ound``, ``-atch``, ``-aunt`` ...) — too many one-letter-apart candidates
remain for the guess budget. Those wins require *probing*: spending a guess on a
non-candidate word that splits the survivors. The full-pool entropy teacher does
this and wins 100%, but probing could not be distilled by behavior cloning (a
per-state soft target can't rank a move whose whole value is a multi-step setup).

So we optimise the real objective instead. The policy plays **unmasked** over the
2315-word head (turn 1 forced to the configured opener, no gradient) and is
rewarded only for the outcome::

    R = (MAX_GUESSES + 1 - turns)  if solved   else   0

Every win beats every loss, so win rate dominates; among wins, fewer turns scores
higher, so it never probes gratuitously. Training is GRPO with teacher-seeded
exploration: for each sampled answer we roll out ``group - 1`` stochastic games
plus one precomputed full-pool-teacher trajectory, baseline each group by the
mean reward of its *sampled* games, and take a PPO-clipped step. The teacher
trajectory guarantees a positive advantage on the trap states (where every
sampled game loses) so a gradient flows there at all; a KL anchor to the frozen
BC policy keeps the ~2300 already-won games from regressing.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import candidate_features, encode_state, pad_batch, pad_to, TOK_PAD
from .evaluate import evaluate_model, summarize
from .model import load_checkpoint, save_checkpoint
from .solver import MAX_GUESSES, best_guess, load_pattern_matrix
from .train import MODELS_DIR, pick_device
from .words import DATA_DIR, load_vocabulary


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
def reward_for(solved_turn: int) -> float:
    """``MAX_GUESSES + 1 - turns`` for a win (so win-in-2 = 5 ... win-in-6 = 1), 0 for a loss."""
    return float(MAX_GUESSES + 1 - solved_turn) if solved_turn > 0 else 0.0


# --------------------------------------------------------------------------- #
# Teacher demonstrations (computed once; independent of the policy)
# --------------------------------------------------------------------------- #
def precompute_teacher_demos(
    p: np.ndarray, vocab, opener_idx: int, answers=None
) -> dict[int, tuple[list, float]]:
    """For each answer, the full-pool teacher's post-opener trajectory.

    Returns ``answer -> ([(tokens, feats, action, stuck), ...], reward)``. The
    opening is forced (shared by every policy) so the demo starts at turn 2; each
    recorded step is the state the teacher faced and the (possibly non-candidate)
    word it probed with. ``stuck`` marks steps where candidates outnumber the
    remaining guesses — the decisions the safety rail can't make, i.e. the ones we
    actually want to imitate. ``answers`` restricts the set (default: all).
    """
    letters = vocab.letters
    n = len(vocab)
    all_words = np.arange(n)
    demos: dict[int, tuple[list, float]] = {}
    for answer in range(n) if answers is None else answers:
        answer = int(answer)
        code0 = int(p[opener_idx, answer])
        guesses = [opener_idx]
        codes = [code0]
        cands = all_words[p[opener_idx, all_words] == code0]
        if opener_idx == answer:  # the opener itself is the answer
            demos[answer] = ([], reward_for(1))
            continue
        steps: list = []
        solved_turn = 0
        for turn in range(2, MAX_GUESSES + 1):
            action = int(cands[0]) if len(cands) == 1 else best_guess(p, cands, pool=all_words)
            stuck = len(cands) > MAX_GUESSES - len(guesses)  # can't enumerate -> a probe
            steps.append(
                (
                    encode_state(guesses, codes, letters),
                    candidate_features(cands, letters),
                    action,
                    stuck,
                )
            )
            code = int(p[action, answer])
            guesses.append(action)
            codes.append(code)
            cands = cands[p[action, cands] == code]
            if action == answer:
                solved_turn = turn
                break
        demos[answer] = (steps, reward_for(solved_turn))
    return demos


# --------------------------------------------------------------------------- #
# On-policy rollouts (unmasked, sampled)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_rollouts(model, p, letters, n, answers, opener_idx, device):
    """Play one game per entry of ``answers`` in lockstep under the safety rail.

    When candidates fit the remaining budget the move is forced to the best
    candidate (deterministic, *not* recorded — the rail handles it at deploy too).
    Only when *stuck* (candidates outnumber the remaining guesses) does the policy
    sample an unmasked action; those are the steps we record and learn from.
    Returns ``(records, reward)`` with ``reward[i]`` the terminal reward of game ``i``.
    """
    all_words = np.arange(n)
    ng = len(answers)
    guesses = [[opener_idx] for _ in range(ng)]
    codes = [[int(p[opener_idx, answers[i]])] for i in range(ng)]
    cands = [all_words[p[opener_idx, all_words] == codes[i][0]] for i in range(ng)]
    solved_turn = [1 if answers[i] == opener_idx else 0 for i in range(ng)]
    done = [st > 0 for st in solved_turn]

    rec_tok, rec_mask, rec_feat, rec_act, rec_logp, rec_game = [], [], [], [], [], []

    for turn in range(2, MAX_GUESSES + 1):
        active = [i for i in range(ng) if not done[i]]
        if not active:
            break
        tok_arrays = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(tok_arrays)
        feats = np.stack([candidate_features(cands[i], letters) for i in active])
        logits = model(
            torch.from_numpy(tokens).to(device),
            torch.from_numpy(kpm).to(device),
            torch.from_numpy(feats).to(device),
        )
        logp = F.log_softmax(logits.float(), dim=1)
        sampled = torch.multinomial(logp.exp(), 1).squeeze(1)
        sampled_logp = logp.gather(1, sampled[:, None]).squeeze(1).cpu().numpy()
        sampled = sampled.cpu().numpy()
        lg = logits.float().cpu().numpy()

        for row, i in enumerate(active):
            c = cands[i]
            stuck = len(c) > MAX_GUESSES - len(guesses[i])
            if stuck:
                a = int(sampled[row])
                rec_tok.append(tokens[row].copy())
                rec_mask.append(kpm[row].copy())
                rec_feat.append(feats[row].copy())
                rec_act.append(a)
                rec_logp.append(float(sampled_logp[row]))
                rec_game.append(i)
            else:  # safe to enumerate: rail forces the best candidate (no gradient)
                a = int(c[np.argmax(lg[row][c])])
            code = int(p[a, answers[i]])
            guesses[i].append(a)
            codes[i].append(code)
            cands[i] = cands[i][p[a, cands[i]] == code]
            if a == answers[i]:
                solved_turn[i] = turn
                done[i] = True

    reward = np.array([reward_for(st) for st in solved_turn], dtype=np.float32)
    records = (rec_tok, rec_mask, rec_feat, rec_act, rec_logp, rec_game)
    return records, reward


# --------------------------------------------------------------------------- #
# Build one training batch: sampled rollouts + teacher demos, GRPO advantages
# --------------------------------------------------------------------------- #
def build_batch(model, ref_unused, p, vocab, demos, chosen, group, opener_idx, device):
    """Roll out ``group`` games per chosen answer (1 teacher demo + rest sampled),
    baseline by sampled-group mean, and flatten to padded step tensors with advantages."""
    letters = vocab.letters
    n = len(vocab)
    n_sampled = group - 1
    answers = np.repeat(chosen, n_sampled)
    slot = np.repeat(np.arange(len(chosen)), n_sampled)  # group id per sampled game

    (s_tok, s_mask, s_feat, s_act, s_logp, s_game), s_reward = sample_rollouts(
        model, p, letters, n, answers, opener_idx, device
    )

    # GRPO baseline: mean reward of the sampled games within each group.
    baseline = np.zeros(len(chosen), dtype=np.float32)
    for g in range(len(chosen)):
        rs = s_reward[slot == g]
        baseline[g] = rs.mean() if len(rs) else 0.0

    # --- sampled steps ---
    tok = [t for t in s_tok]
    mask = [m for m in s_mask]
    feat = [f for f in s_feat]
    act = list(s_act)
    old_logp = list(s_logp)
    adv = [float(s_reward[gi] - baseline[slot[gi]]) for gi in s_game]

    # --- teacher demo steps (advantage vs the same group baseline) ---
    t_tok, t_mask, t_feat, t_act, t_adv = [], [], [], [], []
    for g, answer in enumerate(chosen):
        steps, t_reward = demos[int(answer)]
        a_demo = t_reward - baseline[g]
        for toks, feats, action, stuck in steps:
            if not stuck:  # the rail handles enumerable endgames; learn only the probes
                continue
            padded = pad_to(toks)
            t_tok.append(padded)
            t_mask.append(padded == TOK_PAD)
            t_feat.append(feats)
            t_act.append(int(action))
            t_adv.append(float(a_demo))

    if t_tok:  # old log-probs of the teacher actions under the *current* policy
        tt = torch.from_numpy(np.stack(t_tok)).to(device)
        tm = torch.from_numpy(np.stack(t_mask)).to(device)
        tf = torch.from_numpy(np.stack(t_feat)).to(device)
        with torch.no_grad():
            tlogp = F.log_softmax(model(tt, tm, tf).float(), dim=1)
            t_old = tlogp.gather(1, torch.tensor(t_act, device=device)[:, None]).squeeze(1).cpu().numpy()
        tok += t_tok
        mask += t_mask
        feat += t_feat
        act += t_act
        old_logp += list(t_old)
        adv += t_adv

    if not tok:  # a batch of only easy answers reaches no stuck state — nothing to learn
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


# --------------------------------------------------------------------------- #
# BC anchor: pin the policy's full distribution on *normal* (non-stuck) states
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_anchor_states(ref_model, p, vocab, opener_idx, device, cap, rng):
    """States the safety rail handles at deploy — every non-opening decision in
    ref (BC) masked self-play where candidates still fit the budget. Holding the
    policy's distribution near BC *here* keeps raw play from rotting while the
    gradient is busy reshaping the stuck-state probes. Returns padded tensors."""
    n = len(vocab)
    letters = vocab.letters
    guesses = [[] for _ in range(n)]
    codes = [[] for _ in range(n)]
    cands = [np.arange(n) for _ in range(n)]
    active = list(range(n))
    toks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    feats_all: list[np.ndarray] = []
    for turn in range(MAX_GUESSES):
        if not active:
            break
        tok_arrays = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(tok_arrays)
        feats = np.stack([candidate_features(cands[i], letters) for i in active])
        lg = ref_model(
            torch.from_numpy(tokens).to(device),
            torch.from_numpy(kpm).to(device),
            torch.from_numpy(feats).to(device),
        ).float().cpu().numpy()
        remaining = MAX_GUESSES - turn
        still = []
        for row, i in enumerate(active):
            c = cands[i]
            if opener_idx is not None and turn == 0:
                g = opener_idx
            else:
                if len(c) <= remaining:  # a non-stuck (rail) state worth anchoring
                    toks.append(tokens[row].copy())
                    masks.append(kpm[row].copy())
                    feats_all.append(feats[row].copy())
                g = int(c[np.argmax(lg[row][c])])
            code = int(p[g, i])
            guesses[i].append(g)
            codes[i].append(code)
            cands[i] = cands[i][p[g, cands[i]] == code]
            if g != i:
                still.append(i)
        active = still

    idx = np.arange(len(toks))
    if len(idx) > cap:
        idx = rng.choice(idx, size=cap, replace=False)
    tok = np.stack([toks[j] for j in idx])
    msk = np.stack([masks[j] for j in idx])
    ft = np.stack([feats_all[j] for j in idx])
    anchor = {
        "tokens": torch.from_numpy(tok).to(device),
        "mask": torch.from_numpy(msk).to(device),
        "feats": torch.from_numpy(ft).to(device),
    }
    with torch.no_grad():
        anchor["ref_logp"] = F.log_softmax(
            ref_model(anchor["tokens"], anchor["mask"], anchor["feats"]).float(), dim=1
        )
    return anchor


# --------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------- #
def ppo_update(
    model, ref_model, batch, opt, clip, kl_coef, ent_coef, epochs, mb_size, max_norm,
    anchor=None, anchor_kl_coef=0.0, anchor_mb=512,
):
    tokens, mask, feats = batch["tokens"], batch["mask"], batch["feats"]
    actions, old_logp, adv = batch["actions"], batch["old_logp"], batch["adv"]

    # Scale (don't centre) advantages: GRPO already centres per group, and the
    # teacher's positive sign on the traps must survive. Zero stays zero.
    adv = adv / (adv.std() + 1e-8)

    with torch.no_grad():
        ref_logp = F.log_softmax(ref_model(tokens, mask, feats).float(), dim=1)

    n_anchor = anchor["tokens"].shape[0] if anchor is not None else 0
    n = actions.shape[0]
    stats = {"loss": 0.0, "kl": 0.0, "ent": 0.0, "anchor_kl": 0.0, "clipfrac": 0.0, "nb": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=tokens.device)
        for s in range(0, n, mb_size):
            mb = perm[s : s + mb_size]
            logp_full = F.log_softmax(model(tokens[mb], mask[mb], feats[mb]).float(), dim=1)
            logp_a = logp_full.gather(1, actions[mb, None]).squeeze(1)
            ratio = torch.exp(logp_a - old_logp[mb])
            a = adv[mb]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            probs = logp_full.exp()
            ent = -(probs * logp_full).sum(1).mean()
            kl = (probs * (logp_full - ref_logp[mb])).sum(1).mean()
            loss = -surr + kl_coef * kl - ent_coef * ent

            anchor_kl = torch.zeros((), device=tokens.device)
            if anchor is not None and anchor_kl_coef > 0 and n_anchor:
                ai = torch.randint(n_anchor, (min(anchor_mb, n_anchor),), device=tokens.device)
                a_logp = F.log_softmax(model(anchor["tokens"][ai], anchor["mask"][ai], anchor["feats"][ai]).float(), dim=1)
                anchor_kl = (a_logp.exp() * (a_logp - anchor["ref_logp"][ai])).sum(1).mean()
                loss = loss + anchor_kl_coef * anchor_kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            opt.step()

            stats["loss"] += loss.item()
            stats["kl"] += kl.item()
            stats["ent"] += ent.item()
            stats["anchor_kl"] += float(anchor_kl.detach())
            stats["clipfrac"] += ((ratio - 1.0).abs() > clip).float().mean().item()
            stats["nb"] += 1
    nb = max(stats.pop("nb"), 1)
    return {k: v / nb for k, v in stats.items()}


# --------------------------------------------------------------------------- #
# Prioritised answer sampling
# --------------------------------------------------------------------------- #
def answer_weights(turns: np.ndarray, tau: float, uniform_mix: float) -> np.ndarray:
    """Up-weight answers the current policy loses or wins slowly.

    ``turns`` is the hybrid per-answer guess count (0 = loss). Weight is
    ``exp((Rmax - reward)/tau)`` (a loss is the heaviest), blended with a uniform
    floor so the easy ~2300 games stay represented and don't regress.
    """
    reward = np.where(turns > 0, MAX_GUESSES + 1 - turns, 0.0).astype(np.float64)
    w = np.exp((MAX_GUESSES - reward) / tau)
    w /= w.sum()
    u = np.full_like(w, 1.0 / len(w))
    mix = uniform_mix * u + (1 - uniform_mix) * w
    return mix / mix.sum()


# --------------------------------------------------------------------------- #
# Evaluation: the hybrid (deploy) policy — enumerate when safe, probe when stuck
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_hybrid(model, vocab, p, device) -> tuple[dict, np.ndarray]:
    """Play all answers under the safety rail (mask when candidates fit the budget,
    probe when stuck) — exactly how the policy deploys, so we select on it."""
    turns = evaluate_model(model, vocab, p, device, mask_to_candidates=True, probe_when_stuck=True)
    return summarize(turns), turns


def _fmt(tag, m):
    return (f"{tag} win {m['win_rate']*100:6.2f}%  avg {m['avg_guesses']:.4f}  losses {m['losses']:>2}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="RL post-train the Wordle policy to probe (GRPO + teacher demos).")
    ap.add_argument("--init", type=Path, default=MODELS_DIR / "policy.pt", help="warm-start checkpoint")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy_rl.pt")
    ap.add_argument("--updates", type=int, default=400)
    ap.add_argument("--answers-per-batch", type=int, default=64)
    ap.add_argument("--group", type=int, default=6, help="rollouts per answer (1 teacher demo + rest sampled)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--mb-size", type=int, default=512)
    ap.add_argument("--kl-coef", type=float, default=0.02, help="KL to BC on the stuck states in the batch")
    ap.add_argument("--anchor-kl-coef", type=float, default=0.5, help="KL to BC on frozen normal states (keeps raw play healthy)")
    ap.add_argument("--anchor-cap", type=int, default=6000)
    ap.add_argument("--anchor-mb", type=int, default=512)
    ap.add_argument("--ent-coef", type=float, default=3e-3)
    ap.add_argument("--max-norm", type=float, default=1.0)
    ap.add_argument("--prio-tau", type=float, default=2.0)
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
    if words != vocab.words:
        raise SystemExit("checkpoint vocabulary does not match data/answers.txt")
    opener = getattr(model.config, "opener", None)
    if opener is None:
        raise SystemExit("RL expects a fixed opener (config.opener); train with --opener first")
    opener_idx = vocab.index[opener]
    model.to(device).eval()  # eval() = no dropout, so the policy is well-defined; grads still flow

    ref_model = copy.deepcopy(model).to(device).eval()
    for pm in ref_model.parameters():
        pm.requires_grad_(False)

    print(f"device: {device}  opener: {opener.upper()}  answers={n}")
    print("precomputing teacher demos ...")
    demos = precompute_teacher_demos(p, vocab, opener_idx)
    t_losses = sum(1 for a in range(n) if demos[a][1] == 0.0)
    print(f"  teacher demos cover {n} answers, {t_losses} unwinnable (expected 0)")
    print("collecting BC anchor states ...")
    anchor = collect_anchor_states(ref_model, p, vocab, opener_idx, device, args.anchor_cap, rng)
    print(f"  anchored on {anchor['tokens'].shape[0]} normal states")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    m0, turns = eval_hybrid(model, vocab, p, device)
    raw0 = summarize(evaluate_model(model, vocab, p, device, mask_to_candidates=False))
    print(f"{_fmt('init  (hybrid)', m0)}   |   {_fmt('raw', raw0)}")
    best_key = (m0["win_rate"], -m0["avg_guesses"])
    args.out.parent.mkdir(parents=True, exist_ok=True)

    weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
    last_stats = {"loss": 0.0, "kl": 0.0, "ent": 0.0, "anchor_kl": 0.0, "clipfrac": 0.0}
    for update in range(1, args.updates + 1):
        chosen = rng.choice(n, size=args.answers_per_batch, p=weights, replace=True)
        batch, s_reward = build_batch(model, ref_model, p, vocab, demos, chosen, args.group, opener_idx, device)
        if batch is not None:
            last_stats = ppo_update(
                model, ref_model, batch, opt,
                clip=args.clip, kl_coef=args.kl_coef, ent_coef=args.ent_coef,
                epochs=args.ppo_epochs, mb_size=args.mb_size, max_norm=args.max_norm,
                anchor=anchor, anchor_kl_coef=args.anchor_kl_coef, anchor_mb=args.anchor_mb,
            )

        if update % args.eval_every == 0 or update == args.updates:
            m, turns = eval_hybrid(model, vocab, p, device)
            raw = summarize(evaluate_model(model, vocab, p, device, mask_to_candidates=False))
            weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
            key = (m["win_rate"], -m["avg_guesses"])
            flag = ""
            if key > best_key:
                best_key = key
                save_checkpoint(args.out, model, vocab.words)
                flag = "  <- saved"
            st = last_stats
            print(
                f"upd {update:4d}  R {s_reward.mean():.3f}  "
                f"kl {st['kl']:.3f} aKL {st['anchor_kl']:.4f} ent {st['ent']:.2f}  | "
                f"{_fmt('hybrid', m)}  raw {raw['win_rate']*100:5.1f}%{flag}"
            )

    final, _ = eval_hybrid(model, vocab, p, device)
    raw = summarize(evaluate_model(model, vocab, p, device, mask_to_candidates=False))
    print(f"\n{_fmt('final (hybrid)', final)}   |   {_fmt('raw', raw)}")
    print(f"best saved to {args.out}  (win {best_key[0]*100:.2f}%, avg {-best_key[1]:.4f})")


if __name__ == "__main__":
    main()
