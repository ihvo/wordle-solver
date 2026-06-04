"""Route B: clean categorical RL over the word-seed decoder's word head.

The word-seed model decides *which* word to play via a cross-attention word head, then spells it.
Earlier RL sampled letters (straight-through) → noisy gradients that oscillated. Here we RL the
**word** directly: sample a word from the head's categorical (always a real word — no invalid
losses), GRPO on the head's word log-prob, reward = ``solved ? (7-turns) : 0``. A detached spelling
CE keeps the decoder transcribing the head's current pick without contaminating the head's gradient.
The generative metric (spelling through the decoder) is measured with ``decoder.eval_decoder``; we
also report the head's own (argmax-word) play as the candidacy ceiling.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .decoder import eval_decoder
from .encoding import encode_state, pad_batch
from .model import load_checkpoint, save_checkpoint
from .rl import answer_weights, reward_for
from .solver import MAX_GUESSES, load_pattern_matrix
from .train import MODELS_DIR, pick_device
from .words import load_vocabulary


@torch.no_grad()
def head_eval(model, p, vocab, device, max_guesses=MAX_GUESSES):
    """Play every answer by the head's argmax word (no decoder) — the candidacy ceiling."""
    n = len(vocab); letters = vocab.letters
    opener = vocab.index[model.config.opener]
    guesses = [[opener] for _ in range(n)]
    codes = [[int(p[opener, i])] for i in range(n)]
    turns = np.zeros(n, np.int32)
    turns[np.arange(n) == opener] = 1
    active = [i for i in range(n) if i != opener]
    for turn in range(2, max_guesses + 1):
        if not active:
            break
        toks = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(toks)
        words = model.head_logits(torch.from_numpy(tokens).to(device),
                                  torch.from_numpy(kpm).to(device)).argmax(-1).cpu().numpy()
        still = []
        for row, i in enumerate(active):
            w = int(words[row]); guesses[i].append(w); codes[i].append(int(p[w, i]))
            if w == i:
                turns[i] = turn
            else:
                still.append(i)
        active = still
    won = turns > 0
    return float(won.mean()), float(turns[won].mean())


@torch.no_grad()
def rollouts(model, p, vocab, answers, opener_idx, device, temp=1.0):
    """Sample one game per answer by sampling a *word* from the head each turn (turn 1 = opener)."""
    letters = vocab.letters; ng = len(answers)
    guesses = [[opener_idx] for _ in range(ng)]
    codes = [[int(p[opener_idx, answers[i]])] for i in range(ng)]
    solved = [1 if answers[i] == opener_idx else 0 for i in range(ng)]
    done = [s > 0 for s in solved]
    rec_tok, rec_mask, rec_word, rec_logp, rec_game = [], [], [], [], []
    for turn in range(2, MAX_GUESSES + 1):
        active = [i for i in range(ng) if not done[i]]
        if not active:
            break
        toks = [encode_state(guesses[i], codes[i], letters) for i in active]
        tokens, kpm = pad_batch(toks)
        hl = model.head_logits(torch.from_numpy(tokens).to(device), torch.from_numpy(kpm).to(device))
        lp = (hl / temp).log_softmax(-1)
        words = torch.multinomial(lp.exp(), 1).squeeze(1)
        wlogp = lp.gather(1, words[:, None]).squeeze(1)
        words = words.cpu().numpy(); wlogp = wlogp.cpu().numpy()
        for row, i in enumerate(active):
            w = int(words[row])
            rec_tok.append(tokens[row].copy()); rec_mask.append(kpm[row].copy())
            rec_word.append(w); rec_logp.append(float(wlogp[row])); rec_game.append(i)
            guesses[i].append(w); codes[i].append(int(p[w, answers[i]]))
            if w == answers[i]:
                solved[i] = turn; done[i] = True
    reward = np.array([reward_for(solved[i]) for i in range(ng)], dtype=np.float32)
    return (rec_tok, rec_mask, rec_word, rec_logp, rec_game), reward


def build_batch(model, p, vocab, chosen, group, opener_idx, device):
    answers = np.repeat(chosen, group)
    slot = np.repeat(np.arange(len(chosen)), group)
    (tok, mask, word, logp, game), reward = rollouts(model, p, vocab, answers, opener_idx, device)
    baseline = np.zeros(len(chosen), np.float32)
    for g in range(len(chosen)):
        rs = reward[slot == g]
        baseline[g] = rs.mean() if len(rs) else 0.0
    if not tok:
        return None, reward
    adv = [float(reward[gi] - baseline[slot[gi]]) for gi in game]
    return {
        "tokens": torch.from_numpy(np.stack(tok)).to(device),
        "mask": torch.from_numpy(np.stack(mask)).to(device),
        "word": torch.tensor(word, dtype=torch.long, device=device),
        "old_logp": torch.tensor(logp, dtype=torch.float32, device=device),
        "adv": torch.tensor(adv, dtype=torch.float32, device=device),
    }, reward


def update(model, ref, batch, opt, letters_t, clip, kl_coef, ent_coef, spell_coef, epochs, mb, max_norm):
    tok, msk, word, old_logp, adv = (batch["tokens"], batch["mask"], batch["word"],
                                     batch["old_logp"], batch["adv"])
    adv = adv / (adv.std() + 1e-8)
    with torch.no_grad():
        ref_lp = ref.head_logits(tok, msk).log_softmax(-1)
    n = word.shape[0]
    st = {"loss": 0.0, "kl": 0.0, "ent": 0.0, "spell": 0.0, "nb": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=tok.device)
        for s in range(0, n, mb):
            b = perm[s:s + mb]
            hl = model.head_logits(tok[b], msk[b])
            lp = hl.log_softmax(-1)
            chosen = lp.gather(1, word[b][:, None]).squeeze(1)
            ratio = torch.exp(chosen - old_logp[b]); a = adv[b]
            surr = torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            probs = lp.exp()
            ent = -(probs * lp).sum(-1).mean()
            kl = (probs * (lp - ref_lp[b])).sum(-1).mean()
            tgt = letters_t[hl.argmax(-1)]                              # spell the head's pick
            spell = F.cross_entropy(model.spell_logits(tok[b], msk[b], tgt).reshape(-1, 26), tgt.reshape(-1))
            loss = -surr + kl_coef * kl - ent_coef * ent + spell_coef * spell
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm); opt.step()
            st["loss"] += loss.item(); st["kl"] += kl.item(); st["ent"] += ent.item()
            st["spell"] += spell.item(); st["nb"] += 1
    nb = max(st.pop("nb"), 1)
    return {k: v / nb for k, v in st.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Route B: categorical RL over the word-seed head.")
    ap.add_argument("--init", type=Path, default=MODELS_DIR / "policy_word_seed.pt")
    ap.add_argument("--out", type=Path, default=MODELS_DIR / "policy_word_seed.pt")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--answers-per-batch", type=int, default=96)
    ap.add_argument("--group", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ppo-epochs", type=int, default=2)
    ap.add_argument("--mb-size", type=int, default=1024)
    ap.add_argument("--kl-coef", type=float, default=0.1)
    ap.add_argument("--ent-coef", type=float, default=3e-3)
    ap.add_argument("--spell-coef", type=float, default=1.0)
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
    vocab = load_vocabulary(); p = load_pattern_matrix(vocab); n = len(vocab)
    letters_t = torch.from_numpy(vocab.letters.astype(np.int64)).to(device)

    model, _ = load_checkpoint(args.init, map_location=str(device))
    # Route B: the head picks the word (incl. non-candidate probes); the decoder must spell ANY pick,
    # so the consistency mask (which bans probe letters) must be off — the spelling loss retrains the
    # speller to stand on its own.
    model.config.decoder_hard_mask = False
    opener_idx = vocab.index[model.config.opener]
    model.to(device).eval()
    ref = copy.deepcopy(model).to(device).eval()
    for pm in ref.parameters():
        pm.requires_grad_(False)

    print(f"device: {device}  (word-seed categorical RL / route B)")
    m0, turns = eval_decoder(model, vocab, p, device)
    h0, havg = head_eval(model, p, vocab, device)
    print(f"init  spelled win {m0['win_rate']*100:.2f}% avg {m0['avg_guesses']:.3f} invalid {m0['invalid_rate']*100:.2f}%"
          f"  | head win {h0*100:.2f}% avg {havg:.3f}")
    best_key = (m0["win_rate"], -m0["avg_guesses"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
    last = {"kl": 0.0, "ent": 0.0, "spell": 0.0}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for u in range(1, args.updates + 1):
        chosen = rng.choice(n, size=args.answers_per_batch, p=weights, replace=True)
        batch, s_reward = build_batch(model, p, vocab, chosen, args.group, opener_idx, device)
        if batch is not None:
            last = update(model, ref, batch, opt, letters_t, clip=args.clip, kl_coef=args.kl_coef,
                          ent_coef=args.ent_coef, spell_coef=args.spell_coef, epochs=args.ppo_epochs,
                          mb=args.mb_size, max_norm=args.max_norm)
        if u % args.eval_every == 0 or u == args.updates:
            m, turns = eval_decoder(model, vocab, p, device)
            h, havg = head_eval(model, p, vocab, device)
            weights = answer_weights(turns, args.prio_tau, args.uniform_mix)
            key = (m["win_rate"], -m["avg_guesses"]); flag = ""
            if key > best_key:
                best_key = key; save_checkpoint(args.out, model, vocab.words); flag = "  <- saved"
            print(f"upd {u:4d}  R {s_reward.mean():.2f} kl {last['kl']:.3f} ent {last['ent']:.2f} spell {last['spell']:.3f}"
                  f"  | spelled {m['win_rate']*100:.2f}% avg {m['avg_guesses']:.3f} inv {m['invalid_rate']*100:.2f}%"
                  f"  head {h*100:.2f}%{flag}")

    print(f"best spelled win {best_key[0]*100:.2f}% avg {-best_key[1]:.3f}; saved to {args.out}")


if __name__ == "__main__":
    main()
