# RL post-training: teaching a 1M-param net to probe its way to 100%

*Research log — 2026-06-02. Branch `slm-rl-win-rate`. Author: Ihar + Clawd.*

*Part 3 of 6. Previously: [entropy-teacher-and-cloned-policies](2026-06-01-entropy-teacher-and-cloned-policies.md) → [probing-and-openers](2026-06-02-probing-and-openers.md). Next: [deprecating-the-rail](2026-06-03-deprecating-the-rail.md).*

## TL;DR

The behavior-cloned (BC) Wordle policy was stuck at the candidate-masked ceiling
(**99.7%**, 7 losses). We post-trained the same 1.04M-param transformer with RL — GRPO on
the terminal win reward, teacher-demo exploration, a safety rail folded into the loop, and
a BC anchor — and reached **100% win / 0 losses / 3.481 avg** over all 2315 answers,
matching the entropy solver. The decisive insight: *probing is an RL problem, not a
labeling one.* The full result and the one dead-end (a silent raw-play collapse) are below.

| Policy (all open SLATE) | Win | Avg | Losses |
|---|---|---|---|
| BC, masked (old default) | 99.70% | 3.549 | 7 |
| **RL, hybrid (new default)** | **100.00%** | **3.481** | **0** |
| RL, raw / unmasked | 99.31% | 3.485 | 16 |
| Entropy teacher (full pool) | 100.00% | 3.4955 | 0 |

---

## 1. The question

> "Let's post-train the SLM with RL to improve the win rate."

The BC policy wins 99.7% of games. The teacher (greedy entropy over the full answer list)
wins 100%. Could RL close that gap by improving the net itself, rather than leaning harder
on the solver?

## 2. Diagnosis first — where do the losses actually live?

Before writing any RL, we played four policies over all 2315 answers to see the structure
of the gap. The result reframed the whole task:

| Policy | Losses | Note |
|---|---|---|
| Teacher, **full pool** (may probe) | **0** | 100% |
| Teacher, **masked to candidates** (optimal greedy, no probe) | **7** | the ceiling for *any* candidate-only policy |
| BC, masked | 7 | already there |
| BC, raw (unmasked) | 17 | picks bad non-candidates |

Two facts fell out:

1. **Masked play is already at its ceiling.** The BC net's 7 losses and the *optimal*
   masked teacher's 7 losses are the same set (±1). The earlier SLATE-opener change had
   already harvested the easy gains. So **RL has zero headroom while masked** — to improve
   win rate at all, the policy must go unmasked and *probe*.

2. **The whole game is ~7 trap states.** The survivors — `foyer, patch, pound, vaunt,
   waste, watch, wight` — are neighbour-traps (`-ound`, `-atch`, `-aunt`, `-aste`,
   `-ight`): too many one-letter-apart candidates remain for the guess budget. The
   full-pool teacher wins every one by spending a guess on a **probe** — a non-candidate
   word that splits the survivors. That probe *is* the entire 99.7 → 100 gap.

The action space we'd optimize over is the existing 2315-word head (probing with answer
words is already representable; raw play just picks badly). No architecture change needed.

## 3. Why RL, when behavior cloning already failed at this

A prior negative result (`probing-not-cheaply-learnable`) showed BC of full-pool probe
labels makes the net *worse* — it can't rank a probe from a single argmax label. RL has a
real shot where BC didn't, for one concrete reason:

> A probe's value is a **multi-step return** (spend a guess now to guarantee a win in the
> remaining budget). A per-state soft-CE label can't express that; a terminal win reward
> can. And RL lets the net find *whatever* probe works with its own features, rather than
> reproducing the teacher's specific pick.

## 4. The recipe

- **Reward** (terminal, unshaped): `R = solved ? (MAX_GUESSES + 1 − turns) : 0`. Every win
  (≥1) beats every loss (0), so win rate dominates; among wins, fewer turns scores higher,
  so the net never probes gratuitously. We deliberately added *no* intermediate shaping —
  shaping toward "shrink the candidate set fastest" is the greedy objective BC already
  maxes, and it would actively discourage probing.
- **Algorithm**: GRPO. Per answer, sample a group of rollouts; advantage = reward − group
  mean. No value network (fits the project's "small" ethos). Easy games where every rollout
  wins give ≈0 advantage → gradient *automatically concentrates on the high-variance trap
  states.*
- **Exploration**: teacher-seeded. Each group includes one precomputed full-pool-teacher
  trajectory. On a trap, every sampled rollout loses (R=0) and the demo wins → advantage
  ≠ 0, so a gradient flows where pure sampling would stall forever.
- **Warm start + turn 1**: initialize from `policy.pt`; turn 1 is forced to SLATE (no
  gradient). RL only ever touches turns 2–6.
- **PPO clip + KL anchor** to the frozen BC reference for stability.
- **Prioritized sampling**: up-weight answers the current policy loses or wins slowly.

## 5. The experimental log (three runs)

### Run 1 — pure unmasked training → the hybrid-rail discovery

Trained the policy to play fully unmasked. Raw win rate climbed 99.27% → **99.65%** (8
losses), avg 3.55 → 3.51. Good, but not 100%.

The insight came from *how we evaluated it*. There's a provable rule: **if candidates ≤
remaining guesses, you can't lose by just enumerating them** — probing is only ever needed
when candidates *outnumber* the budget. So we tested a **hybrid** policy: mask to
candidates when safe, unleash the RL-learned probe only when stuck.

| Policy | Win | Losses |
|---|---|---|
| RL, raw | 99.65% | 8 |
| **RL, hybrid** | **99.83%** | **4** (`foyer pound taste wound`) |
| BC, hybrid (control) | 99.65% | 8 |

The hybrid jumped to 99.83% — and BC+hybrid did *not* improve, proving it was the
RL-learned probing, not the rail, doing the work.

### Run 2 — fold the rail into the loop → a silent collapse

If the rail handles the easy states at deploy, we shouldn't waste RL capacity there. So we
folded the rail into training: enumerate deterministically when safe (no gradient), sample
and learn *only* the stuck-state probe decision; select checkpoints by the hybrid metric we
deploy. It worked — **99.91% / 2 losses** (`bound cover`).

But verifying the saved checkpoint exposed a bug:

| Mode | Win | Losses |
|---|---|---|
| RL, hybrid (deploy) | 99.91% | 2 |
| **RL, raw** | **94.56%** | **126** ⚠️ |

Raw play had *collapsed*. The cause: all gradient went to stuck states, and the KL anchor
was *also* only evaluated on those states — so the net's behavior on ordinary states had
nothing holding it to BC and rotted. The rail hid this completely at deploy (it never lets
the net pick on easy states), so the 99.91% was real — but the model had become a rail-
dependent probe specialist, regressing the "plays without the solver" property.

### Run 3 — add a BC anchor → 100%

Fixed the anchor: collect a frozen set of *normal* (non-stuck) states from BC self-play and
add a KL-to-BC term on them in every update, *separate* from the stuck-state policy
gradient. This pins ordinary play to BC while the gradient still reshapes only the probes.

The run reached the goal and held raw play steady:

```
upd  375  hybrid 99.91%  raw 99.4%
upd  550  hybrid 99.96%  raw 99.0%
upd  575  hybrid 100.00% raw 99.1%   <- saved
upd  675  hybrid 100.00% raw 99.3%   <- saved
upd  700  hybrid 100.00% avg 3.4812  raw 99.3%  <- saved
```

Independently verified on a clean checkpoint load:

```
teacher (entropy)      win 100.00%   avg 3.4955   losses 0
model (hybrid)         win 100.00%   avg 3.4812   losses 0   <- new default
model (masked)         win  99.61%   avg 3.5247   losses 9
model (raw, unmasked)  win  99.31%   avg 3.4854   losses 16
```

The hybrid even beats the displayed entropy teacher on *average* (3.481 vs 3.496) at the
same perfect win rate.

## 6. What shipped

- `src/wordle_guesser/rl.py` — the GRPO + teacher-demos + anchor training loop.
- `models/policy_rl.pt` — the 100% checkpoint; new CLI/`evaluate` default.
- Hybrid play wired into `ModelPolicy` / `evaluate_model` via `probe_when_stuck`.
- `--bc` / `--mlp` / `--teacher` preserved; `tests/test_rl.py` added (27 tests pass).

## 7. Lessons

1. **Diagnose before you build.** The masked-ceiling measurement showed the entire task was
   ~7 trap answers and that RL *had* to go unmasked — it set the action space and the whole
   plan in five minutes of evaluation.
2. **Probing is an RL problem, not a labeling one.** The same net that BC couldn't teach to
   probe learned it from the terminal win reward, because the reward can express a
   multi-step return that a label can't.
3. **Align the training objective with the deploy policy.** Folding the safety rail into
   the loop and selecting on the hybrid metric concentrated all gradient on the one decision
   that mattered.
4. **Watch the metric you're *not* optimizing.** Focusing the gradient silently rotted raw
   play to 94.6% while the deploy metric looked perfect. An anchor on the un-optimized
   states fixed it. *Always print the off-objective number.*

## 8. Honest caveats

- **The 100% is on the entire closed game universe.** Wordle has exactly these 2315
  answers, and teacher demos covered every one — so this is optimal coverage of the deploy
  distribution, not a generalization claim.
- **It leans on the rail** (exact candidate tracking), which the product always has. The
  net's own unaided raw play is ~99.3%. The probe itself, though, is now the *net's* — the
  only solver help retained is enumeration, not entropy.
- **Average isn't at the optimum.** Hybrid avg 3.481 already edges the greedy teacher's
  3.4955, but the theoretical optimum is ~3.42 (needs search or the allowed-guess list). Win
  rate was the goal; trimming the average toward ~3.45 is future work (the reward already
  favors speed — a longer, lower-LR tail should help).

## 9. Next

- Push hybrid avg 3.481 → ~3.45 now that win rate is locked.
- The full ~10.6k allowed-guess action space for richer probes (needs a new head).
- Shrink the head at iso-quality.

*See also: `AGENTS.md` (operational guide) and the memory notes
`rl-probing-reaches-100`, `probing-not-cheaply-learnable`, `candidate-set-is-sufficient-statistic`.*
