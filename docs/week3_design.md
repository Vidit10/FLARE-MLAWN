# Week-3 (Phase 3) design notes: detector, sanitizer, robust training, adaptive attacks

This document describes what was implemented in Phase 3, the choices made,
and where they differ from the design in the report (Sec. III, functions
F3–F10 and the robust-training algorithm). Results are in
`docs/meeting3_notes.md`, `results/week3/` and
`notebooks/week3_defence.ipynb`. All numbers come from **our Python
simulator**, not ns-3.

## 1. Pipeline

```
reports s (attacker may add δ)
  → Stage 1  per-user autoencoder: error e = ||s − D0(E(s))||², conformal threshold τ
             distrust w = sigmoid((e − τ)/T), trust ρ = 1 − w            (rra/defence/sanitizer.py)
  → Stage 2  randomized ensemble: D̄ = mean of M decoders on E(s) + ξ_m, ξ_m ~ N(0, σ²)
             s̃ = (1 − w) s + w D̄
  → Stage 3  equivariant PPO actor on (s̃, ρ) → PRB and power shares    (rra/defence/controller.py)
```

| Item | Report | Implemented |
|---|---|---|
| Encoder / decoders | 5 → 16 → 8 → 4 / 4 → 8 → 16 → 5 | same, **plus the slice one-hot as a conditioning input** (Sec. 2) |
| Ensemble | M heads, latent noise σ | M = 5, σ = 0.1 |
| Threshold | (1 − α) quantile on calibration data | split-conformal quantile, α = 0.01 |
| Gate | sigmoid, T = 0.1 τ; hard gate and always-sanitize as ablations | same (`gate = soft / hard / always`) |
| Deterministic sanitizer (Baseline 3) | single decoder, no noise | separate denoising decoder trained the same way, no noise |
| Data | clean runs, many U, split 70/15/15 | 166,736 per-user reports, U = 5–40, 5 clean controllers, seeds 50 000+ |

The detector/sanitizer has 2,367 parameters (encoder 316, detection
decoder 293, each of the 6 denoising decoders 293) and is trained once (about 2 min
on one core). It is then frozen; robust training changes only the actor and
critic, as in the report.

## 2. Change: the autoencoder is conditioned on the slice

The three slices have very different traffic (eMBB: Mb/s, URLLC and mMTC:
kb/s). Without the slice, a clean report of one slice can look like an
attacked report of another, so the reconstruction error mixes "unusual slice"
with "manipulated". The slice is a *trusted* field (it comes from the core
network, not over E2), so the attacker cannot change it, and giving it to the
autoencoder does not open a new attack surface. Switch:
`SanitizerConfig.cond_slice` (`--no-cond` in `experiments/week3_sanitizer.py`).

## 3. Robust training (`rra/agents/ppo.py`, `rra/defence/robust.py`)

* **Fine-tuning, not training from scratch.** Every robust model starts from
  the clean week-2 model of the same seed (actor, critic and λ) and is trained
  for 200 k more steps. Reasons: compute (each adversarial step costs 2–3
  times a clean one), and the clean policy is already a good start. The
  report does not fix this choice. An "ablation" run (sanitizer in the loop,
  no adversary, same 200 k steps) separates the effect of the extra training
  and of the trust input from the effect of the adversary.
* **(a) Madry, sanitizer in the loop** (report F10a and its algorithm): in
  every iteration 4 of the 8 environments are rolled out under attack. The
  perturbation is EOT-PGD against the current actor *through the frozen
  sanitizer* (K = 5 steps, Q = 4 noise draws per step, ε = 0.1, all users,
  all features). PPO then updates on the clean and attacked rollouts
  together.
  * **Change: the attack objective alternates between the report's KL loss
    and the utility loss** (2 attacked environments each). Week 2 showed that
    the utility attack does about twice the damage of the KL attack, so
    training only against KL would leave the strongest attack untrained.
  * The report's K ∈ [5, 10] and Q = 8; training uses K = 5, Q = 4 to keep the
    cost at about 30 s per iteration. Evaluation uses K = 10, Q = 8.
* **(b) SA-PPO, sanitizer in the loop** (report F10b): the actor loss gets
  κ · max_δ D(π(s), π(s + δ)), with the inner maximum found by 3 PGD steps
  through the sanitizer on every minibatch.
  * **Change: D is the KL between the deployed mean share vectors, not the
    Dirichlet KL.** In a test run with the Dirichlet KL (κ = 0.1) the penalty
    started at about 90 and the policy collapsed within 5 iterations: utility
    went from 1.13 to 0.35. The Dirichlet KL grows with the precision, so the
    easiest way to lower it is to flatten the distribution. That does nothing
    for the deployed (mean) action. The mean-action KL does not depend on the
    precision. With it and κ = 1, training stays stable.
* **Asymmetric critic** (allowed by report F7): the critic sees the clean
  reports; it is not deployed.
* **Lagrange multipliers** are updated from all rollouts, clean and
  attacked, so the SLA constraints apply to the mixture that is trained on.

## 4. Baselines

| Row | What it is |
|---|---|
| PPO (undefended) | week-2 model |
| PPO + det. / rand. sanitizer | week-2 model with the sanitizer added at deployment, no retraining |
| **RADAR-style** | Madry training on the bare actor (no sanitizer in training), deployed behind the deterministic sanitizer. This re-implements the two RADAR components the report contrasts with (input sanitization and adversarial training as separate stages; Ergu & Nguyen, IEEE TGCN 2025). RADAR's defensive distillation is not included. It is not the authors' code. |
| SA-PPO | SA-PPO regulariser (mean-action KL, κ = 1) without a sanitizer (Zhang et al., NeurIPS 2020) |
| Proposed (a) / (b) | Sec. 3 |
| Ablation | sanitizer in the loop, no adversary |

## 5. Attacks used in the evaluation (`rra/attacks/gradient.py`)

* **Defence-unaware** (`adaptive=False`): PGD on the bare actor (trust 1, no
  sanitizer); the result is then sent through the real pipeline.
* **Adaptive** (`adaptive=True`, default): PGD differentiates through the
  whole deployed pipeline (detector gate, sanitizer, actor). Against the
  randomized sanitizer every gradient is averaged over Q = 8 fresh noise draws
  (EOT; Athalye et al., ICML 2018), which is the standard way to attack a
  randomized defence.
* **Adaptive + stealth**: after the attack, any user whose report the
  detector would flag gets its perturbation halved, up to 6 times. The attack
  then stays under the detection threshold, at the price of a smaller
  perturbation.
* **Through-sanitizer targeted attack (T2)**: steer the *sanitized* report to
  a chosen target; deterministic vs randomized sanitizer, Q = 1 or 8.
* The black-box search, noise and A1 attacks from weeks 1–2 are unchanged.

### 5b. Rest of the attack suite of report Sec. 5.2 (`rra/attacks/learned.py`, E7)

* **FGSM**: one signed-gradient step from a random start (week-2 attacker),
  adaptive, with EOT against the randomized sanitizer.
* **UAP** (universal adversarial perturbation): one fixed perturbation per
  slice (3 × 5 values), the same for every user and every step. Fitted offline
  by signed-gradient descent of the attacker's utility estimate, through the
  deployed pipeline with EOT (Q = 4), on 840 clean observations of that
  controller (seeds 3000+, U ∈ {10, 20, 30}). At run time it costs nothing.
* **GAN-based attacker** δ = ε·tanh(G(s)): G is a per-user network (reports,
  slice, cell-mean report → 32 → 32 → 5; same weights for every user, so it
  works for any U). The report leaves open whether G is trained by gradient or
  by RL; we train it **by gradient (white-box)** through the deployed pipeline
  with EOT, the stronger choice. Two variants: plain, and *detector-aware*,
  where the defender's detector plays the role of the discriminator (the loss
  adds 0.2 × the expected share of users the soft gate would flag). Against
  controllers without a detector the two variants are identical.
* **Transfer (black-box)**: PGD-util on a surrogate that the attacker trained
  itself (the clean week-2 PPO of the next training seed), without access to
  the deployed model or the defence; also run on PF.
* **DeepFool is not implemented.** DeepFool computes the smallest step across
  a classifier's nearest decision boundary; the allocator outputs continuous
  shares and has no classes, so the method has no direct counterpart. The ε
  sweep (ε ∈ {0.02, 0.05, 0.1, 0.2}) answers the related question of how small
  a perturbation can still do damage. To be confirmed with the professor.

### 5c. Variable users and ε (report Sec. 5.3–5.4)

U ∈ {5, 10, 20, 40} without retraining the detector, the sanitizer or the
allocators. The PPO models were trained on U = 5–30, so U = 40 is outside
their training range; the sanitizer was fitted on U = 5–40, so no U value is
unseen by it (the report asks for U values excluded from the sanitizer's
training; that would need a refit on a narrower U range and is not done).

## 6. Reproducibility

* Every episode seeds torch's generator with `20 000 + seed`
  (`rra/runner.py`), so the sanitizer's noise and EOT draws are the same
  whichever process or worker runs the episode. Week-2 results do not change:
  nothing in week 2 used torch's global generator.
* Commands:

```bash
python -m experiments.week3_sanitizer                  # ~2 min
python -m experiments.week3_train --runs all --jobs N  # 6 runs, ~40-55 min each on one core
python -m experiments.week3_eval --part all --workers N
python -m experiments.week3_e7 --part all --workers N     # attack suite, eps, U sweeps
python -m experiments.week3_seeds                         # combine training seeds
```

The whole sequence for all three seeds is `scripts/week3_all.sh` (also run by
`docker compose run --rm week3`). On Windows, `scripts/run_week3_windows.bat`
trains and evaluates further seeds (Windows has no `fork`, so the evaluation
there is split into shard processes and merged).

## 7. Not done (deliberately, or not yet)

* Training seed 0 was fine-tuned in the container (2 cores); seeds 1 and 2 on
  the author's laptop with the same launcher (`--seeds 1 2`). The detector
  and sanitizer are shared by all seeds. Detection, T2 and the ζ sweep are
  seed 0 only.
* No certified bounds (report Sec. on certification). That needs auto_LiRPA
  and is planned with Phase 5.
* DeepFool (see 5b), TRADES (suggested as an extension, not in the report's
  plan), certification and the telemetry-informed attacker are not done.
* E7 (attack suite, ε sweep, variable U, detection vs ε) is training seed 0
  only.
* No federation (Phase 4).
