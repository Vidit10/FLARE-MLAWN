# Week-2 (Phase 1) design notes: clean PPO allocator and white-box attacks

This document describes what was implemented in Phase 1, the choices made,
and where they differ from the plan in the mid-semester report. Results are
in `docs/meeting2_notes.md` and `results/week2/`.

## 1. Feature scaling (applied as proposed in week 1; still to be confirmed with the professor)

Buffer, demand and average rate are now scaled on a log scale:
`g(x) = log10(1 + x / 1 kbit)` before min–max scaling. SINR and interference
are already in dB and stay linear. This is the default
(`EnvConfig.feature_scale = "log"`). The linear scaling written in the report
is one flag away (`feature_scale="linear"`), and the week-1 script keeps
linear so that its results can be reproduced.

| Feature | ε = 0.02 | ε = 0.05 | ε = 0.10 |
|---|---|---|---|
| SINR, interference | ±1 dB | ±2.5 dB | ±5 dB |
| Buffer | ×1.21 | ×1.62 | ×2.63 |
| Demand | ×1.22 | ×1.64 | ×2.69 |
| Average rate | ×1.24 | ×1.72 | ×2.95 |

This is the same relative change for every slice. Under the old linear
scaling, ε = 0.05 was an absolute 800 kbit of buffer, which is meaningless for
URLLC buffers of 10–100 kbit. **The report's Eq. (norm) and its physical-ε
sentence need a matching update.**

## 2. Allocator (`rra/agents/policy.py`), report Step 4

* **Input per user (11 values, as in the report):** the 5 scaled reports,
  trust ρ (fixed to 1 until the detector exists in Phase 3), the previous PRB
  and power shares, and the slice one-hot.
* **Fixed physics feature (added, nothing learned).** From the 5 scaled
  reports the actor computes how many PRBs the user needs to empty its buffer
  this interval, using the simulator's own link model:
  `n_need = (buffer + demand·T_c) / (B_PRB · SE(SINR) · T_c)`.
  The actor receives `n_need/N` and `log(1+n_need)/log(1+N)`. This is a
  deterministic, differentiable function of attackable inputs. A white-box
  attacker differentiates through it, and a manipulated report changes it.
* **Shared encoder** `f_shared`: 13 → 64 → 64 with tanh, the same weights for
  every user. It is mean-pooled into a cell context `c` (Deep Sets), and `c`
  is extended with the load U/U_max and the total PRB need.
* **Cell head:**
  * a user part maps `[h_u, c]` to one PRB and one power logit per user;
  * a cell part maps `c` to the logits of the "unused" component and to the
    two precisions.
* **Need prior (residual policy).** Each logit is
  `log(prior share) + learned correction`. The prior gives each user the PRBs
  it needs and leaves the rest unused, i.e. shares ≈ need_u / max(N, Σ need).
  The learned corrections start at exactly zero. The untrained policy is
  therefore the prior. Training learns what the prior lacks:
  * slice priorities (URLLC and mMTC protection under overload);
  * fairness;
  * the power split.

  The prior uses only reported values, so the attacker can manipulate it just
  as it can the rest of the input. Section 3 explains why it was needed.
* **Output:** `m = softmax` over {users, unused}, separately for PRBs and
  power. The concentrations are `α = m·A`, with precision `A = exp(a(c))·(U+1)`
  initialised to 20 per component. Training samples from Dir(α); deployment
  uses the mean `m`.
* **Properties checked by tests:**
  * permutation equivariance;
  * feasible actions;
  * the masked Dirichlet log-density, entropy and KL match `torch.distributions`.
* **Encoder and head are separate modules.** Phase 4 federates only the
  encoder (report Step 6).
* **Size:** the actor has 30,598 parameters and the critic 17,732.

## 3. Training (`rra/agents/ppo.py`), PPO + Lagrangian SLA multipliers

* **Reward (report Eq. 6):** `r = Φ − ζ_p P − ζ_n B − Σ_l λ_l V_l`, computed
  from simulator ground truth, with ζ_p = ζ_n = 0.05.
* **Dual update after every iteration:** `λ_l ← [λ_l + η_λ (V̂_l − ν_l)]_+`,
  with ν = (5%, 1%, 5%) for (eMBB, URLLC, mMTC).
* **Critic with four value heads:** one for the return of `Φ − ζP − ζB` and
  one per slice for the return of its violation rate. The advantage is
  `A_r − Σ λ_l A_l`, so a change in λ needs no critic retraining.
* **Time-limit ends are bootstrapped.**
* **Training episodes:** a random mean load U ∈ [5, 30] with user churn.
  Evaluation uses seeds 1000+, which training never sees.

### What did not work, and what was changed (development runs)

1. **One PPO ratio for the whole action did not learn.** This was the first
   version: one ratio for the joint 2 × (U+1)-dimensional Dirichlet, no
   prior, γ = 0.99.
   * After 300 k steps on each of 2 seeds, the mean shares were still within
     about ±8% of equal shares.
   * eMBB users received no more PRBs than URLLC users.

   The likely reason: PPO limits how far the policy may move per update, and for a
   Dirichlet with about 40 components and moderate precision, even a small
   change of the mean shares is a large change of the joint distribution. So
   each update moves the shares only a little.
   **Change:** sample the Dirichlet in its Gamma form, where shares = g / Σg
   and g_u ~ Gamma(α_u) are independent. Then use one ratio per user and
   resource, with the shared cell advantage (as in MAPPO, Yu et al.,
   NeurIPS 2022). The shares are still Dirichlet-distributed.
2. **Even with per-user ratios, learning from equal shares was too slow.**
   With no prior, after 200 k steps at U = 20 the utility was 0.83–0.86,
   against 0.98 for PF (`results/week2/dev/dev_ablation.csv`). The policy has
   to discover from a noisy cell-level reward that a URLLC user needs about 1
   PRB while an eMBB user needs 10 or more.
   **Change:** the need prior (residual policy learning, Silver et al. 2018;
   Johannink et al., ICRA 2019).
3. **γ = 0.99 learned the slice protection much more slowly than γ = 0.9.**
   With the prior, after 200 k steps at U = 20:

   | γ | URLLC violation rate | mMTC violation rate |
   |---|---|---|
   | 0.99 | 55% | 28% |
   | 0.9 | 2.0% | 4.3% |

   The utility already contains history: R̄ is a moving average over about
   20 intervals. A horizon of about 10 intervals (1 s) is therefore enough,
   and much less noisy.
   **Change:** γ = 0.9. The report says 0.99, and **the report needs this
   update.**

These are single-seed development runs with 200 k steps and 4 evaluation
episodes. They are reported only to justify the set-up, not as results.
One more difference: the development runs also had an initial offset of −2
on the "unused" logits. The final set-up removes that offset, so the untrained
policy is exactly the need prior. A test checks this.

| Hyper-parameter | Report plan | Used | Why it changed |
|---|---|---|---|
| Transitions per iteration | 2,048 | 2,048 (8 envs × 256) | — |
| Epochs / minibatch | 10 / 256 | 10 / 256 | — |
| Learning rate actor / critic | 3e-4 / 1e-3 | 3e-4 / 1e-3 | — |
| Clip, λ_GAE | 0.2, 0.95 | same | — |
| γ | 0.99 | **0.9** | Item 3 above |
| PPO ratio | joint (implicit) | **per user and resource** | Item 1 above |
| Policy start | — | **need prior + learned correction** | Item 2 above |
| Hidden width | 64 | 64 | — |
| Entropy coefficient | 0.01 | **0.001** | The entropy is summed over many components, so 0.01 dominated the loss |
| η_λ | 0.01 | **0.05**, λ starts at 0.5 | With about 500 iterations, 0.01 can move λ by at most about 0.5 |
| Seeds | 5 | 3 (this week) | Compute: about 1.5 h for 3 seeds in parallel on 2 CPU cores |
| Steps per seed | — | 1.0 M | — |

## 4. White-box attacks (`rra/attacks/gradient.py`)

* **The hook is the same as before.** The budget, attack-point masks and
  k-user sets are the ones from week 1, and tests check them again.
* **KL objective:** `KL(π(·|s) ‖ π(·|s+δ))` for the PRB and the power
  Dirichlet, in closed form. This is the report's attack loss without the
  sanitizer, which does not exist yet.
* **Utility objective (an addition):** the attacker's one-step estimate of Φ
  for the allocation the policy would output, differentiated through the rate
  model. The attacker sees the true reports before changing them, so it can
  compute this.
* **FGSM-RS:** one step of size 1.25ε from a random start (Wong et al.,
  ICLR 2020). Plain FGSM from δ = 0 does nothing for the KL loss, because its
  gradient is exactly zero there.
* **PGD-10:** step 2.5ε/10, a random start, and projection after every step
  (Madry et al., ICLR 2018). This is Algorithm 1 of the report with Q = 1.
* **PF is not differentiable.** For PF only the black-box search and the
  heuristic attacks apply. Every attack that applies to both controllers is
  run on both.

## 5. Not done yet (deliberately)

* No detector, sanitizer or adversarial training. That is Phase 3.
* No federation. That is Phase 4.
* No ζ sweep for the power/PRB trade-off yet. Only ζ = 0.05 was trained.
