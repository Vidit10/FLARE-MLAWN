# Meeting 2 — week-2 status: clean PPO allocator and white-box attacks

All numbers come from **our Python simulator**, with the log feature scaling
(`docs/week2_design.md`, Sec. 1). They are not ns-3 results. The PPO allocator
has **no defence yet**.

* **PPO:** 3 training seeds × 1 M environment steps.
* **E3 (clean comparison):** 10 evaluation episodes of 300 intervals per
  setting (PPO: 10 × 3 seeds), seeds 1000–1009.
* **E4 (attacks):** 6 episodes (PPO: 6 × 3 seeds), seeds 1000–1005. Every E4
  row uses the same episodes, so rows can be compared directly. E3 and E4 use
  different episode sets, so E3 at U = 20 and E4 "none" differ slightly.
* Raw numbers with 95% confidence intervals are in
  `results/week2/e3_clean.csv` and `results/week2/e4_attacks.csv`.

---

## 1. What was built

* **The report's Step-4 allocator:**
  * permutation-equivariant (Deep Sets) actor;
  * masked Dirichlet shares for PRBs and power, each with an "unused" share;
  * separate shared encoder and cell head, ready for federation;
  * PPO with Lagrangian (RCPO) multipliers for the per-slice SLA targets.
* **White-box attacks on it** (report Algorithm 1 without the sanitizer):
  * FGSM with random start and PGD-10;
  * on the report's KL loss, and on a utility loss (the attacker's one-step
    estimate of Φ).
* **Same attack masks and budgets as week 1.** The black-box search and the
  heuristic attacks run on both PF and PPO.
* **Tests:** 23 pass. They cover Dirichlet maths against `torch.distributions`,
  equivariance, feasibility, attack budgets and masks, and the need prior.

**Three changes from the report were needed to make training work.**
Evidence is in `docs/week2_design.md` Sec. 3 and
`results/week2/dev/dev_ablation.csv`.

1. **One PPO ratio per user (Gamma form of the Dirichlet) instead of one joint
   ratio.** With the joint ratio, the policy stayed at equal shares after
   300 k steps.
2. **A "need" prior.** The policy starts from "give each user the PRBs its
   reports say it needs" and learns corrections on top (residual policy). The
   prior uses only reported values, so attacks act on it too.
3. **γ = 0.9 instead of 0.99.** After 200 k steps, URLLC violations were 2%
   with 0.9 and 55% with 0.99.

## 2. Clean results (E3, Fig. `results/week2/e3_clean.png`)

| U | Controller | Utility Φ | eMBB viol. | URLLC viol. | mMTC viol. | Power used | Cell thr. (Mb/s) |
|---|---|---|---|---|---|---|---|
| 10 | PF | 1.039 | 0.0% | 6.0% | 3.4% | 0.63 | 37.7 |
| 10 | **PPO** | 1.037 | 0.0% | **0.0%** | **0.0%** | 0.74 | 37.5 |
| 20 | PF | 1.019 | 9.0% | 6.6% | 4.3% | 0.98 | 70.0 |
| 20 | **PPO** | 0.991 | 11.5% | **0.1%** | **0.9%** | 0.995 | 63.6 |
| 20 | need prior only | 1.022 | 6.1% | 81% | 58% | 0.95 | 71.2 |
| 30 | PF | 0.920 | 26% | 5.0% | 3.4% | 1.00 | 76.2 |
| 30 | **PPO** | 0.886 | 34% | **1.1%** | **1.7%** | 1.00 | 67.5 |
| 40* | PF | 0.839 | 40% | 6.2% | 3.4% | 1.00 | 81.1 |
| 40* | **PPO** | 0.816 | 46% | **3.2%** | **2.7%** | 1.00 | 72.8 |

\* U = 40 is outside the training range (5–30).

How to read this:

* **PPO protects URLLC and mMTC.**
  * mMTC stays ≤ 1.7% (target 5%) from U = 5 to 30.
  * URLLC meets its 1% target up to U = 20 and is at 1.1% at U = 30.
  * PF has 5–7% URLLC violations at every load.
* **The cost falls on eMBB:**
  * utility is 0–4% lower than PF;
  * cell throughput is 9–12% lower at U ≥ 20;
  * eMBB violations are higher at U ≥ 20.

  This trade is what the Lagrangian constraints ask for.
  * At U ≥ 20, no method in E3 meets the eMBB 5% target.
  * Averaged over the training loads, the eMBB target is not met either, so
    λ_eMBB keeps rising during training (`learning_curves.png`).
* **What PPO learned.** The untrained prior alone has PF-like utility but 81%
  URLLC violations at U = 20. Training added the slice protection.
* **PPO uses more power than PF at light load** (0.74 vs 0.63 of P_max at
  U = 10). The price ζ_p = 0.05 is too low to make it save power. The planned
  ζ sweep (not done yet) is needed for the power trade-off.
* **Decision latency:** 0.43–0.59 ms median and 0.6–1.8 ms p99 for 5–40
  users, on one CPU thread. That is well inside the 10 ms near-RT budget.

## 3. Attack results at U = 20 (E4, Fig. `results/week2/e4_attacks.png`)

Changes are relative to the same controller without attack on the same
episodes. "Users violating" is the share of all users violating their SLA.

| Attack (budget ε) | PF utility | PF users violating | PPO utility | PPO users violating |
|---|---|---|---|---|
| none | 1.013 | 5.3% | 0.987 | 2.6% |
| A2 random noise, 0.05 | −0.4% | 6.2% | −0.2% | 2.8% |
| A1 3 UEs inflate own SINR, 0.1 | +0.1% | 6.2% | 0.0% | 2.6% |
| A2 black-box search, 0.05 | −2.3% | 7.7% | −1.4% | 3.6% |
| A2 black-box search, 0.1 | −6.1% | 13.4% | −4.3% | 6.5% |
| A2 white-box PGD-KL, 0.05 | n/a | n/a | −1.1% | 3.8% |
| A2 white-box PGD-KL, 0.1 | n/a | n/a | −4.6% | 8.9% (URLLC 0.1% → 8.0%) |
| A2 white-box PGD-utility, 0.05 | n/a | n/a | −3.8% | 4.9% |
| A2 white-box PGD-utility, 0.1 | n/a | n/a | **−11.0%** | **11.4%** (eMBB 6.1% → 31.5%, throughput 63.5 → 43.4 Mb/s) |

What this shows:

* **The undefended DRL allocator is vulnerable** to a white-box attacker on
  the E2 link. At ε = 0.1 (±5 dB on SINR, ×2.6–3.0 on buffer, demand and
  rate):
  * utility drops by 11%;
  * cell throughput drops by about a third;
  * four times as many users miss their SLA.

  This is the problem Phase 3 has to fix.
* **The two white-box objectives hurt different things:**
  * The KL attack, the report's loss, mostly breaks URLLC protection.
  * The utility attack hits eMBB throughput, and it is the stronger attack on
    Φ. **We propose to report both as T1 attacks.**
* **FGSM is almost as strong as PGD here,** for example −3.7% vs −4.6% at
  ε = 0.1.
* **Small budgets do little:** ε = 0.02 costs at most 1.2%.
* **Under the same black-box attack, PPO loses less than PF:**
  * utility −4.3% vs −6.1%;
  * users violating 6.5% vs 13.4%.

  PF cannot be attacked by gradients. So "PF is more robust" is only true
  against an attacker class that cannot be used on PF anyway.
* **The selfish-UE attacks (A1) barely matter at the cell level.** Three
  attacking UEs get about 1.2–1.5× their head-count share of PRBs
  (1.47 under PF, 1.22–1.27 under PPO), with wide intervals.

Caveat: with 6 episodes (18 for PPO), the per-row 95% intervals of the
absolute values overlap for the small attacks. The rows use identical
episodes, so the differences are paired, but paired intervals have not been
computed yet.

## 4. Decisions needed from the professor

1. **Accept the three training changes** (Sec. 1): per-user ratio, need prior,
   and γ = 0.9. The report's Step 4 and its PPO hyper-parameter paragraph need
   a matching update.
2. **Log feature scaling** (proposed in week 1). It is applied here; linear
   scaling is still one switch away.
3. **Report two white-box objectives (KL and utility)** rather than only KL.
4. **The URLLC/mMTC vs eMBB trade-off** in clean conditions is acceptable as
   the constrained-RL result, rather than trying to match PF utility exactly.

## 5. Next (Phase 2 → 3, before meeting 3)

* Paired confidence intervals for E4. Per-episode values need to be stored.
* The ζ sweep for the power and PRB trade-off, with a couple of values.
* Phase 3:
  * per-user autoencoder detector with conformal threshold, giving trust ρ;
  * randomized sanitizer;
  * robust training (min-max and SA-PPO variants);
  * the RADAR and SA-PPO baselines;
  * then the same E4 table with the defence in place, including adaptive
    attacks through the sanitizer (EOT).
