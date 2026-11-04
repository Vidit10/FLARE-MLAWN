# Meeting 3 notes: Phase 3 (detector, sanitizer, robust training)

All numbers come from **our Python simulator** (not ns-3). Sections 2–3 are
training seed 0 (intervals are 95 % over episodes). Section 3b repeats the
main table for training seeds 0, 1 and 2 and is the basis for the claims;
seeds 1 and 2 were trained and evaluated on the author's laptop with the same
scripts. Section 3c (E7) covers the rest of the report's experimental design
(Sec. 5.2–5.4) for seed 0. Evaluation: U = 20 users unless stated, 8 episodes
per cell, the same 8 evaluation seeds (1000–1007) for every controller,
attack and training seed, so the differences are paired.
Design and the changes from the report are in `docs/week3_design.md`; the
notebook `notebooks/week3_defence.ipynb` shows the same tables and re-runs
parts of them live.

## Summary (one minute)

1. Week 3 is complete: detector, sanitizer, robust training, the baselines,
   and every experiment of the report's Sec. 5 except DeepFool (Sec. 3c).
2. The detector forces the attacker to choose: perturbations that stay hidden
   do a quarter to a half of the damage of those that do not.
3. The sanitizer on its own does not stop attackers that account for it, and
   offline attacks fitted through it (UAP, GAN) do more damage with it.
   Randomization adds nothing measurable.
4. Sanitizer-in-the-loop SA-PPO (Proposed (b)) is the most robust defence that
   keeps the URLLC SLA against the damaging attacks (ε ≥ 0.1, U ≥ 20), over
   three training seeds and the whole attack suite; where it is not the best
   (ε ≤ 0.05, U = 5) every defence loses at most about 2 %. Its price is about
   3 % clean utility.
5. SA-PPO alone loses even less utility but violates URLLC without any
   attack; Madry training does not stand out from its no-adversary control.
6. Decision needed: re-centre the claim on (2) and (4) (Section 5, point 8).

## 1. What was built

* Per-user detector/sanitizer (2,367 parameters, slice-conditioned
  autoencoder, conformal threshold, soft gate, randomized 5-head ensemble and
  a deterministic baseline).
* Defended controller (sanitizer → trust-aware PPO actor).
* Adaptive attacks: white-box PGD through the whole pipeline, EOT over the
  sanitizer's noise (Q = 8), a stealth variant that keeps every report under
  the detection threshold, and a targeted attack on the sanitized report (T2).
* The rest of the report's attack suite: FGSM, universal perturbation (UAP),
  GAN-based generator attacker (plain and detector-aware), black-box transfer
  attacker; sweeps over ε and over the number of users; detection at U not
  seen in training.
* Robust fine-tuning: Madry adversarial training and SA-PPO, each with and
  without the sanitizer in the loop, plus an ablation without an adversary.
* 10 new unit tests (33 in total).

## 2. Sanitizer and detector

* Trained on 166,736 clean per-user reports. Threshold τ = 2.95·10⁻³,
  false-positive rate on held-out clean reports 1.06 % (target 1 %).
* Denoising (squared error on reports with uniform ±0.1 noise): 0.0149 before
  → 0.0109 randomized, 0.0080 deterministic.

Share of attacked users flagged / AUC (`results/week3/detect.csv`; clean
reports: 0.5 % flagged):

| Attack | ε = 0.05 | ε = 0.1 |
|---|---|---|
| random noise | 16.2 % / 0.80 | 57.6 % / 0.92 |
| A1 SINR inflation | 0.9 % / 0.51 | 1.1 % / 0.52 |
| black-box search | 55.2 % / 0.90 | 84.1 % / 0.98 |
| PGD-KL on the undefended actor | 43.5 % / 0.88 | 81.6 % / 0.97 |
| PGD-util on the undefended actor | 68.5 % / 0.96 | 97.9 % / 1.00 |
| EOT-PGD-util, adaptive | 41.8 % / 0.94 | 66.1 % / 0.95 |
| EOT-PGD-util, adaptive + stealth | 0.6 % / 0.91 | 0.7 % / 0.89 |

* Over the trajectories of all three training seeds (`seeds_detect.csv`) the
  numbers agree with the table within about ±7 percentage points (clean
  flagged 0.7 ± 0.1 %).
* The detector does **not** see SINR-only manipulation (A1): a higher SINR
  alone is still a plausible report. It is caught only if the other features
  disagree with it, and A1 leaves them alone.
* The stealth attack keeps almost every user under the threshold, as
  designed. The AUC stays high because the error still rises, just not above τ.

Targeted attack on the sanitized report (T2, `t2_through_sanitizer.csv`;
share of the distance to the target that the attacker closes, always-sanitize
gate):

| Sanitizer | K = 10 | K = 50 |
|---|---|---|
| none | 100 % | 100 % |
| deterministic | 24.4 % | 25.2 % |
| randomized, Q = 1 | 17.1 % | 18.5 % |
| randomized, Q = 8 (EOT) | 18.4 % | 19.1 % |

The sanitizer limits how far a report can be steered. The randomized one
limits it a little more than the deterministic one, and EOT recovers only a
small part of that. This is the effect the report predicts, but it is small
(about 6 percentage points).

## 3. Main result: E6 (`e6_summary.csv`, `e6_defence.png`)

Utility change under attack vs the same controller without attack, %
(paired, ε = 0.1 unless stated):

| Controller | clean Φ | black-box | PGD-util, unaware | EOT-KL adaptive | EOT-util adaptive | + stealth | EOT-util ε = 0.05 | worst |
|---|---|---|---|---|---|---|---|---|
| PPO (undefended) | 1.000 | −4.0 | −10.7 | −4.3 | −10.7 | −10.7 | −3.7 | −10.7 |
| PPO + det. sanitizer | 0.999 | −4.2 | −2.8 | −3.8 | −10.0 | −4.4 | −5.2 | −10.0 |
| PPO + rand. sanitizer | 0.999 | −3.9 | −3.0 | −3.9 | −9.9 | −4.2 | −5.1 | −9.9 |
| RADAR-style | 1.004 | −2.5 | −0.9 | −2.3 | −6.9 | −2.6 | −4.0 | −6.9 |
| SA-PPO (no sanitizer) | 1.002 | −0.3 | −1.0 | −0.2 | −1.0 | −1.0 | −0.5 | **−1.0** |
| Proposed (a): rand. san. + Madry | 1.002 | −3.2 | −3.2 | −2.0 | −7.6 | −3.7 | −4.3 | −7.6 |
| Proposed (b): rand. san. + SA-PPO | 0.983 | −1.1 | −2.8 | −0.9 | −4.2 | −3.4 | −3.0 | −4.2 |
| Ablation: rand. san., no adversary | 1.003 | −3.7 | −4.3 | −5.7 | −9.9 | −4.8 | −4.7 | −9.9 |
| PF (heuristic) | 1.027 | −6.6 | – | – | – | – | – | −6.6 |

(Random noise and A1 change utility by less than 1.5 % for every controller.)

What this shows:

1. **The sanitizer alone stops attacks that ignore it, not attacks that
   account for it.** Against PGD computed on the bare actor, adding the
   sanitizer cuts the loss from −10.7 % to about −3 %. Against the adaptive
   attack (gradients through the detector, sanitizer and actor, with EOT) the
   loss is −9.9 %, practically the same as without a defence. The randomized
   and deterministic sanitizers give the same result here, so the randomization
   does not add measurable robustness in E6 (it does a little in T2).
2. **Robust training helps; SA-PPO clearly, Madry unclear.** Seed 0 alone:
   against the strongest attack (adaptive EOT-PGD-util) Madry training lowers
   the loss from −9.9 % to −7.6 % (Proposed (a)) and SA-PPO training to
   −4.2 % (Proposed (b)), while the ablation stays at −9.9 %. Over three seeds
   (Section 3b) the SA-PPO gain holds (−3.4 vs −8.7 for the ablation), but the
   Madry gain over the ablation does not stand out from the seed spread.
3. **The best single result is SA-PPO without the sanitizer** (worst case
   −1.0 %). This is not what the report expected. Two qualifications:
   * Its clean behaviour is different. URLLC violations rise from 0.08 % to
     2.5 % of users, the URLLC 99th-percentile delay from 0 to about 50 ms,
     and power use falls to 97 % (all without attack). The policy reacts less
     to the reports, which also makes it hard to attack. The utility Φ hides
     this, because URLLC users carry little weight in Φ.
   * Proposed (b) keeps URLLC as before (0.06 %), but loses 1.7 % clean
     utility.
4. **RADAR-style** (Madry on the bare actor + deterministic sanitizer) does
   slightly better than Proposed (a) on utility (−6.9 % vs −7.6 %; the
   intervals overlap) but lets more users violate their SLA under the KL
   attack (9.9 vs 2.5 percentage points more than without attack).
5. **The stealth attack does less damage** than the plain adaptive attack
   (−4.2 % vs −9.9 % with the sanitizer). Staying under the threshold forces
   a smaller perturbation, so the detector still has an effect on an attacker
   who wants to stay hidden.

## 3b. Three training seeds (`seeds_summary.csv`, `seeds_worst_case.csv`, `seeds_e6.png`)

`python -m experiments.week3_seeds` combines seeds 0, 1, 2. Per seed, each
cell is the mean over the 8 evaluation episodes; the table gives the mean over
the three training seeds and the sample standard deviation between them
(n = 3, so the spread is itself rough).

Utility change under the adaptive EOT-PGD-util attack (ε = 0.1), %:

| Controller | mean ± sd | per seed (0 / 1 / 2) | clean Φ | clean URLLC violations (%) | clean URLLC p99 delay (ms) |
|---|---|---|---|---|---|
| PPO (undefended) | −10.9 ± 0.5 | −10.7 / −11.5 / −10.7 | 0.997 | 0.09 | 0.4 |
| PPO + det. sanitizer | −9.4 ± 1.0 | −10.0 / −8.2 / −10.1 | 0.996 | 0.09 | 0.4 |
| PPO + rand. sanitizer | −9.3 ± 1.1 | −9.9 / −8.0 / −10.0 | 0.996 | 0.09 | 0.4 |
| RADAR-style | −7.0 ± 0.5 | −6.9 / −6.6 / −7.6 | 1.002 | 0.32 | 2.3 |
| SA-PPO (no sanitizer) | **−0.9 ± 0.3** | −1.0 / −0.7 / −1.2 | 0.986 | **2.6** | **108** |
| Proposed (a): rand. san. + Madry | −7.4 ± 0.8 | −7.6 / −6.5 / −8.1 | 1.001 | 0.09 | 0.0 |
| Proposed (b): rand. san. + SA-PPO | −3.4 ± 1.1 | −4.2 / −3.9 / −2.1 | 0.971 | 0.09 | 0.0 |
| Ablation: rand. san., no adversary | −8.7 ± 2.0 | −9.9 / −6.4 / −9.8 | 1.000 | 0.08 | 0.0 |

What the three seeds change:

1. **Confirmed:** the sanitizer alone does not protect against the adaptive
   attack (−9.3 vs −10.9, within the seed spread); SA-PPO training helps most;
   the sanitizer clearly helps against attacks that ignore it (−10.9 → −3.9
   under defence-unaware PGD); the stealth attack does less damage than the
   plain adaptive one for every defended controller.
2. **Not confirmed: Madry training (Proposed (a)) as a gain over the ablation.**
   With seed 0 alone it looked like −7.6 vs −9.9. Over three seeds it is
   −7.4 ± 0.8 vs −8.7 ± 2.0, and the ablation's seed 1 is −6.4, which is
   better than (a)'s seed 1 (−6.5) and in the range of (a). The evidence for a
   Madry gain from the sanitizer-in-the-loop set-up is therefore weak. The
   evidence for the SA-PPO gain is clear: (b) is −3.4 ± 1.1 against −8.7 ± 2.0
   for the ablation, and every (b) seed is better than every ablation seed.
3. **Randomized vs deterministic sanitizer:** no difference in E6
   (−9.3 vs −9.4); RADAR-style (Madry without the sanitizer in training,
   deterministic sanitizer at deployment) is as good as Proposed (a)
   (−7.0 vs −7.4). With the data we have, the randomization and the
   sanitizer-in-the-loop training do not add measurable robustness over
   the RADAR-style pipeline.
4. **SA-PPO without the sanitizer:** its clean URLLC cost is reproduced in
   all three seeds (2.6 % of users violating on average, p99 delay 108 ms),
   so it is a property of the method here, not of one run. Proposed (b)
   keeps URLLC (0.09 %) and loses 2.9 % clean utility (0.971 vs 0.997),
   which is more than in seed 0 alone (1.7 %).

Worst case over all attacks, per seed (`seeds_worst_case.csv`): undefended
−10.9 ± 0.5, RADAR-style −7.0 ± 0.5, Proposed (a) −7.4 ± 0.8, Proposed (b)
−3.4 ± 1.1, SA-PPO −0.9 ± 0.3, ablation −8.7 ± 2.0.

## 3c. The rest of the experimental design: E7 (`results/week3/e7/`)

Training seed 0, U = 20 unless stated, 8 paired episodes per cell. Design in
`docs/week3_design.md` (Sec. 5b–5c); script `experiments/week3_e7.py`.

**Attack suite at ε = 0.1** (utility change vs no attack, %; 95 % intervals
are about ±0.1–1.4):

| Controller | FGSM | UAP | GAN | GAN, detector-aware | transfer (black-box) | EOT-PGD (E6) |
|---|---|---|---|---|---|---|
| PPO (undefended) | −8.1 | −10.1 | −10.0 | −10.0 | −7.7 | −10.7 |
| PPO + det. sanitizer | −5.8 | −13.5 | −13.4 | −3.2 | −4.0 | −10.0 |
| PPO + rand. sanitizer | −5.9 | −13.5 | −13.4 | −3.3 | −3.9 | −9.9 |
| RADAR-style | −2.2 | −11.2 | −11.0 | −2.0 | −1.5 | −6.9 |
| SA-PPO | −0.6 | −0.8 | −0.6 | −0.6 | −0.1 | −1.0 |
| Proposed (a) | −4.6 | −10.9 | −10.9 | −3.4 | −3.1 | −7.6 |
| Proposed (b) | −2.3 | −3.9 | −3.7 | −1.1 | −1.0 | −4.2 |
| Ablation | −6.2 | −12.6 | −12.6 | −3.9 | −4.6 | −9.9 |
| PF | – | – | – | – | −8.2 | – |

* **Offline attacks fitted through the sanitizer (UAP, GAN) are the
  strongest attacks so far** against every sanitizer-based controller except
  Proposed (b). They do *more* damage with the sanitizer than without it
  (−13.5 % vs −10.1 %): the attack is fitted to what the sanitizer outputs, so
  the sanitizer becomes part of what is exploited. They cost nothing at run
  time (one table look-up or one small network per step).
* **The detector-aware GAN stays almost completely hidden** (0.3–0.7 % of
  users flagged, AUC 0.50–0.55 at every ε), but is then limited to about −3 %.
  Together with the stealth PGD, this is the clearest evidence that the
  detector forces a trade-off on the attacker: hide, or do damage.
* **Transfer from the attacker's own model works** on the undefended PPO
  (−7.7 %) and on PF (−8.2 %), so a black-box attacker without access to the
  deployed model is a real threat. The sanitizer and robust training cut it to
  between −1 % and −4 %.

**ε sweep** (utility change %, U = 20):

| | ε | undefended | + rand. san. | RADAR-style | SA-PPO | Proposed (a) | Proposed (b) |
|---|---|---|---|---|---|---|---|
| FGSM | 0.02 | −0.9 | −1.2 | −0.9 | −0.2 | −1.1 | −1.0 |
| | 0.05 | −3.0 | −1.6 | −0.5 | −0.5 | −1.1 | −1.8 |
| | 0.1 | −8.1 | −5.9 | −2.2 | −0.6 | −4.6 | −2.3 |
| | 0.2 | −21.0 | −15.6 | −9.6 | −0.9 | −13.6 | −4.4 |
| EOT-PGD | 0.02 | −1.1 | −1.5 | −1.1 | −0.2 | −1.4 | −1.3 |
| | 0.05 | −3.7 | −5.1 | −4.0 | −0.5 | −4.3 | −3.0 |
| | 0.1 | −10.7 | −9.9 | −6.9 | −1.0 | −7.6 | −4.2 |
| | 0.2 | −28.0 | −24.8 | −20.7 | −1.8 | −21.1 | −8.2 |

* At small budgets (ε ≤ 0.05) the sanitizer does not help and is sometimes
  slightly worse than no defence: the perturbation is mostly under the
  threshold (6 % of users flagged at ε = 0.02), and the soft gate then adds
  little. Robust training separates the methods only from ε = 0.1 on.
* Proposed (b) stays within −8 % even at ε = 0.2 (undefended −28 %).

**Detection vs ε** (`detect_eps.csv`, `e7_detect_eps.png`; attacks against
PPO + rand. sanitizer; clean false-positive rate 0.5 %): share of attacked
users flagged at ε = 0.02 / 0.05 / 0.1 / 0.2 is 7 / 40 / 66 / 96 % for adaptive
EOT-PGD, 4 / 41 / 64 / 80 % for UAP, 4 / 39 / 59 / 100 % for GAN and 0.3–0.4 %
for the detector-aware GAN. Perturbations of ε = 0.02 are practically
undetectable (AUC about 0.8, flagged 4–7 %).

**Variable number of users** (utility change % under adaptive EOT-PGD-util
0.1; the detector, the sanitizer and the allocators are not retrained; PPO was
trained on U = 5–30):

| U | undefended | + rand. san. | RADAR-style | SA-PPO | Proposed (a) | Proposed (b) |
|---|---|---|---|---|---|---|
| 5 | −1.5 | −1.3 | −1.3 | **−7.7** | −0.7 | −2.2 |
| 10 | −2.9 | −2.6 | −1.9 | −1.6 | −1.3 | −0.8 |
| 20 | −10.7 | −9.9 | −6.9 | −1.0 | −7.6 | −4.2 |
| 40 | −12.7 | −12.1 | −6.5 | −1.3 | −10.2 | −5.4 |

* Every method works for U = 5–40 without retraining, including U = 40,
  outside the PPO training range. The detector's clean false-positive rate
  stays at 1.1–1.9 % for every U.
* Attacks hurt little at light load (there are spare resources) and more at
  heavy load. **SA-PPO alone is the exception: at U = 5 it loses 7.7 %**,
  its worst result anywhere. Proposed (b) does not have this weakness.
* Clean utility at U = 40: Proposed (b) 0.789 vs 0.826 for the undefended PPO
  (−4.5 %), so its clean cost grows with the load.

**Detection at U not seen in the detector's training** (`unseen_users.csv`,
`e7_unseen_users.png`): a second detector was fitted on reports from U = 5–20
only (95,501 reports, τ = 5.96·10⁻³, held-out false-positive rate 1.1 %) and
compared with the main one (U = 5–40) on 2 episodes per U (about 280–2,000
user reports per cell, so the clean rates are rough).

| | U = 5 | 10 | 20 | 40 (unseen) |
|---|---|---|---|---|
| clean flagged, U = 5–20 detector | 0.0 % | 0.4 % | 1.1 % | **3.7 %** |
| clean flagged, main detector | 0.0 % | 4.2 % | 0.5 % | 1.4 % |
| PGD-util 0.1 flagged / AUC, U = 5–20 detector | 81 % / 0.97 | 95 % / 1.00 | 99 % / 1.00 | 99 % / 0.99 |
| black-box search 0.1 flagged / AUC, U = 5–20 detector | 71 % / 0.83 | 82 % / 0.96 | 83 % / 0.96 | 82 % / 0.95 |

Detection of attacks carries over to the unseen U = 40 unchanged, but the
false-positive rate there rises to 3.7 % (target 1 %): the conformal
guarantee holds only for data like the calibration data. Calibrating τ over
the whole load range (as the main detector does) avoids this.

**Does SA-PPO use the reports at all?** (`blind_summary.csv`). Because SA-PPO
hardly reacts to any bounded attack, we checked that it is not simply
ignoring the reports: every report was replaced by the mean report of its
slice (no per-user information left). Utility change: undefended −5.2 %,
+ rand. sanitizer −5.0 %, RADAR-style −5.8 %, **SA-PPO −4.4 %**, Proposed (a)
−4.9 %, Proposed (b) −3.1 % (intervals ±1.1–1.8). SA-PPO loses about as much
as the others, so it does use the per-user reports; its robustness is not
blindness. (Its weaker reaction to the reports is still the likely cause of
its URLLC violations.)

## 4. Other results

* **Price ζ for PRB/power** (`seeds_zeta.csv`, 3 training seeds × 8 episodes;
  mean, with the spread between seeds): the week-2 model (ζ = 0.05), the same
  model fine-tuned 200 k more steps with ζ = 0.05 (control: same extra
  training), and with ζ = 0.2.

  | U | power share: week 2 / control / ζ = 0.2 | utility: week 2 / control / ζ = 0.2 | URLLC violations: control / ζ = 0.2 |
  |---|---|---|---|
  | 10 | 0.774 / 0.778 / **0.746** (sd 0.02–0.03) | 1.056 / 1.056 / 1.057 | 0 % / 0 % |
  | 20 | 0.994 / 0.995 / 0.993 | 0.997 / 0.993 / 1.002 | 0.05 % / 0.12 % |
  | 30 | 0.999 / 0.999 / 0.998 | 0.902 / 0.897 / 0.908 | 0.75 % / 1.57 % |

  The higher price lowers power use only at light load, by about 4 % against
  the control, which is about one between-seed standard deviation; at
  U ≥ 20 the allocator uses all the power at every price. The control shows
  that the extra training alone does not change power use. URLLC violations
  at U = 30 double with ζ = 0.2 (still under 2 %). A much larger ζ, or
  training from scratch with the price, would be needed for a real trade-off.
* **Training curves** (`week3_learning.png`): all runs dip at about 70 k
  steps at the same time. They use the same training seeds, so this is a
  stretch of heavy-load episodes, not instability.

## 5. Decisions to confirm with the professor

1. The autoencoder is conditioned on the slice (trusted field). OK?
2. Madry training alternates the KL and the utility objective, instead of
   only the KL loss in the report.
3. SA-PPO uses the KL of the mean action instead of the Dirichlet KL (the
   Dirichlet KL made the policy collapse; see the design notes).
4. Robust models are fine-tuned from the week-2 model (200 k steps), not
   trained from scratch.
5. RADAR is re-implemented without defensive distillation.
6. DeepFool is left out (it needs class decision boundaries; the allocator
   outputs continuous shares). Is the ε sweep an acceptable substitute?
7. The GAN-based attacker is trained by gradient (white-box) through the
   deployed pipeline, and its detector-aware variant uses the defender's
   detector as the discriminator. OK as the report's "GAN-based adaptive
   attacker"?
8. **Main claim.** With the seed results (3b) and E7 (3c), the data support:
   (i) the detector forces the attacker to choose between hiding and doing
   damage; (ii) sanitizer-in-the-loop SA-PPO (Proposed (b)) is the most robust
   defence that keeps the URLLC SLA against the damaging attacks (ε ≥ 0.1,
   U ≥ 20), across the whole attack suite. They do not support a benefit of the randomization, or of
   the sanitizer on its own against attacks that account for it; against
   offline attacks fitted through it, the sanitizer alone makes things worse.
   Should the paper's claim move to (i) + (ii), with SA-PPO alone and
   RADAR-style as the baselines to beat? Alternatives: a stronger Madry
   set-up (more steps, larger K), or training the sanitizer jointly with the
   allocator (report Sec. 7).
9. Should Φ weight URLLC more, or should the SLA metrics be reported next to
   Φ in every table? SA-PPO alone looks best on Φ while violating URLLC.

## 6. Limitations and next steps

* Three training seeds (0, 1, 2) for E6, the price sweep and detection;
  8 evaluation episodes per cell. With n = 3 the between-seed spread is a
  rough estimate; differences below about 2 % utility are not resolved.
  T2 and E7 are seed 0 only.
* Seeds 1 and 2 were trained and evaluated on a different machine (Windows,
  Python venv) from seed 0 (Linux); the results are consistent with seed 0,
  but bit-for-bit equality across machines is not claimed.
* Python simulator only; no ns-3 cross-check of these numbers yet.
* Not done: DeepFool (see 5.6), certified bounds (Phase 5), federation
  (Phase 4), and the extensions suggested in the review notes (TRADES
  frontier, telemetry-informed and SLA-targeting attackers).
