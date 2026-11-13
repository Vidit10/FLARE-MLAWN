# Week-4 (Phase 4) design notes: federation with two-tier trust, and the observer

This implements Steps D–G of the FLARE report (Sec. "Proposed Method") and
the federation experiments of its evaluation plan. Results are in
`docs/meeting4_notes.md`, `results/week4/summary/` and
`notebooks/week4_federation.ipynb`. All numbers come from **our Python
simulator**, not ns-3.

## 1. Set-up

| Item | Report | Implemented |
|---|---|---|
| Cells | C = 8, non-identical traffic | 8 cells (`rra/fed/cells.py`): different slice mix, load range (3–40 users) and neighbour load; same radio set-up as the single-cell model |
| What is shared | "only the shared per-user layers are averaged; each cell keeps its own small output layer" | `actor.shared` (9,216 parameters, 36 KB per cell per round); each cell keeps its head, critic and SLA multipliers |
| Starting point | trained in the digital twin, then online learning | E9: the deployed week-3 model of the seed (Proposed (b)); E8: an untrained actor (need prior), to measure federation gain and steps saved |
| Communication guard | ≤ 80 KB per cell per round | 36 KB |

The cells are simulated independently (interference from neighbours is the
existing random neighbour-load model); coordinated multi-cell scheduling is
out of the report's scope.

## 2. Step D: local learning with trust (`rra/fed/learner.py`)

* Each cell runs the deployed pipeline (detector, randomized sanitizer,
  trust-aware actor) on its own traffic and learns with PPO. **Online
  (E9)**: cautious steps (learning rate 1e-4, 4 epochs, clip 0.1, early stop
  at an approximate KL of 0.02). **From scratch (E8)**: ordinary PPO steps
  (3e-4, 10 epochs, clip 0.2).
* **Trust-weighted samples:** every sample's loss is weighted by the mean
  trust of its users; samples with mean trust below 0.2 are left out.
* **The critic sees the sanitized reports.** In week 3 the critic saw the
  clean reports (allowed in the twin); after deployment the clean reports do
  not exist.
* **Data trust** `D_c = ρ̄_c · exp(−β · MMD_c)`. MMD uses a Gaussian kernel
  (median bandwidth) between the reports the cell received this round and a
  baseline sample cached once from clean rollouts of the starting model.
  **β is calibrated, not hand-set:** the MMD between two clean samples of
  the same cell is the noise level, and β is chosen so that a drift of 5×
  the median noise level halves D_c.

## 3. Step E: aggregation (`rra/fed/aggregate.py`)

| Rule | What it does |
|---|---|
| two-tier (ours) | reference = coordinate-wise median of the updates; `a_c ∝ D_c · max(0, cos(Δ_c, ref))`; every update scaled to at most the reference's norm |
| ablations | without data trust (`D_c = 1`); without the drift term (`D_c = ρ̄_c`) |
| FedAvg | plain mean |
| coordinate median | Yin et al. 2018 |
| FLTrust | the server trains on a clean digital-twin cell each round (its "root dataset"); `a_c = ReLU(cos(Δ_c, Δ_root))`; updates rescaled to the root update's norm |
| local only | no federation |

## 4. Threats

* **Honest but deceived cells:** every report of every user in the cell is
  shifted by a universal perturbation (ε = 0.1; ε = 0.2 as a stronger
  case), fitted once against that cell's starting pipeline. The cell learns
  honestly from what it sees. Deceived cells are chosen by a fixed random
  permutation per seed; k = 0–3 of 8, and 4 as the stress test beyond the
  median's breakdown point.
* **Malicious cell:** trains its local copy to *lower* the reward (negated
  advantages) and sends that update boosted ×8 (= C, the model-replacement
  boost of Bagdasaryan et al., 2020). Its own deployed model is left
  unchanged.
  * **Change:** a sign-flip of an honest update was tried first and did no
    harm. Honest online updates are small and almost uncorrelated between
    cells (cosine 0.1–0.4), so flipping one is just more noise.

## 5. Metrics and the change to Ĵ

* `J = Φ − ζ_p·P − ζ_n·B`, from true KPIs, averaged over 4 evaluation
  episodes per cell (seeds 7000+, the same for every run).
* **Change: Ĵ is normalised with equal shares, not PF.** The report sets
  Ĵ = 0 at the proportional-fair scheduler and Ĵ = 1 at a clean, trained
  allocator. In our simulator PF has the *higher* J in all 8 cells: it does
  not protect the SLAs, and J leaves them out. The denominator would be
  negative. We use `Ĵ = (J − J_equal_share) / (J_ref − J_equal_share)`, where
  J_ref is the same aggregator's run without any affected cell. PF is
  reported next to it, with its SLA violations.
* **Contagion** = mean over cells that are neither deceived nor malicious of
  `1 − Ĵ`. It is also given as the % change of J and as the change of the
  SLA-violation rate (percentage points). Learning mostly changes the
  violations, not J (Sec. 7).
* **Other metrics:** weight on affected cells; honest cells kept
  (C·a_c ≥ 0.5); data-trust AUC; federation gain (vs local only); training
  steps saved (steps to 90 % of the local-only improvement, on violations
  and on J); communication; decision latency.

## 6. Steps F–G (`rra/fed/observer.py`, `experiments/week4_observer.py`)

* **Context summary** κ = [load, slice mix, median of the trusted reports].
  **Library** = the deployed model fine-tuned on each cell's traffic alone
  (25 PPO iterations each), plus each cell's κ and report baseline.
* **Shift-or-attack rule:**
  * **shift:** ≤ 5 % of users distrusted and the nearest library context
    within 0.15;
  * **attack:** ≥ 20 % distrusted, or no library match while some users
    look odd;
  * **no change:** below the drift threshold;
  * **unclear:** otherwise.

  The drift threshold is calibrated on clean windows that are not used for
  the events.
* **Reuse experiment:** after a genuine shift, the cell can keep its policy,
  reuse the matched library policy, start from scratch, or start from a
  **meta-learned initialisation**. The meta-learning is Reptile (Nichol et
  al., 2018), trained on the other cells' traffic, as a simpler stand-in
  for the report's meta-RL baseline (Lotfi & Afghah, 2024). Each then learns
  online and J is measured after every iteration.
* **Observer timeline (one cell):**
  1. clean windows, then a strong new attack (universal perturbation,
     ε = 0.2);
  2. the drop below 0.9× the running level triggers Step F;
  3. on "attack", the twin retrains on an *estimated* attack: the per-slice
     shift of the distrusted reports against the cell's clean reports, used
     as a universal perturbation in the twin;
  4. the candidate is shadow-tested on the replayed attacked windows and
     deployed only if better;
  5. a faulty candidate is shown being rejected by the shadow test, and,
     when forced into service, rolled back on the next drop.

  This is a demonstration of the mechanism on one seed per run, not a
  statistical evaluation.

## 7. Reproducibility

* Every random step is seeded: environment seeds per cell, torch before
  each set-up step and before training, and per evaluation episode. A run
  gives the same numbers whether its cached set-up (baselines, attacks, β)
  existed or not.
* The notebook re-runs round 1 of a stored run from nothing and compares the
  two.
* Commands:

```bash
python -m experiments.week4_launch --grid all --seeds 0 1 2 --jobs N   # 123 runs, about 65 core-hours
python -m experiments.week4_eval --part all                            # tables and figures
# Windows: scripts\run_week4_windows.bat N      Linux/WSL/Docker: bash scripts/week4_all.sh N
```

## 8. Not done

* 10 seeds, as the report asks; we use 3.
* The report's attack budgets ε ∈ {0.01, 0.05, 0.10}; we use 0.1, plus 0.2
  for the stronger poisoning case.
* The meta-learning baseline is Reptile, not the paper's meta-RL method.
* ns-3 (the professor agreed to leave it out of this phase).
* Certified bounds (they are in the robust-RRA design notes, not in the FLARE
  plan).
* The policy-infiltration attack (Ergu et al.).
