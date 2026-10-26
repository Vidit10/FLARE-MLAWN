# Meeting 1 with the professor — foundations

**Goal of the meeting:** answer his two review questions, show that the
pipeline runs end to end, and agree on three decisions.

All numbers below come from **our Python simulator**, with 10 seeds, 300
intervals of 100 ms each, and the first 20 intervals discarded. They are not
ns-3 results and not results of our method, which does not exist yet.

---

## 1. "The digital twin is only in the diagram"

He is right that the report never defined it. Our answer is to **rename it to
"simulation sandbox"** and define it precisely:

> The sandbox is a copy of our simulator, run by the observer rApp in the
> non-RT RIC, configured with the cell's measured context. That context is the
> number of users, the slice mix, the load and the SINR distribution, taken
> from recent trusted KPIs. The rApp uses it for three things only:
> (1) retraining or fine-tuning a policy before deployment,
> (2) shadow-testing a candidate policy against the current one on replayed
> conditions, and
> (3) deciding on rollback.

It is not a live, synchronised replica of the real network. We do not claim
that.

Why this is feasible: the sandbox is exactly `PySimEnv` with its parameters
set from measurements, so it costs us no new component.

## 2. "Is it feasible, and how is the attacker implemented?"

The attacker is a Python function placed between the simulator and the xApp
(`rra/attacks/hook.py`):

```
simulator --reports--> attacker.perturb() --reports'--> xApp --action--> simulator
```

* It changes only reported values and never the real channel or traffic, so
  the damage shows up in the true KPIs.
* An L∞ budget ε applies to the normalised features. For example, ε = 0.05 is
  2.5 dB on SINR.
* Each attack point has a mask, and tests enforce it:
  * **A1, compromised UE:** only its own SINR and interference reports.
  * **A2, E2 link:** any reported feature of any user.
  * **A3, malicious xApp writing to the SDL:** same reach as A2.
* This week's strategies are random noise (control), SINR inflation by k = 3
  UEs, and a query-based black-box search. The gradient attacks on the DRL
  (FGSM/PGD) plug into the same class in Phase 2.

## 3. What runs today (show the two figures)

**Fig. `e1_clean_baselines.png`: PF vs equal share, no attack.**
PF is better on utility and eMBB at every load. For example, at U = 20 its
utility Φ is 1.07 against 0.89.
However, PF misses the URLLC target: 5–7% violations against a 1% target.
PF is slice-unaware and has no headroom, which motivates a slice- and
constraint-aware allocator.

**Fig. `e2_attacks_on_pf.png`: PF under attack at U = 20.**

| Attack | Utility Φ | Users violating SLA |
|---|---|---|
| none | 1.07 | 7.7% |
| A2 random noise, ε = 0.05 | 0.97 | 22% |
| A1, 3 UEs inflate SINR, ε = 0.10 | 1.07 | 8.4% (URLLC 5.0% → 8.4%) |
| A2 query-based search, ε = 0.02 | 0.92 | 24% |
| A2 query-based search, ε = 0.05 | 0.77 | 50% |

* A1 hardly moves the cell-wide numbers. Its effect is unfairness: the three
  attacking UEs take about 1.3× their head-count share of PRBs (wide CI), and
  URLLC violations rise.
* The A2 search at ε = 0.05 makes PF worse than plain equal share (0.89). So
  even a non-learning scheduler that trusts reports is badly exposed, and a
  DRL policy, which also trusts reports, is exposed in the same way.

## 4. Decisions we need from him

1. **The sandbox rename and definition** (section 1). Yes or no.
2. **Log scaling for buffer, demand and rate** (design notes, section 5).
   With linear scaling, ε = 0.05 is 800 kbit of buffer, which is enormous for
   URLLC. Log scaling makes ε a relative change and also helps learning.
3. **Load range U = 5–30.** It goes from light load to overloaded. Is that
   wanted, or should the top be about 20?

Also confirm the scope tiers (Must / Should / Could) from the plan, and the
second simulator. We use the Python simulator for development and ns-3 /
5G-LENA for validation.

## 5. ns-3 status

* The versions are pinned: ns-3.46 + 5G-LENA v4.1.1 + ns3-gym (app-ns-3.36+).
* 5G-LENA's RL-scheduler example now **builds and runs end to end**, driven
  from Python through ns3-gym. A 1 s simulation completed, and the agent set
  per-flow scheduling weights every step.
* It needed five small fixes, all documented in `docs/ns3_setup_wsl.md`:
  * two in ns3-gym's CMake file;
  * one in 5G-LENA's CMake file (its AI-scheduler files are never added to the
    library);
  * one in a 5G-LENA source file (ns-3.46 refuses `Create<>` for Object types);
  * one in ns3gym's Python code (NumPy 2 removed `np.float`).
* In Phase 6 our PRB shares will be mapped onto this scheduler's per-flow
  weights.

## 6. Next (Phase 1, before meeting 2)

* PPO allocator with permutation-equivariant layers and Dirichlet outputs, on
  the clean simulator. The target is to beat PF on Φ while meeting slice SLAs,
  using the Lagrangian multipliers.
* FGSM and PGD gradient attacks on the trained policy, using the same attacker
  class.
