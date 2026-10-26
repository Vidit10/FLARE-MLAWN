# Week-1 (Phase 0) design notes

What was built this week, why it was built that way, and what is deliberately
left out. Everything listed here exists in the code and is covered by tests
(`python -m pytest tests`).

## 1. One interface for every simulator

`rra/env/interface.py` defines the only objects that controllers, attackers and
(later) the federation code ever see:

* `Observation`: what the xApp sees at the start of an interval.
  * **Reported** per-user features. These can be attacked: SINR, buffer,
    demand, interference and average rate.
  * **Trusted** fields. These cannot be attacked: the xApp's own last PRB and
    power shares, and each user's slice.
* `Action`: a PRB share and a power share per user, each summing to at most 1.
  Whatever is left over is unused, which is how the reward's PRB and power
  costs can be lowered.
* `StepInfo`: ground truth after the interval. This includes delivered bits,
  true SINR, delay, SLA flags, and the reward terms Φ, P and B.
* `to_budgets`: turns shares into integer PRBs (largest-remainder rounding)
  and powers. Feasibility is guaranteed by construction.

The Python simulator implements this now. The ns-3 bridge (Phase 6) will
implement the same class, so nothing above it changes when we switch
simulators.

## 2. Python simulator (`rra/env/pysim.py`)

One downlink cell, three slices, one step = one control interval
T_c = 100 ms.

| Part | Model | Source |
|---|---|---|
| Layout | target gNB plus 6 neighbours, ISD 500 m | 3GPP UMa layout |
| User placement | random drop, kept only if the target cell is the strongest (best-server association) | standard cell selection |
| Path loss and LOS | UMa LOS/NLOS, LOS probability, shadowing 4/6 dB | 3GPP TR 38.901, Tables 7.4.1-1 and 7.4.2-1 |
| Fast fading | Gamma(10, 1/10) power gain, averaged over the interval | modelling choice |
| Interference | each neighbour busy on 30–80% of PRBs, slowly drifting | modelling choice |
| Link | 0.6·log2(1+SINR), capped at 7.4 b/s/Hz, zero below −10 dB | TR 36.942 Annex A style |
| Traffic | eMBB 6–15 Mb/s; URLLC 0.2–1 Mb/s; mMTC 20% bursts of 8–80 kbit | report Table III |
| Queue | delivered = min(capacity, data waiting); 16 Mbit buffer cap, overflow dropped | — |
| SLAs | eMBB R̄ ≥ 5 Mb/s; URLLC q/R ≤ 10 ms; mMTC delivery ≥ 99% over 50 intervals | report Sec. III-B |
| Users | birth–death churn around a target load U | — |

Check: the SINR spread over 6,000 dropped users is 5th percentile 0.5 dB,
median 11.5 dB and 95th percentile 40 dB.

Fair comparisons: random draws never depend on the controller's action, so
every method sees identical users, channels and traffic for the same seed. A
test checks this.

**Not modelled** (on purpose, for speed): HARQ and BLER, per-slot scheduling
inside the 100 ms interval, frequency-selective fading, mobility and handover,
uplink, and beamforming. These are the reasons we validate in ns-3 / 5G-LENA
later.

## 3. Baselines (`rra/baselines`)

* **EqualShare**: equal PRBs and power to every active user.
* **PF**: proportional fair. PRBs are given one at a time to the user with the
  highest se/R̄, where R̄ already includes what the user received in this
  interval. Each user is capped at the PRBs needed to empty its reported
  buffer, and power is equal per PRB. PF reads only reported features, so the
  attacker can target it exactly as it will target the DRL xApp.

## 4. Attacker (`rra/attacks/hook.py`)

The attacker sits between the simulator and the xApp and has an L∞ budget ε on
the normalised features.

| Attack point | What it can change |
|---|---|
| A1, compromised UE | SINR and interference of its own reports only |
| A2, E2 interface | any reported feature of any user |
| A3, malicious xApp writing to the SDL | same reach as A2 |

* Strategies:
  * `noise`: random noise, a non-adversarial control.
  * `direction`: push features one way, for example inflate SINR.
  * `search`: a query-based black-box attack. It tries 16 random sign patterns,
    asks the controller what it would do with each, and keeps the pattern that
    is worst for the true network.
* The FGSM/PGD gradient attacks on the DRL policy plug into the same class in
  Phase 2.
* Tests check three things: the budget is never exceeded, entries outside the
  mask (other users, gNB counters for A1) never change, and trusted fields
  never change.

## 5. Things we found this week that need a decision

1. **The ε meaning is very uneven across features.** With linear [0, 1]
   scaling, ε = 0.05 means:
   * 2.5 dB for SINR;
   * 800 kbit for the buffer;
   * 1 Mb/s for demand;
   * 2.5 Mb/s for the average rate.

   For a URLLC user whose buffer is around 10–100 kbit, 800 kbit is a huge
   change. Linear scaling also squeezes those small buffers to about 0.001, so
   a neural network barely sees them.
   **Proposal:** scale buffer, demand and rate on a log scale, for example
   log10(1 + x) over the same ranges. ε then becomes a relative change, the same for every
   user: over the buffer's ~7.2 decades, ε = 0.02 is about ×1.4 and ε = 0.05
   about ×2.3. This needs one line
   changed in the report's normalisation, plus the physical-ε table.
2. **PF misses the URLLC target even with no attack**: 5–7% violations against
   a 1% target. The cause is that PF is slice-unaware and gives exactly the
   PRBs needed, with no headroom. This is expected and is one reason for a
   slice-aware, constraint-aware learned allocator. We should show it as a
   motivation result and not hide it.
3. **The cell is overloaded at U = 30.** eMBB demand (about 12 users ×
   10.5 Mb/s) is above the roughly 80 Mb/s that PF achieves. This is
   intentional, so that the load range covers light to overloaded. The
   professor should agree that this range is wanted.
