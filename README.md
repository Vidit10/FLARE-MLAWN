# robust-rra — Trust-aware, attack-robust DRL radio resource allocation

This is the course project code. This README describes
the state at the end of **Phase 0 (week 1)**: foundations only. No learning
and no defence exist yet.

```
rra/
  config.py            scenario parameters (report Table III)
  env/interface.py     shared simulator interface: Observation / Action / StepInfo / RRAEnv
  env/link.py          3GPP TR 38.901 UMa path loss, LOS probability, link model
  env/pysim.py         fast Python single-cell simulator (3 slices, SLAs, churn)
  baselines/           EqualShare, ProportionalFair (demand-capped, equal PSD)
  attacks/hook.py      attacker between simulator and xApp (A1/A2/A3, noise/direction/search)
  metrics.py           episode metrics from ground truth
  runner.py            one episode: simulator -> attacker -> controller -> simulator
experiments/
  week1_baselines.py   E1 clean baselines, E2 attacks on PF  -> results/week1/
tests/test_phase0.py   13 checks (budgets, conservation, attack masks, fairness of seeds)
docs/
  week1_design.md      modelling choices, what is not modelled, open decisions
  meeting1_notes.md    talking points for meeting 1 (digital twin, attacker, results)
  ns3_setup_wsl.md     ns-3.46 + 5G-LENA v4.1.1 + ns3-gym on Windows/WSL2, with fixes
```

## Run

```bash
pip install numpy matplotlib pytest
python -m pytest -q tests                                    # ~1 s
python -m experiments.week1_baselines --seeds 10             # ~3 min on 1 core
```

## Minimal use

```python
from rra.env.pysim import PySimEnv
from rra.baselines import ProportionalFair
from rra.attacks import Attacker
from rra.runner import run_episode

env = PySimEnv()
pf = ProportionalFair(env)
print(run_episode(env, pf, seed=0, n_users=20))
print(run_episode(env, pf, Attacker(eps=0.05, point="A2", strategy="search"), seed=0, n_users=20))
```

All results produced by this code come from our simulator. Label them that way
in the report until they are checked in ns-3.
