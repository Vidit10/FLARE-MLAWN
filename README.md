# robust-rra — Trust-aware, attack-robust DRL radio resource allocation

This is the course project code. This README describes the state at the end of
**Phase 1 (week 2)**. Phase 1 adds a clean (undefended) PPO allocator and
white-box attacks on it. The defence (detector, sanitizer, robust training)
and the federation are not implemented yet.

```
rra/
  config.py            scenario parameters (report Table III); feature scaling switch
  env/interface.py     shared simulator interface: Observation / Action / StepInfo / RRAEnv
  env/link.py          3GPP TR 38.901 UMa path loss, LOS probability, link model
  env/pysim.py         fast Python single-cell simulator (3 slices, SLAs, churn)
  baselines/           EqualShare, ProportionalFair (demand-capped, equal PSD)
  agents/policy.py     permutation-equivariant actor (Deep Sets, masked Dirichlet), critic
  agents/ppo.py        PPO + Lagrangian SLA multipliers (RCPO), PPOController
  attacks/hook.py      attacker between simulator and xApp (A1/A2/A3, noise/direction/search)
  attacks/gradient.py  white-box FGSM-RS / PGD on the policy (KL or utility objective)
  metrics.py           episode metrics from ground truth
  runner.py            one episode: simulator -> attacker -> controller -> simulator
experiments/
  week1_baselines.py   E1 clean baselines, E2 attacks on PF            -> results/week1/
  train_ppo.py         train one PPO seed                               -> results/week2/ppo_s*/
  week2_dev_ablation.py development runs behind the training set-up    -> results/week2/dev/
  week2_eval.py        E3 clean comparison, E4 attacks, E5 latency      -> results/week2/
tests/
  test_phase0.py       budgets, conservation, attack masks, scaling, fair seeds
  test_phase1.py       Dirichlet maths, equivariance, need prior, gradient-attack budgets
docs/
  week1_design.md      week-1 modelling choices and open decisions
  meeting1_notes.md    talking points for meeting 1
  week2_design.md      allocator, training set-up and its deviations from the report
  meeting2_notes.md    week-2 results and decisions needed
  ns3_setup_wsl.md     ns-3.46 + 5G-LENA v4.1.1 + ns3-gym on Windows/WSL2, with fixes
```

## Run

```bash
pip install numpy matplotlib pytest torch
python -m pytest -q tests                                   # ~5 s
python -m experiments.week1_baselines --seeds 10            # ~3 min, 1 core
for s in 0 1 2; do python -m experiments.train_ppo --seed $s --out results/week2/ppo_s$s; done
                                                            # ~1 h per seed, 1 core
python -m experiments.week2_eval --episodes 10              # evaluates every results/week2/ppo_s*
```

## Minimal use

```python
from rra.env.pysim import PySimEnv
from rra.baselines import ProportionalFair
from rra.agents.ppo import PPOController
from rra.attacks import Attacker, GradientAttacker
from rra.runner import run_episode

env = PySimEnv()
pf = ProportionalFair(env)
ppo = PPOController.load("results/week2/ppo_s0/actor.pt")
print(run_episode(env, pf, seed=1000, n_users=20))
print(run_episode(env, ppo, GradientAttacker(eps=0.05, strategy="pgd", objective="kl"),
                  seed=1000, n_users=20))
```

All results produced by this code come from our Python simulator. Label them
that way in the report until they are checked in ns-3.
