# robust-rra — Trust-aware, attack-robust DRL radio resource allocation

This is the course project code. This README describes the state at the end of
**Phase 3 (week 3)**. Phase 1 (week 2) added a clean (undefended) PPO allocator and
white-box attacks on it. Phase 3 adds the defence: per-user detector, randomized
sanitizer, robust training (Madry and SA-PPO, sanitizer in the loop), the baselines,
and the evaluation of the report's Sec. 5: adaptive attacks against the whole
defended pipeline (EOT, stealth, through-sanitizer), FGSM, UAP, a GAN-based attacker,
a transfer attacker, sweeps over the attack budget and the number of users, and three
training seeds. Federation (Phase 4) and certification (Phase 5) are not implemented yet.

**To present week 3:** open `notebooks/week3_defence.ipynb` (outputs are stored, so it
shows everything without running), or `results/week3/notebook/week3_defence.html` in a
browser. Talking points and the decisions to ask about are in `docs/meeting3_notes.md`.

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
  attacks/gradient.py  white-box FGSM-RS / PGD (KL or utility objective); adaptive, EOT and stealth variants
  attacks/learned.py   UAP, GAN-based generator attacker (plain / detector-aware), transfer attacker
  defence/sanitizer.py per-user autoencoder detector, conformal threshold, randomized / deterministic sanitizer
  defence/controller.py sanitizer -> trust-aware actor pipeline, DefendedController
  defence/robust.py    batched EOT-PGD for adversarial training, SA-PPO inner maximisation
  metrics.py           episode metrics from ground truth
  runner.py            one episode: simulator -> attacker -> controller -> simulator
experiments/
  week1_baselines.py   E1 clean baselines, E2 attacks on PF            -> results/week1/
  train_ppo.py         train one PPO seed                               -> results/week2/ppo_s*/
  week2_dev_ablation.py development runs behind the training set-up    -> results/week2/dev/
  week2_eval.py        E3 clean comparison, E4 attacks, E5 latency      -> results/week2/
  week3_sanitizer.py   collect clean reports, train the detector/sanitizer -> results/week3/sanitizer.pt
  week3_train.py       robust fine-tuning runs (Madry, SA-PPO, baselines) -> results/week3/<run>_s*/
  week3_eval.py        detection, T2, E6 defence table, price sweep      -> results/week3/
  week3_e7.py          attack suite, eps sweep, variable U, detection vs eps, unseen U -> results/week3/e7/
  week3_seeds.py       combine E6, price sweep, detection over seeds 0-2 -> results/week3/seeds_*
scripts/
  week3_all.sh         whole week-3 pipeline for all seeds (Linux / WSL / Docker)
  run_week3_windows.bat / .ps1   week-3 training and E6 for further seeds on Windows
tests/
  test_phase0.py       budgets, conservation, attack masks, scaling, fair seeds
  test_phase1.py       Dirichlet maths, equivariance, need prior, gradient-attack budgets
  test_phase3.py       conformal threshold, sanitizer masks/modes/equivariance, defended controller,
                       adaptive / stealth / UAP / generator / transfer attacks
docs/
  week1_design.md      week-1 modelling choices and open decisions
  meeting1_notes.md    talking points for meeting 1
  week2_design.md      allocator, training set-up and its deviations from the report
  meeting2_notes.md    week-2 results and decisions needed
  week3_design.md      defence design and its deviations from the report
  meeting3_notes.md    week-3 results and decisions needed
  ns3_setup_wsl.md     ns-3.46 + 5G-LENA v4.1.1 + ns3-gym on Windows/WSL2, with fixes
notebooks/
  week2_demo.ipynb     week-2 demonstration notebook; executed, outputs stored in the file
  week3_defence.ipynb  week-3 defence notebook; executed, outputs stored in the file
Dockerfile, docker-compose.yml, requirements.txt    container for the Python-simulator part
```

## Run

```bash
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q tests                                   # 33 tests, ~10 s
python -m experiments.week1_baselines --seeds 10            # ~3 min, 1 core
for s in 0 1 2; do python -m experiments.train_ppo --seed $s --out results/week2/ppo_s$s; done
                                                            # ~1 h per seed, 1 core
python -m experiments.week2_eval --episodes 10              # evaluates every results/week2/ppo_s*
bash scripts/week3_all.sh N        # whole week 3; ~13 h on 2 cores, ~3 h on 12 (Linux, WSL or Docker)
```

`scripts/week3_all.sh` runs, in order: the sanitizer (`week3_sanitizer`), the robust
fine-tuning runs for seeds 0-2 (`week3_train --runs all clean_none --seeds 0 1 2`),
detection / T2 / E6 / price sweep for seed 0 (`week3_eval --part all`), E6, price sweep
and detection for seeds 1-2 (`week3_eval ... --save-dir results/week3/seed<k>`), E7 for
seed 0 (`week3_e7 --part all`) and the combination over seeds (`week3_seeds`).

## Notebooks

`notebooks/week3_defence.ipynb` is the week-3 presentation. It shows every stored week-3 result
(detector, T2, robust training, E6 over three seeds, E7: attack suite, eps sweep, variable U,
detection vs eps and at unseen U, price sweep) and re-runs parts live to show that the stored
numbers are reproduced (about 2 min on 2 cores with `QUICK = True`): it retrains the sanitizer,
re-runs the T2 attack, a subset of E6 and the learned attacks of E7, and compares each with the
stored values. Outputs go to `results/week3/notebook/`, including an HTML copy
(`week3_defence.html`) that opens in any browser.

```bash
jupyter lab notebooks/week3_defence.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/week3_defence.ipynb   # re-run, store outputs
```

`notebooks/week2_demo.ipynb` shows the stored week-2 results, then re-runs a clean comparison and the
white-box attacks live (about 1.5 min on 2 cores with `QUICK = True`). Every table and figure it
produces is saved to `results/notebook/` (CSV, PNG, `manifest.json` with settings and versions, and an
HTML copy of the whole notebook for showing without a Python installation).

```bash
jupyter lab notebooks/week2_demo.ipynb          # interactive
jupyter nbconvert --to notebook --execute --inplace notebooks/week2_demo.ipynb   # re-run and store outputs
jupyter nbconvert --to html notebooks/week2_demo.ipynb --output-dir results/notebook
```

## Docker

The image contains the Python-simulator part (not ns-3).

```bash
docker build -t robust-rra .
docker run --rm -p 127.0.0.1:8888:8888 --cpus 4 -e RRA_WORKERS=4 \
           -v "$PWD/results:/workspace/results" robust-rra      # notebook server on localhost:8888
docker compose run --rm tests                                    # test suite
RRA_CPUS=8 docker compose run --rm train                         # 3 seeds in parallel
RRA_CPUS=8 docker compose run --rm week3                         # whole week-3 pipeline
```

## Using more CPU cores

* **Evaluation:** `--workers N` (or `RRA_WORKERS=N`) runs episodes on N cores. Results are
  identical to a serial run (checked). This uses `fork`, which exists on Linux, WSL and in
  Docker but **not on native Windows**: there `--workers` falls back to one core. On Windows,
  `scripts/run_week3_windows.bat` splits the E6 evaluation into shard processes instead
  (`week3_eval --shard k --n-shards n`, then `--part merge`).
* **Training:** one seed trains on one core. Several seeds run at the same time with
  `python -m experiments.train_seeds --seeds 0 1 2 3 4 --jobs N`. A single seed does not get faster.
* In Docker, the number of cores is set with `--cpus`; under WSL2 it is the `processors` value in
  `%UserProfile%\.wslconfig`.

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
