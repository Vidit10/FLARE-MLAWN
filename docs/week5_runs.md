# Week 5: scaling and full comparison runs

All results come from the Python simulator (`rra/env/pysim.py`), not ns-3.

## What is run

| Grid | Question | Set-up | Runs (3 seeds) | Rough cost |
|---|---|---|---|---|
| `fl_scale` | How do federation size and load change what federation gives and what one malicious cell can do? | N in {1, 2, 4, 8} cells (gNBs) x U in {10, 20, 30, 40} users per cell. Every cell starts from the deployed week-3 model (Proposed (b)) and keeps learning online, as in E9 of week 4: 20 rounds x 2 PPO iterations. For each N, U: local-only learning; for N >= 2, FedAvg and two-tier trust without attack and with one malicious cell. | 192 | ~70 core-hours |
| `fl_learn` | Does federation help cells learn from scratch, and does that depend on N and U? | Same N x U, untrained actor, as E8 of week 4: 30 rounds x 2 PPO iterations, no attack. Local-only for every N; FedAvg and two-tier for N >= 2. | 120 | ~110 core-hours |
| `compare` | How does every controller do under every attack class at each load? | One cell, U in {10, 20, 30, 40}. Controllers: undefended PPO, PPO + deterministic sanitizer, PPO + randomized sanitizer, RADAR-style, SA-PPO, Proposed (a), Proposed (b), ablation (randomized sanitizer, no adversary), PF. Attacks (eps = 0.1): none, noise, black-box search, white-box PGD unaware of the defence, adaptive white-box EOT-PGD. 6 episodes each, the same episodes for every controller. | 516 | ~16 core-hours |

The costs are rough, measured on laptop cores; a desktop core may be up to twice as fast.
`python -m experiments.week5_launch --grid all --dry-run --jobs <cores>` prints the estimate
for a given number of cores.

Details:
* **Cells for N < 8.** The first N cells of `rra.fed.cells.SCALE_ORDER`: N = 1 mixed urban;
  N = 2 adds URLLC factory; N = 4 adds eMBB hot spot and IoT district; N = 8 is all eight cells.
  Each prefix therefore mixes different traffic, and the cells of a smaller federation are also
  in every larger one.
* **Users.** `--users U` fixes the number of users of every cell in training and evaluation
  (week 4 drew it from each cell's own range). Churn within an episode still applies.
* **N = 1** has nothing to aggregate, so only local-only learning is run there. It is the
  single-cell reference for the federated runs.
* **Malicious cell.** One cell (chosen by a fixed permutation per seed) trains to lower the
  reward and sends its update scaled by N (model replacement), as in week 4. With N = 2 that
  cell is half of the federation, beyond what any robust aggregator can handle without a
  trusted reference; the run shows how the aggregators fail there.
* **U = 20 in `compare`** uses the same models, attacks and episodes as the week-3 table E6, so
  those rows repeat E6.
* Optional extras (not in the default grid): `--aggs fedavg median fltrust two_tier` adds
  aggregators; `--scenarios clean malicious deceived` adds max(1, N/4) deceived cells.

## How to run (fresh clone)

Needs Python 3.10 or newer, about 2 GB of disk and no GPU. One run uses one core.

### Linux, macOS or WSL

```bash
git clone https://github.com/Vidit10/FLARE-MLAWN.git && cd FLARE-MLAWN
python3 -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu      # macOS: pip install torch
pip install -r requirements.txt
python -m pytest -q tests                                               # all tests pass in ~30 s
python -m experiments.week5_launch --grid all --smoke --seeds 0 --n-cells 1 2 --users 10 \
       --out-root results/week5_smoke --jobs 4                          # set-up check, ~5 min
python -m experiments.week5_launch --grid all --dry-run --jobs 16       # how long the real runs take
nohup bash scripts/week5_all.sh 16 > week5_run.log 2>&1 &               # all runs, 16 at a time
tail -f results/week5/launch.log                                        # progress: [done/total] run ok
```

Set the number (16 above) to the number of physical cores, or a little less if the machine is
also used for other work. `nohup ... &` keeps the runs going after the terminal is closed (on a
remote server, `tmux` or `screen` also works).

### Windows (PowerShell or Command Prompt)

```bat
git clone https://github.com/Vidit10/FLARE-MLAWN.git
cd FLARE-MLAWN
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest -q tests
python -m experiments.week5_launch --grid all --smoke --seeds 0 --n-cells 1 2 --users 10 --out-root results/week5_smoke --jobs 4
scripts\run_week5_windows.bat 16
```

Keep the window open and the machine awake (Settings > System > Power: sleep "Never")
until it prints "Finished".

### Docker

```bash
docker build -t robust-rra .
RRA_CPUS=16 docker compose run --rm week5
```

### Fewer seeds first, or one grid at a time

Runs are ordered by seed, so stopping early still leaves complete seeds. To run exactly one
seed or one grid:

```bash
python -m experiments.week5_launch --grid compare --seeds 0 1 2 --jobs 16     # ~1 h on 16 cores
python -m experiments.week5_launch --grid fl_scale --seeds 0 --jobs 16
python -m experiments.week5_launch --grid fl_learn --seeds 0 --jobs 16
python -m experiments.week5_eval                                              # tables from whatever is finished
```

Several machines can share the work by giving each a different `--grid` or `--seeds`, then
copying the `results/week5/` folders together before `week5_eval`.

### If it stops

Start the same command again: finished runs are skipped, unfinished ones start over. A failed
run is reported as `FAILED` in the progress lines; its log is in `results/week5/logs/`.

## What to send back

The folder `results/week5/` (without `logs/` if it is large; model files are not written).
`python -m experiments.week5_eval` writes:

* `summary/tables.md`: every table in Markdown;
* `summary/compare_J_table.csv`, `compare_worst_case.csv`, `compare_summary.csv`, `compare_by_users.png`;
* `summary/fl_scale_summary.csv`, `fl_scale_clean.png`, `fl_scale_malicious.png`;
* `summary/fl_learn_summary.csv`, `fl_learn_curves.png`, `fl_learn_vs_local.png`.

## Metrics

* J = Phi - 0.05 P - 0.05 B (utility minus power and PRB use), as in week 4; also the share of
  users violating their SLA and the per-slice violations.
* `compare`: change under attack against the same controller without attack on the same
  episodes; worst case over the four attacks; difference to undefended PPO under the same attack.
* `fl_scale`: honest cells' J and SLA violations; federated vs local-only (same N, U, seed);
  contagion = honest cells with one malicious cell vs the same aggregator without attack; the
  weight the aggregator gave the malicious cell (its fair share is 1/N); upload per round
  (36 KB per cell).
* `fl_learn`: J and violations at the start and the end, violations averaged over training,
  each against local-only learning with the same N, U and seed.
