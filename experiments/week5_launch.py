"""Launch the week-5 runs in parallel, one process per run (Linux, macOS and Windows).

Grids (--grid):
  fl_scale  Federation size and load with the deployed model (cautious online learning,
            as E9 of week 4): N in {1, 2, 4, 8} cells (gNBs) x U in {10, 20, 30, 40} users
            per cell. Local-only learning for every N; for N >= 2 the aggregators in --aggs
            (default fedavg, two_tier) without attack and with one malicious cell
            (--scenarios clean malicious; add "deceived" for max(1, N // 4) deceived cells).
            20 rounds x 2 PPO iterations, evaluation at rounds 0, 10, 20.
  fl_learn  Federation size and load when learning from scratch (as E8 of week 4):
            local-only for every N, the aggregators in --aggs for N >= 2, no attack.
            30 rounds x 2 PPO iterations, evaluation every 5 rounds.
  compare   Single cell: every controller x attack x U in {10, 20, 30, 40}
            (experiments/week5_compare.py), the same evaluation episodes everywhere.
  all       fl_scale, fl_learn and compare.

Runs are ordered by training seed (all of seed 0 first), longest first within a seed, so a
launch that is stopped early still leaves complete results for the first seed(s). A run
whose output is complete is skipped: an interrupted launch is restarted with the same command.

Usage: python -m experiments.week5_launch --grid all --seeds 0 1 2 --jobs 16
       python -m experiments.week5_launch --grid all --dry-run      (list runs, rough time)
       python -m experiments.week5_launch --grid all --smoke --seeds 0 --n-cells 1 2 --users 10
              --out-root results/week5_smoke                       (a few minutes: checks the set-up)
       then: python -m experiments.week5_eval
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from experiments.week5_compare import ATTACK_KEYS, CONTROLLER_KEYS, USERS, out_path, valid

N_CELLS = (1, 2, 4, 8)
SCALE_ROUNDS, SCALE_ITERS, SCALE_EVAL = 20, 2, 10
LEARN_ROUNDS, LEARN_ITERS, LEARN_EVAL = 30, 2, 5


def fl_runs(exp, ns, us, aggs, scenarios):
    init = "pretrained" if exp == "fl_scale" else "scratch"
    rounds, iters, ev = ((SCALE_ROUNDS, SCALE_ITERS, SCALE_EVAL) if exp == "fl_scale"
                         else (LEARN_ROUNDS, LEARN_ITERS, LEARN_EVAL))
    runs = []
    for n in ns:
        for u in us:
            base = dict(kind="fed", exp=exp, init=init, n=n, users=u, rounds=rounds, iters=iters, eval_every=ev)
            runs.append(dict(base, agg="local", deceived=0, malicious=0))
            if n < 2:
                continue                    # one cell: nothing to aggregate
            for agg in aggs:
                runs.append(dict(base, agg=agg, deceived=0, malicious=0))
                if exp != "fl_scale":
                    continue
                if "malicious" in scenarios:
                    runs.append(dict(base, agg=agg, deceived=0, malicious=1))
                if "deceived" in scenarios:
                    runs.append(dict(base, agg=agg, deceived=max(1, n // 4), malicious=0))
    return runs


def compare_runs(us, controllers, attacks_):
    return [dict(kind="compare", controller=c, attack=t, users=u)
            for u in us for c in controllers for t in attacks_ if valid(c, t)]


def fed_dir(r, seed, root):
    return os.path.join(root, r["exp"], f"{r['agg']}_k{r['deceived']}m{r['malicious']}_n{r['n']}_U{r['users']}_s{seed}")


def finished(r, seed, root):
    if r["kind"] == "compare":
        return os.path.exists(out_path(os.path.join(root, "compare"), r["controller"], r["attack"], r["users"], seed))
    p = os.path.join(fed_dir(r, seed, root), "eval.csv")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        rounds = [int(row["round"]) for row in csv.DictReader(f)]
    return bool(rounds) and max(rounds) >= r["rounds"]


def cost_s(r, episodes):
    """Rough single-core time in seconds (measured on a laptop core; machines differ by 2x either way)."""
    u = r["users"] / 20
    if r["kind"] == "compare":
        per_ep = {"none": 1.5, "noise": 1.5, "blackbox": 17, "whitebox": 22, "adaptive": 30}[r["attack"]]
        return episodes * per_ep * u + 10
    per_iter = 6.0 if r["init"] == "pretrained" else 11.0          # one cell, one PPO iteration, U = 20
    cells = r["n"] + (1 if r["agg"] == "fltrust" else 0)
    return cells * r["rounds"] * r["iters"] * per_iter * (0.4 + 0.6 * u) + 60


def command(r, seed, root, episodes):
    if r["kind"] == "compare":
        return [sys.executable, "-m", "experiments.week5_compare", "--controller", r["controller"],
                "--attack", r["attack"], "--users", str(r["users"]), "--seed", str(seed),
                "--episodes", str(episodes), "--out-root", os.path.join(root, "compare")]
    return [sys.executable, "-m", "experiments.week4_fed", "--exp", r["exp"], "--agg", r["agg"],
            "--init", r["init"], "--n-cells", str(r["n"]), "--users", str(r["users"]),
            "--deceived", str(r["deceived"]), "--malicious", str(r["malicious"]),
            "--rounds", str(r["rounds"]), "--iters", str(r["iters"]), "--eval-every", str(r["eval_every"]),
            "--seed", str(seed), "--out-root", root]


def name(r, seed):
    if r["kind"] == "compare":
        return f"compare_U{r['users']}_s{seed}_{r['controller']}__{r['attack']}"
    return f"{r['exp']}_{r['agg']}_k{r['deceived']}m{r['malicious']}_n{r['n']}_U{r['users']}_s{seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all", choices=["all", "fl_scale", "fl_learn", "compare"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=4, help="runs in parallel (one core each)")
    ap.add_argument("--n-cells", type=int, nargs="+", default=list(N_CELLS))
    ap.add_argument("--users", type=int, nargs="+", default=list(USERS))
    ap.add_argument("--aggs", nargs="+", default=["fedavg", "two_tier"],
                    help="aggregators besides local-only (also: median, fltrust)")
    ap.add_argument("--scenarios", nargs="+", default=["clean", "malicious"],
                    choices=["clean", "malicious", "deceived"], help="fl_scale only")
    ap.add_argument("--controllers", nargs="+", default=list(CONTROLLER_KEYS), choices=list(CONTROLLER_KEYS))
    ap.add_argument("--attacks", nargs="+", default=list(ATTACK_KEYS), choices=list(ATTACK_KEYS))
    ap.add_argument("--episodes", type=int, default=6, help="compare: evaluation episodes per run")
    ap.add_argument("--out-root", default="results/week5")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="set-up check only: 2 rounds of 1 iteration, 1 episode (use a separate --out-root)")
    a = ap.parse_args()
    if a.smoke:
        global SCALE_ROUNDS, SCALE_ITERS, SCALE_EVAL, LEARN_ROUNDS, LEARN_ITERS, LEARN_EVAL
        SCALE_ROUNDS = LEARN_ROUNDS = 2
        SCALE_ITERS = LEARN_ITERS = SCALE_EVAL = LEARN_EVAL = 1
        a.episodes = 1

    grid = []
    for exp in ("fl_scale", "fl_learn"):
        if a.grid in ("all", exp):
            grid += fl_runs(exp, a.n_cells, a.users, a.aggs, a.scenarios)
    if a.grid in ("all", "compare"):
        grid += compare_runs(a.users, a.controllers, a.attacks)
    jobs = [(r, s) for s in a.seeds for r in sorted(grid, key=lambda r: -cost_s(r, a.episodes))
            if not finished(r, s, a.out_root)]
    total = len(grid) * len(a.seeds)
    hours = sum(cost_s(r, a.episodes) for r, _ in jobs) / 3600
    print(f"{len(jobs)} runs to do ({total - len(jobs)} of {total} already finished), {a.jobs} at a time; "
          f"roughly {hours:.0f} core-hours, about {hours / max(a.jobs, 1):.1f} h on {a.jobs} cores", flush=True)
    if a.dry_run:
        by = {}
        for r, _ in jobs:
            k = r["exp"] if r["kind"] == "fed" else "compare"
            by[k] = by.get(k, 0) + cost_s(r, a.episodes) / 3600
        for k, v in by.items():
            print(f"  {k}: {sum(1 for r, _ in jobs if (r.get('exp') or 'compare') == k)} runs, ~{v:.0f} core-hours")
        for r, s in jobs[:20]:
            print("  " + name(r, s))
        if len(jobs) > 20:
            print(f"  ... and {len(jobs) - 20} more")
        return
    os.makedirs(os.path.join(a.out_root, "logs"), exist_ok=True)
    t0 = time.time()
    done = [0]

    def run(job):
        r, s = job
        log = os.path.join(a.out_root, "logs", name(r, s) + ".log")
        with open(log, "w") as f:
            code = subprocess.call(command(r, s, a.out_root, a.episodes), stdout=f, stderr=subprocess.STDOUT)
        done[0] += 1
        print(f"[{done[0]}/{len(jobs)}] {name(r, s)} {'ok' if code == 0 else f'FAILED ({code}), see {log}'} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run, jobs))
    print(f"all done in {(time.time() - t0) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
