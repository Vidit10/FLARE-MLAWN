"""Launch the week-4 federated runs in parallel, one process per run (works on Windows too).

Grids:
  e9  poisoning with the deployed (pretrained) model, cautious online learning:
      deceived cells k in {0,1,2,3,4} (4 = stress test beyond the median's
      breakdown point) and one malicious cell, for fedavg, median, fltrust and
      two_tier; ablations of two_tier (no data trust, no drift) at k = 3 and
      with the malicious cell; local-only at k = 0 and k = 3; a stronger
      poisoning case (k = 3, eps = 0.2) for fedavg, fltrust and two_tier.
  e8  learning from scratch: federation gain and training steps saved
      (no attack): local, fedavg, fltrust, two_tier; plus poisoning while
      learning (k = 3) for fedavg, fltrust and two_tier.
  observer  Steps F-G (experiments/week4_observer.py --part all), one run per seed.
A run whose output is complete is skipped, so an interrupted launch can simply
be started again.

Usage: python -m experiments.week4_launch --grid all --seeds 0 1 2 --jobs 10
       (then: python -m experiments.week4_eval --part all)
       python -m experiments.week4_launch --grid e9 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

E9_ROUNDS, E9_ITERS = 20, 2
E8_ROUNDS, E8_ITERS = 30, 2


def e9_runs():
    runs = []
    for agg in ("fedavg", "median", "fltrust", "two_tier"):
        for k in (0, 1, 2, 3, 4):
            runs.append(dict(agg=agg, deceived=k, malicious=0))
        runs.append(dict(agg=agg, deceived=0, malicious=1))
    for agg in ("two_tier_no_data", "two_tier_no_drift"):
        runs += [dict(agg=agg, deceived=3, malicious=0), dict(agg=agg, deceived=0, malicious=1)]
    runs += [dict(agg="local", deceived=0, malicious=0), dict(agg="local", deceived=3, malicious=0)]
    runs += [dict(agg=a, deceived=3, malicious=0, eps=0.2) for a in ("fedavg", "fltrust", "two_tier")]
    return [dict(r, exp="e9", init="pretrained", rounds=E9_ROUNDS, iters=E9_ITERS, eval_every=10) for r in runs]


def e8_runs():
    runs = [dict(agg=a, deceived=0, malicious=0) for a in ("local", "fedavg", "fltrust", "two_tier")]
    runs += [dict(agg=a, deceived=3, malicious=0) for a in ("fedavg", "fltrust", "two_tier")]
    return [dict(r, exp="e8", init="scratch", rounds=E8_ROUNDS, iters=E8_ITERS, eval_every=5) for r in runs]


def observer_runs():
    return [dict(exp="observer", init="pretrained", rounds=0, iters=0)]


def out_dir(r, seed, root):
    if r["exp"] == "observer":
        return os.path.join(root, "observer", f"s{seed}")
    eps = f"_eps{r['eps']:g}" if r.get("eps", 0.1) != 0.1 else ""
    return os.path.join(root, r["exp"], f"{r['agg']}_k{r['deceived']}m{r['malicious']}{eps}_s{seed}")


def finished(r, seed, root):
    if r["exp"] == "observer":
        return os.path.exists(os.path.join(out_dir(r, seed, root), "retrain_events.json"))
    p = os.path.join(out_dir(r, seed, root), "eval.csv")
    if not os.path.exists(p):
        return False
    with open(p) as f:
        rounds = [int(row["round"]) for row in csv.DictReader(f)]
    return bool(rounds) and max(rounds) >= r["rounds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="all", choices=["all", "e8", "e9", "observer"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-root", default="results/week4")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    grid = ((e9_runs() if a.grid in ("all", "e9") else []) + (e8_runs() if a.grid in ("all", "e8") else [])
            + (observer_runs() if a.grid in ("all", "observer") else []))
    # longest runs first, so the last hour is not one slow run on one core
    cost = lambda r: 10_000 if r["exp"] == "observer" else r["rounds"] * r["iters"] * (10 if r["init"] == "scratch" else 4)
    grid.sort(key=lambda r: -cost(r))
    jobs = [(r, s) for r in grid for s in a.seeds if not finished(r, s, a.out_root)]
    print(f"{len(jobs)} runs to do ({len(grid) * len(a.seeds) - len(jobs)} already finished), {a.jobs} at a time",
          flush=True)
    if a.dry_run:
        for r, s in jobs:
            print(out_dir(r, s, a.out_root))
        return
    os.makedirs(os.path.join(a.out_root, "logs"), exist_ok=True)
    t0 = time.time()
    done = [0]

    def run(job):
        r, s = job
        name = os.path.basename(out_dir(r, s, a.out_root))
        log = os.path.join(a.out_root, "logs", f"{r['exp']}_{name}.log")
        if r["exp"] == "observer":
            cmd = [sys.executable, "-m", "experiments.week4_observer", "--part", "all", "--seed", str(s),
                   "--out-root", os.path.join(a.out_root, "observer")]
            with open(log, "w") as f:
                code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            done[0] += 1
            print(f"[{done[0]}/{len(jobs)}] observer/s{s} {'ok' if code == 0 else f'FAILED ({code}), see {log}'} "
                  f"({(time.time() - t0) / 60:.0f} min)", flush=True)
            return
        cmd = [sys.executable, "-m", "experiments.week4_fed", "--exp", r["exp"], "--agg", r["agg"],
               "--init", r["init"], "--deceived", str(r["deceived"]), "--malicious", str(r["malicious"]),
               "--rounds", str(r["rounds"]), "--iters", str(r["iters"]), "--eval-every", str(r["eval_every"]),
               "--seed", str(s), "--out-root", a.out_root, "--eps", str(r.get("eps", 0.1))]
        if r["exp"] == "e9" and r["agg"] == "two_tier" and r["deceived"] == 0 and r["malicious"] == 0:
            cmd.append("--save-models")       # policy library for Steps F-G
        if r["exp"] == "e8" and r["agg"] in ("local", "two_tier") and r["deceived"] == 0:
            cmd.append("--save-models")
        with open(log, "w") as f:
            code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        done[0] += 1
        print(f"[{done[0]}/{len(jobs)}] {r['exp']}/{name} {'ok' if code == 0 else f'FAILED ({code}), see {log}'} "
              f"({(time.time() - t0) / 60:.0f} min)", flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(run, jobs))


if __name__ == "__main__":
    main()
