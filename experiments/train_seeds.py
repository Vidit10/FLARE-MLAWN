"""Train several PPO seeds at the same time, one process (one core) per seed.

Usage: python -m experiments.train_seeds --seeds 0 1 2 3 4 --jobs 5 [-- extra train_ppo flags]
       e.g. ... --seeds 0 1 2 --out-root results/week2 -- --steps 1000000

Each seed is an independent run of experiments.train_ppo (the same settings as
a single run, so a seed gives the same result as when it is trained alone).
Output: <out-root>/ppo_s<seed>/ and <out-root>/train_s<seed>.log.
"""
import argparse
import os
import subprocess
import sys
import time

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--jobs", type=int, default=0, help="processes at once (0 = number of cores)")
    ap.add_argument("--out-root", default="results/week2")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="flags for train_ppo, after --")
    a = ap.parse_args()
    extra = [x for x in a.extra if x != "--"]
    from rra.parallel import available_cores
    jobs = a.jobs or available_cores()
    os.makedirs(a.out_root, exist_ok=True)
    queue, running = list(a.seeds), {}
    t0 = time.time()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    while queue or running:
        while queue and len(running) < jobs:
            s = queue.pop(0)
            log = open(os.path.join(a.out_root, f"train_s{s}.log"), "w")
            cmd = [sys.executable, "-m", "experiments.train_ppo", "--seed", str(s),
                   "--out", os.path.join(a.out_root, f"ppo_s{s}")] + extra
            running[s] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            print(f"[{time.time() - t0:6.0f}s] started seed {s}", flush=True)
        time.sleep(5)
        for s, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[s]
                print(f"[{time.time() - t0:6.0f}s] seed {s} finished with exit code {p.returncode}", flush=True)
