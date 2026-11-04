"""Week 3: train the robust allocators (fine-tuned from the clean week-2 models).

Usage: python -m experiments.week3_train [--runs all] [--seeds 0] [--steps 200000] [--jobs N]

Every run starts from results/week2/ppo_s<seed> (actor, critic, lambda) and
continues PPO training for --steps steps with the settings below. One process
(one core) per run; --jobs runs several at once. Output:
results/week3/<run>_s<seed>/ and results/week3/train_<run>_s<seed>.log.

  run             sanitizer in training   adversary in training            deployed with
  adv_rand        randomized (proposed)   Madry, EOT-PGD through sanitizer  randomized sanitizer   (proposed a)
  sappo_rand      randomized (proposed)   SA-PPO regulariser               randomized sanitizer   (proposed b)
  adv_none        none                    Madry on the bare actor          deterministic sanitizer (RADAR-style)
  sappo_none      none                    SA-PPO regulariser               none                   (SA-PPO baseline)
  clean_rand      randomized              none                             randomized sanitizer   (ablation: no adversary)
  zeta020         none                    none, zeta_p = zeta_n = 0.2      none                   (power/PRB price)
  clean_none      none                    none (zeta = 0.05 as in week 2)   none                   (control for zeta020)

"--runs all" runs the first six; clean_none is the control for the price sweep
and is run by name: --runs clean_none (or --runs all clean_none).
"""
import argparse
import os
import subprocess
import sys
import time

RUNS = {
    "adv_rand":   ["--defence", "rand", "--adv", "madry"],
    "sappo_rand": ["--defence", "rand", "--adv", "sappo"],
    "adv_none":   ["--defence", "none", "--adv", "madry"],
    "sappo_none": ["--defence", "none", "--adv", "sappo"],
    "clean_rand": ["--defence", "rand", "--adv", "none"],
    "zeta020":    ["--defence", "none", "--adv", "none", "--zeta", "0.2"],
    "clean_none": ["--defence", "none", "--adv", "none"],
}
DEFAULT_RUNS = ["adv_rand", "sappo_rand", "adv_none", "sappo_none", "clean_rand", "zeta020"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--jobs", type=int, default=0, help="processes at once (0 = number of cores)")
    ap.add_argument("--out-root", default="results/week3")
    ap.add_argument("--sanitizer", default="results/week3/sanitizer.pt")
    a = ap.parse_args()
    from rra.parallel import available_cores
    names = []
    for r in a.runs:                       # "all" expands to the six default runs
        names += DEFAULT_RUNS if r == "all" else [r]
    queue = [(n, s) for s in a.seeds for n in names]
    jobs = a.jobs or available_cores()
    os.makedirs(a.out_root, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running, t0, failed = {}, time.time(), []
    while queue or running:
        while queue and len(running) < jobs:
            n, s = queue.pop(0)
            tag = f"{n}_s{s}"
            cmd = [sys.executable, "-m", "experiments.train_ppo", "--seed", str(s),
                   "--init-from", f"results/week2/ppo_s{s}", "--steps", str(a.steps),
                   "--sanitizer", a.sanitizer, "--out", os.path.join(a.out_root, tag)] + RUNS[n]
            log = open(os.path.join(a.out_root, f"train_{tag}.log"), "w")
            running[tag] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env), log)
            print(f"[{time.time() - t0:6.0f}s] started {tag}", flush=True)
        time.sleep(5)
        for tag, (p, log) in list(running.items()):
            if p.poll() is not None:
                log.close(); del running[tag]
                print(f"[{time.time() - t0:6.0f}s] {tag} finished (exit {p.returncode})", flush=True)
                if p.returncode != 0:
                    failed.append(tag)
    if failed:
        sys.exit(f"failed runs: {', '.join(failed)}")
