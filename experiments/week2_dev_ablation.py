"""Development runs that decided the week-2 training set-up (1 seed, 200k steps).

Evaluates the saved actors in results/week2/dev/ against PF at U = 20 and 30
(4 evaluation episodes each; seeds 1000-1003).  These are short development
runs, reported only to document why the final set-up was chosen.
"""
import csv
import json
import os

import numpy as np
import torch

from rra.agents.ppo import PPOController
from rra.baselines import ProportionalFair
from rra.env.pysim import PySimEnv
from rra.runner import run_episode

RUNS = [("v_objective", "per-user ratio, no prior, report reward, gamma 0.9"),
        ("v_marginal", "per-user ratio, no prior, PF-marginal reward, gamma 0.9"),
        ("pr_obj99", "per-user ratio, need prior, report reward, gamma 0.99"),
        ("pr_objective", "per-user ratio, need prior, report reward, gamma 0.9")]
KEYS = ("utility", "viol_eMBB", "viol_URLLC", "viol_mMTC", "cell_thr_mbps")

if __name__ == "__main__":
    torch.set_num_threads(1)
    env = PySimEnv()
    rows = []
    ctrls = [("PF", "proportional fair (reference)", ProportionalFair(env))]
    ctrls += [(r, desc, PPOController.load(f"results/week2/dev/{r}/actor.pt")) for r, desc in RUNS]
    for U in (20, 30):
        for name, desc, c in ctrls:
            rs = [run_episode(env, c, seed=s, n_users=U) for s in range(1000, 1004)]
            row = {"run": name, "setup": desc, "U": U}
            row.update({k: round(float(np.nanmean([r[k] for r in rs])), 4) for k in KEYS})
            rows.append(row); print(row, flush=True)
    with open("results/week2/dev/dev_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
