"""Week 5: every controller under every attack class, at 10, 20, 30 and 40 users (Python simulator).

One call evaluates one (controller, attack, users, training seed) on the same evaluation
episodes (seeds 1000 .. 1000 + episodes - 1) and writes one CSV, so a launcher can run the
calls in parallel and skip the finished ones (experiments/week5_launch.py --grid compare).

Controllers (week-2/3 models of the given training seed, results/week2 and results/week3):
  ppo            PPO (undefended)                         -- also the "no attack" reference
  ppo_det        PPO + deterministic sanitizer
  ppo_rand       PPO + randomized sanitizer
  radar          RADAR-style (adversarial training, deterministic sanitizer)
  sappo          SA-PPO (no sanitizer)
  prop_a         Proposed (a): randomized sanitizer + adversarial training
  prop_b         Proposed (b): randomized sanitizer + SA-PPO (the deployed model of week 4)
  ablation       Ablation: randomized sanitizer, trained without adversary
  pf             proportional fair scheduler (no learning; not attacked by gradient attacks)
Attacks (eps = 0.1 on the normalised reports unless stated):
  none           no attack
  noise          random noise (A2)
  blackbox       black-box search (A2, 16 candidates per step, no gradients)
  whitebox       white-box PGD on the utility, unaware of the defence
  adaptive       white-box EOT-PGD on the utility, aware of the (randomized) sanitizer
The definitions are those of the week-3 evaluation (experiments/week3_eval.py), so the
U = 20 rows repeat the week-3 table E6 on the same episodes.

Usage: python -m experiments.week5_compare --controller prop_b --attack adaptive --users 30 --seed 0
Output: results/week5/compare/U<U>/s<seed>/<controller>__<attack>.csv (one row per episode)
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from experiments.week3_eval import CONTROLLERS, attacks
from rra.agents.ppo import PPOController
from rra.baselines import ProportionalFair
from rra.config import EnvConfig
from rra.defence.controller import DefendedController
from rra.defence.sanitizer import Sanitizer
from rra.env.pysim import PySimEnv
from rra.runner import run_episode

# short key -> index into week3_eval.CONTROLLERS (None = proportional fair)
CONTROLLER_KEYS = {"ppo": 0, "ppo_det": 1, "ppo_rand": 2, "radar": 3, "sappo": 4,
                   "prop_a": 5, "prop_b": 6, "ablation": 7, "pf": None}
# short key -> name in week3_eval.attacks()
ATTACK_KEYS = {"none": "none", "noise": "noise 0.1", "blackbox": "black-box search 0.1",
               "whitebox": "PGD-util 0.1, defence-unaware", "adaptive": "EOT-PGD-util 0.1, adaptive"}
GRADIENT_ATTACKS = ("whitebox", "adaptive")
USERS = (10, 20, 30, 40)
METRICS = ["J", "utility", "any_violation", "viol_eMBB", "viol_URLLC", "viol_mMTC", "cell_thr_mbps",
           "power_frac", "prb_frac", "jain_eMBB", "p5_rate_eMBB_mbps", "urllc_p99_delay_ms"]
ZETA_P = ZETA_N = 0.05            # J = Phi - zeta_p P - zeta_n B (as in week 4)


def controller_name(key):
    return "PF" if CONTROLLER_KEYS[key] is None else CONTROLLERS[CONTROLLER_KEYS[key]][0]


def valid(ctrl_key, attack_key):
    """PF has no gradient, so the gradient attacks do not apply to it."""
    return not (ctrl_key == "pf" and attack_key in GRADIENT_ATTACKS)


def out_path(root, ctrl_key, attack_key, users, seed):
    return os.path.join(root, f"U{users}", f"s{seed}", f"{ctrl_key}__{attack_key}.csv")


def build_controller(key, env, san, seed):
    if CONTROLLER_KEYS[key] is None:
        return ProportionalFair(env)
    name, run, mode = CONTROLLERS[CONTROLLER_KEYS[key]]
    path = os.path.join(run.format(s=seed), "actor.pt")
    if not os.path.exists(path):
        raise SystemExit(f"model not found: {path}")
    actor = PPOController.load(path).actor
    return DefendedController(actor, san if mode != "none" else None, mode=mode, name=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True, choices=list(CONTROLLER_KEYS))
    ap.add_argument("--attack", required=True, choices=list(ATTACK_KEYS))
    ap.add_argument("--users", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0, help="training seed of the models")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--out-root", default="results/week5/compare")
    a = ap.parse_args()
    if not valid(a.controller, a.attack):
        raise SystemExit("gradient attacks do not apply to PF")
    torch.set_num_threads(1)
    env = PySimEnv(EnvConfig(feature_scale="log"))
    san = Sanitizer.load("results/week3/sanitizer.pt")
    ctrl = build_controller(a.controller, env, san, a.seed)
    fac = {n: f for n, f, _ in attacks()}[ATTACK_KEYS[a.attack]]
    rows = []
    t0 = time.time()
    for j in range(a.episodes):
        s = 1000 + j
        t = time.time()
        att = fac(ctrl) if fac else None
        r = run_episode(env, ctrl, att, seed=s, n_users=a.users)
        r["J"] = r["utility"] - ZETA_P * r["power_frac"] - ZETA_N * r["prb_frac"]
        rows.append({"controller": controller_name(a.controller), "controller_key": a.controller,
                     "attack": ATTACK_KEYS[a.attack], "attack_key": a.attack, "users": a.users,
                     "train_seed": a.seed, "episode_seed": s,
                     **{k: r.get(k, float("nan")) for k in METRICS},
                     "flag_rate": getattr(ctrl, "flag_rate", float("nan")),
                     "attacker_prb_ratio": r.get("attacker_prb_ratio", float("nan")),
                     "runtime_s": round(time.time() - t, 2)})
        print(f"episode {s}: J {r['J']:.3f}, violations {r['any_violation']:.3f} ({time.time() - t0:.0f} s)",
              flush=True)
    out = out_path(a.out_root, a.controller, a.attack, a.users, a.seed)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    os.replace(tmp, out)            # written in one piece: a file that exists is complete


if __name__ == "__main__":
    main()
