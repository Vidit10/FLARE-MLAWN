"""Train the clean PPO allocator (Phase 1) in the Python simulator.

Usage: python -m experiments.train_ppo --seed 0 --out results/week2/ppo_s0

Defaults are the settings used for the week-2 results (see docs/week2_design.md):
need prior, per-user PPO ratios, report reward, gamma = 0.9, 1 M steps.
"""
import argparse

import torch

from rra.agents.ppo import PPOConfig, train
from rra.config import EnvConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--zeta", type=float, default=0.05, help="zeta_p = zeta_n")
    ap.add_argument("--scale", default="log", choices=["log", "linear"])
    ap.add_argument("--reward", default="objective", choices=["objective", "marginal", "debug_embb"])
    ap.add_argument("--users", type=int, nargs=2, default=[5, 30])
    ap.add_argument("--no-churn", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--precision", type=float, default=20.0)
    ap.add_argument("--lr-actor", type=float, default=3e-4)
    ap.add_argument("--prior", default="need", choices=["none", "need"])
    ap.add_argument("--ratio", default="per_user", choices=["per_user", "joint"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = PPOConfig(total_steps=a.steps, seed=a.seed, zeta_p=a.zeta, zeta_n=a.zeta,
                    reward_mode=a.reward, train_users=tuple(a.users), churn=not a.no_churn,
                    gamma=a.gamma, init_precision=a.precision, lr_actor=a.lr_actor,
                    ratio_mode=a.ratio, prior=a.prior)
    train(cfg, a.out, EnvConfig(feature_scale=a.scale))
