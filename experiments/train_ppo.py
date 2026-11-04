"""Train the clean PPO allocator (Phase 1) in the Python simulator.

Usage: python -m experiments.train_ppo --seed 0 --out results/week2/ppo_s0

Defaults are the settings used for the week-2 results (see docs/week2_design.md):
need prior, per-user PPO ratios, report reward, gamma = 0.9, 1 M steps.

Robust training (week 3, docs/week3_design.md), e.g. the proposed variant (a):
  python -m experiments.train_ppo --seed 0 --init-from results/week2/ppo_s0 --steps 300000 \
      --defence rand --sanitizer results/week3/sanitizer.pt --adv madry --out results/week3/adv_rand_s0
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
    ap.add_argument("--init-from", default=None, help="run directory to fine-tune from")
    ap.add_argument("--defence", default="none", choices=["none", "det", "rand"],
                    help="sanitizer inside the training loop")
    ap.add_argument("--sanitizer", default="results/week3/sanitizer.pt")
    ap.add_argument("--adv", default="none", choices=["none", "madry", "sappo"])
    ap.add_argument("--adv-eps", type=float, default=0.1)
    ap.add_argument("--adv-frac", type=float, default=0.5)
    ap.add_argument("--adv-steps", type=int, default=5)
    ap.add_argument("--adv-eot", type=int, default=4)
    ap.add_argument("--adv-objective", default="mix", choices=["kl", "util", "mix"])
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--sappo-steps", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    cfg = PPOConfig(total_steps=a.steps, seed=a.seed, zeta_p=a.zeta, zeta_n=a.zeta,
                    reward_mode=a.reward, train_users=tuple(a.users), churn=not a.no_churn,
                    gamma=a.gamma, init_precision=a.precision, lr_actor=a.lr_actor,
                    ratio_mode=a.ratio, prior=a.prior, init_from=a.init_from, defence=a.defence,
                    sanitizer_path=a.sanitizer if a.defence != "none" else None, adv=a.adv,
                    adv_eps=a.adv_eps, adv_frac=a.adv_frac, adv_steps=a.adv_steps, adv_eot=a.adv_eot,
                    adv_objective=a.adv_objective, sappo_kappa=a.kappa, sappo_steps=a.sappo_steps)
    train(cfg, a.out, EnvConfig(feature_scale=a.scale))
