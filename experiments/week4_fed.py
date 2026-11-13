"""One federated run (FLARE report, Algorithm "One federated round"), Python simulator.

Eight cells (rra/fed/cells.py) run the deployed pipeline (detector, randomized
sanitizer, trust-aware actor) and learn locally between rounds (Step D). After
every round the aggregator combines the updates of the shared per-user layers
(Step E); every cell keeps its own head, critic and SLA multipliers.

Two set-ups (--init):
  pretrained  every cell starts from the deployed week-3 model of this seed
              (Proposed (b): sanitizer-in-the-loop SA-PPO, results/week3/sappo_rand_s<seed>),
              then keeps learning online with small PPO steps (report: "after deployment").
              Used for the poisoning experiments (E9).
  scratch     every cell starts from an untrained actor (need prior only) and
              learns with ordinary PPO steps. Used for federation gain and
              training steps saved (E8).

Threats (E9):
  --deceived k   k cells are fed manipulated reports ("honest but deceived"):
                 a universal perturbation (UAP, eps) fitted against that cell's
                 initial pipeline, applied to every report of the cell, every interval.
  --malicious m  m further cells attack the federation on purpose: they train to lower the
                 reward and send that update boosted by --mal-scale (default 8 = number of
                 cells, the model-replacement boost of Bagdasaryan et al., AISTATS 2020).
Which cells: a fixed permutation of the cells per seed; deceived cells come first.

Scaling (week 5): --n-cells N uses the first N cells of rra.fed.cells.SCALE_ORDER instead of
all eight, and --users U fixes the number of users of every cell (training and evaluation)
instead of drawing it from the cell's range. The output directory name then ends in
_n<N>_U<U>, and cached set-up files are keyed by cell and U.

Output (results/week4/<exp>/<agg>_<tag>_s<seed>/):
  rounds.csv   per round and cell: trust, drift (MMD), data trust, cosine, weight, training KPIs
  eval.csv     per evaluation point, cell and episode: J = Phi - zeta_p P - zeta_n B and the KPIs,
               clean (every cell) and under the cell's attack (deceived cells)
  config.json  everything needed to repeat the run
Usage: python -m experiments.week4_fed --exp e9 --agg two_tier --deceived 2 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import time
import zlib
from dataclasses import asdict

import numpy as np
import torch

from rra.agents.policy import Actor, Critic
from rra.attacks import UAPAttacker, collect_states, fit_uap
from rra.baselines import ProportionalFair
from rra.defence.controller import DefendedController
from rra.defence.sanitizer import Sanitizer
from rra.env.pysim import PySimEnv
from rra.fed.aggregate import AGGREGATORS, aggregate, data_trust, mmd2
from rra.fed.cells import CELLS, ROOT_CELL, cell_env_config, cell_subset
from rra.fed.learner import CellLearner, LocalConfig, set_shared, shared_vector
from rra.runner import run_episode

KPIS = ["utility", "power_frac", "prb_frac", "viol_eMBB", "viol_URLLC", "viol_mMTC", "any_violation",
        "cell_thr_mbps", "jain_eMBB", "p5_rate_eMBB_mbps", "urllc_p99_delay_ms"]


def J_of(r, cfg: LocalConfig) -> float:
    return r["utility"] - cfg.zeta_p * r["power_frac"] - cfg.zeta_n * r["prb_frac"]


def initial_model(init: str, seed: int):
    if init == "pretrained":
        d = f"results/week3/sappo_rand_s{seed}"
        a = Actor(64, prior="need"); a.load_state_dict(torch.load(f"{d}/actor.pt", map_location="cpu"))
        c = Critic(64); c.load_state_dict(torch.load(f"{d}/critic.pt", map_location="cpu"))
        return a, c, np.load(f"{d}/lambda.npy")
    torch.manual_seed(seed)
    return Actor(64, prior="need"), Critic(64), np.full(3, 0.5)


def cached(path, make):
    """Compute once, share between runs (atomic write: parallel runs may race)."""
    if os.path.exists(path):
        return np.load(path)
    torch.manual_seed(zlib.crc32(os.path.basename(path).encode()))   # same result wherever it is computed
    v = make()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npy"
    np.save(tmp, v)
    try:
        os.replace(tmp, path)
    except PermissionError:          # Windows: another run is reading the same file; its copy is identical
        if not os.path.exists(path):
            raise
        os.remove(tmp)
    return v


def cell_reports(env, ctrl, spec, seed, n_episodes=3):
    """Reports of active users in clean rollouts of a controller (the baseline cached in the non-RT RIC)."""
    states = collect_states(env, ctrl, n_episodes=n_episodes,
                            users=(spec.users[0], (spec.users[0] + spec.users[1]) // 2, spec.users[1]),
                            seed=seed, every=4)
    return np.concatenate([o.reported_norm[o.active] for o in states]).astype(np.float32)


def evaluate(learners, attacked, cfg, n_ep, seed_base):
    rows = []
    for L in learners:
        env = PySimEnv(L.env_cfg)
        ctrl = L.controller()
        rng = np.random.default_rng(seed_base + L.idx)
        for j in range(n_ep):
            s = seed_base + 100 * L.idx + j
            U = int(rng.integers(L.spec.users[0], L.spec.users[1] + 1))
            conds = [("clean", None)]
            if L.idx in attacked:
                conds.append(("attacked", attacked[L.idx]))
            for cond, att in conds:
                r = run_episode(env, ctrl, att, seed=s, n_users=U)
                rows.append({"cell": L.idx, "episode": j, "seed": s, "users": U, "condition": cond,
                             "J": J_of(r, cfg), **{k: r.get(k, float("nan")) for k in KPIS}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="e9", help="experiment name (output sub-directory)")
    ap.add_argument("--agg", default="two_tier", choices=AGGREGATORS)
    ap.add_argument("--init", default="pretrained", choices=["pretrained", "scratch"])
    ap.add_argument("--deceived", type=int, default=0)
    ap.add_argument("--malicious", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--mal-scale", type=float, default=None,
                    help="boost of the malicious update (default: number of cells, model replacement)")
    ap.add_argument("--n-cells", type=int, default=8, help="number of cells (gNBs) in the federation, 1..8")
    ap.add_argument("--users", type=int, default=None, help="fixed number of users per cell (default: cell's range)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--iters", type=int, default=2, help="PPO iterations (2048 intervals each) per cell per round")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-episodes", type=int, default=4)
    ap.add_argument("--beta", type=float, default=None, help="drift weight; default: calibrated from clean data")
    ap.add_argument("--no-trust-weighting", action="store_true")
    ap.add_argument("--save-models", action="store_true")
    ap.add_argument("--out-root", default="results/week4")
    a = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    custom = a.n_cells != len(CELLS) or a.users is not None
    if custom:
        cell_ids, specs = cell_subset(a.n_cells, a.users)
    else:                                   # the week-4 federation: all eight cells in their own order
        cell_ids, specs = list(range(len(CELLS))), list(CELLS)
    C = len(specs)
    if a.deceived + a.malicious > C:
        raise SystemExit(f"{a.deceived} deceived + {a.malicious} malicious cells > {C} cells")
    tag = f"k{a.deceived}m{a.malicious}" + (f"_eps{a.eps:g}" if a.eps != 0.1 else "")
    if custom:
        tag += f"_n{C}" + (f"_U{a.users}" if a.users is not None else "")
    # cache keys: unchanged for the week-4 set-up; by cell and U otherwise
    ck = (lambda i: f"c{cell_ids[i]}_U{a.users}") if custom else (lambda i: f"c{i}")
    out = os.path.join(a.out_root, a.exp, f"{a.agg}_{tag}_s{a.seed}")
    os.makedirs(out, exist_ok=True)

    mal = a.mal_scale if a.mal_scale is not None else float(C)
    if a.init == "pretrained":
        cfg = LocalConfig(trust_weighting=not a.no_trust_weighting, malicious_scale=mal)
    else:   # learning from scratch: ordinary PPO steps
        cfg = LocalConfig(lr_actor=3e-4, lr_critic=1e-3, epochs=10, clip=0.2, target_kl=1e9,
                          trust_weighting=not a.no_trust_weighting, malicious_scale=mal)
    san = Sanitizer.load("results/week3/sanitizer.pt")
    actor0, critic0, lam0 = initial_model(a.init, a.seed)
    perm = np.random.default_rng(1000 + a.seed).permutation(C)
    deceived = [int(c) for c in perm[:a.deceived]]
    malicious = [int(c) for c in perm[a.deceived:a.deceived + a.malicious]]
    cache = os.path.join(a.out_root, "cache")

    # per-cell clean baseline of reports (non-RT RIC) and the deceived cells' attack
    base_reports, attackers = {}, {}
    for i, spec in enumerate(specs):
        env = PySimEnv(cell_env_config(spec))
        ctrl = DefendedController(actor0, san, mode=cfg.mode)
        base_reports[i] = cached(f"{cache}/baseline_{a.init}_s{a.seed}_{ck(i)}.npy",
                                 lambda: cell_reports(env, ctrl, spec, seed=40_000 + 100 * i))
        if i in deceived:
            table = cached(f"{cache}/uap_{a.init}_s{a.seed}_{ck(i)}_eps{a.eps:g}.npy",
                           lambda: fit_uap(env, ctrl, collect_states(env, ctrl, n_episodes=4,
                                           users=spec.users, seed=30_000 + 100 * i), a.eps, seed=a.seed))
            attackers[i] = UAPAttacker(eps=a.eps, table=table)

    # drift weight beta: MMD between two clean samples of the same cell is the noise level;
    # beta is set so that a drift of 5x the median noise level halves the data trust.
    def clean_noise():
        vals = []
        for i, spec in enumerate(specs):
            env = PySimEnv(cell_env_config(spec))
            other = cell_reports(env, DefendedController(actor0, san, mode=cfg.mode), spec,
                                 seed=45_000 + 100 * i, n_episodes=2)
            vals.append(np.sqrt(max(mmd2(other, base_reports[i]), 0.0)))
        return np.array(vals)
    noise_key = "" if not custom else f"_n{C}_U{a.users}"
    noise = cached(f"{cache}/mmd_noise_{a.init}_s{a.seed}{noise_key}.npy", clean_noise)
    beta = a.beta if a.beta is not None else float(np.log(2) / (5 * np.median(noise)))

    # torch's generator drives the PPO sampling and the sanitizer's noise; seed it after the
    # (possibly cached) set-up, so a run gives the same numbers whether the cache existed or not
    torch.manual_seed(10_000 * (a.seed + 1))
    learners = [CellLearner(i, spec, cfg, actor0, critic0, lam0, san, seed=10_000 * (a.seed + 1) + i,
                            attacker=attackers.get(i), malicious=i in malicious)
                for i, spec in enumerate(specs)]
    root_spec = ROOT_CELL if a.users is None else dataclasses.replace(ROOT_CELL, users=(a.users, a.users))
    root = (CellLearner(99, root_spec, cfg, actor0, critic0, lam0, san, seed=10_000 * (a.seed + 1) + 99)
            if a.agg == "fltrust" else None)
    n_shared = shared_vector(actor0).numel()
    json.dump({"args": vars(a), "local": asdict(cfg), "cells": [asdict(c) for c in specs], "cell_ids": cell_ids,
               "deceived": deceived, "malicious": malicious, "beta": beta, "mmd_noise": noise.tolist(),
               "shared_params": n_shared, "bytes_per_round_per_cell": 4 * n_shared},
              open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"{out}: deceived {deceived}, malicious {malicious}, beta {beta:.3g}, "
          f"{n_shared} shared parameters ({4 * n_shared / 1024:.0f} KB per cell per round)", flush=True)

    eval_rows = [dict(r, round=0) for r in evaluate(learners, attackers, cfg, a.eval_episodes, 7000)]
    round_rows = []
    glob = shared_vector(actor0)
    for rnd in range(1, a.rounds + 1):
        res = [L.local_round(a.iters) for L in learners]
        mmd = [float(np.sqrt(max(mmd2(r["reports"], base_reports[L.idx], seed=rnd), 0.0)))
               for r, L in zip(res, learners)]
        D = [data_trust(r["rho_bar"], m, beta) for r, m in zip(res, mmd)]
        rho = [r["rho_bar"] for r in res]
        if a.agg == "local":
            w, info = np.full(C, np.nan), {}
        else:
            rootd = None
            if root is not None:
                set_shared(root.actor, glob)
                rootd = root.local_round(a.iters)["delta"]
            upd, w, info = aggregate(a.agg, [r["delta"] for r in res], D, rho, rootd)
            glob = glob + upd
            for L in learners:
                set_shared(L.actor, glob)
        cos = info.get("cos", [np.nan] * C)
        for L, r, m, d, c, wi in zip(learners, res, mmd, D, cos, w):
            round_rows.append({"round": rnd, "cell": L.idx, "deceived": L.idx in deceived,
                               "malicious": L.idx in malicious, "rho_bar": r["rho_bar"],
                               "kept_frac": r["kept_frac"], "mmd": m, "data_trust": d, "cos": c,
                               "weight": wi, "train_utility": r["utility"], "train_viol_eMBB": r["viol"][0],
                               "train_viol_URLLC": r["viol"][1], "train_viol_mMTC": r["viol"][2],
                               "kl": r["kl"], "steps": r["steps"], "update_norm": float(r["delta"].norm())})
        print(f"round {rnd}: weights {np.round(w, 3).tolist()} data trust {np.round(D, 3).tolist()} "
              f"train utility {np.mean([r['utility'] for r in res]):.3f} ({time.time() - t0:.0f} s)", flush=True)
        if rnd % a.eval_every == 0 or rnd == a.rounds:
            eval_rows += [dict(r, round=rnd) for r in evaluate(learners, attackers, cfg, a.eval_episodes, 7000)]
        for name, rows in (("rounds.csv", round_rows), ("eval.csv", eval_rows)):
            with open(os.path.join(out, name), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    if a.save_models:
        for L in learners:
            d = os.path.join(out, f"cell{L.idx}"); os.makedirs(d, exist_ok=True)
            torch.save(L.actor.state_dict(), f"{d}/actor.pt"); torch.save(L.critic.state_dict(), f"{d}/critic.pt")
            np.save(f"{d}/lambda.npy", L.lam)
    meta = json.load(open(os.path.join(out, "config.json")))
    meta["runtime_s"] = round(time.time() - t0, 1)
    json.dump(meta, open(os.path.join(out, "config.json"), "w"), indent=2)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
