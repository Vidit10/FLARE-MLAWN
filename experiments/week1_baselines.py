"""Week-1 experiment (Phase 0): clean baselines and a first attack demo.

All numbers come from our Python simulator (rra/env/pysim.py), not from ns-3.

E1  PF vs EqualShare, no attack, mean load U in {5, 10, 20, 30}.
E2  PF under attack at U = 20:
      - A2 random noise (non-adversarial control),
      - A1: k = 3 compromised UEs inflate their own SINR (selfish),
      - A2: query-based search over all users (worst case of this week).
Usage:  python -m experiments.week1_baselines [--seeds 10] [--out results/week1]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from rra.attacks import Attacker, eps_in_physical_units  # noqa: E402
from rra.baselines import EqualShare, ProportionalFair  # noqa: E402
from rra.env.pysim import PySimEnv  # noqa: E402
from rra.runner import run_episode  # noqa: E402

# reference categorical palette (first three slots), neutral ink for text
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def ci95(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float(np.mean(x)) if x.size else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def summarise(rows, keys):
    return {k: ci95([r[k] for r in rows]) for k in keys}


METRICS = ["utility", "viol_eMBB", "viol_URLLC", "viol_mMTC", "any_violation",
           "cell_thr_mbps", "prb_frac", "power_frac", "jain_eMBB",
           "p5_rate_eMBB_mbps", "urllc_p99_delay_ms", "dropped_frac"]


def e1(env, seeds):
    res = {}
    for U in (5, 10, 20, 30):
        for ctrl in (EqualShare(env), ProportionalFair(env)):
            rows = [run_episode(env, ctrl, seed=s, n_users=U) for s in seeds]
            res[(ctrl.name, U)] = summarise(rows, METRICS)
    return res


def e2(env, seeds, U=20):
    pf = ProportionalFair(env)
    attacks = [("none", None)]
    attacks.append(("A2 noise e=0.05", Attacker(eps=0.05, point="A2", strategy="noise")))
    for e in (0.02, 0.05, 0.10):
        attacks.append((f"A1 SINR+ k=3 e={e:g}",
                        Attacker(eps=e, point="A1", k=3, strategy="direction",
                                 features=["sinr_db"], direction={"sinr_db": +1})))
    for e in (0.02, 0.05, 0.10):
        attacks.append((f"A2 search e={e:g}", Attacker(eps=e, point="A2", strategy="search",
                                                       n_candidates=16)))
    res = {}
    for name, att in attacks:
        rows = [run_episode(env, pf, att, seed=s, n_users=U) for s in seeds]
        keys = METRICS + (["attacker_prb_ratio"] if "attacker_prb_ratio" in rows[0] else [])
        res[name] = summarise(rows, keys)
    return res


def plot_e1(res, path):
    loads = [5, 10, 20, 30]
    panels = [("utility", "Utility $\\Phi$ (higher is better)"),
              ("viol_eMBB", "eMBB SLA violation rate"),
              ("viol_URLLC", "URLLC SLA violation rate"),
              ("prb_frac", "PRBs used (fraction)")]
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.8))
    for ax, (k, title) in zip(axs, panels):
        for name, col, mk in (("EqualShare", ORANGE, "s"), ("PF", BLUE, "o")):
            m = [res[(name, U)][k][0] for U in loads]
            c = [res[(name, U)][k][1] for U in loads]
            ax.errorbar(loads, m, yerr=c, color=col, lw=2, marker=mk, ms=6, capsize=3,
                        label=name)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("mean number of users U", fontsize=8, color=INK2)
        ax.set_xticks(loads)
        style(ax)
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Clean baselines in the Python simulator (mean and 95% CI over seeds)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_e2(res, path):
    names = list(res.keys())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (k, title) in zip(axs, (("utility", "Utility $\\Phi$ of PF (higher is better)"),
                                    ("any_violation", "Share of users violating their SLA"))):
        m = [res[n][k][0] for n in names]
        c = [res[n][k][1] for n in names]
        cols = [INK2 if n == "none" else AQUA if "noise" in n else ORANGE if n.startswith("A1")
                else BLUE for n in names]
        y = np.arange(len(names))[::-1]
        ax.barh(y, m, xerr=c, color=cols, height=0.62, capsize=3, error_kw={"ecolor": INK2})
        ax.axvline(res["none"][k][0], color=INK2, lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axs[0] else [], fontsize=8)
        ax.set_title(title, fontsize=9, color=INK)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
    fig.suptitle("PF under observation attacks, U = 20 (Python simulator; dashed line = no attack)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="results/week1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = PySimEnv()
    seeds = list(range(a.seeds))
    t0 = time.time()
    r1 = e1(env, seeds)
    r2 = e2(env, seeds)
    plot_e1(r1, os.path.join(a.out, "e1_clean_baselines.png"))
    plot_e2(r2, os.path.join(a.out, "e2_attacks_on_pf.png"))
    with open(os.path.join(a.out, "e1_clean_baselines.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "U"] + [f"{k}{s}" for k in METRICS for s in ("_mean", "_ci95")])
        for (name, U), d in r1.items():
            w.writerow([name, U] + [f"{v:.4g}" for k in METRICS for v in d[k]])
    with open(os.path.join(a.out, "e2_attacks_on_pf.csv"), "w", newline="") as f:
        keys = METRICS + ["attacker_prb_ratio"]
        w = csv.writer(f)
        w.writerow(["attack"] + [f"{k}{s}" for k in keys for s in ("_mean", "_ci95")])
        for name, d in r2.items():
            w.writerow([name] + [f"{v:.4g}" for k in keys for v in d.get(k, (float('nan'),) * 2)])
    meta = {"seeds": a.seeds, "episode_len": env.cfg.episode_len, "warmup": 20,
            "eps_physical_at_0.05": eps_in_physical_units(env.bounds, 0.05),
            "runtime_s": round(time.time() - t0, 1), "simulator": "rra.env.pysim (Python)"}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
