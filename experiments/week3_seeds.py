"""Combine the E6 results of several training seeds (week 3).

Usage: python -m experiments.week3_seeds [--root results/week3] [--seeds 0 1 2]

Seed 0 is read from <root>/e6_episodes.csv, seed k > 0 from <root>/seed<k>/e6_episodes.csv.
For every training seed, the per-cell metric is the mean over the 8 evaluation episodes (the same
evaluation seeds for every controller, attack and training seed). The table gives the mean over
training seeds and the spread between them (sample standard deviation, n = number of seeds),
because the training seed, not the episode, is the unit that a claim about a method rests on.
Also combines, where present for every seed, the price sweep (zeta_sweep.csv) and the
detector results (detect.csv).
Output: <root>/seeds_summary.csv, <root>/seeds_worst_case.csv, <root>/seeds_e6.png,
        <root>/seeds_zeta.csv, <root>/seeds_detect.csv
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(root, seeds):
    parts = []
    for s in seeds:
        p = os.path.join(root, "e6_episodes.csv" if s == 0 else f"seed{s}/e6_episodes.csv")
        d = pd.read_csv(p); d["train_seed"] = s; parts.append(d)
    return pd.concat(parts, ignore_index=True)


def per_seed(d):
    """One row per (training seed, controller, attack): means over the evaluation episodes,
    and the paired utility change against the same controller without attack."""
    g = d.groupby(["train_seed", "controller", "attack"])
    m = g[["utility", "any_violation", "viol_URLLC", "viol_eMBB", "viol_mMTC", "power_frac",
           "urllc_p99_delay_ms"]].mean().reset_index()
    base = d[d.attack == "none"].set_index(["train_seed", "controller", "seed"]).utility
    d = d.copy()
    d["chg"] = 100 * (d.utility.values / base.reindex(
        list(zip(d.train_seed, d.controller, d.seed))).values - 1)
    chg = d.groupby(["train_seed", "controller", "attack"]).chg.mean().rename("utility_change_pct")
    return m.merge(chg.reset_index(), on=["train_seed", "controller", "attack"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/week3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    d = load(a.root, a.seeds)
    ps = per_seed(d)
    order_c = list(dict.fromkeys(d.controller)); order_a = list(dict.fromkeys(d.attack))
    agg = ps.groupby(["controller", "attack"]).agg(
        seeds=("train_seed", "nunique"),
        utility=("utility", "mean"), utility_sd=("utility", "std"),
        utility_change_pct=("utility_change_pct", "mean"), utility_change_sd=("utility_change_pct", "std"),
        utility_change_min=("utility_change_pct", "min"), utility_change_max=("utility_change_pct", "max"),
        any_violation=("any_violation", "mean"), viol_URLLC=("viol_URLLC", "mean"),
        power_frac=("power_frac", "mean"), urllc_p99_delay_ms=("urllc_p99_delay_ms", "mean")).reset_index()
    agg["controller"] = pd.Categorical(agg.controller, order_c)
    agg["attack"] = pd.Categorical(agg.attack, order_a)
    agg = agg.sort_values(["controller", "attack"])
    agg.round(4).to_csv(os.path.join(a.root, "seeds_summary.csv"), index=False)

    # worst case over attacks (per seed, then across seeds), model attacks only (not PF)
    w = ps[ps.controller != "PF"].groupby(["train_seed", "controller"]).utility_change_pct.min().unstack(0)
    w = w.reindex(order_c[:-1] if "PF" in order_c else order_c)
    w["mean"] = w.mean(axis=1); w["sd"] = w.iloc[:, :len(a.seeds)].std(axis=1)
    w.round(2).to_csv(os.path.join(a.root, "seeds_worst_case.csv"))
    print(w.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 3.8))
    sub = agg[agg.attack == "EOT-PGD-util 0.1, adaptive"]
    sub = sub[sub.controller != "PF"]
    y = np.arange(len(sub))
    ax.barh(y, sub.utility_change_pct, xerr=sub.utility_change_sd, color="#4477aa", error_kw={"lw": 1})
    for k, s in enumerate(a.seeds):
        v = ps[(ps.train_seed == s) & (ps.attack == "EOT-PGD-util 0.1, adaptive")].set_index("controller")
        ax.scatter([v.loc[c, "utility_change_pct"] for c in sub.controller], y, s=14, color="k", zorder=3,
                   label="single seed" if k == 0 else None)
    ax.set_yticks(y); ax.set_yticklabels(sub.controller, fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("utility change under adaptive EOT-PGD-util, eps = 0.1 (%)")
    ax.set_title(f"mean and spread over {len(a.seeds)} training seeds (Python simulator)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(a.root, "seeds_e6.png"), dpi=150, bbox_inches="tight")

    z = combine_other(a.root, a.seeds, "zeta_sweep.csv", ["model", "U"],
                      ["utility", "power_frac", "prb_frac", "viol_URLLC", "viol_eMBB", "cell_thr_mbps"], "seeds_zeta.csv")
    if z is not None:
        print(z.round(4).to_string(index=False))
    dt = combine_other(a.root, a.seeds, "detect.csv", ["attack", "eps"], ["flagged", "auc"], "seeds_detect.csv")
    if dt is not None:
        print(dt.round(3).to_string(index=False))


def combine_other(root, seeds, name, keys, values, out):
    """Mean and spread over seeds of a per-seed table (skipped unless every seed has it)."""
    paths = [os.path.join(root, name if s == 0 else f"seed{s}/{name}") for s in seeds]
    if not all(os.path.exists(p) for p in paths):
        print(f"{name}: not available for every seed, skipped")
        return None
    d = pd.concat([pd.read_csv(p).assign(train_seed=s) for p, s in zip(paths, seeds)], ignore_index=True)
    d = d.dropna(subset=keys)
    order = list(dict.fromkeys(map(tuple, d[keys].values)))
    g = d.groupby(keys, sort=False)
    res = pd.concat([g[values].mean().add_suffix(""), g[values].std().add_suffix("_sd"),
                     g.train_seed.nunique().rename("seeds")], axis=1).reset_index()
    res = res.set_index(keys).loc[order].reset_index()
    res.round(5).to_csv(os.path.join(root, out), index=False)
    return res


if __name__ == "__main__":
    main()
