"""Week-5 analysis: tables and figures from results/week5 (Python simulator).

Parts (--part, default all; a part whose runs are missing is skipped):
  fl_scale  federation size N x users U with the deployed model: J and SLA violations of
            honest cells, federated vs local-only (same N, U, seed), and the harm done by one
            malicious cell (contagion: honest cells' change against the same aggregator
            without attack), the weight the malicious cell received, upload per round.
  fl_learn  federation size N x users U, learning from scratch: final J and violations,
            violations averaged over training (area under the curve), federated vs local.
  compare   single cell, every controller x attack x U: J, utility, violations; change
            against the same controller without attack (paired episodes); worst case over
            the attacks; difference to undefended PPO under the same attack.
Every table is the mean over training seeds, with the sample standard deviation over seeds
("_sd"), because the training seed is the unit a claim rests on.

Output: results/week5/summary/*.csv, *.png and tables.md (all tables in Markdown).
Usage: python -m experiments.week5_eval [--root results/week5] [--part all]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

AGG_LABEL = {"local": "local only", "fedavg": "FedAvg", "median": "median", "fltrust": "FLTrust",
             "two_tier": "two-tier (proposed)"}
AGG_COL = {"local": "#777777", "fedavg": "#d62728", "median": "#9467bd", "fltrust": "#ff7f0e",
           "two_tier": "#1f77b4"}
CTRL_ORDER = ["ppo", "ppo_det", "ppo_rand", "radar", "sappo", "prop_a", "prop_b", "ablation", "pf"]
ATT_ORDER = ["none", "noise", "blackbox", "whitebox", "adaptive"]
ATT_LABEL = {"none": "no attack", "noise": "noise", "blackbox": "black-box", "whitebox": "white-box (unaware)",
             "adaptive": "white-box (adaptive)"}
BYTES_PER_CELL = 4 * 9216
RUN_RE = re.compile(r"(.+)_k(\d+)m(\d+)_n(\d+)_U(\d+)_s(\d+)$")


def mean_sd(d, keys, cols):
    g = d.groupby(keys)
    return (g[cols].mean().join(g[cols].std().add_suffix("_sd")).join(g.size().rename("seeds"))
            .reset_index())


def md_table(d, floatfmt="{:.3f}"):
    cols = list(d.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v)
                         else ("" if isinstance(v, (float, np.floating)) else str(v)))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ===================================================================== federated runs
def load_fed(root, exp):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, exp, "*_s*"))):
        m = RUN_RE.match(os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "eval.csv")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        ev = pd.read_csv(os.path.join(d, "eval.csv"))
        if ev["round"].max() < cfg["args"]["rounds"]:
            print(f"skip unfinished {d}")
            continue
        rd = pd.read_csv(os.path.join(d, "rounds.csv")) if os.path.exists(os.path.join(d, "rounds.csv")) else None
        k, mal = int(m.group(2)), int(m.group(3))
        runs.append({"agg": m.group(1), "k": k, "m": mal, "n": int(m.group(4)), "U": int(m.group(5)),
                     "seed": int(m.group(6)), "scenario": "clean" if k + mal == 0 else ("malicious" if mal else "deceived"),
                     "cfg": cfg, "eval": ev, "rounds": rd})
    return runs


def per_cell(ev, rnd=None):
    rnd = ev["round"].max() if rnd is None else rnd
    e = ev[(ev["round"] == rnd) & (ev.condition == "clean")]
    return e.groupby("cell")[["J", "any_violation", "viol_URLLC"]].mean()


def summarise_fl_scale(root, out):
    runs = load_fed(root, "fl_scale")
    if not runs:
        return None
    key = {(r["agg"], r["scenario"], r["n"], r["U"], r["seed"]): r for r in runs}
    rows = []
    for r in runs:
        aff = set(r["cfg"]["deceived"]) | set(r["cfg"]["malicious"])
        cur = per_cell(r["eval"])
        honest = [c for c in cur.index if c not in aff]
        row = {"agg": r["agg"], "scenario": r["scenario"], "n": r["n"], "U": r["U"], "seed": r["seed"],
               "J_honest": cur.J[honest].mean(), "viol_honest": cur.any_violation[honest].mean(),
               "J_change_vs_start_pct": float((100 * (cur.J[honest] / per_cell(r["eval"], 0).J[honest] - 1)).mean()),
               "upload_KB_per_round": 0.0 if r["agg"] == "local" else r["n"] * BYTES_PER_CELL / 1024,
               "runtime_min": r["cfg"].get("runtime_s", np.nan) / 60}
        loc = key.get(("local", "clean", r["n"], r["U"], r["seed"]))
        if loc is not None:
            lc = per_cell(loc["eval"])
            row["J_vs_local_pct"] = float((100 * (cur.J[honest] / lc.J[honest] - 1)).mean())
            row["viol_vs_local_pp"] = float((100 * (cur.any_violation[honest] - lc.any_violation[honest])).mean())
        ref = key.get((r["agg"], "clean", r["n"], r["U"], r["seed"]))
        if r["scenario"] != "clean" and ref is not None:
            rc = per_cell(ref["eval"])
            row["contagion_J_pct"] = float((100 * (cur.J[honest] / rc.J[honest] - 1)).mean())
            row["contagion_viol_pp"] = float((100 * (cur.any_violation[honest] - rc.any_violation[honest])).mean())
        rd = r["rounds"]
        if rd is not None and aff and rd.weight.notna().any():
            row["weight_on_attacker"] = float(rd[rd.cell.isin(aff)].groupby("round").weight.sum().mean())
            row["fair_share"] = len(aff) / r["n"]
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(out, "fl_scale_runs.csv"), index=False)
    num = [c for c in d.columns if c not in ("agg", "scenario", "n", "U", "seed")]
    s = mean_sd(d, ["scenario", "agg", "n", "U"], num).sort_values(["scenario", "agg", "n", "U"])
    s.to_csv(os.path.join(out, "fl_scale_summary.csv"), index=False)
    plot_fl_scale(d, out)
    return s


def heat(ax, piv, title, fmt="{:+.1f}", cmap="RdBu", center=0.0):
    v = piv.to_numpy(float)
    lim = np.nanmax(np.abs(v - center)) if np.isfinite(v).any() else 1
    ax.imshow(v, cmap=cmap, vmin=center - lim, vmax=center + lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    ax.set_xlabel("users per cell U"); ax.set_ylabel("cells N")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if np.isfinite(v[i, j]):
                ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title(title, fontsize=8)


def plot_fl_scale(d, out):
    aggs = [a for a in AGG_LABEL if a != "local" and a in set(d["agg"])]
    mal = d[d.scenario == "malicious"]
    if not mal.empty:
        fig, axs = plt.subplots(2, len(aggs), figsize=(3.6 * len(aggs), 6), squeeze=False)
        for j, agg in enumerate(aggs):
            m = mal[mal["agg"] == agg]
            if m.empty:
                continue
            heat(axs[0, j], m.pivot_table(index="n", columns="U", values="contagion_J_pct"),
                 f"{AGG_LABEL[agg]}: honest cells' J change\nwith one malicious cell (%)")
            heat(axs[1, j], m.pivot_table(index="n", columns="U", values="weight_on_attacker"),
                 f"{AGG_LABEL[agg]}: weight given to\nthe malicious cell (fair share = 1/N)", fmt="{:.2f}",
                 cmap="Reds", center=0.0)
        fig.suptitle("One malicious cell vs federation size (Python simulator, mean over seeds)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fl_scale_malicious.png"), dpi=150); plt.close(fig)
    clean = d[d.scenario == "clean"]
    if not clean.empty:
        us = sorted(clean.U.unique())
        fig, axs = plt.subplots(2, len(us), figsize=(3.0 * len(us), 5.2), squeeze=False)
        for j, u in enumerate(us):
            for agg in AGG_LABEL:
                v = clean[(clean["agg"] == agg) & (clean.U == u)]
                if v.empty:
                    continue
                for i, col in enumerate(("J_honest", "viol_honest")):
                    m = v.groupby("n")[col].agg(["mean", "std"])
                    axs[i, j].errorbar(m.index, m["mean"], m["std"], marker="o", ms=3, capsize=2,
                                       color=AGG_COL[agg], label=AGG_LABEL[agg])
            axs[0, j].set_title(f"U = {u}", fontsize=8)
            axs[1, j].set_xlabel("cells N")
            for i in range(2):
                axs[i, j].set_xscale("log", base=2); axs[i, j].set_xticks([1, 2, 4, 8], [1, 2, 4, 8])
                axs[i, j].spines[["top", "right"]].set_visible(False)
        axs[0, 0].set_ylabel("J (mean over cells)"); axs[1, 0].set_ylabel("users violating SLA (share)")
        axs[0, 0].legend(fontsize=6, frameon=False)
        fig.suptitle("Deployed model, online learning, no attack (Python simulator, mean +/- sd over seeds)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fl_scale_clean.png"), dpi=150); plt.close(fig)


def summarise_fl_learn(root, out):
    runs = load_fed(root, "fl_learn")
    if not runs:
        return None
    rows, curves = [], []
    for r in runs:
        ev = r["eval"][r["eval"].condition == "clean"]
        steps_per_round = r["cfg"]["args"]["iters"] * 2048
        c = ev.groupby("round")[["J", "any_violation"]].mean().reset_index()
        for _, x in c.iterrows():
            curves.append({"agg": r["agg"], "n": r["n"], "U": r["U"], "seed": r["seed"], "round": int(x["round"]),
                           "steps": int(x["round"]) * steps_per_round, "J": x.J, "any_violation": x.any_violation})
        rows.append({"agg": r["agg"], "n": r["n"], "U": r["U"], "seed": r["seed"],
                     "J_start": c.J.iloc[0], "J_end": c.J.iloc[-1],
                     "viol_start": c.any_violation.iloc[0], "viol_end": c.any_violation.iloc[-1],
                     "viol_mean_over_training": c.any_violation.mean(),
                     "runtime_min": r["cfg"].get("runtime_s", np.nan) / 60})
    d = pd.DataFrame(rows)
    loc = d[d["agg"] == "local"].set_index(["n", "U", "seed"])
    for col, new, f in (("J_end", "J_end_vs_local_pct", lambda a, b: 100 * (a / b - 1)),
                        ("viol_end", "viol_end_vs_local_pp", lambda a, b: 100 * (a - b)),
                        ("viol_mean_over_training", "viol_auc_vs_local_pp", lambda a, b: 100 * (a - b))):
        d[new] = [f(r[col], loc[col].get((r.n, r.U, r.seed), np.nan)) for _, r in d.iterrows()]
    d.to_csv(os.path.join(out, "fl_learn_runs.csv"), index=False)
    pd.DataFrame(curves).to_csv(os.path.join(out, "fl_learn_curves.csv"), index=False)
    num = [c for c in d.columns if c not in ("agg", "n", "U", "seed")]
    s = mean_sd(d, ["agg", "n", "U"], num).sort_values(["agg", "n", "U"])
    s.to_csv(os.path.join(out, "fl_learn_summary.csv"), index=False)
    cv = pd.DataFrame(curves)
    ns, us = sorted(cv.n.unique()), sorted(cv.U.unique())
    fig, axs = plt.subplots(len(ns), len(us), figsize=(2.8 * len(us), 2.2 * len(ns)), squeeze=False, sharex=True)
    for i, n in enumerate(ns):
        for j, u in enumerate(us):
            ax = axs[i, j]
            for agg in AGG_LABEL:
                v = cv[(cv["agg"] == agg) & (cv.n == n) & (cv.U == u)]
                if v.empty:
                    continue
                m = v.groupby("steps").any_violation.agg(["mean", "std"])
                ax.plot(m.index / 1000, m["mean"], marker="o", ms=2, color=AGG_COL[agg], label=AGG_LABEL[agg])
                ax.fill_between(m.index / 1000, m["mean"] - m["std"], m["mean"] + m["std"], color=AGG_COL[agg],
                                alpha=0.12)
            ax.set_title(f"N = {n}, U = {u}", fontsize=7); ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=6)
    axs[0, 0].legend(fontsize=6, frameon=False)
    for ax in axs[-1]:
        ax.set_xlabel("training steps per cell (thousands)", fontsize=7)
    for ax in axs[:, 0]:
        ax.set_ylabel("SLA violations", fontsize=7)
    fig.suptitle("Learning from scratch: users violating their SLA (Python simulator, mean +/- sd over seeds)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fl_learn_curves.png"), dpi=150); plt.close(fig)
    fed = d[d["agg"] != "local"]
    aggs = [a for a in AGG_LABEL if a in set(fed["agg"])]
    if aggs:
        fig, axs = plt.subplots(1, len(aggs), figsize=(3.6 * len(aggs), 3), squeeze=False)
        for j, agg in enumerate(aggs):
            heat(axs[0, j], fed[fed["agg"] == agg].pivot_table(index="n", columns="U", values="viol_auc_vs_local_pp"),
                 f"{AGG_LABEL[agg]} vs local: SLA violations over\ntraining (pp, negative = federation helps)")
        fig.tight_layout(); fig.savefig(os.path.join(out, "fl_learn_vs_local.png"), dpi=150); plt.close(fig)
    return s


# ===================================================================== single-cell comparison
def summarise_compare(root, out):
    files = glob.glob(os.path.join(root, "compare", "U*", "s*", "*.csv"))
    if not files:
        return None
    e = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    e.to_csv(os.path.join(out, "compare_episodes.csv"), index=False)
    cols = ["J", "utility", "any_violation", "viol_URLLC", "viol_eMBB", "cell_thr_mbps", "flag_rate"]
    # paired change against the same controller, users and seed without attack (same episode seeds)
    base = e[e.attack_key == "none"].set_index(["controller_key", "users", "train_seed", "episode_seed"])
    idx = list(zip(e.controller_key, e.users, e.train_seed, e.episode_seed))
    e["J_change_pct"] = 100 * (e.J.to_numpy() / base.J.reindex(idx).to_numpy() - 1)
    e["viol_change_pp"] = 100 * (e.any_violation.to_numpy() - base.any_violation.reindex(idx).to_numpy())
    # against undefended PPO under the same attack (same episodes)
    blind = e[e.controller_key == "ppo"].set_index(["attack_key", "users", "train_seed", "episode_seed"])
    idx2 = list(zip(e.attack_key, e.users, e.train_seed, e.episode_seed))
    e["J_vs_undefended_pct"] = 100 * (e.J.to_numpy() / blind.J.reindex(idx2).to_numpy() - 1)
    per_seed = e.groupby(["controller_key", "controller", "attack_key", "users", "train_seed"])[
        cols + ["J_change_pct", "viol_change_pp", "J_vs_undefended_pct"]].mean().reset_index()
    s = mean_sd(per_seed, ["controller_key", "controller", "attack_key", "users"],
                cols + ["J_change_pct", "viol_change_pp", "J_vs_undefended_pct"])
    s["c_ord"] = s.controller_key.map(CTRL_ORDER.index); s["a_ord"] = s.attack_key.map(ATT_ORDER.index)
    s = s.sort_values(["users", "c_ord", "a_ord"]).drop(columns=["c_ord", "a_ord"])
    s.to_csv(os.path.join(out, "compare_summary.csv"), index=False)
    # worst case over the attacks (per seed, then mean over seeds)
    att = per_seed[per_seed.attack_key != "none"]
    wi = att.loc[att.groupby(["controller_key", "users", "train_seed"]).J.idxmin()]
    wc = wi.groupby(["controller_key", "controller", "users"]).agg(
        worst_J=("J", "mean"), worst_J_sd=("J", "std"), worst_J_change_pct=("J_change_pct", "mean"),
        worst_viol=("any_violation", "mean"), seeds=("train_seed", "nunique"),
        worst_attack=("attack_key", lambda x: x.mode().iloc[0])).reset_index()
    clean = s[s.attack_key == "none"].set_index(["controller_key", "users"]).J
    wc["J_no_attack"] = [clean.get((c, u), np.nan) for c, u in zip(wc.controller_key, wc.users)]
    wc["c_ord"] = wc.controller_key.map(CTRL_ORDER.index)
    wc = wc.sort_values(["users", "c_ord"]).drop(columns="c_ord")
    wc.to_csv(os.path.join(out, "compare_worst_case.csv"), index=False)
    # one wide table: J per controller (rows) and attack (columns), one block per U
    wide = s.pivot_table(index=["users", "controller_key"], columns="attack_key", values="J").reindex(
        columns=[a for a in ATT_ORDER if a in set(s.attack_key)]).reset_index()
    wide["c_ord"] = wide.controller_key.map(CTRL_ORDER.index)
    wide = wide.sort_values(["users", "c_ord"]).drop(columns="c_ord")
    wide.to_csv(os.path.join(out, "compare_J_table.csv"), index=False)
    plot_compare(s, out)
    return s, wc, wide


def plot_compare(s, out):
    atts = [a for a in ATT_ORDER if a in set(s.attack_key)]
    ctrls = [c for c in CTRL_ORDER if c in set(s.controller_key)]
    cmap = plt.get_cmap("tab10")
    fig, axs = plt.subplots(3, len(atts), figsize=(3.0 * len(atts), 8.0), squeeze=False, sharey="row")
    rows = (("J", "J"), ("J_change_pct", "J change vs no attack (%)"), ("any_violation", "users violating SLA (share)"))
    for j, att in enumerate(atts):
        for i, (col, lab) in enumerate(rows):
            ax = axs[i, j]
            for k, c in enumerate(ctrls):
                v = s[(s.controller_key == c) & (s.attack_key == att)].sort_values("users")
                if v.empty:
                    continue
                ax.errorbar(v.users, v[col], v[col + "_sd"], marker="o", ms=3, capsize=2, color=cmap(k),
                            label=v.controller.iloc[0], lw=2.2 if c == "prop_b" else 1.0)
            ax.set_xticks(sorted(s.users.unique())); ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(ATT_LABEL[att], fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("users U")
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
    seen = {}
    for ax in axs.flat:                               # PF is missing from the gradient-attack panels
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    axs[1, -1].legend(seen.values(), seen.keys(), fontsize=6, frameon=False, loc="center left",
                      bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Every controller under every attack class, eps = 0.1 (Python simulator, mean +/- sd over "
                 "training seeds)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, "compare_by_users.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===================================================================== report
def write_tables(out, scale, learn, comp):
    parts = ["# Week 5: scaling and full comparison (Python simulator)",
             "All numbers come from the Python simulator (rra/env/pysim.py). Mean over training seeds; "
             "`_sd` is the standard deviation over seeds. Produced by `python -m experiments.week5_eval`.", ""]
    if comp is not None:
        s, wc, wide = comp
        parts += ["## Single cell: J of every controller under every attack (eps = 0.1)", "",
                  md_table(wide.rename(columns={"users": "U", "controller_key": "controller"})), "",
                  "## Worst case over the attacks", "",
                  md_table(wc[["users", "controller", "J_no_attack", "worst_J", "worst_J_sd", "worst_J_change_pct",
                               "worst_viol", "worst_attack", "seeds"]]), "",
                  "## Change under attack (paired episodes) and against undefended PPO", "",
                  md_table(s[s.attack_key != "none"][["users", "controller_key", "attack_key", "J", "J_change_pct",
                                                       "viol_change_pp", "J_vs_undefended_pct", "flag_rate",
                                                       "seeds"]]), ""]
    if scale is not None:
        keep = [c for c in ["scenario", "agg", "n", "U", "J_honest", "viol_honest", "J_vs_local_pct",
                            "viol_vs_local_pp", "contagion_J_pct", "contagion_viol_pp", "weight_on_attacker",
                            "fair_share", "upload_KB_per_round", "seeds"] if c in scale.columns]
        parts += ["## Federation size N x users U, deployed model (online learning)", "",
                  "`contagion_*`: honest cells with one malicious cell vs the same aggregator without attack. "
                  "`*_vs_local_*`: federated vs local-only learning with the same N, U and seed.", "",
                  md_table(scale[keep]), ""]
    if learn is not None:
        keep = [c for c in ["agg", "n", "U", "J_start", "J_end", "viol_start", "viol_end",
                            "viol_mean_over_training", "J_end_vs_local_pct", "viol_end_vs_local_pp",
                            "viol_auc_vs_local_pp", "seeds"] if c in learn.columns]
        parts += ["## Federation size N x users U, learning from scratch", "", md_table(learn[keep]), ""]
    open(os.path.join(out, "tables.md"), "w", encoding="utf-8").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/week5")
    ap.add_argument("--part", default="all", choices=["all", "fl_scale", "fl_learn", "compare"])
    a = ap.parse_args()
    out = os.path.join(a.root, "summary")
    os.makedirs(out, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    scale = summarise_fl_scale(a.root, out) if a.part in ("all", "fl_scale") else None
    learn = summarise_fl_learn(a.root, out) if a.part in ("all", "fl_learn") else None
    comp = summarise_compare(a.root, out) if a.part in ("all", "compare") else None
    write_tables(out, scale, learn, comp)
    for nm, x in (("fl_scale", scale), ("fl_learn", learn)):
        if x is not None:
            print(f"\n== {nm}\n{x.round(3).to_string(index=False)}")
    if comp is not None:
        print(f"\n== compare: J per controller and attack\n{comp[2].round(3).to_string(index=False)}")
        print(f"\n== compare: worst case\n{comp[1].round(3).to_string(index=False)}")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
