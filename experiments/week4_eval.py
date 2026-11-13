"""Week-4 analysis: tables and figures from the federated runs and the observer runs.

Reads results/week4/{e8,e9}/<agg>_k<k>m<m>_s<seed>/ (experiments/week4_fed.py)
and results/week4/observer/s<seed>/ (experiments/week4_observer.py); writes
results/week4/summary/.

Measures (FLARE report, Table "Performance metrics"; deviations noted):
  J            Phi - zeta_p P - zeta_n B per cell, averaged over the evaluation
               episodes (true KPIs; SLAs are a separate guard).
  normalisation  The report normalises J between the proportional-fair scheduler
               (0) and a clean, well-trained allocator (1). In our simulator PF has
               the HIGHER J in every cell (it does not protect the SLAs, which J
               leaves out), so that scale is not defined. We use equal shares as
               the floor instead: J^ = (J - J_ES) / (J_ref - J_ES), with J_ref the
               same aggregator's run without any affected cell. PF is reported
               separately with its SLA violations.
  contagion    mean over cells that are neither deceived nor malicious of
               (1 - J^), i.e. the share of the clean-to-equal-share gap lost because
               other cells were affected (0 = no harm). Also given as % change of J.
  weight on affected cells   sum of the aggregator's weights a_c over deceived and
               malicious cells, averaged over rounds.
  honest cells kept   share of (round, honest cell) pairs with C * a_c >= 0.5.
  data-trust AUC   how well D_c separates deceived cells from the others (per round).
  federation gain  % change of mean J, federated vs local-only, no attack (E8, E9).
  training steps saved  steps (per cell) until 90 % of the local-only gain is reached,
               federated vs local-only (E8).
Usage: python -m experiments.week4_eval --part all
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

AGG_LABEL = {"local": "local only", "fedavg": "FedAvg", "median": "coordinate median", "fltrust": "FLTrust",
             "two_tier": "two-tier trust (ours)", "two_tier_no_data": "ours without data trust",
             "two_tier_no_drift": "ours without drift term"}
AGG_ORDER = list(AGG_LABEL)
C = 8


def auc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    ranks = pd.Series(allv).rank().to_numpy()
    rp = ranks[neg.size:].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def load_runs(root, exp):
    runs = []
    for d in sorted(glob.glob(os.path.join(root, exp, "*_s*"))):
        m = re.match(r"(.+)_k(\d+)m(\d+)(?:_eps([\d.]+))?_s(\d+)$", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "eval.csv")):
            continue
        cfg = json.load(open(os.path.join(d, "config.json")))
        ev = pd.read_csv(os.path.join(d, "eval.csv"))
        rd = pd.read_csv(os.path.join(d, "rounds.csv")) if os.path.exists(os.path.join(d, "rounds.csv")) else None
        if ev["round"].max() < cfg["args"]["rounds"]:
            print(f"skip unfinished {d}")
            continue
        runs.append({"dir": d, "agg": m.group(1), "k": int(m.group(2)), "m": int(m.group(3)),
                     "eps": float(m.group(4)) if m.group(4) else 0.1, "seed": int(m.group(5)),
                     "cfg": cfg, "eval": ev, "rounds": rd})
    return runs


def references(root, seeds, n_ep=4):
    """J of equal shares and PF on the same evaluation episodes (cached)."""
    out = os.path.join(root, "summary", "references.csv")
    if os.path.exists(out):
        return pd.read_csv(out)
    from rra.baselines import EqualShare, ProportionalFair
    from rra.env.pysim import PySimEnv
    from rra.fed.cells import CELLS, cell_env_config
    from rra.runner import run_episode
    rows = []
    for i, spec in enumerate(CELLS):
        env = PySimEnv(cell_env_config(spec))
        rng = np.random.default_rng(7000 + i)
        for j in range(n_ep):
            s = 7000 + 100 * i + j
            U = int(rng.integers(spec.users[0], spec.users[1] + 1))
            for name, ctrl in (("equal share", EqualShare(env)), ("PF", ProportionalFair(env))):
                r = run_episode(env, ctrl, None, seed=s, n_users=U)
                rows.append({"controller": name, "cell": i, "episode": j,
                             "J": r["utility"] - 0.05 * r["power_frac"] - 0.05 * r["prb_frac"],
                             "utility": r["utility"], "any_violation": r["any_violation"],
                             "viol_URLLC": r["viol_URLLC"]})
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    d.to_csv(out, index=False)
    return d


def cell_J(run, rnd=None, cond="clean"):
    ev = run["eval"]
    rnd = ev["round"].max() if rnd is None else rnd
    e = ev[(ev["round"] == rnd) & (ev.condition == cond)]
    return e.groupby("cell")[["J", "any_violation", "viol_URLLC", "cell_thr_mbps", "jain_eMBB"]].mean()


# ===================================================================== E9
def summarise_e9(root, refs):
    runs = load_runs(root, "e9")
    if not runs:
        return None
    es = refs[refs.controller == "equal share"].groupby("cell").J.mean()
    pf = refs[refs.controller == "PF"].groupby("cell")[["J", "any_violation"]].mean()
    by = {(r["agg"], r["k"], r["m"], r["seed"]): r for r in runs}
    rows, trust_rows = [], []
    for r in runs:
        ref = by.get((r["agg"], 0, 0, r["seed"]))
        if ref is None:
            continue
        aff = set(r["cfg"]["deceived"]) | set(r["cfg"]["malicious"])
        Jr, J0 = cell_J(r), cell_J(ref)
        start = cell_J(r, 0)
        honest = [c for c in Jr.index if c not in aff]
        jhat = (Jr.J - es) / (J0.J - es)
        row = {"agg": r["agg"], "k": r["k"], "m": r["m"], "eps": r["eps"], "seed": r["seed"],
               "contagion": float((1 - jhat[honest]).mean()),
               "contagion_J_pct": float((100 * (Jr.J[honest] / J0.J[honest] - 1)).mean()),
               "honest_J_change_vs_start_pct": float((100 * (Jr.J[honest] / start.J[honest] - 1)).mean()),
               "contagion_viol_pp": float((100 * (Jr.any_violation[honest] - J0.any_violation[honest])).mean()),
               "honest_any_violation": float(Jr.any_violation[honest].mean()),
               "honest_viol_URLLC": float(Jr.viol_URLLC[honest].mean())}
        if r["cfg"]["deceived"]:
            att = cell_J(r, cond="attacked")
            dec = r["cfg"]["deceived"]
            row["deceived_J_attacked_vs_ref_pct"] = float((100 * (att.J[dec] / J0.J[dec] - 1)).mean())
        rd = r["rounds"]
        if rd is not None:
            w = rd.pivot_table(index="round", columns="cell", values="weight")
            has_w = w.shape[1] == C and w.notna().all().all()   # local-only runs have no weights
            if has_w and aff:
                row["weight_on_affected"] = float(w[list(aff)].sum(axis=1).mean())
            if has_w:
                hw = w[[c for c in w.columns if c not in aff]]
                row["honest_kept"] = float((C * hw >= 0.5).to_numpy().mean())
                row["honest_min_Ca"] = float((C * hw).to_numpy().min())
            if r["cfg"]["deceived"]:
                dec = set(r["cfg"]["deceived"])
                pos = rd[rd.cell.isin(dec)].data_trust
                neg = rd[~rd.cell.isin(aff)].data_trust
                trust_rows.append({"agg": r["agg"], "k": r["k"], "seed": r["seed"], "auc_data_trust": 1 - auc(neg, pos),
                                   "auc_rho_only": 1 - auc(rd[~rd.cell.isin(aff)].rho_bar, rd[rd.cell.isin(dec)].rho_bar),
                                   "D_deceived": float(pos.mean()), "D_honest": float(neg.mean())})
        rows.append(row)
    d = pd.DataFrame(rows)
    out = os.path.join(root, "summary")
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "e9_runs.csv"), index=False)
    num = [c for c in d.columns if c not in ("agg", "k", "m", "eps", "seed")]
    g = d.groupby(["agg", "k", "m", "eps"])
    s = g[num].mean().join(g[num].std().add_suffix("_sd")).join(g.seed.nunique().rename("seeds")).reset_index()
    s["agg_order"] = s["agg"].map(AGG_ORDER.index)
    s = s.sort_values(["agg_order", "eps", "m", "k"]).drop(columns="agg_order")
    s.to_csv(os.path.join(out, "e9_summary.csv"), index=False)
    if trust_rows:
        t = pd.DataFrame(trust_rows)
        t.groupby(["agg", "k"]).mean(numeric_only=True).reset_index().to_csv(os.path.join(out, "e9_data_trust.csv"),
                                                                             index=False)
    pfrow = pd.DataFrame({"cell": es.index, "J_equal_share": es.values, "J_PF": pf.J.values,
                          "PF_any_violation": pf.any_violation.values})
    pfrow.to_csv(os.path.join(out, "references_per_cell.csv"), index=False)
    return s


def plot_e9(root):
    p = os.path.join(root, "summary", "e9_summary.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    cols = dict(zip(AGG_ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for agg in AGG_ORDER:
        v = s[(s["agg"] == agg) & (s.m == 0) & (s.eps == 0.1)]
        if len(v) > 1:
            ax[0].errorbar(v.k, v.contagion, yerr=v.contagion_sd, marker="o", ms=3, capsize=2, color=cols[agg],
                           label=AGG_LABEL[agg])
            if "weight_on_affected" in v and v.weight_on_affected.notna().any():
                ax[1].plot(v.k, v.weight_on_affected, marker="o", ms=3, color=cols[agg])
    ax[0].axvspan(3.5, 4.5, color="#eee", zorder=0)
    ax[0].set_title("contagion in honest cells (1 - J^), deceived cells", fontsize=9)
    ax[1].plot([0, 4], [0, 4 / C], ls=":", color="#999", label="equal weights")
    ax[1].set_title("total weight given to deceived cells", fontsize=9)
    for x in ax[:2]:
        x.set_xlabel("deceived cells out of 8 (shaded: beyond the median's breakdown point)")
        x.set_xticks(range(5)); x.spines[["top", "right"]].set_visible(False)
    ax[0].axhline(0, color="#999", lw=0.8); ax[0].legend(fontsize=7, frameon=False)
    mal = s[(s.m == 1) & (s.k == 0) & (s.eps == 0.1)]
    mal = mal.set_index("agg").reindex([a for a in AGG_ORDER if a in set(mal["agg"])])
    ax[2].barh(range(len(mal)), mal.contagion, xerr=mal.contagion_sd, color=[cols[a] for a in mal.index])
    ax[2].set_yticks(range(len(mal))); ax[2].set_yticklabels([AGG_LABEL[a] for a in mal.index], fontsize=8)
    ax[2].invert_yaxis(); ax[2].axvline(0, color="#999", lw=0.8)
    ax[2].set_title("contagion with one malicious cell (update x8)", fontsize=9)
    ax[2].spines[["top", "right"]].set_visible(False)
    fig.suptitle("Federation of 8 cells, deployed model learning online (Python simulator, mean +/- sd over seeds)",
                 fontsize=9, y=1.02)
    fig.savefig(os.path.join(root, "summary", "e9_contagion.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ===================================================================== E8
def summarise_e8(root, refs):
    runs = load_runs(root, "e8")
    if not runs:
        return None
    rows = []
    for r in runs:
        ev = r["eval"]; e = ev[ev.condition == "clean"]
        steps_per_round = r["cfg"]["args"]["iters"] * 2048
        for (rnd, cell), g in e.groupby(["round", "cell"]):
            rows.append({"agg": r["agg"], "k": r["k"], "seed": r["seed"], "round": rnd, "steps": rnd * steps_per_round,
                         "cell": cell, "J": g.J.mean(), "any_violation": g.any_violation.mean(),
                         "deceived": cell in r["cfg"]["deceived"]})
    d = pd.DataFrame(rows)
    out = os.path.join(root, "summary")
    os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, "e8_curves.csv"), index=False)
    res = []
    for seed in sorted(d.seed.unique()):
        loc = d[(d["agg"] == "local") & (d.k == 0) & (d.seed == seed)]
        if loc.empty:
            continue
        last = loc["round"].max()
        for agg in d["agg"].unique():
            for k in sorted(d.k.unique()):
                f = d[(d["agg"] == agg) & (d.k == k) & (d.seed == seed)]
                if f.empty:
                    continue
                honest = f[~f.deceived]
                cells = sorted(honest.cell.unique())
                gain, saved, vsaved, vgain = [], [], [], []
                for c in cells:
                    lcv = loc[loc.cell == c].sort_values("round"); fcv = honest[honest.cell == c].sort_values("round")
                    v0, vl = lcv.any_violation.iloc[0], lcv[lcv["round"] == last].any_violation.iloc[0]
                    vf = fcv[fcv["round"] == last].any_violation.iloc[0]
                    vgain.append(100 * (vf - vl))
                    if vl < v0:                         # local learning cut the violations: when does each get 90 % there?
                        target = v0 - 0.9 * (v0 - vl)
                        vsaved.append((lcv[lcv.any_violation <= target].steps.min(),
                                       fcv[fcv.any_violation <= target].steps.min()))
                    lc = loc[loc.cell == c].sort_values("round"); fc = honest[honest.cell == c].sort_values("round")
                    j0, jl = lc.J.iloc[0], lc[lc["round"] == last].J.iloc[0]
                    jf = fc[fc["round"] == last].J.iloc[0]
                    gain.append(100 * (jf / jl - 1))
                    if jl > j0:
                        target = j0 + 0.9 * (jl - j0)
                        sl = lc[lc.J >= target].steps.min(); sf = fc[fc.J >= target].steps.min()
                        saved.append((sl, sf))
                sl = np.array([x[0] for x in saved], float); sf = np.array([x[1] for x in saved], float)
                vl_ = np.array([x[0] for x in vsaved], float); vf_ = np.array([x[1] for x in vsaved], float)
                res.append({"agg": agg, "k": k, "seed": seed, "gain_vs_local_pct": float(np.mean(gain)),
                            "viol_vs_local_pp": float(np.mean(vgain)),
                            "cells_viol_improved_locally": len(vsaved),
                            "steps_to_90pct_viol_local": float(np.nanmean(vl_)) if vsaved else np.nan,
                            "steps_to_90pct_viol_this": float(np.nanmean(vf_)) if vsaved else np.nan,
                            "reached_90pct_viol_share": float(np.mean(np.isfinite(vf_))) if vsaved else np.nan,
                            "cells_that_improved_locally": len(saved),
                            "steps_to_90pct_local": float(np.nanmean(sl)) if saved else np.nan,
                            "steps_to_90pct_this": float(np.nanmean(sf)) if saved else np.nan,
                            "reached_90pct_share": float(np.mean(np.isfinite(sf))) if saved else np.nan,
                            "local_gain_from_start_pct": float(np.mean(
                                [100 * (loc[(loc.cell == c) & (loc["round"] == last)].J.iloc[0]
                                        / loc[(loc.cell == c) & (loc["round"] == 0)].J.iloc[0] - 1) for c in cells]))})
    s = pd.DataFrame(res)
    if not s.empty:
        g = s.groupby(["agg", "k"])
        num = [c for c in s.columns if c not in ("agg", "k", "seed")]
        s2 = g[num].mean().join(g[num].std().add_suffix("_sd")).join(g.seed.nunique().rename("seeds")).reset_index()
        s2.to_csv(os.path.join(out, "e8_summary.csv"), index=False)
        return s2
    return None


def plot_e8(root):
    p = os.path.join(root, "summary", "e8_curves.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    d = d[~d.deceived]
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
    cols = dict(zip(AGG_ORDER, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for row, (col, lab) in enumerate((("any_violation", "users violating their SLA (share)"),
                                      ("J", "J (mean over honest cells)"))):
        for k, x in zip((0, 3), axs[row]):
            for agg in AGG_ORDER:
                v = d[(d["agg"] == agg) & (d.k == k)]
                if v.empty:
                    continue
                m = v.groupby(["seed", "steps"])[col].mean().groupby("steps").agg(["mean", "std"])
                x.plot(m.index / 1000, m["mean"], marker="o", ms=3, color=cols[agg], label=AGG_LABEL[agg])
                x.fill_between(m.index / 1000, m["mean"] - m["std"], m["mean"] + m["std"], color=cols[agg], alpha=0.12)
            if row == 0:
                x.set_title("no attack" if k == 0 else f"{k} of 8 cells deceived (honest cells shown)", fontsize=9)
            else:
                x.set_xlabel("training steps per cell (thousands)")
            x.spines[["top", "right"]].set_visible(False)
        axs[row, 0].set_ylabel(lab, fontsize=8)
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Learning from scratch: federated vs local (Python simulator, mean +/- sd over seeds)", fontsize=9, y=1.02)
    fig.savefig(os.path.join(root, "summary", "e8_learning.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ===================================================================== observer
def summarise_observer(root):
    out = os.path.join(root, "summary")
    os.makedirs(out, exist_ok=True)
    cl = [pd.read_csv(p).assign(seed=int(re.search(r"s(\d+)", p).group(1)))
          for p in glob.glob(os.path.join(root, "observer", "s*", "classify.csv"))]
    res = {}
    if cl:
        d = pd.concat(cl, ignore_index=True)
        d["kind"] = np.where(d.event.str.startswith("stealthy"), "stealthy attack", d.truth)
        conf = pd.crosstab(d.kind, d.decision)
        conf.to_csv(os.path.join(out, "observer_confusion.csv"))
        sh = d[d.truth == "shift"]
        res["shift_match_accuracy"] = float((sh.nearest_cell == sh.true_target).mean())
        res["confusion"] = conf
    ad = [pd.read_csv(p).assign(seed=int(re.search(r"s(\d+)", p).group(1)))
          for p in glob.glob(os.path.join(root, "observer", "s*", "adapt.csv"))]
    if ad:
        d = pd.concat(ad, ignore_index=True)
        d.to_csv(os.path.join(out, "observer_adapt_all.csv"), index=False)
        rows = []
        sc = "lagrangian_score" if "lagrangian_score" in d else "J"
        for (seed, cell), g in d.groupby(["seed", "cell"]):
            # the score can be near zero or negative, so the target is set on the absolute scale:
            # 90 % of the way from the worst starting score to the best final score of any strategy
            level = g[g.iteration == g.iteration.max()][sc].max()
            lo = g[g.iteration == 0][sc].min()
            target = lo + 0.9 * (level - lo)
            for strat, h in g.groupby("strategy"):
                h = h.sort_values("iteration")
                ok = h[h[sc] >= target]
                rows.append({"seed": seed, "cell": cell, "strategy": strat,
                             "score_after_shift": h[sc].iloc[0], "score_end": h[sc].iloc[-1],
                             "gap_to_best_end": level - h[sc].iloc[-1],
                             "J_after_shift": h.J.iloc[0], "J_end": h.J.iloc[-1],
                             "violation_after_shift": h.any_violation.iloc[0] if "any_violation" in h else np.nan,
                             "violation_end": h.any_violation.iloc[-1] if "any_violation" in h else np.nan,
                             "intervals_to_90pct": float(ok.intervals.min()) if len(ok) else np.nan})
        a = pd.DataFrame(rows)
        a.groupby("strategy").mean(numeric_only=True).drop(columns=["seed", "cell"]).to_csv(
            os.path.join(out, "observer_adapt.csv"))
        res["adapt"] = a
    return res


def plot_observer(root):
    out = os.path.join(root, "summary")
    p = os.path.join(out, "observer_adapt_all.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        pairs = list(d.groupby(["cell", "target_cell"]).groups)
        fig, ax = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 3.2), sharey=False)
        ax = np.atleast_1d(ax)
        for x, (c, t) in zip(ax, pairs):
            v = d[(d.cell == c) & (d.target_cell == t)]
            sc = "lagrangian_score" if "lagrangian_score" in v else "J"
            for strat, g in v.groupby("strategy"):
                m = g.groupby("iteration")[sc].mean()
                x.plot(m.index * 2048 / 1000, m.values, marker="o", ms=3, label=strat)
            x.set_title(f"{v.new_traffic.iloc[0]} traffic arrives in cell {c}", fontsize=8)
            x.set_xlabel("intervals after the shift (thousands)"); x.spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("J - sum lambda V (Lagrangian score)"); ax[0].legend(fontsize=7, frameon=False)
        fig.suptitle("Adaptation after a genuine traffic shift (Python simulator, mean over seeds)", fontsize=9, y=1.03)
        fig.savefig(os.path.join(out, "observer_adapt.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    tl = sorted(glob.glob(os.path.join(root, "observer", "s*", "retrain_timeline.csv")))
    if tl:
        d = pd.read_csv(tl[0])
        ev = json.load(open(os.path.join(os.path.dirname(tl[0]), "retrain_events.json")))
        fig, ax = plt.subplots(figsize=(9, 3.4))
        cmap = {"clean": "#4477aa", "new attack": "#cc6677", "faulty model forced": "#999933"}
        for ph, g in d.groupby("phase", sort=False):
            ax.plot(g.window, g.J, marker="o", ms=4, color=cmap.get(ph, "k"), label=ph)
        ax.plot(d.window, 0.9 * d.level, ls=":", color="#999", label="drop threshold (0.9 x running level)")
        for e in ev:
            ax.axvline(e["window"] + 0.5, color="#555", lw=0.8, ls="--")
            ax.text(e["window"] + 0.6, ax.get_ylim()[1], e["event"], fontsize=7, va="top", rotation=90)
        ax.set_xlabel("window (one episode of 300 intervals)"); ax.set_ylabel("J of the cell")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=7, frameon=False, loc="lower left")
        ax.set_title("Observer (Step G): drop, attack diagnosis, twin retraining, shadow test, rollback "
                     "(one cell, Python simulator)", fontsize=9)
        fig.savefig(os.path.join(out, "observer_timeline.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "e8", "e9", "observer", "plot"])
    ap.add_argument("--root", default="results/week4")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    refs = references(a.root, a.seeds)
    if a.part in ("all", "e9"):
        s = summarise_e9(a.root, refs)
        if s is not None:
            print(s.round(4).to_string(index=False))
    if a.part in ("all", "e8"):
        s = summarise_e8(a.root, refs)
        if s is not None:
            print(s.round(4).to_string(index=False))
    if a.part in ("all", "observer"):
        r = summarise_observer(a.root)
        if "confusion" in r:
            print(r["confusion"].to_string())
    plot_e9(a.root); plot_e8(a.root); plot_observer(a.root)


if __name__ == "__main__":
    main()
