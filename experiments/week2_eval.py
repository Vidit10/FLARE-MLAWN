"""Week-2 (Phase 1) evaluation: clean PPO vs baselines, and attacks on both.

All numbers come from the Python simulator (rra/env/pysim.py), with the
log feature scaling (EnvConfig.feature_scale = "log").

E3  Clean comparison: EqualShare, PF, the untrained need prior alone
    ("NeedPrior": what the PPO policy starts from) and the trained PPO, at mean
    load U in {5, 10, 20, 30} (the training range) and U = 40 (outside it).
E4  Attacks at U = 20 on PF and PPO:
      common:   none, A2 random noise, A1 SINR inflation by k = 3 UEs,
                A2 query-based search (black-box, 16 queries per interval);
      PPO only: white-box gradient attacks (FGSM-RS and PGD-10 on the
                report's KL loss, PGD-10 on the model-based utility loss).
E5  Decision latency of the PPO actor (one CPU thread) for 5..40 users.

Evaluation episodes use seeds 1000+ (training never used them); PPO results
pool all trained seeds.
Usage: python -m experiments.week2_eval [--episodes 10] [--out results/week2]
       (or run --part e3 / e4common / e4grad in parallel, then --part combine)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rra.agents.policy import Actor  # noqa: E402
from rra.agents.ppo import PPOController  # noqa: E402
from rra.attacks import Attacker, GradientAttacker, eps_in_physical_units  # noqa: E402
from rra.baselines import EqualShare, ProportionalFair  # noqa: E402
from rra.config import EnvConfig  # noqa: E402
from rra.env.pysim import PySimEnv  # noqa: E402
from rra.runner import run_episode  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"PF": BLUE, "EqualShare": ORANGE, "PPO": AQUA, "NeedPrior": YELLOW}
MARKERS = {"EqualShare": "s", "PF": "o", "PPO": "D", "NeedPrior": "^"}

METRICS = ["utility", "power_frac", "prb_frac", "viol_eMBB", "viol_URLLC", "viol_mMTC",
           "any_violation", "cell_thr_mbps", "jain_eMBB", "p5_rate_eMBB_mbps",
           "urllc_p99_delay_ms", "dropped_frac"]


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
        return (float(x.mean()) if x.size else float("nan")), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


def objective(row, zeta=0.05):
    """Per-interval objective without the SLA terms: Phi - zeta_p P - zeta_n B."""
    return row["utility"] - zeta * row["power_frac"] - zeta * row["prb_frac"]


def run_many(env, make_ctrl_list, att_factory, eval_seeds, U):
    rows = []
    for ctrl in make_ctrl_list:
        for s in eval_seeds:
            att = att_factory() if att_factory else None
            r = run_episode(env, ctrl, att, seed=s, n_users=U)
            r["objective"] = objective(r)
            rows.append(r)
    return rows


def summarise(rows):
    keys = METRICS + ["objective"] + (["attacker_prb_ratio"] if "attacker_prb_ratio" in rows[0] else [])
    return {k: ci95([r[k] for r in rows]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--out", default="results/week2")
    ap.add_argument("--part", default="all", choices=["all", "e3", "e4common", "e4grad", "combine", "replot"],
                    help="run one part (parts can run in parallel); 'combine' merges E4 parts")
    ap.add_argument("--grad-split", type=int, default=0, choices=[0, 1, 2],
                    help="e4grad only: 1 = first half of the gradient attacks, 2 = second half")
    a = ap.parse_args()
    torch.set_num_threads(1)
    env = PySimEnv(EnvConfig(feature_scale="log"))
    eval_seeds = list(range(1000, 1000 + a.episodes))
    actor_paths = sorted(glob.glob(os.path.join(a.out, "ppo_s*", "actor.pt")))
    ppos = [PPOController.load(p) for p in actor_paths]
    print("PPO models:", actor_paths, flush=True)
    t0 = time.time()

    if a.part == "replot":       # redraw E3 and learning curves from the saved CSVs
        e3, loads = {}, []
        for r in csv.DictReader(open(os.path.join(a.out, "e3_clean.csv"))):
            U = int(r["U"]); loads += [U] if U not in loads else []
            e3[(r["controller"], U)] = {k[:-5]: (float(r[k]), float(r[k[:-5] + "_ci95"]))
                                        for k in r if k.endswith("_mean")}
        plot_e3(e3, loads, os.path.join(a.out, "e3_clean.png"))
        plot_learning(sorted(glob.glob(os.path.join(a.out, "ppo_s*", "train_log.csv"))),
                      os.path.join(a.out, "learning_curves.png"))
        return
    if a.part == "combine":
        e4 = {}
        for part in ("e4common", "e4grad", "e4grad1", "e4grad2"):
            fp = os.path.join(a.out, f"_{part}.json")
            if os.path.exists(fp):
                for row in json.load(open(fp)):
                    e4[(row["controller"], row["attack"])] = {k: tuple(v) for k, v in row["stats"].items()}
        write_e4(e4, a.out)
        return
    if a.part in ("e4common", "e4grad"):
        run_e4(a, env, ppos, eval_seeds, t0)
        return

    # ---------------------------------------------------------------- E3
    loads = [5, 10, 20, 30, 40]
    e3 = {}
    for U in loads:
        e3[("EqualShare", U)] = summarise(run_many(env, [EqualShare(env)], None, eval_seeds, U))
        e3[("PF", U)] = summarise(run_many(env, [ProportionalFair(env)], None, eval_seeds, U))
        e3[("NeedPrior", U)] = summarise(run_many(env, [PPOController(Actor(prior="need"))], None,
                                                   eval_seeds, U))
        e3[("PPO", U)] = summarise(run_many(env, ppos, None, eval_seeds, U))
        print(f"E3 U={U} done {time.time() - t0:.0f}s", flush=True)
    keys = METRICS + ["objective"]
    with open(os.path.join(a.out, "e3_clean.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "U"] + [f"{k}{s}" for k in keys for s in ("_mean", "_ci95")])
        for (name, U), d in e3.items():
            w.writerow([name, U] + [f"{v:.4g}" for k in keys for v in d[k]])
    plot_e3(e3, loads, os.path.join(a.out, "e3_clean.png"))

    if a.part == "all":
        for part in ("e4common", "e4grad"):
            a.part = part
            run_e4(a, env, ppos, eval_seeds, t0)
        a.part = "all"
        e4 = {}
        for part in ("e4common", "e4grad"):
            for row in json.load(open(os.path.join(a.out, f"_{part}.json"))):
                e4[(row["controller"], row["attack"])] = {k: tuple(v) for k, v in row["stats"].items()}
        write_e4(e4, a.out)

    # ---------------------------------------------------------------- E5
    lat = {}
    for Ul in (5, 10, 20, 30, 40):
        obs = env.reset(seed=7, n_users=Ul, churn=False)
        ctrl = ppos[0]
        for _ in range(20):
            ctrl.act(obs)
        ts = []
        for _ in range(300):
            t = time.perf_counter(); ctrl.act(obs); ts.append(time.perf_counter() - t)
        lat[Ul] = {"median_ms": float(np.median(ts) * 1e3), "p99_ms": float(np.percentile(ts, 99) * 1e3)}

    plot_learning(sorted(glob.glob(os.path.join(a.out, "ppo_s*", "train_log.csv"))),
                  os.path.join(a.out, "learning_curves.png"))
    meta = {"episodes_per_setting": a.episodes, "eval_seeds": eval_seeds, "ppo_models": actor_paths,
            "feature_scale": "log", "eps_physical": {e: eps_in_physical_units(env.bounds, e)
                                                     for e in (0.02, 0.05, 0.1)},
            "ppo_decision_latency_1thread": lat, "runtime_s": round(time.time() - t0, 1),
            "simulator": "rra.env.pysim (Python)"}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


def attack_list(part):
    if part == "e4common":
        L = [("none", None)]
        L.append(("A2 noise e=0.05", lambda: Attacker(eps=0.05, point="A2", strategy="noise")))
        L.append(("A1 SINR+ k=3 e=0.1", lambda: Attacker(
            eps=0.1, point="A1", k=3, strategy="direction", features=["sinr_db"],
            direction={"sinr_db": +1})))
        for e in (0.02, 0.05, 0.1):
            L.append((f"A2 search e={e:g}",
                      (lambda e=e: Attacker(eps=e, point="A2", strategy="search", n_candidates=16))))
        return L
    L = []
    for e in (0.02, 0.05, 0.1):
        L.append((f"A2 FGSM-KL e={e:g}", (lambda e=e: GradientAttacker(eps=e, strategy="fgsm", objective="kl"))))
        L.append((f"A2 PGD-KL e={e:g}", (lambda e=e: GradientAttacker(eps=e, strategy="pgd", objective="kl"))))
        L.append((f"A2 PGD-util e={e:g}", (lambda e=e: GradientAttacker(eps=e, strategy="pgd", objective="util"))))
    L.append(("A1 PGD-KL k=3 e=0.1", lambda: GradientAttacker(eps=0.1, point="A1", k=3,
                                                              strategy="pgd", objective="kl")))
    return L


def run_e4(a, env, ppos, eval_seeds, t0, U=20):
    """Attacks at U = 20.  'e4common' runs on PF and PPO, 'e4grad' on PPO only."""
    pf = ProportionalFair(env)
    rows = []
    attacks = attack_list(a.part)
    if a.part == "e4grad" and a.grad_split:
        h = (len(attacks) + 1) // 2
        attacks = attacks[:h] if a.grad_split == 1 else attacks[h:]
    tag = a.part + (f"{a.grad_split}" if a.part == "e4grad" and a.grad_split else "")
    for name, fac in attacks:
        targets = [("PF", [pf]), ("PPO", ppos)] if a.part == "e4common" else [("PPO", ppos)]
        for ctrl, cl in targets:
            st = summarise(run_many(env, cl, fac, eval_seeds, U))
            rows.append({"controller": ctrl, "attack": name, "stats": st})
        print(f"E4 {name} done {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(a.out, f"_{tag}.json"), "w") as f:
            json.dump(rows, f)


def write_e4(e4, out):
    keys4 = METRICS + ["objective", "attacker_prb_ratio"]
    with open(os.path.join(out, "e4_attacks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller", "attack"] + [f"{k}{s}" for k in keys4 for s in ("_mean", "_ci95")])
        for (ctrl, name), d in e4.items():
            w.writerow([ctrl, name] + [f"{v:.4g}" for k in keys4 for v in d.get(k, (float("nan"),) * 2)])
    plot_e4(e4, os.path.join(out, "e4_attacks.png"))


def plot_e3(res, loads, path):
    panels = [("utility", "Utility $\\Phi$ (higher is better)"),
              ("viol_eMBB", "eMBB SLA violation rate"),
              ("viol_URLLC", "URLLC SLA violation rate"),
              ("power_frac", "Power used (fraction of $P_{max}$)")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (k, title) in zip(axs, panels):
        for name in ("PF", "EqualShare", "PPO", "NeedPrior"):
            m = [res[(name, U)][k][0] for U in loads]
            c = [res[(name, U)][k][1] for U in loads]
            ax.errorbar(loads, m, yerr=c, color=COLORS[name], lw=2 if name != "NeedPrior" else 1.2,
                        ls="-" if name != "NeedPrior" else "--", marker=MARKERS[name], ms=6,
                        capsize=3, label=name if name != "NeedPrior" else "need prior only (untrained)")
        ax.axvspan(35, 42, color=GRID, alpha=0.6, lw=0)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel("mean number of users U (shaded: outside training range)", fontsize=7, color=INK2)
        ax.set_xticks(loads)
        style(ax)
    axs[2].axhline(0.01, color=INK2, lw=1, ls=":")
    axs[2].annotate("target 1%", xy=(40, 0.01), xytext=(26, 0.35), fontsize=7, color=INK2,
                    arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.6})
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Clean comparison in the Python simulator (mean and 95% CI over evaluation episodes)",
                 fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_e4(res, path):
    names = []
    for (_, n) in res:
        if n not in names:
            names.append(n)
    fig, axs = plt.subplots(1, 2, figsize=(12, 0.42 * len(names) + 1.9))
    y = np.arange(len(names))[::-1]
    for ax, k in zip(axs, ("utility", "any_violation")):
        for ctrl in ("PF", "PPO"):
            base = res[(ctrl, "none")][k][0]
            m, c = [], []
            for n in names:
                v, ci = res.get((ctrl, n), {k: (np.nan, 0)})[k]
                if k == "utility":      # change relative to the same controller without attack
                    # no error bar here: the episode-to-episode spread (in the CSV) is not
                    # the uncertainty of a paired difference on identical episodes
                    m.append(100 * (v / base - 1)); c.append(0.0)
                else:
                    m.append(v); c.append(ci)
            ax.barh(y + (0.2 if ctrl == "PF" else -0.2), m, xerr=c, height=0.38,
                    color=COLORS[ctrl], label=ctrl, capsize=2, error_kw={"ecolor": INK2, "lw": 0.8})
            if k == "any_violation":
                ax.axvline(base, color=COLORS[ctrl], lw=1, ls="--")
        if k == "utility":
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_title("Utility $\\Phi$: change vs. the same controller without attack (%)",
                         fontsize=9, color=INK)
            for yi, n in zip(y, names):
                for ctrl, dy in (("PF", 0.2), ("PPO", -0.2)):
                    if (ctrl, n) in res and n != "none":
                        v = 100 * (res[(ctrl, n)][k][0] / res[(ctrl, "none")][k][0] - 1)
                        ax.text(min(v, 0) - 0.15, yi + dy, f"{v:.1f}", fontsize=6.5, color=INK2,
                                va="center", ha="right")
            lo = ax.get_xlim()[0]
            ax.set_xlim(left=lo - 1.2)
        else:
            ax.set_title("Share of users violating their SLA (dashed: without attack)", fontsize=9, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels(names if ax is axs[0] else [], fontsize=8)
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("PF and PPO under observation attacks, U = 20 (Python simulator; same evaluation "
                 "episodes in every row; gradient attacks apply to PPO only)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _smooth(y, w=15):
    y = np.asarray(y, float)
    if y.size < w:
        return y
    c = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


def plot_learning(paths, path):
    if not paths:
        return
    panels = [("utility", "Utility $\\Phi$ (training rollouts)"),
              ("viol_eMBB", "eMBB violation rate"), ("viol_URLLC", "URLLC violation rate"),
              ("lambda", "Multipliers $\\lambda$")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    for i, p in enumerate(paths):
        rows = list(csv.DictReader(open(p)))
        steps = np.array([float(r["steps"]) for r in rows]) / 1e6
        for ax, (k, title) in zip(axs, panels):
            if k == "lambda":
                for kk, col in (("lambda_eMBB", BLUE), ("lambda_URLLC", ORANGE), ("lambda_mMTC", AQUA)):
                    ax.plot(steps, [float(r[kk]) for r in rows], color=col, lw=1.5,
                            label=kk.split("_")[1] if i == 0 else None)
            else:
                y = [float(r[k]) for r in rows]
                ax.plot(steps, y, color=GRID, lw=0.6)
                ax.plot(steps, _smooth(y), color=INK2, lw=1.4)
            ax.set_title(title, fontsize=9, color=INK)
            ax.set_xlabel("environment steps (millions)", fontsize=8, color=INK2)
            style(ax)
    axs[2].axhline(0.01, color=INK, lw=0.8, ls=":")
    axs[1].axhline(0.05, color=INK, lw=0.8, ls=":")
    axs[3].legend(frameon=False, fontsize=8)
    fig.suptitle(f"PPO training, {len(paths)} seeds (light: per iteration; dark: 15-iteration "
                 "moving average; dotted: SLA target)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
