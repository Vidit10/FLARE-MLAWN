"""Week-3 (Phase 3) evaluation: the defence against non-adaptive and adaptive attacks.

All numbers come from the Python simulator (rra/env/pysim.py), log feature scaling.

Parts (--part):
  detect  Detector quality: per-user detection score of clean vs attacked reports
          (AUC, true-positive rate at the conformal threshold, false-positive rate).
  t2      Through-sanitizer targeted attack (report F9, tier T2): can an attacker
          steer the *sanitized* report to a chosen target? Deterministic vs
          randomized sanitizer, attacker with Q = 1 or Q = 8 noise draws per step.
  e6      Main table: every controller under every attack at U = 20, on the same
          episodes (seeds 1000+). Per-episode values are stored, so changes are
          paired (same episodes with and without attack).
  plot    Figures from the saved CSVs.
  all     detect, t2, e6, plot.

Usage: python -m experiments.week3_eval --part all [--episodes 6] [--workers N] [--seed 0]
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
import torch  # noqa: E402

from rra.agents.policy import obs_features  # noqa: E402
from rra.agents.ppo import PPOController  # noqa: E402
from rra.attacks import Attacker, GradientAttacker  # noqa: E402
from rra.baselines import ProportionalFair  # noqa: E402
from rra.config import EnvConfig  # noqa: E402
from rra.defence.controller import DefendedController  # noqa: E402
from rra.defence.sanitizer import Sanitizer  # noqa: E402
from rra.env.pysim import PySimEnv  # noqa: E402
from rra.parallel import pmap  # noqa: E402
from rra.runner import run_episode  # noqa: E402

U_EVAL = 20
METRICS = ["utility", "any_violation", "viol_eMBB", "viol_URLLC", "viol_mMTC", "cell_thr_mbps",
           "power_frac", "jain_eMBB", "urllc_p99_delay_ms"]

# name, actor run directory, sanitizer mode at deployment
CONTROLLERS = [
    ("PPO (undefended)", "results/week2/ppo_s{s}", "none"),
    ("PPO + det. sanitizer", "results/week2/ppo_s{s}", "det"),
    ("PPO + rand. sanitizer", "results/week2/ppo_s{s}", "rand"),
    ("RADAR-style", "results/week3/adv_none_s{s}", "det"),
    ("SA-PPO", "results/week3/sappo_none_s{s}", "none"),
    ("Proposed (a): rand. san. + adv. training", "results/week3/adv_rand_s{s}", "rand"),
    ("Proposed (b): rand. san. + SA-PPO", "results/week3/sappo_rand_s{s}", "rand"),
    ("Ablation: rand. san., no adversary", "results/week3/clean_rand_s{s}", "rand"),
]


def attacks(eps_main=0.1):
    """(name, factory(controller) -> attacker, gradient-based?)"""
    e = eps_main

    def grad(**kw):
        def f(ctrl):
            # Q = 8 noise draws against the randomized sanitizer; 1 when the pipeline is deterministic
            eot = 8 if getattr(ctrl, "mode", "none") == "rand" else 1
            return GradientAttacker(eot=eot, **kw)
        return f

    return [
        ("none", None, False),
        (f"noise {e:g}", lambda c: Attacker(eps=e, point="A2", strategy="noise"), False),
        (f"A1 SINR+ k=3 {e:g}", lambda c: Attacker(eps=e, point="A1", k=3, strategy="direction",
                                                   features=["sinr_db"], direction={"sinr_db": +1}), False),
        (f"black-box search {e:g}", lambda c: Attacker(eps=e, point="A2", strategy="search", n_candidates=16), False),
        (f"PGD-util {e:g}, defence-unaware", grad(eps=e, objective="util", adaptive=False), True),
        (f"EOT-PGD-KL {e:g}, adaptive", grad(eps=e, objective="kl"), True),
        (f"EOT-PGD-util {e:g}, adaptive", grad(eps=e, objective="util"), True),
        (f"EOT-PGD-util {e:g}, adaptive + stealth", grad(eps=e, objective="util", stealth=True), True),
        ("EOT-PGD-util 0.05, adaptive", grad(eps=0.05, objective="util"), True),
    ]


def load_controllers(san, seed):
    out = []
    for name, run, mode in CONTROLLERS:
        path = os.path.join(run.format(s=seed), "actor.pt")
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found", flush=True)
            continue
        actor = PPOController.load(path).actor
        out.append(DefendedController(actor, san if mode != "none" else None, mode=mode, name=name))
    return out


def ci95(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2:
        return (float(x.mean()) if x.size else float("nan")), float("nan")
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(x.size))


# ================================================================ E6
E6_COLS = ["controller", "attack", "seed"] + METRICS + ["flag_rate", "attacker_prb_ratio", "runtime_s"]


def run_e6(a, env, san):
    ctrls = load_controllers(san, a.seed) + [ProportionalFair(env)]
    seeds = list(range(1000, 1000 + a.episodes))
    jobs, keys = [], []
    for ctrl in ctrls:
        is_pf = isinstance(ctrl, ProportionalFair)
        for an, fac, is_grad in attacks():
            if is_pf and is_grad:
                continue
            for s in seeds:
                def job(ctrl=ctrl, fac=fac, s=s):
                    t = time.time()
                    att = fac(ctrl) if fac else None
                    r = run_episode(env, ctrl, att, seed=s, n_users=U_EVAL)
                    r["flag_rate"] = getattr(ctrl, "flag_rate", float("nan"))
                    r["runtime_s"] = time.time() - t
                    return r
                jobs.append(job)
                keys.append((getattr(ctrl, "name", "PF"), an, s))
    name = "e6_episodes.csv"
    if a.n_shards > 1:
        # shard k of n takes episodes k, k + n, k + 2n, ... (for machines without fork, e.g. Windows:
        # run the shards as separate processes, then --part merge)
        jobs, keys = jobs[a.shard::a.n_shards], keys[a.shard::a.n_shards]
        name = f"e6_episodes_shard{a.shard}of{a.n_shards}.csv"
    print(f"E6: {len(jobs)} episodes on {a.workers} worker(s)", flush=True)
    t0 = time.time()
    rows = pmap(jobs, a.workers)
    print(f"E6 done in {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(a.out, name), "w", newline="") as f:
        w = csv.writer(f); w.writerow(E6_COLS)
        for (c, an, s), r in zip(keys, rows):
            w.writerow([c, an, s] + [f"{r.get(k, float('nan')):.6g}" for k in E6_COLS[3:]])
    if a.n_shards <= 1:
        summarise_e6(a.out)


def merge_e6(out, n):
    """Put the shard files back together in the original episode order."""
    parts = [list(csv.reader(open(os.path.join(out, f"e6_episodes_shard{k}of{n}.csv")))) for k in range(n)]
    total = sum(len(p) - 1 for p in parts)
    with open(os.path.join(out, "e6_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(parts[0][0])
        for i in range(total):
            w.writerow(parts[i % n][1 + i // n])
    print(f"merged {n} shards, {total} episodes", flush=True)
    summarise_e6(out)


def read_episodes(out):
    rows = list(csv.DictReader(open(os.path.join(out, "e6_episodes.csv"))))
    for r in rows:
        for k in r:
            if k not in ("controller", "attack"):
                r[k] = float(r[k])
    return rows


def summarise_e6(out):
    rows = read_episodes(out)
    by = {}
    for r in rows:
        by.setdefault((r["controller"], r["attack"]), {})[r["seed"]] = r
    ctrls = list(dict.fromkeys(r["controller"] for r in rows))
    atts = list(dict.fromkeys(r["attack"] for r in rows))
    base_ctrl = "PPO (undefended)"
    summ = []
    for c in ctrls:
        for an in atts:
            if (c, an) not in by:
                continue
            cur = by[(c, an)]
            row = {"controller": c, "attack": an, "episodes": len(cur)}
            for k in METRICS + ["flag_rate"]:
                row[k], row[k + "_ci95"] = ci95([v[k] for v in cur.values()])
            own = by.get((c, "none"), {})
            sd = sorted(set(cur) & set(own))
            # paired change against the same controller without attack (same episodes)
            row["utility_change_pct"], row["utility_change_pct_ci95"] = ci95(
                [100 * (cur[s]["utility"] / own[s]["utility"] - 1) for s in sd])
            row["violating_change_pp"], row["violating_change_pp_ci95"] = ci95(
                [100 * (cur[s]["any_violation"] - own[s]["any_violation"]) for s in sd])
            # paired difference against the undefended PPO under the same attack
            ref = by.get((base_ctrl, an), {})
            sr = sorted(set(cur) & set(ref))
            row["utility_vs_undefended"], row["utility_vs_undefended_ci95"] = ci95(
                [cur[s]["utility"] - ref[s]["utility"] for s in sr])
            summ.append(row)
    cols = list(summ[0].keys())
    with open(os.path.join(out, "e6_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in summ:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    return summ


# ================================================================ detection
def auc(neg, pos):
    x = np.concatenate([neg, pos]); r = x.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def run_detect(a, env, san):
    """Per-user detection scores of clean vs attacked reports (attacks on the undefended PPO
    and adaptive attacks on PPO + randomized sanitizer)."""
    ppo = PPOController.load(f"results/week2/ppo_s{a.seed}/actor.pt")
    dctrl = DefendedController(ppo.actor, san, "rand", name="PPO + rand. sanitizer")
    obs_list = []
    for s in (2000, 2001):
        obs = env.reset(seed=s, n_users=U_EVAL)
        for t in range(150):
            if t >= 20 and t % 5 == 0:
                obs_list.append(obs)
            obs, _, _ = env.step(ppo.act(obs))
    tau = float(san.tau)

    def scores(ob):
        x, _ = obs_features(ob)
        with torch.no_grad():
            e = san.error(torch.from_numpy(ob.reported_norm.astype(np.float32)),
                          torch.from_numpy(x[:, 8:11])).numpy()
        return e[ob.active]

    clean = np.concatenate([scores(o) for o in obs_list])
    cases = []
    for e in (0.05, 0.1):
        cases += [
            (f"noise {e:g}", e, lambda e=e: Attacker(eps=e, point="A2", strategy="noise"), ppo),
            (f"A1 SINR+ {e:g}", e, lambda e=e: Attacker(eps=e, point="A1", strategy="direction",
                                                        features=["sinr_db"], direction={"sinr_db": +1}), ppo),
            (f"black-box search {e:g}", e, lambda e=e: Attacker(eps=e, point="A2", strategy="search"), ppo),
            (f"PGD-KL {e:g} (undefended)", e, lambda e=e: GradientAttacker(eps=e, objective="kl"), ppo),
            (f"PGD-util {e:g} (undefended)", e, lambda e=e: GradientAttacker(eps=e, objective="util"), ppo),
            (f"EOT-PGD-util {e:g} adaptive", e, lambda e=e: GradientAttacker(eps=e, objective="util", eot=8), dctrl),
            (f"EOT-PGD-util {e:g} adaptive + stealth", e,
             lambda e=e: GradientAttacker(eps=e, objective="util", eot=8, stealth=True), dctrl),
        ]

    def job(case):
        name, e, fac, ctrl = case
        att = fac(); att.reset(env, ctrl, seed=11)
        sc = []
        for ob in obs_list:
            sc.append(scores(att.perturb(ob)))
        return np.concatenate(sc)

    pos = pmap([(lambda c=c: job(c)) for c in cases], a.workers)
    res = [{"attack": "clean (false-positive rate)", "eps": 0.0, "auc": float("nan"),
            "flagged": float(np.mean(clean > tau)), "n_users": int(clean.size)}]
    for (name, e, _, _), p in zip(cases, pos):
        res.append({"attack": name, "eps": e, "auc": auc(clean, p), "flagged": float(np.mean(p > tau)),
                    "n_users": int(p.size)})
    with open(os.path.join(a.out, "detect.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    np.savez_compressed(os.path.join(a.out, "detect_scores.npz"), clean=clean,
                        **{c[0]: p for c, p in zip(cases, pos)})
    for r in res:
        print(f"{r['attack']:40s} AUC {r['auc']:.3f}  flagged {100 * r['flagged']:.1f}%")


# ================================================================ T2
def run_t2(a, san):
    """Targeted through-sanitizer attack on clean test reports (always-sanitize gate).

    Target: the report of a 'starving' user, s + 0.1 * (+1, +1, +1, 0, -1) (better channel,
    more buffer and demand, lower average rate), i.e. a report that would attract PRBs.
    The attacker has budget eps = 0.1, so the target is exactly reachable at the input;
    the question is whether it is reachable at the *output* of the sanitizer.
    """
    d = np.load(os.path.join(a.out, "clean_reports.npz"))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["S"]), 2000, replace=False)
    s = torch.from_numpy(d["S"][idx]); oh = torch.eye(3)[torch.from_numpy(d["C"][idx]).long()]
    m = torch.ones(len(idx), dtype=torch.bool)
    eps = 0.1
    tgt = (s + eps * torch.tensor([1., 1., 1., 0., -1.])).clamp(0, 1)
    res = []
    for mode in ("det", "rand"):
        def out_dist(delta, n_eval=16):
            with torch.no_grad():
                ds = [((san(s + delta, oh, m, mode, "always")[0] - tgt) ** 2).sum(-1).sqrt()
                      for _ in range(n_eval if mode == "rand" else 1)]
            return torch.stack(ds).mean(0)
        base = out_dist(torch.zeros_like(s))
        for Q in ((1,) if mode == "det" else (1, 8)):
            for K in (10, 50):
                torch.manual_seed(0)
                delta = torch.zeros_like(s)
                for _ in range(K):
                    delta.requires_grad_(True)
                    loss = 0.0
                    for _ in range(Q):
                        st = san(torch.clamp(s + delta, 0, 1), oh, m, mode, "always")[0]
                        loss = loss + ((st - tgt) ** 2).sum()
                    g, = torch.autograd.grad(loss / Q, delta)
                    delta = torch.clamp(delta.detach() - (2.5 * eps / K) * g.sign(), -eps, eps)
                    delta = torch.clamp(s + delta, 0, 1) - s
                dist = out_dist(delta)
                res.append({"sanitizer": mode, "Q": Q, "K": K,
                            "gap_closed_pct": float(100 * (1 - (dist / base.clamp(min=1e-9)).mean())),
                            "dist_before": float(base.mean()), "dist_after": float(dist.mean())})
                print(res[-1], flush=True)
    # reference: no sanitizer, the attacker simply sends the target
    res.append({"sanitizer": "none", "Q": 0, "K": 0, "gap_closed_pct": 100.0,
                "dist_before": float(((s - tgt) ** 2).sum(-1).sqrt().mean()), "dist_after": 0.0})
    with open(os.path.join(a.out, "t2_through_sanitizer.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


# ================================================================ zeta
def run_zeta(a, env):
    """Power/PRB price: week-2 model (zeta = 0.05), the same model fine-tuned for 200 k steps with
    zeta = 0.05 (control: same extra training) and with zeta = 0.2."""
    models = [("zeta 0.05 (week-2 model)", f"results/week2/ppo_s{a.seed}"),
              ("zeta 0.05 (fine-tuned 200 k steps)", f"{a.models}/clean_none_s{a.seed}"),
              ("zeta 0.2 (fine-tuned 200 k steps)", f"{a.models}/zeta020_s{a.seed}")]
    models = [(n, r) for n, r in models if os.path.exists(os.path.join(r, "actor.pt"))]
    seeds = list(range(1000, 1000 + a.episodes))
    jobs, keys = [], []
    for name, run in models:
        ctrl = PPOController.load(os.path.join(run, "actor.pt"))
        for U in (10, 20, 30):
            for s in seeds:
                jobs.append(lambda ctrl=ctrl, U=U, s=s: run_episode(env, ctrl, None, seed=s, n_users=U))
                keys.append((name, U))
    rows = pmap(jobs, a.workers)
    res = []
    for name, _ in models:
        for U in (10, 20, 30):
            rr = [r for k, r in zip(keys, rows) if k == (name, U)]
            row = {"model": name, "U": U}
            for k in ("utility", "power_frac", "prb_frac", "viol_eMBB", "viol_URLLC", "viol_mMTC", "cell_thr_mbps"):
                row[k], row[k + "_ci95"] = ci95([r[k] for r in rr])
            res.append(row)
    with open(os.path.join(a.out, "zeta_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0])); w.writeheader()
        for r in res:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in res:
        print(f"{r['model']:34s} U={r['U']:2d} utility {r['utility']:.3f} power {r['power_frac']:.3f} "
              f"PRB {r['prb_frac']:.3f} URLLC {100 * r['viol_URLLC']:.1f}% eMBB {100 * r['viol_eMBB']:.1f}%")


# ================================================================ plots
COL = {"PPO (undefended)": "#9a9893", "PPO + det. sanitizer": "#eda100", "PPO + rand. sanitizer": "#eb6834",
       "RADAR-style": "#7d5bd6", "SA-PPO": "#2a78d6", "Proposed (a): rand. san. + adv. training": "#1baf7a",
       "Proposed (b): rand. san. + SA-PPO": "#0e7c55", "Ablation: rand. san., no adversary": "#c9c7c1",
       "PF": "#3b3a36"}


def plot_e6(out, path=None):
    summ = list(csv.DictReader(open(os.path.join(out, "e6_summary.csv"))))
    ctrls = list(dict.fromkeys(r["controller"] for r in summ))
    atts = list(dict.fromkeys(r["attack"] for r in summ))
    get = {(r["controller"], r["attack"]): r for r in summ}
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.55 * len(atts) + 2.2), sharey=True)
    h = 0.8 / len(ctrls)
    y = np.arange(len(atts))
    for k, c in enumerate(ctrls):
        for ax, key, scale in ((axes[0], "utility", 1), (axes[1], "any_violation", 100)):
            vals = [float(get[(c, an)][key]) * scale if (c, an) in get else np.nan for an in atts]
            errs = [float(get[(c, an)][key + "_ci95"]) * scale if (c, an) in get else 0 for an in atts]
            ax.barh(y + (k - (len(ctrls) - 1) / 2) * h, vals, h, xerr=np.nan_to_num(errs),
                    color=COL.get(c, "#888"), label=c, error_kw={"lw": 0.6, "capsize": 1})
    axes[0].set_yticks(y); axes[0].set_yticklabels(atts, fontsize=8); axes[0].invert_yaxis()
    axes[0].set_xlabel("utility Φ (higher is better)"); axes[0].set_xlim(0.6, 1.1)
    axes[1].set_xlabel("users violating their SLA (%, lower is better)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.0, -0.08 - 0.3 / len(atts)), ncol=3,
                   frameon=False, fontsize=8)
    fig.suptitle(f"E6: defences under attack, U = {U_EVAL} (Python simulator; mean and 95% interval)", fontsize=10)
    fig.savefig(path or os.path.join(out, "e6_defence.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_learning(out, path=None):
    runs = [("adv_rand", "Proposed (a)"), ("sappo_rand", "Proposed (b)"), ("adv_none", "RADAR-style training"),
            ("sappo_none", "SA-PPO"), ("clean_rand", "Ablation")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2))
    colours = dict(zip([r for r, _ in runs], plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for r, lab in runs:
        p = os.path.join(out, f"{r}_s0", "train_log.csv")
        if not os.path.exists(p):
            continue
        d = list(csv.DictReader(open(p)))
        st = np.array([float(x["steps"]) for x in d])
        u = np.array([float(x["utility"]) for x in d])
        k = min(9, len(u))
        sm = np.convolve(u, np.ones(k) / k, mode="valid")
        ax[0].plot(st[k - 1:], sm, label=lab, color=colours[r])
        ua = np.array([float(x["utility_attacked"]) for x in d])
        if np.isfinite(ua).any():
            ax[1].plot(st[k - 1:], np.convolve(ua, np.ones(k) / k, mode="valid"), label=lab, color=colours[r])
    ax[0].set_title("training utility, clean rollouts (smoothed)", fontsize=9)
    ax[1].set_title("training utility, attacked rollouts (Madry runs)", fontsize=9)
    for x in ax:
        x.set_xlabel("fine-tuning steps (thousands)")
        x.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}"))
        x.spines[["top", "right"]].set_visible(False)
        x.legend(frameon=False, fontsize=7)
    fig.savefig(path or os.path.join(out, "week3_learning.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "detect", "t2", "e6", "zeta", "summary", "plot", "merge"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="training seed of the models to evaluate")
    ap.add_argument("--out", default="results/week3", help="where the sanitizer and trained models are")
    ap.add_argument("--shard", type=int, default=0, help="E6 only: which shard to run (0 .. n-shards - 1)")
    ap.add_argument("--n-shards", type=int, default=1, help="E6 only: split the episodes into this many shards")
    ap.add_argument("--save-dir", default=None,
                    help="where to write the results (default: --out; use e.g. results/week3/seed1 for other seeds)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.workers is None:
        from rra.parallel import default_workers
        a.workers = default_workers()
    env = PySimEnv(EnvConfig(feature_scale="log"))
    san = Sanitizer.load(os.path.join(a.out, "sanitizer.pt"))
    a.models = a.out                       # where the trained week-3 models are
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        a.out = a.save_dir
    t0 = time.time()
    if a.part in ("all", "detect"):
        run_detect(a, env, san)
    if a.part in ("all", "t2"):
        run_t2(a, san)
    if a.part in ("all", "e6"):
        run_e6(a, env, san)
    if a.part in ("all", "zeta"):
        run_zeta(a, env)
    if a.part == "summary":
        summarise_e6(a.out)
    if a.part == "merge":
        merge_e6(a.out, a.n_shards)
    if a.n_shards > 1 and a.part == "e6":
        return
    if a.part in ("all", "e6", "summary", "plot", "merge"):
        plot_e6(a.out)
        if not a.save_dir:
            plot_learning(a.out)
    meta_p = os.path.join(a.out, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.update({"simulator": "rra.env.pysim (Python)", "users": U_EVAL, "episodes": a.episodes,
                 "eval_seeds": list(range(1000, 1000 + a.episodes)), "model_seed": a.seed,
                 f"runtime_s_{a.part}": round(time.time() - t0, 1)})
    json.dump(meta, open(meta_p, "w"), indent=2)


if __name__ == "__main__":
    main()
