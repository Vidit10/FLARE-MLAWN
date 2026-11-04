"""Week-3 evaluation E7: the rest of the report's experimental design (Sec. 5.2-5.4).

All numbers come from the Python simulator (rra/env/pysim.py), log feature scaling.
Same controllers, evaluation seeds (1000+) and paired comparisons as E6
(experiments/week3_eval.py); the "no attack" episodes at U = 20 are taken from E6.

Parts (--part):
  fit     Fit the offline attacks against every deployed controller (white-box,
          through the whole pipeline, EOT over the sanitizer's noise): universal
          perturbation (UAP) and generator attacker (GAN, plain and detector-aware).
          Fitting data: clean observations of that controller, seeds 3000+.
          Saved to <out>/fitted/ and reused by the other parts.
  suite   Attack suite of Sec. 5.2 at eps = 0.1, U = 20: FGSM, UAP, GAN,
          GAN detector-aware, and a black-box transfer attack (PGD on the clean PPO
          of another training seed).
  eps     Sweep eps in {0.02, 0.05, 0.1, 0.2} for FGSM and adaptive EOT-PGD
          (eps1 < eps2 < eps3 of Sec. 5.2); eps 0.05/0.1 of PGD come from E6.
  users   Variable-user evaluation (Sec. 5.3): U in {5, 10, 20, 40}, clean and
          adaptive EOT-PGD-util 0.1. U = 20 comes from E6. The detector and the
          sanitizer are never retrained; U = 40 is outside the PPO training range (5-30).
  detect  Detector true-positive rate and AUC as a function of eps (Sec. 5.4),
          including UAP and the generator attacks.
  unseen  Detection at U values excluded from the detector's training (Sec. 5.3):
          a second detector/sanitizer is fitted on reports from U = 5-20 only and
          compared with the main one (U = 5-40) at U in {5, 10, 20, 40}.
  summary / plot   Tables and figures from the saved CSVs.
  blind   Diagnostic (not an attack): every report is replaced by the mean report of
          its slice in that cell, so the allocator gets no per-user information.
          A controller whose utility hardly changes is not using the reports.
  all     fit, suite, eps, users, detect, unseen, blind, summary, plot.

Usage: python -m experiments.week3_e7 --part all [--episodes 8] [--workers N] [--seed 0]
Output: results/week3/e7/ (seed 0) or --out for other seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from rra.agents.policy import obs_features  # noqa: E402
from rra.agents.ppo import PPOController  # noqa: E402
from rra.attacks import (Attacker, GeneratorAttacker, GradientAttacker, TransferAttacker,  # noqa: E402
                         UAPAttacker, collect_states, fit_generator, fit_uap)
from rra.attacks.learned import Generator  # noqa: E402
from rra.baselines import ProportionalFair  # noqa: E402
from rra.config import EnvConfig  # noqa: E402
from rra.defence.controller import DefendedController  # noqa: E402
from rra.defence.sanitizer import Sanitizer  # noqa: E402
from rra.env.pysim import PySimEnv  # noqa: E402
from rra.parallel import pmap  # noqa: E402
from rra.runner import run_episode  # noqa: E402

from experiments import week3_eval as W3  # noqa: E402

EPS_MAIN = 0.1
EPS_SWEEP = (0.02, 0.05, 0.1, 0.2)
USERS = (5, 10, 20, 40)
BETA_STEALTH = 0.2           # weight of the detector term of the detector-aware generator
SUBSET = ["PPO (undefended)", "PPO + rand. sanitizer", "RADAR-style", "SA-PPO",
          "Proposed (a): rand. san. + adv. training", "Proposed (b): rand. san. + SA-PPO"]
SHORT = {"PPO (undefended)": "undefended", "PPO + det. sanitizer": "det_san",
         "PPO + rand. sanitizer": "rand_san", "RADAR-style": "radar", "SA-PPO": "sappo",
         "Proposed (a): rand. san. + adv. training": "proposed_a",
         "Proposed (b): rand. san. + SA-PPO": "proposed_b",
         "Ablation: rand. san., no adversary": "ablation"}
COLS = ["controller", "attack", "eps", "users", "seed"] + W3.METRICS + ["flag_rate", "runtime_s"]


def eot_for(ctrl):
    return 8 if getattr(ctrl, "mode", "none") == "rand" else 1


def surrogate_for(seed):
    """Attacker's own model for the transfer attack: clean PPO of another training seed."""
    s = (seed + 1) % 3
    return DefendedController(PPOController.load(f"results/week2/ppo_s{s}/actor.pt").actor, None,
                              mode="none", name=f"surrogate ppo_s{s}")


# ================================================================== fitting
def fit_path(out, ctrl_name, eps):
    return os.path.join(out, "fitted", f"{SHORT[ctrl_name]}_eps{eps:g}.pt")


def run_fit(a, env, ctrls, pairs):
    """pairs: list of (controller name, eps). Fits UAP, GAN and detector-aware GAN."""
    os.makedirs(os.path.join(a.out, "fitted"), exist_ok=True)
    todo = [(c, e) for c, e in pairs if not os.path.exists(fit_path(a.out, c, e))]
    print(f"fit: {len(todo)} (controller, eps) pairs on {a.workers} worker(s)", flush=True)

    def job(c, e):
        t = time.time()
        ctrl = ctrls[c]
        states = collect_states(env, ctrl)
        table = fit_uap(env, ctrl, states, e)
        g = fit_generator(env, ctrl, states, e)
        g_da = fit_generator(env, ctrl, states, e, beta=BETA_STEALTH)
        return {"uap": torch.from_numpy(table), "gan": g.state_dict(), "gan_da": g_da.state_dict(),
                "eps": e, "controller": c, "n_states": len(states), "fit_s": time.time() - t}

    res = pmap([(lambda c=c, e=e: job(c, e)) for c, e in todo], a.workers)
    for r in res:
        torch.save(r, fit_path(a.out, r["controller"], r["eps"]))
        print(f"  fitted {r['controller']} eps {r['eps']:g}: {r['n_states']} states, {r['fit_s']:.0f} s", flush=True)


def load_fit(out, ctrl_name, eps):
    d = torch.load(fit_path(out, ctrl_name, eps), map_location="cpu")
    gens = {}
    for k in ("gan", "gan_da"):
        g = Generator(eps); g.load_state_dict(d[k]); g.eval()
        for p in g.parameters():
            p.requires_grad_(False)
        gens[k] = g
    return d["uap"].numpy().astype(float), gens["gan"], gens["gan_da"]


# ================================================================== episodes
def run_cells(a, env, cells, name):
    """cells: list of (controller, attack name, eps, factory or None, U, seed)."""
    print(f"{name}: {len(cells)} episodes on {a.workers} worker(s)", flush=True)

    def job(ctrl, fac, U, s):
        t = time.time()
        r = run_episode(env, ctrl, fac(ctrl) if fac else None, seed=s, n_users=U)
        r["flag_rate"] = getattr(ctrl, "flag_rate", float("nan"))
        r["runtime_s"] = time.time() - t
        return r

    t0 = time.time()
    res = pmap([(lambda c=c, f=f, U=U, s=s: job(c, f, U, s)) for c, _, _, f, U, s in cells], a.workers)
    print(f"{name} done in {time.time() - t0:.0f} s", flush=True)
    rows = []
    for (c, an, e, _, U, s), r in zip(cells, res):
        rows.append({"controller": getattr(c, "name", "PF"), "attack": an, "eps": e, "users": U, "seed": s,
                     **{k: r.get(k, float("nan")) for k in W3.METRICS + ["flag_rate", "runtime_s"]}})
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(os.path.join(a.out, f"{name}_episodes.csv"), index=False, float_format="%.6g")
    return df


def fgsm(e):
    return lambda c: GradientAttacker(eps=e, strategy="fgsm", objective="util", eot=eot_for(c))


def pgd(e):
    return lambda c: GradientAttacker(eps=e, objective="util", eot=eot_for(c))


def run_suite(a, env, ctrls, pf):
    seeds = list(range(1000, 1000 + a.episodes))
    sur = surrogate_for(a.seed)
    cells = []
    for c in ctrls.values():
        table, g, g_da = load_fit(a.out, c.name, EPS_MAIN)
        atts = [(f"FGSM-util {EPS_MAIN:g}, adaptive", fgsm(EPS_MAIN)),
                (f"UAP {EPS_MAIN:g}, adaptive", lambda c, t=table: UAPAttacker(eps=EPS_MAIN, table=t)),
                (f"GAN {EPS_MAIN:g}, adaptive", lambda c, g=g: GeneratorAttacker(eps=EPS_MAIN, gen=g)),
                (f"GAN {EPS_MAIN:g}, adaptive + detector-aware",
                 lambda c, g=g_da: GeneratorAttacker(eps=EPS_MAIN, gen=g)),
                (f"transfer PGD-util {EPS_MAIN:g} (black-box)",
                 lambda c: TransferAttacker(eps=EPS_MAIN, objective="util", surrogate=sur))]
        for an, f in atts:
            cells += [(c, an, EPS_MAIN, f, W3.U_EVAL, s) for s in seeds]
    cells += [(pf, f"transfer PGD-util {EPS_MAIN:g} (black-box)", EPS_MAIN,
               lambda c: TransferAttacker(eps=EPS_MAIN, objective="util", surrogate=sur), W3.U_EVAL, s)
              for s in seeds]
    run_cells(a, env, cells, "suite")


def run_eps(a, env, ctrls):
    seeds = list(range(1000, 1000 + a.episodes))
    cells = []
    for n in SUBSET:
        c = ctrls[n]
        for e in EPS_SWEEP:
            if e != EPS_MAIN:                       # FGSM 0.1 is in the suite
                cells += [(c, f"FGSM-util {e:g}, adaptive", e, fgsm(e), W3.U_EVAL, s) for s in seeds]
            if e not in (0.05, 0.1):                # PGD 0.05 and 0.1 are in E6
                cells += [(c, f"EOT-PGD-util {e:g}, adaptive", e, pgd(e), W3.U_EVAL, s) for s in seeds]
    run_cells(a, env, cells, "eps")


def run_users(a, env, ctrls):
    seeds = list(range(1000, 1000 + a.episodes))
    cells = []
    for n in SUBSET:
        c = ctrls[n]
        for U in USERS:
            if U == W3.U_EVAL:                      # U = 20 is E6
                continue
            cells += [(c, "none", 0.0, None, U, s) for s in seeds]
            cells += [(c, f"EOT-PGD-util {EPS_MAIN:g}, adaptive", EPS_MAIN, pgd(EPS_MAIN), U, s) for s in seeds]
    run_cells(a, env, cells, "users")


# ================================================================== detection vs eps
def run_detect(a, env, san):
    ppo = PPOController.load(f"results/week2/ppo_s{a.seed}/actor.pt")
    name = "PPO + rand. sanitizer"
    dctrl = DefendedController(ppo.actor, san, "rand", name=name)
    obs_list = []
    for s in (2000, 2001):
        obs = env.reset(seed=s, n_users=W3.U_EVAL)
        for t in range(150):
            if t >= 20 and t % 5 == 0:
                obs_list.append(obs)
            obs, _, _ = env.step(ppo.act(obs))
    tau = float(san.tau)

    def scores(ob):
        x, _ = obs_features(ob)
        with torch.no_grad():
            e = san.error(torch.from_numpy(ob.reported_norm.astype(np.float32)), torch.from_numpy(x[:, 8:11])).numpy()
        return e[ob.active]

    clean = np.concatenate([scores(o) for o in obs_list])
    cases = []
    for e in EPS_SWEEP:
        table, g, g_da = load_fit(a.out, name, e)
        cases += [("noise", e, lambda e=e: Attacker(eps=e, point="A2", strategy="noise"), ppo),
                  ("PGD-util, defence-unaware", e, lambda e=e: GradientAttacker(eps=e, objective="util"), ppo),
                  ("EOT-PGD-util, adaptive", e, lambda e=e: GradientAttacker(eps=e, objective="util", eot=8), dctrl),
                  ("UAP, adaptive", e, lambda e=e, t=table: UAPAttacker(eps=e, table=t), dctrl),
                  ("GAN, adaptive", e, lambda e=e, g=g: GeneratorAttacker(eps=e, gen=g), dctrl),
                  ("GAN, adaptive + detector-aware", e, lambda e=e, g=g_da: GeneratorAttacker(eps=e, gen=g), dctrl)]

    def job(case):
        _, e, fac, ctrl = case
        att = fac(); att.reset(env, ctrl, seed=11)
        return np.concatenate([scores(att.perturb(ob)) for ob in obs_list])

    print(f"detect: {len(cases)} cases on {a.workers} worker(s)", flush=True)
    pos = pmap([(lambda c=c: job(c)) for c in cases], a.workers)
    rows = [{"attack": "clean (false-positive rate)", "eps": 0.0, "flagged": float(np.mean(clean > tau)),
             "auc": float("nan"), "n_users": int(clean.size)}]
    for (n, e, _, _), p in zip(cases, pos):
        rows.append({"attack": n, "eps": e, "flagged": float(np.mean(p > tau)), "auc": W3.auc(clean, p),
                     "n_users": int(p.size)})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "detect_eps.csv"), index=False, float_format="%.4g")


# ================================================================== report-blind diagnostic
class SliceMeanReports(Attacker):
    """Replace every active user's reports by the mean of its slice (no budget)."""

    def perturb(self, obs):
        r = obs.reported_norm.copy()
        for k in range(3):
            rows = obs.active & (obs.slice_id == k)
            if rows.any():
                r[rows] = r[rows].mean(0)
        return self.apply_delta(obs, r - obs.reported_norm)


def run_blind(a, env, ctrls):
    seeds = list(range(1000, 1000 + a.episodes))
    cells = [(ctrls[n], "slice-mean reports (diagnostic)", 1.0,
              lambda c: SliceMeanReports(eps=1.0, point="A2"), W3.U_EVAL, s) for n in SUBSET for s in seeds]
    df = run_cells(a, env, cells, "blind")
    base = e6_rows(a); base = base[base.attack == "none"]
    out = _agg(_paired(df, base), ["controller", "attack"])
    out.round(5).to_csv(os.path.join(a.out, "blind_summary.csv"), index=False)
    print(out[["controller", "utility", "utility_change_pct", "utility_change_ci95", "viol_URLLC"]].round(3).to_string())


# ================================================================== unseen U
def run_unseen(a, env, san):
    import glob
    from rra.baselines import EqualShare
    from rra.defence.sanitizer import SanitizerConfig, collect_reports, train_sanitizer
    path = os.path.join(a.out, "sanitizer_u5_20.pt")
    if not os.path.exists(path):
        t = time.time()
        ctrls = [EqualShare(env), ProportionalFair(env)] + [
            PPOController.load(p) for p in sorted(glob.glob("results/week2/ppo_s*/actor.pt"))]
        S, C = collect_reports(env, ctrls, 60, users=(5, 20), seed=60_000)
        san20, stats = train_sanitizer(S, C, SanitizerConfig(**json.load(open(
            os.path.join(a.root, "sanitizer.json")))["config"]))
        stats["users"] = [5, 20]
        san20.save(path, stats)
        print(f"unseen: fitted the U = 5-20 detector on {len(S)} reports ({time.time() - t:.0f} s)", flush=True)
    san20 = Sanitizer.load(path)
    ppo = PPOController.load(f"results/week2/ppo_s{a.seed}/actor.pt")
    dets = {"main detector (fitted on U = 5-40)": san, "detector fitted on U = 5-20": san20}

    def job(U):
        obs_list = []
        for s in (2100, 2101):
            obs = env.reset(seed=s, n_users=U)
            for t in range(150):
                if t >= 20 and t % 5 == 0:
                    obs_list.append(obs)
                obs, _, _ = env.step(ppo.act(obs))
        atts = {"none": None,
                "noise 0.1": Attacker(eps=0.1, point="A2", strategy="noise"),
                "black-box search 0.1": Attacker(eps=0.1, point="A2", strategy="search"),
                "PGD-util 0.1 (undefended)": GradientAttacker(eps=0.1, objective="util")}
        rows = []
        for an, att in atts.items():
            if att is not None:
                att.reset(env, ppo, seed=11)
            seen = [att.perturb(o) if att else o for o in obs_list]
            for dn, d in dets.items():
                sc = []
                for ob, o0 in zip(seen, obs_list):
                    x, _ = obs_features(ob)
                    with torch.no_grad():
                        e = d.error(torch.from_numpy(ob.reported_norm.astype(np.float32)),
                                    torch.from_numpy(x[:, 8:11])).numpy()
                    sc.append(e[o0.active])
                rows.append({"detector": dn, "users": U, "attack": an, "scores": np.concatenate(sc),
                             "tau": float(d.tau)})
        return rows

    print(f"unseen: U in {USERS} on {a.workers} worker(s)", flush=True)
    res = [r for rows in pmap([(lambda U=U: job(U)) for U in USERS], a.workers) for r in rows]
    out = []
    for r in res:
        clean = next(q["scores"] for q in res if q["detector"] == r["detector"] and q["users"] == r["users"]
                     and q["attack"] == "none")
        out.append({"detector": r["detector"], "users": r["users"], "attack": r["attack"],
                    "flagged": float(np.mean(r["scores"] > r["tau"])),
                    "auc": float("nan") if r["attack"] == "none" else W3.auc(clean, r["scores"]),
                    "n_users": int(r["scores"].size)})
    pd.DataFrame(out).to_csv(os.path.join(a.out, "unseen_users.csv"), index=False, float_format="%.4g")


# ================================================================== summary
def _paired(df, base):
    """Add the paired utility change against the same controller without attack (same U, seed)."""
    b = base.set_index(["controller", "users", "seed"])["utility"]
    key = list(zip(df.controller, df.users, df.seed))
    ref = b.reindex(key).to_numpy()
    df = df.copy()
    df["utility_change_pct"] = 100 * (df.utility.to_numpy() / ref - 1)
    bv = base.set_index(["controller", "users", "seed"])["any_violation"].reindex(key).to_numpy()
    df["violating_change_pp"] = 100 * (df.any_violation.to_numpy() - bv)
    return df


def _agg(df, by):
    def ci(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return 1.96 * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else np.nan
    g = df.groupby(by, sort=False)
    out = g.agg(episodes=("seed", "count"), utility=("utility", "mean"),
                utility_change_pct=("utility_change_pct", "mean"),
                utility_change_ci95=("utility_change_pct", ci),
                any_violation=("any_violation", "mean"), violating_change_pp=("violating_change_pp", "mean"),
                viol_URLLC=("viol_URLLC", "mean"), flag_rate=("flag_rate", "mean")).reset_index()
    return out


def e6_rows(a):
    p = os.path.join(a.e6_dir, "e6_episodes.csv")
    d = pd.read_csv(p)
    d["users"] = W3.U_EVAL
    d["eps"] = d.attack.str.extract(r"(\d+\.\d+)").astype(float).fillna(0.0)[0]
    return d


def summarise(a):
    e6 = e6_rows(a)
    base20 = e6[e6.attack == "none"]
    out = {}
    if os.path.exists(os.path.join(a.out, "suite_episodes.csv")):
        s = pd.read_csv(os.path.join(a.out, "suite_episodes.csv"))
        ref = e6[e6.attack.isin(["EOT-PGD-util 0.1, adaptive", "EOT-PGD-util 0.1, adaptive + stealth",
                                 "black-box search 0.1"])]
        s = pd.concat([_paired(s, base20), _paired(ref, base20)], ignore_index=True)
        out["suite"] = _agg(s, ["controller", "attack"])
    if os.path.exists(os.path.join(a.out, "eps_episodes.csv")):
        s = pd.read_csv(os.path.join(a.out, "eps_episodes.csv"))
        parts = [s, e6[e6.attack.isin(["EOT-PGD-util 0.05, adaptive", "EOT-PGD-util 0.1, adaptive"])]]
        if os.path.exists(os.path.join(a.out, "suite_episodes.csv")):
            su = pd.read_csv(os.path.join(a.out, "suite_episodes.csv"))
            parts.append(su[su.attack == f"FGSM-util {EPS_MAIN:g}, adaptive"])
        s = _paired(pd.concat(parts, ignore_index=True), base20)
        s = s[s.controller.isin(SUBSET)]
        s["method"] = np.where(s.attack.str.startswith("FGSM"), "FGSM-util, adaptive", "EOT-PGD-util, adaptive")
        out["eps"] = _agg(s, ["controller", "method", "eps"]).sort_values(["controller", "method", "eps"])
    if os.path.exists(os.path.join(a.out, "users_episodes.csv")):
        s = pd.read_csv(os.path.join(a.out, "users_episodes.csv"))
        s = pd.concat([s, e6[e6.attack.isin(["none", "EOT-PGD-util 0.1, adaptive"]) & e6.controller.isin(SUBSET)]],
                      ignore_index=True)
        base = s[s.attack == "none"]
        out["users"] = _agg(_paired(s, base), ["controller", "attack", "users"]).sort_values(
            ["controller", "attack", "users"])
    for k, v in out.items():
        v.round(5).to_csv(os.path.join(a.out, f"{k}_summary.csv"), index=False)
    return out


# ================================================================== plots
def plot(a):
    S = {k: pd.read_csv(os.path.join(a.out, f"{k}_summary.csv"))
         for k in ("suite", "eps", "users") if os.path.exists(os.path.join(a.out, f"{k}_summary.csv"))}
    cols = dict(zip(list(SHORT), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    cols["PF"] = "#444444"
    if "suite" in S:
        d = S["suite"]
        atts = list(dict.fromkeys(d.attack))
        ctrl = [c for c in list(SHORT) + ["PF"] if c in set(d.controller)]
        fig, ax = plt.subplots(figsize=(10, 0.55 * len(atts) + 1.6))
        h = 0.8 / len(ctrl)
        for j, c in enumerate(ctrl):
            v = d[d.controller == c].set_index("attack").reindex(atts)
            ax.barh(np.arange(len(atts)) + j * h, v.utility_change_pct, h, xerr=v.utility_change_ci95,
                    color=cols[c], label=c, error_kw={"lw": 0.6})
        ax.set_yticks(np.arange(len(atts)) + 0.4 - h / 2); ax.set_yticklabels(atts, fontsize=8); ax.invert_yaxis()
        ax.axvline(0, color="#999", lw=0.8)
        ax.set_xlabel("utility change vs. no attack (%), paired, 95% interval")
        ax.set_title(f"Attack suite at eps = 0.1, U = 20, training seed {a.seed} (Python simulator)", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
        ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(os.path.join(a.out, "e7_suite.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    if "eps" in S:
        d = S["eps"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for k, meth in enumerate(["FGSM-util, adaptive", "EOT-PGD-util, adaptive"]):
            for c in SUBSET:
                v = d[(d.controller == c) & (d.method == meth)]
                ax[k].errorbar(v.eps, v.utility_change_pct, yerr=v.utility_change_ci95, marker="o", ms=3,
                               color=cols[c], label=c, lw=1.2, capsize=2)
            ax[k].set_title(meth, fontsize=9); ax[k].set_xlabel("eps"); ax[k].set_xticks(EPS_SWEEP)
            ax[k].axhline(0, color="#999", lw=0.8); ax[k].spines[["top", "right"]].set_visible(False)
        ax[0].set_ylabel("utility change (%)"); ax[1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Robustness vs. attack budget, U = 20, training seed {a.seed} (Python simulator)", fontsize=9, y=1.02)
        fig.savefig(os.path.join(a.out, "e7_eps.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    if "users" in S:
        d = S["users"]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
        for c in SUBSET:
            n = d[(d.controller == c) & (d.attack == "none")]
            t = d[(d.controller == c) & (d.attack != "none")]
            ax[0].plot(n.users, n.utility, marker="o", ms=3, color=cols[c], label=c)
            ax[1].errorbar(t.users, t.utility_change_pct, yerr=t.utility_change_ci95, marker="o", ms=3,
                           color=cols[c], capsize=2)
            if c != "PPO (undefended)" and c != "SA-PPO":
                ax[2].plot(n.users, 100 * n.flag_rate, marker="o", ms=3, color=cols[c], ls="--")
                ax[2].plot(t.users, 100 * t.flag_rate, marker="o", ms=3, color=cols[c])
        ax[0].set_title("utility without attack", fontsize=9)
        ax[1].set_title("utility change, EOT-PGD-util 0.1 adaptive (%)", fontsize=9)
        ax[2].set_title("users flagged (%): dashed clean, solid attacked", fontsize=9)
        for x in ax:
            x.set_xlabel("users U"); x.set_xticks(USERS); x.spines[["top", "right"]].set_visible(False)
        ax[1].axhline(0, color="#999", lw=0.8); ax[0].legend(fontsize=6, frameon=False)
        fig.suptitle(f"Variable number of users, training seed {a.seed}; PPO trained on U = 5-30, "
                     "sanitizer on U = 5-40 (Python simulator)", fontsize=9, y=1.04)
        fig.savefig(os.path.join(a.out, "e7_users.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    p = os.path.join(a.out, "detect_eps.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fpr = float(d.loc[d.eps == 0, "flagged"].iloc[0])
        d = d[d.eps > 0]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        for n in dict.fromkeys(d.attack):
            v = d[d.attack == n]
            ax[0].plot(v.eps, 100 * v.flagged, marker="o", ms=3, label=n)
            ax[1].plot(v.eps, v.auc, marker="o", ms=3, label=n)
        ax[0].axhline(100 * fpr, color="#999", ls=":", lw=0.8)
        ax[0].set_ylabel("attacked users flagged (%)"); ax[1].set_ylabel("AUC")
        ax[1].axhline(0.5, color="#999", ls=":", lw=0.8)
        for x in ax:
            x.set_xlabel("eps"); x.set_xticks(EPS_SWEEP); x.spines[["top", "right"]].set_visible(False)
        ax[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Per-user detector vs. attack budget (dotted: clean false-positive rate {100 * fpr:.1f}%), "
                     "U = 20 (Python simulator)", fontsize=9, y=1.02)
        fig.savefig(os.path.join(a.out, "e7_detect_eps.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    p = os.path.join(a.out, "unseen_users.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        acol = dict(zip(dict.fromkeys(d.attack), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
        for (dn, an), v in d.groupby(["detector", "attack"], sort=False):
            ls = "-" if dn.startswith("main") else "--"
            lab = an if dn.startswith("main") else None
            ax[0].plot(v.users, 100 * v.flagged, marker="o", ms=3, ls=ls, color=acol[an], label=lab)
            if an != "none":
                ax[1].plot(v.users, v.auc, marker="o", ms=3, ls=ls, color=acol[an])
        ax[0].plot([], [], color="#555", ls="-", label="main detector (U = 5-40)")
        ax[0].plot([], [], color="#555", ls="--", label="detector fitted on U = 5-20")
        ax[0].set_ylabel("users flagged (%)"); ax[1].set_ylabel("AUC")
        ax[0].axvspan(20, 40, color="#eee", zorder=0); ax[1].axvspan(20, 40, color="#eee", zorder=0)
        for x in ax:
            x.set_xlabel("users U"); x.set_xticks(USERS); x.spines[["top", "right"]].set_visible(False)
        fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, -0.02), ncol=3)
        fig.suptitle("Detection at U not seen in training (shaded: outside the U = 5-20 detector's "
                     "training range); 'none' = clean false-positive rate", fontsize=9, y=1.02)
        fig.savefig(os.path.join(a.out, "e7_unseen_users.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "fit", "suite", "eps", "users", "detect", "unseen", "blind", "summary", "plot"])
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="training seed of the models to evaluate")
    ap.add_argument("--root", default="results/week3", help="where the sanitizer is")
    ap.add_argument("--out", default=None, help="default: <root>/e7 for seed 0, <root>/seed<k>/e7 otherwise")
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.workers is None:
        from rra.parallel import default_workers
        a.workers = default_workers()
    a.e6_dir = a.root if a.seed == 0 else os.path.join(a.root, f"seed{a.seed}")
    a.out = a.out or os.path.join(a.e6_dir, "e7")
    os.makedirs(a.out, exist_ok=True)
    env = PySimEnv(EnvConfig(feature_scale="log"))
    san = Sanitizer.load(os.path.join(a.root, "sanitizer.pt"))
    ctrls = {c.name: c for c in W3.load_controllers(san, a.seed)}
    t0 = time.time()
    if a.part in ("all", "fit"):
        pairs = [(n, EPS_MAIN) for n in ctrls] + [("PPO + rand. sanitizer", e) for e in EPS_SWEEP if e != EPS_MAIN]
        run_fit(a, env, ctrls, pairs)
    if a.part in ("all", "suite"):
        run_suite(a, env, ctrls, ProportionalFair(env))
    if a.part in ("all", "eps"):
        run_eps(a, env, ctrls)
    if a.part in ("all", "users"):
        run_users(a, env, ctrls)
    if a.part in ("all", "detect"):
        run_detect(a, env, san)
    if a.part in ("all", "unseen"):
        run_unseen(a, env, san)
    if a.part in ("all", "blind"):
        run_blind(a, env, ctrls)
    if a.part in ("all", "suite", "eps", "users", "summary"):
        summarise(a)
    if a.part in ("all", "suite", "eps", "users", "detect", "unseen", "summary", "plot"):
        plot(a)
    meta_p = os.path.join(a.out, "meta.json")
    meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
    meta.update({"simulator": "rra.env.pysim (Python)", "episodes": a.episodes, "model_seed": a.seed,
                 "eval_seeds": list(range(1000, 1000 + a.episodes)), "eps_main": EPS_MAIN,
                 "eps_sweep": list(EPS_SWEEP), "users": list(USERS), "beta_detector_aware": BETA_STEALTH,
                 f"runtime_s_{a.part}": round(time.time() - t0, 1)})
    json.dump(meta, open(meta_p, "w"), indent=2)


if __name__ == "__main__":
    main()
