"""Week-4 evaluation of Steps F-G (FLARE report contribution 3), Python simulator.

Parts (--part):
  library   The policy library: the deployed model of this seed fine-tuned on each
            cell's traffic alone (ordinary PPO steps, --lib-iters iterations of 2048
            intervals), and each cell's context summary kappa and report baseline.
  classify  Step F, shift vs attack. Events, each one window (an episode of
            300 intervals) in one cell after a change:
              no change     the same traffic, new seed
              shift         the cell's traffic becomes another cell's (7 per cell)
              attack all    UAP on every user, eps 0.1 and 0.05
              attack k=3    UAP on 3 users, eps 0.1
              stealthy      UAP on every user, eps 0.02 (mostly under the detector)
            The thresholds are calibrated on clean windows not used for events;
            output: the decision for every event and a confusion table.
  adapt     Adaptation after a genuine shift (four pairs of cells): keep the
            current policy and learn online, reuse the library policy of the most
            similar context (Step F) and learn online, learn from scratch, or start
            from a meta-learned initialisation (Reptile, Nichol et al. 2018; trained on
            the other cells' traffic). After every iteration: J, the SLA violations, and
            the Lagrangian score J - sum_l lambda_l V_l with the deployed model's multipliers
            (J alone would reward an allocator that breaks the SLAs).
  retrain   Step G, one cell at a steady load of 20 users, a timeline: clean windows, then a strong new attack;
            the observer sees the drop, Step F calls it an attack, the twin retrains
            on the estimated attack, the candidate is shadow-tested and deployed only
            if better; a faulty candidate is shown being rejected and, if deployed by
            force, rolled back. This is a demonstration of the mechanism (one seed),
            not a statistical evaluation.
  all       library, classify, adapt, retrain.

Usage: python -m experiments.week4_observer --part all --seed 0
Output: results/week4/observer/s<seed>/
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from rra.agents.policy import Actor, Critic
from rra.attacks import UAPAttacker, collect_states, fit_uap
from rra.defence.controller import DefendedController
from rra.defence.sanitizer import Sanitizer
from rra.env.pysim import PySimEnv
from rra.fed.aggregate import mmd2
from rra.fed.cells import CELLS, cell_env_config
from rra.fed.learner import CellLearner, LocalConfig
from rra.fed.observer import ObserverState, ShiftRule, cell_summary, nearest
from rra.runner import run_episode

from experiments.week4_fed import J_of, initial_model

FAST = LocalConfig(lr_actor=3e-4, lr_critic=1e-3, epochs=10, clip=0.2, target_kl=1e9)   # library, twin, scratch
SLOW = LocalConfig()                                                                    # online, after deployment
SHIFT_PAIRS = [(4, 5), (1, 0), (2, 6), (3, 7)]   # (cell, the cell whose traffic it takes on)


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def window(env, ctrl, attacker, seed, n_users, every=5):
    """One episode; returns reports seen, slices, trust (sampled every `every` intervals) and KPIs."""
    obs = env.reset(seed=seed, n_users=n_users)
    torch.manual_seed(20_000 + seed)
    ctrl.reset()
    if attacker is not None:
        attacker.reset(env, ctrl, seed=10_000 + seed)
    R, S, T, nu = [], [], [], []
    from rra.metrics import EpisodeLog
    log = EpisodeLog(warmup=20)
    for t in range(env.cfg.episode_len):
        seen = attacker.perturb(obs) if attacker is not None else obs
        act = ctrl.act(seen)
        if t >= 20 and t % every == 0:
            a = obs.active
            R.append(seen.reported_norm[a]); S.append(obs.slice_id[a]); T.append(ctrl.last_trust[a]); nu.append(a.sum())
        obs, info, done = env.step(act)
        log.add(info)
        if done:
            break
    r = log.summary()
    return np.concatenate(R), np.concatenate(S), np.concatenate(T), float(np.mean(nu)), r


def load_actor(d):
    a = Actor(64, prior="need"); a.load_state_dict(torch.load(f"{d}/actor.pt", map_location="cpu"))
    c = Critic(64); c.load_state_dict(torch.load(f"{d}/critic.pt", map_location="cpu"))
    return a, c, np.load(f"{d}/lambda.npy")


def users_of(spec, seed):
    return int(np.random.default_rng(seed).integers(spec.users[0], spec.users[1] + 1))


# ===================================================================== library
def run_library(a, san):
    actor0, critic0, lam0 = initial_model("pretrained", a.seed)
    lib = {}
    for i, spec in enumerate(CELLS):
        d = os.path.join(a.out, "library", f"cell{i}")
        if not os.path.exists(f"{d}/actor.pt"):
            t = time.time()
            torch.manual_seed(60_000 + 100 * a.seed + i)
            L = CellLearner(i, spec, FAST, actor0, critic0, lam0, san, seed=60_000 + 100 * a.seed + i)
            for _ in range(a.lib_iters):
                L._iteration()
            os.makedirs(d, exist_ok=True)
            torch.save(L.actor.state_dict(), f"{d}/actor.pt"); torch.save(L.critic.state_dict(), f"{d}/critic.pt")
            np.save(f"{d}/lambda.npy", L.lam)
            print(f"library: cell {i} ({spec.name}) {a.lib_iters} iterations, {time.time() - t:.0f} s", flush=True)
        # context summary and report baseline of this cell under its own policy (clean)
        env = PySimEnv(cell_env_config(spec))
        ctrl = DefendedController(load_actor(d)[0], san, "rand")
        reps, sl, tr, nus = [], [], [], []
        for j in range(2):
            s = 80_000 + 100 * i + j
            R, S, T, n, _ = window(env, ctrl, None, s, users_of(spec, s))
            reps.append(R); sl.append(S); tr.append(T); nus.append(n)
        R, S, T = np.concatenate(reps), np.concatenate(sl), np.concatenate(tr)
        lib[i] = {"kappa": cell_summary(R, S, T, np.mean(nus)), "baseline": R}
    np.savez(os.path.join(a.out, "library", "contexts.npz"),
             **{f"kappa{i}": v["kappa"] for i, v in lib.items()}, **{f"base{i}": v["baseline"] for i, v in lib.items()})
    return lib


def load_library(a):
    z = np.load(os.path.join(a.out, "library", "contexts.npz"))
    return {i: {"kappa": z[f"kappa{i}"], "baseline": z[f"base{i}"]} for i in range(len(CELLS))}


# ===================================================================== classify (Step F)
def run_classify(a, san, lib):
    rule = ShiftRule()
    kap = {str(i): v["kappa"] for i, v in lib.items()}
    uap = {}
    rows = []
    # calibrate on clean windows (seeds 90 000+, never used for events): the largest share of
    # distrusted users and the largest move of the context a clean window shows, with a 1.5x margin
    cal_f, cal_d = [], []
    for i, spec in enumerate(CELLS):
        env = PySimEnv(cell_env_config(spec))
        ctrl = DefendedController(load_actor(os.path.join(a.out, "library", f"cell{i}"))[0], san, "rand")
        for j in range(2):
            s = 90_000 + 100 * i + j
            R, S, T, n, _ = window(env, ctrl, None, s, users_of(spec, s))
            cal_f.append(float((T < 0.5).mean()))
            cal_d.append(float(np.linalg.norm(cell_summary(R, S, T, n) - lib[i]["kappa"])))
    rule.f_odd = float(1.5 * max(cal_f))
    rule.d_change = float(1.5 * max(cal_d))
    print(f"classify: calibrated f_odd {rule.f_odd:.3f}, d_change {rule.d_change:.3f}", flush=True)

    for i, spec in enumerate(CELLS):
        own = os.path.join(a.out, "library", f"cell{i}")
        ctrl = DefendedController(load_actor(own)[0], san, "rand")
        env_i = PySimEnv(cell_env_config(spec))
        if i not in uap:
            torch.manual_seed(31_000 + 100 * i + a.seed)
            uap[i] = {e: fit_uap(env_i, ctrl, collect_states(env_i, ctrl, n_episodes=3, users=spec.users,
                                                               seed=31_000 + 100 * i), e, seed=a.seed)
                      for e in (0.02, 0.05, 0.1)}
        events = [("no change", "no change", spec, None, None)]
        events += [("shift", f"shift to {CELLS[j].name}", CELLS[j], None, None) for j in range(len(CELLS)) if j != i]
        events += [("attack", "attack all, eps 0.1", spec, 0.1, None), ("attack", "attack all, eps 0.05", spec, 0.05, None),
                   ("attack", "attack 3 users, eps 0.1", spec, 0.1, 3),
                   ("attack", "stealthy, eps 0.02", spec, 0.02, None)]
        for e_idx, (truth, name, sp, eps, k) in enumerate(events):
            env = PySimEnv(cell_env_config(sp))
            att = UAPAttacker(eps=eps, k=k, table=uap[i][eps]) if eps else None
            s = 85_000 + 1000 * i + e_idx
            R, S, T, n, r = window(env, ctrl, att, s, users_of(sp, s))
            kappa = cell_summary(R, S, T, n)
            drift = float(np.sqrt(max(mmd2(R, lib[i]["baseline"]), 0)))
            f_flag = float((T < 0.5).mean())
            match, d_lib = nearest(kappa, kap, exclude=[str(i)])
            d_own = float(np.linalg.norm(kappa - lib[i]["kappa"]))
            dec = rule.decide(f_flag, d_lib, d_own)
            rows.append({"cell": i, "event": name, "truth": truth, "decision": dec, "drift": drift,
                         "flagged": f_flag, "nearest_cell": int(match), "d_lib": d_lib, "d_own": d_own,
                         "true_target": CELLS.index(sp) if truth == "shift" else -1, "J": J_of(r, SLOW)})
        print(f"classify: cell {i} done", flush=True)
    save_csv(rows, os.path.join(a.out, "classify.csv"))
    d = pd.DataFrame(rows)
    d["kind"] = np.where(d.event.str.startswith("stealthy"), "stealthy attack", d.truth)
    conf = pd.crosstab(d.kind, d.decision)
    conf.to_csv(os.path.join(a.out, "classify_confusion.csv"))
    print(conf.to_string(), flush=True)
    json.dump({"rule": rule.__dict__, "calibration_flagged": cal_f, "calibration_d_own": cal_d},
              open(os.path.join(a.out, "classify_rule.json"), "w"), indent=2)


# ===================================================================== adapt (Step F: reuse)
def reptile_init(a, san, exclude, iters=20, step=0.5):
    actor, critic, lam = initial_model("pretrained", a.seed)
    rng = np.random.default_rng(70_000 + a.seed)
    torch.manual_seed(70_000 + a.seed + 1000 * sum(exclude))
    cells = [i for i in range(len(CELLS)) if i not in exclude]
    for m in range(iters):
        i = int(rng.choice(cells))
        L = CellLearner(i, CELLS[i], FAST, actor, critic, lam, san, seed=71_000 + 100 * a.seed + m)
        L._iteration()
        with torch.no_grad():
            for p, q in zip(actor.parameters(), L.actor.parameters()):
                p.add_(step * (q - p))
            for p, q in zip(critic.parameters(), L.critic.parameters()):
                p.add_(step * (q - p))
    return actor, critic, lam


def run_adapt(a, san, lib):
    rows = []
    lam_ref = initial_model("pretrained", a.seed)[2]
    for src, dst in SHIFT_PAIRS:
        spec = CELLS[dst]                                       # the new traffic of cell src
        starts = {"keep current policy": (load_actor(os.path.join(a.out, "library", f"cell{src}")), SLOW),
                  "reuse library policy": (load_actor(os.path.join(a.out, "library", f"cell{dst}")), SLOW),
                  "from scratch": (initial_model("scratch", a.seed), FAST),
                  "meta-learned start": (reptile_init(a, san, exclude=[dst]), SLOW)}
        for strat, ((act, cri, lam), cfg) in starts.items():
            t = time.time()
            torch.manual_seed(72_000 + 100 * a.seed + 10 * src + dst)
            L = CellLearner(src, spec, cfg, act, cri, lam, san, seed=72_000 + 100 * a.seed + 10 * src + dst)
            env = PySimEnv(cell_env_config(spec))
            for it in range(a.adapt_iters + 1):
                if it > 0:
                    L._iteration()
                rs = [run_episode(env, L.controller(), None, seed=s, n_users=users_of(spec, s))
                      for s in (95_000 + 10 * dst, 95_001 + 10 * dst)]
                v = {k: float(np.nanmean([r[k] for r in rs])) for k in
                     ("viol_eMBB", "viol_URLLC", "viol_mMTC", "any_violation")}
                J = float(np.mean([J_of(r, SLOW) for r in rs]))
                # the allocator's own objective: J minus the SLA violations priced by the deployed
                # model's multipliers (kept fixed, so all strategies are scored on one scale)
                score = J - float(np.dot(lam_ref, np.nan_to_num([v["viol_eMBB"], v["viol_URLLC"], v["viol_mMTC"]])))
                rows.append({"cell": src, "new_traffic": spec.name, "target_cell": dst, "strategy": strat,
                             "iteration": it, "intervals": it * cfg.n_steps * cfg.n_envs, "J": J, **v,
                             "lagrangian_score": score})
            print(f"adapt: {CELLS[src].name} -> {spec.name}: {strat} ({time.time() - t:.0f} s)", flush=True)
        save_csv(rows, os.path.join(a.out, "adapt.csv"))


# ===================================================================== retrain (Step G)
def run_retrain(a, san, lib):
    i = 3                                                    # mixed urban
    spec = CELLS[i]
    env = PySimEnv(cell_env_config(spec))
    live = load_actor(os.path.join(a.out, "library", f"cell{i}"))
    ctrl = DefendedController(copy.deepcopy(live[0]), san, "rand")
    torch.manual_seed(33_000 + a.seed)
    atk_table = fit_uap(env, ctrl, collect_states(env, ctrl, n_episodes=3, users=spec.users, seed=33_000), 0.2,
                        seed=a.seed)
    attack = UAPAttacker(eps=0.2, table=atk_table)
    rule = ShiftRule(**json.load(open(os.path.join(a.out, "classify_rule.json")))["rule"]) \
        if os.path.exists(os.path.join(a.out, "classify_rule.json")) else ShiftRule()
    kap = {str(j): v["kappa"] for j, v in lib.items()}
    obs_state = ObserverState()
    rows, events = [], []
    seeds = [96_000 + w for w in range(40)]
    deployed, previous = ctrl, None
    saved_reports, clean_R, clean_S = [], [], []

    def J_window(c, att, s):          # steady load (20 users), so that J moves with the attack, not the load
        R, S, T, n, r = window(env, c, att, s, 20)
        return J_of(r, SLOW), R, S, T, n

    w = 0
    for phase, n_win, att in (("clean", 6, None), ("new attack", 14, attack)):
        for _ in range(n_win):
            s = seeds[w]
            J, R, S, T, n = J_window(deployed, att, s)
            if phase == "clean":
                clean_R.append(R); clean_S.append(S)
            drop = obs_state.update(J)
            f_flag = float((T < 0.5).mean())
            kappa = cell_summary(R, S, T, n)
            dec = (rule.decide(f_flag, nearest(kappa, kap, exclude=[str(i)])[1],
                               float(np.linalg.norm(kappa - lib[i]["kappa"]))) if drop else "")
            rows.append({"window": w, "phase": phase, "J": J, "level": obs_state.level, "drop": drop,
                         "flagged": f_flag, "decision": dec, "model": "candidate" if previous else "original"})
            if drop and dec == "attack" and previous is None and not events:
                saved_reports.append(R[T < 0.5])
                # retrain in the digital twin on the estimated attack: per-slice mean shift of the
                # distrusted reports against the cell's clean baseline (an estimated universal perturbation)
                est = np.zeros((3, 5))
                Rf, Sf = R[T < 0.5], S[T < 0.5]
                cR, cS = np.concatenate(clean_R), np.concatenate(clean_S)
                for k in range(3):
                    if (Sf == k).any() and (cS == k).any():
                        est[k] = np.median(Rf[Sf == k], 0) - np.median(cR[cS == k], 0)
                est = np.clip(est, -0.2, 0.2)
                t = time.time()
                torch.manual_seed(97_000 + a.seed)
                twin = CellLearner(i, spec, FAST, live[0], live[1], live[2], san, seed=97_000 + a.seed,
                                   attacker=UAPAttacker(eps=0.2, table=est))
                for _ in range(a.twin_iters):
                    twin._iteration()
                cand = DefendedController(twin.actor, san, "rand")
                # faulty candidate (for the rollback demonstration): trained to lower the reward
                bad = CellLearner(i, spec, FAST, live[0], live[1], live[2], san, seed=97_100 + a.seed, malicious=True)
                for _ in range(a.twin_iters):
                    bad._iteration()
                bad_ctrl = DefendedController(bad.actor, san, "rand")
                # shadow test on replayed conditions: the same 6 attacked windows for every model (paired)
                rep_seeds = [96_900 + q for q in range(6)]
                sh = {nm: float(np.mean([J_window(c, attack, s_)[0] for s_ in rep_seeds]))
                      for nm, c in (("live", deployed), ("candidate", cand), ("faulty candidate", bad_ctrl))}
                events.append({"window": w, "event": "retrained in twin", "twin_s": round(time.time() - t, 1),
                               **{f"shadow_J_{k}": v for k, v in sh.items()},
                               "deploy_candidate": sh["candidate"] > sh["live"],
                               "deploy_faulty": sh["faulty candidate"] > sh["live"]})
                print(f"retrain: window {w}: shadow test {sh}", flush=True)
                if sh["candidate"] > sh["live"]:
                    previous, deployed = deployed, cand
                    obs_state = ObserverState(level=obs_state.level)
                bad_hold = bad_ctrl
            w += 1
    # rollback demonstration: deploy the faulty candidate by force, the observer rolls back on the drop
    if events:
        before = deployed
        deployed = bad_hold
        for _ in range(4):
            s = seeds[w] if w < len(seeds) else 96_500 + w
            J, R, S, T, n = J_window(deployed, attack, s)
            drop = obs_state.update(J)
            faulty_live = deployed is bad_hold
            rows.append({"window": w, "phase": "faulty model forced", "J": J, "level": obs_state.level, "drop": drop,
                         "flagged": float((T < 0.5).mean()), "decision": "roll back" if (drop and faulty_live) else "",
                         "model": "faulty" if faulty_live else "candidate"})
            if drop and deployed is bad_hold:
                deployed = before
                events.append({"window": w, "event": "rolled back"})
            w += 1
    save_csv(rows, os.path.join(a.out, "retrain_timeline.csv"))
    json.dump(events, open(os.path.join(a.out, "retrain_events.json"), "w"), indent=2, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "library", "classify", "adapt", "retrain"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lib-iters", type=int, default=25)
    ap.add_argument("--adapt-iters", type=int, default=8)
    ap.add_argument("--twin-iters", type=int, default=10)
    ap.add_argument("--out-root", default="results/week4/observer")
    a = ap.parse_args()
    torch.set_num_threads(1)
    a.out = os.path.join(a.out_root, f"s{a.seed}")
    os.makedirs(a.out, exist_ok=True)
    san = Sanitizer.load("results/week3/sanitizer.pt")
    t0 = time.time()
    lib = run_library(a, san) if a.part in ("all", "library") else load_library(a)
    if a.part in ("all", "classify"):
        run_classify(a, san, lib)
    if a.part in ("all", "adapt"):
        run_adapt(a, san, lib)
    if a.part in ("all", "retrain"):
        run_retrain(a, san, lib)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
