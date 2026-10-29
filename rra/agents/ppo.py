"""PPO with Lagrangian SLA multipliers (RCPO) for the per-cell allocator.

Report Sec. III-D and IV Step 4:
    r_t = Phi_t - zeta_p P_t - zeta_n B_t - sum_l lambda_l V_{l,t}
    lambda_l <- [lambda_l + eta_lambda (V_hat_l - nu_l)]_+     after every iteration
Rewards come from simulator ground truth (StepInfo), never from reports.

Episodes end by a time limit (300 intervals), not by a terminal state, so the
value of the last state is bootstrapped.
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..baselines import Controller
from ..config import EnvConfig
from ..env.interface import N_SLICES, Action, Observation
from ..env.pysim import PySimEnv
from .policy import (Actor, Critic, dir_entropy, dir_log_prob, dir_sample,
                     gamma_entropy, gamma_log_prob, gamma_sample, n_params,
                     obs_features, shares_from_gamma)


@dataclass
class PPOConfig:
    total_steps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 256               # per env -> 2048 transitions per iteration (report)
    epochs: int = 10
    minibatch: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    clip: float = 0.2
    gamma: float = 0.9
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    hidden: int = 64
    zeta_p: float = 0.05
    zeta_n: float = 0.05
    nu: Tuple[float, float, float] = (0.05, 0.01, 0.05)   # tolerated violation rate per slice
    eta_lambda: float = 0.05
    lambda_init: float = 0.5
    train_users: Tuple[int, int] = (5, 30)
    churn: bool = True
    reward_mode: str = "objective"   # see _reward_part
    # "per_user": one PPO ratio per user and resource, using the Gamma form of
    # the Dirichlet (MAPPO-style, shared advantage).  "joint": one ratio for
    # the whole 2 x (U+1)-dimensional Dirichlet action (did not learn, see
    # docs/week2_design.md).
    ratio_mode: str = "per_user"
    r_ref_bps: float = 1e6           # must match EnvConfig.r_ref_bps
    init_precision: float = 20.0
    prior: str = "need"              # "none" | "need" (see policy.Actor)
    seed: int = 0


class PPOController(Controller):
    """Deployable xApp policy: deterministic Dirichlet mean (report Step 9)."""
    name = "PPO"

    def __init__(self, actor: Actor, stochastic: bool = False):
        self.actor = actor
        self.stochastic = stochastic

    @torch.no_grad()
    def act(self, obs: Observation) -> Action:
        x, m = obs_features(obs)
        out = self.actor(torch.from_numpy(x)[None], torch.from_numpy(m)[None])
        if self.stochastic:
            prb = dir_sample(out.alpha_prb, out.cmask)[0].numpy()
            pw = dir_sample(out.alpha_pow, out.cmask)[0].numpy()
        else:
            prb, pw = (t[0].numpy() for t in out.mean())
        return Action(prb_share=prb[:-1].astype(float), power_share=pw[:-1].astype(float))

    @classmethod
    def load(cls, path: str, hidden: int = 64, prior: Optional[str] = None) -> "PPOController":
        if prior is None:   # read it from the run's config.json when present
            cfg_path = os.path.join(os.path.dirname(path), "config.json")
            prior = json.load(open(cfg_path))["ppo"].get("prior", "none") if os.path.exists(cfg_path) else "none"
        actor = Actor(hidden, prior=prior)
        actor.load_state_dict(torch.load(path, map_location="cpu"))
        actor.eval()
        return cls(actor)


def _reward_part(info, cfg: "PPOConfig", prev_avg: np.ndarray) -> float:
    """Reward without the SLA terms.

    objective : Phi_t - zeta_p P_t - zeta_n B_t  (report Eq. 6; Phi uses the
                moving-average rate Rbar_t)
    marginal  : (1/U) sum_u R_{u,t} / (Rbar_{u,t-1} + R_ref) - zeta_p P - zeta_n B.
                This is t_w times the first-order change of Phi caused by this
                interval's rates, i.e. the PF metric.  It gives each action an
                immediate signal instead of one smeared over ~t_w intervals.
    debug_embb: share of PRBs given to eMBB users (pipeline sanity check only)
    """
    a = info.active
    cost = cfg.zeta_p * info.power_frac + cfg.zeta_n * info.prb_frac
    if cfg.reward_mode == "objective":
        return info.utility - cost
    if cfg.reward_mode == "marginal":
        if not a.any():
            return -cost
        return float(np.mean(info.rate_bps[a] / (prev_avg[a] + cfg.r_ref_bps))) - cost
    if cfg.reward_mode == "debug_embb":
        return float(info.n_prb[a & (info.slice_id == 0)].sum() / max(info.n_prb.sum(), 1))
    raise ValueError(cfg.reward_mode)


def _violations(info) -> np.ndarray:
    return np.nan_to_num(info.slice_violation, nan=0.0)


def train(cfg: PPOConfig, out_dir: str, env_cfg: Optional[EnvConfig] = None,
          log_every: int = 5) -> Actor:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env_cfg = env_cfg or EnvConfig()
    envs = [PySimEnv(env_cfg) for _ in range(cfg.n_envs)]
    U = envs[0].u_max
    actor, critic = Actor(cfg.hidden, cfg.init_precision, env_cfg, cfg.prior), Critic(cfg.hidden)
    opt_a = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor, eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic, eps=1e-5)
    lam = np.full(N_SLICES, cfg.lambda_init)
    nu = np.array(cfg.nu)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"ppo": asdict(cfg), "env_feature_scale": env_cfg.feature_scale,
                   "actor_params": n_params(actor), "critic_params": n_params(critic)}, f, indent=2)

    def new_episode(i):
        n = int(rng.integers(cfg.train_users[0], cfg.train_users[1] + 1))
        return envs[i].reset(seed=int(rng.integers(1 << 30)), n_users=n, churn=cfg.churn)

    obs = [new_episode(i) for i in range(cfg.n_envs)]
    T, N = cfg.n_steps, cfg.n_envs
    n_iters = cfg.total_steps // (T * N)
    log_rows: List[dict] = []
    t_start = time.time()
    ep_stats: List[np.ndarray] = []

    for it in range(n_iters):
        X = np.zeros((T, N, U, 11), np.float32); M = np.zeros((T, N, U), bool)
        G = np.zeros((T, N, 1), np.float32)
        A_prb = np.zeros((T, N, U + 1), np.float32); A_pow = np.zeros((T, N, U + 1), np.float32)
        LP = np.zeros((T, N), np.float32)
        G1 = np.zeros((T, N, U + 1), np.float32); G2 = np.zeros((T, N, U + 1), np.float32)
        LPC = np.zeros((T, N, 2, U + 1), np.float32)     # per-component log-probs
        R = np.zeros((T, N, 4), np.float32)           # [Phi - zP - zB, V_eMBB, V_URLLC, V_mMTC]
        VAL = np.zeros((T, N, 4), np.float32); BOOT = np.zeros((T, N, 4), np.float32)
        DONE = np.zeros((T, N), bool)
        present = np.zeros((T, N, N_SLICES), bool)
        kpi = []
        for t in range(T):
            for i in range(N):
                X[t, i], M[t, i] = obs_features(obs[i])
                G[t, i, 0] = obs[i].n_active / U
            xt, mt, gt = torch.from_numpy(X[t]), torch.from_numpy(M[t]), torch.from_numpy(G[t])
            with torch.no_grad():
                out = actor(xt, mt)
                g1, g2 = gamma_sample(out.alpha_prb, out.cmask), gamma_sample(out.alpha_pow, out.cmask)
                a1, a2 = shares_from_gamma(g1), shares_from_gamma(g2)
                lp = dir_log_prob(out.alpha_prb, a1, out.cmask) + dir_log_prob(out.alpha_pow, a2, out.cmask)
                lpc = torch.stack([gamma_log_prob(out.alpha_prb, g1, out.cmask),
                                   gamma_log_prob(out.alpha_pow, g2, out.cmask)], 1)
                VAL[t] = critic(xt, mt, gt).numpy()
            A_prb[t], A_pow[t], LP[t] = a1.numpy(), a2.numpy(), lp.numpy()
            G1[t], G2[t], LPC[t] = g1.numpy(), g2.numpy(), lpc.numpy()
            for i in range(N):
                prev_avg = envs[i].avg_rate.copy()
                nxt, info, done = envs[i].step(Action(A_prb[t, i, :-1].astype(float),
                                                      A_pow[t, i, :-1].astype(float)))
                R[t, i, 0] = _reward_part(info, cfg, prev_avg)
                R[t, i, 1:] = _violations(info)
                present[t, i] = np.isfinite(info.slice_violation)
                kpi.append([info.utility, info.power_frac, info.prb_frac, *info.slice_violation])
                if done:
                    DONE[t, i] = True
                    xf, mf = obs_features(nxt)
                    with torch.no_grad():
                        BOOT[t, i] = critic(torch.from_numpy(xf)[None], torch.from_numpy(mf)[None],
                                            torch.tensor([[nxt.n_active / U]], dtype=torch.float32)).numpy()[0]
                    nxt = new_episode(i)
                obs[i] = nxt
        # value of the state after the rollout
        with torch.no_grad():
            xl = np.stack([obs_features(o)[0] for o in obs]); ml = np.stack([o.active for o in obs])
            gl = np.array([[o.n_active / U] for o in obs], np.float32)
            last_v = critic(torch.from_numpy(xl), torch.from_numpy(ml), torch.from_numpy(gl)).numpy()

        # GAE per value head (time-limit ends are bootstrapped, not terminal)
        ADV = np.zeros_like(R); gae = np.zeros((N, 4), np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else VAL[t + 1]
            nv = np.where(DONE[t][:, None], BOOT[t], nv)
            delta = R[t] + cfg.gamma * nv - VAL[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * np.where(DONE[t][:, None], 0.0, gae)
            ADV[t] = gae
        RET = ADV + VAL
        adv_total = ADV[..., 0] - (ADV[..., 1:] * lam).sum(-1)

        # PPO update
        flat = lambda a: torch.from_numpy(a.reshape(T * N, *a.shape[2:]))
        bx, bm, bg = flat(X), flat(M), flat(G)
        b1, b2, blp = flat(A_prb), flat(A_pow), flat(LP)
        bg1, bg2, blpc = flat(G1), flat(G2), flat(LPC)
        badv, bret = flat(adv_total.astype(np.float32)), flat(RET.astype(np.float32))
        stats = {"kl": [], "clipfrac": [], "ent": [], "vloss": []}
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * N)
            for s in range(0, T * N, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                out = actor(bx[idx], bm[idx])
                adv = badv[idx]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                if cfg.ratio_mode == "joint":
                    lp = (dir_log_prob(out.alpha_prb, b1[idx], out.cmask)
                          + dir_log_prob(out.alpha_pow, b2[idx], out.cmask))
                    ratio = torch.exp(lp - blp[idx])
                    pg = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv).mean()
                    ent = (dir_entropy(out.alpha_prb, out.cmask) + dir_entropy(out.alpha_pow, out.cmask)).mean()
                else:
                    lpc = torch.stack([gamma_log_prob(out.alpha_prb, bg1[idx], out.cmask),
                                       gamma_log_prob(out.alpha_pow, bg2[idx], out.cmask)], 1)
                    cm = out.cmask.unsqueeze(1).expand_as(lpc).float()
                    ratio = torch.exp((lpc - blpc[idx]) * cm)
                    a_ = adv[:, None, None]
                    surr = torch.min(ratio * a_, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_)
                    ncomp = cm.sum((1, 2))
                    pg = -((surr * cm).sum((1, 2)) / ncomp).mean()
                    ent = ((torch.stack([gamma_entropy(out.alpha_prb, out.cmask),
                                         gamma_entropy(out.alpha_pow, out.cmask)], 1) * cm).sum((1, 2))
                           / ncomp).mean()
                loss_a = pg - cfg.ent_coef * ent
                opt_a.zero_grad(); loss_a.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm); opt_a.step()
                v = critic(bx[idx], bm[idx], bg[idx])
                loss_c = (((v - bret[idx]) / critic.scale) ** 2).mean()
                opt_c.zero_grad(); loss_c.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm); opt_c.step()
                with torch.no_grad():
                    if cfg.ratio_mode == "joint":
                        stats["kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                        stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())
                    else:   # averaged over existing components
                        stats["kl"].append(((((ratio - 1) - torch.log(ratio)) * cm).sum() / cm.sum()).item())
                        stats["clipfrac"].append(((((ratio - 1).abs() > cfg.clip).float() * cm).sum()
                                                  / cm.sum()).item())
                    stats["ent"].append(ent.item()); stats["vloss"].append(loss_c.item())

        # dual (lambda) update from the rollout's mean violation per slice
        V = R[..., 1:]
        v_hat = np.array([V[..., l][present[..., l]].mean() if present[..., l].any() else 0.0
                          for l in range(N_SLICES)])
        lam = np.maximum(0.0, lam + cfg.eta_lambda * (v_hat - nu))

        k = np.array(kpi, float)
        row = {"iter": it + 1, "steps": (it + 1) * T * N, "time_s": round(time.time() - t_start, 1),
               "utility": k[:, 0].mean(), "power_frac": k[:, 1].mean(), "prb_frac": k[:, 2].mean(),
               "viol_eMBB": v_hat[0], "viol_URLLC": v_hat[1], "viol_mMTC": v_hat[2],
               "lambda_eMBB": lam[0], "lambda_URLLC": lam[1], "lambda_mMTC": lam[2],
               "kl": np.mean(stats["kl"]), "clipfrac": np.mean(stats["clipfrac"]),
               "entropy": np.mean(stats["ent"]), "vloss": np.mean(stats["vloss"])}
        log_rows.append(row)
        if (it + 1) % log_every == 0 or it == n_iters - 1:
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
            with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys())); w.writeheader(); w.writerows(log_rows)
            torch.save(actor.state_dict(), os.path.join(out_dir, "actor.pt"))
            torch.save(critic.state_dict(), os.path.join(out_dir, "critic.pt"))
            np.save(os.path.join(out_dir, "lambda.npy"), lam)
    return actor
