"""Step D of the FLARE report: one cell's local learning between federated rounds.

Each cell runs the deployed pipeline (detector and randomized sanitizer, then
the trust-aware actor) on its own traffic and keeps learning from its own
experience with small, cautious PPO updates:

* every sample (one control interval) counts in proportion to the average
  trust rho of its users, and samples whose average trust is below
  ``low_trust`` are left out (report Step D);
* PPO uses a small learning rate, a tight clip and an early stop on the
  approximate KL, so one round cannot move the policy far;
* the reward comes from the true KPIs, never from the (possibly manipulated)
  reports; the critic sees what the xApp sees (sanitized reports), because
  after deployment the clean reports are not available.

A cell whose reports are manipulated (``attacker``) follows the protocol
honestly but learns from corrupted inputs ("honest but deceived"). A cell
with ``malicious=True`` attacks the federation on purpose: it trains its
local copy to *lower* the reward (negated advantages, the inverted-objective
model-poisoning attack) and sends that update scaled by ``malicious_scale``.
A plain sign flip of an honest update is not used: honest online updates are
small and nearly uncorrelated between cells, so flipping one does no harm.

The PPO code follows rra/agents/ppo.py (per-user Gamma-form ratios, four value
heads, Lagrangian SLA multipliers) without the week-3 adversarial-training
parts.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from ..agents.policy import (Actor, Critic, gamma_entropy, gamma_log_prob, gamma_sample, obs_features,
                             shares_from_gamma)
from ..agents.ppo import _reward_part, _violations
from ..env.interface import N_SLICES, Action
from ..env.pysim import PySimEnv
from .cells import CellSpec, cell_env_config


@dataclass
class LocalConfig:
    n_envs: int = 8
    n_steps: int = 256              # per env -> 2048 transitions per iteration
    epochs: int = 4
    minibatch: int = 256
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    clip: float = 0.1
    target_kl: float = 0.02         # stop the epochs early once the policy has moved this far
    gamma: float = 0.9
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    zeta_p: float = 0.05
    zeta_n: float = 0.05
    nu: tuple = (0.05, 0.01, 0.05)
    eta_lambda: float = 0.05
    r_ref_bps: float = 1e6
    reward_mode: str = "objective"
    trust_weighting: bool = True
    low_trust: float = 0.2
    mode: str = "rand"              # sanitizer at deployment
    gate: str = "soft"
    malicious_scale: float = 3.0


def shared_vector(actor: Actor) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(actor.shared.parameters()).detach().clone()


def set_shared(actor: Actor, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec.clone(), actor.shared.parameters())


class CellLearner:
    def __init__(self, idx: int, spec: CellSpec, cfg: LocalConfig, actor: Actor, critic: Critic,
                 lam: np.ndarray, san, seed: int, attacker=None, malicious: bool = False):
        self.idx, self.spec, self.cfg, self.san = idx, spec, cfg, san
        self.env_cfg = cell_env_config(spec)
        self.envs = [PySimEnv(self.env_cfg) for _ in range(cfg.n_envs)]
        self.actor, self.critic = copy.deepcopy(actor), copy.deepcopy(critic)
        self.lam = np.array(lam, float).copy()
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor, eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic, eps=1e-5)
        self.rng = np.random.default_rng(seed)
        self.attacker = attacker
        self.malicious = malicious
        if attacker is not None:
            attacker.reset(self.envs[0], None, seed=seed + 99)
        self.obs = [self._new_episode(i) for i in range(cfg.n_envs)]
        self.steps = 0

    # ------------------------------------------------------------------ helpers
    def _new_episode(self, i):
        n = int(self.rng.integers(self.spec.users[0], self.spec.users[1] + 1))
        return self.envs[i].reset(seed=int(self.rng.integers(1 << 30)), n_users=n, churn=True)

    def _features(self, obs_list):
        """Actor/critic input after the attacker and the sanitizer, the received reports and trust."""
        U = self.envs[0].u_max
        XC = np.zeros((len(obs_list), U, 11), np.float32); M = np.zeros((len(obs_list), U), bool)
        for i, o in enumerate(obs_list):
            seen = self.attacker.perturb(o) if self.attacker is not None else o
            XC[i], M[i] = obs_features(seen)
        X = XC.copy()
        rho = M.astype(np.float32)
        if self.san is not None and self.cfg.mode != "none":
            with torch.no_grad():
                st, r, _ = self.san(torch.from_numpy(XC[..., :5]), torch.from_numpy(XC[..., 8:11]),
                                    torch.from_numpy(M), self.cfg.mode, self.cfg.gate)
            X[..., :5] = st.numpy(); X[..., 5] = r.numpy(); rho = r.numpy()
        X[~M] = 0.0
        return X, M, XC[..., :5], rho

    # ------------------------------------------------------------------ one round
    def local_round(self, n_iters: int) -> Dict:
        """Run n_iters PPO iterations on this cell's experience; return the round's statistics."""
        cfg = self.cfg
        before = shared_vector(self.actor)
        rows, reports, weights, util = [], [], [], []
        for _ in range(n_iters):
            st = self._iteration()
            rows.append(st); reports.append(st.pop("reports")); weights.append(st.pop("weights"))
        delta = shared_vector(self.actor) - before
        if self.malicious:     # boosted update of a model trained to do harm
            delta = cfg.malicious_scale * delta
            set_shared(self.actor, before)   # its own deployed model is not damaged
        rep = np.concatenate(reports)
        w = np.concatenate(weights)
        return {"delta": delta, "reports": rep, "rho_bar": float(w.mean()),
                "kept_frac": float((w >= cfg.low_trust).mean()),
                "utility": float(np.mean([r["utility"] for r in rows])),
                "viol": np.mean([r["viol"] for r in rows], 0), "kl": float(np.mean([r["kl"] for r in rows])),
                "steps": self.steps}

    def _iteration(self) -> Dict:
        cfg = self.cfg
        T, N, U = cfg.n_steps, cfg.n_envs, self.envs[0].u_max
        X = np.zeros((T, N, U, 11), np.float32); M = np.zeros((T, N, U), bool)
        Gl = np.zeros((T, N, 1), np.float32); W = np.zeros((T, N), np.float32)
        G1 = np.zeros((T, N, U + 1), np.float32); G2 = np.zeros((T, N, U + 1), np.float32)
        LPC = np.zeros((T, N, 2, U + 1), np.float32)
        R = np.zeros((T, N, 4), np.float32); VAL = np.zeros((T, N, 4), np.float32)
        BOOT = np.zeros((T, N, 4), np.float32); DONE = np.zeros((T, N), bool)
        present = np.zeros((T, N, N_SLICES), bool)
        rep_sample, util = [], []
        for t in range(T):
            X[t], M[t], seen, rho = self._features(self.obs)
            Gl[t, :, 0] = [o.n_active / U for o in self.obs]
            W[t] = (rho * M[t]).sum(-1) / np.maximum(M[t].sum(-1), 1)
            if t % 16 == 0:
                rep_sample.append(seen[M[t]])
            xt, mt = torch.from_numpy(X[t]), torch.from_numpy(M[t])
            with torch.no_grad():
                out = self.actor(xt, mt)
                g1, g2 = gamma_sample(out.alpha_prb, out.cmask), gamma_sample(out.alpha_pow, out.cmask)
                a1, a2 = shares_from_gamma(g1), shares_from_gamma(g2)
                LPC[t] = torch.stack([gamma_log_prob(out.alpha_prb, g1, out.cmask),
                                      gamma_log_prob(out.alpha_pow, g2, out.cmask)], 1).numpy()
                VAL[t] = self.critic(xt, mt, torch.from_numpy(Gl[t])).numpy()
            G1[t], G2[t] = g1.numpy(), g2.numpy()
            a1, a2 = a1.numpy(), a2.numpy()
            for i in range(N):
                prev_avg = self.envs[i].avg_rate.copy()
                nxt, info, done = self.envs[i].step(Action(a1[i, :-1].astype(float), a2[i, :-1].astype(float)))
                R[t, i, 0] = _reward_part(info, cfg, prev_avg)
                R[t, i, 1:] = _violations(info)
                present[t, i] = np.isfinite(info.slice_violation)
                util.append(info.utility)
                if done:
                    DONE[t, i] = True
                    xf, mf, _, _ = self._features([nxt])
                    with torch.no_grad():
                        BOOT[t, i] = self.critic(torch.from_numpy(xf), torch.from_numpy(mf),
                                                 torch.tensor([[nxt.n_active / U]], dtype=torch.float32)).numpy()[0]
                    nxt = self._new_episode(i)
                self.obs[i] = nxt
        self.steps += T * N
        xl, ml, _, _ = self._features(self.obs)
        with torch.no_grad():
            last_v = self.critic(torch.from_numpy(xl), torch.from_numpy(ml),
                                 torch.tensor([[o.n_active / U] for o in self.obs], dtype=torch.float32)).numpy()
        ADV = np.zeros_like(R); gae = np.zeros((N, 4), np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else VAL[t + 1]
            nv = np.where(DONE[t][:, None], BOOT[t], nv)
            delta = R[t] + cfg.gamma * nv - VAL[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * np.where(DONE[t][:, None], 0.0, gae)
            ADV[t] = gae
        RET = ADV + VAL
        adv_total = ADV[..., 0] - (ADV[..., 1:] * self.lam).sum(-1)
        if self.malicious:
            adv_total = -adv_total

        # sample weights: average user trust; very low trust -> left out (report Step D)
        w = W if cfg.trust_weighting else np.ones_like(W)
        w = np.where(W < cfg.low_trust, 0.0, w).astype(np.float32)
        flat = lambda a: torch.from_numpy(np.ascontiguousarray(a).reshape(T * N, *a.shape[2:]))
        bx, bm, bg = flat(X), flat(M), flat(Gl)
        bg1, bg2, blpc = flat(G1), flat(G2), flat(LPC)
        badv, bret, bw = flat(adv_total.astype(np.float32)), flat(RET.astype(np.float32)), flat(w)
        kls = []
        stop = False
        for _ in range(cfg.epochs):
            perm = torch.randperm(T * N)
            for s in range(0, T * N, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                wi = bw[idx]
                if wi.sum() <= 0:
                    continue
                out = self.actor(bx[idx], bm[idx])
                adv = badv[idx]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                lpc = torch.stack([gamma_log_prob(out.alpha_prb, bg1[idx], out.cmask),
                                   gamma_log_prob(out.alpha_pow, bg2[idx], out.cmask)], 1)
                cm = out.cmask.unsqueeze(1).expand_as(lpc).float()
                ratio = torch.exp((lpc - blpc[idx]) * cm)
                a_ = adv[:, None, None]
                surr = torch.min(ratio * a_, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_)
                ncomp = cm.sum((1, 2))
                per = (surr * cm).sum((1, 2)) / ncomp
                ent = (torch.stack([gamma_entropy(out.alpha_prb, out.cmask),
                                    gamma_entropy(out.alpha_pow, out.cmask)], 1) * cm).sum((1, 2)) / ncomp
                loss_a = -((per + cfg.ent_coef * ent) * wi).sum() / wi.sum()
                self.opt_a.zero_grad(); loss_a.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm); self.opt_a.step()
                v = self.critic(bx[idx], bm[idx], bg[idx])
                loss_c = ((((v - bret[idx]) / self.critic.scale) ** 2).mean(-1) * wi).sum() / wi.sum()
                self.opt_c.zero_grad(); loss_c.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm); self.opt_c.step()
                with torch.no_grad():
                    kl = ((((ratio - 1) - torch.log(ratio)) * cm).sum() / cm.sum()).item()
                kls.append(kl)
                if kl > 1.5 * cfg.target_kl:
                    stop = True
                    break
            if stop:
                break
        V = R[..., 1:]
        v_hat = np.array([V[..., l][present[..., l]].mean() if present[..., l].any() else 0.0
                          for l in range(N_SLICES)])
        self.lam = np.maximum(0.0, self.lam + cfg.eta_lambda * (v_hat - np.array(cfg.nu)))
        return {"utility": float(np.mean(util)), "viol": v_hat, "kl": float(np.mean(kls)) if kls else 0.0,
                "reports": np.concatenate(rep_sample), "weights": W.reshape(-1)}

    # ------------------------------------------------------------------ deployment view
    def controller(self):
        from ..defence.controller import DefendedController
        return DefendedController(self.actor, self.san, mode=self.cfg.mode, name=f"cell {self.idx}")
