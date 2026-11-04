"""Attacks of report Sec. 5.2 that are fitted before deployment.

All of them use the same hook, budget and masks as ``Attacker`` (hook.py).

* ``UAPAttacker`` - universal adversarial perturbation (Moosavi-Dezfooli et
  al., CVPR 2017): one fixed perturbation per slice (3 x 5 values), the same
  for every user of that slice and every time step. It is fitted once, offline,
  on clean observations, and then needs no computation and no access to the
  current reports at run time.
* ``GeneratorAttacker`` - the report's GAN-based adaptive attacker,
  delta = eps * tanh(G(s)). G is a small per-user network (same weights for
  every user, so it works for any number of users) that sees the user's
  reports, its slice and the mean report of the cell. It is trained **by
  gradient (white-box)** through the deployed pipeline, with EOT over the
  sanitizer's noise. In the detector-aware variant the defender's detector
  plays the discriminator's role: the loss adds beta times the expected share
  of users the detector would flag, so G learns perturbations that look clean.
  The report leaves open whether G is trained by gradient or by RL; we use
  gradient training, which is the stronger (white-box) choice.
* ``TransferAttacker`` - black-box transfer attack (Papernot et al., 2017):
  PGD on a *surrogate* allocator that the attacker trained itself (here the
  clean PPO of another training seed), without knowledge of the deployed
  model or the defence. The perturbation is then sent to the real pipeline.

The fitting objective of UAP and G is the attacker's one-step utility
estimate (``predicted_utility_torch``), the same as for the ``util`` PGD
attack, averaged over a batch of observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..agents.policy import obs_features
from ..env.interface import N_REPORTED, N_SLICES, Observation
from .gradient import GradientAttacker, predicted_utility_torch
from .hook import Attacker


# ======================================================================= data
def collect_states(env, controller, n_episodes: int = 6, users: Sequence[int] = (10, 20, 30),
                   seed: int = 3000, every: int = 2, warmup: int = 20) -> List[Observation]:
    """Clean observations visited by ``controller`` (fitting data for UAP / G).

    Seeds 3000+ never overlap evaluation (1000+), detection (2000+) or the
    sanitizer's data (50 000+).
    """
    out = []
    for ep in range(n_episodes):
        controller.reset()
        obs = env.reset(seed=seed + ep, n_users=int(users[ep % len(users)]))
        for t in range(env.cfg.episode_len):
            if t >= warmup and t % every == 0:
                out.append(obs.copy())
            obs, _, done = env.step(controller.act(obs))
            if done:
                break
    return out


def _stack(obs_list):
    feats = [obs_features(o) for o in obs_list]
    x0 = torch.from_numpy(np.stack([f[0] for f in feats]))
    m = torch.from_numpy(np.stack([f[1] for f in feats]))
    return x0, m


def _pred_util(env, controller, obs_list, s_adv, x0, m, Q):
    """Mean predicted utility of the allocation chosen on the perturbed reports
    (the attacker wants it low)."""
    B = len(obs_list)
    out, e = controller.policy_torch(s_adv.repeat_interleave(Q, 0), x0.repeat_interleave(Q, 0),
                                     m.repeat_interleave(Q, 0))
    prb, pw = out.mean()
    u = torch.stack([predicted_utility_torch(env, obs_list[b], prb[b * Q:(b + 1) * Q, :-1],
                                             pw[b * Q:(b + 1) * Q, :-1]).mean() for b in range(B)])
    return u.mean(), e


def _entry_mask(obs_list, k=None):
    # A2: every reported feature of every active user
    return torch.stack([torch.from_numpy(o.active.astype(np.float32)) for o in obs_list]).unsqueeze(-1)


# ======================================================================== UAP
@dataclass
class UAPAttacker(Attacker):
    strategy: str = "uap"
    table: Optional[np.ndarray] = None          # (N_SLICES, N_REPORTED), |.| <= eps

    def perturb(self, obs: Observation) -> Observation:
        if self.eps <= 0 or self.table is None:
            return obs
        return self.apply_delta(obs, self.table[obs.slice_id])


@torch.enable_grad()            # callers may be inside no_grad()
def fit_uap(env, controller, obs_list: List[Observation], eps: float, iters: int = 300,
            batch: int = 16, eot: int = 4, seed: int = 0, log=None) -> np.ndarray:
    """Signed-gradient descent of the predicted utility over the per-slice table, projected on the budget."""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    table = ((torch.rand((N_SLICES, N_REPORTED), generator=g) * 2 - 1) * eps)
    alpha = eps / 10
    Q = max(1, eot)
    for it in range(iters):
        idx = rng.choice(len(obs_list), size=min(batch, len(obs_list)), replace=False)
        ob = [obs_list[i] for i in idx]
        x0, m = _stack(ob)
        sid = torch.from_numpy(np.stack([o.slice_id for o in ob])).long()
        table = table.detach().requires_grad_(True)
        delta = table[sid] * _entry_mask(ob)
        s_adv = (x0[..., :5] + delta).clamp(0, 1)
        util, _ = _pred_util(env, controller, ob, s_adv, x0, m, Q)
        grad, = torch.autograd.grad(util, table)
        table = (table.detach() - alpha * grad.sign()).clamp(-eps, eps)   # lower the utility
        if log and it % 50 == 0:
            log(f"  uap iter {it}: predicted utility {util.item():.4f}")
    return table.detach().numpy().astype(float)


# ================================================================== generator
class Generator(nn.Module):
    """Per-user generator: (reports, slice, cell mean report) -> delta in [-eps, eps]."""

    def __init__(self, eps: float, hidden: int = 32):
        super().__init__()
        self.eps = eps
        self.net = nn.Sequential(nn.Linear(2 * N_REPORTED + N_SLICES, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, N_REPORTED))

    def forward(self, s, onehot, m):
        mf = m.float().unsqueeze(-1)
        ctx = (s * mf).sum(-2, keepdim=True) / mf.sum(-2, keepdim=True).clamp(min=1)
        h = torch.cat([s, onehot, ctx.expand_as(s)], -1)
        return self.eps * torch.tanh(self.net(h)) * mf


@dataclass
class GeneratorAttacker(Attacker):
    strategy: str = "gan"
    gen: Optional[Generator] = field(default=None, repr=False)

    @torch.no_grad()
    def perturb(self, obs: Observation) -> Observation:
        if self.eps <= 0 or self.gen is None:
            return obs
        x0, m = obs_features(obs)
        x0, m = torch.from_numpy(x0)[None], torch.from_numpy(m)[None]
        d = self.gen(x0[..., :5], x0[..., 8:11], m)[0]
        return self.apply_delta(obs, d.numpy().astype(float))


@torch.enable_grad()
def fit_generator(env, controller, obs_list: List[Observation], eps: float, iters: int = 600,
                  batch: int = 16, eot: int = 4, beta: float = 0.0, lr: float = 1e-3,
                  seed: int = 0, log=None) -> Generator:
    """Train G through the deployed pipeline; beta > 0 adds the detector term."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = Generator(eps)
    opt = torch.optim.Adam(gen.parameters(), lr=lr)
    san = getattr(controller, "san", None)
    detector = san is not None and getattr(controller, "mode", "none") != "none"
    Q = max(1, eot)
    actor_params = list(controller.actor.parameters())
    for p in actor_params:
        p.requires_grad_(False)
    try:
        for it in range(iters):
            idx = rng.choice(len(obs_list), size=min(batch, len(obs_list)), replace=False)
            ob = [obs_list[i] for i in idx]
            x0, m = _stack(ob)
            delta = gen(x0[..., :5], x0[..., 8:11], m)
            s_adv = (x0[..., :5] + delta).clamp(0, 1)
            loss, _ = _pred_util(env, controller, ob, s_adv, x0, m, Q)   # minimised
            if beta > 0 and detector:
                w = san.weight(san.error(s_adv, x0[..., 8:11]), "soft")
                loss = loss + beta * (w * m.float()).sum() / m.float().sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if log and it % 100 == 0:
                log(f"  generator iter {it}: loss {loss.item():.4f}")
    finally:
        for p in actor_params:
            p.requires_grad_(True)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)
    return gen


# =================================================================== transfer
@dataclass
class TransferAttacker(GradientAttacker):
    """PGD on a surrogate controller; the result is sent to the real pipeline."""
    surrogate: object = None

    def _policy(self):
        return self.surrogate.policy_torch
