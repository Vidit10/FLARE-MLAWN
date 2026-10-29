"""Permutation-equivariant actor-critic with masked Dirichlet outputs.

Report, Sec. IV Step 4.  For every active user u the actor builds

    y_u = [s_u (5 reported, scaled), rho_u (trust), n_{t-1}/N, p_{t-1}/P_max, slice one-hot (3)]  in R^11

and computes  h_u = f_shared(y_u),  c = mean_v h_v  (Deep Sets).  The cell head
maps [h_u, c] to one logit per user for PRBs and one for power, and c to the
logits of the extra "unused" component and to the Dirichlet precisions:

    m   = softmax over {users, unused}  (one softmax for PRBs, one for power)
    A   = exp(a(c)) * (U + 1)           (precision; larger = less exploration)
    alpha_i = m_i * A

so the action distribution is Dir(alpha) for PRB shares and Dir(alpha') for
power shares.  Permuting users permutes the shares (equivariance) and the
policy works for any number of users up to u_max.  In deployment the mean m
is used.

The encoder ``shared`` and the ``head`` are separate modules on purpose: in
Phase 4 only ``shared`` is federated, each cell keeps its own head.

The critic has its own encoder (it is never deployed, so it may see clean
reports) and four value heads: one for the return of
Phi - zeta_p P - zeta_n B, one for the return of each slice's violation rate.
With Lagrange multipliers lambda the advantage is A_r - sum_l lambda_l A_l, so
changing lambda needs no critic retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..env.interface import N_REPORTED, N_SLICES, Observation

FEAT_DIM = N_REPORTED + 1 + 2 + N_SLICES      # 11
N_VALUE_HEADS = 1 + N_SLICES                  # reward part + one per slice


def obs_features(obs: Observation, reported_norm: Optional[np.ndarray] = None,
                 trust: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (features (u_max, 11) float32, mask (u_max,) bool)."""
    s = obs.reported_norm if reported_norm is None else reported_norm
    rho = np.ones(obs.active.size) if trust is None else trust
    onehot = np.eye(N_SLICES)[obs.slice_id]
    x = np.concatenate([s, rho[:, None], obs.last_prb_share[:, None],
                        obs.last_power_share[:, None], onehot], axis=1)
    x[~obs.active] = 0.0
    return x.astype(np.float32), obs.active.copy()


def mlp(i, h, o, act=nn.Tanh):
    return nn.Sequential(nn.Linear(i, h), act(), nn.Linear(h, h), act(), nn.Linear(h, o))


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    return (h * m).sum(-2) / m.sum(-2).clamp(min=1.0)


# --------------------------------------------------------------- Dirichlet
# Tensors have a trailing component axis of size u_max + 1 (last = "unused");
# ``cmask`` marks which components exist (active users + unused).

def dir_log_prob(alpha, x, cmask):
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    lx = torch.log(x.clamp(min=1e-12))
    a0 = (alpha * cmask).sum(-1)
    return (torch.lgamma(a0) - (torch.lgamma(a) * cmask).sum(-1)
            + ((a - 1.0) * lx * cmask).sum(-1))


def dir_entropy(alpha, cmask):
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    k = cmask.sum(-1).float()
    a0 = (alpha * cmask).sum(-1)
    log_b = (torch.lgamma(a) * cmask).sum(-1) - torch.lgamma(a0)
    return (log_b + (a0 - k) * torch.digamma(a0)
            - ((a - 1.0) * torch.digamma(a) * cmask).sum(-1))


def dir_kl(alpha, beta, cmask):
    """KL( Dir(alpha) || Dir(beta) ) over the components in cmask (closed form)."""
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    b = torch.where(cmask, beta, torch.ones_like(beta))
    a0 = (a * cmask).sum(-1)
    b0 = (b * cmask).sum(-1)
    return (torch.lgamma(a0) - (torch.lgamma(a) * cmask).sum(-1)
            - torch.lgamma(b0) + (torch.lgamma(b) * cmask).sum(-1)
            + ((a - b) * (torch.digamma(a) - torch.digamma(a0).unsqueeze(-1)) * cmask).sum(-1))


def dir_sample(alpha, cmask):
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    g = torch.distributions.Gamma(a, torch.ones_like(a)).sample()
    g = torch.where(cmask, g.clamp(min=1e-30), torch.zeros_like(g))
    return g / g.sum(-1, keepdim=True)


# ------------------------------------------------- Gamma form of the Dirichlet
# If g_i ~ Gamma(alpha_i, 1) independently, then g / sum(g) ~ Dir(alpha).
# Training treats the per-user g_i as the sampled action (one factor per user),
# so PPO can clip one probability ratio per user instead of one joint ratio
# over up to 2 x 41 components (see ppo.py).  The shares the network receives
# are still Dirichlet-distributed.

def gamma_sample(alpha, cmask):
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    g = torch.distributions.Gamma(a, torch.ones_like(a)).sample()
    return torch.where(cmask, g.clamp(min=1e-30), torch.zeros_like(g))


def gamma_log_prob(alpha, g, cmask):
    """Per-component log density of Gamma(alpha, 1) at g; 0 where cmask is False."""
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    lg = torch.log(g.clamp(min=1e-30))
    lp = (a - 1.0) * lg - g - torch.lgamma(a)
    return torch.where(cmask, lp, torch.zeros_like(lp))


def gamma_entropy(alpha, cmask):
    a = torch.where(cmask, alpha, torch.ones_like(alpha))
    h = a + torch.lgamma(a) + (1.0 - a) * torch.digamma(a)
    return torch.where(cmask, h, torch.zeros_like(h))


def shares_from_gamma(g):
    return g / g.sum(-1, keepdim=True).clamp(min=1e-30)


# ------------------------------------------------------------------ network
@dataclass
class PolicyOut:
    alpha_prb: torch.Tensor    # (B, u_max + 1)
    alpha_pow: torch.Tensor    # (B, u_max + 1)
    cmask: torch.Tensor        # (B, u_max + 1) bool

    def mean(self):
        return (self.alpha_prb / self.alpha_prb.sum(-1, keepdim=True),
                self.alpha_pow / self.alpha_pow.sum(-1, keepdim=True))


class NeedFeature(nn.Module):
    """Fixed (not learned), differentiable physics feature computed from the
    scaled reports: how many PRBs the user needs to empty its buffer.

        need_bits = buffer + demand * T_c
        bits/PRB  = B_PRB * SE(SINR) * T_c        (same link model as the simulator)
        n_need    = need_bits / bits/PRB

    Returns [min(1, n_need / N), log(1 + n_need) / log(1 + N)].  It adds no
    new input: it is a deterministic function of the 5 attackable reports, so a
    white-box attacker differentiates through it like through the rest of the
    network.
    """

    def __init__(self, env_cfg=None):
        super().__init__()
        from ..config import EnvConfig
        from ..env.pysim import PySimEnv
        env_cfg = env_cfg or EnvConfig()
        b = PySimEnv(env_cfg).bounds
        lo, hi = b._g(b.low), b._g(b.high)
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        self.register_buffer("logf", torch.tensor(b.log_feat, dtype=torch.bool))
        self.register_buffer("x0", torch.tensor(b.x0, dtype=torch.float32))
        r = env_cfg.radio
        self.alpha, self.se_max, self.n_prb = r.se_alpha, r.se_max, float(r.n_prb)
        self.bits_scale = r.prb_bw_hz * env_cfg.t_c
        self.t_c = env_cfg.t_c

    def raw(self, s):
        y = self.lo + s * (self.hi - self.lo)
        return torch.where(self.logf, self.x0 * (torch.pow(10.0, torch.where(self.logf, y, torch.zeros_like(y))) - 1.0), y)

    def need_prbs(self, s):
        """PRBs needed to empty the buffer this interval (capped at N)."""
        raw = self.raw(s)
        sinr = torch.pow(10.0, raw[..., 0] / 10.0)
        se = torch.clamp(self.alpha * torch.log2(1.0 + sinr), min=1e-3, max=self.se_max)
        need = (raw[..., 1].clamp(min=0) + raw[..., 2].clamp(min=0) * self.t_c) / (self.bits_scale * se)
        return torch.clamp(need, max=self.n_prb)

    def forward(self, s):                       # s: (..., 5) scaled reports
        need = self.need_prbs(s)
        return torch.stack([need / self.n_prb, torch.log1p(need) / float(np.log1p(self.n_prb))], -1)


class Actor(nn.Module):
    def __init__(self, hidden: int = 64, init_precision: float = 20.0, env_cfg=None,
                 prior: str = "none"):
        super().__init__()
        self.need = NeedFeature(env_cfg)
        # prior="need": the logits are  log(prior share) + learned correction,
        # where the prior gives each user the PRBs it needs (from its reports)
        # and leaves the rest unused; the network learns the residual
        # (residual policy learning).  prior="none": equal shares at start.
        self.prior = prior
        self.shared = mlp(FEAT_DIM + 2, hidden, hidden)        # federated in Phase 4
        self.head = nn.ModuleDict({                            # stays in the cell
            "user": mlp(2 * hidden + 2, hidden, 2),            # PRB and power logit per user
            "cell": mlp(hidden + 2, hidden, 4),                # unused logits (2) + log precision (2)
        })
        self.log_prec0 = float(np.log(init_precision))
        # start close to "equal shares, little unused"
        nn.init.zeros_(self.head["user"][-1].weight); nn.init.zeros_(self.head["user"][-1].bias)
        nn.init.zeros_(self.head["cell"][-1].weight)
        with torch.no_grad():
            # Without a prior: start with little left unused.  With the need
            # prior: all corrections start at exactly zero (policy = prior).
            off = -2.0 if prior == "none" else 0.0
            self.head["cell"][-1].bias.copy_(torch.tensor([off, off, 0.0, 0.0]))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> PolicyOut:
        nf = self.need(x[..., :5]) * mask.unsqueeze(-1)       # (B, U, 2)
        h = self.shared(torch.cat([x, nf], -1))               # (B, U, H)
        n_act = mask.sum(-1, keepdim=True).float()
        # Deep-Sets context: mean of embeddings, plus load and total PRB need
        glob = torch.cat([n_act / mask.shape[-1], nf[..., 0].sum(-1, keepdim=True).clamp(max=4.0) / 4.0], -1)
        c = torch.cat([masked_mean(h, mask), glob], -1)       # (B, H + 2)
        u = self.head["user"](torch.cat([h, c.unsqueeze(1).expand(-1, h.shape[1], -1)], -1))
        g = self.head["cell"](c)                                                # (B, 4)
        cmask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
        if self.prior == "need":
            nd = self.need.need_prbs(x[..., :5]) * mask
            left = (self.need.n_prb - nd.sum(-1, keepdim=True)).clamp(min=0.0)
            base = torch.log(torch.cat([nd, left], -1) + 0.5)
        else:
            base = torch.zeros_like(cmask, dtype=x.dtype)
        outs = []
        for k in range(2):
            logits = base + torch.cat([u[..., k], g[:, k:k + 1]], -1)
            logits = logits.masked_fill(~cmask, -1e9)
            m = torch.softmax(logits, -1)
            prec = torch.exp((g[:, 2 + k:3 + k] + self.log_prec0).clamp(-3.0, 9.0)) * (n_act + 1.0)
            outs.append((m * prec).clamp(min=1e-3))
        return PolicyOut(outs[0], outs[1], cmask)


class Critic(nn.Module):
    def __init__(self, hidden: int = 64, n_global: int = 1, scale: float = 100.0):
        super().__init__()
        self.enc = mlp(FEAT_DIM, hidden, hidden)
        self.out = mlp(hidden + n_global, hidden, N_VALUE_HEADS)
        self.scale = scale           # returns are O(1/(1-gamma)); predict them /scale

    def forward(self, x, mask, glob):
        c = masked_mean(self.enc(x), mask)
        return self.out(torch.cat([c, glob], -1)) * self.scale


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
