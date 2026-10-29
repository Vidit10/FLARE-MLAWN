"""White-box gradient attacks on the DRL allocator (attacker tier T1).

Same hook, budget and masks as ``Attacker`` (hook.py); only the way the
perturbation is chosen differs.

Objectives
----------
* ``kl``   - the report's attack loss (Eq. latk without the sanitizer, which
             does not exist yet):  KL( pi(.|s) || pi(.|s+delta) ), summed over
             the PRB and the power Dirichlet.  Pushes the policy as far as
             possible from what it would do on the true reports.
* ``util`` - model-based: minimise the attacker's one-step estimate of the
             cell utility Phi for the allocation the policy would output
             (same estimate as ``predicted_utility`` in hook.py, written in
             torch so it can be differentiated).  The attacker sees the true
             reports before changing them, so it can compute this.

Methods
-------
* ``fgsm`` - one signed-gradient step of size 1.25 eps from a random start in
             [-eps, eps] ("FGSM with random start", Wong et al., ICLR 2020).
             The random start is needed because the KL loss has zero gradient
             at delta = 0.
* ``pgd``  - K signed-gradient steps of size 2.5 eps / K from a random start,
             projected back onto the budget, the mask and [0, 1] after every
             step (Madry et al., ICLR 2018; Algorithm 1 of the report with Q = 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..agents.policy import dir_kl, obs_features
from ..env.interface import Observation
from .hook import Attacker


@dataclass
class GradientAttacker(Attacker):
    strategy: str = "pgd"          # fgsm | pgd
    objective: str = "kl"          # kl | util
    steps: int = 10

    def __post_init__(self):
        auto_name = not self.name
        super().__post_init__()
        if auto_name:
            k = "all" if self.k is None else self.k
            self.name = f"{self.point}-{self.strategy}-{self.objective}-eps{self.eps:g}-k{k}"

    def perturb(self, obs: Observation) -> Observation:
        if self.eps <= 0:
            return obs
        if self.strategy not in ("fgsm", "pgd"):
            return super().perturb(obs)
        actor = self.controller.actor
        x0, m = obs_features(obs)
        x0, m = torch.from_numpy(x0)[None], torch.from_numpy(m)[None]
        s0 = x0[..., :5].clone()
        emask = torch.from_numpy(self.entry_mask(obs).astype(np.float32))[None]
        with torch.no_grad():
            clean = actor(x0, m)
        g = torch.Generator().manual_seed(int(self._rng.integers(1 << 31)))
        delta = (torch.rand(s0.shape, generator=g) * 2 - 1) * self.eps * emask
        if self.strategy == "fgsm":
            steps, alpha = 1, 1.25 * self.eps
        else:
            steps, alpha = self.steps, 2.5 * self.eps / self.steps
        with torch.enable_grad():              # callers may be inside no_grad()
            for _ in range(steps):
                delta = (torch.clamp(s0 + delta, 0, 1) - s0).detach().requires_grad_(True)
                x = torch.cat([s0 + delta, x0[..., 5:]], -1) * m.unsqueeze(-1)
                out = actor(x, m)
                if self.objective == "kl":
                    loss = (dir_kl(clean.alpha_prb, out.alpha_prb, out.cmask)
                            + dir_kl(clean.alpha_pow, out.alpha_pow, out.cmask)).sum()
                elif self.objective == "util":
                    prb, pw = out.mean()
                    loss = -predicted_utility_torch(self.env, obs, prb[0, :-1], pw[0, :-1])
                else:
                    raise ValueError(self.objective)
                grad, = torch.autograd.grad(loss, delta)
                delta = delta.detach() + alpha * grad.sign()
                delta = torch.clamp(delta, -self.eps, self.eps) * emask
                delta = torch.clamp(s0 + delta, 0, 1) - s0
        return self.apply_delta(obs, delta[0].numpy().astype(float))


def predicted_utility_torch(env, true_obs: Observation, prb: torch.Tensor,
                            pw: torch.Tensor) -> torch.Tensor:
    """Differentiable version of hook.predicted_utility (no integer rounding)."""
    r = env.cfg.radio
    rep = torch.from_numpy(true_obs.reported.astype(np.float32))
    act = torch.from_numpy(true_obs.active)
    n = env.n_prb * prb
    gain = torch.where(prb > 1e-6, pw / prb.clamp(min=1e-6), torch.zeros_like(prb))
    sinr = torch.pow(10.0, rep[:, 0] / 10.0) * gain
    se = torch.clamp(r.se_alpha * torch.log2(1.0 + sinr), max=r.se_max)
    need = rep[:, 1] + rep[:, 2] * env.t_c
    bits = torch.minimum(n * r.prb_bw_hz * se * env.t_c, need)
    tw = env.cfg.avg_rate_window
    rbar = (1 - 1 / tw) * rep[:, 4] + (bits / env.t_c) / tw
    u = torch.log1p(rbar / env.cfg.r_ref_bps)
    return u[act].mean()
