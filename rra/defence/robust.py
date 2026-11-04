"""Batched white-box attack used inside robust training (report F10, Algorithm).

During a rollout, the environments chosen for adversarial rollouts all get
their perturbation in one batched EOT-PGD call against the *current* actor,
routed through the frozen detector/sanitizer (sanitizer-in-the-loop). This is
the same attack as ``rra.attacks.GradientAttacker`` (A2, all users, all
features), written for a batch of environments.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from ..agents.policy import dir_kl
from ..attacks.gradient import predicted_utility_torch
from .controller import pipeline


def batch_attack(actor, san, mode: str, gate: str, x0: torch.Tensor, m: torch.Tensor,
                 eps: float, steps: int, eot: int, objectives: Sequence[str],
                 env=None, true_obs: Optional[List] = None,
                 gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """Return delta (B, U, 5) for the scaled reports x0[..., :5].

    objectives[b] is "kl" (report's L_atk) or "util" (one-step utility, needs
    env and the true observation of row b).
    """
    B, U, _ = x0.shape
    Q = max(1, eot)
    xq = x0.repeat_interleave(Q, 0)
    mq = m.repeat_interleave(Q, 0)
    s0 = xq[..., :5]
    emask = m.unsqueeze(-1).float()
    with torch.no_grad():
        clean, _ = pipeline(actor, san, s0, xq, mq, mode, gate)
    delta = (torch.rand((B, U, 5), generator=gen) * 2 - 1) * eps * emask
    alpha = 2.5 * eps / steps
    kl_rows = torch.tensor([o == "kl" for o in objectives]).repeat_interleave(Q)
    util_rows = [b for b, o in enumerate(objectives) if o == "util"]
    for p in actor.parameters():
        p.requires_grad_(False)
    try:
        with torch.enable_grad():
            for _ in range(steps):
                delta = (torch.clamp(x0[..., :5] + delta, 0, 1) - x0[..., :5]).detach().requires_grad_(True)
                out, _ = pipeline(actor, san, s0 + delta.repeat_interleave(Q, 0), xq, mq, mode, gate)
                loss = torch.zeros(())
                if kl_rows.any():
                    kl = (dir_kl(clean.alpha_prb, out.alpha_prb, out.cmask)
                          + dir_kl(clean.alpha_pow, out.alpha_pow, out.cmask))
                    loss = loss + kl[kl_rows].sum() / Q
                if util_rows:
                    prb, pw = out.mean()
                    for b in util_rows:
                        r = slice(b * Q, (b + 1) * Q)
                        loss = loss - predicted_utility_torch(env, true_obs[b], prb[r, :-1], pw[r, :-1]).mean()
                grad, = torch.autograd.grad(loss, delta)
                delta = delta.detach() + alpha * grad.sign()
                delta = torch.clamp(delta, -eps, eps) * emask
                delta = torch.clamp(x0[..., :5] + delta, 0, 1) - x0[..., :5]
    finally:
        for p in actor.parameters():
            p.requires_grad_(True)
    return delta.detach()


def mean_kl(ref, out) -> torch.Tensor:
    """KL between the deployed mean share vectors (PRB and power), per row.

    Used by SA-PPO instead of the Dirichlet KL: the Dirichlet KL grows with the
    precision, so the penalty could be lowered by flattening the policy without
    making the deployed (mean) action any more stable. See docs/week3_design.md.
    """
    tot = 0.0
    for a, b in ((ref.alpha_prb, out.alpha_prb), (ref.alpha_pow, out.alpha_pow)):
        p = a / a.sum(-1, keepdim=True)
        q = b / b.sum(-1, keepdim=True)
        t = p * (torch.log(p.clamp(min=1e-8)) - torch.log(q.clamp(min=1e-8)))
        tot = tot + (t * ref.cmask).sum(-1)
    return tot


def kl_pgd(actor, san, mode: str, gate: str, s: torch.Tensor, x0: torch.Tensor, m: torch.Tensor,
           ref, eps: float, steps: int) -> torch.Tensor:
    """Inner maximisation of SA-PPO: delta maximising mean_kl(ref, pi(San(s + delta)))."""
    emask = m.unsqueeze(-1).float()
    delta = (torch.rand_like(s) * 2 - 1) * eps * emask
    alpha = 2.5 * eps / steps
    ref = type(ref)(ref.alpha_prb.detach(), ref.alpha_pow.detach(), ref.cmask)
    for p in actor.parameters():
        p.requires_grad_(False)
    try:
        for _ in range(steps):
            delta = (torch.clamp(s + delta, 0, 1) - s).detach().requires_grad_(True)
            out, _ = pipeline(actor, san, s + delta, x0, m, mode, gate)
            loss = mean_kl(ref, out).sum()
            grad, = torch.autograd.grad(loss, delta)
            delta = delta.detach() + alpha * grad.sign()
            delta = torch.clamp(delta, -eps, eps) * emask
            delta = torch.clamp(s + delta, 0, 1) - s
    finally:
        for p in actor.parameters():
            p.requires_grad_(True)
    return delta.detach()
