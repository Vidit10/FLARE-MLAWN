"""Deployed pipeline: reports -> detector/sanitizer -> PPO actor -> shares.

``DefendedController`` is the xApp of the report's Fig. 1 with all three
stages. It exposes ``policy_torch`` so that an adaptive white-box attacker can
differentiate through the whole pipeline (detector gate, randomized
sanitizer and actor), as the report's threat model requires.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ..agents.policy import Actor, PolicyOut, obs_features
from ..agents.ppo import PPOController
from ..env.interface import Action, Observation
from .sanitizer import Sanitizer


def pipeline(actor: Actor, san: Optional[Sanitizer], s, x0, m, mode: str = "rand",
             gate: str = "soft", xi=None):
    """Actor output for scaled reports s (B, U, 5); x0 holds the trusted features.

    Returns (PolicyOut, detection error e (B, U) or None).
    """
    if san is None or mode == "none":
        x = torch.cat([s, x0[..., 5:]], -1) * m.unsqueeze(-1)
        return actor(x, m), None
    onehot = x0[..., 8:11]
    st, rho, e = san(s, onehot, m, mode=mode, gate=gate, xi=xi)
    x = torch.cat([st, rho.unsqueeze(-1), x0[..., 6:]], -1) * m.unsqueeze(-1)
    return actor(x, m), e


class DefendedController(PPOController):
    """PPO actor behind the per-user detector and sanitizer (deterministic mean action)."""

    def __init__(self, actor: Actor, san: Optional[Sanitizer], mode: str = "rand",
                 gate: str = "soft", name: str = "Defended"):
        super().__init__(actor)
        self.san, self.mode, self.gate, self.name = san, mode, gate, name
        self.last_trust: Optional[np.ndarray] = None
        self.last_error: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> None:
        self.n_seen, self.n_flagged = 0, 0      # users seen / flagged (trust < 0.5) this episode

    @property
    def flag_rate(self) -> float:
        return self.n_flagged / max(self.n_seen, 1)

    def policy_torch(self, s, x0, m, xi=None):
        return pipeline(self.actor, self.san, s, x0, m, self.mode, self.gate, xi)

    @torch.no_grad()
    def act(self, obs: Observation) -> Action:
        x, m = obs_features(obs)
        x, m = torch.from_numpy(x)[None], torch.from_numpy(m)[None]
        out, e = self.policy_torch(x[..., :5], x, m)
        if self.san is not None and self.mode != "none":
            self.last_error = e[0].numpy()
            self.last_trust = (1 - self.san.weight(e, self.gate))[0].numpy() * obs.active
            self.n_seen += int(obs.active.sum())
            self.n_flagged += int(((self.last_trust < 0.5) & obs.active).sum())
        prb, pw = (t[0].numpy() for t in out.mean())
        return Action(prb_share=prb[:-1].astype(float), power_share=pw[:-1].astype(float))
