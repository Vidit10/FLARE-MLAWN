"""Run one episode: simulator -> (attacker) -> controller -> simulator."""
from __future__ import annotations

from typing import Dict, Optional

from .attacks import Attacker
from .env.interface import RRAEnv
from .metrics import EpisodeLog


def run_episode(env: RRAEnv, controller, attacker: Optional[Attacker] = None,
                seed: int = 0, n_users: Optional[int] = None, steps: Optional[int] = None,
                warmup: int = 20) -> Dict[str, float]:
    obs = env.reset(seed=seed, n_users=n_users)
    controller.reset()
    if attacker is not None:
        attacker.reset(env, controller, seed=10_000 + seed)
    log = EpisodeLog(warmup=warmup)
    steps = steps or env.cfg.episode_len
    got, fair = [], []          # PRBs taken by attacker-controlled users vs equal share
    for t in range(steps):
        if attacker is not None:
            ctrl_mask = attacker.controlled_users(obs)
            seen = attacker.perturb(obs)
        else:
            seen = obs
        action = controller.act(seen)
        obs, info, done = env.step(action)
        log.add(info)
        if attacker is not None and attacker.k is not None and t >= warmup:
            got.append(info.n_prb[ctrl_mask].sum() / max(info.n_prb.sum(), 1))
            fair.append(ctrl_mask.sum() / max(info.active.sum(), 1))
        if done:
            break
    out = log.summary()
    if got:
        # >1 means the controlled users got more than their head-count share
        out["attacker_prb_ratio"] = float(sum(got) / max(sum(fair), 1e-9))
    return out
