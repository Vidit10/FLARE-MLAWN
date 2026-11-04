"""Stages 1-2 of the report: per-user detector and randomized sanitizer.

One small autoencoder is applied to every user separately, so the same
weights work for any number of active users U (report Sec. III, F3-F6).

Per user u, with  s = the 5 scaled reports (attackable) and  c = the slice
one-hot (trusted, from the core network):

    z      = E_phi(s, c)                                  encoder   (5+3) -> 16 -> 8 -> 4
    e      = || s - D_0(z, c) ||^2                         detection error (D_0: (4+3) -> 8 -> 16 -> 5)
    tau    = conformal (1 - alpha) quantile of e on clean calibration reports
    w      = sigmoid((e - tau) / T)                        distrust weight, T = 0.1 tau
    Dbar   = (1/M) sum_m D_m(z + xi_m, c),  xi_m ~ N(0, sigma^2 I) fresh per call
    s~     = (1 - w) s + w Dbar                            sanitized report
    rho    = 1 - w                                         trust, fed to the actor

Difference from the report: the encoder and decoders also receive the slice
one-hot. It is trusted (not reported over E2) and the three slices have very
different traffic, so a clean eMBB report can look like an attacked URLLC
report and the other way round. Conditioning removes that ambiguity. It is a
switch (``cond_slice``).

Modes (``mode``):
* ``rand`` - the proposed randomized self-ensemble (M heads, latent noise).
* ``det``  - Baseline 3: one deterministic denoising decoder, no noise.
* ``none`` - pass-through (s~ = s, rho = 1).
Gates (``gate``): ``soft`` (the formula above), ``hard`` (w = 1[e > tau],
the draft's Eq. 22), ``always`` (w = 1).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..env.interface import N_REPORTED, N_SLICES


def _mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


@dataclass
class SanitizerConfig:
    latent: int = 4
    n_heads: int = 5                 # M
    sigma: float = 0.1               # latent noise of the ensemble
    alpha: float = 0.01              # target false-positive rate of the detector
    temp_frac: float = 0.1           # T = temp_frac * tau
    cond_slice: bool = True
    train_eps: float = 0.1           # uniform perturbation used to train the denoisers
    epochs_ae: int = 60
    epochs_den: int = 40
    batch: int = 512
    lr: float = 3e-3
    weight_decay: float = 1e-5
    seed: int = 0


class Sanitizer(nn.Module):
    def __init__(self, cfg: SanitizerConfig = SanitizerConfig()):
        super().__init__()
        self.cfg = cfg
        c = N_SLICES if cfg.cond_slice else 0
        L = cfg.latent
        self.enc = _mlp([N_REPORTED + c, 16, 8, L])
        self.dec0 = _mlp([L + c, 8, 16, N_REPORTED], nn.Sigmoid())            # detection decoder
        self.heads = nn.ModuleList(_mlp([L + c, 8, 16, N_REPORTED], nn.Sigmoid())
                                   for _ in range(cfg.n_heads))                 # randomized ensemble
        self.det_head = _mlp([L + c, 8, 16, N_REPORTED], nn.Sigmoid())         # Baseline 3
        self.register_buffer("tau", torch.tensor(float("nan")))

    # ------------------------------------------------------------ pieces
    def _c(self, s, onehot):
        return [onehot] if self.cfg.cond_slice else []

    def encode(self, s, onehot):
        return self.enc(torch.cat([s] + self._c(s, onehot), -1))

    def reconstruct(self, s, onehot):
        z = self.encode(s, onehot)
        return self.dec0(torch.cat([z] + self._c(s, onehot), -1))

    def error(self, s, onehot):
        """Detection score e = ||s - D_0(E(s))||^2 per user."""
        return ((s - self.reconstruct(s, onehot)) ** 2).sum(-1)

    def denoise(self, s, onehot, mode: str = "rand", xi: Optional[torch.Tensor] = None):
        z = self.encode(s, onehot)
        c = self._c(s, onehot)
        if mode == "det":
            return self.det_head(torch.cat([z] + c, -1))
        if xi is None:
            xi = torch.randn((self.cfg.n_heads,) + z.shape, device=z.device) * self.cfg.sigma
        outs = [h(torch.cat([z + xi[m]] + c, -1)) for m, h in enumerate(self.heads)]
        return torch.stack(outs, 0).mean(0)

    def weight(self, e, gate: str = "soft"):
        tau = self.tau
        if gate == "soft":
            return torch.sigmoid((e - tau) / (self.cfg.temp_frac * tau))
        if gate == "hard":
            return (e > tau).float()
        if gate == "always":
            return torch.ones_like(e)
        raise ValueError(gate)

    def forward(self, s, onehot, mask, mode: str = "rand", gate: str = "soft",
                xi: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """s: (..., U, 5) scaled reports; returns (s~, rho, e), zero for inactive users."""
        if mode == "none":
            return s, mask.float(), torch.zeros_like(s[..., 0])
        e = self.error(s, onehot)
        w = self.weight(e, gate)
        clean = self.denoise(s, onehot, mode, xi)
        st = (1 - w.unsqueeze(-1)) * s + w.unsqueeze(-1) * clean
        m = mask.unsqueeze(-1).float()
        return st * m, (1 - w) * mask.float(), e * mask.float()

    # ------------------------------------------------------------ save/load
    def save(self, path: str, extra: Optional[dict] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        meta = {"config": asdict(self.cfg), "tau": float(self.tau)}
        meta.update(extra or {})
        json.dump(meta, open(os.path.splitext(path)[0] + ".json", "w"), indent=2)

    @classmethod
    def load(cls, path: str) -> "Sanitizer":
        meta = json.load(open(os.path.splitext(path)[0] + ".json"))
        san = cls(SanitizerConfig(**meta["config"]))
        san.load_state_dict(torch.load(path, map_location="cpu"))
        san.eval()
        for p in san.parameters():
            p.requires_grad_(False)
        return san


# ================================================================ data
def collect_reports(env, controllers, n_episodes: int, users: Tuple[int, int] = (5, 40),
                    steps: int = 150, seed: int = 50_000) -> Tuple[np.ndarray, np.ndarray]:
    """Clean per-user reports (scaled) and slice ids from simulator runs.

    The controllers take turns, so the data covers the allocations they make
    (buffers and rates depend on the allocation). Seeds start at 50 000 so
    that they never overlap evaluation seeds (1000+) or training seeds.
    """
    rng = np.random.default_rng(seed)
    S, C = [], []
    for ep in range(n_episodes):
        ctrl = controllers[ep % len(controllers)]
        ctrl.reset()
        U = int(rng.integers(users[0], users[1] + 1))
        obs = env.reset(seed=seed + ep, n_users=U)
        for t in range(steps):
            if t >= 20:                         # skip the warm-up of the moving averages
                a = obs.active
                S.append(obs.reported_norm[a].astype(np.float32)); C.append(obs.slice_id[a])
            obs, _, done = env.step(ctrl.act(obs))
            if done:
                break
    return np.concatenate(S), np.concatenate(C)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Split-conformal (1 - alpha) quantile: P(e_new > tau) <= alpha for clean data."""
    n = len(scores)
    k = min(n, int(math.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])


def train_sanitizer(S: np.ndarray, C: np.ndarray, cfg: SanitizerConfig = SanitizerConfig(),
                    log=print) -> Tuple[Sanitizer, dict]:
    """70/15/15 split: train the autoencoder, calibrate tau, test; then the denoisers."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.permutation(len(S))
    n_tr, n_cal = int(0.70 * len(S)), int(0.15 * len(S))
    split = {"train": idx[:n_tr], "cal": idx[n_tr:n_tr + n_cal], "test": idx[n_tr + n_cal:]}
    s_all = torch.from_numpy(S.astype(np.float32))
    c_all = torch.from_numpy(np.eye(N_SLICES, dtype=np.float32)[C])
    san = Sanitizer(cfg)

    def batches(ids):
        p = torch.from_numpy(rng.permutation(ids))
        for i in range(0, len(p), cfg.batch):
            b = p[i:i + cfg.batch]
            yield s_all[b], c_all[b]

    # Stage 1: autoencoder (encoder + detection decoder), reconstruction loss
    params = list(san.enc.parameters()) + list(san.dec0.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs_ae)
    for ep in range(cfg.epochs_ae):
        for s, c in batches(split["train"]):
            loss = san.error(s, c).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    with torch.no_grad():
        e = {k: san.error(s_all[v], c_all[v]).numpy() for k, v in split.items()}
    tau = conformal_threshold(e["cal"], cfg.alpha)
    san.tau.fill_(tau)
    log(f"autoencoder: mean error train {e['train'].mean():.2e}, tau {tau:.3e}, "
        f"test false-positive rate {np.mean(e['test'] > tau):.4f} (target {cfg.alpha})")

    # Stage 2: denoising heads with the encoder frozen
    for p in san.enc.parameters():
        p.requires_grad_(False)
    params = list(san.heads.parameters()) + list(san.det_head.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs_den)
    for ep in range(cfg.epochs_den):
        for s, c in batches(split["train"]):
            d = (torch.rand_like(s) * 2 - 1) * cfg.train_eps
            sp = (s + d).clamp(0, 1)
            loss = (((s - san.denoise(sp, c, "rand")) ** 2).sum(-1).mean()
                    + ((s - san.denoise(sp, c, "det")) ** 2).sum(-1).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    san.eval()
    for p in san.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        st, ct = s_all[split["test"]], c_all[split["test"]]
        d = (torch.rand_like(st) * 2 - 1) * cfg.train_eps
        sp = (st + d).clamp(0, 1)
        den = {m: float(((st - san.denoise(sp, ct, m)) ** 2).sum(-1).mean()) for m in ("rand", "det")}
        noisy = float(((st - sp) ** 2).sum(-1).mean())
    log(f"denoisers: squared error on perturbed test reports {noisy:.4f} -> rand {den['rand']:.4f}, "
        f"det {den['det']:.4f}")
    stats = {"n_reports": int(len(S)), "split": {k: int(len(v)) for k, v in split.items()},
             "tau": tau, "fpr_test": float(np.mean(e["test"] > tau)),
             "err_mean": {k: float(v.mean()) for k, v in e.items()},
             "denoise_sq_err": {"perturbed_input": noisy, **den}}
    return san, stats
