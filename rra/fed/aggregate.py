"""Step E of the FLARE report: combining the cells' updates.

All aggregators take the cells' updates of the shared (per-user) layers,
Delta_c = theta_c(after local round) - theta(global), as flat vectors, and
return the update applied to the global shared layers plus each cell's weight.

* ``fedavg``  - plain average (McMahan et al., 2017).
* ``median``  - coordinate-wise median (Yin et al., 2018).
* ``fltrust`` - FLTrust (Cao et al., NDSS 2021): the server trains on a small
                clean root dataset each round; a cell's weight is
                ReLU(cos(Delta_c, Delta_root)), and every update is rescaled
                to the root update's norm.
* ``two_tier`` - ours (report Step E):
                reference  Delta~ = coordinate-wise median of the updates
                weight     a_c  proportional to  D_c * max(0, cos(Delta_c, Delta~))
                scaling    Delta_c <- Delta_c * min(1, |Delta~| / |Delta_c|)
                where D_c = rho_bar_c * exp(-beta * MMD_c) is the cell's data
                trust (Step D). Ablations: ``two_tier_no_data`` (D_c = 1) and
                ``two_tier_no_drift`` (D_c = rho_bar_c).
* ``local``   - no federation: every cell keeps its own update (reference).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

AGGREGATORS = ("local", "fedavg", "median", "fltrust", "two_tier", "two_tier_no_data", "two_tier_no_drift")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def aggregate(name: str, deltas: Sequence[torch.Tensor], data_trust: Optional[Sequence[float]] = None,
              rho_bar: Optional[Sequence[float]] = None,
              root: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, np.ndarray, Dict]:
    """Return (global update, weight per cell (sums to 1, or all 0), diagnostics)."""
    D = torch.stack(list(deltas))                         # (C, P)
    C = D.shape[0]
    info: Dict = {}
    if name == "fedavg":
        w = np.full(C, 1.0 / C)
        return D.mean(0), w, info
    if name == "median":
        return D.median(0).values, np.full(C, np.nan), info
    if name == "fltrust":
        if root is None:
            raise ValueError("fltrust needs the server's root update")
        cos = np.array([_cos(d, root) for d in D])
        a = np.maximum(cos, 0.0)
        scaled = torch.stack([d * (root.norm() / d.norm().clamp(min=1e-12)) for d in D])
        info["cos"] = cos
        if a.sum() <= 0:
            return torch.zeros_like(root), np.zeros(C), info
        w = a / a.sum()
        return (torch.from_numpy(w).float()[:, None] * scaled).sum(0), w, info
    if name.startswith("two_tier"):
        ref = D.median(0).values
        cos = np.array([_cos(d, ref) for d in D])
        if name == "two_tier_no_data":
            trust = np.ones(C)
        elif name == "two_tier_no_drift":
            trust = np.asarray(rho_bar, float)
        else:
            trust = np.asarray(data_trust, float)
        a = trust * np.maximum(cos, 0.0)
        rn = ref.norm()
        scaled = torch.stack([d * torch.clamp(rn / d.norm().clamp(min=1e-12), max=1.0) for d in D])
        info["cos"] = cos
        info["trust_used"] = trust
        if a.sum() <= 0:
            return ref, np.zeros(C), info
        w = a / a.sum()
        return (torch.from_numpy(w).float()[:, None] * scaled).sum(0), w, info
    raise ValueError(name)


# ------------------------------------------------------------------ data trust (Step D)
def mmd2(x: np.ndarray, y: np.ndarray, n: int = 400, seed: int = 0, bandwidth: Optional[float] = None) -> float:
    """Squared maximum mean discrepancy with a Gaussian kernel (Gretton et al., 2012).

    Biased (V-statistic) estimate on up to n rows of each sample. The kernel
    bandwidth is the median pairwise distance of the pooled sample unless given.
    """
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), size=min(n, len(x)), replace=False)]
    y = y[rng.choice(len(y), size=min(n, len(y)), replace=False)]
    z = np.concatenate([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    if bandwidth is None:
        bandwidth = float(np.sqrt(np.median(d2[d2 > 0]))) if (d2 > 0).any() else 1.0
    k = np.exp(-d2 / (2 * bandwidth ** 2))
    nx = len(x)
    return float(k[:nx, :nx].mean() + k[nx:, nx:].mean() - 2 * k[:nx, nx:].mean())


def data_trust(rho_bar: float, mmd: float, beta: float) -> float:
    """D_c = rho_bar_c * exp(-beta * MMD_c) (report Step D)."""
    return float(rho_bar * np.exp(-beta * mmd))
