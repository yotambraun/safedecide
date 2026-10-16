"""Temperature scaling: one scalar per question kind, fitted on a held-out split.

Temperature scaling never changes the argmax, so accuracy is untouched; it
only fixes over/under-confidence. Conformal calibration is applied on top.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from .schema import KINDS


def _nll_at(log_t: torch.Tensor, logits: list[np.ndarray], targets: list[np.ndarray]) -> torch.Tensor:
    t = log_t.exp()
    loss = 0.0
    for z, y in zip(logits, targets):
        zt = torch.as_tensor(z, dtype=torch.float64) / t
        loss = loss - (torch.as_tensor(y, dtype=torch.float64) * torch.log_softmax(zt, -1)).sum()
    return loss / len(logits)


def fit_temperature(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> float:
    if len(logits) == 0:
        return 1.0
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = _nll_at(log_t, list(logits), list(targets))
        loss.backward()
        return loss

    opt.step(closure)
    return float(np.clip(log_t.exp().item(), 0.05, 20.0))


def fit_temperatures(
    logits: Sequence[np.ndarray], targets: Sequence[np.ndarray], kinds: Sequence[str]
) -> dict[str, float]:
    """One temperature per kind (noul / choice / score)."""
    out = {}
    for kind in KINDS:
        idx = [i for i, k in enumerate(kinds) if k == kind]
        out[kind] = fit_temperature([logits[i] for i in idx], [targets[i] for i in idx]) if idx else 1.0
    return out


def softmax_np(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=float) / t
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()
