"""Temperature scaling: one scalar per question kind, fitted on a held-out split.

Temperature scaling never changes the argmax, so accuracy is untouched; it
only fixes over/under-confidence. An optional per-question bias (fitted on
held-out predictions, never on test) can move the argmax. Conformal
calibration is applied on top of both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import torch

from .schema import KINDS, Decision


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


# --------------------------------------------------------------------------- #
# Per-question bias
# --------------------------------------------------------------------------- #
def question_key(d: Decision) -> str:
    """Identifies a question by its kind, wording and options (descriptions included)."""
    text = "\x1f".join([d.kind, d.question, *(d.option_text(i) for i in range(d.num_options))])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def fit_bias(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray], l2: float = 0.01) -> np.ndarray:
    """One additive log-bias per option minimising soft cross-entropy, with a small L2 pull to zero."""
    z = torch.as_tensor(np.stack(logits), dtype=torch.float64)
    y = torch.as_tensor(np.stack(targets), dtype=torch.float64)
    b = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([b], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(y * torch.log_softmax(z + b, -1)).sum(-1).mean() + l2 * (b**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    out = b.detach().numpy()
    return out - out.mean()  # softmax ignores a shared shift


def fit_question_bias(
    logits: Sequence[np.ndarray],
    decisions: Sequence[Decision],
    temperatures: dict[str, float],
    min_count: int = 30,
    l2: float = 0.01,
) -> dict[str, list[float]]:
    """Per-question option biases on top of temperature scaling (labelled, held-out ``decisions``).

    A question asked often enough in ``decisions`` (``min_count``) gets one log-bias per
    option, fitted on its temperature-scaled logits; every other question is left alone.
    """
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(decisions):
        groups.setdefault(question_key(d), []).append(i)
    out = {}
    for key, idx in groups.items():
        if len(idx) < min_count:
            continue
        scaled = [np.asarray(logits[i], dtype=float) / temperatures.get(decisions[i].kind, 1.0) for i in idx]
        out[key] = [round(float(v), 6) for v in fit_bias(scaled, [np.asarray(decisions[i].target) for i in idx], l2)]
    return out


def calibrated_probs(
    logits: Sequence[np.ndarray],
    decisions: Sequence[Decision],
    temperatures: dict[str, float],
    question_bias: dict[str, list[float]] | None = None,
) -> list[np.ndarray]:
    """softmax(logits / T_kind + bias_question): the probabilities every other step sees."""
    out = []
    for z, d in zip(logits, decisions):
        z = np.asarray(z, dtype=float) / temperatures.get(d.kind, 1.0)
        b = question_bias.get(question_key(d)) if question_bias else None
        out.append(softmax_np(z + np.asarray(b) if b is not None and len(b) == len(z) else z))
    return out
