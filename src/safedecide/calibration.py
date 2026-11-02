"""Temperature scaling: one scalar per question kind, fitted on a held-out split.

Temperature scaling never changes the argmax, so accuracy is untouched; it
only fixes over/under-confidence. Both fits are small convex-ish problems solved
with SciPy's L-BFGS-B, so this module does not need torch. An optional per-question bias (fitted on
held-out predictions, never on test) can move the argmax. Conformal
calibration is applied on top of both.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from .schema import KINDS, Decision

T_MIN, T_MAX = 0.05, 20.0
_LBFGS = {"ftol": 1e-12, "gtol": 1e-9, "maxiter": 1000}


def _by_width(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Ragged rows stacked into one (logits, targets) pair of 2-D arrays per option count."""
    groups: dict[int, tuple[list, list]] = {}
    for z, y in zip(logits, targets):
        zs, ys = groups.setdefault(len(z), ([], []))
        zs.append(z)
        ys.append(y)
    return [(np.asarray(zs, dtype=float), np.asarray(ys, dtype=float)) for zs, ys in groups.values()]


def fit_temperature(logits: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> float:
    """The temperature minimising mean soft cross-entropy (NLL), within [0.05, 20]."""
    if len(logits) == 0:
        return 1.0
    groups, n = _by_width(logits, targets), len(logits)

    def nll_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        t = np.exp(x[0])  # optimise log T
        loss = grad = 0.0
        for z, y in groups:
            s = z / t
            logp = log_softmax(s, axis=1)
            loss -= float((y * logp).sum())
            grad -= float(((np.exp(logp) * y.sum(1, keepdims=True) - y) * s).sum())  # d loss / d log T
        return loss / n, np.array([grad / n])

    bounds = [(np.log(T_MIN), np.log(T_MAX))]
    res = minimize(nll_and_grad, np.zeros(1), jac=True, method="L-BFGS-B", bounds=bounds, options=_LBFGS)
    return float(np.clip(np.exp(res.x[0]), T_MIN, T_MAX))


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
    z = np.stack(logits).astype(float)
    y = np.stack(targets).astype(float)

    def loss_and_grad(b: np.ndarray) -> tuple[float, np.ndarray]:
        logp = log_softmax(z + b, axis=1)
        loss = -(y * logp).sum(1).mean() + l2 * float(b @ b)
        grad = (np.exp(logp) * y.sum(1, keepdims=True) - y).mean(0) + 2 * l2 * b
        return float(loss), grad

    out = minimize(loss_and_grad, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B", options=_LBFGS).x
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
