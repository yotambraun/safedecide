"""Evaluation metrics on ragged predictions (each decision has its own K).

``probs`` is a list of 1-D numpy arrays, ``targets`` the matching soft
targets, ``labels`` the argmax reference answers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def top_confidence(probs: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([float(p.max()) for p in probs])


def predictions(probs: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([int(p.argmax()) for p in probs])


def accuracy(probs, labels) -> float:
    return float((predictions(probs) == np.asarray(labels)).mean())


def nll(probs, targets) -> float:
    vals = [-(t * np.log(np.clip(p, 1e-12, 1.0))).sum() for p, t in zip(probs, targets)]
    return float(np.mean(vals))


def kl_from_gold(probs, targets) -> float:
    """Mean KL(gold || prediction), the typed-decisions benchmark's distribution metric."""
    vals = []
    for p, t in zip(probs, targets):
        t = np.asarray(t, dtype=float)
        nz = t > 0
        vals.append(float((t[nz] * (np.log(t[nz]) - np.log(np.clip(np.asarray(p)[nz], 1e-12, 1.0)))).sum()))
    return float(np.mean(vals))


def brier(probs, targets) -> float:
    return float(np.mean([((p - t) ** 2).sum() for p, t in zip(probs, targets)]))


def rps(probs, targets) -> float:
    vals = [((np.cumsum(p) - np.cumsum(t)) ** 2).sum() / (len(p) - 1) for p, t in zip(probs, targets)]
    return float(np.mean(vals)) if vals else float("nan")


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def reliability_curve(confidence, correct, n_bins: int = 15):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            xs.append(float(confidence[in_bin].mean()))
            ys.append(float(correct[in_bin].mean()))
            ns.append(int(in_bin.sum()))
    return xs, ys, ns


def summarize(
    probs, targets, kinds: Sequence[str] | None = None, labels: Sequence[int] | None = None
) -> dict[str, float]:
    """All point metrics in one dict. ``labels`` default to the argmax of ``targets``."""
    if labels is None:
        labels = [int(np.argmax(t)) for t in targets]
    conf = top_confidence(probs)
    correct = predictions(probs) == np.asarray(labels)
    out = {
        "n": len(probs),
        "accuracy": accuracy(probs, labels),
        "nll": nll(probs, targets),
        "kl_from_gold": kl_from_gold(probs, targets),
        "brier": brier(probs, targets),
        "ece": ece(conf, correct),
        "mean_confidence": float(conf.mean()),
    }
    if kinds is not None:
        ordinal = [i for i, k in enumerate(kinds) if k == "score"]
        if ordinal:
            out["rps_score_questions"] = rps([probs[i] for i in ordinal], [targets[i] for i in ordinal])
        for kind in sorted(set(kinds)):
            idx = [i for i, k in enumerate(kinds) if k == kind]
            out[f"accuracy_{kind}"] = float(np.mean([correct[i] for i in idx]))
    return out


def bootstrap_ci(
    probs: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
    groups: Sequence[str] | None = None,
    labels: Sequence[int] | None = None,
) -> dict[str, list[float]]:
    """Percentile bootstrap intervals for accuracy, NLL, Brier and ECE.

    With ``groups`` (e.g. the state each decision was asked about) whole groups
    are resampled, so correlated questions about the same case don't make the
    interval look tighter than it is.
    """
    rng = np.random.default_rng(seed)
    n = len(probs)
    if groups is None:
        clusters = [np.array([i]) for i in range(n)]
    else:
        index: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            index.setdefault(g, []).append(i)
        clusters = [np.array(v) for v in index.values()]
    labels = np.array([int(np.argmax(t)) for t in targets] if labels is None else labels)
    conf = top_confidence(probs)
    correct = predictions(probs) == labels
    per_nll = np.array([-(t * np.log(np.clip(p, 1e-12, 1.0))).sum() for p, t in zip(probs, targets)])
    per_brier = np.array([((p - t) ** 2).sum() for p, t in zip(probs, targets)])
    draws: dict[str, list[float]] = {"accuracy": [], "nll": [], "brier": [], "ece": []}
    for _ in range(n_boot):
        i = np.concatenate([clusters[c] for c in rng.integers(0, len(clusters), len(clusters))])
        draws["accuracy"].append(float(correct[i].mean()))
        draws["nll"].append(float(per_nll[i].mean()))
        draws["brier"].append(float(per_brier[i].mean()))
        draws["ece"].append(ece(conf[i], correct[i]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in draws.items()}
