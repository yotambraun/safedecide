"""Distribution-free guarantees for decision models.

Two complementary tools. Both only need a held-out calibration set of
labelled decisions and assume calibration and deployment data are
exchangeable (i.i.d. from the same distribution). They make **no**
assumption that the model's probabilities are calibrated.

1. Selective risk control ("automate or escalate")
   Find a confidence threshold lambda such that, with probability >= 1-delta
   over the draw of the calibration set,

       P(model is wrong | confidence >= lambda) <= alpha.

   Method: Learn-then-Test (Angelopoulos et al. 2021) with an exact
   Clopper-Pearson p-value per threshold on a fixed grid, tested by
   fixed-sequence testing from a few fixed starting thresholds: each start
   walks down the grid at level delta / n_starts and stops at its first
   failure. Pick the lowest certified lambda = highest automation rate.
   The headline metric is the automation rate at that lambda: "share of
   decisions handled with a guaranteed error <= alpha".

2. Split conformal prediction sets (LAC / THR)
   Sets C(x) with P(y in C(x)) >= 1-alpha. Automating only singleton sets
   guarantees P(automated AND wrong) <= alpha as a share of *all* traffic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta as beta_dist

DEFAULT_GRID = np.round(np.concatenate([np.linspace(0.0, 0.9, 46), np.linspace(0.902, 0.999, 50)]), 4)
# Where the fixed-sequence chains start (high to low). Fixed in advance, never chosen from data.
DEFAULT_STARTS = (0.99, 0.95, 0.90, 0.80)


# --------------------------------------------------------------------------- #
# Selective risk control
# --------------------------------------------------------------------------- #
def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided (1-delta) upper confidence bound on a binomial rate."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta_dist.ppf(1.0 - delta, k + 1, n - k))


@dataclass
class SelectiveRule:
    threshold: float  # automate when confidence >= threshold (inf = never)
    alpha: float
    delta: float
    n_calibration: int
    calibration_automation: float
    calibration_error_ucb: float

    def to_dict(self) -> dict:
        d = asdict(self)
        if math.isinf(d["threshold"]):
            d["threshold"] = None
        return d


def calibrate_selective(
    confidence: np.ndarray,
    correct: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.1,
    grid: Sequence[float] = DEFAULT_GRID,
    starts: Sequence[float] | None = DEFAULT_STARTS,
) -> SelectiveRule:
    """Lowest threshold whose selective error is certified <= alpha with probability >= 1 - delta.

    ``starts`` gives the fixed-sequence chains; ``starts=None`` falls back to
    Bonferroni over the whole grid (valid too, but far more conservative).
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    grid_desc = np.sort(np.asarray(grid, dtype=float))[::-1]
    if starts is None:
        chains, level = [[lam] for lam in grid_desc], delta / len(grid_desc)
    else:
        chains, level = [grid_desc[grid_desc <= s] for s in starts], delta / len(starts)
    best = SelectiveRule(math.inf, alpha, delta, len(confidence), 0.0, 1.0)
    for chain in chains:
        for lam in chain:
            sel = confidence >= lam
            n = int(sel.sum())
            k = int((~correct[sel]).sum())
            ucb = clopper_pearson_upper(k, n, level)
            if ucb > alpha:
                break  # a chain stops at its first failure
            if lam < best.threshold:
                best = SelectiveRule(float(lam), alpha, delta, len(confidence), n / len(confidence), ucb)
    return best


def apply_selective(rule: SelectiveRule, confidence: np.ndarray, correct: np.ndarray) -> dict:
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    sel = confidence >= rule.threshold
    n_auto = int(sel.sum())
    err = float((~correct[sel]).mean()) if n_auto else 0.0
    return {
        "automation_rate": n_auto / max(1, len(confidence)),
        "selective_error": err,
        "n_automated": n_auto,
        "n": len(confidence),
        "within_alpha": err <= rule.alpha,
    }


def automation_at_alpha(
    confidence: np.ndarray,
    correct: np.ndarray,
    alpha: float = 0.05,
    delta: float = 0.1,
    cal_fraction: float = 0.5,
    n_splits: int = 100,
    seed: int = 0,
    groups: Sequence[str] | None = None,
    starts: Sequence[float] | None = DEFAULT_STARTS,
) -> dict:
    """Repeated random cal/test splits on one labelled pool.

    Returns the mean automation rate at a guaranteed error <= alpha, the
    realised test error, and how often the realised error exceeded alpha
    (should be <= delta if the guarantee holds).

    With ``groups`` (e.g. the case each question belongs to) a whole group goes
    to one side of the split, so correlated questions never straddle cal and test.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    rng = np.random.default_rng(seed)
    if groups is None:
        members = [np.array([i]) for i in range(len(confidence))]
    else:
        index: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            index.setdefault(g, []).append(i)
        members = [np.array(v) for v in index.values()]
    autos, errs, violations, thresholds = [], [], 0, []
    for _ in range(n_splits):
        perm = rng.permutation(len(members))
        cut = int(len(members) * cal_fraction)
        cal = np.concatenate([members[j] for j in perm[:cut]])
        test = np.concatenate([members[j] for j in perm[cut:]])
        rule = calibrate_selective(confidence[cal], correct[cal], alpha, delta, starts=starts)
        res = apply_selective(rule, confidence[test], correct[test])
        autos.append(res["automation_rate"])
        errs.append(res["selective_error"])
        thresholds.append(rule.threshold)
        violations += int(not res["within_alpha"])
    finite = [t for t in thresholds if math.isfinite(t)]
    return {
        "alpha": alpha,
        "delta": delta,
        "automation_rate_mean": float(np.mean(autos)),
        "automation_rate_std": float(np.std(autos)),
        "selective_error_mean": float(np.mean(errs)),
        "violation_rate": violations / n_splits,
        "threshold_median": float(np.median(finite)) if finite else None,
        "n_splits": n_splits,
        "per_split": {"automation_rate": autos, "selective_error": errs},
    }


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray, points: int = 100):
    """Empirical selective error vs coverage (no guarantee, for plotting)."""
    order = np.argsort(-np.asarray(confidence))
    wrong = ~np.asarray(correct, dtype=bool)[order]
    cum_err = np.cumsum(wrong) / np.arange(1, len(wrong) + 1)
    idx = np.unique(np.linspace(0, len(wrong) - 1, points).astype(int))
    coverage = (idx + 1) / len(wrong)
    return coverage.tolist(), cum_err[idx].tolist()


# --------------------------------------------------------------------------- #
# Split conformal prediction sets
# --------------------------------------------------------------------------- #
def lac_qhat(true_class_probs: np.ndarray, alpha: float) -> float:
    """Threshold on the non-conformity score 1 - p_y."""
    scores = 1.0 - np.asarray(true_class_probs, dtype=float)
    n = len(scores)
    level = math.ceil((n + 1) * (1 - alpha)) / n
    if level > 1.0:
        return 1.0
    return float(np.quantile(scores, level, method="higher"))


def prediction_set(p: np.ndarray, qhat: float) -> list[int]:
    return [i for i, pi in enumerate(p) if 1.0 - pi <= qhat]


def conformal_set_stats(
    cal_probs: Sequence[np.ndarray],
    cal_labels: Sequence[int],
    test_probs: Sequence[np.ndarray],
    test_labels: Sequence[int],
    alpha: float = 0.05,
) -> dict:
    qhat = lac_qhat(np.array([p[y] for p, y in zip(cal_probs, cal_labels)]), alpha)
    sets = [prediction_set(p, qhat) for p in test_probs]
    sizes = np.array([len(s) for s in sets])
    covered = np.array([y in s for s, y in zip(sets, test_labels)])
    singleton = sizes == 1
    auto_wrong = singleton & ~covered
    return {
        "alpha": alpha,
        "qhat": qhat,
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "singleton_rate": float(singleton.mean()),
        "empty_rate": float((sizes == 0).mean()),
        "automated_and_wrong_rate": float(auto_wrong.mean()),
    }
