"""Distribution-free guarantees for decision models.

Two complementary tools. Both only need a held-out calibration set of
labelled decisions and assume the units they certify are exchangeable between
calibration and deployment. They make **no** assumption that the model's
probabilities are calibrated.

The units are the rows passed in. Several questions about one case are
correlated, so treating them as separate rows is an approximation that can make
the bound optimistic. :func:`calibrate_selective_by_case` certifies with whole
cases as the units and keeps every question; :func:`one_per_case` keeps one
random decision per case. Both are valid when whole cases are independent
(docs/METHOD.md, section 2).

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
# What a certificate counts as one unit: each decision (assumes decisions are exchangeable),
# each case with every decision kept, or one random decision of each case (the last two
# assume only that whole cases are).
UNITS = ("decision", "case", "one-per-case")


def check_unit(unit: str) -> str:
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {', '.join(UNITS)}; got {unit!r}")
    return unit


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


def one_per_case(cases: Sequence, seed: int = 0) -> np.ndarray:
    """Positions of one decision per case, drawn uniformly at random within each case (seeded).

    Certify on ``confidence[idx], correct[idx]`` to rely on exchangeable cases rather than
    exchangeable decisions. Positions come back in input order.
    """
    members: dict = {}
    for i, c in enumerate(cases):
        members.setdefault(c, []).append(i)
    rng = np.random.default_rng(seed)
    return np.sort(np.array([m[int(rng.integers(len(m)))] for m in members.values()], dtype=int))


# Bet fractions for the case-level test: midpoint quadrature of a Beta(1, 1/2) prior on
# b * (1 - alpha), which leans towards the largest admissible bet. Fixed, never fitted.
_BET_FRACTIONS = 0.999 * (1.0 - (1.0 - (np.arange(200) + 0.5) / 200) ** 2)


def _case_counts(confidence: np.ndarray, correct: np.ndarray, codes: np.ndarray, n_cases: int, lam: float):
    """Per case: decisions automated at ``lam``, how many of them are wrong, and the case size."""
    sel = confidence >= lam
    automated = np.bincount(codes, weights=sel, minlength=n_cases)
    wrong = np.bincount(codes, weights=sel & ~correct, minlength=n_cases)
    return automated, wrong


def case_level_evalue(automated: np.ndarray, wrong: np.ndarray, size: np.ndarray, alpha: float) -> float:
    """E-value against "the error among automated decisions exceeds ``alpha``", one term per case.

    For case *i* with ``size`` questions, ``automated`` of them automated and ``wrong`` of those
    wrong, ``x_i = (wrong_i - alpha * automated_i) / size_i`` lies in [-alpha, 1 - alpha], and
    the error rate of a uniformly random question of a random case, among automated ones, is
    at most alpha exactly when E[x] <= 0. For a fixed bet b in [0, 1/(1 - alpha)),
    prod_i (1 - b * x_i) has expectation prod_i (1 - b * E[x]) <= 1 under the null when cases
    are independent, whatever the dependence between questions of one case. The average over
    the fixed bets in ``_BET_FRACTIONS`` is therefore an e-value, and rejecting when it reaches
    1/level has type-I error <= level (Markov). The order of the cases does not matter.
    """
    x = (np.asarray(wrong, float) - alpha * np.asarray(automated, float)) / np.asarray(size, float)
    values, counts = np.unique(x[x != 0], return_counts=True)  # cases with nothing automated give 1
    if not len(values):
        return 1.0
    bets = _BET_FRACTIONS / (1.0 - alpha)
    log_wealth = (counts[None, :] * np.log1p(-bets[:, None] * values[None, :])).sum(axis=1)
    top = log_wealth.max()
    return float(math.exp(top) * np.mean(np.exp(log_wealth - top))) if top < 700 else math.inf


def _case_error_ucb(automated, wrong, size, level: float) -> float:
    """Smallest alpha' on a 0.001 grid above which the case-level test rejects at ``level``."""
    if automated.sum() == 0:
        return 1.0
    ucb = 1.0
    for a in np.arange(999, 0, -1) / 1000:
        if case_level_evalue(automated, wrong, size, a) < 1.0 / level:
            break
        ucb = float(a)
    return ucb


def calibrate_selective_by_case(
    confidence: np.ndarray,
    correct: np.ndarray,
    cases: Sequence,
    alpha: float = 0.05,
    delta: float = 0.1,
    grid: Sequence[float] = DEFAULT_GRID,
    starts: Sequence[float] | None = DEFAULT_STARTS,
) -> SelectiveRule:
    """Like :func:`calibrate_selective`, with whole cases as the units and every decision kept.

    ``cases`` gives the case of each decision. The guarantee needs cases to be independent
    draws, not decisions: questions about one case may be correlated in any way. It bounds
    the error of a uniformly random question of a new case, among automated ones, which is the
    per-decision error when every case asks the same number of questions (as
    :func:`one_per_case` does, but without discarding the other questions). Each threshold is
    tested with :func:`case_level_evalue`, in the same fixed-sequence chains.

    In the returned rule, ``n_calibration`` counts cases, ``calibration_automation`` is the
    share of decisions automated, and ``calibration_error_ucb`` is the case-level upper bound
    at the chain level.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(cases) != len(confidence):
        raise ValueError(f"got {len(cases)} cases for {len(confidence)} decisions")
    _, codes = np.unique(np.asarray([str(c) for c in cases]), return_inverse=True)
    n_cases = int(codes.max()) + 1 if len(codes) else 0
    size = np.bincount(codes, minlength=n_cases).astype(float)
    grid_desc = np.sort(np.asarray(grid, dtype=float))[::-1]
    if starts is None:
        chains, level = [[lam] for lam in grid_desc], delta / len(grid_desc)
    else:
        chains, level = [grid_desc[grid_desc <= s] for s in starts], delta / len(starts)
    best = SelectiveRule(math.inf, alpha, delta, n_cases, 0.0, 1.0)
    tested: dict[float, bool] = {}
    for chain in chains:
        for lam in chain:
            lam = float(lam)
            if lam not in tested:
                automated, wrong = _case_counts(confidence, correct, codes, n_cases, lam)
                tested[lam] = case_level_evalue(automated, wrong, size, alpha) >= 1.0 / level
            if not tested[lam]:
                break  # a chain stops at its first failure
            if lam < best.threshold:
                best = SelectiveRule(lam, alpha, delta, n_cases, float((confidence >= lam).mean()), 1.0)
    if math.isfinite(best.threshold):
        automated, wrong = _case_counts(confidence, correct, codes, n_cases, best.threshold)
        best.calibration_error_ucb = _case_error_ucb(automated, wrong, size, level)
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
    unit: str = "decision",
) -> dict:
    """Repeated random cal/test splits on one labelled pool.

    Returns the mean automation rate at a guaranteed error <= alpha, the
    realised test error, and how often the realised error exceeded alpha
    (should be <= delta if the guarantee holds).

    With ``groups`` (e.g. the case each question belongs to) a whole group goes
    to one side of the split, so correlated questions never straddle cal and test.
    ``unit`` is what the threshold is certified on (see :data:`UNITS`): every decision
    (``"decision"``), whole calibration groups with every decision kept (``"case"``,
    :func:`calibrate_selective_by_case`), or one random decision of each calibration
    group (``"one-per-case"``, :func:`one_per_case`). It is applied to every test decision.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    check_unit(unit)
    if unit != "decision" and groups is None:
        raise ValueError(f"unit={unit!r} needs groups (the case of each decision)")
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
        cal_groups = [members[j] for j in perm[:cut]]
        if unit == "one-per-case":
            cal = np.array([m[rng.integers(len(m))] for m in cal_groups], dtype=int)
        else:
            cal = np.concatenate(cal_groups)
        test = np.concatenate([members[j] for j in perm[cut:]])
        if unit == "case":
            cal_cases = np.concatenate([np.full(len(m), j) for j, m in enumerate(cal_groups)])
            rule = calibrate_selective_by_case(confidence[cal], correct[cal], cal_cases, alpha, delta, starts=starts)
        else:
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
        "n_certified": len(finite),
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
