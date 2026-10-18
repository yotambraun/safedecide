import numpy as np

from safedecide.conformal import (
    DEFAULT_GRID,
    apply_selective,
    automation_at_alpha,
    calibrate_selective,
    clopper_pearson_upper,
    conformal_set_stats,
    lac_qhat,
)


def _synthetic(n, seed=0):
    """Confidence is informative: P(correct) = confidence (perfectly calibrated)."""
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.3, 1.0, n)
    correct = rng.uniform(0, 1, n) < conf
    return conf, correct


def test_clopper_pearson_bounds():
    assert clopper_pearson_upper(0, 0, 0.1) == 1.0
    ub = clopper_pearson_upper(5, 1000, 0.05)
    assert 0.005 < ub < 0.02
    assert clopper_pearson_upper(10, 10, 0.1) == 1.0


def test_selective_guarantee_holds_on_fresh_data():
    alpha, delta, violations, trials = 0.05, 0.1, 0, 200
    for s in range(trials):
        c_cal, k_cal = _synthetic(3000, seed=s)
        c_te, k_te = _synthetic(20000, seed=10_000 + s)
        rule = calibrate_selective(c_cal, k_cal, alpha, delta)
        res = apply_selective(rule, c_te, k_te)
        violations += int(res["selective_error"] > alpha)
    # promised: violation probability <= delta
    assert violations / trials <= delta


def test_guarantee_holds_when_error_is_not_monotone_in_confidence():
    """Fixed-sequence testing must stay valid where higher confidence is not always safer.

    The scenario is hard enough that skipping the multiple-testing correction
    breaks the guarantee, so passing here means something.
    """
    alpha, delta, trials = 0.05, 0.1, 300

    def draw(n, rng):
        conf = rng.uniform(0.5, 1.0, n)
        bump = np.where((conf > 0.93) & (conf < 0.97), 0.08, 0.0)  # a bad band near the top
        return conf, rng.uniform(0, 1, n) >= 0.4 * (1 - conf) + bump

    rng = np.random.default_rng(7)
    c_big, k_big = draw(400_000, rng)  # stands in for the population

    def violation_rate(**kwargs):
        violations = 0
        for _ in range(trials):
            c, k = draw(3000, rng)
            rule = calibrate_selective(c, k, alpha, **kwargs)
            if np.isfinite(rule.threshold):
                violations += int((~k_big[c_big >= rule.threshold]).mean() > alpha)
        return violations / trials

    assert violation_rate(delta=delta) <= delta + 0.03  # Monte Carlo slack
    # every threshold tested at level delta with no correction: invalid here
    assert violation_rate(delta=delta * len(DEFAULT_GRID), starts=None) > delta + 0.03


def test_fixed_sequence_is_at_least_as_strong_as_bonferroni_here():
    c, k = _synthetic(600, seed=3)
    fixed = calibrate_selective(c, k, 0.05, 0.1)
    bonferroni = calibrate_selective(c, k, 0.05, 0.1, starts=None)
    assert fixed.calibration_automation >= bonferroni.calibration_automation


def test_selective_automates_something_when_model_is_good():
    c, k = _synthetic(5000)
    rule = calibrate_selective(c, k, 0.05, 0.1)
    assert np.isfinite(rule.threshold)
    assert rule.calibration_automation > 0.05


def test_selective_refuses_when_model_is_random():
    rng = np.random.default_rng(1)
    c = rng.uniform(0, 1, 2000)
    k = rng.uniform(0, 1, 2000) < 0.5  # confidence carries no signal
    rule = calibrate_selective(c, k, 0.05, 0.1)
    assert not np.isfinite(rule.threshold)
    assert apply_selective(rule, c, k)["automation_rate"] == 0.0


def test_automation_at_alpha_keys():
    c, k = _synthetic(4000)
    out = automation_at_alpha(c, k, 0.05, 0.1, n_splits=5)
    assert 0 <= out["automation_rate_mean"] <= 1
    assert out["violation_rate"] <= 1


def test_conformal_set_coverage():
    rng = np.random.default_rng(0)
    probs, labels = [], []
    for _ in range(6000):
        logits = rng.normal(size=4) * 2
        p = np.exp(logits) / np.exp(logits).sum()
        probs.append(p)
        labels.append(int(rng.choice(4, p=p)))  # labels drawn from p: calibrated model
    stats = conformal_set_stats(probs[:3000], labels[:3000], probs[3000:], labels[3000:], alpha=0.1)
    assert stats["coverage"] >= 0.88
    assert stats["automated_and_wrong_rate"] <= 0.1 + 0.02


def test_lac_qhat_saturates_for_tiny_sets():
    assert lac_qhat(np.array([0.9, 0.8]), alpha=0.01) == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    from safedecide.metrics import bootstrap_ci, summarize

    rng = np.random.default_rng(0)
    probs, targets = [], []
    for _ in range(800):
        p = rng.dirichlet([1, 1, 1])
        probs.append(p)
        targets.append(np.eye(3)[rng.choice(3, p=p)])
    ci = bootstrap_ci(probs, targets, n_boot=300)
    point = summarize(probs, targets)
    for k in ("accuracy", "nll", "brier"):
        assert ci[k][0] <= point[k] <= ci[k][1]
