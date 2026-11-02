import numpy as np
import pytest

from safedecide.conformal import (
    DEFAULT_GRID,
    apply_selective,
    automation_at_alpha,
    calibrate_selective,
    calibrate_selective_by_case,
    case_level_evalue,
    clopper_pearson_upper,
    conformal_set_stats,
    lac_qhat,
    one_per_case,
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
    assert 0 <= out["n_certified"] <= 5
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


def test_one_per_case_keeps_one_seeded_random_decision_per_case():
    cases = ["a", "a", "b", "c", "c", "c", "b"]
    idx = one_per_case(cases, seed=0)
    assert sorted(cases[i] for i in idx) == ["a", "b", "c"]
    assert list(idx) == sorted(idx)  # input order is kept
    assert np.array_equal(idx, one_per_case(cases, seed=0))  # reproducible
    picks = {tuple(one_per_case(cases, seed=s)) for s in range(50)}
    assert len(picks) > 1  # random within a case, not always the first
    assert list(one_per_case(["x", "y", "z"])) == [0, 1, 2]


def _clustered(n_cases, rng, size=5):
    """The extreme of clustering: every question about a case shares its confidence and its outcome."""
    conf = rng.uniform(0.5, 1.0, n_cases)
    correct = rng.uniform(0, 1, n_cases) < conf
    return np.repeat(conf, size), np.repeat(correct, size), np.repeat(np.arange(n_cases), size)


def test_one_per_case_stays_valid_on_clustered_data_where_all_rows_do_not():
    """Correlated questions make the all-rows certificate optimistic; one decision per case restores it."""
    alpha, delta, trials = 0.1, 0.1, 300
    rng = np.random.default_rng(11)
    c_pop, k_pop, _ = _clustered(200_000, rng, size=1)

    def violation_rate(pick_one):
        violations = 0
        for _ in range(trials):
            c, k, cases = _clustered(400, rng)
            idx = one_per_case(cases, seed=int(rng.integers(1 << 31))) if pick_one else slice(None)
            rule = calibrate_selective(c[idx], k[idx], alpha, delta)
            if np.isfinite(rule.threshold):
                violations += int((~k_pop[c_pop >= rule.threshold]).mean() > alpha)
        return violations / trials

    assert violation_rate(pick_one=True) <= delta + 0.03  # Monte Carlo slack
    assert violation_rate(pick_one=False) > delta + 0.03  # about 0.17 here


def test_automation_at_alpha_one_per_case_needs_groups():
    c, k = _synthetic(1000)
    groups = [str(i // 5) for i in range(1000)]
    for unit in ("case", "one-per-case"):
        out = automation_at_alpha(c, k, 0.1, 0.1, n_splits=5, groups=groups, unit=unit)
        assert 0 <= out["automation_rate_mean"] <= 1
        with pytest.raises(ValueError, match="groups"):
            automation_at_alpha(c, k, 0.1, 0.1, n_splits=5, unit=unit)
    with pytest.raises(ValueError, match="unit must be one of"):
        automation_at_alpha(c, k, 0.1, 0.1, n_splits=5, groups=groups, unit="row")


def test_case_level_evalue_is_the_mixture_of_fixed_bets():
    from safedecide.conformal import _BET_FRACTIONS

    automated, wrong, size = np.array([2, 0, 3, 1]), np.array([0, 0, 1, 1]), np.array([2, 4, 3, 2])
    alpha = 0.2
    x = (wrong - alpha * automated) / size
    bets = _BET_FRACTIONS / (1 - alpha)
    expected = np.mean([np.prod(1 - b * x) for b in bets])
    assert case_level_evalue(automated, wrong, size, alpha) == pytest.approx(expected, rel=1e-12)
    # the order of cases does not matter, and a case with nothing automated contributes nothing
    order = [3, 1, 0, 2]
    assert case_level_evalue(automated[order], wrong[order], size[order], alpha) == pytest.approx(expected, rel=1e-12)
    keep = automated > 0
    assert case_level_evalue(automated[keep], wrong[keep], size[keep], alpha) == pytest.approx(expected, rel=1e-12)
    assert case_level_evalue(np.zeros(3), np.zeros(3), np.ones(3), alpha) == 1.0
    # every bet keeps 1 - b * x > 0, even for a case whose every question is automated and wrong
    assert case_level_evalue(np.array([5]), np.array([5]), np.array([5]), alpha) > 0


def test_by_case_does_not_depend_on_how_the_rows_are_ordered_or_cases_named():
    c, k, cases = _clustered(300, np.random.default_rng(4), size=3)
    rule = calibrate_selective_by_case(c, k, cases, 0.2, 0.1)
    assert np.isfinite(rule.threshold) and rule.n_calibration == 300
    perm = np.random.default_rng(5).permutation(len(c))
    renamed = [f"ticket-{x * 7}" for x in cases[perm]]
    again = calibrate_selective_by_case(c[perm], k[perm], renamed, 0.2, 0.1)
    assert again == rule
    assert rule.calibration_error_ucb <= 0.2 + 1e-9
    assert rule.calibration_error_ucb >= (~k[c >= rule.threshold]).mean()


def test_by_case_stays_valid_on_clustered_data_where_all_rows_do_not():
    """Every question of a case shares its outcome: the worst case for clustering.

    At 20% the case-level test certifies in about 92% of draws and goes over budget in
    about 0.3%; the all-rows rule goes over in about 16% (delta is 10%).
    """
    alpha, delta, trials = 0.2, 0.1, 300
    rng = np.random.default_rng(12)
    c_pop, k_pop, _ = _clustered(200_000, rng, size=1)
    violations = {"case": 0, "decision": 0}
    certified = 0
    for _ in range(trials):
        c, k, cases = _clustered(400, rng)
        for unit, rule in (
            ("case", calibrate_selective_by_case(c, k, cases, alpha, delta)),
            ("decision", calibrate_selective(c, k, alpha, delta)),
        ):
            if np.isfinite(rule.threshold):
                certified += unit == "case"
                violations[unit] += int((~k_pop[c_pop >= rule.threshold]).mean() > alpha)
    assert certified / trials > 0.5  # the test has power here, so a low violation rate means something
    assert violations["case"] / trials <= delta + 0.03  # Monte Carlo slack
    assert violations["decision"] / trials > delta + 0.03


def test_by_case_stays_valid_when_error_is_not_monotone_in_confidence():
    """A band of high confidence with high error, and correlated errors inside a case.

    The e-value test is conservative, so even testing every threshold uncorrected rarely
    goes over budget here; this checks that the chains keep it valid while it certifies.
    """
    alpha, delta, trials = 0.2, 0.1, 100
    rng = np.random.default_rng(13)

    def draw(n_cases, size=4, sd=0.7):
        conf = np.repeat(rng.uniform(0.5, 1.0, n_cases), size) + rng.normal(0, 0.01, n_cases * size)
        p_wrong = np.where((conf > 0.93) & (conf < 0.97), 0.5, 0.8 * (1 - conf))
        case_effect = np.repeat(rng.normal(0, sd, n_cases), size)
        wrong = rng.uniform(0, 1, n_cases * size) < np.clip(p_wrong * np.exp(case_effect - sd**2 / 2), 0, 1)
        return conf, ~wrong, np.repeat(np.arange(n_cases), size)

    c_pop, k_pop, _ = draw(200_000)
    rules = [calibrate_selective_by_case(*draw(1500), alpha, delta) for _ in range(trials)]
    certified = [r for r in rules if np.isfinite(r.threshold)]
    assert len(certified) / trials > 0.9
    violations = sum(int((~k_pop[c_pop >= r.threshold]).mean() > alpha) for r in certified)
    assert violations / trials <= delta + 0.03


def test_by_case_keeps_more_than_one_per_case_when_questions_are_nearly_independent():
    rng = np.random.default_rng(14)
    n_cases, size = 400, 5
    conf = rng.uniform(0.5, 1.0, n_cases * size)
    correct = rng.uniform(0, 1, n_cases * size) < conf
    groups = [str(i // size) for i in range(n_cases * size)]
    runs = {
        u: automation_at_alpha(conf, correct, 0.2, 0.1, n_splits=20, groups=groups, unit=u)
        for u in ("case", "one-per-case")
    }
    assert runs["case"]["automation_rate_mean"] > runs["one-per-case"]["automation_rate_mean"] + 0.05
    assert runs["case"]["violation_rate"] <= 0.1 + 0.1  # 20 splits: loose check that nothing is badly off


def test_by_case_input_errors():
    with pytest.raises(ValueError, match="cases for"):
        calibrate_selective_by_case(np.ones(3), np.ones(3, bool), ["a", "b"], 0.1, 0.1)
    rule = calibrate_selective_by_case(np.array([]), np.array([], bool), [], 0.1, 0.1)
    assert not np.isfinite(rule.threshold) and rule.n_calibration == 0
