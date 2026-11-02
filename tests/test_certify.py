"""Certifying thresholds from confidences and correctness (pipeline.certify_rules), no model involved."""

import json
import warnings

import numpy as np
import pytest

from safedecide.pipeline import ClusteredCalibrationWarning, certify_rules
from safedecide.schema import case_of, labelled_decisions_from_jsonl


def _pool(n_cases, size, seed=0):
    rng = np.random.default_rng(seed)
    conf = rng.uniform(0.5, 1.0, n_cases * size)
    correct = rng.uniform(0, 1, n_cases * size) < conf
    return conf, correct, [f"case{i // size}" for i in range(n_cases * size)]


def test_repeated_cases_warn_and_point_to_the_case_unit():
    conf, correct, cases = _pool(100, 5)
    with pytest.warns(ClusteredCalibrationWarning, match="unit='case'"):
        rules, info = certify_rules(conf, correct, cases, alpha=0.1, delta=0.1)
    assert info == {"n_selective": 500, "n_cases": 100, "unit": "decision"}
    assert [r["alpha"] for r in rules] == [0.05, 0.1, 0.2]
    assert all("statement" in r and r["n_calibration"] == 500 for r in rules)


def test_one_per_case_certifies_on_one_decision_per_case_without_warning():
    conf, correct, cases = _pool(100, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rules, info = certify_rules(conf, correct, cases, alpha=0.1, delta=0.1, unit="one-per-case", seed=3)
    assert info == {"n_selective": 100, "n_cases": 100, "unit": "one-per-case"}
    assert all(r["n_calibration"] == 100 for r in rules)
    again, _ = certify_rules(conf, correct, cases, alpha=0.1, delta=0.1, unit="one-per-case", seed=3)
    assert again == rules  # seeded


def test_case_unit_keeps_every_decision_without_warning():
    conf, correct, cases = _pool(100, 5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        rules, info = certify_rules(conf, correct, cases, alpha=0.1, delta=0.1, unit="case")
    assert info == {"n_selective": 500, "n_cases": 100, "unit": "case"}
    assert all(r["n_calibration"] == 100 for r in rules)  # cases are the units
    assert all("every question counted" in r["statement"] for r in rules)


def test_unknown_unit_is_an_error():
    conf, correct, cases = _pool(10, 2)
    with pytest.raises(ValueError, match="unit must be one of"):
        certify_rules(conf, correct, cases, alpha=0.1, delta=0.1, unit="cases")


def test_one_decision_per_case_needs_no_warning():
    conf, correct, cases = _pool(300, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, info = certify_rules(conf, correct, cases, alpha=0.1, delta=0.1)
    assert info["n_cases"] == 300


def test_case_comes_from_the_case_field_or_the_state(tmp_path):
    rows = [
        {"state": "s1", "question": {"type": "noul"}, "answer": True},
        {"state": "s1", "question": {"type": "noul"}, "answer": False},
        {"state": "s2 v1", "case": "ticket-7", "question": {"type": "noul"}, "answer": True},
        {"state": "s2 v2", "case": "ticket-7", "question": {"type": "noul"}, "answer": True},
    ]
    path = tmp_path / "mine.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cases = [case_of(d) for d in labelled_decisions_from_jsonl(str(path))]
    assert cases[0] == cases[1] != cases[2]
    assert cases[2] == cases[3]
