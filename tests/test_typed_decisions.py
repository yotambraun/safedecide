import numpy as np
import pytest

from safedecide.schema import Decision
from safedecide.typed_decisions import (
    PriorAdapter,
    UniformAdapter,
    eval_results_yaml,
    markdown_table,
    score_typed_decisions,
)


def _decision(i, kind="choice", workflow="customer_service", qid="category", target=(0.7, 0.2, 0.1), gold=None):
    options = ["false", "true"] if kind == "noul" else [f"o{j}" for j in range(len(target))]
    return Decision(
        id=f"c{i}:{qid}",
        kind=kind,
        state=f"case {i}",
        question=qid,
        options=options,
        target=list(target),
        source="typed_decisions",
        domain=workflow,
        meta={"question_id": qid, "gold_label": gold},
    )


def test_uniform_reference_values():
    d = _decision(0, target=(0.5, 0.5, 0.0))
    s = score_typed_decisions(UniformAdapter().predict_proba([d]), [d])["overall"]
    # KL(gold || uniform) = sum g log(g / (1/3)) = log 3 - H(gold)
    assert s["kl_from_gold"] == pytest.approx(np.log(3) - np.log(2))
    assert s["brier"] == pytest.approx((0.5 - 1 / 3) ** 2 * 2 + (1 / 3) ** 2)


def test_explicit_gold_label_breaks_ties():
    d = _decision(0, kind="noul", qid="needs_review", target=(0.5, 0.5), gold="true")
    assert d.label == 1
    s = score_typed_decisions([np.array([0.3, 0.7])], [d])["overall"]
    assert s["accuracy"] == 1.0


def test_breakdowns_and_reports():
    decisions = [_decision(i, workflow=w) for i, w in enumerate(["customer_service", "invoice_processing"] * 5)]
    prior = PriorAdapter(decisions)
    probs = prior.predict_proba(decisions)
    s = score_typed_decisions(probs, decisions)
    assert set(s["by_workflow"]) == {"customer_service", "invoice_processing"}
    assert s["overall"]["accuracy"] == 1.0
    assert "| Prior |" in markdown_table({"Prior": s})
    y = eval_results_yaml(s, "https://example.com", "me", "2026-10-16", "fitted on train")
    assert "kl_from_gold" in y and "revision" not in y


@pytest.mark.network
def test_reproduces_benchmark_uniform_row():
    """The benchmark README lists Uniform at KL 0.444 / Brier 0.238 on the full test split."""
    pytest.importorskip("datasets")
    from safedecide.data import load_typed_decisions

    try:
        test = load_typed_decisions("test")
    except Exception as e:  # offline
        pytest.skip(f"dataset not reachable: {e}")
    assert len(test) == 2000
    s = score_typed_decisions(UniformAdapter().predict_proba(test), test)["overall"]
    assert s["kl_from_gold"] == pytest.approx(0.444, abs=0.001)
    assert s["brier"] == pytest.approx(0.238, abs=0.001)
