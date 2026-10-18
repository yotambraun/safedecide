"""Scoring on the LocalLLaMA/typed-decisions benchmark, with its own conventions.

Official tasks (``eval.yaml``): accuracy, ``kl_from_gold``, ``brier`` and ``ece`` on
config ``all``, split ``test`` (400 cases, 2,000 decisions).

* accuracy: top option against the gold ``label``
* kl_from_gold: KL(gold || prediction), mean over decisions
* brier: squared error summed over options against the soft gold, mean over decisions
* ece: the benchmark leaves the binning to submitters; we use 15 equal-width bins on
  the top-option confidence and say so in every report

KL and Brier reproduce the benchmark's Uniform reference row (0.444 / 0.238); see
``tests/test_typed_decisions.py``. Results go in the README's "fitted on train" table:
the released model was trained on the ``train`` split and never on ``test``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .metrics import brier, ece, kl_from_gold
from .schema import Decision

DATASET_ID = "LocalLLaMA/typed-decisions"
ECE_BINS = 15


def _scores(probs: Sequence[np.ndarray], decisions: Sequence[Decision]) -> dict[str, float]:
    targets = [np.asarray(d.target) for d in decisions]
    correct = np.array([int(np.argmax(p)) == d.label for p, d in zip(probs, decisions)])
    conf = np.array([float(np.max(p)) for p in probs])
    return {
        "n": len(decisions),
        "accuracy": float(correct.mean()),
        "kl_from_gold": kl_from_gold(probs, targets),
        "brier": brier(probs, targets),
        "ece": ece(conf, correct, ECE_BINS),
    }


def score_typed_decisions(probs: Sequence[np.ndarray], decisions: Sequence[Decision]) -> dict[str, Any]:
    """Official metrics overall and per question type, workflow and question."""

    def by(key) -> dict[str, dict[str, float]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, d in enumerate(decisions):
            groups[key(d)].append(i)
        return {k: _scores([probs[i] for i in v], [decisions[i] for i in v]) for k, v in sorted(groups.items())}

    return {
        "overall": _scores(probs, decisions),
        "by_kind": by(lambda d: d.kind),
        "by_workflow": by(lambda d: d.domain),
        "by_question": by(lambda d: f"{d.domain}/{d.meta.get('question_id', '?')}"),
        "ece_bins": ECE_BINS,
    }


def markdown_table(scores_by_model: dict[str, dict[str, Any]]) -> str:
    head = "| Model | Accuracy ↑ | KL from gold ↓ | Brier ↓ | ECE ↓ | noul | choice | score |\n"
    head += "|---|---|---|---|---|---|---|---|\n"
    rows = []
    for name, s in scores_by_model.items():
        o, k = s["overall"], s["by_kind"]
        per_kind = " | ".join(f"{k[t]['accuracy']:.3f}" if t in k else "–" for t in ("noul", "choice", "score"))
        rows.append(
            f"| {name} | {o['accuracy']:.3f} | {o['kl_from_gold']:.3f} | {o['brier']:.3f} | {o['ece']:.3f} | {per_kind} |"
        )
    return head + "\n".join(rows) + f"\n\nECE: {ECE_BINS} equal-width bins on top-option confidence.\n"


def eval_results_yaml(
    scores: dict[str, Any],
    source_url: str,
    user: str,
    date: str,
    notes: str,
    source_name: str = "Full test distributions",
) -> str:
    """``.eval_results/typed-decisions.yaml`` for the model repo (Hub eval-results format).

    No ``revision`` pin: entries pinned to a stale dataset revision drop off the
    auto-populated leaderboard.
    """
    import yaml

    o = scores["overall"]
    entries = [
        {
            "dataset": {"id": DATASET_ID, "task_id": task},
            "value": round(float(o[task]), 4),
            "date": date,
            "source": {"url": source_url, "name": source_name, "user": user},
            "notes": notes,
        }
        for task in ("accuracy", "kl_from_gold", "brier", "ece")
    ]
    return yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=100)


# --------------------------------------------------------------------------- #
# Reference rows, as in the benchmark README
# --------------------------------------------------------------------------- #
class UniformAdapter:
    """Same probability on every option."""

    name = "Uniform (reference)"

    def predict_proba(self, decisions):
        return [np.full(d.num_options, 1.0 / d.num_options) for d in decisions]


class PriorAdapter:
    """Mean gold distribution per (workflow, question) on ``train``; ignores the input."""

    name = "Prior (reference)"

    def __init__(self, train: Sequence[Decision]):
        acc: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        for d in train:
            acc[(d.domain, d.meta.get("question_id", d.question))].append(np.asarray(d.target, dtype=float))
        self.prior = {k: np.mean(v, axis=0) for k, v in acc.items()}

    def predict_proba(self, decisions):
        out = []
        for d in decisions:
            p = self.prior.get((d.domain, d.meta.get("question_id", d.question)))
            out.append(
                p / p.sum() if p is not None and len(p) == d.num_options else np.full(d.num_options, 1 / d.num_options)
            )
        return out
