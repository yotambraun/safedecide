"""Core data types.

A *decision* is one typed question about one state, in the System One format
used by Jev, Clef, d3, Laya and Phocinae:

    state + question + a closed set of options  ->  a probability per option

Everything in this package (training, conformal calibration, benchmarking)
works on lists of :class:`Decision`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

KINDS = ("noul", "choice", "score")

# How boolean options are rendered to the encoder. "noul" is TypeSafe's name
# for a yes/no question.
NOUL_RENDER = {"false": "no", "true": "yes"}


@dataclass
class Decision:
    """One typed decision with a soft target distribution over its options."""

    id: str
    kind: str
    state: str
    question: str
    options: list[str]
    target: list[float] | None = None
    option_descriptions: list[str | None] | None = None
    source: str = ""
    domain: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")
        if len(self.options) < 2:
            raise ValueError(f"decision {self.id} has fewer than 2 options")
        if self.target is not None:
            if len(self.target) != len(self.options):
                raise ValueError(f"decision {self.id}: target/options length mismatch")
            total = float(sum(self.target))
            if total <= 0:
                raise ValueError(f"decision {self.id}: target sums to {total}")
            self.target = [float(t) / total for t in self.target]

    @property
    def num_options(self) -> int:
        return len(self.options)

    @property
    def label(self) -> int:
        """Index of the reference answer.

        The dataset's explicit gold label when it has one (it settles ties in the
        soft target), otherwise the argmax of the soft target.
        """
        gold = self.meta.get("gold_label")
        if gold is not None and gold in self.options:
            return self.options.index(gold)
        if self.target is None:
            raise ValueError(f"decision {self.id} has no target")
        return max(range(len(self.target)), key=self.target.__getitem__)

    @property
    def is_uniform(self) -> bool:
        """True when the target carries no preference at all (e.g. 0.5/0.5)."""
        if self.target is None:
            return False
        return max(self.target) - min(self.target) < 1e-6

    def option_text(self, i: int) -> str:
        """Human-readable text for option ``i`` as fed to the encoder."""
        opt = self.options[i]
        if self.kind == "noul":
            opt = NOUL_RENDER.get(opt.lower(), opt)
        desc = None
        if self.option_descriptions is not None:
            desc = self.option_descriptions[i]
        return f"{opt}: {desc}" if desc else opt


def decisions_from_systemone_request(request: dict[str, Any]) -> list[Decision]:
    """Convert a ``POST /v1/systemone`` body into :class:`Decision` objects.

    Accepts the TypeSafe/Clef/d3 shape::

        {"state": ..., "questions": {qid: {"type", "instructions", "criteria"}}}

    ``criteria`` is a dict (choice), a list (score) or optional (noul).
    """
    state = request["state"]
    if not isinstance(state, str):
        import json

        state = json.dumps(state, ensure_ascii=False)
    questions = request["questions"]
    out: list[Decision] = []
    for qid, q in questions.items():
        kind = q.get("type", "choice")
        text = q.get("instructions") or qid
        criteria = q.get("criteria")
        if kind == "noul":
            options = ["false", "true"]
            descs = None
            if isinstance(criteria, dict):
                descs = [criteria.get("false"), criteria.get("true")]
        elif kind == "score":
            if not isinstance(criteria, list) or len(criteria) < 2:
                raise ValueError(f"score question {qid} needs a list of >=2 criteria")
            options = [str(i) for i in range(len(criteria))]
            descs = list(criteria)
        else:
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"choice question {qid} needs a dict of >=2 criteria")
            options = list(criteria.keys())
            descs = list(criteria.values())
        out.append(
            Decision(
                id=qid,
                kind=kind,
                state=state,
                question=text,
                options=options,
                option_descriptions=descs,
            )
        )
    return out
