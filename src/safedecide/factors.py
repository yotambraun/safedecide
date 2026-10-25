"""Auxiliary training questions from the typed-decisions ``factors``.

Every typed-decisions ``train`` case records the latent facts it was generated from:
for an invoice the issue (``duplicate_submission``, ``price_variance``, ...), whether it
is overdue, the payment terms. The benchmark does not give them to models as input, and
neither do we. During training only, each usable fact becomes one more typed question
about the same state with a one-hot target ("Fact about this case: issue." ->
``duplicate submission``), so the encoder learns to find those facts in the text.

At inference nothing changes: the model sees the state and the real question. Test-split
factors are never loaded (:func:`safedecide.data.load_typed_decisions` refuses).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .schema import Decision

MAX_CHOICES = 15  # more distinct values than this is an identifier or a number, not a category
MAX_SCORE_LEVELS = 10
MAX_VALUE_CHARS = 40  # longer strings are sentences copied from the state (task, issue_description)
FACTOR_SOURCE = "typed_decisions_factors"


@dataclass(frozen=True)
class FactorQuestion:
    workflow: str
    name: str
    kind: str  # noul (bool), choice (short strings) or score (small integers, ordered)
    values: tuple[Any, ...]  # raw values, in option order

    @property
    def question(self) -> str:
        return f"Fact about this case: {self.name.replace('_', ' ')}."

    @property
    def options(self) -> list[str]:
        if self.kind == "noul":
            return ["false", "true"]
        return [str(v).replace("_", " ") for v in self.values]

    def target(self, value: Any) -> list[float] | None:
        if self.kind == "noul":
            return [0.0, 1.0] if value else [1.0, 0.0]
        if value not in self.values:
            return None
        return [1.0 if v == value else 0.0 for v in self.values]


def case_id(d: Decision) -> str:
    return d.id.rsplit(":", 1)[0]


def _cases(decisions: Sequence[Decision]) -> dict[str, Decision]:
    """One decision per case (all five share the state and the factors)."""
    out: dict[str, Decision] = {}
    for d in decisions:
        if "factors" in d.meta:
            out.setdefault(case_id(d), d)
    return out


def _kind(values: list[Any]) -> str | None:
    if not all(isinstance(v, bool | int | str) for v in values):
        return None  # lists, floats, missing values
    distinct = set(values)
    if len(distinct) < 2:
        return None
    if all(isinstance(v, bool) for v in values):
        return "noul"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "score" if len(distinct) <= MAX_SCORE_LEVELS else None
    if all(isinstance(v, str) for v in values):
        if len(distinct) <= MAX_CHOICES and max(len(v) for v in distinct) <= MAX_VALUE_CHARS:
            return "choice"
    return None  # mixed types, too many distinct values, or sentences


def factor_vocabulary(decisions: Sequence[Decision]) -> dict[tuple[str, str], FactorQuestion]:
    """The usable factors per workflow, with their option sets, from ``decisions``' cases."""
    seen: dict[tuple[str, str], list[Any]] = {}
    for d in _cases(decisions).values():
        for name, value in d.meta["factors"].items():
            seen.setdefault((d.domain, name), []).append(value)
    vocab = {}
    for (workflow, name), values in sorted(seen.items()):
        kind = _kind(values)
        if kind is None:
            continue
        ordered = sorted(set(values)) if kind in ("score", "choice") else [False, True]
        vocab[(workflow, name)] = FactorQuestion(workflow, name, kind, tuple(ordered))
    return vocab


def factor_decisions(decisions: Sequence[Decision], vocab: dict[tuple[str, str], FactorQuestion]) -> list[Decision]:
    """One auxiliary decision per (case, usable factor) of the cases in ``decisions``."""
    out = []
    for cid, d in _cases(decisions).items():
        for name, value in d.meta["factors"].items():
            fq = vocab.get((d.domain, name))
            target = fq.target(value) if fq else None
            if target is None:
                continue
            out.append(
                Decision(
                    id=f"{cid}:factor:{name}",
                    kind=fq.kind,
                    state=d.state,
                    question=fq.question,
                    options=fq.options,
                    target=target,
                    source=FACTOR_SOURCE,
                    domain=d.domain,
                    meta={"question_id": f"factor:{name}", "gold_label": None},
                )
            )
    return out
