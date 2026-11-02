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

# The answers a labelled ``noul`` line may give, compared case-insensitively. JSON
# ``true``/``false`` and the integers 1/0 are accepted too; anything else is an error.
NOUL_ANSWERS = {"true": "true", "yes": "true", "1": "true", "false": "false", "no": "false", "0": "false"}


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


def noul_answer(answer: Any) -> str:
    """``"true"`` or ``"false"`` for a labelled yes/no answer; raises ``ValueError`` on anything else.

    Accepted: JSON booleans, the integers 1 and 0, and the strings true/false, yes/no and 1/0
    in any case. A typo, ``null`` or ``"maybe"`` is refused rather than read as one side.
    """
    if isinstance(answer, bool):
        return "true" if answer else "false"
    if isinstance(answer, int) and answer in (0, 1):
        return "true" if answer else "false"
    if isinstance(answer, str) and answer.strip().lower() in NOUL_ANSWERS:
        return NOUL_ANSWERS[answer.strip().lower()]
    raise ValueError(f"noul answer {answer!r} is not one of true/false, yes/no, 1/0")


def labelled_decisions_from_jsonl(path: str) -> list[Decision]:
    """Read labelled decisions for calibration: one JSON object per line.

    Each line holds one question in the ``/v1/systemone`` shape plus the right answer::

        {"state": ..., "question": {"type", "instructions", "criteria"}, "answer": ...}

    ``answer`` is an option name for ``choice``, the level index for ``score``, and for
    ``noul`` one of the spellings :func:`noul_answer` accepts (true/false, yes/no, 1/0, in
    any case, or a JSON boolean). Anything else raises ``ValueError`` naming the line.

    Questions sharing a ``state`` stay on the same side of every calibration split.
    """
    import json

    out: list[Decision] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if "answer" not in row:
                    raise ValueError("missing 'answer'")
                qid = str(row.get("id", f"line{n}"))
                (d,) = decisions_from_systemone_request({"state": row["state"], "questions": {qid: row["question"]}})
                answer = noul_answer(row["answer"]) if d.kind == "noul" else str(row["answer"])
                if answer not in d.options:
                    raise ValueError(f"answer {row['answer']!r} is not one of {d.options}")
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                detail = f"missing {e}" if isinstance(e, KeyError) else str(e)
                raise ValueError(f"{path}:{n}: {detail}") from e
            d.target = [1.0 if o == answer else 0.0 for o in d.options]
            d.source = "jsonl"
            out.append(d)
    if not out:
        raise ValueError(f"{path} holds no decisions")
    return out
