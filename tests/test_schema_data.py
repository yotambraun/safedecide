import json

import pytest

from safedecide.adapters import decision_to_systemone_question, parse_systemone_answer
from safedecide.data import describe, jev_row_to_decision, split_by_group, typed_decisions_case_to_decisions
from safedecide.schema import Decision, decisions_from_systemone_request, labelled_decisions_from_jsonl


def test_decision_normalises_and_labels():
    d = Decision("x", "choice", "s", "q", ["a", "b", "c"], [1, 3, 0])
    assert d.target == pytest.approx([0.25, 0.75, 0.0])
    assert d.label == 1
    assert not d.is_uniform


def test_noul_rendering():
    d = Decision("x", "noul", "s", "q", ["false", "true"], [0.5, 0.5])
    assert d.option_text(1) == "yes"
    assert d.is_uniform


def test_bad_kind_rejected():
    with pytest.raises(ValueError):
        Decision("x", "freeform", "s", "q", ["a", "b"])


def test_jev_row():
    row = {
        "id": "r1",
        "kind": "score",
        "options": ["0", "1", "2"],
        "target": [0.1, 0.2, 0.7],
        "state": "st",
        "question": "rate",
        "domain": "d",
        "family": "f",
        "source": "openjev_v2",
    }
    d = jev_row_to_decision(row)
    assert d.kind == "score" and d.label == 2 and d.source == "openjev_v2"


def test_typed_decisions_case():
    row = {
        "id": "case1",
        "workflow": "customer_service",
        "state": json.dumps({"ticket": "charged twice"}),
        "questions": json.dumps(
            {
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": "money", "tech": "bugs"},
                },
                "urgent": {"type": "noul", "instructions": "Urgent?"},
                "risk": {"type": "score", "instructions": "Risk?", "criteria": ["low", "mid", "high"]},
            }
        ),
        "gold": json.dumps(
            {
                "team": {"probabilities": {"billing": 0.9, "tech": 0.1}},
                "urgent": {"probabilities": {"false": 0.3, "true": 0.7}},
                "risk": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}},
            }
        ),
    }
    ds = typed_decisions_case_to_decisions(row)
    by = {d.id.split(":")[1]: d for d in ds}
    assert by["team"].label == 0
    assert by["urgent"].label == 1
    assert by["risk"].options == ["0", "1", "2"] and by["risk"].label == 2
    assert describe(ds)["n"] == 3


def test_split_by_group_has_no_leakage():
    ds = [Decision(f"{i}", "noul", f"state{i // 3}", "q", ["false", "true"], [0.2, 0.8]) for i in range(30)]
    a, b = split_by_group(ds, 0.5)
    assert not {d.state for d in a} & {d.state for d in b}
    assert len(a) + len(b) == 30


def test_systemone_roundtrip():
    req = {
        "state": "server down",
        "questions": {
            "team": {"type": "choice", "criteria": {"billing": None, "tech": "bugs"}},
            "urgent": {"type": "noul", "instructions": "urgent?"},
            "sev": {"type": "score", "criteria": ["low", "high"]},
        },
    }
    ds = decisions_from_systemone_request(req)
    assert [d.kind for d in ds] == ["choice", "noul", "score"]
    q = decision_to_systemone_question(ds[0])
    assert q["criteria"] == {"billing": None, "tech": "bugs"}


@pytest.mark.parametrize(
    "answer,expected",
    [
        ({"probabilities": {"yes": 0.8, "no": 0.2}}, [0.2, 0.8]),
        ({"noul": 0.7}, [0.3, 0.7]),
        ({"probabilities": {"false": 0.4, "true": 0.6}}, [0.4, 0.6]),
    ],
)
def test_parse_noul_variants(answer, expected):
    d = Decision("x", "noul", "s", "q", ["false", "true"])
    assert parse_systemone_answer(d, answer) == pytest.approx(expected)


def test_parse_choice_top_only():
    d = Decision("x", "choice", "s", "q", ["a", "b", "c", "d"])
    p = parse_systemone_answer(d, {"choice": "c", "confidence": 0.7})
    assert p.argmax() == 2 and p.sum() == pytest.approx(1.0)


def test_labelled_decisions_from_jsonl(tmp_path):
    rows = [
        {
            "state": "s1",
            "question": {
                "type": "choice",
                "instructions": "team?",
                "criteria": {"billing": "money", "technical": "bugs"},
            },
            "answer": "technical",
        },
        {"state": "s1", "question": {"type": "noul", "instructions": "urgent."}, "answer": "yes"},
        {
            "id": "sev",
            "state": "s2",
            "question": {"type": "score", "instructions": "severity?", "criteria": ["none", "low", "high"]},
            "answer": 2,
        },
    ]
    path = tmp_path / "mine.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    choice, noul, score = labelled_decisions_from_jsonl(str(path))
    assert choice.options == ["billing", "technical"] and choice.label == 1
    assert noul.kind == "noul" and noul.label == 1
    assert score.id == "sev" and score.options == ["0", "1", "2"] and score.label == 2
    assert choice.state == noul.state  # same case: stays on one side of every split

    path.write_text(json.dumps({**rows[0], "answer": "sales"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not one of"):
        labelled_decisions_from_jsonl(str(path))
