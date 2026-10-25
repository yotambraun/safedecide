"""Dataset loaders. Every loader returns ``list[Decision]``.

Sources and their licensing (checked 2026-10-15):

* ``SargeDev/jev-distill-corpus-v3`` (Apache-2.0 card), three streams:
    - ``openjev_v2`` (94.8k rows): from ZefanCai/Open-Jev, CC0 at source.  CLEAN.
    - ``yuri_v1`` (148k rows): memory-relevance pairs labelled by a 32B teacher
      over open QA datasets; many rows are uniform 0.5/0.5 and are dropped.
    - ``yuri_v3`` (498k rows): labels distilled from TypeSafe Jev 1.13. Other
      projects trained on Jev outputs ship non-commercial. RESEARCH ONLY.
* ``LocalLLaMA/typed-decisions`` (Apache-2.0): 4 business workflows,
  5 questions per case, soft gold labels.

Profiles:
    clean    -> openjev_v2 + typed-decisions train  (release under Apache-2.0)
    research -> all three streams + typed-decisions train (release non-commercial)
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from .schema import Decision

JEV_DISTILL = "SargeDev/jev-distill-corpus-v3"
TYPED_DECISIONS = "LocalLLaMA/typed-decisions"

PROFILES: dict[str, tuple[str, ...]] = {
    "clean": ("openjev_v2",),
    "research": ("openjev_v2", "yuri_v1", "yuri_v3"),
}

PROFILE_LICENSE = {"clean": "apache-2.0", "research": "other"}

TD_SPLIT_SEED = 0


# --------------------------------------------------------------------------- #
# jev-distill-corpus-v3
# --------------------------------------------------------------------------- #
def jev_row_to_decision(row: dict[str, Any]) -> Decision:
    """Convert one jev-distill row (schema documented on the dataset card)."""
    return Decision(
        id=str(row["id"]),
        kind=row["kind"],
        state=str(row["state"]),
        question=str(row["question"]),
        options=[str(o) for o in row["options"]],
        target=[float(t) for t in row["target"]],
        source=row.get("source", ""),
        domain=row.get("domain", ""),
        meta={"family": row.get("family", "")},
    )


def load_jev_distill(
    split: str,
    sources: Sequence[str] = PROFILES["clean"],
    max_rows: int | None = None,
    drop_uniform: bool = True,
    seed: int = 0,
    max_options: int = 16,
) -> list[Decision]:
    """Load one split of jev-distill-corpus-v3, filtered by stream.

    Splits: train, validation, calibration, test, ood, test_set_30k.
    Note: ``test_set_30k`` was carved out of train/validation/calibration, so
    train on ``train`` and evaluate on ``test_set_30k`` / ``ood``.
    """
    from datasets import load_dataset

    ds = load_dataset(JEV_DISTILL, split=split)
    allowed = set(sources)
    ds = ds.filter(lambda src: src in allowed, input_columns="source")
    ds = ds.shuffle(seed=seed)
    out: list[Decision] = []
    for row in ds:  # lazy: only the rows we keep are turned into Python objects
        if len(row["options"]) > max_options:
            continue
        try:
            d = jev_row_to_decision(row)
        except ValueError:
            continue
        if drop_uniform and d.is_uniform:
            continue
        out.append(d)
        if max_rows is not None and len(out) >= max_rows:
            break
    return out


# --------------------------------------------------------------------------- #
# LocalLLaMA/typed-decisions
# --------------------------------------------------------------------------- #
def _maybe_json(x: Any) -> Any:
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x


def typed_decisions_case_to_decisions(row: dict[str, Any], with_factors: bool = False) -> list[Decision]:
    """One typed-decisions case (state + 5 questions) -> 5 decisions.

    ``with_factors`` keeps the case's latent ``factors`` in each decision's ``meta``,
    for auxiliary training questions (:mod:`safedecide.factors`). They are never model input.
    """
    state = _maybe_json(row["state"])
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    questions = _maybe_json(row["questions"])
    gold = _maybe_json(row["gold"])
    out: list[Decision] = []
    for qid, q in questions.items():
        kind = q.get("type", "choice")
        g = gold.get(qid, {}) if isinstance(gold, dict) else {}
        probs: dict[str, float] = {str(k): float(v) for k, v in g.get("probabilities", {}).items()}
        criteria = q.get("criteria")
        if kind == "noul":
            options = ["false", "true"]
            descs = [criteria.get("false"), criteria.get("true")] if isinstance(criteria, dict) else None
        elif kind == "score":
            if isinstance(criteria, list):
                options = [str(i) for i in range(len(criteria))]
                descs = [str(c) for c in criteria]
            else:
                options = sorted(probs, key=lambda s: int(s))
                descs = None
        else:
            if isinstance(criteria, dict):
                options = list(criteria.keys())
                descs = [criteria[k] for k in options]
            else:
                options = list(probs.keys())
                descs = None
        target = [probs.get(o, 0.0) for o in options] if probs else None
        if target is not None and sum(target) <= 0:
            target = None
        out.append(
            Decision(
                id=f"{row['id']}:{qid}",
                kind=kind,
                state=state_text,
                question=q.get("instructions") or qid,
                options=options,
                option_descriptions=descs,
                target=target,
                source="typed_decisions",
                domain=row.get("workflow", ""),
                meta={"question_id": qid, "gold_label": None if g.get("label") is None else str(g["label"])},
            )
        )
        if with_factors:
            out[-1].meta["factors"] = _maybe_json(row.get("factors")) or {}
    return out


def load_typed_decisions(split: str = "test", config: str = "all", with_factors: bool = False) -> list[Decision]:
    """``with_factors`` is refused for ``test``: test factors are never loaded, not even unused."""
    from datasets import load_dataset

    if with_factors and split != "train":
        raise ValueError("factors are only loaded for the train split")
    ds = load_dataset(TYPED_DECISIONS, config, split=split)
    out: list[Decision] = []
    for row in ds:
        out.extend(d for d in typed_decisions_case_to_decisions(row, with_factors) if d.target is not None)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def split_by_group(
    decisions: Sequence[Decision], frac: float, seed: int = 0, key=lambda d: d.state
) -> tuple[list[Decision], list[Decision]]:
    """Split so that decisions sharing a state never cross the split (no leakage)."""
    groups: dict[str, list[Decision]] = {}
    for d in decisions:
        groups.setdefault(key(d), []).append(d)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    cut = int(len(keys) * frac)
    a = [d for k in keys[:cut] for d in groups[k]]
    b = [d for k in keys[cut:] for d in groups[k]]
    return a, b


def describe(decisions: Iterable[Decision]) -> dict[str, Any]:
    decisions = list(decisions)
    return {
        "n": len(decisions),
        "by_kind": dict(Counter(d.kind for d in decisions)),
        "by_source": dict(Counter(d.source for d in decisions)),
        "top_domains": dict(Counter(d.domain for d in decisions).most_common(10)),
        "mean_options": round(sum(d.num_options for d in decisions) / max(1, len(decisions)), 2),
        "uniform_targets": sum(d.is_uniform for d in decisions),
    }


def typed_decisions_splits(
    val_frac: float = 0.1, cal_frac: float = 0.1, with_factors: bool = False
) -> dict[str, list[Decision]]:
    """typed-decisions ``train`` carved by case into train / validation / calibration.

    The benchmark only ships train and test, and ``test`` is never used for
    anything but the final evaluation. The carve uses a fixed seed so that every
    caller (training, model selection, calibration) sees the same disjoint parts,
    whatever seed the training run uses.
    """
    td = load_typed_decisions("train", with_factors=with_factors)
    val, rest = split_by_group(td, val_frac, seed=TD_SPLIT_SEED)
    cal, train = split_by_group(rest, cal_frac / (1 - val_frac), seed=TD_SPLIT_SEED)
    return {"train": train, "validation": val, "calibration": cal}


def k_folds(decisions: Sequence[Decision], k: int, seed: int = 0, key=lambda d: d.state) -> list[list[Decision]]:
    """``k`` disjoint folds, split by ``key`` (one case never spans two folds)."""
    groups: dict[str, list[Decision]] = {}
    for d in decisions:
        groups.setdefault(key(d), []).append(d)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    return [[d for g in keys[i::k] for d in groups[g]] for i in range(k)]


def typed_decisions_folds(k: int = 5, with_factors: bool = True) -> dict[str, Any]:
    """Cross-validation folds over typed-decisions ``train``, outside the calibration carve.

    The pool is the train and validation parts of :func:`typed_decisions_splits`
    (1,080 cases); the calibration carve is returned unchanged (the same 600
    decisions that certified the released model) and is not in any fold.
    """
    splits = typed_decisions_splits(with_factors=with_factors)
    pool = splits["train"] + splits["validation"]
    return {"folds": k_folds(pool, k, seed=TD_SPLIT_SEED), "calibration": splits["calibration"]}


def load_profile(
    profile: str,
    split: str = "train",
    max_rows: int | None = None,
    include_typed_decisions: bool = True,
    seed: int = 0,
    typed_decisions_repeat: int = 1,
) -> list[Decision]:
    """A training mix: jev-distill streams of ``profile`` + the typed-decisions carve.

    ``split`` is train, validation or calibration. ``typed_decisions_repeat``
    oversamples the (small) typed-decisions part of the train split.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")
    if split not in ("train", "validation", "calibration"):
        raise ValueError(f"unknown split {split!r}")
    decisions = load_jev_distill(split, PROFILES[profile], max_rows=max_rows, seed=seed)
    if include_typed_decisions:
        td = typed_decisions_splits()[split]
        decisions += td * (typed_decisions_repeat if split == "train" else 1)
    random.Random(seed).shuffle(decisions)
    return decisions
