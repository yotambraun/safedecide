"""High-level steps: calibrate a checkpoint, then benchmark any adapter."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .calibration import fit_temperatures, softmax_np
from .conformal import (
    apply_selective,
    automation_at_alpha,
    calibrate_selective,
    conformal_set_stats,
    risk_coverage_curve,
)
from .data import split_by_group
from .metrics import bootstrap_ci, summarize, top_confidence
from .predict import CONFIG_NAME, SafeDecider, resolve_model_dir
from .schema import Decision


def guarantee_statement(alpha: float, delta: float) -> str:
    return (
        f"With probability >= {1 - delta:.0%} over the calibration draw, the error rate among "
        f"decisions marked 'automate' is <= {alpha:.0%}, for data exchangeable with the "
        f"calibration set. Everything else is marked 'escalate'."
    )


def calibrate_checkpoint(
    path: str,
    calibration_set: Sequence[Decision],
    alpha: float = 0.05,
    delta: float = 0.1,
    temp_fraction: float = 0.5,
    seed: int = 0,
    output_dir: str | None = None,
    temperature_set: Sequence[Decision] | None = None,
    alphas: Sequence[float] = (0.05, 0.10, 0.20),
) -> dict:
    """Fit per-kind temperatures on one half of the calibration set and the
    selective threshold on the other half (disjoint by case, so the guarantee is valid).
    Writes everything into ``safedecide_config.json`` next to the weights.

    With ``temperature_set`` (labelled data disjoint from ``calibration_set``,
    e.g. the validation split) the temperatures are fitted there and the whole
    calibration set goes to the threshold.

    A rule is certified at every level in ``alphas`` (plus ``alpha``), each a
    separate guarantee; ``alpha`` is the default one. Pick the level from your
    error budget, not from which level automates most.

    ``path`` is a local checkpoint or a Hub repo id. A Hub model is first
    downloaded into ``output_dir`` (default: a folder named after the repo), and
    the re-calibrated copy lives there.
    """
    if not Path(path).is_dir():
        path = resolve_model_dir(path, local_dir=output_dir or path.rstrip("/").split("/")[-1])
    decider = SafeDecider.from_pretrained(path)
    cal = list(calibration_set)
    if temperature_set is None:
        temp_set, conf_set = split_by_group(cal, temp_fraction, seed=seed)  # questions about one case stay together
    else:
        temp_set, conf_set = list(temperature_set), cal

    logits_t = decider.logits(temp_set)
    temps = fit_temperatures(logits_t, [np.array(d.target) for d in temp_set], [d.kind for d in temp_set])

    logits_c = decider.logits(conf_set)
    probs_c = [softmax_np(z, temps[d.kind]) for z, d in zip(logits_c, conf_set)]
    conf = top_confidence(probs_c)
    correct = np.array([int(p.argmax()) == d.label for p, d in zip(probs_c, conf_set)])
    rules = []
    for a in sorted({*alphas, alpha}):
        rule_d = calibrate_selective(conf, correct, a, delta).to_dict()
        rule_d["statement"] = guarantee_statement(a, delta)
        rules.append(rule_d)

    decider.config.update(
        {
            "temperatures": temps,
            "selective_rule": next(r for r in rules if r["alpha"] == alpha),
            "selective_rules": rules,
            "calibration": {
                "n_temperature": len(temp_set),
                "n_selective": len(conf_set),
                "sources": dict(Counter(d.source or "unknown" for d in cal)),
            },
        }
    )
    decider.save_config(path)
    return {**decider.config, "path": str(path)}


def benchmark(
    adapter,
    decisions: Sequence[Decision],
    alphas: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
    delta: float = 0.1,
    n_splits: int = 100,
    name: str | None = None,
    probs: Sequence[np.ndarray] | None = None,
) -> dict:
    """Point metrics + automation-at-guaranteed-error on one labelled pool.

    Pass ``probs`` to score predictions you already have instead of running ``adapter``.
    """
    decisions = list(decisions)
    probs = list(adapter.predict_proba(decisions) if probs is None else probs)
    targets = [np.array(d.target) for d in decisions]
    labels = [d.label for d in decisions]
    conf = top_confidence(probs)
    correct = np.array([int(p.argmax()) == y for p, y in zip(probs, labels)])
    groups = [d.state for d in decisions]  # questions about the same case stay together
    res = {
        "model": name or getattr(adapter, "name", "model"),
        "metrics": summarize(probs, targets, [d.kind for d in decisions], labels),
        "ci95": bootstrap_ci(probs, targets, groups=groups, labels=labels),
        "automation": {
            f"alpha={a}": automation_at_alpha(conf, correct, a, delta, n_splits=n_splits, groups=groups) for a in alphas
        },
        "conformal_sets": {},
        "risk_coverage": dict(zip(("coverage", "error"), risk_coverage_curve(conf, correct))),
        "_conf": conf.tolist(),
        "_correct": correct.tolist(),
    }
    perm = np.random.default_rng(0).permutation(len(decisions))
    half = len(decisions) // 2
    cal_i, test_i = perm[:half], perm[half:]
    for a in alphas:
        res["conformal_sets"][f"alpha={a}"] = conformal_set_stats(
            [probs[i] for i in cal_i],
            [labels[i] for i in cal_i],
            [probs[i] for i in test_i],
            [labels[i] for i in test_i],
            a,
        )
    return res


def shift_test(
    adapter,
    in_distribution: Sequence[Decision],
    shifted: Sequence[Decision],
    alpha: float = 0.05,
    delta: float = 0.1,
) -> dict:
    """Calibrate on in-distribution data, deploy on shifted data.

    The guarantee is only promised under exchangeability; this measures what
    actually happens when that assumption breaks.
    """
    p_in = adapter.predict_proba(list(in_distribution))
    p_sh = adapter.predict_proba(list(shifted))
    c_in = top_confidence(p_in)
    k_in = np.array([int(p.argmax()) == d.label for p, d in zip(p_in, in_distribution)])
    c_sh = top_confidence(p_sh)
    k_sh = np.array([int(p.argmax()) == d.label for p, d in zip(p_sh, shifted)])
    rule = calibrate_selective(c_in, k_in, alpha, delta)
    return {
        "alpha": alpha,
        "threshold": rule.to_dict()["threshold"],
        "in_distribution": apply_selective(rule, c_in, k_in),
        "shifted": apply_selective(rule, c_sh, k_sh),
    }


def shipped_rule_check(path: str, decisions: Sequence[Decision]) -> dict:
    """Apply the threshold saved with the checkpoint, unchanged, to new labelled data.

    This is what a user gets out of the box, without re-calibrating. The promise
    only holds if ``decisions`` are exchangeable with the calibration set.
    """
    from .conformal import SelectiveRule

    decider = SafeDecider.from_pretrained(path)
    rule_d = decider.config.get("selective_rule")
    if not rule_d:
        raise ValueError(f"{path} is not calibrated; run calibrate_checkpoint first")
    fields = {k: rule_d[k] for k in SelectiveRule.__dataclass_fields__}
    fields["threshold"] = float("inf") if fields["threshold"] is None else fields["threshold"]
    rule = SelectiveRule(**fields)
    probs = decider.predict_proba(list(decisions))
    conf = top_confidence(probs)
    correct = np.array([int(p.argmax()) == d.label for p, d in zip(probs, decisions)])
    return {"threshold": rule_d["threshold"], "alpha": rule.alpha, **apply_selective(rule, conf, correct)}


def save_predictions(
    results_dir: str, pool: str, model: str, decisions: Sequence[Decision], probs: Sequence[np.ndarray]
) -> str:
    """``<results_dir>/predictions/<pool>/<model-slug>.jsonl``: one line per decision, so every
    reported number can be recomputed. Returns the path."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-").lower()
    path = Path(results_dir, "predictions", pool, f"{slug}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d, p in zip(decisions, probs):
            row = {
                "id": d.id,
                "model": model,
                "kind": d.kind,
                "options": d.options,
                "probs": [round(float(x), 6) for x in p],
            }
            row["target"], row["label"] = d.target, d.options[d.label]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def save_results(results: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    Path(path).write_text(json.dumps(clean, indent=2), encoding="utf-8")


def load_config(path: str) -> dict:
    p = Path(path) / CONFIG_NAME
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
