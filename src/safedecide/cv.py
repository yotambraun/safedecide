"""K-fold cross-validation on the typed-decisions ``train`` split.

Every choice for a leaderboard model (stage-1 length, stage-2 schedule, the factor
questions, the per-question bias) is made here, on out-of-fold predictions for the
1,080 ``train`` cases outside the calibration carve (5,400 decisions instead of a
600-decision validation carve). ``test`` is never touched, and the calibration carve
stays untouched until the final model's threshold is certified on it.

The folds are nested so that every out-of-fold number is honest: in fold ``i`` the
model trains on the other folds minus a 10% inner carve, which picks the checkpoint
(early stopping) and fits the temperatures; fold ``i`` itself is only predicted, once.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import shutil
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .calibration import calibrated_probs, fit_question_bias, fit_temperatures
from .data import split_by_group
from .factors import case_id, factor_decisions
from .predict import SafeDecider
from .schema import Decision
from .train import TrainConfig, steps_per_epoch, train
from .typed_decisions import score_typed_decisions

INNER_FRACTION = 0.1


def _aux(decisions: Sequence[Decision], vocab) -> list[Decision]:
    return factor_decisions(decisions, vocab) if vocab else []


def run_fold(cfg: TrainConfig, folds: Sequence[Sequence[Decision]], i: int, factor_vocab=None, seed: int = 0) -> dict:
    """Train on every fold but ``i`` (minus the inner carve), predict fold ``i``. Returns its record."""
    held = list(folds[i])
    rest = [d for j, f in enumerate(folds) if j != i for d in f]
    inner_val, inner_train = split_by_group(rest, INNER_FRACTION, seed=seed)
    aux = _aux(inner_train, factor_vocab)
    t0 = time.time()
    summary = train(cfg, inner_train, inner_val, aux_set=aux)
    decider = SafeDecider.from_pretrained(summary["best_dir"], full_precision=True)
    temps = fit_temperatures(
        decider.logits(inner_val), [np.array(d.target) for d in inner_val], [d.kind for d in inner_val]
    )
    logits = decider.logits(held)
    del decider
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # the next fold loads a fresh model
    shutil.rmtree(Path(cfg.output_dir) / cfg.run_name, ignore_errors=True)  # fold checkpoints are not kept
    return {
        "fold": i,
        "ids": [d.id for d in held],
        "logits": [[round(float(v), 5) for v in z] for z in logits],
        "temperatures": temps,
        "best_step": summary["best_step"],
        "steps": summary["steps"],
        "steps_per_epoch": summary["steps_per_epoch"],
        "stopped_early": summary["stopped_early"],
        "n_train": len(inner_train),
        "n_aux_pool": len(aux),
        "minutes": round((time.time() - t0) / 60, 2),
        "history": [{k: h[k] for k in ("step", "nll", "accuracy")} for h in summary["history"]],
    }


def cross_validate(
    name: str,
    cfg: TrainConfig,
    folds: Sequence[Sequence[Decision]],
    cv_dir: str,
    work_dir: str,
    factor_vocab=None,
) -> dict:
    """All folds of one recipe. Each fold is written to ``cv_dir/name/fold<i>.json`` as it
    finishes and reused on a rerun, so a disconnect costs at most one fold."""
    out = Path(cv_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(len(folds)):
        path = out / f"fold{i}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
            print(f"{name} fold {i}: done earlier, reused")
            continue
        fold_cfg = dataclasses.replace(cfg, run_name=f"{name}-fold{i}", output_dir=work_dir)
        rec = run_fold(fold_cfg, folds, i, factor_vocab)
        path.write_text(json.dumps(rec), encoding="utf-8")
        print(f"{name} fold {i}: best step {rec['best_step']} of {rec['steps']}, {rec['minutes']} min")
        records.append(rec)
    result = {
        "name": name,
        "config": dataclasses.asdict(cfg),
        "factors": bool(factor_vocab),
        "folds": records,
    }
    (out / "summary.json").write_text(json.dumps({k: v for k, v in result.items() if k != "folds"}), encoding="utf-8")
    return result


def oof_probs(
    cv: dict, decisions: Sequence[Decision], question_bias: dict[str, list[float]] | None = None
) -> list[np.ndarray]:
    """Out-of-fold probabilities aligned with ``decisions``, each with its own fold's temperatures."""
    by_id = {}
    for rec in cv["folds"]:
        for did, z in zip(rec["ids"], rec["logits"]):
            by_id[did] = (np.asarray(z), rec["temperatures"])
    out = []
    for d in decisions:
        z, temps = by_id[d.id]
        out.extend(calibrated_probs([z], [d], temps, question_bias))
    return out


def crossfit_bias(cv: dict, decisions: Sequence[Decision], min_count: int = 30) -> list[np.ndarray]:
    """Out-of-fold probabilities with a per-question bias fitted on the *other* folds only,
    which measures what the bias would do to unseen cases."""
    by_id = {d.id: d for d in decisions}
    out_by_id = {}
    for rec in cv["folds"]:
        others = [r for r in cv["folds"] if r is not rec]
        fit_d = [by_id[i] for r in others for i in r["ids"]]
        fit_z = [
            np.asarray(z) / r["temperatures"].get(by_id[i].kind, 1.0)
            for r in others
            for i, z in zip(r["ids"], r["logits"])
        ]
        bias = fit_question_bias(fit_z, fit_d, {}, min_count=min_count)
        held = [by_id[i] for i in rec["ids"]]
        probs = calibrated_probs([np.asarray(z) for z in rec["logits"]], held, rec["temperatures"], bias)
        out_by_id.update({d.id: p for d, p in zip(held, probs)})
    return [out_by_id[d.id] for d in decisions]


def bias_from_cv(cv: dict, decisions: Sequence[Decision], min_count: int = 30) -> dict[str, list[float]]:
    """The per-question bias for a final model: fitted on all out-of-fold predictions."""
    by_id = {d.id: d for d in decisions}
    ds, zs = [], []
    for r in cv["folds"]:
        for i, z in zip(r["ids"], r["logits"]):
            ds.append(by_id[i])
            zs.append(np.asarray(z) / r["temperatures"].get(by_id[i].kind, 1.0))
    return fit_question_bias(zs, ds, {}, min_count=min_count)


def mean_temperatures(cv: dict) -> dict[str, float]:
    kinds = cv["folds"][0]["temperatures"]
    return {k: float(np.mean([r["temperatures"][k] for r in cv["folds"]])) for k in kinds}


def refit_step(cv: dict, cfg: TrainConfig, full_train: Sequence[Decision], aux: Sequence[Decision] = ()) -> int:
    """The step at which to stop a refit on all of ``full_train``: the folds' median best
    epoch, in the refit's steps per epoch (its LR schedule is sized the same way)."""
    best_epochs = [r["best_step"] / r["steps_per_epoch"] for r in cv["folds"]]
    return max(1, round(statistics.median(best_epochs) * steps_per_epoch(cfg, full_train, aux)))


def score(probs: Sequence[np.ndarray], decisions: Sequence[Decision]) -> dict:
    return score_typed_decisions(probs, decisions)


def paired_accuracy_delta(
    probs_a: Sequence[np.ndarray],
    probs_b: Sequence[np.ndarray],
    decisions: Sequence[Decision],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Accuracy of A minus B on the same decisions, with a 95% bootstrap interval over cases
    (the five questions about one case are resampled together)."""
    a = np.array([int(np.argmax(p)) == d.label for p, d in zip(probs_a, decisions)], dtype=float)
    b = np.array([int(np.argmax(p)) == d.label for p, d in zip(probs_b, decisions)], dtype=float)
    cases: dict[str, list[int]] = {}
    for i, d in enumerate(decisions):
        cases.setdefault(case_id(d), []).append(i)
    diff_sum = np.array([(a[idx] - b[idx]).sum() for idx in cases.values()])
    count = np.array([len(idx) for idx in cases.values()])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(cases), size=(n_boot, len(cases)))
    boots = diff_sum[pick].sum(1) / count[pick].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float((a - b).mean()), "lo": float(lo), "hi": float(hi)}


def comparison_table(
    probs_by_name: dict[str, Sequence[np.ndarray]], decisions: Sequence[Decision], baseline: str
) -> str:
    """Markdown: out-of-fold scores per recipe and the paired accuracy gain over ``baseline``."""
    head = "| Recipe | Accuracy ↑ | vs " + baseline + " (95% CI) | KL ↓ | Brier ↓ | ECE ↓ | noul | choice | score |\n"
    head += "|---|---|---|---|---|---|---|---|---|\n"
    rows = []
    for name, probs in probs_by_name.items():
        s = score(probs, decisions)
        o, k = s["overall"], s["by_kind"]
        if name == baseline:
            gain = "–"
        else:
            g = paired_accuracy_delta(probs, probs_by_name[baseline], decisions)
            gain = f"{g['delta']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]"
        kinds = " | ".join(f"{k[t]['accuracy']:.3f}" if t in k else "–" for t in ("noul", "choice", "score"))
        rows.append(
            f"| {name} | {o['accuracy']:.4f} | {gain} | {o['kl_from_gold']:.4f} | {o['brier']:.4f} | "
            f"{o['ece']:.3f} | {kinds} |"
        )
    return head + "\n".join(rows) + f"\n\nOut-of-fold, {len(decisions):,} decisions. ECE: 15 equal-width bins.\n"
