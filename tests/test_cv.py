"""Cross-validation, factor questions and the per-question bias (the v0.2 training recipe)."""

import json
import random

import numpy as np
import pytest

from safedecide.calibration import calibrated_probs, fit_question_bias, question_key
from safedecide.cv import paired_accuracy_delta
from safedecide.data import k_folds, load_typed_decisions
from safedecide.factors import FACTOR_SOURCE, factor_decisions, factor_vocabulary
from safedecide.schema import Decision

TEAMS = {"refund": "billing", "invoice": "billing", "crash": "technical", "outage": "technical", "login": "account"}


def cases(n, seed, workflow="customer_service"):
    """Typed-decisions-shaped synthetic cases: two questions per state, plus factors."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        w = rng.choice(sorted(TEAMS))
        urgent = w in ("crash", "outage")
        state = f"customer says {w} please help {seed} {i}"
        factors = {"word": w, "urgent": urgent, "turns": rng.choice([1, 3, 5]), "constraints": ["x"]}
        opts = ["billing", "technical", "account"]
        meta = {"question_id": "team", "gold_label": None, "factors": factors}
        out.append(
            Decision(
                f"s{seed}c{i}:team",
                "choice",
                state,
                "which team?",
                opts,
                [0.9 if o == TEAMS[w] else 0.05 for o in opts],
                domain=workflow,
                meta=meta,
            )
        )
        out.append(
            Decision(
                f"s{seed}c{i}:urgent",
                "noul",
                state,
                "is this urgent?",
                ["false", "true"],
                [0.1, 0.9] if urgent else [0.9, 0.1],
                domain=workflow,
                meta={"question_id": "urgent", "gold_label": None, "factors": factors},
            )
        )
    return out


def test_k_folds_split_by_case_and_cover_everything():
    ds = cases(50, 0)
    folds = k_folds(ds, 5, seed=0)
    assert sum(len(f) for f in folds) == len(ds)
    states = [{d.state for d in f} for f in folds]
    for i in range(5):
        for j in range(i + 1, 5):
            assert states[i].isdisjoint(states[j])
    assert k_folds(ds, 5, seed=0)[2][0].id == folds[2][0].id  # deterministic


def test_factor_vocabulary_keeps_categories_and_skips_the_rest():
    ds = cases(40, 1)
    ds[0].meta["factors"]["note"] = "a very long sentence copied straight from the state text"
    vocab = factor_vocabulary(ds)
    kinds = {name: fq.kind for (_, name), fq in vocab.items()}
    assert kinds == {"word": "choice", "urgent": "noul", "turns": "score"}  # lists and long text skipped
    assert vocab[("customer_service", "turns")].values == (1, 3, 5)


def test_factor_decisions_are_one_hot_and_one_per_case():
    ds = cases(10, 2)
    vocab = factor_vocabulary(ds)
    aux = factor_decisions(ds, vocab)
    assert len(aux) == 10 * len(vocab)
    assert all(d.source == FACTOR_SOURCE and max(d.target) == 1.0 for d in aux)
    word = next(d for d in aux if d.id.endswith(":factor:word"))
    case = next(d for d in ds if d.id.startswith(word.id.split(":factor:")[0]))
    assert word.options[word.label] == case.meta["factors"]["word"]
    assert word.state == case.state


def test_test_split_factors_are_never_loaded():
    with pytest.raises(ValueError, match="only loaded for the train split"):
        load_typed_decisions("test", with_factors=True)


def test_question_bias_moves_only_its_own_question():
    biased = [Decision(f"b{i}", "choice", f"s{i}", "q1", ["a", "b", "c"], [0.1, 0.8, 0.1]) for i in range(60)]
    other = Decision("o", "choice", "s", "q2", ["a", "b", "c"], [0.8, 0.1, 0.1])
    logits = [np.array([1.0, 0.0, 0.0])] * len(biased)
    bias = fit_question_bias(logits, biased, {"choice": 1.0})
    assert list(bias) == [question_key(biased[0])]
    p_biased, p_other = calibrated_probs([np.array([1.0, 0.0, 0.0])] * 2, [biased[0], other], {}, bias)
    assert p_biased.argmax() == 1  # the bias overturns a one-logit lead learnt from the gold
    assert p_other.argmax() == 0 and np.allclose(p_other, calibrated_probs([np.array([1.0, 0, 0])], [other], {})[0])
    assert fit_question_bias(logits[:10], biased[:10], {}, min_count=30) == {}  # too few to fit


def test_paired_accuracy_delta():
    ds = cases(100, 3)
    gold = [np.asarray(d.target) for d in ds]
    wrong = [np.roll(t, 1) for t in gold]
    same = paired_accuracy_delta(gold, gold, ds)
    assert same["delta"] == 0.0 and same["lo"] == 0.0 == same["hi"]
    better = paired_accuracy_delta(gold, wrong, ds)
    assert better["delta"] == 1.0 and better["lo"] > 0.9


# --------------------------------------------------------------------------- #
# With a tiny model
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    d = tmp_path_factory.mktemp("tiny")
    words = set(TEAMS) | set(TEAMS.values()) | {"customer", "says", "please", "help", "which", "team", "is"}
    words |= {"this", "urgent", "option", "question", "yes", "no", "choice", "noul", "score", "fact", "about"}
    words |= {"case", "word", "turns", "1", "3", "5"}
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "|", "[", "]", ":", "?", "."] + sorted(words)
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        num_labels=1,
    )
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(d)
    BertTokenizer(str(d / "vocab.txt")).save_pretrained(d)
    return str(d)


def tiny_cfg(tiny_dir, tmp_path, **kw):
    from safedecide.train import TrainConfig

    base = dict(
        run_name="t",
        init_from=tiny_dir,
        output_dir=str(tmp_path),
        max_length=40,
        max_pairs_per_batch=30,
        epochs=3,
        lr=5e-4,
        eval_every=5,
    )
    return TrainConfig(**{**base, **kw})


@pytest.mark.e2e
def test_patience_stop_step_and_aux(tiny_dir, tmp_path):
    from safedecide.train import steps_per_epoch, train

    ds = cases(60, 4)
    still = tiny_cfg(tiny_dir, tmp_path, lr=0.0, epochs=20, patience=2)  # lr 0: the NLL never improves
    s = train(still, ds, cases(10, 5))
    assert s["stopped_early"] and s["best_step"] == 5 and s["steps"] == 15

    stop = tiny_cfg(tiny_dir, tmp_path, run_name="stop", stop_step=7)
    s = train(stop, ds, [])
    assert s["steps"] == 7 and s["best_step"] == 7 and s["best_val_nll"] is None
    assert json.loads((tmp_path / "stop" / "best" / "safedecide_config.json").read_text())["best_step"] == 7

    aux = factor_decisions(ds, factor_vocabulary(ds))
    with_aux = tiny_cfg(tiny_dir, tmp_path, aux_fraction=0.5)
    assert steps_per_epoch(with_aux, ds, aux) > steps_per_epoch(with_aux, ds)
    assert steps_per_epoch(tiny_cfg(tiny_dir, tmp_path), ds, aux) == steps_per_epoch(with_aux, ds)  # fraction 0


@pytest.mark.e2e
def test_cross_validation_roundtrip(tiny_dir, tmp_path, monkeypatch):
    import safedecide.cv as cv
    from safedecide.cv import (
        bias_from_cv,
        comparison_table,
        cross_validate,
        crossfit_bias,
        mean_temperatures,
        oof_probs,
        refit_step,
    )

    pool = cases(45, 6)
    folds = k_folds(pool, 3, seed=0)
    vocab = factor_vocabulary(pool)
    cfg = tiny_cfg(tiny_dir, tmp_path, epochs=4, patience=3, aux_fraction=0.5)
    res = cross_validate("r", cfg, folds, str(tmp_path / "cv"), str(tmp_path / "work"), vocab)
    assert len(res["folds"]) == 3 and res["factors"]
    assert not (tmp_path / "work" / "r-fold0").exists()  # fold checkpoints are not kept
    held = {i for r in res["folds"] for i in r["ids"]}
    assert held == {d.id for d in pool}  # every decision predicted exactly once, out of fold

    probs = oof_probs(res, pool)
    assert len(probs) == len(pool) and all(np.isclose(p.sum(), 1.0) for p in probs)
    assert len(crossfit_bias(res, pool, min_count=5)) == len(pool)
    assert bias_from_cv(res, pool, min_count=5)
    assert set(mean_temperatures(res)) == {"noul", "choice", "score"}
    assert refit_step(res, cfg, pool) >= 1
    assert "| a |" in comparison_table({"a": probs, "b": probs}, pool, baseline="a")

    def no_training(*a, **k):
        raise AssertionError("a finished fold was retrained")

    monkeypatch.setattr(cv, "run_fold", no_training)
    again = cross_validate("r", cfg, folds, str(tmp_path / "cv"), str(tmp_path / "work"), vocab)
    assert [r["ids"] for r in again["folds"]] == [r["ids"] for r in res["folds"]]


@pytest.mark.e2e
def test_predictor_and_certification_use_the_bias(tiny_dir, tmp_path):
    import shutil

    from safedecide.pipeline import calibrate_checkpoint
    from safedecide.predict import CONFIG_NAME, SafeDecider

    model_dir = tmp_path / "m"
    shutil.copytree(tiny_dir, model_dir)
    ds = cases(40, 7)
    team = next(d for d in ds if d.meta["question_id"] == "team")
    config = {"max_length": 40, "temperatures": {"noul": 1.0, "choice": 1.0, "score": 1.0}}
    config["question_bias"] = {question_key(team): [0.0, 0.0, 50.0]}  # always "account"
    (model_dir / CONFIG_NAME).write_text(json.dumps(config), encoding="utf-8")

    decider = SafeDecider.from_pretrained(str(model_dir), device="cpu")
    probs = decider.predict_proba(ds)
    assert all(p.argmax() == 2 for p, d in zip(probs, ds) if d.meta["question_id"] == "team")
    assert all(p[2] > 0.99 for p, d in zip(probs, ds) if d.meta["question_id"] == "team")
    raw = decider.predict_proba(ds, calibrated=False)
    assert all(p.max() < 0.9 for p, d in zip(raw, ds) if d.meta["question_id"] == "team")  # untrained: near-uniform

    out = calibrate_checkpoint(str(model_dir), ds, alpha=0.2, delta=0.1, keep_temperatures=True)
    assert out["question_bias"] == config["question_bias"]  # kept, and applied to what was certified


@pytest.mark.network
def test_real_folds_keep_the_v01_calibration_carve():
    pytest.importorskip("datasets")
    from safedecide.data import typed_decisions_folds, typed_decisions_splits

    f = typed_decisions_folds(5)
    pool = [d for fold in f["folds"] for d in fold]
    assert [len(fold) for fold in f["folds"]] == [1080] * 5
    assert [d.id for d in f["calibration"]] == [d.id for d in typed_decisions_splits()["calibration"]]
    assert {d.state for d in pool}.isdisjoint({d.state for d in f["calibration"]})
    vocab = factor_vocabulary(pool)
    assert len(vocab) == 31 and vocab[("invoice_processing", "issue")].kind == "choice"
