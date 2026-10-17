"""End-to-end smoke test with a tiny randomly initialised BERT (no downloads).

Covers: train -> save best -> calibrate -> decide -> benchmark -> ONNX export
-> parity -> server. Slow-ish (~1 min on CPU); skip with -m "not e2e".
"""

import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from safedecide.schema import Decision  # noqa: E402

WORDS = ["refund", "charge", "invoice", "crash", "error", "login", "password", "outage", "slow", "card"]
TEAM = {
    "refund": "billing",
    "charge": "billing",
    "invoice": "billing",
    "card": "billing",
    "crash": "technical",
    "error": "technical",
    "outage": "technical",
    "slow": "technical",
    "login": "account",
    "password": "account",
}


def synthetic(n, seed):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        w = rng.choice(WORDS)
        state = f"customer says {w} {rng.choice(WORDS)} please help"
        opts = ["billing", "technical", "account"]
        rng.shuffle(opts)
        target = [0.9 if o == TEAM[w] else 0.05 for o in opts]
        out.append(Decision(f"c{seed}-{i}", "choice", state, "which team?", opts, target))
        urgent = w in ("outage", "crash")
        out.append(
            Decision(
                f"n{seed}-{i}",
                "noul",
                state,
                "is this urgent?",
                ["false", "true"],
                [0.1, 0.9] if urgent else [0.9, 0.1],
            )
        )
    return out


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    d = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "|", "[", "]", ":", "?"] + sorted(
        set(WORDS)
        | set(TEAM.values())
        | {
            "customer",
            "says",
            "please",
            "help",
            "which",
            "team",
            "is",
            "this",
            "urgent",
            "option",
            "question",
            "yes",
            "no",
            "choice",
            "noul",
        }
    )
    (d / "vocab.txt").write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizer(str(d / "vocab.txt"))
    cfg = BertConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        num_labels=1,
    )
    torch.manual_seed(0)
    model = BertForSequenceClassification(cfg)
    return tok, model, d


@pytest.mark.e2e
def test_full_pipeline(tiny, tmp_path):
    from safedecide.adapters import LocalAdapter
    from safedecide.pipeline import benchmark, calibrate_checkpoint, shift_test
    from safedecide.predict import SafeDecider
    from safedecide.train import TrainConfig, train

    tok, model, _ = tiny
    cfg = TrainConfig(
        run_name="tiny",
        output_dir=str(tmp_path),
        max_length=48,
        max_pairs_per_batch=40,
        epochs=15,
        lr=5e-4,
        warmup_ratio=0.1,
        eval_every=40,
        val_max=200,
        w_conftr=0.1,
        conftr_start_frac=0.5,
    )
    summary = train(cfg, synthetic(300, 1), synthetic(60, 2), model=model, tokenizer=tok)
    best = summary["best_dir"]
    assert summary["history"][-1]["accuracy"] > 0.8

    config = calibrate_checkpoint(best, synthetic(300, 3), alpha=0.1, delta=0.1)
    assert set(config["temperatures"]) == {"noul", "choice", "score"}
    assert "statement" in config["selective_rule"]

    decider = SafeDecider.from_pretrained(best, device="cpu")
    resp = decider.decide(
        {
            "state": "customer says refund please help",
            "questions": {
                "team": {
                    "type": "choice",
                    "instructions": "which team?",
                    "criteria": {"billing": None, "technical": None, "account": None},
                },
                "urgent": {"type": "noul", "instructions": "is this urgent?"},
            },
        }
    )
    assert resp["answers"]["team"]["choice"] == "billing"
    assert resp["answers"]["team"]["action"] in {"automate", "escalate"}

    # order invariance is exact by construction
    d1 = Decision("a", "choice", "customer says crash", "which team?", ["billing", "technical", "account"])
    d2 = Decision("b", "choice", "customer says crash", "which team?", ["account", "billing", "technical"])
    p1, p2 = decider.predict_proba([d1, d2])
    assert np.allclose(p1, p2[[1, 2, 0]], atol=1e-5)

    res = benchmark(LocalAdapter(best), synthetic(200, 4), n_splits=5)
    assert res["metrics"]["accuracy"] > 0.8
    assert "alpha=0.05" in res["automation"]
    shift = shift_test(LocalAdapter(best), synthetic(150, 5), synthetic(150, 6))
    assert "shifted" in shift

    from safedecide.report import (
        ablation_table,
        markdown_table,
        plot_automation,
        plot_automation_vs_alpha,
        plot_reliability,
        plot_risk_coverage,
        plot_validity,
    )

    assert "Automation @ 5%" in markdown_table([res])
    assert "Paired Δ" in ablation_table(res, res, (0.05, 0.1))
    for plot in (plot_risk_coverage, plot_automation, plot_validity):
        assert plot([res], str(tmp_path / f"{plot.__name__}.png"))
    assert plot_automation_vs_alpha([res], str(tmp_path / "ava.png"))
    assert plot_reliability(res, str(tmp_path / "rel.png"))


@pytest.mark.e2e
def test_onnx_export_and_parity(tiny, tmp_path):
    pytest.importorskip("onnxruntime")
    from safedecide.export import check_parity, cpu_latency, export_onnx

    tok, model, _ = tiny
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    paths = export_onnx(str(tmp_path))
    assert check_parity(str(tmp_path), paths["fp32"]) < 1e-3
    lat = cpu_latency(str(tmp_path), paths["int8"], runs=5)
    assert lat["p50_ms"] > 0


@pytest.mark.e2e
def test_server(tiny, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from safedecide.server import create_app

    tok, model, _ = tiny
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    client = TestClient(create_app(str(tmp_path)))
    assert client.get("/health").json()["status"] == "ok"
    r = client.post(
        "/v1/systemone",
        json={"state": "customer says login", "questions": {"u": {"type": "noul", "instructions": "urgent?"}}},
    )
    assert r.status_code == 200
    assert 0 <= r.json()["answers"]["u"]["noul"] <= 1
    bad = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "choice"}}})
    assert bad.status_code == 422


@pytest.mark.e2e
def test_server_api_key(tiny, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from safedecide.server import create_app

    tok, model, _ = tiny
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    monkeypatch.setenv("SAFEDECIDE_API_KEY", "s3cret")
    client = TestClient(create_app(str(tmp_path)))
    body = {"state": "customer says login", "questions": {"u": {"type": "noul", "instructions": "urgent?"}}}
    assert client.post("/v1/systemone", json=body).status_code == 401
    ok = client.post("/v1/systemone", json=body, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


@pytest.mark.e2e
def test_hub_repo_id_is_downloaded(tiny, tmp_path, monkeypatch):
    """A Hub repo id must load its calibration config, not silently fall back to 'escalate everything'."""
    import json

    import huggingface_hub

    from safedecide.hub import render_model_card
    from safedecide.pipeline import calibrate_checkpoint
    from safedecide.predict import CONFIG_NAME, SafeDecider

    tok, model, _ = tiny
    hub_copy = tmp_path / "hub_copy"
    model.save_pretrained(hub_copy)
    tok.save_pretrained(hub_copy)
    rule = {"threshold": 0.0, "alpha": 0.05, "delta": 0.1, "statement": "test"}
    (hub_copy / CONFIG_NAME).write_text(json.dumps({"selective_rule": rule}), encoding="utf-8")

    def fake_snapshot_download(repo_id, local_dir=None, **kwargs):
        assert repo_id == "someone/tiny-model"
        if local_dir is None:
            return str(hub_copy)
        import shutil

        shutil.copytree(hub_copy, local_dir, dirs_exist_ok=True)
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    decider = SafeDecider.from_pretrained("someone/tiny-model", device="cpu")
    assert decider.threshold == 0.0

    # Re-calibrating a Hub model writes into a local copy, never into the Hub cache.
    out = tmp_path / "recalibrated"
    cfg = calibrate_checkpoint("someone/tiny-model", synthetic(100, 7), output_dir=str(out))
    assert cfg["path"] == str(out)
    assert (out / CONFIG_NAME).exists()

    # The card contains non-ASCII characters (α, ≤); it must round-trip as UTF-8 on every OS.
    card = render_model_card(str(out), "someone/tiny-model", [])
    assert "≤" in card


@pytest.mark.e2e
def test_step_and_time_budget(tiny, tmp_path):
    from safedecide.train import TrainConfig, train

    tok, model, _ = tiny
    cfg = TrainConfig(run_name="budget", output_dir=str(tmp_path), max_length=48, max_pairs_per_batch=40, epochs=50)
    cfg.max_steps, cfg.eval_every = 7, 1000
    summary = train(cfg, synthetic(100, 8), synthetic(20, 9), model=model, tokenizer=tok)
    assert summary["steps"] == 7
    assert 0 < summary["train_step_seconds"] <= summary["seconds_per_step"]  # steps 6-7 only, no eval or save
    assert (tmp_path / "budget" / "best" / "safedecide_config.json").exists()  # final eval always saves

    cfg = TrainConfig(run_name="clock", output_dir=str(tmp_path), max_length=48, max_pairs_per_batch=40, epochs=50)
    cfg.max_minutes, cfg.eval_every = 0.0, 1000
    summary = train(cfg, synthetic(100, 8), synthetic(20, 9), model=model, tokenizer=tok)
    assert summary["steps"] == 1
    assert summary["train_step_seconds"] is None  # too short to time


@pytest.mark.e2e
def test_stage_two_starts_from_checkpoint(tiny, tmp_path):
    """Stage 2 loads stage-1 weights but keeps recording the original base model."""
    import json

    from safedecide.train import TrainConfig, train

    tok, model, _ = tiny
    stage1 = TrainConfig(run_name="stage1", output_dir=str(tmp_path), max_length=48, max_pairs_per_batch=40)
    stage1.max_steps, stage1.eval_every = 3, 1000
    best = train(stage1, synthetic(60, 10), synthetic(20, 11), model=model, tokenizer=tok)["best_dir"]

    stage2 = TrainConfig(
        run_name="stage2", output_dir=str(tmp_path), max_length=48, max_pairs_per_batch=40, init_from=best
    )
    stage2.max_steps, stage2.eval_every = 2, 1000
    summary = train(stage2, synthetic(60, 12), synthetic(20, 13))
    cfg = json.loads((tmp_path / "stage2" / "best" / "safedecide_config.json").read_text(encoding="utf-8"))
    assert summary["steps"] == 2
    assert cfg["base_model"] == "jhu-clsp/mmBERT-small"
    assert cfg["init_from"] == "best"
