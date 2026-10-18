import json
import re

import huggingface_hub
import pytest

from safedecide.hub import push_model, push_results_dataset, render_model_card


class FakeApi:
    calls: list = []

    def create_repo(self, repo_id, **kw):
        FakeApi.calls.append(("create_repo", repo_id, kw))

    def upload_folder(self, **kw):
        FakeApi.calls.append(("upload_folder", kw["repo_id"], kw))


@pytest.fixture
def model_dir(tmp_path):
    rule = {"threshold": 0.9, "alpha": 0.1, "delta": 0.1, "statement": "the promise", "calibration_automation": 0.3}
    rules = [{**rule, "alpha": 0.05, "threshold": None, "calibration_automation": 0.0}, rule]
    cfg = {
        "profile": "clean",
        "base_model": "jhu-clsp/mmBERT-small",
        "selective_rule": rule,
        "selective_rules": rules,
        "temperatures": {},
    }
    (tmp_path / "safedecide_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return tmp_path


def test_card_metadata_and_latency_table(model_dir):
    latency = {
        "max_abs_logit_diff": {"fp32": 3e-6, "int8": 0.4},
        "latency": {
            "fp32": {"p50_ms": 50.0, "p90_ms": 55.0, "threads": 1, "n_options": 3, "file_mb": 570.0},
            "int8": {"p50_ms": 27.0, "p90_ms": 31.0, "threads": 1, "n_options": 3, "file_mb": 141.5},
        },
    }
    shipped = {"alpha": 0.1, "automation_rate": 0.262, "selective_error": 0.0439, "n": 2000}
    card = render_model_card(str(model_dir), "me/model", [], latency=latency, shipped=shipped)
    assert card.startswith("---\nlicense: apache-2.0\n")
    assert "(2,000 decisions, never used for training or calibration) automated 26.2%" in card
    assert "4.4% of those were wrong (budget 10%)" in card
    assert "license_name" not in card  # only for the research profile
    assert "| int8 | 141.5 MB | 27.0 ms" in card
    assert "the promise" in card
    assert "\nbase_model: jhu-clsp/mmBERT-small\n" in card
    assert not re.search(r"\$\{?[A-Za-z_]", card), "unfilled template placeholder"
    assert "| 5% | not certified | 0% (escalates everything) |" in card
    assert "| 10% (default) | 0.900 | 30.0% |" in card


def test_push_model_and_results(model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    FakeApi.calls = []
    asset = tmp_path / "fig.png"
    asset.write_bytes(b"png")
    url = push_model(str(model_dir), "me/model", "# card", private=True, extra_files={"assets/fig.png": str(asset)})
    assert url == "https://huggingface.co/me/model"
    assert (model_dir / "README.md").read_text(encoding="utf-8") == "# card"
    assert (model_dir / "assets" / "fig.png").exists()
    assert FakeApi.calls[0] == ("create_repo", "me/model", {"repo_type": "model", "private": True, "exist_ok": True})

    results = tmp_path / "results"
    results.mkdir()
    push_results_dataset(str(results), "me/results", "me/model", private=True)
    assert "me/model" in (results / "README.md").read_text(encoding="utf-8")
    assert FakeApi.calls[-1][2]["repo_type"] == "dataset"
