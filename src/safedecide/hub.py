"""Model card rendering and Hugging Face Hub upload."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from string import Template

from .data import PROFILE_LICENSE
from .report import markdown_table


def render_model_card(
    model_dir: str,
    repo_id: str,
    results: list[dict],
    ablation_md: str = "",
    shift: dict | None = None,
    latency: dict | None = None,
    alpha: float = 0.05,
    results_repo: str | None = None,
) -> str:
    cfg = json.loads(Path(model_dir, "safedecide_config.json").read_text(encoding="utf-8"))
    rule = cfg.get("selective_rule", {})
    profile = cfg.get("profile", "clean")
    template = Template(resources.files("safedecide").joinpath("model_card.md").read_text(encoding="utf-8"))
    shift_md = "Not measured."
    if shift:
        s_in, s_sh = shift["in_distribution"], shift["shifted"]
        shift_md = (
            f"Threshold calibrated in-distribution at α={shift['alpha']:.0%}. "
            f"In-distribution: {s_in['automation_rate']:.1%} automated, {s_in['selective_error']:.2%} error. "
            f"Shifted (OOD split): {s_sh['automation_rate']:.1%} automated, **{s_sh['selective_error']:.2%} error**."
        )
    lat_md = "Not measured."
    if latency:
        lines = ["| Precision | Size | p50 | p90 | Max logit diff vs PyTorch |", "|---|---|---|---|---|"]
        for k, lat in latency["latency"].items():
            lines.append(
                f"| {k} | {lat['file_mb']} MB | {lat['p50_ms']:.1f} ms | {lat['p90_ms']:.1f} ms | "
                f"{latency['max_abs_logit_diff'][k]:.2g} |"
            )
        first = next(iter(latency["latency"].values()))
        lat_md = (
            f"onnxruntime on {first['threads']} CPU thread, one {first['n_options']}-option decision.\n\n"
            + "\n".join(lines)
            + "\n\nThe guarantee is calibrated on the full-precision model (PyTorch or fp32 ONNX). "
            "int8 is faster but shifts the probabilities slightly, so its automate/escalate flags are "
            "not covered unless you re-calibrate it."
        )
    return template.safe_substitute(
        repo_id=repo_id,
        results_repo=results_repo or f"{repo_id}-results",
        license=PROFILE_LICENSE.get(profile, "other"),
        license_extra=(
            ""
            if profile == "clean"
            else "license_name: research-non-commercial\nlicense_link: https://github.com/yotambraun/safedecide#data-and-licensing\n"
        ),
        profile=profile,
        base_model=cfg.get("base_model", ""),
        alpha=f"{rule.get('alpha', alpha):.0%}" if rule.get("alpha") is not None else f"{alpha:.0%}",
        delta=f"{rule.get('delta', 0.1):.0%}",
        threshold=rule.get("threshold"),
        statement=rule.get("statement", ""),
        temperatures=json.dumps(cfg.get("temperatures", {})),
        leaderboard=markdown_table(results, alpha) if results else "_run the benchmark first_",
        ablation=ablation_md or "_not run_",
        shift=shift_md,
        latency=lat_md,
    )


def push_model(
    model_dir: str, repo_id: str, card: str, private: bool = False, extra_files: dict[str, str] | None = None
) -> str:
    from huggingface_hub import HfApi

    Path(model_dir, "README.md").write_text(card, encoding="utf-8")
    for dest, src in (extra_files or {}).items():
        target = Path(model_dir, dest)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(src).read_bytes())
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=model_dir, repo_type="model", commit_message="Upload model")
    return f"https://huggingface.co/{repo_id}"


RESULTS_CARD = """---
license: apache-2.0
tags:
- typed-decisions
- calibration
- conformal-prediction
---

# {repo_id}

Raw evaluation outputs behind the numbers in the [{model_repo}](https://huggingface.co/{model_repo})
model card: benchmark summaries (`*.json`), figures (`*.png`) and per-decision predictions
(`predictions/<pool>/<model>.jsonl`, one line per decision with the predicted
probabilities, the soft gold target and the gold label).

Produced by the evaluation notebooks in https://github.com/yotambraun/safedecide.
"""


def push_results_dataset(results_dir: str, repo_id: str, model_repo: str, private: bool = False) -> str:
    """Upload the whole results folder (summaries, figures, per-decision predictions)."""
    from huggingface_hub import HfApi

    Path(results_dir, "README.md").write_text(
        RESULTS_CARD.format(repo_id=repo_id, model_repo=model_repo), encoding="utf-8"
    )
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, folder_path=results_dir, repo_type="dataset", commit_message="Upload results")
    return f"https://huggingface.co/datasets/{repo_id}"
