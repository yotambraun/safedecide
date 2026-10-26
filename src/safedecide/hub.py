"""Model card rendering and Hugging Face Hub upload."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from string import Template

from .data import PROFILE_LICENSE
from .report import markdown_table


def rules_table(rules: list[dict], default_alpha: float | None) -> str:
    """Markdown table of the certified error levels saved with a checkpoint."""
    lines = ["| Error level α | Threshold | Automated on the calibration carve |", "|---|---|---|"]
    for r in rules:
        level = f"{r['alpha']:.0%}" + (" (default)" if r["alpha"] == default_alpha else "")
        if r.get("threshold") is None:
            lines.append(f"| {level} | not certified | 0% (escalates everything) |")
        else:
            lines.append(f"| {level} | {r['threshold']:.3f} | {r['calibration_automation']:.1%} |")
    return "\n".join(lines)


def latency_table(latency: dict) -> str:
    """Markdown summary of ``export.cpu_latency`` and ``export.check_parity`` results."""
    lines = ["| Precision | Size | p50 | p90 | Max logit diff vs PyTorch |", "|---|---|---|---|---|"]
    for k, lat in latency["latency"].items():
        lines.append(
            f"| {k} | {lat['file_mb']} MB | {lat['p50_ms']:.1f} ms | {lat['p90_ms']:.1f} ms | "
            f"{latency['max_abs_logit_diff'][k]:.2g} |"
        )
    first = next(iter(latency["latency"].values()))
    return (
        f"onnxruntime on {first['threads']} CPU thread, one {first['n_options']}-option decision.\n\n"
        + "\n".join(lines)
        + "\n\nThe guarantee is calibrated on the full-precision model (PyTorch or fp32 ONNX). "
        "int8 is smaller and faster but changes the logits (last column), so its automate/escalate "
        "flags are not covered unless you re-calibrate it."
    )


def render_model_card(
    model_dir: str,
    repo_id: str,
    results: list[dict],
    ablation_md: str = "",
    shift: dict | None = None,
    latency: dict | None = None,
    alpha: float = 0.05,
    results_repo: str | None = None,
    shipped: dict | None = None,
) -> str:
    """``shipped`` is ``pipeline.shipped_rule_check`` on held-out data: the default rule, unchanged."""
    cfg = json.loads(Path(model_dir, "safedecide_config.json").read_text(encoding="utf-8"))
    rule = cfg.get("selective_rule", {})
    rules_md = rules_table(cfg.get("selective_rules") or ([rule] if rule else []), rule.get("alpha"))
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
    lat_md = latency_table(latency) if latency else "Not measured."
    shipped_md = ""
    if shipped:
        shipped_md = (
            f"Out of the box, the default threshold applied unchanged to typed-decisions `test` "
            f"({shipped['n']:,} decisions, never used for training or calibration) automated "
            f"{shipped['automation_rate']:.1%} of them, and {shipped['selective_error']:.1%} of those were wrong "
            f"(budget {shipped['alpha']:.0%})."
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
        rules=rules_md,
        shipped=shipped_md,
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
pretty_name: SafeDecide evaluation results
tags:
- typed-decisions
- calibration
- conformal-prediction
- selective-prediction
configs:
- config_name: typed_decisions_test
  data_files: predictions/typed_decisions_test/*.jsonl
  default: true
- config_name: jev_test_30k
  data_files: predictions/jev_test_30k/*.jsonl
- config_name: jev_ood
  data_files: predictions/jev_ood/*.jsonl
---

# SafeDecide evaluation results

Every number in the [{model_repo}](https://huggingface.co/{model_repo}) and
[yotambr/safedecide-400m](https://huggingface.co/yotambr/safedecide-400m) model cards can be
recomputed from the files here. The viewer shows the per-decision predictions, one tab per
test set:

| Tab | Test set |
|---|---|
| `typed_decisions_test` | [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `test`: 400 cases, 2,000 decisions |
| `jev_test_30k` | jev-distill `test_set_30k` (sample), never trained on |
| `jev_ood` | jev-distill `ood` (sample): the distribution-shift test |

Each row is one decision scored by one model:

| Column | Meaning |
|---|---|
| `id` | `<case>:<question>`; questions about the same case share the prefix |
| `model` | which model produced the row (released: `SafeDecide (no ConfTr)` is SafeDecide-140M v0.1, and `SafeDecide-400M v0.2`) |
| `kind` | `noul` (yes/no), `choice` or `score` |
| `options` | the answer options, in order |
| `probs` | the model's calibrated probability for each option |
| `target` | the benchmark's soft gold distribution |
| `label` | the benchmark's gold answer |

```python
from datasets import load_dataset

rows = load_dataset("{repo_id}", "typed_decisions_test", split="train")
released = rows.filter(lambda r: r["model"] == "SafeDecide (no ConfTr)")
```

Also here: the summary tables (`leaderboard.md`, `ablation.md`, `automation_board.md`), the
same data as JSON (`typed_decisions_test.json`, `typed_decisions_official.json`,
`automation_board.json`, `shift.json`, `run_info.json`) and the figures (`*.png`).
Phocinae-Largha-150M-v1 rows are its shipped weights, which saw 500 of the 2,000
typed-decisions test decisions in training.

`v0.2/` holds the v0.2 round, whose choices were made on `train` and sealed before `test`
was scored once: `recipe.json` (its SHA-256 is the seal posted in
[typed-decisions discussion #39](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/discussions/39)),
the recipe selection (`selection.md`), the cross-validation out-of-fold predictions on
`train` (`cv/`), the test results (`test_results.md`, `typed_decisions_official.json`,
`typed_decisions_test.json`, `run_info.json`), and the same test predictions, with v0.1's,
recomputed in full precision on CPU (`full_precision/`, scores in `scores.json`). The sealed
test pass ran on a GPU with bf16 autocast; the `SafeDecide-* v0.2` rows in the viewer are
that pass.

Produced by the notebooks in [yotambraun/safedecide](https://github.com/yotambraun/safedecide).
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
