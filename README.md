# SafeDecide

[![CI](https://github.com/yotambraun/safedecide/actions/workflows/ci.yml/badge.svg)](https://github.com/yotambraun/safedecide/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/safedecide)](https://pypi.org/project/safedecide/)
[![Python](https://img.shields.io/pypi/pyversions/safedecide)](https://pypi.org/project/safedecide/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/yotambraun/safedecide/blob/main/LICENSE)

A 140M-parameter decision model, and a library that tells you how much of your
traffic any decision model can handle on its own with a guaranteed error rate.

Decision models such as Jev, Clef, d3, Laya and Phocinae answer typed questions
(yes/no, pick one, rate 0–N) with a probability per option. Deploying one means
choosing which answers to act on and which to send to a person. SafeDecide makes
that choice with a statistical guarantee:

> With probability at least 1 − δ over the calibration data, the error rate among
> the decisions marked `automate` is at most α.

The guarantee needs only a few hundred labelled examples from your own traffic. It
does not assume the model is calibrated, which matters because calibration of
these models shifts with task and phrasing (an
[independent audit](https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md)
measured a 0.125 mean probability gap between asking the same thing as a yes/no
and as a two-option choice).

[Model](https://huggingface.co/yotambr/safedecide-140m) ·
[Demo](https://huggingface.co/spaces/yotambr/safedecide) ·
[Raw results](https://huggingface.co/datasets/yotambr/safedecide-results) ·
[Method](https://github.com/yotambraun/safedecide/blob/main/docs/METHOD.md)

## Results

On [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(test split, 2,000 decisions; models fitted on its `train` split, never on `test`):

| Model | Params | Accuracy | KL from gold ↓ | Brier ↓ | Automation @ 5% ↑ | Realised error |
|---|---|---|---|---|---|---|
| SafeDecide (Conformal Training) | 140M | – | – | – | – | – |
| SafeDecide (no Conformal Training) | 140M | – | – | – | – | – |
| Phocinae-Largha-150M-v1 | 144M | – | – | – | – | – |
| Prior (ignores the input) | – | – | – | – | – | – |

*Automation @ 5%*: share of decisions automated with the error among them
guaranteed ≤ 5% (δ = 10%), mean over 100 random calibration/test splits by case.
Full tables with 95% intervals, the ablation and the distribution-shift test are in
the [model card](https://huggingface.co/yotambr/safedecide-140m); every
per-decision prediction is in the [results dataset](https://huggingface.co/datasets/yotambr/safedecide-results).

## Install

```bash
pip install "safedecide[all]"
```

`safedecide` needs Python 3.11+. The extras: `data` (dataset loaders), `export`
(ONNX), `serve` (HTTP server), `report` (figures); `all` installs them all.

## Use the model

```python
from safedecide import SafeDecider

decider = SafeDecider.from_pretrained("yotambr/safedecide-140m")
resp = decider.decide(
    {
        "state": "Agent wants to run `rm -rf /var/data` on prod to free disk space.",
        "questions": {
            "allow": {"type": "noul", "instructions": "Allow without human approval?"},
            "risk": {
                "type": "score",
                "instructions": "Risk level",
                "criteria": ["none", "low", "medium", "high", "critical"],
            },
        },
    }
)
resp["answers"]["allow"]["action"]  # "automate" or "escalate"
resp["guarantee"]["statement"]  # the promise, in words
```

The model ships a threshold certified at each error level of 5%, 10% and 20%.
`decide(request, alpha=0.2)` picks one (or `?alpha=0.2` on the server); a level
the calibration data could not certify escalates everything.

The request and response follow the `/v1/systemone` shape, so
`safedecide serve --model yotambr/safedecide-140m` is a drop-in local server.

## Calibrate on your own traffic

The shipped threshold is calibrated on public data. For the guarantee to hold on
*your* traffic, re-calibrate on labelled examples from it:

```python
from safedecide.pipeline import calibrate_checkpoint

calibrate_checkpoint(
    "yotambr/safedecide-140m", my_labelled_decisions, alpha=0.05, delta=0.1, output_dir="safedecide-mine"
)
decider = SafeDecider.from_pretrained("safedecide-mine")
```

Or wrap any model whose confidences you already have:

```python
from safedecide import calibrate_selective

rule = calibrate_selective(confidence, correct, alpha=0.05, delta=0.1)
rule.threshold  # automate when confidence >= threshold
```

## Score any decision model

`Automation @ α` works for any model that returns probabilities:

```bash
safedecide benchmark --model runs/safedecide-conftr/best \
                     --http https://your-systemone-endpoint
```

## How the guarantee works

For each threshold λ on a grid fixed in advance, an exact Clopper–Pearson upper
bound on the error among calibration decisions with confidence ≥ λ, tested by
fixed-sequence testing from four fixed starting thresholds (δ split four ways);
the lowest certified threshold wins. This is Learn-then-Test (Angelopoulos et
al., 2021) with binomial p-values. The model is
trained with proper scoring rules plus Conformal Training (Stutz et al., 2022), which
pushes it towards confident answers where it can afford them.
[docs/METHOD.md](https://github.com/yotambraun/safedecide/blob/main/docs/METHOD.md) has the details and the assumptions.

## Reproduce

Two Colab notebooks reproduce every number:

1. [`notebooks/01_train_gpu.ipynb`](https://github.com/yotambraun/safedecide/blob/main/notebooks/01_train_gpu.ipynb): two-stage
   training, calibration and evaluation, about 80 minutes on one A100.
2. [`notebooks/02_publish.ipynb`](https://github.com/yotambraun/safedecide/blob/main/notebooks/02_publish.ipynb): baseline,
   figures, ONNX export and model card: about 20 minutes on a T4 (or about three hours on CPU).

From a shell, the same steps are `safedecide train | calibrate | benchmark | export | card`
(`safedecide --help`).

## Development

```bash
uv sync                       # creates .venv with the dev dependencies
uv run pytest                 # unit, end-to-end (tiny model) and benchmark-reference tests
uv run ruff check . && uv run ruff format --check .
```

These commands work the same on Linux, macOS and Windows, and CI runs the test suite on all three.
`uv sync` installs the default PyTorch build; for a smaller CPU-only install on Linux, first run
`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Data and licensing

| Profile | Training data | Model licence |
|---|---|---|
| `clean` (default) | jev-distill `openjev_v2` stream (from Open-Jev, CC0) + typed-decisions `train` (Apache-2.0) | Apache-2.0 |
| `research` | + jev-distill `yuri_v1` and `yuri_v3` (labels distilled from TypeSafe's Jev) | research, non-commercial |

The released model uses `clean`. Evaluation never touches training rows.

## Limitations

- The guarantee is marginal (over the calibration distribution), not per class or
  per customer, and it assumes deployment data is exchangeable with calibration
  data. The shift test in the model card shows what happens when it is not.
- It covers agreement with the reference labels. On typed-decisions those labels
  come from a teacher model, so "error" means disagreement with that teacher.
- Decisions, not reasoning: questions that need multi-step reasoning or world
  knowledge belong to a generative model.
- English only in evaluation. Single training seed per configuration.
- Not for fully automated decisions with legal, medical, financial or employment
  effects on people.

## Citation

```bibtex
@software{braun2026safedecide,
  author = {Braun, Yotam},
  title  = {SafeDecide: small decision models with guaranteed automation},
  year   = {2026},
  url    = {https://github.com/yotambraun/safedecide}
}
```

Built on [mmBERT](https://huggingface.co/jhu-clsp/mmBERT-small) (JHU CLSP),
[jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3),
[Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) and
[typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).
