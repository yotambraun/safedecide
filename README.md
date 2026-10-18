# SafeDecide

[![CI](https://github.com/yotambraun/safedecide/actions/workflows/ci.yml/badge.svg)](https://github.com/yotambraun/safedecide/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/safedecide)](https://pypi.org/project/safedecide/)
[![Downloads](https://static.pepy.tech/badge/safedecide)](https://pepy.tech/projects/safedecide)
[![Python](https://img.shields.io/pypi/pyversions/safedecide)](https://pypi.org/project/safedecide/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/yotambraun/safedecide/blob/main/LICENSE)

**A small decision model that knows when not to decide.**

Decision models such as Jev, Clef, d3 and Laya answer typed questions (yes or no,
pick one, rate 0 to N) with a probability for each option. Putting one in
production comes down to one question: which answers can you act on, and which
should go to a person? SafeDecide answers it with a threshold that is certified on
labelled data, not guessed:

> With probability at least 1 − δ over the calibration data, at most α of the
> decisions marked `automate` are wrong.

It ships as a 140M model you can run anywhere, including in a browser tab, and as a
library that puts the same guarantee on top of any decision model you already use.

<p align="center">
  <a href="https://huggingface.co/spaces/yotambr/safedecide">
    <img src="https://raw.githubusercontent.com/yotambraun/safedecide/main/docs/assets/demo-escalate.png" width="540" alt="The demo on a held-out invoice: the model leans towards approve at 0.52, below the certified line of 0.70, so the decision is escalated. The benchmark's answer is reject.">
  </a>
</p>

<p align="center"><em>A held-out invoice that duplicates one already paid. The model leans towards
<b>approve</b>; the benchmark says <b>reject</b>. Its confidence is below the certified line,
so it goes to a person.</em> <a href="https://huggingface.co/spaces/yotambr/safedecide">Try it in your browser</a>.</p>

[Demo](https://huggingface.co/spaces/yotambr/safedecide) ·
[Model](https://huggingface.co/yotambr/safedecide-140m) ·
[Raw results](https://huggingface.co/datasets/yotambr/safedecide-results) ·
[Method](https://github.com/yotambraun/safedecide/blob/main/docs/METHOD.md)

## Why a certified threshold

Picking a confidence cut-off by eye fails quietly. The probabilities these models
return move with the task and even with the phrasing: an
[independent audit](https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md)
measured a 0.125 mean gap between asking the same thing as a yes/no and as a
two-option choice. SafeDecide does not trust the probabilities. It tests candidate
thresholds against labelled decisions with exact binomial bounds and keeps the
lowest one it can certify, so the promise holds even when the model is
miscalibrated. On typed-decisions, 300 labelled decisions are enough to certify a 20%
budget and about 600 a 10% budget; 5% needs more than that.

## Results

On [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(test split: 400 cases, 2,000 decisions). SafeDecide was fitted on the benchmark's
`train` split and never saw `test`.

| Model | Params | Accuracy ↑ | KL from gold ↓ | Brier ↓ | Automation @ 10% ↑ | Realised error |
|---|---|---|---|---|---|---|
| **SafeDecide** (released) | 140M | 0.754 | 0.126 | 0.069 | **33.2%** ± 5.2 | 6.3% |
| SafeDecide with Conformal Training (ablation) | 140M | 0.753 | 0.130 | 0.070 | 29.7% ± 5.2 | 6.3% |
| Phocinae-Largha-150M-v1, shipped weights (trained on 500 test decisions) | 144M | 0.906 | 0.055 | 0.025 | 93.9% ± 2.9 | 7.8% |
| Prior (ignores the input) | – | 0.487 | 0.328 | 0.181 | 0.0% | – |

*Automation @ 10%* is the share of decisions the model may take alone with the error
among them certified at most 10% (δ = 10%), as mean ± std over 100 random
calibration/test splits by case. Out of the box, the threshold shipped with the
model, applied unchanged to all 2,000 test decisions, automated 26.2% of them, and
4.4% of those were wrong.

At 5% nothing can be certified on the 600 calibration decisions. The benchmark's
gold is the mean of three samples from a teacher model, and a fresh sample from that
teacher matches it only 73.5% of the time, so even confident answers carry label noise.

### Every model Hugging Face scored, under one calibration

Hugging Face published per-decision results for the models it ran on typed-decisions
(bucket [`SaylorTwift/typed-decisions-evals`](https://huggingface.co/buckets/SaylorTwift/typed-decisions-evals)).
[`scripts/automation_board.py`](https://github.com/yotambraun/safedecide/blob/main/scripts/automation_board.py)
applies the same calibration, case splits and correctness rule to every row, ours
included. That scorer counts a tie in the gold as correct, so accuracies differ
slightly from the table above.

![Share of typed-decisions test decisions automated with a certified error of at most 10%, for every model Hugging Face scored](https://raw.githubusercontent.com/yotambraun/safedecide/main/results/automation_board.png)

Two things stand out. The order is not the accuracy order: at a 5% budget,
JEV-27B-VL (accuracy 0.688) can take 20.4% of decisions alone, while d3 (0.749) can
take 0.6%. And a 140M model sits next to 27B general models; read that with the
labels, since SafeDecide was fitted on `train` and they were scored zero-shot.

Full tables with 95% intervals, the Conformal Training ablation and the
distribution-shift test are in [`results/`](https://github.com/yotambraun/safedecide/tree/main/results)
and the [model card](https://huggingface.co/yotambr/safedecide-140m). Every
per-decision prediction is in the [results dataset](https://huggingface.co/datasets/yotambr/safedecide-results).

## Quickstart

```bash
pip install "safedecide[all]"
```

Python 3.11 or newer. The extras are `data` (dataset loaders), `export` (ONNX),
`serve` (HTTP server) and `report` (figures); `all` installs them all.

The request below is a held-out customer-service case from typed-decisions, in the
same `/v1/systemone` shape as Jev, Clef and d3:

```python
from safedecide import SafeDecider

decider = SafeDecider.from_pretrained("yotambr/safedecide-140m")
response = decider.decide(
    {
        "state": {
            "account": {
                "lifetime_value_usd": 228,
                "prior_tickets_90d": 0,
                "seats": 1,
                "tenure_months": 3,
                "tier": "standard",
            },
            "thread": [
                {"role": "customer", "text": "My payment didn’t go through, and I’m now locked out of my account."}
            ],
        },
        "questions": {
            "topic": {
                "type": "choice",
                "instructions": "What is this customer conversation primarily about?",
                "criteria": {
                    "account": "Login, plan changes, profile or access management.",
                    "billing": "A charge, invoice, subscription or payment problem.",
                    "delivery": "Shipping, fulfilment or delivery of a physical item.",
                    "refund": "The customer is explicitly asking for money back.",
                    "technical": "The product or service is not working as expected.",
                },
            },
            "needs_human": {
                "type": "noul",
                "instructions": "This conversation requires a human agent rather than automated handling.",
                "criteria": {
                    "false": "Automation can carry this to resolution.",
                    "true": "A human must take over: judgement, authority or empathy is required.",
                },
            },
        },
    }
)

response["answers"]["topic"]  # choice "billing", confidence 0.92, action "automate"
response["answers"]["needs_human"]  # noul 0.23 (no), confidence 0.77, action "automate"
response["guarantee"]["statement"]  # the promise behind every "automate", in words
```

The model carries a certified threshold for error budgets of 5%, 10% (the default)
and 20%. Pass `alpha=0.2` to `decide` to use another; a budget the calibration data
could not certify escalates everything rather than promise too much.

To serve it behind the same API as Jev, run
`safedecide serve --model yotambr/safedecide-140m` and add `?alpha=0.2` to a request
to change the budget.

## Calibrate on your own data

The shipped thresholds were certified on typed-decisions. Inputs that look different
are outside that guarantee, so before you rely on the flag in production,
re-certify on labelled decisions from your own traffic. Ticket systems usually hold
these labels already: the team that closed the ticket, whether the refund went
through.

Write one labelled question per line, in the same shape as a request:

```json
{"state": {"thread": "..."}, "question": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "...", "technical": "..."}}, "answer": "billing"}
{"state": {"thread": "..."}, "question": {"type": "noul", "instructions": "This needs a human."}, "answer": true}
```

`answer` is the option name for `choice`, `true` or `false` for `noul`, and the level
number for `score`. Then:

```bash
safedecide calibrate --model yotambr/safedecide-140m --data my_decisions.jsonl --output-dir safedecide-mine
```

```python
decider = SafeDecider.from_pretrained("safedecide-mine")  # same model, your thresholds
```

The model keeps the temperatures it shipped with, so every line of your file goes to
the threshold (`--refit-temperatures` spends half of it refitting them instead). How
many lines you need, measured on held-out typed-decisions questions the temperatures
never saw:

| Labelled decisions | 20% budget certified | 10% budget certified |
|---|---|---|
| 300 | in 46 of 50 random draws | in 4 of 50 |
| 450 | in 50 of 50 | in 15 of 50 |
| 600 | yes (66% automated) | yes (23% automated) |

A more accurate model on your data needs fewer; a budget that cannot be certified
escalates everything rather than promise too much.

Already running another decision model? The guarantee needs only its confidences and
whether each answer was right:

```python
from safedecide import calibrate_selective

rule = calibrate_selective(confidence, correct, alpha=0.1, delta=0.1)
rule.threshold  # automate when confidence >= threshold (inf: nothing could be certified)
```

## Measure any decision model

`safedecide benchmark` scores any server that speaks `/v1/systemone` on typed-decisions
and jev-distill, and reports how much it can automate at each certified budget,
with figures:

```bash
safedecide benchmark --http http://localhost:8000 --out results/
```

## How the guarantee works

1. Fix a grid of thresholds in advance.
2. For each threshold, take the calibration decisions at or above it and compute an
   exact Clopper–Pearson upper bound on their error rate.
3. Walk down the grid from four fixed starting points (fixed-sequence testing, δ
   split four ways), stop each walk at its first failure, and keep the lowest
   threshold any walk certified.

This is Learn-then-Test (Angelopoulos et al., 2021) with binomial p-values. It needs
exchangeable data, not calibrated probabilities. The released model is trained with
proper scoring rules (cross-entropy against teacher distributions, Brier score, and
ranked probability score for ordinal questions); Conformal Training (Stutz et al.,
2022) was tested as an ablation and is not used.
[docs/METHOD.md](https://github.com/yotambraun/safedecide/blob/main/docs/METHOD.md)
has the proof sketch, the assumptions and the known weaknesses.

## Limitations

- The guarantee covers data like the calibration set. On shifted data it can break:
  in the shift test, the error among automated decisions rose from 3.3% to 8.4%
  against a 5% budget. Re-calibrate on your own traffic.
- It is marginal over the calibration distribution, not per class, per task or per
  customer.
- On typed-decisions the reference labels come from a teacher model, so "error"
  means disagreement with that teacher.
- Decisions, not reasoning: questions that need multi-step reasoning or world
  knowledge belong to a generative model.
- Evaluated in English only, with one training seed per configuration.
- Not for fully automated decisions with legal, medical, financial or employment
  effects on people.

## Reproduce

Two Colab notebooks reproduce every number:

1. [`notebooks/01_train_gpu.ipynb`](https://github.com/yotambraun/safedecide/blob/main/notebooks/01_train_gpu.ipynb):
   two-stage training, calibration and evaluation, about 80 minutes on one A100.
2. [`notebooks/02_publish.ipynb`](https://github.com/yotambraun/safedecide/blob/main/notebooks/02_publish.ipynb):
   certification, baseline, figures, ONNX export and model card, about 20 minutes on a T4.

The same steps run from a shell with `safedecide train | calibrate | benchmark | export | card`
(see `safedecide --help`).

## Development

```bash
uv sync                       # creates .venv with the dev dependencies
uv run pytest                 # unit, end-to-end (tiny model) and benchmark-reference tests
uv run ruff check . && uv run ruff format --check .
```

These commands work the same on Linux, macOS and Windows, and CI runs the tests on
all three. `uv sync` installs the default PyTorch build; for a smaller CPU-only
install on Linux, first run `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Data and licensing

| Profile | Training data | Model licence |
|---|---|---|
| `clean` (default) | jev-distill `openjev_v2` stream (from Open-Jev, CC0) + typed-decisions `train` (Apache-2.0) | Apache-2.0 |
| `research` | + jev-distill `yuri_v1` and `yuri_v3` (labels distilled from TypeSafe's Jev) | research, non-commercial |

The released model uses `clean`. Evaluation never touches training rows.

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
[Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev),
[typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) and
the per-decision results Hugging Face published for it.
