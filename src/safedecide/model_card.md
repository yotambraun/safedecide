---
license: $license
$license_extrabase_model: $base_model
library_name: transformers
pipeline_tag: text-classification
language:
- en
datasets:
- SargeDev/jev-distill-corpus-v3
- LocalLLaMA/typed-decisions
tags:
- systemone
- typed-decisions
- decision-model
- conformal-prediction
- calibration
- selective-prediction
- onnx
- transformers.js
---

# $repo_id

**A small decision model that knows when not to decide.**

Give it a state and a typed question (yes/no, pick-one, or score). It returns a
probability per option *and* an `automate` / `escalate` flag backed by a
statistical guarantee:

> $statement

The guarantee does **not** rely on the model's probabilities being calibrated.
It comes from split conformal risk control (exact Clopper–Pearson bounds,
Bonferroni over a fixed threshold grid) on a held-out calibration set.

Code: [github.com/yotambraun/safedecide](https://github.com/yotambraun/safedecide) ·
raw predictions behind every number: [$results_repo](https://huggingface.co/datasets/$results_repo)

## Why this exists

System One decision models (Jev, Clef, d3, Laya, Phocinae, …) return
probabilities your code can branch on. Their calibration moves with task and
phrasing ([one audit](https://github.com/jujumilk3/jev-calibration-audit/blob/main/FINDINGS.md)
measured a 0.125 mean gap between the same question asked as yes/no and as a
two-option choice), so a threshold tuned on someone else's data says little about
yours. This model ships with the tool that turns a re-check on your own data into
a guarantee, and was trained to automate as much as that guarantee allows.

## Results

Evaluated on [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
`test` (400 cases, 2,000 decisions), the jev-distill `test_set_30k` and its `ood`
split. The model was fitted on typed-decisions `train` (it belongs in that
benchmark's "fitted on train" table) and never saw `test`.

Headline metric: **automation rate at a guaranteed error ≤ $alpha**, i.e. the
share of decisions the model may take alone while the error among those
decisions stays within budget (δ = $delta).

$leaderboard

Numbers are means over 100 random calibration/test splits of the same labelled
pool, split by case. Brackets are 95% bootstrap intervals over cases. "Violation rate" is how often the realised error on the test half
exceeded the budget. It is an empirical sanity check of the guarantee (which
promises the true error exceeds the budget with probability ≤ δ); the finite
test half adds some sampling noise of its own.

### Ablation: does Conformal Training help?

$ablation

### What happens under distribution shift

$shift

The guarantee assumes deployment data looks like the calibration data. If it
doesn't, re-calibrate on ~500 labelled examples from your own traffic
(`safedecide calibrate`). Ticket systems usually already hold these labels:
which team closed the ticket, whether the refund was approved.

### Speed

$latency

## Usage

```python
from safedecide import SafeDecider

decider = SafeDecider.from_pretrained("$repo_id")
decider.decide(
    {
        "state": "I was charged twice for the same subscription this month.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Payments and invoices",
                    "technical": "Bugs and outages",
                    "account": "Login and access",
                },
            },
            "urgent": {"type": "noul", "instructions": "Does this need a reply today?"},
        },
    }
)
# -> answers.team.choice == "billing", answers.team.action == "automate" | "escalate"
```

Serve it behind the same `/v1/systemone` API as Jev: `safedecide serve --model $repo_id`.

## How it works

* **Architecture:** each (state, question, option) triple is scored by a
  cross-encoder with one logit; softmax over a question's options. Option-order
  invariance is exact by construction. Standard `AutoModelForSequenceClassification`,
  so it exports to ONNX and runs in the browser with transformers.js.
* **Training:** soft-label cross-entropy against teacher distributions + Brier
  score (proper scoring) + ranked probability score for ordinal questions +
  Conformal Training size penalty (Stutz et al., ICLR 2022). Two stages: jev-distill
  `openjev_v2` plus the typed-decisions train carve, then the typed-decisions train
  carve alone at 640 tokens (no typed-decisions input is truncated).
* **Calibration:** per-kind temperature scaling (temperatures: `$temperatures`),
  then the selective threshold (`$threshold`) on a disjoint split of the
  typed-decisions calibration carve (by case, never `test`).

## Data and license

Data profile: **$profile**.

* `clean`: the `openjev_v2` stream of jev-distill-corpus-v3 (CC0 at source) plus
  the typed-decisions train split (Apache-2.0). Released under Apache-2.0.
* `research`: additionally the `yuri_v1` and `yuri_v3` streams. `yuri_v3` labels
  were distilled from TypeSafe Jev 1.13; checkpoints trained on it are released
  for research and non-commercial use only.

Test data: `test_set_30k` and `ood` splits of jev-distill (never trained on) and
the typed-decisions test split.

## Limitations

* Decisions, not text. Questions that need multi-step reasoning or world
  knowledge belong to a generative model.
* The guarantee is marginal over the calibration distribution, not per class
  and not per customer.
* English-first; the base encoder is multilingual (1800+ languages) but other
  languages are not evaluated here.
* Not for fully automated decisions with legal, medical, financial or
  employment effects on people.

## Citation

```bibtex
@misc{braun2026safedecide,
  title  = {SafeDecide: small decision models with guaranteed automation},
  author = {Yotam Braun},
  year   = {2026},
  url    = {https://huggingface.co/$repo_id}
}
```
