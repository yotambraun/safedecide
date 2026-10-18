# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-18

### Added

- SafeDecide-140M: an option-scoring cross-encoder on mmBERT-small with exact option-order
  invariance, trained in two stages (jev-distill `openjev_v2` plus typed-decisions `train`,
  then typed-decisions `train` alone) with soft cross-entropy, Brier and RPS.
- Selective risk control: Learn-then-Test with exact Clopper–Pearson bounds and
  fixed-sequence testing; every checkpoint is certified at 5%, 10% and 20% error, and
  `decide(..., alpha=...)` or `?alpha=` on the server picks the level.
- `safedecide calibrate --data my_decisions.jsonl`: certify the thresholds on your own
  labelled decisions, one `/v1/systemone`-shaped question per line.
- *Automation @ α* benchmark for any `/v1/systemone` model, a distribution-shift test, and
  `scripts/automation_board.py`, which applies the same calibration to every model in
  Hugging Face's published typed-decisions results.
- typed-decisions scorer with the benchmark's conventions (accuracy, KL from gold, Brier, ECE),
  checked against its published Uniform reference row; Prior and Uniform reference adapters.
- 95% bootstrap intervals by case, case-grouped calibration/test splits, validity and
  automation-vs-α figures, paired Conformal Training ablation, raw per-decision prediction export.
- Adapters for any `/v1/systemone` server and for Phocinae (in-process engine).
- ONNX fp32/int8 export with parity and latency checks, a `/v1/systemone` server, and a
  static Space that runs the model in the browser with transformers.js.
- Training time budget (`max_steps`, `max_minutes`) for fixed-cost GPU runs.
