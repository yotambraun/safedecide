# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `notebooks/03_improve_gpu.ipynb`: chooses the v0.2 training recipe by nested 5-fold
  cross-validation on typed-decisions `train` (5,400 out-of-fold decisions; the 600-decision
  calibration carve and `test` stay untouched), refits the winner, seals the result
  (SHA-256 of a recipe file that records every choice and weight hash), and scores `test`
  only in a second pass that checks the seal.
- Training: early stopping (`patience`), a refit stop step that keeps the LR schedule
  (`stop_step`), training without a validation set, and training-only auxiliary decisions
  drawn fresh each epoch (`aux_fraction`).
- `safedecide.factors`: the typed-decisions `factors` of each `train` case as training-only
  questions. They are never model input, and test-split factors are never loaded.
- Per-question bias on top of temperature scaling (`question_bias` in
  `safedecide_config.json`), applied by the predictor and by threshold certification alike.
- `safedecide.cv`: cross-validation runner with per-fold resume, out-of-fold scoring,
  paired accuracy differences with case-bootstrap intervals, and the final refit step.
- `configs/typed_decisions_v2.yaml` (up to 10 epochs, early stopping) and
  `configs/typed_decisions_large.yaml` (Ettin-encoder-400m).

### Changed

- Colab setup clones the public repository first and uses `GH_TOKEN` only if that fails.

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
