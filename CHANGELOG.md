# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-26

### Added

- SafeDecide-400M ([yotambr/safedecide-400m](https://huggingface.co/yotambr/safedecide-400m)):
  the same option-scoring cross-encoder on Ettin-encoder-400m, fine-tuned on typed-decisions
  `train` alone. On typed-decisions `test`: accuracy 0.787 (v0.1: 0.754; paired gain +3.3
  points, 95% interval +1.4 to +5.2), KL 0.083, Brier 0.045, and 52.4% of decisions
  automated at a certified 10% error (v0.1: 33.2%). It certifies 5%, 10% and 20%; it takes
  605 ms per 3-option decision on one CPU thread (140M: 80 ms).
- SafeDecide-140M v0.2, sealed with the 400M and scored once: 0.748 (−0.6, no gain), so the
  released 140M stays v0.1. Its weights are on the 140M repo's `v0.2` branch.
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
- `LocalAdapter` predicts in full precision on GPU too, as the thresholds are certified and
  as the model runs on CPU. The v0.1 and v0.2 test passes used bf16 autocast on an A100;
  in full precision on CPU the 400M scores 0.7875 (bf16: 0.787), the 140M v0.2 0.755
  (0.748) and v0.1 0.7545 (0.754).

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
