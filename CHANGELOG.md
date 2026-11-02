# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - Unreleased

### Added

- `revision=` on `SafeDecider.from_pretrained`, `calibrate_checkpoint`, `LocalAdapter` and
  the server, and `--revision` on `safedecide calibrate` and `safedecide serve`: load a Hub
  branch, tag or commit, e.g. the 140M repo's `v0.2` branch, or pin a commit so the weights
  and their certified thresholds cannot change under you. A revision with a local folder is
  an error rather than ignored.
- `one_per_case` certification: `safedecide calibrate --one-per-case`,
  `calibrate_checkpoint(..., one_per_case=True)`, `pipeline.certify_rules` and
  `safedecide.one_per_case` (positions of one seeded-random decision per case) certify on
  one decision per case, which is valid when whole cases are exchangeable. A labelled line
  may name its case with a `"case"` field (default: its state). `automation_at_alpha`
  takes `one_per_case=True` to measure the cost: on the saved typed-decisions `test`
  predictions of SafeDecide-140M v0.1, automation at a 10% budget falls from 32.4% to 3.9%.
- `safedecide decide`: answers one `/v1/systemone` request (a JSON object) or many (JSON
  Lines) from a file or stdin, and prints each result as JSON with every answer's
  probabilities, its `action` (automate or escalate) and the `guarantee`; `--alpha` picks a
  certified level, `--revision` a Hub revision, `--jsonl` forces one line per result.
- `calibrate` warns (`ClusteredCalibrationWarning`) when calibration decisions repeat
  cases and `one_per_case` is off. The calibration record in `safedecide_config.json` adds
  `n_cases` and `one_per_case`.

### Fixed

- `labelled_decisions_from_jsonl` (and so `safedecide calibrate --data`) read any `noul`
  answer other than true/yes/1 as `false`, so a typo, `null` or `"maybe"` silently became a
  label. It now accepts true/false, yes/no and 1/0 in any case, JSON booleans and the
  integers 1/0, and raises an error naming the file and line for anything else. Malformed
  lines and lines without an `answer` are reported the same way.
- `SafeDecider.logits` (and so `predict_proba` and `decide`) matched results to inputs by
  object identity, so passing the same `Decision` object twice left a slot empty and
  crashed. Results are now matched by position.
- `calibrate_checkpoint` ignored `output_dir` for a local checkpoint and overwrote the
  checkpoint's own `safedecide_config.json`, its certified thresholds, in place. With
  `output_dir` it now copies the checkpoint there (without `onnx/` and `assets/`, as a Hub
  download does) and re-calibrates the copy. Without `output_dir` it still calibrates in
  place, as the training notebooks do, but warns when that replaces certified thresholds.

### Changed

- `import safedecide`, the statistics (`calibrate_selective`, `automation_at_alpha`,
  `one_per_case`, ...), `safedecide.calibration` and `safedecide.pipeline` no longer import
  torch or transformers; `safedecide.SafeDecider` loads them on first access. The public
  API is unchanged.
- Temperature and per-question bias fitting use SciPy's L-BFGS-B instead of torch's LBFGS.
  On random test problems both reach the same optimum (temperatures within 1.5e-4
  relative; SciPy's objective was never higher). Saved temperatures and thresholds are not
  affected.
- The package's project URLs list the 400M model next to the 140M one.
- The release workflow's wheel smoke test runs `scripts/smoke_installed.py`: it checks the
  package was imported from the install, that the model card template and `py.typed` are in
  the wheel, and renders a card, reads labelled JSONL and certifies a rule, all offline.
- docs/METHOD.md stated that calibration decisions are i.i.d. draws. The assumption the
  bound uses is exchangeability of the units it certifies; questions about one case are
  correlated, and with each decision counted as a unit the Clopper–Pearson bound can be
  optimistic. METHOD.md, the README and the model card template now say so, and say that
  the shipped thresholds and published automation figures (unchanged) count decisions as
  units.
- `safedecide calibrate --model <local folder>` needs `--output-dir DIR` (a re-calibrated
  copy) or `--in-place` (replace the thresholds in that folder); it no longer overwrites a
  local checkpoint by default.

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
