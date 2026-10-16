# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0]

### Added

- Option-scoring cross-encoder on mmBERT-small with exact option-order invariance.
- Training with soft cross-entropy, Brier, RPS and Conformal Training (Stutz et al., 2022).
- Selective risk control: automate/escalate threshold with a Clopper–Pearson guarantee.
- *Automation @ α* benchmark for any `/v1/systemone` model, plus a distribution-shift test.
- typed-decisions scorer with the benchmark's conventions (accuracy, KL from gold, Brier, ECE),
  checked against its published Uniform reference row; Prior and Uniform reference adapters.
- 95% bootstrap intervals by case, case-grouped calibration/test splits, validity and
  automation-vs-α figures, paired ConfTr ablation, raw per-decision prediction export.
- Adapters for any `/v1/systemone` server and for Phocinae (in-process engine).
- ONNX fp32/int8 export with parity and latency checks, `/v1/systemone` server, Gradio demo Space.
- Training time budget (`max_steps`, `max_minutes`) for fixed-cost GPU runs.
