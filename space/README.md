---
title: SafeDecide
emoji: 🚦
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
short_description: A decision model that knows when not to decide
models:
  - yotambr/safedecide-140m
datasets:
  - yotambr/safedecide-results
  - LocalLLaMA/typed-decisions
---

The [SafeDecide](https://github.com/yotambraun/safedecide) model running in your browser with
[transformers.js](https://github.com/huggingface/transformers.js). Each answer is marked automate or
escalate against an error budget certified on held-out data. The page tokenizes and truncates
exactly like the Python package, so its scores match PyTorch to within 1e-5 and the certified
thresholds apply unchanged. Nothing you type leaves the browser.

Local test: `python -m http.server` next to a checkpoint folder `x/` (with `onnx/model.onnx`),
then open `index.html?model=x&local=1`.
