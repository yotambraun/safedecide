---
title: SafeDecide
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: true
license: apache-2.0
short_description: A decision model that knows when not to decide
models:
  - yotambr/safedecide-140m
---

Runs the ONNX export of [SafeDecide](https://github.com/yotambraun/safedecide) on CPU.
The `MODEL_REPO` Space variable selects the model repo.
