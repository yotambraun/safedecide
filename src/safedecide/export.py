"""ONNX export (fp32 + dynamic int8) and a CPU latency benchmark.

Layout follows the transformers.js convention so the same repo runs in the
browser: ``onnx/model.onnx`` and ``onnx/model_quantized.onnx``.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def export_onnx(model_dir: str, out_dir: str | None = None, opset: int = 17) -> dict[str, str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out = Path(out_dir or model_dir) / "onnx"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    kwargs = {}
    try:  # ModernBERT/mmBERT: plain attention + no torch.compile for tracing
        kwargs = {"attn_implementation": "eager", "reference_compile": False}
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    sample = tok(["option: yes || question: is this urgent?"], ["the server is down"], return_tensors="pt")
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    fp32 = out / "model.onnx"
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(sample[n] for n in names),
            str(fp32),
            input_names=names,
            output_names=["logits"],
            dynamic_axes={**{n: {0: "batch", 1: "sequence"} for n in names}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = out / "model_quantized.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    return {"fp32": str(fp32), "int8": str(int8)}


def check_parity(model_dir: str, onnx_path: str, n: int = 8) -> float:
    """Max |logit difference| between PyTorch and ONNX on a few inputs."""
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    firsts = [f"option: {o} || question: route this ticket" for o in ["billing", "technical", "account"] * 3][:n]
    seconds = ["I was charged twice for the same subscription"] * len(firsts)
    enc = tok(firsts, seconds, padding=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**enc).logits.squeeze(-1).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feed = {i.name: enc[i.name].numpy() for i in sess.get_inputs()}
    got = sess.run(None, feed)[0].squeeze(-1)
    return float(np.abs(ref - got).max())


def cpu_latency(model_dir: str, onnx_path: str, n_options: int = 3, runs: int = 50, threads: int = 1) -> dict:
    """p50/p90 latency for one decision (= n_options pairs in one batch)."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
    firsts = [f"option: opt{i} || question: which team should handle this?" for i in range(n_options)]
    seconds = ["Customer reports the checkout page times out after the last deploy."] * n_options
    enc = tok(firsts, seconds, padding=True, return_tensors="np")
    feed = {i.name: enc[i.name] for i in sess.get_inputs()}
    for _ in range(5):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p90_ms": float(np.percentile(times, 90)),
        "threads": threads,
        "n_options": n_options,
        "file_mb": round(Path(onnx_path).stat().st_size / 1e6, 1),
    }
