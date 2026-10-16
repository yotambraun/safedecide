"""Hugging Face Space: try SafeDecide on CPU (onnxruntime, no GPU).

Set MODEL_REPO in the Space settings (Variables) to the model repo. For a local
test, point it at a checkpoint folder: MODEL_REPO=runs/x/best python space/app.py
Self-contained on purpose: needs only onnxruntime and a tokenizer, not safedecide.
"""

import json
import os
import time

import gradio as gr
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_REPO = os.environ.get("MODEL_REPO", "yotambraun/safedecide-140m")


def model_file(name: str) -> str:
    """A file from the model repo, or from a local checkpoint folder when testing."""
    if os.path.isdir(MODEL_REPO):
        return os.path.join(MODEL_REPO, name)
    return hf_hub_download(MODEL_REPO, name)


tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
with open(model_file("safedecide_config.json"), encoding="utf-8") as f:
    config = json.load(f)
# fp32, not int8: the automate/escalate threshold was calibrated on full-precision outputs.
session = ort.InferenceSession(model_file("onnx/model.onnx"), providers=["CPUExecutionProvider"])
TEMPS = config.get("temperatures", {})
RULE = config.get("selective_rule", {})
THRESHOLD = RULE.get("threshold")
THRESHOLD = float("inf") if THRESHOLD is None else float(THRESHOLD)
MAX_LEN = int(config.get("max_length", 384))
NOUL = {"false": "no", "true": "yes"}


def score(kind, state, question, options, descriptions):
    firsts = []
    for o, d in zip(options, descriptions):
        text = NOUL.get(o, o) if kind == "noul" else o
        text = f"{text}: {d}" if d else text
        firsts.append(f"[{kind}] option: {text} || question: {question}")
    enc = tokenizer(
        firsts, [state] * len(firsts), truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="np"
    )
    feed = {i.name: enc[i.name] for i in session.get_inputs()}
    t0 = time.perf_counter()
    logits = session.run(None, feed)[0].squeeze(-1) / TEMPS.get(kind, 1.0)
    ms = (time.perf_counter() - t0) * 1000
    p = np.exp(logits - logits.max())
    return p / p.sum(), ms


def run(kind, state, question, options_text):
    if kind == "noul":
        options, descs = ["false", "true"], [None, None]
    else:
        lines = [ln.strip() for ln in options_text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return "Give at least two options, one per line (`name: description`).", None
        options = [ln.split(":", 1)[0].strip() for ln in lines]
        descs = [ln.split(":", 1)[1].strip() if ":" in ln else None for ln in lines]
    p, ms = score(kind, state, question, options, descs)
    top = int(p.argmax())
    automate = p[top] >= THRESHOLD
    shown = [NOUL.get(o, o) if kind == "noul" else o for o in options]
    verdict = "✅ **Automate**" if automate else "↗️ **Escalate to a human / bigger model**"
    md = (
        f"### {shown[top]}  ({p[top]:.0%})\n\n{verdict}\n\n"
        f"Threshold {THRESHOLD:.3f} · {ms:.0f} ms on CPU\n\n"
        f"_{RULE.get('statement', '')}_"
    )
    return md, {o: float(pi) for o, pi in zip(shown, p)}


EXAMPLES = [
    [
        "choice",
        "I was charged twice for the same subscription this month and support hasn't answered.",
        "Which team should handle this ticket?",
        "billing: payments, invoices, refunds\ntechnical: bugs, outages, errors\naccount: login, access, profile",
    ],
    [
        "noul",
        "The agent wants to run `rm -rf /var/data` on the production host to free disk space.",
        "Should this action be allowed without human approval?",
        "",
    ],
    [
        "score",
        "Checkout API p99 latency jumped to 8.4s and the TLS certificate expires in 40 minutes.",
        "How severe is this incident?",
        "0: no impact\n1: minor\n2: degraded\n3: major\n4: critical",
    ],
]

with gr.Blocks(title="SafeDecide") as demo:
    gr.Markdown(
        "# SafeDecide\nA small decision model that knows when **not** to decide. "
        f"Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})"
    )
    with gr.Row():
        with gr.Column():
            kind = gr.Radio(["choice", "noul", "score"], value="choice", label="Question type")
            state = gr.Textbox(lines=4, label="State (ticket, log, agent step, JSON…)")
            question = gr.Textbox(label="Question")
            options = gr.Textbox(lines=4, label="Options, one per line (`name: description`). Ignored for noul.")
            btn = gr.Button("Decide", variant="primary")
        with gr.Column():
            verdict = gr.Markdown()
            probs = gr.Label(label="Probabilities")
    btn.click(run, [kind, state, question, options], [verdict, probs])
    gr.Examples(EXAMPLES, [kind, state, question, options])

demo.launch()
