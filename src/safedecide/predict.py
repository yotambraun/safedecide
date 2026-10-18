"""Inference: load a trained checkpoint and answer System One requests."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .calibration import softmax_np
from .model import autocast_dtype, forward_decisions, load_backbone, make_batches
from .schema import Decision, decisions_from_systemone_request

CONFIG_NAME = "safedecide_config.json"


def resolve_model_dir(name_or_path: str, local_dir: str | None = None) -> str:
    """Local directory for a checkpoint: the path itself, or a Hub download.

    ``local_dir`` puts the download in a normal folder instead of the Hub cache,
    which is what you want before re-calibrating (calibration writes into it).
    """
    if Path(name_or_path).is_dir():
        return str(name_or_path)
    from huggingface_hub import snapshot_download

    return snapshot_download(name_or_path, local_dir=local_dir, ignore_patterns=["onnx/*", "assets/*"])


class SafeDecider:
    def __init__(self, model, tokenizer, config: dict[str, Any] | None = None, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.path: str | None = None

    # ------------------------------------------------------------------ io
    @classmethod
    def from_pretrained(cls, name_or_path: str, device=None) -> SafeDecider:
        """Load a local checkpoint directory or a Hugging Face Hub repo id."""
        path = resolve_model_dir(name_or_path)
        tokenizer, model = load_backbone(path)
        cfg_path = Path(path) / CONFIG_NAME
        config = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        decider = cls(model, tokenizer, config, device)
        decider.path = path
        return decider

    def save_config(self, path: str) -> None:
        Path(path, CONFIG_NAME).write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    @property
    def max_length(self) -> int:
        return int(self.config.get("max_length", 384))

    @property
    def temperatures(self) -> dict[str, float]:
        return self.config.get("temperatures", {"noul": 1.0, "choice": 1.0, "score": 1.0})

    def rule(self, alpha: float | None = None) -> dict:
        """The certified rule at error level ``alpha`` (default: the one saved as default)."""
        default = self.config.get("selective_rule") or {}
        if alpha is None:
            return default
        rules = self.config.get("selective_rules") or [default]
        for r in rules:
            if r.get("alpha") is not None and math.isclose(r["alpha"], alpha):
                return r
        raise ValueError(f"no rule certified at alpha={alpha}; levels: {[r.get('alpha') for r in rules]}")

    def threshold_for(self, alpha: float | None = None) -> float:
        t = self.rule(alpha).get("threshold")
        return math.inf if t is None else float(t)

    @property
    def threshold(self) -> float:
        return self.threshold_for()

    # ------------------------------------------------------------ predict
    @torch.no_grad()
    def logits(self, decisions: Sequence[Decision], max_pairs: int = 256) -> list[np.ndarray]:
        """Raw per-option logits (ragged), in input order."""
        out: list[np.ndarray | None] = [None] * len(decisions)
        index = {id(d): i for i, d in enumerate(decisions)}
        amp_dtype = autocast_dtype(self.device)
        for batch in make_batches(decisions, max_pairs, shuffle=False):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                lg, _ = forward_decisions(self.model, self.tokenizer, batch, self.max_length, self.device)
            lg = lg.float().cpu().numpy()
            for b, d in enumerate(batch):
                out[index[id(d)]] = lg[b, : d.num_options]
        return out  # type: ignore[return-value]

    def predict_proba(self, decisions: Sequence[Decision], calibrated: bool = True) -> list[np.ndarray]:
        temps = self.temperatures if calibrated else {}
        return [softmax_np(z, temps.get(d.kind, 1.0)) for z, d in zip(self.logits(decisions), decisions)]

    # ---------------------------------------------------- System One API
    def decide(self, request: dict[str, Any], alpha: float | None = None) -> dict[str, Any]:
        """Answer a ``/v1/systemone`` request; adds an automate/escalate flag.

        ``alpha`` picks one of the certified error levels (default: the model's default).
        """
        t0 = time.perf_counter()
        rule = self.rule(alpha)
        threshold = self.threshold_for(alpha)
        decisions = decisions_from_systemone_request(request)
        probs = self.predict_proba(decisions)
        answers: dict[str, Any] = {}
        for d, p in zip(decisions, probs):
            top = int(p.argmax())
            conf = float(p[top])
            ans: dict[str, Any] = {
                "type": d.kind,
                "confidence": conf,
                "probabilities": {o: float(pi) for o, pi in zip(d.options, p)},
                "action": "automate" if conf >= threshold else "escalate",
            }
            if d.kind == "noul":
                ans["noul"] = float(p[1])
            elif d.kind == "score":
                ans["score"] = float(sum(i * pi for i, pi in enumerate(p)))
                ans["choice"] = d.options[top]
            else:
                ans["choice"] = d.options[top]
            answers[d.id] = ans
        return {
            "object": "structured_decision",
            "model": self.config.get("model_name", "safedecide"),
            "answers": answers,
            "guarantee": {
                "alpha": rule.get("alpha"),
                "delta": rule.get("delta"),
                "statement": rule.get("statement"),
            },
            "usage": {"output_tokens": 0},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
