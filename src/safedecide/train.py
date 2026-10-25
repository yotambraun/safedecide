"""Training loop (plain PyTorch, single GPU, Colab-friendly)."""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .losses import total_loss
from .metrics import summarize
from .model import autocast_dtype, forward_decisions, load_backbone, make_batches, targets_tensor
from .predict import CONFIG_NAME, SafeDecider
from .schema import Decision


@dataclass
class TrainConfig:
    run_name: str = "safedecide-base"
    base_model: str = "jhu-clsp/mmBERT-small"
    init_from: str | None = None  # start from these weights (a stage-1 checkpoint); base_model stays the record
    profile: str = "clean"
    max_train_rows: int | None = None
    typed_decisions_repeat: int = 1
    output_dir: str = "runs"
    max_length: int = 384
    max_pairs_per_batch: int = 64
    grad_accum: int = 1
    epochs: float = 2.0
    max_steps: int | None = None  # cap on optimizer steps; the LR schedule is sized to it
    max_minutes: float | None = None  # wall-clock safety stop (keeps the best checkpoint so far)
    patience: int | None = None  # stop after this many evaluations without a better validation NLL
    stop_step: int | None = None  # stop here without resizing the LR schedule (a refit at a validated step)
    aux_fraction: float = 0.0  # auxiliary decisions drawn per epoch, as a fraction of the train set
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    w_brier: float = 0.5
    w_rps: float = 0.5
    w_conftr: float = 0.0
    conftr_alpha: float = 0.05
    conftr_temperature: float = 0.1
    conftr_start_frac: float = 0.3  # warm up with proper scoring before ConfTr kicks in
    eval_every: int = 500
    val_max: int = 3000
    gradient_checkpointing: bool = False
    seed: int = 42
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> TrainConfig:
        import yaml

        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.extra.update(extra)
        return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_model(model, tokenizer, decisions: Sequence[Decision], max_length: int) -> dict:
    decider = SafeDecider(model, tokenizer, {"max_length": max_length}, device=str(next(model.parameters()).device))
    probs = decider.predict_proba(decisions, calibrated=False)
    model.train()
    targets = [np.array(d.target) for d in decisions]
    return summarize(probs, targets, [d.kind for d in decisions], [d.label for d in decisions])


def _epoch_set(cfg: TrainConfig, train_set: Sequence[Decision], aux_set: Sequence[Decision], epoch: int):
    """The train set plus a fresh draw of auxiliary decisions for this epoch."""
    n_aux = min(len(aux_set), round(cfg.aux_fraction * len(train_set)))
    return list(train_set) + random.Random(cfg.seed + epoch).sample(list(aux_set), n_aux)


def steps_per_epoch(cfg: TrainConfig, train_set: Sequence[Decision], aux_set: Sequence[Decision] = ()) -> int:
    batches = make_batches(_epoch_set(cfg, train_set, aux_set, 0), cfg.max_pairs_per_batch)
    return math.ceil(len(batches) / cfg.grad_accum)


def _param_groups(model, weight_decay: float):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name.lower() or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def train(
    cfg: TrainConfig,
    train_set: Sequence[Decision],
    val_set: Sequence[Decision],
    model=None,
    tokenizer=None,
    aux_set: Sequence[Decision] = (),
) -> dict:
    """Train and keep the checkpoint with the best validation NLL.

    NLL against soft targets is a strictly proper score, so model selection
    rewards honest probabilities, not just accuracy. With an empty ``val_set``
    nothing is evaluated and the last step is kept (set ``stop_step`` to a step
    chosen on validation elsewhere). ``aux_set`` adds training-only decisions
    (see :mod:`safedecide.factors`); they are never evaluated.
    """
    from transformers import get_linear_schedule_with_warmup

    seed_everything(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None or tokenizer is None:
        tokenizer, model = load_backbone(cfg.init_from or cfg.base_model, cfg.gradient_checkpointing)
    model.to(device).train()

    out_dir = Path(cfg.output_dir) / cfg.run_name
    best_dir = out_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    val_set = list(val_set)[: cfg.val_max]
    epoch_steps = steps_per_epoch(cfg, train_set, aux_set)
    total_steps = max(1, int(epoch_steps * cfg.epochs))
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
    amp_dtype = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    best_nll, best_step, bad_evals, stopped_early = float("inf"), 0, 0, False
    step, epoch, history = 0, 0, []
    end_step = min(total_steps, cfg.stop_step or total_steps)  # the schedule stays sized to total_steps
    t0 = t_last = time.time()
    step_seconds = []  # optimizer-step durations, without evaluation and the first few warm-up steps
    log = log_path.open("w", encoding="utf-8")
    while step < end_step:
        epoch_set = _epoch_set(cfg, train_set, aux_set, epoch)
        batches = make_batches(epoch_set, cfg.max_pairs_per_batch, shuffle=True, seed=cfg.seed + epoch)
        optimizer.zero_grad(set_to_none=True)  # drop a partial accumulation left over from the last epoch
        for i, batch in enumerate(batches):
            k = max(d.num_options for d in batch)
            target = targets_tensor(batch, k, device)
            labels = torch.tensor([d.label for d in batch], device=device)
            is_ordinal = torch.tensor([d.kind == "score" for d in batch], device=device)
            w_conftr = cfg.w_conftr if step >= cfg.conftr_start_frac * total_steps else 0.0
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                logits, mask = forward_decisions(model, tokenizer, batch, cfg.max_length, device)
            loss, parts = total_loss(
                logits.float(),
                mask,
                target,
                is_ordinal,
                labels,
                cfg.w_brier,
                cfg.w_rps,
                w_conftr,
                cfg.conftr_alpha,
                cfg.conftr_temperature,
            )
            scaler.scale(loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            rec = {
                "step": step,
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "loss": float(loss.detach()),
                **parts,
            }
            now = time.time()  # float(loss) above synchronised the GPU, so this is the real step time
            if step > 5:
                step_seconds.append(now - t_last)
            t_last = now
            if step % 50 == 0:
                print(f"step {step}/{end_step} loss {rec['loss']:.4f} ce {parts['ce']:.4f} ({time.time() - t0:.0f}s)")
            out_of_time = cfg.max_minutes is not None and time.time() - t0 > cfg.max_minutes * 60
            if out_of_time:
                print(f"time budget of {cfg.max_minutes} min reached at step {step}/{end_step}")
                end_step = step
            if val_set and (step % cfg.eval_every == 0 or step == end_step):
                m = evaluate_model(model, tokenizer, val_set, cfg.max_length)
                rec["val"] = m
                history.append({"step": step, **m})
                print(f"  val nll {m['nll']:.4f} acc {m['accuracy']:.4f} ece {m['ece']:.4f}")
                if m["nll"] < best_nll:
                    best_nll, best_step, bad_evals = m["nll"], step, 0
                    _save(cfg, model, tokenizer, best_dir, step, m)
                else:
                    bad_evals += 1
                    if cfg.patience is not None and bad_evals >= cfg.patience and step < end_step:
                        print(f"  no better validation NLL in {bad_evals} evaluations: stopping at step {step}")
                        end_step, stopped_early = step, True
                t_last = time.time()
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if step >= end_step:
                break
        epoch += 1
    log.close()
    if not val_set:
        best_step = step
        _save(cfg, model, tokenizer, best_dir, step, None)
    summary = {
        "best_dir": str(best_dir),
        "best_val_nll": best_nll if val_set else None,
        "best_step": best_step,
        "steps": step,
        "steps_per_epoch": epoch_steps,
        "stopped_early": stopped_early,
        "minutes": round((time.time() - t0) / 60, 2),
        "seconds_per_step": round((time.time() - t0) / max(1, step), 3),
        "train_step_seconds": round(statistics.median(step_seconds), 4) if step_seconds else None,
        "history": history,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _save(cfg: TrainConfig, model, tokenizer, best_dir: Path, step: int, val: dict | None) -> None:
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    cfg_out = {
        "max_length": cfg.max_length,
        "base_model": cfg.base_model,
        "init_from": Path(cfg.init_from).name if cfg.init_from else None,
        "profile": cfg.profile,
        "run_name": cfg.run_name,
        "best_step": step,
        "val": val,
    }
    (best_dir / CONFIG_NAME).write_text(json.dumps(cfg_out, indent=2), encoding="utf-8")
