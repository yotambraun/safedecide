"""Option-scoring cross-encoder.

Each (state, question, option) triple is scored independently by an encoder
with a single-logit head; a softmax over a decision's options gives its
distribution. Two consequences, both deliberate:

* **Exact option-order invariance.** Reordering options permutes the logits
  and nothing else. No position bias by construction.
* **Standard architecture.** The checkpoint is a plain
  ``AutoModelForSequenceClassification(num_labels=1)``, so it exports to ONNX
  and runs in transformers.js / onnxruntime-web without custom code.

The cost is K forward passes for K options instead of one shared pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .schema import Decision

NEG_INF = -1e4  # safe in fp16/bf16


def pair_text(d: Decision, i: int) -> tuple[str, str]:
    """(first, second) segments for option ``i``.

    The option comes first so truncation can only ever cut the state or the
    end of the question, never the option being scored.
    """
    first = f"[{d.kind}] option: {d.option_text(i)} || question: {d.question}"
    return first, d.state


@dataclass
class EncodedBatch:
    inputs: dict[str, torch.Tensor]
    owner: torch.Tensor  # [P] decision index per pair
    slot: torch.Tensor  # [P] option index per pair
    num_decisions: int
    max_options: int


def encode_decisions(tokenizer, decisions: Sequence[Decision], max_length: int = 384) -> EncodedBatch:
    firsts, seconds, owner, slot = [], [], [], []
    for b, d in enumerate(decisions):
        for i in range(d.num_options):
            f, s = pair_text(d, i)
            firsts.append(f)
            seconds.append(s)
            owner.append(b)
            slot.append(i)
    inputs = tokenizer(
        firsts,
        seconds,
        truncation="longest_first",
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return EncodedBatch(
        inputs=dict(inputs),
        owner=torch.tensor(owner, dtype=torch.long),
        slot=torch.tensor(slot, dtype=torch.long),
        num_decisions=len(decisions),
        max_options=max(d.num_options for d in decisions),
    )


def scatter_logits(flat: torch.Tensor, batch: EncodedBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """[P] pair logits -> padded [B, K] logits and boolean mask."""
    logits = flat.new_full((batch.num_decisions, batch.max_options), NEG_INF)
    mask = torch.zeros(batch.num_decisions, batch.max_options, dtype=torch.bool, device=flat.device)
    owner = batch.owner.to(flat.device)
    slot = batch.slot.to(flat.device)
    logits[owner, slot] = flat
    mask[owner, slot] = True
    return logits, mask


def forward_decisions(
    model, tokenizer, decisions: Sequence[Decision], max_length: int = 384, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder on all options of ``decisions``. Returns (logits [B,K], mask [B,K])."""
    device = device or next(model.parameters()).device
    batch = encode_decisions(tokenizer, decisions, max_length)
    inputs = {k: v.to(device) for k, v in batch.inputs.items()}
    flat = model(**inputs).logits.squeeze(-1)
    return scatter_logits(flat, batch)


def targets_tensor(decisions: Sequence[Decision], k: int, device=None) -> torch.Tensor:
    t = torch.zeros(len(decisions), k, device=device)
    for b, d in enumerate(decisions):
        t[b, : d.num_options] = torch.tensor(d.target, device=device)
    return t


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    z = logits.float() / temperature
    z = z.masked_fill(~mask, float("-inf"))
    return torch.softmax(z, dim=-1)


def make_batches(
    decisions: Sequence[Decision], max_pairs: int, shuffle: bool = True, seed: int = 0
) -> list[list[Decision]]:
    """Greedy batching by total option count, so memory stays flat across kinds."""
    import random

    order = list(range(len(decisions)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batches, cur, pairs = [], [], 0
    for i in order:
        k = decisions[i].num_options
        if cur and pairs + k > max_pairs:
            batches.append(cur)
            cur, pairs = [], 0
        cur.append(decisions[i])
        pairs += k
    if cur:
        batches.append(cur)
    return batches


def autocast_dtype(device: str) -> torch.dtype | None:
    """bf16 on GPUs that support it natively (Ampere+), fp16 otherwise, None on CPU.

    ``torch.cuda.is_bf16_supported()`` also returns True when bf16 is only
    emulated (e.g. T4), which is much slower than fp16, so check for native support.
    """
    if not str(device).startswith("cuda"):
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def load_backbone(name_or_path: str, gradient_checkpointing: bool = False):
    """Load tokenizer + single-logit classification model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    kwargs = {"num_labels": 1}
    model = AutoModelForSequenceClassification.from_pretrained(name_or_path, **kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return tokenizer, model
