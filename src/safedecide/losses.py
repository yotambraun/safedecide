"""Training losses.

* ``soft_ce``   cross-entropy against the teacher's soft distribution (= KL + const)
* ``brier``     multi-class Brier score: a strictly proper scoring rule that
                punishes over-confidence harder than log-loss near 0/1
* ``rps``       ranked probability score for ordinal ``score`` questions:
                predicting 4 when the answer is 5 costs less than predicting 0
* ``conftr_size``  Conformal Training (Stutz et al., ICLR 2022): simulate split
                conformal prediction inside each batch and penalise the size of
                the resulting prediction sets, so the model learns to produce
                more singleton sets (= more decisions it can automate).

All losses take padded ``logits [B,K]`` + ``mask [B,K]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    z = logits.float().masked_fill(~mask, float("-inf"))
    return F.log_softmax(z, dim=-1)


def soft_ce(logits, mask, target) -> torch.Tensor:
    logp = _masked_log_softmax(logits, mask).masked_fill(~mask, 0.0)
    return -(target * logp).sum(-1).mean()


def brier(logits, mask, target) -> torch.Tensor:
    p = _masked_log_softmax(logits, mask).exp().masked_fill(~mask, 0.0)
    return ((p - target) ** 2).masked_fill(~mask, 0.0).sum(-1).mean()


def rps(logits, mask, target, is_ordinal: torch.Tensor) -> torch.Tensor:
    """Ranked probability score, averaged over ordinal rows only."""
    if not bool(is_ordinal.any()):
        return logits.new_zeros(())
    p = _masked_log_softmax(logits, mask).exp().masked_fill(~mask, 0.0)
    cdf_p = p.cumsum(-1)
    cdf_t = target.cumsum(-1)
    k = mask.sum(-1).clamp(min=2).float()
    per_row = ((cdf_p - cdf_t) ** 2).masked_fill(~mask, 0.0).sum(-1) / (k - 1)
    return per_row[is_ordinal].mean()


def conftr_size(
    logits: torch.Tensor,
    mask: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.05,
    sigmoid_temperature: float = 0.1,
    target_size: float = 1.0,
    cal_fraction: float = 0.5,
) -> torch.Tensor:
    """ConfTr size loss with the THR (LAC) conformal predictor.

    1. Split the batch into a calibration half and a prediction half.
    2. On the calibration half, conformity = probability of the true option;
       the threshold tau is its alpha-quantile (finite-sample corrected).
    3. On the prediction half, soft set membership = sigmoid((p_k - tau)/T);
       penalise expected set size above ``target_size``.

    Gradients flow through both the threshold and the memberships, as in the
    paper. Returns 0 for batches too small to split meaningfully.
    """
    b = logits.size(0)
    if b < 8:
        return logits.new_zeros(())
    p = _masked_log_softmax(logits, mask).exp().masked_fill(~mask, 0.0)
    perm = torch.randperm(b, device=logits.device)
    n_cal = max(2, int(b * cal_fraction))
    cal, pred = perm[:n_cal], perm[n_cal:]
    conformity = p[cal, labels[cal]]
    level = min(1.0, alpha * (1.0 + 1.0 / n_cal))
    tau = torch.quantile(conformity, level)
    membership = torch.sigmoid((p[pred] - tau) / sigmoid_temperature)
    membership = membership.masked_fill(~mask[pred], 0.0)
    size = membership.sum(-1)
    return F.relu(size - target_size).mean()


def total_loss(
    logits,
    mask,
    target,
    is_ordinal,
    labels,
    w_brier: float = 0.5,
    w_rps: float = 0.5,
    w_conftr: float = 0.0,
    alpha: float = 0.05,
    conftr_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts = {
        "ce": soft_ce(logits, mask, target),
        "brier": brier(logits, mask, target),
        "rps": rps(logits, mask, target, is_ordinal),
    }
    loss = parts["ce"] + w_brier * parts["brier"] + w_rps * parts["rps"]
    if w_conftr > 0:
        parts["conftr"] = conftr_size(logits, mask, labels, alpha, conftr_temperature)
        loss = loss + w_conftr * parts["conftr"]
    return loss, {k: float(v.detach()) for k, v in parts.items()}
