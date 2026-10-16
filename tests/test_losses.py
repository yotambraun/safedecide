import torch

from safedecide.losses import brier, conftr_size, rps, soft_ce, total_loss

NEG = -1e4


def _batch():
    logits = torch.tensor([[2.0, 0.0, NEG], [0.0, 1.0, 3.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.2, 0.8]])
    return logits, mask, target


def test_soft_ce_ignores_padding():
    logits, mask, target = _batch()
    loss = soft_ce(logits, mask, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad[0, 2] == 0  # padded slot gets no gradient


def test_brier_zero_for_perfect_prediction():
    logits = torch.tensor([[20.0, -20.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    target = torch.tensor([[1.0, 0.0]])
    assert brier(logits, mask, target) < 1e-6


def test_rps_only_on_ordinal_rows():
    logits, mask, target = _batch()
    assert rps(logits, mask, target, torch.tensor([False, False])) == 0
    assert rps(logits, mask, target, torch.tensor([False, True])) > 0


def test_rps_respects_order():
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = torch.tensor([[0, 0, 0, 0, 1.0]])
    near = torch.tensor([[NEG, NEG, NEG, 5.0, NEG]])
    far = torch.tensor([[5.0, NEG, NEG, NEG, NEG]])
    ordinal = torch.tensor([True])
    assert rps(near, mask, target, ordinal) < rps(far, mask, target, ordinal)


def test_conftr_differentiable_and_rewards_sharp_models():
    torch.manual_seed(0)
    b, k = 64, 4
    labels = torch.randint(0, k, (b,))
    mask = torch.ones(b, k, dtype=torch.bool)
    sharp = torch.full((b, k), -3.0)
    sharp[torch.arange(b), labels] = 3.0
    flat = torch.zeros(b, k, requires_grad=True)
    loss_flat = conftr_size(flat, mask, labels, alpha=0.1)
    loss_flat.backward()
    assert flat.grad is not None
    torch.manual_seed(0)
    assert conftr_size(sharp, mask, labels, alpha=0.1) < loss_flat.detach()


def test_conftr_skips_tiny_batches():
    logits = torch.zeros(4, 3)
    assert conftr_size(logits, torch.ones(4, 3, dtype=torch.bool), torch.zeros(4, dtype=torch.long)) == 0


def test_total_loss_parts():
    logits, mask, target = _batch()
    loss, parts = total_loss(logits, mask, target, torch.tensor([False, True]), torch.tensor([0, 2]), w_conftr=0.1)
    assert set(parts) == {"ce", "brier", "rps", "conftr"}
    assert torch.isfinite(loss)
