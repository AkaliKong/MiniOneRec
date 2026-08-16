import pathlib
import sys

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rq"))

from models.rqvae import RQVAE  # noqa: E402
from models.vq import VectorQuantizer  # noqa: E402


def _quantizer(**kwargs):
    quantizer = VectorQuantizer(
        n_e=4,
        e_dim=2,
        kmeans_init=False,
        enable_balance_loss=True,
        balance_temperature=0.05,
        **kwargs,
    )
    with torch.no_grad():
        quantizer.embedding.weight.copy_(
            torch.tensor(
                [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]
            )
        )
    return quantizer


def test_balance_loss_penalizes_collapsed_assignments():
    quantizer = _quantizer()
    uniform_batch = quantizer.embedding.weight.detach().clone()
    quantizer(uniform_batch, use_sk=False)
    uniform_loss = quantizer.balance_loss.item()

    collapsed_batch = uniform_batch[0].repeat(8, 1)
    quantizer(collapsed_batch, use_sk=False)
    collapsed_loss = quantizer.balance_loss.item()

    assert uniform_loss < 1e-4
    assert collapsed_loss > 1.0


def test_balance_loss_has_gradients():
    quantizer = _quantizer()
    inputs = torch.tensor([[-0.8, -0.7], [-0.9, -0.6]], requires_grad=True)
    quantizer(inputs, use_sk=False)
    quantizer.balance_loss.backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert quantizer.embedding.weight.grad is not None


def test_dead_codes_are_reinitialized_from_residuals():
    quantizer = _quantizer(
        usage_ema_decay=0.0,
        dead_code_threshold=0.2,
        dead_code_patience=2,
        dead_code_jitter=0.0,
    )
    original_weights = quantizer.embedding.weight.detach().clone()
    collapsed_batch = torch.tensor([[-1.0, -1.0]]).repeat(8, 1)

    quantizer(collapsed_batch, use_sk=False)
    quantizer(collapsed_batch, use_sk=False)
    quantizer(collapsed_batch, use_sk=False)

    assert quantizer.last_reset_count == 3
    assert not torch.equal(original_weights[1:], quantizer.embedding.weight[1:])
    assert torch.equal(
        quantizer.inactive_steps[1:], torch.ones(3, dtype=torch.long)
    )


def test_dead_code_recovery_does_not_break_backward():
    quantizer = _quantizer(
        usage_ema_decay=0.0,
        dead_code_threshold=0.2,
        dead_code_patience=1,
        dead_code_jitter=0.0,
    )
    collapsed_batch = torch.tensor(
        [[-1.0, -1.0]], requires_grad=True
    ).repeat(8, 1)
    output, quantization_loss, _ = quantizer(
        collapsed_batch, use_sk=False
    )
    (output.mean() + quantization_loss + quantizer.balance_loss).backward()

    second_output, second_quantization_loss, _ = quantizer(
        collapsed_batch.detach(), use_sk=False
    )
    (
        second_output.mean()
        + second_quantization_loss
        + quantizer.balance_loss
    ).backward()
    assert quantizer.last_reset_count == 3


def test_eval_does_not_change_training_usage_ema():
    quantizer = _quantizer(usage_ema_decay=0.0)
    quantizer.train()
    quantizer(torch.tensor([[-1.0, -1.0]]).repeat(4, 1), use_sk=False)
    training_usage = quantizer.ema_usage.clone()

    quantizer.eval()
    quantizer(torch.tensor([[1.0, 1.0]]).repeat(4, 1), use_sk=False)

    assert torch.equal(training_usage, quantizer.ema_usage)


def test_rqvae_total_loss_includes_balance_penalty():
    model = RQVAE(
        in_dim=2,
        num_emb_list=[4],
        e_dim=2,
        layers=[],
        sk_epsilons=[0.0],
        balance_loss_weight=0.5,
        balance_temperature=0.05,
    )
    model.train()
    inputs = torch.tensor([[-1.0, -1.0]]).repeat(8, 1)
    output, quant_loss, _ = model(inputs, use_sk=False)
    total_loss, reconstruction_loss = model.compute_loss(
        output, quant_loss, xs=inputs
    )
    expected = (
        reconstruction_loss
        + model.quant_loss_weight * quant_loss
        + model.balance_loss_weight * model.rq.get_balance_loss()
    )

    assert torch.allclose(total_loss, expected)
    assert model.rq.get_balance_loss().item() >= 0


def test_usage_metrics_are_bounded():
    quantizer = _quantizer()
    quantizer(torch.tensor([[-1.0, -1.0]]).repeat(4, 1), use_sk=False)
    metrics = quantizer.get_usage_metrics()

    assert 0.0 <= metrics["batch_utilization"] <= 1.0
    assert 1.0 <= metrics["ema_perplexity"] <= 4.0
    assert 0.0 <= metrics["ema_max_share"] <= 1.0
