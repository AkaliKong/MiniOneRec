import math

import torch

from history_compression import HistoryCompressionConfig, format_compressed_history, split_history
from metrics_utils import binary_auc, binary_logloss, ranking_metrics
from models.ctr_head import CTRHead


def test_ctr_head_forward_backward():
    head = CTRHead(hidden_size=8, dropout=0.0)
    hidden = torch.randn(4, 8, requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    logits = head(hidden).squeeze(-1)
    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (4,)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()


def test_history_compression_correctness():
    history = [f"<sid_{i}>" for i in range(120)]
    long_hist, short_hist = split_history(history, threshold=100)

    assert len(long_hist) == 20
    assert len(short_hist) == 100

    cfg = HistoryCompressionConfig(
        use_history_compression=True,
        history_threshold=100,
        compression_type="attention",
    )
    formatted, meta = format_compressed_history(history, cfg)

    assert formatted.startswith("<compressed_attention:")
    assert ", <sid_20>" in formatted
    assert meta["long_history_len"] == 20
    assert meta["short_history_len"] == 100


def test_evaluation_ctr_metrics_outputs():
    y_true = [1, 0, 1, 0]
    y_prob = [0.9, 0.2, 0.7, 0.1]

    auc = binary_auc(y_true, y_prob)
    ll = binary_logloss(y_true, y_prob)
    ranks = ranking_metrics(
        predictions=[["a", "b", "c"], ["x", "y", "z"]],
        targets=["a", "q"],
        ks=[3, 10],
    )

    assert 0.0 <= auc <= 1.0
    assert ll > 0.0 and math.isfinite(ll)
    assert "HR@3" in ranks and "NDCG@10" in ranks
