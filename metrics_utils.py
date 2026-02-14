"""Ranking and CTR metric utilities."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _safe_clip_probs(y_prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.clip(y_prob, eps, 1.0 - eps)


def binary_logloss(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_prob_arr = _safe_clip_probs(np.asarray(y_prob, dtype=np.float64))
    if y_true_arr.size == 0:
        return float("nan")
    return float(-np.mean(y_true_arr * np.log(y_prob_arr) + (1.0 - y_true_arr) * np.log(1.0 - y_prob_arr)))


def binary_auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_score_arr = np.asarray(y_score, dtype=np.float64)

    pos = np.sum(y_true_arr == 1)
    neg = np.sum(y_true_arr == 0)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(y_score_arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score_arr) + 1)

    pos_ranks = ranks[y_true_arr == 1]
    auc = (np.sum(pos_ranks) - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def ranking_metrics(predictions: Sequence[Sequence[str]], targets: Sequence[str], ks: Iterable[int]) -> Dict[str, float]:
    ks_list = sorted(set(int(k) for k in ks))
    hr = {k: 0.0 for k in ks_list}
    ndcg = {k: 0.0 for k in ks_list}

    n = max(len(predictions), 1)
    for pred, target in zip(predictions, targets):
        target_clean = str(target).strip(' \n\"')
        pred_clean = [str(x).strip(' \n\"') for x in pred]

        hit_idx = None
        for idx, cand in enumerate(pred_clean):
            if cand == target_clean:
                hit_idx = idx
                break

        if hit_idx is None:
            continue

        for k in ks_list:
            if hit_idx < k:
                hr[k] += 1.0
                ndcg[k] += 1.0 / math.log2(hit_idx + 2)

    out: Dict[str, float] = {}
    for k in ks_list:
        out[f"HR@{k}"] = hr[k] / n
        out[f"NDCG@{k}"] = ndcg[k] / n
    return out


def ctr_metrics(y_true: Sequence[float], y_prob: Sequence[float]) -> Dict[str, float]:
    return {
        "CTR_AUC": binary_auc(y_true, y_prob),
        "CTR_LogLoss": binary_logloss(y_true, y_prob),
    }
