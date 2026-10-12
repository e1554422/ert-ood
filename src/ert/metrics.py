"""OOD 指标 (复用 banded_ood.metrics) + 宏平均 + 配对 bootstrap。

约定: 分数越大越像 ID (与 OpenOOD 一致)。
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r"D:\CHANGJIANG\banded_depth_ood\src")
from banded_ood.metrics import ood_metrics  # noqa: E402

GROUPS = {"near": ("ssb_hard", "ninco"), "far": ("inaturalist", "textures", "openimage_o")}


def evaluate(id_scores: np.ndarray, ood_scores: dict[str, np.ndarray]) -> dict:
    per = {k: ood_metrics(id_scores, v) for k, v in ood_scores.items()}
    out = {"per_dataset": per}
    for g, names in GROUPS.items():
        present = [n for n in names if n in per]
        if present:
            out[g] = {m: float(np.mean([per[n][m] for n in present])) for m in ("auroc", "fpr95")}
    return out


def _fpr95(id_s: np.ndarray, ood_s: np.ndarray) -> float:
    thr = np.quantile(id_s, 0.05)
    return float(np.mean(ood_s >= thr))


def _auroc(id_s: np.ndarray, ood_s: np.ndarray) -> float:
    from scipy.stats import rankdata

    r = rankdata(np.concatenate([id_s, ood_s]))
    n1, n0 = len(id_s), len(ood_s)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def paired_bootstrap(
    id_a: np.ndarray, ood_a: np.ndarray, id_b: np.ndarray, ood_b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, list[float]]:
    """方法 A 相对 B 的 ΔAUROC / ΔFPR95 的 95% CI (ID 与 OOD 样本各自重采样, A/B 共享索引)。"""
    rng = np.random.default_rng(seed)
    d_auc, d_fpr = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(id_a), len(id_a))
        o = rng.integers(0, len(ood_a), len(ood_a))
        d_auc.append(_auroc(id_a[i], ood_a[o]) - _auroc(id_b[i], ood_b[o]))
        d_fpr.append(_fpr95(id_a[i], ood_a[o]) - _fpr95(id_b[i], ood_b[o]))
    q = lambda v: [float(np.mean(v)), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {"d_auroc": q(d_auc), "d_fpr95": q(d_fpr)}
