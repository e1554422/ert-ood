"""末层/逐层 post-hoc 基线, 复用 banded_ood 的已审计实现。全部返回"越大越 ID"的分数。

  msp / energy     : 原生 1000 类头 logits
  maha_pp          : 末层 L2 归一化 + 类条件共享协方差 Mahalanobis (Müller & Hein 2025)
  relmaha          : Relative Mahalanobis (Ren et al. 2021), 未归一化
  knn              : 末层 L2 归一化 k-NN (Sun et al. 2022), k=10 (ImageNet 1% bank 惯例下的 Maha++ 文献设置)
  xmaha            : X-Maha 冻结特征混合 Mahalanobis 核心 (不含其 PEFT 微调)
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.special import logsumexp, softmax

sys.path.insert(0, r"D:\CHANGJIANG\banded_depth_ood\src")
sys.path.insert(0, r"D:\CHANGJIANG\banded_depth_ood\scripts")
from banded_ood.estimators import (  # noqa: E402
    SingleLayerMahalanobis,
    VarianceWeightedFeatureMixingMahalanobis,
)
from run_formal_knn import ExactKNN  # noqa: E402


def msp(logits: np.ndarray) -> np.ndarray:
    return softmax(logits.astype(np.float64), axis=1).max(1)


def energy(logits: np.ndarray) -> np.ndarray:
    return logsumexp(logits.astype(np.float64), axis=1)


def fit_baselines(z_fit: np.ndarray, y_fit: np.ndarray, device: str = "cuda", knn_k: int = 10) -> dict:
    """z_fit: [L, N, D]。返回已拟合估计器字典。"""
    last = z_fit[-1].astype(np.float64)
    return {
        "maha_pp": SingleLayerMahalanobis(normalize=True, score_device=device).fit(last, y_fit),
        "relmaha": SingleLayerMahalanobis(relative=True, score_device=device).fit(last, y_fit),
        "knn": ExactKNN(k=knn_k, device=device, batch_size=1024).fit(last),
        "xmaha": VarianceWeightedFeatureMixingMahalanobis(score_device=device).fit(
            [z.astype(np.float64) for z in z_fit], y_fit
        ),
    }


def score_baselines(fitted: dict, data: dict) -> dict[str, np.ndarray]:
    z = data["z"]
    out = {
        "maha_pp": fitted["maha_pp"].score_samples(z[-1].astype(np.float64)),
        "relmaha": fitted["relmaha"].score_samples(z[-1].astype(np.float64)),
        "knn": fitted["knn"].score_samples(z[-1].astype(np.float64)),
        "xmaha": fitted["xmaha"].score_samples([l.astype(np.float64) for l in z]),
    }
    if "logits" in data:
        out["msp"] = msp(data["logits"])
        out["energy"] = energy(data["logits"])
    return out
