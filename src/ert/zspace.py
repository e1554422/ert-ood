"""视觉表征空间轨迹 (De la Jara et al. 2026 §4.1 复现; 亦即核心消融 A 的"表征基线")。

类路线 μ_c^l = E[z^l | Y=c]; 测试时以末层 Maha++ 最近类 ĉ 锚定一次, r^l = z^l - μ_ĉ^l。
"""
from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def maha_nearest_class(estimator, last: np.ndarray, device: str = "cuda", batch: int = 4096) -> np.ndarray:
    """用已拟合的 SingleLayerMahalanobis(normalize=True) 的几何求末层最近类。"""
    mu = torch.as_tensor(estimator.means_, device=device, dtype=torch.float64)
    P = torch.as_tensor(estimator.precision_, device=device, dtype=torch.float64)
    muP = mu @ P
    mu_term = (muP * mu).sum(1)
    out = np.empty(last.shape[0], dtype=np.int64)
    for s in range(0, last.shape[0], batch):
        x = torch.as_tensor(last[s : s + batch], device=device, dtype=torch.float64)
        if estimator.normalize:
            x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        d = mu_term[None] - 2 * x @ muP.T  # x^T P x 对所有类相同, 省略
        out[s : s + batch] = d.argmin(1).cpu().numpy()
    return estimator.classes_[out]


def class_means(z: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """z: [L, N, D] -> μ: [L, C, D] (float32)。"""
    L, N, D = z.shape
    mu = np.zeros((L, n_classes, D), dtype=np.float64)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    for l in range(L):
        np.add.at(mu[l], y, z[l].astype(np.float64))
    return (mu / np.maximum(counts, 1)[None, :, None]).astype(np.float32)


def residuals(z: np.ndarray, mu: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    return (z.astype(np.float32) - mu[:, anchors]).astype(np.float16)
