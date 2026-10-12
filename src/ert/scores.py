"""打分与残差化融合 (论文 §打分与残差化融合)。

约定: 各分量与融合输出都是"越大越 OOD"的异常分数; 交给 metrics 前取负 (越大越 ID)。
"""
from __future__ import annotations

import numpy as np


def layer_aggregate(values: np.ndarray, layers: list[int] | np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Σ_l ω_l x^l, 只在 L_eff 上聚合; 默认均匀权重 ω_l = 1/|L_eff|。values: [L, N]。"""
    layers = np.asarray(layers)
    w = np.full(len(layers), 1.0 / len(layers)) if weights is None else np.asarray(weights)
    return (w[:, None] * values[layers]).sum(0)


def effective_layers(id_val_accuracy: np.ndarray, threshold: float = 0.05) -> list[int]:
    """L_eff: ID 准确率超过阈值的层 (纯 ID 规则, 末层恒保留), 取连续尾段以保证相邻转移有定义。"""
    ok = id_val_accuracy > threshold
    ok[-1] = True
    start = len(ok) - 1
    while start > 0 and ok[start - 1]:
        start -= 1
    return list(range(start, len(ok)))


class ResidualFusion:
    """按给定顺序逐项残差化叠加: 每个后加分量先在 ID 校准集上对已并入分量做 OLS, 只留残差。

    score = s̃_1 + Σ_{k≥2} w_k · (s̃_k - Σ_j a_kj · s̃_j^{res} - b_k),  s̃ 为校准集 z 标准化分量。
    第一个分量即基础项 (纯证据版为 U; 对照版为 Maha++)。
    """

    def __init__(self, order: list[str], weights: dict[str, float]):
        self.order = order
        self.weights = weights

    def _z(self, comps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: (comps[k] - self.mu[k]) / self.sd[k] for k in self.order}

    def fit(self, calib: dict[str, np.ndarray]) -> "ResidualFusion":
        self.mu = {k: float(np.mean(calib[k])) for k in self.order}
        self.sd = {k: float(np.std(calib[k]) + 1e-12) for k in self.order}
        z = self._z(calib)
        self.coef: dict[str, np.ndarray] = {}
        basis = [z[self.order[0]]]
        for k in self.order[1:]:
            X = np.column_stack([*basis, np.ones_like(basis[0])])
            beta, *_ = np.linalg.lstsq(X, z[k], rcond=None)
            self.coef[k] = beta
            basis.append(z[k] - X @ beta)
        return self

    def components(self, comps: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        z = self._z(comps)
        out = {self.order[0]: z[self.order[0]]}
        basis = [z[self.order[0]]]
        for k in self.order[1:]:
            X = np.column_stack([*basis, np.ones_like(basis[0])])
            res = z[k] - X @ self.coef[k]
            out[k] = res
            basis.append(res)
        return out

    def score(self, comps: dict[str, np.ndarray]) -> np.ndarray:
        parts = self.components(comps)
        s = parts[self.order[0]].copy()
        for k in self.order[1:]:
            s += self.weights[k] * parts[k]
        return s
