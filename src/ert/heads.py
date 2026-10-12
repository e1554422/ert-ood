"""逐层 EDL 分类头 (论文 §每层的 EDL 分类头)。

e = Softplus(a), α = e + 1。12 层的线性头作为一个批量张量 [L, D, C] 同时训练。
K 折交叉拟合给出拟合集上的 out-of-fold 证据, 避免头在自己的训练样本上过度自信
而污染类路线 ν_c 与转移模型; 对校准/测试集取 K 个折模型 α 的平均。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def kl_dirichlet_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL[Dir(α) || Dir(1)], 最后一维为类别。"""
    c = alpha.shape[-1]
    s = alpha.sum(-1)
    return (
        torch.lgamma(s)
        - math.lgamma(c)
        - torch.lgamma(alpha).sum(-1)
        + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(s)[..., None])).sum(-1)
    )


def edl_loss(alpha: torch.Tensor, y: torch.Tensor, kind: str, kl_weight: float) -> torch.Tensor:
    """Sensoy et al. 2018 的三种 EDL 数据项 + 退火 KL(去掉正确类证据后的 α̃)。

    alpha: [..., N, C]; y: [N] 类别索引。返回标量均值。
    """
    onehot = F.one_hot(y, alpha.shape[-1]).to(alpha.dtype)
    s = alpha.sum(-1, keepdim=True)
    if kind == "mse":
        p = alpha / s
        data = ((onehot - p) ** 2 + p * (1 - p) / (s + 1)).sum(-1)
    elif kind == "digamma":
        data = (onehot * (torch.digamma(s) - torch.digamma(alpha))).sum(-1)
    elif kind == "log":
        data = (onehot * (torch.log(s) - torch.log(alpha))).sum(-1)
    else:
        raise ValueError(kind)
    alpha_tilde = onehot + (1 - onehot) * alpha
    return (data + kl_weight * kl_dirichlet_uniform(alpha_tilde)).mean()


class LayerwiseLinearEDL(nn.Module):
    """L 个独立线性头的批量实现, 输入按层标准化。"""

    def __init__(self, n_layers: int, dim: int, n_classes: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_layers, dim, n_classes) * dim**-0.5 * 0.01)
        self.bias = nn.Parameter(torch.zeros(n_layers, 1, n_classes))
        self.register_buffer("mean", mean)  # [L,1,D]
        self.register_buffer("std", std)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [L, B, D] -> log α: [L, B, C]  (log α = log(softplus(a)+1), 数值稳定)。"""
        a = torch.baddbmm(self.bias, (z - self.mean) / self.std, self.weight)
        return torch.log1p(F.softplus(a))


@dataclass
class HeadConfig:
    loss: str = "digamma"
    epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    anneal_epochs: int = 10
    kl_coef: float = 1.0
    n_folds: int = 5
    seed: int = 0
    device: str = "cuda"


def _train_one(z: torch.Tensor, y: torch.Tensor, n_classes: int, cfg: HeadConfig) -> LayerwiseLinearEDL:
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    mean = z.mean(1, keepdim=True)
    std = z.std(1, keepdim=True) + 1e-6
    model = LayerwiseLinearEDL(z.shape[0], z.shape[2], n_classes, mean, std).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = z.shape[1]
    steps = cfg.epochs * math.ceil(n / cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=steps, pct_start=0.1)
    for epoch in range(cfg.epochs):
        kl_w = cfg.kl_coef * min(1.0, epoch / max(cfg.anneal_epochs, 1))
        for idx in torch.randperm(n, generator=gen).split(cfg.batch_size):
            idx = idx.to(cfg.device)
            log_alpha = model(z[:, idx].float())
            loss = edl_loss(log_alpha.exp(), y[idx], cfg.loss, kl_w) * z.shape[0]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def _predict(models: list[LayerwiseLinearEDL], z: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    """K 个模型 α 的算术平均, 返回 float16 log α [L, N, C]。"""
    out = []
    for s in range(0, z.shape[1], batch):
        zb = torch.from_numpy(z[:, s : s + batch]).to(device).float()
        la = torch.stack([m(zb) for m in models])  # [K,L,B,C]
        out.append((torch.logsumexp(la, 0) - math.log(len(models))).half().cpu())
    return torch.cat(out, 1).numpy()


class CrossFitEDL:
    """K 折交叉拟合的逐层 EDL 头。"""

    def __init__(self, cfg: HeadConfig):
        self.cfg = cfg
        self.models: list[LayerwiseLinearEDL] = []

    def fit(self, z: np.ndarray, y: np.ndarray, folds: np.ndarray, n_classes: int = 1000) -> np.ndarray:
        """在拟合集上训练 K 个折模型, 返回拟合集的 OOF log α [L, N, C] (float16)。"""
        dev = self.cfg.device
        zt = torch.from_numpy(z).to(dev).half()
        yt = torch.from_numpy(y.astype(np.int64)).to(dev)
        oof = np.empty((z.shape[0], z.shape[1], n_classes), dtype=np.float16)
        for k in range(self.cfg.n_folds):
            tr = np.flatnonzero(folds != k)
            te = np.flatnonzero(folds == k)
            torch.manual_seed(self.cfg.seed * 100 + k)
            cfg = HeadConfig(**{**self.cfg.__dict__, "seed": self.cfg.seed * 100 + k})
            m = _train_one(zt[:, torch.from_numpy(tr).to(dev)], yt[torch.from_numpy(tr).to(dev)], n_classes, cfg)
            self.models.append(m)
            oof[:, te] = _predict([m], z[:, te], dev)
        return oof

    def predict(self, z: np.ndarray) -> np.ndarray:
        return _predict(self.models, z, self.cfg.device)
