"""共享一步转移模型 + 转移意外度 (论文 §转移模型与转移意外度)。

f_θ(r^l + e^l) -> r^{l+1}: 两隐层 GELU MLP (hidden 1024), 深度嵌入加在第一层隐空间
(等价于在输入空间加 e 后的线性投影, 且更一般)。输出头三选一:
  nig   : 每维 (γ, ν, α, β), 损失 = Amini et al. 2020 NIG NLL + λ·|y-γ|(2ν+α);
          意外度 = 逐维 Student-t NLL 之和 (NIG 边缘分布)。
  mse   : 点预测 + MSE; 意外度 = Σ (r - r̂)² / (σ̂²_{l,d} + ε), σ̂² 为拟合集均方误差 (De la Jara 式 6)。
  gauss : 点预测 + 每 (转移, 维) 一个可学的同方差 log σ; 意外度 = Gaussian NLL 之和。
训练配方与 De la Jara Table D2 一致: AdamW lr 3e-4, wd 1e-5, 80 epoch, batch 512。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

LOG_PI = math.log(math.pi)


def nig_nll(y, gamma, nu, alpha, beta):
    """Amini et al. 2020 式 (8); 恰为 Student-t(2α, γ, β(1+ν)/(να)) 的负对数密度。"""
    omega = 2 * beta * (1 + nu)
    return (
        0.5 * (LOG_PI - torch.log(nu))
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def nig_epistemic(nu, alpha, beta):
    """Var[μ] = β / (ν(α-1))。"""
    return beta / (nu * (alpha - 1))


@dataclass
class TransitionConfig:
    head: str = "nig"
    hidden: int = 1024
    epochs: int = 80
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-5
    nig_reg: float = 0.01
    seed: int = 0
    device: str = "cuda"


class TransitionMLP(nn.Module):
    def __init__(self, dim: int, n_transitions: int, cfg: TransitionConfig):
        super().__init__()
        self.head = cfg.head
        self.inp = nn.Linear(dim, cfg.hidden)
        self.depth = nn.Embedding(n_transitions, cfg.hidden)
        self.mid = nn.Linear(cfg.hidden, cfg.hidden)
        n_out = {"nig": 4, "mse": 1, "gauss": 1}[cfg.head]
        self.out = nn.Linear(cfg.hidden, n_out * dim)
        if cfg.head == "gauss":
            self.log_sigma = nn.Parameter(torch.zeros(n_transitions, dim))
        self.dim = dim

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.gelu(self.inp(r) + self.depth(t))
        h = F.gelu(self.mid(h))
        o = self.out(h)
        if self.head != "nig":
            res = {"mu": o}
            if self.head == "gauss":
                res["log_sigma"] = self.log_sigma[t]
            return res
        g, n, a, b = o.split(self.dim, -1)
        return {"mu": g, "nu": F.softplus(n) + 1e-4, "alpha": F.softplus(a) + 1.0 + 1e-4, "beta": F.softplus(b) + 1e-4}


class TransitionModel:
    """在 [L, N, D] 的残差轨迹上训练共享一步转移模型并给出逐样本意外度。"""

    def __init__(self, cfg: TransitionConfig):
        self.cfg = cfg

    # -- 标准化: 每 (层, 维) 用拟合集统计量 --
    def _norm(self, r: torch.Tensor) -> torch.Tensor:
        return (r - self.mean) / self.std

    def fit(self, r: np.ndarray, log_every: int = 10) -> list[float]:
        cfg, dev = self.cfg, self.cfg.device
        torch.manual_seed(cfg.seed)
        rt = torch.from_numpy(r).to(dev).float()
        self.mean = rt.mean(1, keepdim=True)
        self.std = rt.std(1, keepdim=True) + 1e-4
        rt = self._norm(rt).half()
        L, N, D = rt.shape
        T = L - 1
        self.model = TransitionMLP(D, T, cfg).to(dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        gen = torch.Generator().manual_seed(cfg.seed)
        history = []
        for epoch in range(cfg.epochs):
            perm = torch.randperm(N * T, generator=gen).to(dev)
            total, count = 0.0, 0
            for idx in perm.split(cfg.batch_size):
                t, i = idx // N, idx % N
                x, y = rt[t, i].float(), rt[t + 1, i].float()
                out = self.model(x, t)
                loss = self._loss(out, y).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
                count += len(idx)
            history.append(total / count)
            if log_every and (epoch % log_every == 0 or epoch == cfg.epochs - 1):
                print(f"[transition/{cfg.head}] epoch {epoch} loss {history[-1]:.4f}", flush=True)
        self.model.eval()
        if cfg.head == "mse":  # σ̂²_{l,d}: 拟合集上的均方预测误差 + 1e-6
            self.sigma2 = self._sq_error(r).mean(1) + 1e-6  # [T, D]
        return history

    def _loss(self, out: dict, y: torch.Tensor) -> torch.Tensor:
        head = self.cfg.head
        if head == "nig":
            nll = nig_nll(y, out["mu"], out["nu"], out["alpha"], out["beta"])
            reg = (y - out["mu"]).abs() * (2 * out["nu"] + out["alpha"])
            return (nll + self.cfg.nig_reg * reg).sum(-1)
        if head == "mse":
            return ((y - out["mu"]) ** 2).sum(-1)
        return (0.5 * ((y - out["mu"]) / out["log_sigma"].exp()) ** 2 + out["log_sigma"]).sum(-1)

    @torch.no_grad()
    def _run(self, r: np.ndarray, batch: int = 2048):
        dev = self.cfg.device
        L, N, D = r.shape
        for s in range(0, N, batch):
            rt = self._norm(torch.from_numpy(r[:, s : s + batch]).to(dev).float())
            for t in range(L - 1):
                tt = torch.full((rt.shape[1],), t, device=dev, dtype=torch.long)
                yield s, t, rt[t + 1], self.model(rt[t], tt)

    @torch.no_grad()
    def _sq_error(self, r: np.ndarray) -> torch.Tensor:
        L, N, D = r.shape
        out = torch.empty(L - 1, N, D, device=self.cfg.device)
        for s, t, y, o in self._run(r):
            out[t, s : s + len(y)] = (y - o["mu"]) ** 2
        return out

    @torch.no_grad()
    def surprise(self, r: np.ndarray) -> dict[str, np.ndarray]:
        """逐样本意外度 D(x) 及逐转移分解 [T, N]; nig 头额外给认知不确定性均值。"""
        L, N, D = r.shape
        per_t = np.zeros((L - 1, N), dtype=np.float64)
        epi = np.zeros((L - 1, N), dtype=np.float64) if self.cfg.head == "nig" else None
        head = self.cfg.head
        for s, t, y, o in self._run(r):
            if head == "nig":
                v = nig_nll(y, o["mu"], o["nu"], o["alpha"], o["beta"]).sum(-1)
                epi[t, s : s + len(y)] = nig_epistemic(o["nu"], o["alpha"], o["beta"]).mean(-1).cpu().numpy()
            elif head == "mse":
                v = ((y - o["mu"]) ** 2 / self.sigma2[t]).sum(-1)
            else:
                v = (0.5 * ((y - o["mu"]) / o["log_sigma"].exp()) ** 2 + o["log_sigma"]).sum(-1)
            per_t[t, s : s + len(y)] = v.cpu().numpy()
        res = {"D": per_t.sum(0), "per_transition": per_t}
        if epi is not None:
            res["epistemic"] = epi.mean(0)
        return res
