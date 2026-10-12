"""证据状态、类别路线、锚定与证据残差 (论文 §证据状态 / §类别路线与证据残差)。

所有函数在 [L, N, C] 的 log α 上操作 (float16 存储, 计算时转 float32)。
"""
from __future__ import annotations

import numpy as np
import torch


def _t(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(x)).to(device).float()


def summaries(log_alpha: np.ndarray, device: str = "cuda", batch: int = 8192) -> dict[str, np.ndarray]:
    """逐层 S, 空缺度 u=C/S, 类别熵 H, 预测类与最大概率。返回各 [L, N]。"""
    L, N, C = log_alpha.shape
    keys = ("log_S", "u", "H", "pred", "pmax")
    out = {k: np.empty((L, N), dtype=np.int64 if k == "pred" else np.float32) for k in keys}
    for s in range(0, N, batch):
        la = _t(log_alpha[:, s : s + batch], device)
        log_s = torch.logsumexp(la, -1)
        logp = la - log_s[..., None]
        p = logp.exp()
        out["log_S"][:, s : s + batch] = log_s.cpu().numpy()
        out["u"][:, s : s + batch] = torch.exp(np.log(C) - log_s).cpu().numpy()
        out["H"][:, s : s + batch] = (-(p * logp).sum(-1)).cpu().numpy()
        pm, pr = p.max(-1)
        out["pmax"][:, s : s + batch] = pm.cpu().numpy()
        out["pred"][:, s : s + batch] = pr.cpu().numpy()
    return out


def evidence_state(log_alpha: torch.Tensor, kind: str = "full") -> torch.Tensor:
    """v^l。full: [log α_1..log α_C, log S] (C+1 维); compact: [p, u] (C+1 维)。"""
    log_s = torch.logsumexp(log_alpha, -1, keepdim=True)
    if kind == "full":
        return torch.cat([log_alpha, log_s], -1)
    if kind == "compact":
        c = log_alpha.shape[-1]
        return torch.cat([(log_alpha - log_s).exp(), c * torch.exp(-log_s)], -1)
    raise ValueError(kind)


def class_routes(log_alpha: np.ndarray, y: np.ndarray, n_classes: int, kind: str = "full",
                 device: str = "cuda", batch: int = 8192) -> np.ndarray:
    """ν_c^l = E[v^l | Y=c]  -> [L, C, C+1] (float32)。"""
    L, N, C = log_alpha.shape
    sums = torch.zeros(L, n_classes, C + 1, device=device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    for s in range(0, N, batch):
        v = evidence_state(_t(log_alpha[:, s : s + batch], device), kind)
        sums.index_add_(1, yt[s : s + batch], v)
    counts = torch.bincount(yt, minlength=n_classes).clamp_min(1).float()
    return (sums / counts[None, :, None]).cpu().numpy()


def anchor_residuals(log_alpha: np.ndarray, routes: np.ndarray, mode: str = "hard", kind: str = "full",
                     device: str = "cuda", batch: int = 4096, out_dtype=np.float16) -> tuple[np.ndarray, np.ndarray]:
    """末层锚定一次 ĉ 后沿用其整条路线: r^l = v^l - ν_ĉ^l (hard) 或 v^l - Σ_c p_c^L ν_c^l (soft)。

    返回 (r [L, N, C+1], ĉ [N])。
    """
    L, N, C = log_alpha.shape
    nu = _t(routes, device)
    r = np.empty((L, N, C + 1), dtype=out_dtype)
    anchors = np.empty(N, dtype=np.int64)
    for s in range(0, N, batch):
        la = _t(log_alpha[:, s : s + batch], device)
        v = evidence_state(la, kind)
        p_last = torch.softmax(la[-1], -1)  # α^L / S^L
        c_hat = p_last.argmax(-1)
        if mode == "hard":
            ref = nu[:, c_hat]  # [L,B,C+1]
        elif mode == "soft":
            ref = torch.einsum("bc,lcd->lbd", p_last, nu)
        else:
            raise ValueError(mode)
        r[:, s : s + batch] = (v - ref).cpu().numpy().astype(out_dtype)
        anchors[s : s + batch] = c_hat.cpu().numpy()
    return r, anchors
