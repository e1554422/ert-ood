"""逐层特征加载 + OpenOOD v1.5 ImageNet-1K 切分 + ID-only 拟合/校准划分。

所有拟合与选择步骤只能读 ``FIT_ALLOWED`` 中的集合; 读 OOD 集合必须经 ``load_eval``,
并由 ``OODGuard`` 记录, 便于在 tests 中断言"选择阶段未触碰 OOD"。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ERT_ROOT = Path(__file__).resolve().parents[2]
BANDED_FEATURES = Path(r"D:\CHANGJIANG\banded_depth_ood\artifacts\features")
ID_FEATURES = ERT_ROOT / "runs" / "features" / "vit_augreg"
MANIFESTS = ERT_ROOT / "runs" / "manifests"

NEAR_OOD = {"ssb_hard": "ssb_hard", "ninco": "far_ninco"}
FAR_OOD = {"inaturalist": "far_inaturalist", "textures": "far_texture", "openimage_o": "far_openimage_o"}
EXPECTED_COUNTS = {"ssb_hard": 49000, "ninco": 5879, "inaturalist": 10000, "textures": 5160, "openimage_o": 15869}
N_LAYERS = 12


def _load_dir(path: Path, layers: range | list[int] | None = None) -> dict[str, np.ndarray]:
    layers = range(N_LAYERS) if layers is None else layers
    out = {"z": np.stack([np.load(path / f"layer_{l:02d}.npy").astype(np.float32) for l in layers])}
    for name in ("labels", "logits"):
        f = path / f"{name}.npy"
        if f.exists():
            out[name] = np.load(f)
    return out


@dataclass
class OODGuard:
    """记录 OOD 读取; 进入 ``selection`` 上下文时读取 OOD 直接报错。"""

    accessed: list[str] = field(default_factory=list)
    selecting: bool = False

    def touch(self, name: str) -> None:
        if self.selecting:
            raise RuntimeError(f"OOD set '{name}' accessed during an ID-only selection step")
        self.accessed.append(name)

    def __enter__(self):
        self.selecting = True
        return self

    def __exit__(self, *exc):
        self.selecting = False


GUARD = OODGuard()


def load_id_train() -> dict[str, np.ndarray]:
    """100k ImageNet 训练子集 (ID-only 拟合池)。z: [L, N, D]。"""
    return _load_dir(ID_FEATURES / "train100k")


def load_id_eval() -> dict[str, dict[str, np.ndarray]]:
    """OpenOOD ID val(5k) / test(45k), 从 val50k 中按官方列表切出。"""
    full = _load_dir(ID_FEATURES / "val50k")
    splits = json.loads((MANIFESTS / "openood_id_splits.json").read_text())
    return {
        name: {k: (v[:, idx] if k == "z" else v[idx]) for k, v in full.items()}
        for name, idx in ((n, np.asarray(i)) for n, i in splits.items())
    }


def load_ood(name: str) -> dict[str, np.ndarray]:
    GUARD.touch(name)
    folder = {**NEAR_OOD, **FAR_OOD}[name]
    data = _load_dir(BANDED_FEATURES / folder)
    n = data["z"].shape[1]
    if n != EXPECTED_COUNTS[name]:
        raise ValueError(f"{name}: expected {EXPECTED_COUNTS[name]} samples, found {n}")
    return data


def fit_calib_split(labels: np.ndarray, seed: int, calib_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """按类分层把 ID 拟合池切成 F(拟合) / C(校准)。"""
    rng = np.random.default_rng(seed)
    fit, calib = [], []
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        k = int(round(len(idx) * calib_frac))
        calib.append(idx[:k])
        fit.append(idx[k:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(calib))


def kfold_ids(labels: np.ndarray, k: int, seed: int) -> np.ndarray:
    """按类分层的 K 折编号, 用于 EDL 头交叉拟合 (OOF 证据)。"""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(labels), dtype=np.int64)
    for c in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == c))
        fold[idx] = np.arange(len(idx)) % k
    return fold


def per_class_subset(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """De la Jara 同口径的 50/类 拟合子集。"""
    rng = np.random.default_rng(seed)
    idx = [rng.permutation(np.flatnonzero(labels == c))[:per_class] for c in np.unique(labels)]
    return np.sort(np.concatenate(idx))
