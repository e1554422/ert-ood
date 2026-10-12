"""2026-10-12 · P0 · 基线复现 + De la Jara 表征轨迹复现 (闸门 G0)。

目的: 在 ViT-B/16 AugReg (timm augreg_in21k_ft_in1k) + OpenOOD v1.5 ImageNet-1K 上,
按 De la Jara 同口径 (每类 50 拟合 / 30 校准, 三种子) 复现
MSP / Energy / Maha++ / RelMaha / kNN / X-Maha, 以及 z 空间 path-surprise
(类路线残差 + 共享两隐层 MLP + MSE/σ̂² + ID 残差化 + 固定 0.3 融合到 Maha++)。
对照原文表 P5 的 ViT-B/16-augreg 行 (AUROC/FPR95, 五数据集宏平均):
  MSP 87.5/46.5 · kNN 87.8/50.1 · Maha++ 92.7/27.8 (+path 93.8/23.8) · RelMaha 92.0/30.9 · X-Maha 94.3/22.1
闸门 G0: Maha++ 两指标均在 ±1.0 内, 且 +path 使 FPR95 下降。
结论: G0 通过。宏平均 AUROC/FPR95: Maha++ 92.9/27.5 (原文 92.7/27.8), +path 93.9/23.9 (93.8/23.8), X-Maha 94.4/21.8 (94.3/22.1), RelMaha 92.5/30.4, kNN 88.9/47.6, MSP 88.0/45.7。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ert import data as D  # noqa: E402
from ert.baselines import fit_baselines, score_baselines  # noqa: E402
from ert.metrics import evaluate  # noqa: E402
from ert.scores import ResidualFusion  # noqa: E402
from ert.transition import TransitionConfig, TransitionModel  # noqa: E402
from ert.zspace import class_means, maha_nearest_class, residuals  # noqa: E402

PAPER = {"msp": (87.5, 46.5), "knn": (87.8, 50.1), "maha_pp": (92.7, 27.8), "maha_pp+path": (93.8, 23.8),
         "relmaha": (92.0, 30.9), "xmaha": (94.3, 22.1)}
OOD_SETS = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]


def fit_calib_per_class(y: np.ndarray, n_fit: int, n_cal: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit, cal = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        k = min(n_fit, len(idx) - 1)
        fit.append(idx[:k])
        cal.append(idx[k : k + n_cal])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(cal))


def macro_all(res: dict) -> tuple[float, float]:
    per = res["per_dataset"]
    return (100 * np.mean([per[n]["auroc"] for n in OOD_SETS]), 100 * np.mean([per[n]["fpr95"] for n in OOD_SETS]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-fit", type=int, default=50)
    ap.add_argument("--n-cal", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--tag", default="parity50")
    args = ap.parse_args()
    out_dir = ROOT / "results" / "p0"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train = D.load_id_train()
    id_eval = D.load_id_eval()
    test = id_eval["id_test"]
    oods = {n: D.load_ood(n) for n in OOD_SETS}
    print(f"loaded in {time.time() - t0:.0f}s; train {train['z'].shape}, test {test['z'].shape}", flush=True)
    native_acc = float((test["logits"].argmax(1) == test["labels"]).mean())
    print(f"native top-1 on id_test: {native_acc:.4f}", flush=True)

    all_runs = []
    for seed in args.seeds:
        fit_idx, cal_idx = fit_calib_per_class(train["labels"], args.n_fit, args.n_cal, seed)
        y_fit = train["labels"][fit_idx]
        z_fit = train["z"][:, fit_idx]
        z_cal = {"z": train["z"][:, cal_idx]}
        print(f"[seed {seed}] fit {len(fit_idx)} calib {len(cal_idx)}", flush=True)

        with D.GUARD:  # 拟合阶段禁止读 OOD
            fitted = fit_baselines(z_fit, y_fit)
        sets = {"id_test": test, **oods}
        scores = {name: score_baselines(fitted, d) for name, d in sets.items()}
        cal_scores = score_baselines(fitted, z_cal)

        # --- De la Jara path surprise in z space ---
        with D.GUARD:
            mu = class_means(z_fit, y_fit, 1000)
            r_fit = residuals(z_fit, mu, y_fit)  # 训练样本用真实类路线
            tm = TransitionModel(TransitionConfig(head="mse", epochs=args.epochs, seed=seed))
            tm.fit(r_fit)

        def path_D(z: np.ndarray) -> np.ndarray:
            anchors = maha_nearest_class(fitted["maha_pp"], z[-1])
            return tm.surprise(residuals(z, mu, anchors))["D"]

        cal_D = path_D(z_cal["z"])
        fus = ResidualFusion(["base", "path"], {"path": 0.3}).fit({"base": -cal_scores["maha_pp"], "path": cal_D})
        for name, d in sets.items():
            scores[name]["path_D"] = -path_D(d["z"])
            scores[name]["maha_pp+path"] = -fus.score({"base": -scores[name]["maha_pp"], "path": -scores[name]["path_D"]})

        run = {"seed": seed, "methods": {}}
        for m in scores["id_test"]:
            res = evaluate(scores["id_test"][m], {n: scores[n][m] for n in OOD_SETS})
            res["macro5"] = dict(zip(("auroc", "fpr95"), macro_all(res)))
            run["methods"][m] = res
            print(f"  {m:14s} AUROC {res['macro5']['auroc']:.1f}  FPR95 {res['macro5']['fpr95']:.1f}"
                  f"  | near {100*res['near']['auroc']:.1f}/{100*res['near']['fpr95']:.1f}"
                  f"  far {100*res['far']['auroc']:.1f}/{100*res['far']['fpr95']:.1f}", flush=True)
        all_runs.append(run)
        (out_dir / f"baselines_{args.tag}_seed{seed}.json").write_text(json.dumps(run, indent=1))

    summary = {"tag": args.tag, "native_top1": native_acc, "paper_table_P5": PAPER, "methods": {}}
    for m in all_runs[0]["methods"]:
        a = np.array([[r["methods"][m]["macro5"]["auroc"], r["methods"][m]["macro5"]["fpr95"]] for r in all_runs])
        summary["methods"][m] = {"auroc_mean": a[:, 0].mean(), "auroc_std": a[:, 0].std(),
                                 "fpr95_mean": a[:, 1].mean(), "fpr95_std": a[:, 1].std()}
    mpp, path = summary["methods"]["maha_pp"], summary["methods"]["maha_pp+path"]
    gate = {
        "G0_maha_pp_within_1pt": bool(abs(mpp["auroc_mean"] - 92.7) <= 1.0 and abs(mpp["fpr95_mean"] - 27.8) <= 1.0),
        "G0_path_reduces_fpr95": bool(path["fpr95_mean"] < mpp["fpr95_mean"]),
    }
    gate["G0_pass"] = gate["G0_maha_pp_within_1pt"] and gate["G0_path_reduces_fpr95"]
    summary["gate"] = gate
    (out_dir / f"summary_{args.tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["methods"], indent=1))
    print("GATE", gate, f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
