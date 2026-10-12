"""2026-10-12 · P1 · 逐层 EDL 头 + 静态证据分量 U/G (闸门 G1)。

目的: 在 ID 拟合集 F (80k, 5 折交叉拟合) 上训练 12 层线性 EDL 头, 比较 digamma / mse / log
三种 EDL 损失; 报告逐层 ID 准确率、ECE、u/H 的 ID-OOD 均值与逐层 AUROC (仅诊断)。
用 ID-only 规则 (id_val 准确率 > 5%) 确定 L_eff, 在 L_eff 上聚合 U、G, 校准集 C 上残差化融合
S_static = U + 0.3·G̃。主损失 (digamma) 的 log α 缓存到 runs/evidence/ 供 P2–P7 复用。
闸门 G1: 纯证据 S_static 在 Far 上的 AUROC 不低于 MSP。
结论: (运行后填写)
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
from ert.baselines import msp  # noqa: E402
from ert.heads import CrossFitEDL, HeadConfig  # noqa: E402
from ert.metrics import evaluate  # noqa: E402
from ert.scores import ResidualFusion, effective_layers, layer_aggregate  # noqa: E402
from ert.state import summaries  # noqa: E402

OOD_SETS = ["ssb_hard", "ninco", "inaturalist", "textures", "openimage_o"]
EVID = ROOT / "runs" / "evidence"


def ece(pmax: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pmax, edges) - 1, 0, bins - 1)
    return float(sum(abs(pmax[idx == b].mean() - correct[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--losses", nargs="+", default=["digamma", "mse", "log"])
    ap.add_argument("--main-loss", default="digamma")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--acc-threshold", type=float, default=0.05)
    args = ap.parse_args()
    out_dir = ROOT / "results" / "p1"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train = D.load_id_train()
    id_eval = D.load_id_eval()
    oods = {n: D.load_ood(n) for n in OOD_SETS}
    print(f"loaded {time.time() - t0:.0f}s", flush=True)

    report = {"config": vars(args), "runs": []}
    for loss in args.losses:
        for seed in args.seeds:
            t1 = time.time()
            fit_idx, cal_idx = D.fit_calib_split(train["labels"], seed)
            y_fit = train["labels"][fit_idx]
            folds = D.kfold_ids(y_fit, 5, seed)
            with D.GUARD:
                heads = CrossFitEDL(HeadConfig(loss=loss, epochs=args.epochs, seed=seed))
                la_fit = heads.fit(train["z"][:, fit_idx], y_fit, folds)
                la = {"fit": la_fit, "calib": heads.predict(train["z"][:, cal_idx]),
                      "id_val": heads.predict(id_eval["id_val"]["z"])}
                summ = {k: summaries(v) for k, v in la.items()}
                acc_val = (summ["id_val"]["pred"] == id_eval["id_val"]["labels"][None]).mean(1)
                L_eff = effective_layers(acc_val.copy(), args.acc_threshold)
            la["id_test"] = heads.predict(id_eval["id_test"]["z"])
            summ["id_test"] = summaries(la["id_test"])
            for n in OOD_SETS:
                la[n] = heads.predict(oods[n]["z"])
                summ[n] = summaries(la[n])

            y_test = id_eval["id_test"]["labels"]
            correct = summ["id_test"]["pred"] == y_test[None]
            per_layer = []
            for l in range(12):
                row = {"layer": l, "acc_val": float(acc_val[l]), "acc_test": float(correct[l].mean()),
                       "ece_test": ece(summ["id_test"]["pmax"][l], correct[l]),
                       "u_id": float(summ["id_test"]["u"][l].mean()), "H_id": float(summ["id_test"]["H"][l].mean())}
                for q in ("u", "H"):
                    r = evaluate(-summ["id_test"][q][l], {n: -summ[n][q][l] for n in OOD_SETS})
                    row[f"{q}_auroc_near"], row[f"{q}_auroc_far"] = r["near"]["auroc"], r["far"]["auroc"]
                    row[f"{q}_ood_mean"] = float(np.mean([summ[n][q][l].mean() for n in OOD_SETS]))
                per_layer.append(row)

            comps = {k: {"U": layer_aggregate(s["u"], L_eff), "G": layer_aggregate(s["H"], L_eff),
                         "uL": s["u"][-1]} for k, s in summ.items()}
            fus = ResidualFusion(["U", "G"], {"G": 0.3}).fit(comps["calib"])
            sets = ["id_test", *OOD_SETS]
            variants = {
                "S_static(U+0.3G~)": {k: -fus.score(comps[k]) for k in sets},
                "U_only": {k: -comps[k]["U"] for k in sets},
                "G_only": {k: -comps[k]["G"] for k in sets},
                "u_last_only": {k: -comps[k]["uL"] for k in sets},
                "msp_native": {k: msp((id_eval["id_test"] if k == "id_test" else oods[k])["logits"]) for k in sets},
            }
            res = {v: evaluate(s["id_test"], {n: s[n] for n in OOD_SETS}) for v, s in variants.items()}
            run = {"loss": loss, "seed": seed, "L_eff": L_eff, "per_layer": per_layer,
                   "scores": {v: {"near": r["near"], "far": r["far"]} for v, r in res.items()},
                   "seconds": time.time() - t1}
            report["runs"].append(run)
            print(f"[{loss} s{seed}] L_eff={L_eff} acc_val(L12)={acc_val[-1]:.3f} "
                  f"{time.time() - t1:.0f}s", flush=True)
            for v, r in res.items():
                print(f"   {v:20s} near {100*r['near']['auroc']:.1f}/{100*r['near']['fpr95']:.1f}"
                      f"  far {100*r['far']['auroc']:.1f}/{100*r['far']['fpr95']:.1f}", flush=True)

            if loss == args.main_loss:  # 缓存主损失的证据与索引, 供后续阶段复用
                d = EVID / f"{loss}_seed{seed}"
                d.mkdir(parents=True, exist_ok=True)
                for k, v in la.items():
                    np.save(d / f"log_alpha_{k}.npy", v)
                np.savez(d / "index.npz", fit_idx=fit_idx, cal_idx=cal_idx, folds=folds, L_eff=np.asarray(L_eff))
            del la, summ
            (out_dir / "edl_heads_report.json").write_text(json.dumps(report, indent=1))

    # 闸门 G1 (主损失, 三种子均值)
    main_runs = [r for r in report["runs"] if r["loss"] == args.main_loss]
    s_far = np.mean([r["scores"]["S_static(U+0.3G~)"]["far"]["auroc"] for r in main_runs])
    m_far = np.mean([r["scores"]["msp_native"]["far"]["auroc"] for r in main_runs])
    report["gate"] = {"G1_static_far_auroc": s_far, "msp_far_auroc": m_far, "G1_pass": bool(s_far >= m_far)}
    (out_dir / "edl_heads_report.json").write_text(json.dumps(report, indent=1))
    print("GATE", report["gate"], f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
