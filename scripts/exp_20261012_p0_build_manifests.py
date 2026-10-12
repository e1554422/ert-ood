"""2026-10-12 · P0 · 构建 ID 特征提取用的 PATH LABEL 清单。

目的: 把 ImageNet1K-trajectory 的 jsonl 清单 (reference/statistics 训练子集 100k,
validation 50k) 转成 extract_vit_features.py 可读的文本清单, 并按 OpenOOD v1.5
val/test 列表给出 val 50k 内的 id_val(5k)/id_test(45k) 索引。
结论: (运行后填写)
"""
from __future__ import annotations

import json
from pathlib import Path

BANDED = Path(r"D:\CHANGJIANG\banded_depth_ood")
TRAJ = BANDED / "data" / "datasets" / "ImageNet1K-trajectory"
OPENOOD = BANDED / "data" / "openood" / "benchmark_imglist" / "imagenet"
OUT = Path(__file__).resolve().parents[1] / "runs" / "manifests"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    train = read_jsonl(TRAJ / "manifests" / "reference.jsonl") + read_jsonl(
        TRAJ / "manifests" / "statistics.jsonl"
    )
    val = read_jsonl(TRAJ / "manifests" / "validation.jsonl")
    assert len(train) == 100_000 and len(val) == 50_000
    assert len({r["relative_path"] for r in train}) == 100_000

    for name, rows in (("train100k", train), ("val50k", val)):
        lines = [f"{r['relative_path']} {r['label']}" for r in rows]
        (OUT / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    position = {r["filename"]: i for i, r in enumerate(val)}
    splits = {}
    for split, fname in (("id_val", "val_imagenet.txt"), ("id_test", "test_imagenet.txt")):
        names = [Path(l.split()[0]).name for l in (OPENOOD / fname).read_text().splitlines() if l]
        splits[split] = sorted(position[n] for n in names)
    assert len(splits["id_val"]) == 5_000 and len(splits["id_test"]) == 45_000
    assert not set(splits["id_val"]) & set(splits["id_test"])
    (OUT / "openood_id_splits.json").write_text(json.dumps(splits), encoding="utf-8")
    print({k: len(v) for k, v in splits.items()}, "train", len(train), "val", len(val))


if __name__ == "__main__":
    main()
