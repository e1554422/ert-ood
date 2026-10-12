# 实验索引（单一事实来源）

| 日期 | 阶段 | 脚本 | 主要内容 | 结论 |
|---|---|---|---|---|
| 2026-10-12 | P0 | scripts/exp_20261012_p0_build_manifests.py | 构建 ID 特征提取清单 + OpenOOD ID val/test 索引 | 100k/50k；id_val 5000 / id_test 45000，互不相交 |
| 2026-10-12 | P0 | scripts/run_p0_extract.ps1 | tmux 提取 ViT-B/16 AugReg 逐层 final_LN(CLS) | 进行中 |
