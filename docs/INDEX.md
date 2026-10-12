# 实验索引（单一事实来源）

| 日期 | 阶段 | 脚本 | 主要内容 | 结论 |
|---|---|---|---|---|
| 2026-10-12 | P0 | scripts/exp_20261012_p0_build_manifests.py | 构建 ID 特征提取清单 + OpenOOD ID val/test 索引 | 100k/50k；id_val 5000 / id_test 45000，互不相交 |
| 2026-10-12 | P0 | scripts/run_p0_extract.ps1 | tmux 提取 ViT-B/16 AugReg 逐层 final_LN(CLS) | 完成：train100k + val50k（约 95 分钟） |
| 2026-10-12 | P0 | scripts/exp_20261012_p0_baselines_reproduction.py | 50/30 每类三种子复现 MSP/Energy/Maha++/RelMaha/kNN/X-Maha + z 空间 path-surprise(0.3 融合) | ★ G0 通过：Maha++ 92.9/27.5（原文 92.7/27.8），+path 93.9/23.9（93.8/23.8），X-Maha 94.4/21.8（94.3/22.1）；三种子 std≤0.3 |
| 2026-10-12 | P1 | scripts/exp_20261012_p1_edl_heads.py | 逐层 EDL 头 5 折交叉拟合，digamma/mse/log 三损失，L_eff，静态 U+0.3G̃ | 排队中；合成数据冒烟：bias 0 初始化致 u 恒定（已修为 -5），大范数 OOD 上 vacuity 反向 |
