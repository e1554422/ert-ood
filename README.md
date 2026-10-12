# ERT — Evidential Representation Trajectory for OOD Detection

论文《证据表征轨迹：基于逐层证据演化的分布外检测》（CVPR 2027 投稿）的代码与论文同仓库。

- 完整实验方案（论文点 → 实验 → 闸门）: [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)
- 实验索引（日期 | 脚本 | 内容 | 结论）: [docs/INDEX.md](docs/INDEX.md)
- 代码 tag ↔ 论文 tag ↔ 结果: [VERSIONS.md](VERSIONS.md)
- 论文主稿: [paper/draft.md](paper/draft.md)（LaTeX 在 `paper/latex/`，经 subtree 同步 Overleaf）

## 布局
`src/ert/` 方法实现（data / heads / state / transition / scores / baselines / metrics）·
`scripts/exp_YYYYMMDD_<phase>_<content>.py` 实验脚本 · `results/` 入库的小产物 ·
`runs/`（不入库）特征缓存与日志 · `tests/` 单元测试。

## 运行环境（Windows + RTX 4090）
```
set PYTHONUTF8=1 & set KMP_DUPLICATE_LIB_OK=TRUE
C:\Users\wangl4\Miniconda3\envs\terramind\python.exe -m pytest -q
```
长任务一律在 tmux（MSYS2）中运行，见 `scripts/run_*.ps1`。
复用 `D:\CHANGJIANG\banded_depth_ood` 中已审计的特征提取、Maha++/kNN/X-Maha 与指标实现。

## 协议红线
拟合、校准、层选择与融合权重**只用 ID 数据**；OOD 读取经 `ert.data.GUARD` 记录，选择阶段读取 OOD 会直接报错。
