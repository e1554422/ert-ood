# ERT（证据表征轨迹）完整实验方案 + 代码/论文版本迭代

## Context
用户的论文初稿（CVPR 2027）方法与假设已写完，实验、相关工作和结论还没写。本计划要做四件事：
1. 把初稿每一个可检验的点映射成具体实验，每个实验写明产物和"继续/止损"闸门；
2. 新建独立项目文件夹 `D:\CHANGJIANG\ert_ood\`，代码和论文一起做版本迭代；
3. 每完成一个阶段就打 tag，推送到 GitHub 私有仓库，并把 LaTeX 同步到 Overleaf；
4. 规模按用户选择：**先用 ImageNet-1K 单骨干把路线验证通**，闸门全部通过后再扩展骨干。

**单骨干选 `timm vit_base_patch16_224.augreg_in21k_ft_in1k`**，理由有三：
- 这正是 De la Jara 表 P5 的 "ViT-B/16-augreg" 行，可以逐格对比。该行 ImageNet-1K 数字（AUROC/FPR95）：

  | 方法 | 原值 | +path |
  |---|---|---|
  | Maha++ | 92.7 / 27.8 | 93.8 / 23.8 |
  | X-Maha | 94.3 / 22.1 | — |
  | kNN | 87.8 / 50.1 | — |
  | MSP | 87.5 / 46.5 | — |

- 本机已经缓存了这个 checkpoint 的逐层特征（`final_LN(CLS)`），覆盖 iNat、Textures、OpenImage-O、NINCO、SSB-hard 以及 ImageNet-C/R/V2/ES。
- 该原文已报告：固定层置换后性能基本不变。这直接威胁初稿"有方向演化"的主张，必须正面检验。

## 已有可复用资产（不重写）
- **特征提取**：`banded_depth_ood/scripts/extract_vit_features.py`，已缓存 OOD 集在 `banded_depth_ood/artifacts/features/<set>/layer_XX.npy`。
- **ID 数据**：`banded_depth_ood/data/datasets/ImageNet1K-trajectory/`，包含 100k 训练子集（reference 50k + statistics 50k，seed 20260914）和 50k ImageNet val。ID 的 val 5k / test 45k 按 OpenOOD 列表切分。
- **OpenOOD v1.5 严格协议**：`banded_depth_ood/configs/openood_v15_strict.json` 和 `src/banded_ood/openood_v15.py`。
- **基线**：
  - Maha++ 在 `src/banded_ood/estimators.py:MMPlusPlus`；
  - kNN 在 `scripts/run_formal_knn.py:ExactKNN`；
  - MSP/Energy 在 `scripts/run_formal_output_baselines.py`；
  - X-Maha 官方代码在 `external/X-Maha`，也用它补全 `wei2025xmaha` 的真实元数据。
- **指标**：`src/banded_ood/metrics.py:ood_metrics`。
- **EDL/NIG 损失**：
  - EDL 的 KL 与退火在 `CARAE_v4_paper/01_core_code/origin_edl_loss.py`；
  - NIG 损失在 `banded_depth_ood/.../losses/nig_regression_loss.py` 和 `models/nig_head.py`；
  - 密度式证据头在 `layerwise_density_evidence/src/density_evidence/evidence.py:LayerEvidenceHead` 和 `uce_loss`，作为止损备选。
- **已知结论（必须纳入设计）**：
  - 监督 ViT 逐层独立 probe 的 top-1 准确率只有 1.1%→81.6%，浅层几乎不可解码；
  - "样本错配对照"才是判定轨迹信息的决定性判据，打乱层序掉分不算证据；
  - 未校准 softmax 上的轨迹量是温度伪影。
  - 见记忆 prediction-dynamics-falsified。

## 新项目结构 `D:\CHANGJIANG\ert_ood\`（独立 git 仓库）
```
ert_ood/
  README.md  VERSIONS.md  CHANGELOG.md   # VERSIONS: code tag ↔ paper tag ↔ 结果表
  pyproject.toml                         # 以 editable 方式依赖 banded_ood（复用，不复制）
  configs/  p0_baselines.yaml ... p7_ablations.yaml
  src/ert/
    data.py         # 读缓存特征 + OpenOOD 切分 + fit/calib 划分
    heads.py        # 逐层 EDL 头（Softplus, α=e+1），5 折交叉拟合 → OOF 证据
    state.py        # v^l=[logα, logS]，类路线 ν_c^l，硬/软锚定，残差 r^l
    competition.py  # top-k 竞争剖面，路线切换（主导/占比/KL）
    transition.py   # 共享 MLP + 深度嵌入；输出头可选 NIG / MSE+σ̂ / 同方差 Gauss
    scores.py       # U, G, D, 证据内部残差化融合
    baselines.py    # 封装 MMPlusPlus / kNN / X-Maha / MSP / De la Jara 表征轨迹复现
    metrics.py      # 封装 ood_metrics + 配对 bootstrap
  scripts/  exp_YYYYMMDD_<phase>_<content>.py   # 脚本头注明 日期/目的/结论
  tests/    test_edl_loss.py test_nig_nll.py test_scores.py test_splits.py
  results/  <phase>/*.json *.csv  figures/      # 小产物入库
  runs/     (gitignored) 大缓存与 ckpt
  docs/     EXPERIMENT_PLAN.md（本方案）  INDEX.md（日期|脚本|内容|结论，沿用项目索引格式）
  paper/    draft.md（初稿 Markdown 主版本）  latex/（CVPR 模板，subtree 同步 Overleaf）
```

## 数据与协议（全程只用 ID 数据）
- **ID 拟合池 = 100k 训练子集**：
  - 其中 80k 作为 F（拟合集）：EDL 头 5 折交叉拟合，得到每个样本的 out-of-fold 证据。这是为了避免头在自己训练样本上过度自信，进而污染路线和转移模型。
  - 类路线 ν_c 和转移模型都在 F 的 OOF 状态上拟合。
  - 另外 20k 作为 C（校准集）：用于标准化和残差化系数 a·、b·，以及记录 ID 的 σ̂。
- **测试**：
  - ID 用 OpenOOD test_imagenet 45k；
  - Near：SSB-hard、NINCO；
  - Far：iNat、Textures、OpenImage-O；
  - 补充协变量：ImageNet-C/R/V2/ES。
  - 按数据集做宏平均。
- **基线两种口径**：
  - 50/类，与原文同口径，用来校验复现；
  - 100k，与 ERT 同数据预算，保证公平。
- **融合权重**：在 C 上 z 标准化后，**预注册 w1=w2=0.3**（2026-10-12 决策：与 De la Jara 式 7 的先验固定常数 0.3 相同，保证与表征基线同一融合规则；1.0 与网格扫描只作审计图）。初稿写的是"在开发集上用小网格定权"，但没有 ID-only 的定权准则，所以改成固定权重，这一点同步写进论文。
- **统计**：3 个种子（交叉拟合划分 × 转移模型初始化），报告均值±标准差。ERT 相对基线的 ΔFPR95 和 ΔAUROC 用样本级配对 bootstrap（1000 次）给 95% CI。

## 论文点 → 实验映射与闸门

| 阶段 | 对应初稿章节 | 实验 | 闸门（不过则执行右侧） |
|---|---|---|---|
| **P0** 基建+复现 | §实验主表骨架、参考文献 | ① 提取 ViT-AugReg 在 100k 训练子集和 50k val 上的逐层特征（tmux 中跑，约 30–40 分钟）。② 跑 MSP / kNN / Maha++ / RelMaha / X-Maha。③ **复现 De la Jara 表征轨迹**：z 空间残差 + MLP + MSE + 残差化叠加到 Maha++。④ 从 external/X-Maha 补全 wei2025xmaha 元数据 | Maha++ 与 92.7/27.8 相差在 ±1 以内，且 +path 的改进方向与原文一致 → 否则先排查预处理和协议，不进入 P1 |
| **P1** 逐层 EDL 头 | §每层 EDL 分类头、§证据状态 | 线性头 768→1000，比较两种 EDL 损失：Sensoy MSE 版与 digamma(CE) 版，KL 退火 λ_t=min(1,t/10)。逐层报告 top-1、ECE、ID/OOD 上 u^l 和 H^l 的均值，以及逐层 AUROC。用 ID-only 规则确定有效层段 L_eff（ID 准确率 > 5% 的层） | **G1（关键）**：纯证据项 U+G 在 Far 上不低于 MSP，且 U 的逐层曲线可以解释。不过 → 换成密度式证据头（LayerEvidenceHead/UCE，仍是纯证据），并在论文中如实写出 EDL vacuity 在 ID-only 训练下的已知局限 |
| **P2** H2 证据路线 | §待验证假设 H2 | 逐类逐层计算 u_c^l、e_y^l、max 错误类证据、间距 m^l、H^l。把归一化曲线做 k-means，统计逐步积累型 / 后期重组型 / 早分离晚压缩型各占多少比例，配典型类别示意图 | 描述性实验，不设闸门。如果只出现一种模式，就改写 H2 的表述 |
| **P3** H1 身份连续性 | §H1 | 用 ridge 回归（λ=1）和 MLP 拟合 r^l→r^{l+1}，比较三种设置：真实配对、**样本错配**、层序打乱。报告 R²、ΔR²，以及 bootstrap 1000 次的 CI。证据空间和 z 空间都做 | **G3**：过半相邻层对的 ΔR² CI 不含 0。不过 → 转移意外度在机理上不成立，论文退为静态证据（U+G̃）版本，D 降级为消融 |
| **P4** 竞争读取/冲突 | §竞争读取与冲突分数 | top-k 剖面（k=3/5/10），路线切换的闭式量：主导性、占比、top-k 上的 KL。在后半段层上汇总 | 预注册主版本 G=Σω·H^l（与打分章节一致），路线切换版作为消融 |
| **P5** 转移模型 | §转移模型与转移意外度 | 共享 MLP（hidden 1024）+ 深度嵌入；AdamW，lr 3e-4，wd 1e-5，80 epoch，batch 512（与原文 Table D2 一致）。三种输出头：NIG、MSE+σ̂（D_ERT）、同方差 Gauss。逐 OOD 集报告 NIG 的认知不确定性 Var[μ] | 在 C 上验证损失收敛，且 ID 上的 NLL 稳定 |
| **P6** 打分与 H3 | §打分与残差化融合、§H3 | 计算 S=−(U+G̃+D̃)。H3 包含：corr(D,u^L)；末层 / +原始轨迹 / +残差化轨迹三者对比；随机噪声对照；正序与反序融合；分量逐项去除 | **G6**：控制 u^L 之后，D̃ 仍能让 FPR95 改善，且 CI 不含 0 |
| **P7** 消融 | §消融设计 | **A**：证据空间 vs z 空间 vs [p,u] 紧凑版 vs z 线性投影到 C+1 维（PCA/LDA 等维对照）。**B**：NIG vs MSE vs 同方差 Gauss。支撑消融：+Maha++ 基础项、软锚定、MLP vs GRU vs DeepSets（无序） | 结果全部如实报告。若 DeepSets≈MLP，就删去"有方向演化"的主张 |
| **P8** 扩展 | §实验主表 | 闸门都通过后依次加入 DINOv2-B、CLIP-B/16、DeiT3-B，再加 Swin-B、ResNet-50，再扩到 ImageNet-200 / CIFAR。与原文表 P5 逐格对比 | — |

## 写实验前要先修正的初稿不一致（paper-v0.2 落实）
1. **分数方向冲突**：问题设定写的是"ID 得分高"，但 S_OOD=U+… 是 OOD 得分高。统一为 S(x)=−S_OOD(x)。
2. **冲突分数有两个定义**：方法节是路线切换 KL，打分节是熵聚合 G。明确主版本是熵版，路线切换版作为消融。
3. **转移模型输入不一致**：一处写"竞争剖面"（k 维），一处写 r^l（C+1 维，D 的公式就是按这个求和）。主版本取 r^l ∈ R^{C+1}，竞争剖面作为消融。
4. **w1、w2 的选取方式**：改为预注册固定权重（见上文"融合权重"）。
5. **补写 L_eff 规则**：浅层不可解码，层权重 ω_l 只覆盖 ID-only 规则选出的层。
6. **补局限与相关工作**：EDL vacuity 在只用 ID 训练时是否忠实存在争议（Bengs et al. NeurIPS'22；Jürgens et al. ICML'24），相关工作需要讨论，并在 P1 里用实测回应。

## 版本迭代与同步
| tag（code / paper） | 触发条件 | 论文同步内容 |
|---|---|---|
| v0.1 / paper-v0.1 | 项目骨架 + 初稿导入 + P0 复现通过 | 导入初稿，补全 X-Maha 文献，写好主表的基线列 |
| v0.2 / paper-v0.2 | P1 通过 | 修正上面 6 条不一致，写方法细节与 EDL 头诊断 |
| v0.3 / paper-v0.3 | P2+P3 | H1/H2 图表与结论 |
| v0.4 / paper-v0.4 | P4–P6 | ERT 主结果（单骨干）与 H3 |
| v0.5 / paper-v0.5 | P7 | 消融节 |
| v1.0 / paper-v1.0 | P8 | 多骨干主表、结论、局限 |

每个阶段收尾时：
1. `pytest` 全部通过；
2. 结果 JSON/CSV 和图写入 `results/`；
3. 更新 `docs/INDEX.md`、`VERSIONS.md`、`CHANGELOG.md`，并在 `paper/draft.md` 加修订记录；
4. 依次执行 `git commit`、`git tag`、`git push origin main --tags`；
5. 执行 `git subtree push --prefix paper/latex overleaf master`；
6. 更新记忆。

`paper/latex` 用 pandoc 从 `draft.md` 生成（初稿注释里已写好命令，配一个剥离 callout 的 lua filter），之后再人工微调。

**需要用户提供**：
- 在 GitHub 网页上新建的空私有仓库 `ert-ood` 的 URL（用 Git Credential Manager 推送）；
- 一个新建 Overleaf 项目的 git URL（与现有的 PRDC 项目分开）。

**运行规范**：
- 长任务全部在 tmux 中跑（MSYS2，runner ps1 设置 PATHEXT）；
- 环境变量：`PYTHONUTF8=1`、`KMP_DUPLICATE_LIB_OK=TRUE`、`num_workers=0`；
- python 用 `C:\Users\wangl4\Miniconda3\envs\terramind\python.exe`。

## 验证
- **单元测试**：
  - EDL 损失与 KL 的数值和手算一致；
  - α=[1,1] 时 u=1，α=[100,100] 时 u=0.01 且熵最大（初稿中的示例直接写成测试）；
  - NIG 的 Student-t NLL 与 scipy.stats.t.logpdf 对齐；
  - 划分没有泄漏，fit、calib、test 的文件名两两不相交；
  - 任何选择步骤都没有读取 OOD 数据（在 data.py 里加断言守卫）。
- **端到端**：P0 复现数字与原文表 P5 的 ViT-B/16-augreg 行对上；每个阶段的闸门都输出一个 JSON 判定（pass/fail + 数值），写进 `results/gates.json`。
- **同步检查**：每次 push 之后，`git ls-remote origin --tags` 能看到新 tag，Overleaf 项目能正常编译。
