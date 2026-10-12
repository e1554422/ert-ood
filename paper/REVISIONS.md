# 论文修订记录

| 版本 | 日期 | 修改 | 依据 |
|---|---|---|---|
| paper-v0.1 | 2026-10-12 | 导入初稿原文（未改动） | — |

## 待落实（paper-v0.2，见 docs/EXPERIMENT_PLAN.md「初稿不一致」）
1. 分数方向统一为 S(x) = −S_OOD(x)。
2. 冲突分数主版本 = 熵聚合 G；路线切换 KL 降为消融。
3. 转移模型主输入 = r^l ∈ R^{C+1}；竞争剖面降为消融。
4. w1, w2 预注册固定为 0.3（与 De la Jara 式 7 同一常数，保证公平），1.0 及网格仅作审计。
5. 补写 L_eff（ID 准确率 > 5% 的连续尾段层）规则。
6. 补 EDL vacuity 忠实性争议（Bengs et al. NeurIPS'22；Jürgens et al. ICML'24）。
7. 参考文献：Maha++ = Müller & Hein, ICML 2025, arXiv:2505.18032；X-Maha 为 NeurIPS 2025（作者待核）。
