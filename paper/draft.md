> [!info] 论文初稿 · CVPR 2027
> **中文标题**　证据表征轨迹：基于逐层证据演化的分布外检测
> **英文标题**　Evidential Representation Trajectory for Out-of-Distribution Detection
> **验证路线**　纯证据打分（空缺+冲突+转移，不依赖外部检测器）· `log α + log S` 证据状态 · 硬锚定 · NIG + Student-t NLL 转移意外度 · 一步 MLP 主干
> **当前状态**　方法与假设已成稿；相关工作 / 实验 / 结论待补充
> **关键词**　分布外检测 · 证据深度学习 · 逐层表征 · 不确定性估计

> [!abstract] 摘要
> 分布外（Out-of-Distribution, OOD）检测是部署可信视觉系统的核心前提。现有最强的 post-hoc 逐层方法（如 X-Mahalanobis [@wei2025xmaha]）把每一层的信息压缩成一个到最近类中心的标量距离，再加权求和，由此丢失了两类关键信息：**其一**，无法区分"该层缺乏任何类别证据"与"证据在多个竞争类别间分散"这两种不确定性来源；**其二**，逐层距离独立计算后相加，既不记录样本在不同深度上支持哪个类别，也不刻画相邻层之间的表征转移是否典型。
>
> 本文提出 **证据表征轨迹（Evidential Representation Trajectory, ERT）**，将逐层表征重新理解为沿网络深度逐步累积证据的过程。我们在冻结骨干的每个原生块后接一个轻量证据头，用证据化深度学习（EDL [@sensoy2018edl]）在统一的类别坐标系中读取每层的狄利克雷证据；进而定义类别条件证据残差，并用带法向逆伽马（NIG [@amini2020evidential]）输出头的一步转移模型度量层间证据演化的"意外度"。最终 OOD 分数为**纯证据分数**：由逐层空缺度、逐层证据冲突与层间转移意外度三类自身证据信号，在分布内校准集上逐项残差化叠加得到，**不以任何外部检测器（如 Maha++）为基础项**。整个框架**仅使用分布内数据训练、骨干保持冻结、严格 post-hoc**。我们给出一条确定的验证路线及其假设检验（身份连续性、可重复证据路线、超出末层不确定性的增量信息），用于系统评估 ERT 在近 OOD 与远 OOD 上的互补优势。

---

# 引言

深度网络对输入的处理是逐层进行的：浅层偏向纹理与局部统计等低层属性，深层逐渐聚合为类别语义，而末层为完成分类会主动压缩与类别无关的变化。这种压缩恰恰可能丢弃对识别分布偏移有用的线索。因此，相比只使用 logits 或倒数第二层特征的多数 post-hoc 方法，在冻结编码器的多个深度上分别建模的逐层方法展现出明显优势 [@wei2025xmaha]。

然而，当前逐层范式存在一个根本局限：它把每一层的信息都压缩成一个到最近类中心的距离，再合并为单一标量分数。这一压缩丢失了两类对 OOD 检测至关重要的信息。

> [!warning] 现有逐层范式的两处信息损失
> **① 不确定性来源不可分。** 一个较大的层内距离既可能意味着样本在该层缺乏支持任何类别的证据（**空缺**），也可能意味着证据分散在多个相互竞争的类别之间（**冲突**）；标量距离无法区分这两种来源。
>
> **② 层间轨迹被抹除。** 逐层距离各自独立计算后再加权求和，既不记录样本在不同深度上分别支持哪个类别，也不刻画相邻层之间的表征转移是否典型。对于近 OOD 样本，末层往往给出自信但错误的类别判断，而这种误判在中间层是否已有迹象，正是加权求和无法捕捉的。此外，固定或启发式的层权重难以适应不同偏移，且最佳结果常依赖额外微调，与严格 post-hoc 设定相冲突。

证据化深度学习 [@sensoy2018edl] 为上述问题提供了自然的框架。它不直接输出点估计概率，而是输出一组证据，并以二阶分布表达不确定性：在分类任务中，证据参数化一个狄利克雷分布，其总证据强度的倒数刻画"缺乏证据"的程度，证据在各类之间的分散程度刻画"证据冲突"的程度；在回归任务中，NIG 分布 [@amini2020evidential] 将预测误差分解为偶然不确定性与认知不确定性。受此启发，我们将逐层表征重新理解为一个沿深度逐步累积证据的过程：在每一层，模型既读取各类别之间的竞争格局以形成分类证据，也评估当前层到下一层的表征转移是否符合分布内的典型模式以形成转移证据。由此，OOD 样本不再仅仅表现为"距离某个类中心过远"，而可以表现为**证据累积不足、跨层证据相互冲突，或层间转移超出分布内的认知范围**。

**本文的主要贡献：**

| # | 贡献 |
|:---:|:---|
| 1 | 提出 **ERT**——仅用分布内数据训练、骨干保持冻结的证据化逐层读取器，将逐层表征建模为证据累积轨迹 |
| 2 | 在统一类别证据坐标系中定义**类别条件证据残差**，用 NIG 转移模型度量层间演化意外度，显式分离**空缺 / 冲突 / 转移异常**三类互补信号 |
| 3 | 给出严格 post-hoc 的**纯证据残差化融合协议**：三类证据信号在彼此之间逐项残差化叠加，不依赖任何外部检测器分数，权重预先固定，不用任何 OOD 数据选超参 |
| 4 | 提出并给出三个可检验假设（身份连续性、可重复证据路线、增量信息）的验证方案 |

---

# 相关工作

> [!todo] 待补充
> post-hoc OOD 检测 · 逐层/多尺度方法 · X-Mahalanobis 与 Maha++ · 证据化深度学习与 NIG · 表征轨迹分析。本节在初稿后期补全并配齐引用键。

---

# 方法

## 问题设定

设数据集 $D=\{(x_i,y_i)\}_{i=1}^{N}$ 来自真实但未知的分布 $p$，其中每个 $x_i\in\mathbb{R}^d$ 独立同分布采样自 $p$，$y_i\in\{1,\dots,C\}$ 为其类别标签。在 OOD 检测场景下，部署时遇到的未见数据 $\{\tilde{x}_j\}_{j=1}^{n}$ 可能来自真实分布 $p$ 与一个未知偏移分布 $\tilde{p}$ 的混合分布（偏移既可是语义层面——出现训练中未见的类别，也可是协变量层面——成像条件、纹理或噪声发生改变）：

$$\tilde{x}\sim \alpha\, p(\tilde{x}) + (1-\alpha)\,\tilde{p}(\tilde{x})$$

其中 $\alpha$ 是未知混合系数。由于 $\alpha$ 与 $\tilde{p}$ 均未知，我们无法直接从 $\tilde{p}$ 采样，也不对二者做任何假设，检测器的全部组件只能由分布内数据估计。目标是构建评分函数 $S(x)$ 与阈值 $\tau$，使分布内样本得分高、分布外样本得分低，当 $S(\tilde{x})<\tau$ 时判定 $\tilde{x}$ 为异常。

本文基于逐层表征展开。代表方法 X-Mahalanobis [@wei2025xmaha] 在冻结编码器的多个深度 $h_1(x),\dots,h_L(x)$ 上分别建模：先在每一层拟合典型性度量 $T_l(x)$，例如到最近类中心的 Mahalanobis 距离

$$T_l(x)=\min_{c}\,\big(h_l(x)-\hat{\mu}_{c,l}\big)^{\top}\hat{\Sigma}_l^{-1}\big(h_l(x)-\hat{\mu}_{c,l}\big)$$

其中 $\hat{\mu}_{c,l},\hat{\Sigma}_l$ 为该层的类均值与类间共享协方差；再由汇总函数 $A$ 得 $S(x)=A\big(T_1(x),\dots,T_L(x)\big)$。X-Mahalanobis 以层权重 $\{w_l\}$ 对逐层距离加权求和，取各类别中 $\sum_l w_l\,T_l(x,c)$ 的最小值取负为 $S(x)$。本文据此构建一个冻结骨干、仅用分布内数据的证据化深度读取器，将空缺、冲突与转移异常三类证据作为互补 OOD 信号。

---

## 逐层轨迹演化框架

### 骨干与层表示

对图像 $x$，从冻结视觉编码器提取与原文一致的原生全局读出 $z^1(x),\dots,z^L(x)$：

- **ViT / DeiT / CLIP**：每个 block 的 CLS token 或原生全局 token；
- **Swin**：每个 stage 的全局池化输出；
- **ResNet / ConvNeXt**：每个 stage 的全局平均池化特征。

这样保持与原文相同的全局轨迹范围，避免 token 或特征图的对齐成为额外混杂因素。

### 每层的 EDL 分类头

对每层表示 $z^l$ 构建轻量证据头 $h_l(z^l)\to a^l\in\mathbb{R}^C$（$C$ 为分布内类别数），用 Softplus 保证每类证据非负：

$$e_k^l(x)=\mathrm{Softplus}\big(a_k^l(x)\big),\qquad \alpha_k^l(x)=e_k^l(x)+1$$

总证据量（狄利克雷浓度）、预测类别概率与空缺度（vacuity）：

$$S^l(x)=\sum_{k=1}^{C}\alpha_k^l(x),\qquad p_k^l(x)=\frac{\alpha_k^l(x)}{S^l(x)},\qquad u^l(x)=\frac{C}{S^l(x)}$$

其中 $p^l$ 表示该层倾向哪个类别，$S^l$ 表示积累了多少分类证据，$u^l$ 表示是否仍缺可用证据。

> [!note] 两类不确定性的精确对应
> **空缺度** $u^l=C/S^l$ ——证据总量不足 → **认知不确定性**（模型知识不足，可通过增训消除）。$S^l\to\infty$ 时 $u^l\to 0$（极确定），$S^l\to C$ 时 $u^l\to 1$（完全空缺）。
>
> **类别熵** $u_{\text{mut}}^l(x)=-\sum_{k=1}^{C}p_k^l(x)\log p_k^l(x)$ ——证据足但在竞争类别间分散 → **偶然不确定性**（样本本身模糊，再多数据也消不掉）。
>
> 二者正是"空缺"与"冲突"在逐层 EDL 下的精确对应：二分类中 $\alpha=[1,1]$ 给出 $u_{\text{vac}}=1$（**纯空缺**）；而 $\alpha=[100,100]$ 给出 $u_{\text{vac}}=0.01$ 却 $p=[0.5,0.5]$ 熵极高（**纯冲突**）。
> ⚠️ 此处的认知/偶然分解属 EDL **分类**框架，与后文「转移模型与转移意外度」一节 NIG **回归**框架对连续残差转移的同名分解分属两个层面，不应混淆。

每个头独立训练，只用分布内留出数据，采用 EDL 损失加退火 KL 正则：

$$\mathcal{L}_{\text{EDL}}=\mathcal{L}_{\text{cls-Dirichlet}}+\lambda_t\cdot \mathrm{KL}\big[\mathrm{Dir}(\tilde{\alpha})\,\|\,\mathrm{Dir}(\mathbf{1})\big]$$

对分布内样本，错误类别证据很小，KL 大，损失压低错误类别证据；对 OOD / 模糊样本，各类证据都弱，$\mathrm{Dir}(\tilde{\alpha})$ 接近均匀，KL 小、惩罚轻，允许输出高空缺度。退火系数 $\lambda_t$ 随训练轮数升高。所有层共享同一组类别坐标。

### 证据状态

只用 $p^l$ 不够——各类概率之和恒为 $1$，会丢掉"证据总量"这一 EDL 最重要的维度。我们把每层证据状态定义为

$$v^l(x)=\big[\log\alpha_1^l(x),\dots,\log\alpha_C^l(x),\ \log S^l(x)\big]\in\mathbb{R}^{C+1}$$

它保留每类相对证据与总证据量，对极大证据值数值更稳定，且各层都处在同一 $C+1$ 维类别坐标系中，比原始 $z^l$ 更易跨层对齐。信息较少的紧凑版 $v_{\text{compact}}^l(x)=[\,p^l(x),u^l(x)\,]$ 用作对照。

### 类别路线与证据残差

训练样本 $i$ 真实类别为 $y_i=c$。对每类每层计算平均证据状态

$$\nu_c^l=\mathbb{E}\big[v^l(X)\mid Y=c\big]$$

即"类别 $c$ 的证据路线" $\nu_c^1\to\cdots\to\nu_c^L$。对训练样本定义类别条件证据残差 $r_i^l=v_i^l-\nu_{y_i}^l$，与原文的表示残差 $r_i^l=z_i^l-\mu_{y_i}^l$ 在结构上完全对应，只是把视觉表示空间换成 EDL 的类别证据空间。

**测试时类别锚定。** 测试时不知真实标签，故只在末层锚定一次（主实验用硬锚定）：

$$\hat{c}(x)=\arg\max_k\,p_k^L(x)$$

随后所有层相对同一路线算残差 $r^l(x)=v^l(x)-\nu_{\hat{c}(x)}^l$——并非每层重新选类，而是末层选一次 $\hat{c}$ 后沿用其整条路线。作为独立消融，软锚定用末层概率对全部锚点加权：

$$\bar{\nu}^l(x)=\sum_c p_c^L(x)\,\nu_c^l,\qquad r_{\text{soft}}^l(x)=v^l(x)-\bar{\nu}^l(x)$$

硬锚定更接近原文、解释直接；软锚定对相似类别、模糊样本及近 OOD 可能更鲁棒。我们不默认软锚定更好，将其作为独立消融。

### 竞争读取与冲突分数

有了逐层全类狄利克雷，竞争剖面可直接在证据空间构造，无需另算 Mahalanobis 距离。先在末层取证据最高的 $k$ 个类并按末层排名编号；每层记录这 $k$ 类的 $\log\alpha$（按末层排名排列）、$\log S^l$、该层主导类在末层 top-$k$ 中的排名，以及第一名与第二名的证据差。共享类别坐标解决跨层对齐，排名编号使状态类别无关。路线切换以闭式计算：末层第一名在第 $l$ 层是否仍主导、其证据占比，以及第 $l$ 层与末层狄利克雷在这 $k$ 类上的 KL 散度。沿后半段各层汇总即得 **冲突分数**，直接估计"给定末层判定后前面各层分配是否异常"。

### 转移模型与转移意外度

在分布内数据上训练一个共享的、带深度嵌入的一步 MLP，由第 $l$ 层竞争剖面预测第 $l+1$ 层竞争剖面。给每层分配专属深度嵌入 $e_{\text{depth}}^l$，预测形式为 $f_\theta(r^l+e_{\text{depth}}^l)$。不同于原始方案的点预测加固定方差，我们将输出头换成 NIG [@amini2020evidential]：对每一维输出 $(\gamma,\nu,\alpha,\beta)$——$\gamma$ 均值、$\nu$ 样本证据量、$\alpha$ 形状、$\beta$ 尺度，边缘分布为 Student-t。

模型不再预测"下一层残差等于某定值"，而是对每一维预测一个 Student-t 分布，并以其对真实观测的下一层竞争剖面计算负对数似然（NLL），作为 **转移意外度**（新版 ERT surprise）：分布内样本落在高概率区 NLL 小，OOD 样本落在尾部 NLL 大。

作为与 EDL + MSE 单点预测对齐的基线，原始 ERT 意外度以预测误差经分布内方差归一化的平方和给出：

$$D_{\text{ERT}}(x)=\sum_{l=1}^{L-1}\sum_{d=1}^{C+1}\frac{\big(r_d^{l+1}(x)-\hat{r}_d^{l+1}(x)\big)^2}{\hat{\sigma}_{l,d}^2+\epsilon}$$

其中 $\hat{r}_d^{l+1}(x)$ 为转移模型对第 $l{+}1$ 层第 $d$ 维残差的预测，$\hat{\sigma}_{l,d}^2$ 为分布内拟合集上该维预测误差方差，$\epsilon$ 为数值稳定项。NIG 版本以逐维 Student-t NLL 取代上式中的归一化平方误差。NIG 还可显式分离两种不确定性：偶然 $\mathbb{E}[\sigma^2]=\beta/(\alpha-1)$（数据固有噪声）与认知 $\mathrm{Var}[\mu]=\beta/\big(\nu(\alpha-1)\big)$（模型知识不足，OOD 核心信号）。我们单独报告认知不确定性用于诊断，考察近 OOD 是否主要触发其升高。

转移意外度与冲突分数虽同用类别条件残差，刻画对象不同：**冲突分数**关注某层更新中样本特异能量的占比；**转移意外度**关注该样本特异残差从当前层变到下一层时是否符合分布内学到的条件转移模式。

> [!important] 核心主张
> OOD 不一定只表现为"末层证据少"；它还可能表现为**证据积累、类别竞争消解或不确定性收缩的路径，不符合分布内类别的正常机制**。

---

## 打分与残差化融合

> [!important] 设计原则：纯证据打分
> ERT 的 OOD 分数**完全由自身的证据量、证据分散度与证据转移意外度构成，不以任何外部检测器（如 Mahalanobis / Maha++ / kNN）为基础项**。这是本文与"以 Maha++ 为基础项、轨迹仅作残差化补充"的表征轨迹方法 [@delajara2026traj] 的根本区别：后者的 OOD 判别力主要来自末层几何距离检测器，轨迹只是修正；而本文主张**二阶证据分布本身——跨层累积与演化——已足以独立刻画分布偏移**，不需要借助外部基础模型得到的分数。

### 三类证据分量

全部分量来自 ERT 自身的逐层 EDL 头与 NIG 转移模型，不涉及任何几何距离或外部检测器：

**① 空缺分量（远 OOD 主导）**——逐层空缺度聚合，刻画"各层是否普遍缺乏任何类别证据"：

$$U(x)=\sum_{l=1}^{L}\omega_l\,u^l(x),\qquad u^l(x)=\frac{C}{S^l(x)}$$

其中 $\omega_l$ 为预先固定的非负层权重（默认均匀，$\omega_l=1/L$）。远 OOD 样本在多数层证据总量 $S^l$ 都偏低，$U$ 自然偏高。这取代了原方案中"以 Maha++ 为基础项"的角色——基础判别力改由证据总量不足提供。

**② 冲突分量（近 OOD 与模糊样本）**——逐层类别熵聚合，刻画"证据足但在竞争类别间分散"：

$$G(x)=\sum_{l=1}^{L}\omega_l\,H^l(x),\qquad H^l(x)=-\sum_{k=1}^{C}p_k^l(x)\log p_k^l(x)$$

它捕捉 $U$ 无法区分的那类近 OOD：证据总量看似充足（$u^l$ 不高），但分散在多个类别间。

**③ 转移意外度（本文核心信号，近 OOD）**——层间证据演化是否偏离分布内条件转移模式，由 NIG 转移模型逐维 Student-t NLL 给出：

$$D(x)=\sum_{l=1}^{L-1}\sum_{d=1}^{C+1}-\log t_{\nu_{l,d}}\!\big(r_d^{l+1}(x)\,;\,\gamma_{l,d},\,\sigma_{l,d}\big)$$

其中 $(\gamma,\nu,\alpha,\beta)$ 为转移模型对第 $l{+}1$ 层第 $d$ 维残差的 NIG 输出，$t$ 为其诱导的 Student-t 边缘分布。分布内样本落在高概率区 NLL 小，证据演化反常的近 OOD 落在尾部 NLL 大。此分量是表征轨迹方法 [@delajara2026traj] 的 MSE/固定方差 surprise 所没有的分布式刻画（见 §转移模型与转移意外度）。

### 证据内部残差化融合

三分量**仅在彼此之间**残差化叠加，不引入任何外部分数。按"空缺 → 冲突 → 转移意外度"顺序，每加一个新分量，先在分布内校准集上对已并入的分量做线性回归、只留残差，再以预先固定权重加入：

$$\tilde{G}=G-a_1 U-b_1,\qquad \tilde{D}=D-a_2 U-a_3\tilde{G}-b_2$$
$$S_{\text{OOD}}(x)=U(x)+w_1\,\tilde{G}(x)+w_2\,\tilde{D}(x)$$

残差化确保每个后加分量只贡献前面分量尚未编码的信息：$\tilde{G}$ 只保留"冲突"中独立于"空缺"的部分，$\tilde{D}$ 只保留"转移异常"中独立于静态空缺与冲突的部分。三分量机理互补：

| 分量 | 擅长场景 | 机理 | 来源 |
|:---|:---|:---|:---|
| $U$ | **远 OOD** | 各层证据总量普遍不足（空缺） | 逐层 EDL 头 |
| $\tilde{G}$ | 模糊 / 部分近 OOD | 证据足但跨类分散（冲突） | 逐层 EDL 头 |
| $\tilde{D}$ | **近 OOD** | 末层证据未必塌缩，但层间证据演化偏离 ID 条件转移 | NIG 转移模型 |

其中所有分量用留出的分布内校准集标准化，$a_\cdot,b_\cdot$ 由校准集线性拟合，$w_1,w_2$ 预先固定（在单独开发集上于小网格中各定一次，之后不按模型/数据集/骨干调整）。

> [!check] 实验协议
> 拟合集与校准集**都只含分布内数据**；使用**三个随机种子**；**任何超参（含 $w_1,w_2$）都不用 OOD 数据选择**。事后可画权重扫描曲线作稳健性审计，但不据此选权重。

> [!note] 与 Maha++ 的关系：对照而非依赖
> 为量化"放弃外部几何检测器"的代价与收益，我们额外报告一个**非主方法**的对照变体 $S_{\text{OOD}}^{+\text{Maha}}=\text{Maha++}+w\cdot\tilde{D}$（即把 Maha++ 作为基础项、ERT 转移分数残差化叠加，对应 [@delajara2026traj] 的融合范式）。主方法始终是上面的**纯证据版** $S_{\text{OOD}}$；该对照仅用于诚实报告纯证据打分相对"挂靠强末层检测器"在远 OOD 绝对指标上损失多少、在独立性与可解释性上换回多少。

---

# 待验证假设与评估方案

> [!tip] 本文选定的验证路线
> 纯证据打分（空缺+冲突+转移，不依赖外部检测器）· `log α + log S` 证据状态 · **硬锚定为主** · **NIG + Student-t NLL** 转移意外度 · **一步 MLP** 主干转移模型。围绕以下三个假设组织评估。

**H1 — 身份连续性（证据轨迹具类别条件连续性）**
真实配对 $r^l(x_i)\to r^{l+1}(x_i)$ 应显著优于打乱身份的对照 $r^l(x_j)\to r^{l+1}(x_i)$（$j\neq i$）。指标：留出集 $R^2$、MSE、$\Delta R^2$（真实减打乱）、bootstrap 置信区间。这是验证"证据轨迹不是一组独立层快照"的最关键机制检验。

**H2 — 可重复证据路线（架构剖面）**
对每类可视化并统计：类平均空缺度 $u_c^l$、正确类证据 $e_y^l$、最大错误类证据 $\max_{k\neq y}e_k^l$、证据间距 $m^l=e_y^l-\max_{k\neq y}e_k^l$、类别熵 $H^l$。预计至少出现三种模式——**逐步积累型**（$S^l\uparrow$、$u^l\downarrow$、间距增大）、**后期重组型**（中间层竞争持续、末层才消解）、**早分离晚压缩型**（中间层已明确、末层为判别压缩细粒度）。不假设所有类别/架构不确定性都单调下降。

**H3 — 增量信息（轨迹分数超出末层 EDL 不确定性）**
关键不是证明 $D_{\text{ERT}}$ 自己能检测 OOD，而是证明 $D_{\text{ERT}}\not\approx f(u^L)$。方法：算 $\mathrm{corr}(D_{\text{ERT}},u^L)$；做分布内残差化；比较"只用末层 / 末层+原始轨迹 / 末层+残差化轨迹"；以随机噪声或打乱层顺序作对照；做正反序顺序融合。只有在控制末层不确定性后轨迹仍改善 FPR95 与 AUROC，才足以说明它不是在重复编码末层不确定性。

---

# 消融设计

表征轨迹方法 [@delajara2026traj] 已在**视觉表征空间** $z^l$ 上验证了"轨迹残差 + 一步 MLP + 残差化融合"这一结构的有效性。因此本文的消融不重复论证轨迹结构本身，而**集中证明两处本文独有的设计到底带来了什么**：(A) 把轨迹从视觉表征空间迁到 **EDL 证据空间**；(B) 把 surprise 从 MSE/固定方差换成 **NIG + Student-t NLL**。这两项是 ERT 相对 [@delajara2026traj] 唯一的实质增量，必须单独量化，否则无法说明 ERT 不是其改写。

> [!important] 两个核心消融（本文立论所系）

**核心消融 A — 证据空间 vs 视觉表征空间（本文最关键对照）**
在**完全相同**的轨迹结构、锚定方式、转移模型、残差化融合下，只替换轨迹所处的空间：

| 变体 | 每层状态 $v^l$ | 说明 |
|:---|:---|:---|
| **ERT（本文）** | $[\log\alpha_1^l,\dots,\log\alpha_C^l,\log S^l]$ | EDL 证据空间 |
| 表征基线 [@delajara2026traj] | $z^l$（原生全局读出） | 视觉表征空间 |
| 紧凑证据对照 | $[p^l,u^l]$ | 去掉证据总量维，验证 $\log S^l$ 是否必要 |

结论指向：若 ERT 在**近 OOD** 与**冲突型样本**上显著优于表征基线，说明收益来自"证据空间显式分离空缺/冲突"，而非轨迹结构本身——这是 ERT 独立于 [@delajara2026traj] 的直接证据。紧凑对照掉点则证明 $\log S^l$（证据总量）不可或缺。为排除"仅因维度/坐标系不同"的混杂，附加一个把 $z^l$ 线性投影到 $C{+}1$ 维的等维对照。

**核心消融 B — NIG + Student-t NLL vs MSE 单点 surprise**
在相同证据空间与转移主干下，只替换转移模型输出头与意外度定义：

| 变体 | 输出头 | 意外度 | 说明 |
|:---|:---|:---|:---|
| **ERT（本文）** | NIG $(\gamma,\nu,\alpha,\beta)$ | 逐维 Student-t NLL | 分布式、含认知/偶然分离 |
| MSE 基线 [@delajara2026traj] | 点预测 + 固定 $\hat\sigma^2$ | 归一化平方误差 $D_{\text{ERT}}$ | 对应原方案 |
| 同方差 Gaussian NLL | 点预测 + 全局 $\sigma$ | Gaussian NLL | 隔离"重尾"与"逐样本方差"各自的贡献 |

结论指向：若 NIG 版在**近 OOD** FPR95 上优于 MSE 基线，说明逐样本、重尾的分布式转移建模对捕捉"证据演化落在尾部"有价值；进一步单独报告 NIG 的**认知不确定性** $\beta/(\nu(\alpha{-}1))$ 是否随近 OOD 升高，作为机理证据。

> [!note] 支撑性消融（验证设计选择，非立论核心）

| 维度 | 主方案 | 对照 / 消融 | 结论指向 |
|:---|:---|:---|:---|
| 打分范式 | **纯证据** $U+w_1\tilde G+w_2\tilde D$ | +Maha++ 基础项变体 | 量化放弃外部检测器在远 OOD 的代价与换回的独立性 |
| 证据分量 | 空缺+冲突+转移 | 去冲突 / 去转移 / 仅末层空缺 | 逐分量增益；验证三信号互补非冗余 |
| 锚定方式 | 硬锚定 | 软锚定 | 边界样本与近 OOD 的鲁棒性 |
| 转移结构 | 一步 MLP | GRU（时序）/ 无序集合编码器 | 有序明显更好 ⇒ 证据沿深度**有方向演化**；集合相当 ⇒ 只是各层独立统计起作用 |

---

# 实验

> [!todo] 待补充
> ImageNet 规模基准 · 近/远 OOD 数据集 · 骨干覆盖 · 与 X-Mahalanobis / Maha++ 的主表对比 · FPR95 / AUROC · 三种子均值方差 · 权重扫描稳健性审计 · 假设与评估一节 的 H1–H3 图表。

主结果表骨架（复杂多级表头直接写 LaTeX，Pandoc 原样保留）：

> [!example]- 展开：主结果表 LaTeX 代码
> ````latex
> \begin{table}[t]
>   \centering
>   \caption{ImageNet 规模下与逐层 post-hoc 基线的对比（三随机种子均值）。}
>   \label{tab:main}
>   \setlength{\tabcolsep}{4pt}
>   \begin{tabular}{lcccc}
>     \toprule
>     \multirow{2}{*}{方法} & \multicolumn{2}{c}{Near-OOD} & \multicolumn{2}{c}{Far-OOD} \\
>     \cmidrule(lr){2-3}\cmidrule(lr){4-5}
>      & AUROC$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & FPR95$\downarrow$ \\
>     \midrule
>     X-Mahalanobis~\cite{wei2025xmaha} & -- & -- & -- & -- \\
>     Maha++                            & -- & -- & -- & -- \\
>     \textbf{ERT (ours)}               & -- & -- & -- & -- \\
>     \bottomrule
>   \end{tabular}
> \end{table}
> ````

---

# 结论

> [!todo] 待补充
> 总结 ERT 将逐层表征建模为证据累积轨迹、显式分离空缺/冲突/转移三类信号、严格 post-hoc 融合的贡献与局限。

---

# 参考文献

正文用 Pandoc 引用语法 `[@key]`，转换时由 citeproc 依下表的 `.bib` 自动生成文献表。以下链接已核对（最后核对日期 2026-10-11）。

| 引用键 | 文献 | 链接 |
|:---|:---|:---|
| `delajara2026traj` | De la Jara I.M., Rodriguez-Opazo C., Damirchi H., Gould S., Ranasinghe D. *Representation Trajectories Matters: Complementary Evidence for OOD Detection and Image Classification.* 2026.（University of Adelaide / ANU / Naval Group Pacific — 本文最核心的区分对象，见下方说明） | [alphaXiv 2607.26565](https://www.alphaxiv.org/abs/2607.26565) |
| `wei2025xmaha` | X-Mahalanobis: Cross-Layer Mahalanobis Distance for OOD Detection.（⚠️ 真实元数据待核实——此为 `delajara2026traj` 引用的一个末层基线，非 2607.26565 本身） | 待确认 |
| `sensoy2018edl` | Sensoy M., Kaplan L., Kandemir M. *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS 2018. | [arXiv:1806.01768](https://arxiv.org/abs/1806.01768) · [DOI](https://doi.org/10.48550/arXiv.1806.01768) |
| `amini2020evidential` | Amini A., Schwarting W., Soleimany A., Rus D. *Deep Evidential Regression.* NeurIPS 2020. | [arXiv:1910.02600](https://arxiv.org/abs/1910.02600) · [DOI](https://doi.org/10.48550/arXiv.1910.02600) |

> [!warning] 关于 `wei2025xmaha` 的核对说明
> 后两篇（Sensoy 2018 / Amini 2020）已在 OpenAlex 与 arXiv 核实，标题、作者、年份、链接均确认无误。
> **重要更正：** alphaXiv ID `2607.26565` 的真实论文是 **De la Jara et al., *Representation Trajectories Matters*（2026）**，并非 X-Mahalanobis。该论文是本文最核心的区分对象（它已在**视觉表征空间**提出轨迹残差 + 一步 MLP + 残差化融合 + 身份连续性验证）。本文的增量仅在于：把轨迹迁至 **EDL 证据空间**、把 surprise 换成 **NIG + Student-t NLL**、改用**纯证据打分**。X-Mahalanobis 与 Maha++（Mahalanobis++）是该论文引用的**末层基线**，需各自单独检索正确元数据后补入 `.bib`。

供 Zotero / Pandoc 使用的 BibTeX 条目：

> [!example]- 展开：`main.bib` 条目
> ```bibtex
> @article{sensoy2018edl,
>   title   = {Evidential Deep Learning to Quantify Classification Uncertainty},
>   author  = {Sensoy, Murat and Kaplan, Lance and Kandemir, Melih},
>   journal = {Advances in Neural Information Processing Systems (NeurIPS)},
>   year    = {2018},
>   eprint  = {1806.01768},
>   archivePrefix = {arXiv},
>   url     = {https://arxiv.org/abs/1806.01768}
> }
>
> @article{amini2020evidential,
>   title   = {Deep Evidential Regression},
>   author  = {Amini, Alexander and Schwarting, Wilko and Soleimany, Ava and Rus, Daniela},
>   journal = {Advances in Neural Information Processing Systems (NeurIPS)},
>   year    = {2020},
>   eprint  = {1910.02600},
>   archivePrefix = {arXiv},
>   url     = {https://arxiv.org/abs/1910.02600}
> }
>
> @article{delajara2026traj,
>   title   = {Representation Trajectories Matters: Complementary Evidence for OOD Detection and Image Classification},
>   author  = {De la Jara, Ignacio M. and Rodriguez-Opazo, Cristian and Damirchi, Hamed and Gould, Stephen and Ranasinghe, Damith},
>   year    = {2026},
>   note    = {Preprint; verify final venue/identifiers before submission},
>   url     = {https://www.alphaxiv.org/abs/2607.26565}
> }
>
> % ⚠️ X-Mahalanobis 与 Maha++ 为上文引用的末层基线，真实元数据待各自核实后补全
> @article{wei2025xmaha,
>   title   = {X-Mahalanobis: Cross-Layer Mahalanobis Distance for Out-of-Distribution Detection},
>   author  = {TBD},
>   year    = {2025},
>   note    = {Baseline cited in delajara2026traj; metadata to be confirmed}
> }
> ```

> [!note] 投稿提示
> 正式投稿前用 Zotero Better BibTeX 导出 `main.bib` 并核对所有引用键，尤其是上面标 ⚠️ 的 `wei2025xmaha`。

<!--
Pandoc 转 CVPR .tex 命令（转换前请移除 callout/折叠等 Obsidian 专有语法，或改用 lua filter 剥离）：
pandoc 论文初稿.md -o main.tex --template=cvpr --number-sections \
  --filter pandoc-crossref --citeproc --bibliography=main.bib
章节映射：# → \section，## → \subsection，### → \subsubsection（不手动编号，交给 LaTeX 自动编号）。
-->
