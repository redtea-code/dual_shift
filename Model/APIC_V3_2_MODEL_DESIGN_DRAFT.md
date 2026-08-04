# APIC v3_2 模型设计草案

**状态：** 设计审阅版；当前仓库代码仅为机制原型，正式 revision 4 实验默认阻断
**方法族：** `APIC_v3_2`
**主代码变体：** `v3_2_balanced_style_memory`
**主实验名：** `apic_v3_2_x`
**协议修订：** `protocol_revision: 4`

> 实现状态说明（2026-08-04）：`Model/dual_shift/apis_v3_2.py` 已提供教师冻结、固定
> prototype、support gate、相对统计迁移和审计接口，但尚未满足本文第 4--12 节的全部冻结
> 约束。差距与关闭条件见
> `review/analysis/27_apic_v3_2_implementation_review_2026-08-04.md`。在该审阅的 P0/P1
> 条目关闭前，配置中的 `formal_run_allowed` 必须保持 `false`。

## 1. 设计动机与证据

APIC v3 在 CN vs AD 与 MCI vs AD、ADNI/NACC 双方向、seed 42/43 的八个
checkpoint 诊断中表现出一致的近恒等退化：

- `condition_gate` 约为 `0.025--0.046`；
- layer1 相对 RMS 约为 `3e-8--1e-6`，layer2 约为 `6e-5--1.2e-4`；
- embedding cosine distance 与 JS divergence 接近 0；
- 6100 个 target scan 中仅出现 1 次 clean/shifted 预测翻转；
- 多数 checkpoint 的 prototype 硬分配由单一 slot 支配；
- 同时存在 train 与 source-validation/target 之间的明显过拟合。

因此 v3_2 不把失败解释为“扰动太强”，而将首要问题定义为：

```text
非平稳且塌缩的 style memory
    -> assignment 高熵、低置信度
    -> 乘法 gate 被压到接近 0
    -> shifted path 退化为 clean path
    -> APIC 不再提供有效训练增强
```

v3_2 的目标不是无条件放大扰动，而是建立一个可验证的闭环：

1. 样式坐标系在训练期间保持稳定；
2. 原型具有可审计的 source support；
3. 原型能明确映射为浅层特征统计目标；
4. 只对受支持样本实施非零且有上界的干预；
5. 不受支持样本严格回退到 clean path；
6. shifted path 不重复污染 BatchNorm running statistics；
7. 推理仍只使用 clean image path。

## 2. 研究边界

APIC v3_2 的主分析仍为严格 image-only：训练和推理的诊断路径只接收预处理后的
3D T1 MRI `X`。field strength、manufacturer、scanner、sequence、TR/TE/TI 等采集字段：

- 不进入 style teacher、prototype bank、gate、分类头或 checkpoint 选择；
- 只用于训练完成后的描述性分层与机制审计；
- 不用于定义或修正 prototype；
- 不支持“学习 MRI 参数因果效应”或“连续扫描参数校正”的主张。

NACC 的多个扫描字段近零方差，且 ADNI/NACC 的协议支持不对称。因此本方法只能主张
source-observed image style support 内的训练期稳健化，不能主张目标协议适配。

## 3. 总体结构

```text
clean warm-up
    -> freeze shallow style teacher T
    -> source-train descriptor pass
    -> robust scaling + PCA
    -> balanced fixed prototype bank P

training image x
    -> student clean path -> clean logits
    -> teacher descriptor s(x)
    -> source-support test + alternative prototype selection
    -> prototype layer-stat target
    -> bounded layer1/layer2 statistic residual
    -> shifted continuation reusing clean-batch BN moments without a second update
    -> shifted logits

inference image x
    -> student clean path only -> logits
```

主版本不包含标签对抗器、target-domain memory、测试时适配、图像生成器或额外分割网络。

## 4. 稳定样式教师

### 4.1 教师构造

先用 clean CE 完成配置中预先冻结 epoch 数的 warm-up。warm-up 长度不得根据 source-validation
或 external target 表现选择；三种正式比较方法共享相同总优化 epoch，B0 描述符遍历不计入优化步数。
warm-up 结束后对 student 的 `stem + layer1 + layer2` 做 deep copy（不得共享 parameter/buffer
storage）得到教师 `T`，并永久设置为 `eval()` 与 `requires_grad=False`。教师参数、BN running
statistics 和样式投影在后续训练中均不更新。

这一设计使同一个样本在不同 epoch 的样式坐标可比较，避免 student 特征漂移与 EMA memory
互相追逐。

### 4.2 显式样式描述符

对教师的 layer1、layer2 特征计算：

\[
d(x)=\operatorname{concat}(\mu_1,\log\sigma_1,
\mu_2,\log\sigma_2)
\]

其中通道均值和标准差按空间维计算，标准差在取对数前固定使用配置中的 `eps_sigma` 下界。
为避免首版增加未冻结的频率分箱定义，`v3_2-A` 不使用频率能量；频率补充只能作为后续独立
消融。所有 robust center、scale 以及 PCA 均只在 source-train 的
`mechanism_fit` 子集上拟合并冻结：

\[
s(x)=\operatorname{PCA}_{D}(\operatorname{RobustScale}(d(x))),\quad
D=\min(16,\operatorname{rank}_{fit})
\]

robust scaling 固定为逐维 median/IQR；零或低于 `eps_scale` 的 IQR 维度从描述符中删除，而不是
除以任意小量。PCA 不 whitening，原型距离固定为 PCA 空间欧氏距离。主版本不使用可训练
style encoder，以消除随机投影与分类损失共同漂移造成的不可辨识性。

`mechanism_fit` 与 `mechanism_calibration` 是从 source-train subject 中一次性、subject-disjoint
划出的机制子集；二者仍可进入分类训练，但后者不得参与 robust scaler、PCA、prototype center
或 layer-stat target 的拟合，只用于冻结 support、RMS 与 occupancy 阈值。拆分清单及 SHA-256
必须在 E1/E2 前固定。

## 5. 平衡固定原型库

### 5.1 初始化

在 `mechanism_fit` 的 `s(x)` 上一次性建立 `K=4` 个原型。初始化采用配置中固定 `n_init`、
初始化种子和最大迭代数的 weighted k-means。每个 scan 的权重为其 subject 扫描数的倒数；
若有效维度为 0、不同描述符或 subject 数少于 K，或者没有任何 restart 满足预注册的最小
subject occupancy，B0 直接失败，不通过丢弃小簇、复制中心或临时改变 K 修复。该定义避免
多次扫描受试者主导中心。

每个原型同时保存：

\[
p_k=(c_k,\mu^{(1)}_k,\sigma^{(1)}_k,
\mu^{(2)}_k,\sigma^{(2)}_k,n_k,r_k)
\]

其中 `c_k` 是 PCA 样式中心；layer 均值目标固定为该簇逐通道 weighted median，layer 标准差
目标固定为逐通道 weighted median log-standard-deviation 再取指数，权重均为 subject-balanced。
`n_k` 是有效 subject 支持量。`r_k` 使用
`mechanism_calibration` 中被分配到该簇的距离按 subject-balanced weighted quantile 冻结为预注册
分位数；若校准支持量不足或
`r_k <= eps_distance`，B0 失败。

这一定义解决了 v3 中“latent target 无法明确映射回 layer feature target”的问题。

### 5.2 冻结策略

`v3_2-A` 的 prototype bank 在正式训练中完全冻结。只有 `v3_2-A` 同时通过机制 Gate M0
和性能 Gate S1 后，才允许研究 `v3_2-B` 的 queue + balanced Sinkhorn EMA 更新。

主实验不得将“每个 slot 已初始化”报告为“每个 slot 被有效使用”；必须同时报告有效 slot、
occupancy、assignment entropy、簇间距、样本支持量，以及冻结后在 source-validation 上计算的
assignment-label 关联。标签仅用于拒绝诊断相关原型，不参与原型拟合、目标选择或参数调整。该
审计使用每个 subject 的多数硬 assignment 和一次 subject-level 标签，NMI 固定为
`normalized_mutual_info_score(average_method="arithmetic")`，不得按 scan 数重复计权。

### 5.3 目标选择

对样本先找到事实原型 `k_src`，再从下列候选中选择 `k_tgt`：

- 排除 `k_src`；
- `n_k` 达到最小支持量；
- 原型间距离位于 `mechanism_fit` 冻结的 `[delta_min, delta_max]`；
- 优先从两个最近的合法替代原型中确定性采样。

采样必须使用无状态 hash，例如 `(run_seed, epoch, subject_id, scan_id)`；checkpoint 诊断固定使用
该 checkpoint 的 epoch，正式 clean inference 不执行该采样。不得依赖 DataLoader 顺序、worker
数或设备数；同一样本的 layer1/layer2 使用同一 `k_tgt`。不得使用 target cohort
信息、诊断标签或扫描元数据选择目标。

## 6. Support-aware 门控

### 6.1 硬支持掩码

定义：

\[
m_i=I[d(s_i,c_{src})\le r_{src}]
I[n_{src}\ge n_{min}]I[n_{tgt}\ge n_{min}]
I[\delta_{min}\le d(c_{src},c_{tgt})\le\delta_{max}]
\]

任何条件不满足时，`m_i=0`，shifted path 必须与 clean path 数值一致。该回退是模型语义，
不能被损失函数覆盖。

### 6.2 非退化确定性门控

受支持样本使用：

\[
g_i=m_i\left[g_{min}+(g_{max}-g_{min})
\operatorname{Clip}_{[0,1]}\left(1-\frac{d(s_i,c_{src})}{r_{src}}\right)\right]
\]

`v3_2-A` 的 gate 不可训练且不接收 anatomy summary、标签或分类梯度。它只按样本在事实簇内的
相对位置确定强度；目标分离度仅作为 6.1 的硬合法性条件。这样避免多个小量把 gate 压到近零，
也避免诊断相关解剖信号隐式控制增强强度。

候选初值为 `g_min=0.20`、`g_max=0.80`。最终值只可通过 mechanism calibration 与合成机制
实验按照预提交的有限候选集和确定性排序规则一次性冻结；查看 E1 输出后不得追加候选，查看
source-validation 分类性能或 external target 后均不得修改。

## 7. 可解释的浅层干预

教师原型的绝对统计不能直接套用到持续更新的 student 特征上。对事实原型 `src` 与目标原型
`tgt`，先在冻结教师坐标中计算相对统计位移，再施加到 student 当前统计：

\[
\mu^{(l)}_{i\rightarrow tgt}=\mu(F_l)+
\rho_{\mu,l}(\mu^{(l)}_{tgt}-\mu^{(l)}_{src})
\]

\[
\sigma^{(l)}_{i\rightarrow tgt}=\sigma(F_l)\odot
\exp\{\rho_{\sigma,l}(\log\sigma^{(l)}_{tgt}-
\log\sigma^{(l)}_{src})\}
\]

`rho` 首轮固定为 1，并对均值位移和 log-scale ratio 使用 source-train 分位数裁剪。这样 target
语义是明确的“从 source style A 向 source style B 移动”，同时不要求 student 的绝对特征统计
在后续 epoch 与 warm-up 教师保持相同。

干预点固定为现有 `DualShiftBackbone` 中整个 `layer1`/`layer2` stage 的输出，即 residual addition
和末尾 ReLU 之后、进入下一 stage 之前的位置，与现有 APIC hook 保持一致。随后对 layer `l` 的
student 特征 `F_l` 构造目标统计变换。layer1 的 `F_l` 是 clean stage1 输出；layer2 的 `F_l` 是
已经传播 layer1 干预后的 shifted stage2 输出，不能错误地从 clean layer2 特征另起一条不相连的
分支：

\[
\hat F_l=\sigma^{(l)}_{i\rightarrow tgt}
\frac{F_l-\mu(F_l)}{\sigma(F_l)+\epsilon}
+\mu^{(l)}_{i\rightarrow tgt}
\]

再使用受限残差：

\[
F_l^{shift}=F_l+g_i\alpha_l
\operatorname{ClipRelativeRMS}(\hat F_l-F_l;F_l,r^{(l)}_{base,max})
\]

对每个样本定义 `RelRMS(A;F)=RMS(A)/(RMS(F)+eps_rms)`，并定义：

\[
\operatorname{ClipRelativeRMS}(A;F,r)=
A\min\left(1,\frac{r}{\operatorname{RelRMS}(A;F)+\epsilon_{rms}}\right)
\]

RMS 均覆盖该样本的 channel 与空间维。该算子只缩小超过基础上界的残差，不放大零残差；实际
干预的 relative RMS 使用 `F_l^{shift}-F_l` 重新计算。非零下界由 source-only calibration 冻结
`alpha_l` 并由 Gate M0 审计，不作为模型可优化的损失。

首版只在 layer1 和 layer2 干预；layer3/layer4 不直接改动。每层分别记录真实 relative RMS，
不能再用系数范数代替实际干预强度。

## 8. 双路径与 BatchNorm

每个训练 batch 先执行 clean path，允许其按正常语义更新 BN running statistics，并缓存每个
BN 层本次 clean forward 实际使用的 batch mean/variance。随后执行 shifted continuation；它必须
复用对应 clean forward 的 batch moments（缓存值 stop-gradient），使用相同 affine 参数，但不再
更新 running statistics。禁止让 clean 使用 batch statistics、shifted 改用历史 running statistics，
否则零干预时两条路径也不构成数值对照。

实现接口应显式支持类似：

```text
bn_context = backbone.forward_clean(..., update_running_stats=True)
backbone.forward_shifted(..., bn_moments=bn_context.detached_moments,
                          update_running_stats=False)
```

网络应拆为可复用的 stage continuation。若存在 dropout 或其他随机层，shifted continuation 必须
复用 clean mask，或在两条路径中同时禁用该随机层。禁止依赖一次完整 clean forward 加一次默认
train forward，因为它会让 shifted branch 额外更新 BN。对 `m_i=0` 的样本，最终 shifted
embedding/logits 直接复用 clean 值，保证严格回退；辅助损失也不得纳入这些样本。

当前分类头含 dropout，因此训练时必须显式生成一次 dropout mask，并将同一 mask 分别施加到
clean/shifted pooled embedding 后再计算 logits；不得通过两次普通 `nn.Dropout.forward` 隐式取得
不同 mask。source-validation 与推理保持 `eval()`，不使用 dropout。

validation、source-test、external target 和正式推理全部只走 clean path。source-validation 的
shifted checkpoint 诊断必须从同一次 `eval()` clean forward 分叉，clean 与 shifted 均使用相同的
冻结 running statistics；它仅供机制诊断，不进入正式性能表。external target 不参与 checkpoint
机制诊断或选择，正式训练结束后只做一次 clean-path 评估。

## 9. 损失函数

对所有样本计算 clean CE。分类损失按样本保持总权重为 1，避免 supported 样本因多一条分支而
被隐式加权：

\[
L_{cls}=\frac{1}{B}\sum_i\left[
\left(1-\frac{m_i}{2}\right)CE(y_i,z_i^{clean})+
\frac{m_i}{2}CE(y_i,z_i^{shift})\right]
\]

\[
L=L_{cls}+\lambda_{js}L_{JS}+\lambda_{deep}L_{deep}
+\lambda_{style}L_{style}
\]

- `L_JS`：clean/shifted 预测分布一致性；
- `L_deep`：dropout 之前的 layer4 或最终 pooled image embedding 的 cosine consistency；
- `L_style`：shifted layer1/layer2 的均值、标准差向 7 节定义的 student-relative target 靠近；
- 无支持样本时，整个目标严格退化为 `L_cleanCE`。

所有 shifted 辅助项均定义为 `B^{-1}\sum_i m_i\ell_i`，而不是除以 `sum(m_i)`；这样每个支持
样本的最大权重不随 batch 中支持数量变化。`L_style` 对每层的均值与 log-standard-deviation
使用逐通道 Huber 后先按通道、再按层取均值。零支持 batch 返回可微的数值 0；`m_i`、目标原型
选择和 BN moments 均 stop-gradient。supported fraction 必须按类别与 subject 另行报告，以检查
残余的标签条件增强。

RMS band 是机制校准与失败判定指标，不作为可优化损失，防止 student 通过放大分母、扭曲特征
或饱和 gate 来“学会通过”审计项。revision 4 在 E1 前固定 band 为 layer1
`[0.001, 0.03]`、layer2 `[0.003, 0.05]`，不得把 band 宽度作为提高 hit rate 的搜索变量；
`alpha_l` 与 strength ramp 再由预提交的有限候选集和 `mechanism_calibration` 无标签统计一次性选择。

删除 v3 的单向 `lambda_intervention * coefficient_l2` 收缩项，因为该项与一致性损失共同奖励
零干预。首轮不加入 label adversary；若后续诊断显示 style assignment 与 label 的关联明显高于
扫描属性且跨 seed 稳定，再作为 `v3_2-B` 独立消融。

## 10. 训练阶段

| 阶段 | epoch/动作 | 可训练部分 | APIC 状态 |
|---|---|---|---|
| W0 | clean warm-up | student backbone + classifier | 关闭 |
| B0 | source descriptor pass | 无梯度 | 建立并冻结 teacher/PCA/prototypes |
| W1 | strength ramp | student | fixed gate；supported-only，`alpha` 线性升高 |
| J0 | joint training | student | 固定 teacher/prototypes/gate，使用相对统计位移，RMS band 仅审计 |
| Eval | validation/test/target | 无 | clean path only |

若 B0 未满足有效原型数、最小 occupancy 或分离度要求，run 应标记为
`mechanism_not_initialized` 并停止，不得自动退回单 prototype 后继续报告 APIC 结果。

## 11. Checkpoint 选择

三种正式方法使用完全相同的 performance selector，且只使用 source-validation 的 subject-level
聚合结果：

1. collapse guard：sensitivity、specificity 和逐类 recall 均不低于冻结阈值；
2. 主排序：3-epoch EMA 的 balanced accuracy；epoch 1--2 使用截至当前的可用窗口；
3. 依次以 macro-F1、AUC、较低 Brier 和较早 epoch 打破并列。

APIC v3_2 在上述 performance selector 选出的唯一 checkpoint 上额外接受 mechanism guard：最近
3 个可用 epoch 内有效 prototype、supported fraction、RMS band hit rate 和 BN 对照均须合格。
不得因 mechanism guard 失败而改选另一个 checkpoint；该 run 直接判为机制失败。可以附加报告
“最佳机制合格 checkpoint”，但它不进入主性能比较。日志字段必须命名为实际指标，不得把
composite score 写成 `val_auc`。

## 12. 强制审计输出

每个 epoch 至少保存：

- `prototype_subject_counts`、`effective_slots`、`max_slot_share`；
- assignment entropy、nearest/target distance 分位数、source-validation assignment-label NMI；
- `supported_intervention_frac` 与各失败原因计数；
- gate mean/p05/p50/p95；
- layer1/layer2 relative RMS p05/p50/p95 与 band hit rate；
- shallow style target progress；
- clean/shifted CE、JS、deep cosine distance、prediction flip rate；
- clean 与 shifted 分支的 BN running-stat update 次数、moment 来源和配对标识；
- source-validation subject-level BA、macro-F1、AUC、Brier 和 collapse guard。

checkpoint 必须保存 teacher/PCA/prototype 指纹、配置 hash、split manifest hash、Git commit 和
完整环境指纹。teacher、robust scaler、PCA 与 prototype 的完整状态必须随 checkpoint 保存，或由
不可变 artifact URI 加 SHA-256 引用；只有指纹而无法恢复状态的 checkpoint 不视为可复现。

## 13. 失败判定

以下任一情况均视为机制失败，不应用 target 性能为其辩护：

- prototype 初始化不足或单 slot 支配超过冻结阈值；
- supported fraction 长期为 0，或不受支持样本仍产生非零 shift；
- supported 样本实际 RMS 长期低于 band；
- shifted branch 二次更新 BN running statistics，或未使用与对应 clean forward 相同的 BN moments；
- clean reload 概率不能在 `1e-5` 内复现正式预测；
- source-validation 出现单类预测或明显性能塌缩；
- source-validation assignment-label 关联或逐类 supported fraction 差异超过冻结上界；
- 只有个别 seed 改善而整体 Gate S1 不通过。

## 14. 分阶段实现建议

### APIC v3_2-A：主实现

只实现固定教师、source-only PCA、平衡固定原型、显式 layer-stat target、support gate、
supported-only RMS band、BN 隔离和 BA 对齐选点。这一版本是首轮性能筛选的唯一候选。

### APIC v3_2-B：条件性研究

仅在 v3_2-A 同时通过机制和性能 Gate 后研究：

- queue + balanced Sinkhorn 的缓慢 prototype EMA；
- `K=8` 与 `K=4` 的容量消融；
- 独立 label-adversary 消融；
- X+D 扩展轴。

v3_2-B 不得与 v3_2-A 混用同一结果目录或覆盖其 checkpoint。
