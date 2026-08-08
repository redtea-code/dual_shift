# CAPM-ABL-P1：特征尺度、Patch-Table 交互消融与参数调整计划

> 历史计划声明：本计划固定了 `subjects_all_paired` 的 subject-wide 排除，
> 仅用于复现其对应 revision。后续 scan-filtered 实验不得沿用本计划的纳排、
> manifest 或结果目录；请使用 `34_scan_filtered_capm_execution_plan_2026-08-08.md`。

## 基本信息

- 日期：2026-08-07
- 负责人：cyh
- 状态：计划中
- Git commit：`dab4dd8`
- 关联文档：`docs/IE_CAPM_APIC_V3_2_ALIGNED_EXPERIMENT_PROTOCOL.md`、`docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`、`docs/APIS_APIC_V2_V3_V3_2_EXPERIMENT_LESSONS.md`、`review/plans/32_ie_capm_paired_experiment_plan_2026-08-07.md`
- 关联实现：`Model/ablation/scale_table_transformer.py`
- 关联测试：`tests/test_scale_table_transformer_ablation.py`
- 关联数据文件：正式运行前生成；当前无结果文件

## 1. 实验指导与依据

### 1.1 研究问题

本实验回答两个相互区分的问题：

1. CAPM 类融合发生在不同语义深度和空间分辨率时，性能与机制表现是否改变？
2. 在相同特征尺度下，使用 Transformer 建模 image patch 与三个表格变量的交互，是否优于卷积门控、图像自注意力和原始 patchwise CAPM？

核心候选不是独立的域泛化模块。它是在 CAPM 的变量特异空间场上增加影像证据约束：

```text
r_v(z_v)                         CAPM 的第 v 个变量空间场
g_v(F)                           由影像 patch 与表格 token 交互得到的门控
m(F,z) = sigmoid(sum_v g_v(F) * r_v(z_v))
F_out = F + Norm((1 - m(F,z)) * F)
```

当 `g_v(F)=1` 时严格退化为同一实现中的 CAPM 控制。Transformer cross-attention 的作用应由
`transformer_cross - transformer_self` 识别，不能只用 `transformer_cross - image_only` 归因。

### 1.2 假设与基线

- H1：相同 175-token 预算下，`layer3_patch2` 与 `layer4_pixel` 的差异主要反映语义深度和局部粒度差异，而不是 token 数量差异。
- H2：`layer5_pixel` 的 36-token 极端可以检验过度下采样是否损失细粒度人口学-影像交互。
- H3：若表格交互有效，`transformer_cross` 应稳定优于容量匹配的 `transformer_self`，且不能由 table-free 或卷积门控对照解释。
- H4：原始 patchwise CAPM 是历史机制对照；它的调制为 `X_adj=(2-gamma)X`，不等同于当前 CAPM，也不承担严格等价基线职责。其来源固定为 `redtea-code/Causal_fusion@d1a37d9`。

主要基线为同尺度 `capm`；机制基线为 `conv_gate` 和 `transformer_self`；纯影像基线为
`image_only`；历史对照为 `original_capm`。

### 1.3 数据与主张边界

本计划只支持“融合结构、预测性能和可审计交互机制”的结论，不支持以下表述：

- 扫描参数、场强或厂商效应的因果校正；
- 模型识别了 NACC 或 ADNI 中不可可靠识别的 scan 参数；
- 单个 seed 或单个方向证明了稳定域泛化；
- Transformer 注意力权重本身构成因果解释。

扫描参数、site、vendor、protocol 和 field strength 均不进入模型。两个方向分别报告，不使用双向平均掩盖
`NACC_to_ADNI` 的 unsupported-protocol 问题。

### 1.4 冻结任务、输入和纳排规则

```text
task          = MCI_vs_AD
labels        = MCI=0, AD=1
directions    = ADNI_to_NACC, NACC_to_ADNI
table         = [age, sex, education]
preprocessing = skullstrip+n4+mni+crop+normalize
input_shape   = [160, 196, 160]
split_seed    = 42
```

该尺寸必须在正式运行前从冻结 manifest 对应的实际 NIfTI 文件抽样核验，并把核验结果与 manifest hash 一起归档。
若实际尺寸不同，必须生成新的 protocol revision；禁止由数据加载器静默 resize 后继续使用本计划的 token 数。

表格变量固定为三个，顺序、缺失值处理和标准化在所有 table-aware 变体间完全一致。不得在本轮加入
MMSE、CDRSB、ADAS11、FAQ、APOE4、site 或任何采集参数。

使用 `data/claim/paired_holdout_subjects.json` 中的 `subjects_all_paired`，在任何 split 前从 ADNI
删除全部 73 个 1.5T/3T 配对 subject 的所有扫描、访视和表格记录。不得替换为 7 天或 30 天子集。
当 ADNI 是 source 或 external target 时均执行该排除；NACC 使用全部合格 subject。

| direction | source | source split | untouched external target |
| --- | --- | --- | --- |
| `ADNI_to_NACC` | 排除配对 subject 后的 ADNI | subject-level 60/20/20 | 全部合格 NACC |
| `NACC_to_ADNI` | 全部合格 NACC | subject-level 60/20/20 | 排除配对 subject 后的 ADNI |

同一 subject 的全部 scan/visit 必须位于同一 partition。6/2/2 是 subject 比例，不是 scan 比例。
External target 不参与训练、调参、early stopping、正则权重选择或 checkpoint 选择。

### 1.5 固定训练与评估协议

| 项目 | 固定值 |
| --- | --- |
| train / evaluation batch size | `4 / 2` |
| epochs | `50` |
| optimizer learning rate | `1e-4` |
| weight decay | `1e-4` |
| classification loss | class-weighted cross-entropy |
| training seeds | screening `42`；confirmation `42,43` |
| split seed | `42` |
| checkpoint selector | source-validation balanced accuracy；并列时选更早 epoch |
| primary endpoint | external target subject-mean balanced accuracy |
| bootstrap | subject-level，200 次 |
| label conflict | `earliest_visit` |

所有结构变体共享 backbone depth、分类头、数据增强、优化预算、sampler 和 checkpoint 规则。只有表中明确列为
自变量的结构或超参数允许改变。

## 2. 分阶段实验

### 2.1 E0：实现、几何与数值验收

真实数据训练前必须通过：

1. `layer3_patch2` 在 `160x196x160` 输入下得到 `10x13x10` 特征图，右侧补齐为 `10x14x10`，得到 `5x7x5=175` tokens；
2. `layer4_pixel` 得到 `5x7x5=175` tokens；
3. `layer5_pixel` 得到 `3x4x3=36` tokens；
4. patch 不能整除特征图时只能做显式右侧 padding，并在 audit 中记录 padding，禁止静默丢弃边界体素；
5. `force_capm=True` 的门控全为 1，FP32 CPU 下与公式路径最大绝对误差不超过 `1e-6`；
6. `original_capm` 在 `gamma=0` 时严格满足 `X_adj=2X`；较小的奇数特征维度只能通过记录在 audit 中的右侧 padding 对齐，超出预期几何时拒绝运行；
7. 普通训练前向不保存完整 attention matrix；只有 `return_audit=True` 时导出；
8. 所有变体完成 forward、backward、checkpoint save/reload，logits 和正则项无 NaN/Inf；
9. 实际运行的 `experiment_signature()` 与 resolved config 一致。

任一项失败则停止，不进入 E1。

### 2.2 E1：最小真实数据 smoke

固定 `seed=42`，每个方向仅使用 source train/validation，运行 2--3 epoch：

| ID | preset | interaction | 目的 |
| --- | --- | --- | --- |
| M0 | `layer4_pixel` | `image_only` | 验证纯影像路径 |
| M1 | `layer4_pixel` | `capm` | 验证严格 CAPM 控制 |
| M2 | `layer4_pixel` | `transformer_cross` | 验证完整候选路径 |

验收要求：损失下降、预测同时包含两类、sensitivity/specificity 均非零、梯度有限、token count 正确、
checkpoint reload 后固定 batch 的概率最大误差不超过 `1e-5`。失败时只修工程问题，不查看 external target。

### 2.3 E2：特征尺度与交互结构消融

E2 只用 `seed=42` 的 source validation 做筛选，两个方向采用相同矩阵和选择规则。

#### E2a：尺度 × Transformer 信息来源

| ID | preset | tokens | interaction | 解释 |
| --- | --- | ---:| --- | --- |
| S1 | `layer3_patch2` | 175 | `transformer_self` | 浅层、patch-level 容量对照 |
| S2 | `layer4_pixel` | 175 | `transformer_self` | 深层、pixel-level 容量对照 |
| S3 | `layer5_pixel` | 36 | `transformer_self` | 极粗粒度容量对照 |
| S4 | `layer3_patch2` | 175 | `transformer_cross` | 浅层 patch-table 交互 |
| S5 | `layer4_pixel` | 175 | `transformer_cross` | 深层 pixel-table 交互 |
| S6 | `layer5_pixel` | 36 | `transformer_cross` | 极粗粒度 table 交互 |

S1/S2 是等 token 预算比较，但 backbone stage 和通道数仍不同，因此结论只能写为“整体尺度配置差异”，
不能写成纯粹的 patch-size 因果效应。`transformer_cross - transformer_self` 才是表格交互的直接增量。

#### E2b：最佳尺度下的交互机制矩阵

最佳尺度只由 source validation 选择。固定该尺度后运行：

| ID | interaction | 目的 |
| --- | --- | --- |
| I0 | `image_only` | 纯影像基线 |
| I1 | `capm` | 当前 CAPM 主基线 |
| I2 | `conv_gate` | IE-CAPM 卷积影像门控 |
| I3 | `original_capm` | Causal_fusion 原始 patchwise 历史对照 |
| I4 | `transformer_self` | Transformer 容量与图像上下文对照 |
| I5 | `transformer_cross` | 完整 patch-table 交互候选 |

尺度选择规则按顺序执行：

1. 淘汰类别塌缩、NaN/Inf、token/几何不一致或 reload 不一致的配置；
2. 最大化 source-validation subject-mean BA；
3. BA 差异小于 `0.005` 时，选择参数量更小、token 更少的配置；
4. 仍并列时按 `layer4_pixel`、`layer3_patch2`、`layer5_pixel` 的预注册顺序选择。

不得根据 external target、attention 图是否“好看”或单个 scan 结果选择尺度。

### 2.4 E3：Transformer 与正则参数调整

仅对 E2 选出的 `transformer_cross` 尺度调参。采用逐组、有限候选搜索；每组选择后冻结，再进入下一组。
所有候选只用 source validation，`seed=42`。

#### E3a：模型容量

| 组 | transformer_dim | num_heads | dropout |
| --- | ---:| ---:| ---:|
| C0 | 64 | 4 | 0.1 |
| C1 | 128 | 4 | 0.1 |
| C2 | 256 | 8 | 0.1 |

#### E3b：随机正则与初始化

固定 E3a 胜出容量后，分别单因素筛选：

```text
transformer_dropout = [0.0, 0.1, 0.2]
gate_init           = [0.80, 0.90, 0.95]
```

`classifier_dropout=0.3`、FFN ratio、学习率和 weight decay 在本轮保持不变，避免同时改变优化器与结构。

#### E3c：机制正则权重包

禁止做任意组合式大网格，只比较三个预注册权重包：

| bundle | basis_tv | basis_orth | gate_anchor | gate_floor | modulation_preservation |
| --- | ---:| ---:| ---:| ---:| ---:|
| R0 | 0 | 0 | 0 | 0 | 0 |
| R1 | `1e-5` | `1e-5` | `1e-4` | `1e-3` | `1e-3` |
| R2 | `1e-4` | `1e-4` | `1e-3` | `1e-2` | `1e-2` |

参数选择仍以 source-validation subject-mean BA 为第一排序。若差异小于 `0.005`，依次选择更小容量、
更低 dropout、`gate_init` 更接近默认 `0.95`、更弱正则。选择过程、失败候选和完整 source-validation
结果必须保留，不能只保存胜出项。

### 2.5 E4：冻结确认矩阵与一次性 target 评估

两个方向的配置必须在打开任一方向 external target 结果前同时冻结并写入只读 resolved config。
每个方向只能利用自己的 source train/validation 选择参数；不得把一个方向的 target 表现反馈给另一个方向。

确认矩阵：

```text
2 directions x 2 seeds x 6 interactions = 24 runs
seeds = 42, 43
interactions = image_only, capm, conv_gate, original_capm,
               transformer_self, transformer_cross
```

E2/E3 中与最终冻结配置、commit、manifest 和训练协议完全一致的 seed42 运行可以复用，否则必须重跑。
所有 source-selected checkpoint 锁定后，才一次性执行 source test 和 external target 推理。

### 2.6 E5：条件性扩展

只有 E4 的 screening gate 通过后才允许：

- 增加 seeds `44,45,46`，形成五 seed 稳定性矩阵；
- 在不改变模型输入的前提下，做 demographic subgroup 与 common-support 描述性分析；
- 实现 table-conditioned gate negative control，检验收益是否可由表格 shortcut 单独解释；
- 研究第二个 Transformer block、不同 FFN ratio 或更多 patch 尺度；
- 重新运行完全对齐的 APIC v3_2 作为历史方法比较。

E4 未通过时停止并归档，不通过追加 patch size、扩大 Transformer 或查看 target 后重新调参追逐正结果。

## 3. 预注册判定标准

### 3.1 工程 Gate G0

E0/E1 全部通过；任何数值不一致、隐性体素丢弃、attention 显存异常、NaN/Inf、类别塌缩或 checkpoint
不可复现均为 No-Go。

### 3.2 结构 Gate G1

`transformer_cross` 只有同时满足以下条件才作为主候选：

1. 在两个方向的 source validation 均不发生类别塌缩；
2. 相对同尺度 `transformer_self` 的 seed42 BA 增量在两个方向同号；
3. 相对同尺度 `capm` 没有任一方向超过 `0.02` 的绝对 BA 损失；
4. gate mean 不低于实现中的 `min_gate_mean=0.65`，effective/raw field ratio 不低于 `0.60`；
5. attention、gate、raw/effective field 均有限，且不同输入样本的 gate 不完全恒定。

G1 只决定是否值得进入确认矩阵，不构成 target 性能结论。

### 3.3 确认 Gate G2

E4 后分别对每个方向判断，不以双向平均替代：

1. `transformer_cross - capm` 的 target subject-mean BA 在 seeds 42/43 中方向一致；
2. 两 seed 平均 delta BA 为正；
3. `transformer_cross - transformer_self` 的平均 delta BA 为正；
4. sensitivity 或 specificity 不得为 0，预测分布不得塌缩；
5. AUROC、Brier/ECE 和 subgroup gap 不出现与 BA 增益明显冲突的系统性恶化；
6. gate 与 effective-field 摘要在两个 seed 间保持同一数量级。

通过 G2 只允许进入五 seed 扩展，不能直接声称“稳定显著增益”。五 seed 结果必须报告逐 seed 配对差值、
胜负计数、均值/标准差和 subject-level paired bootstrap 95% CI。

## 4. 可复现记录

- 配置文件：待新增冻结 YAML；建议命名 `journal_capm_scale_transformer_ablation_mci_ad.yaml`
- 数据与划分版本：复用 APIC v3_2 对齐后的排除 manifest；记录 manifest SHA-256
- 配对排除文件：`data/claim/paired_holdout_subjects.json`；记录 SHA-256 和实际排除 ID/数量
- 随机种子：`split_seed=42`；screening `42`；confirmation `42,43`
- 环境与硬件：Windows，`D:/Anaconda/envs/segment/python.exe`；每个方向避免跨硬件混跑
- 启动命令：launcher/config 完成后冻结；当前无正式命令
- 工作区状态：正式运行必须使用 clean commit；当前计划编写时工作区 dirty
- 产物位置：建议 `outputs/journal/capm_scale_transformer_ablation_mci_ad/`
- 结果归档：建议 `review/records/capm_ablation/<environment>/`

每个 `phase × direction × seed × preset × interaction` 单元必须保存：

- resolved config、config hash、git commit、环境与 GPU 信息；
- split manifest、subject 交集检查和排除清单；
- source-validation checkpoint 选择历史；
- source test 与 target 的 scan-level 原始预测及 subject-level 聚合预测；
- token grid/count、参数量、峰值显存和训练时间；
- gate、raw field、effective field、modulation 以及按 head/变量汇总的 attention；
- 200 次 subject bootstrap 的指标和失败原因。

规范长表唯一键为：

```text
task, direction, seed, preset, interaction, split, aggregation, status
```

未运行、失败和不适用分别写为 `not_started`、`failed`、`NA`，不得用 `0.5` 或空值替代。

## 5. 分析与结果

### 5.1 结果表

| phase | direction | seed | preset | interaction | source-val BA | target BA | AUROC | Brier/ECE | gate/effective ratio | status |
| --- | --- | ---:| --- | --- | ---:| ---:| ---:| --- | --- | --- |
| E2 | 待运行 | 42 | `layer3_patch2` | `transformer_self` | NA | 不读取 | NA | NA | NA | `not_started` |
| E2 | 待运行 | 42 | `layer4_pixel` | `transformer_cross` | NA | 不读取 | NA | NA | NA | `not_started` |
| E4 | 待运行 | 42/43 | frozen | `capm` | NA | NA | NA | NA | NA | `not_started` |
| E4 | 待运行 | 42/43 | frozen | `transformer_cross` | NA | NA | NA | NA | NA | `not_started` |

### 5.2 分析要求

- 相对基线：逐方向、逐 seed 报告相对 CAPM 和 Transformer-self 的绝对 delta；
- 尺度解释：同时报告 feature shape、token count、通道数和参数量，禁止只写 patch size；
- 机制解释：attention 仅作交互审计，必须结合 gate 与 effective/raw field ratio；
- 异常与局限：完整保留失败运行、类别塌缩、显存不足和缺失指标；
- 结果结论：分别判断工程可行性、机制非恒等、预测增益和跨 seed 稳定性，不互相替代。

## 6. 建议下一步实验指导

- 建议动作：先实现冻结配置与 launcher，完成 E0/E1，再执行 E2 的 12 个 source-only seed42 单元；
- 建议依据：当前代码已有三种尺度和六种交互接口，但尚无与训练入口绑定的正式配置；
- 固定条件：三个人口学变量、73 名配对 subject 排除、subject-level 6/2/2、MCI_vs_AD、两个方向和主终点保持不变；
- 进入条件：G0 通过后进入 E2，G1 通过后进入 E3/E4，G2 通过后才增加三枚 seed；
- 禁止事项：不使用 target 调参，不追加临床量表或 scan 参数，不按 scan 划分，不删除失败单元，不以注意力图替代性能和机制证据。

## 提交前检查

- [x] 研究问题、基线、数据边界和阶段 Gate 已预先定义。
- [x] 三变量、配对 subject 排除和 subject-level 6/2/2 已固定。
- [x] 参数候选、选择顺序和 target 隔离规则已固定。
- [ ] 正式配置、launcher、manifest hash 和启动命令待实现后补充。
- [ ] 结果产生后补齐主表、失败清单、bootstrap 和机制审计。
