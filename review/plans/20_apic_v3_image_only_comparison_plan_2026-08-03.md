# APIC-V3-S1：Image-only 多任务双向两种子对比实验计划

## 基本信息

- 日期：2026-08-03
- 负责人：cyh / 远程实验协作者
- 状态：计划中；代码与合成 smoke 已完成，远程真实数据预检后方可启动正式矩阵
- Git commit：`ffc57b3`（当前 `v3_style_memory` 实现基线）
- 关联文档：`docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`、`review/plans/19_support_aware_paired_protocol_execution_plan_2026-08-03.md`
- 关联数据文件：`E:/2.causal/ADNI_NACC_descriptive_statistics.xlsx`；正式 screening manifest 与结果文件待生成

## 1. 实验指导与依据

### 1.1 研究问题

在不向模型提供 scanner/acquisition metadata、target 标签或 target 统计量的前提下，APIC v3 的图像风格记忆与有界残差干预，是否比相同输入模态下的 clean CE 和 MixStyle 获得更稳定的 ADNI/NACC 双向跨队列泛化？

本计划中的论文方法名冻结为 **APIC v3**。当前仓库代码仍使用：

```text
method_family = APIC_v3
code_variant  = v3_style_memory
legacy_label  = APIS_v3
```

在代码完成统一改名之前，配置、日志和结果表必须同时保存 `method_family` 与 `code_variant`，不得把 APIC v3 与 APIS v2 结果混写。

### 1.2 数据依据与主张边界

NACC 的场强、TI、flip angle、matrix rows 和 sequence family 等多个字段为零方差或近零方差，且 ADNI/NACC 的厂家和协议支持差异明显。因此：

- acquisition metadata 不进入 APIC v3、CE 或 MixStyle 的诊断路径；
- acquisition metadata 只允许用于描述性统计、评价分层和独立机制分析；
- 不主张模型学习了 MRI 参数的因果效应或连续响应；
- 不允许用 NACC->ADNI 1.5T 结果声称模型从 NACC 学会了 1.5T 校正；
- 两 seed 结果只作筛选，不作显著性或最终性能主张。

### 1.3 模态定义

| 符号 | 模态 | 字段/内容 | 在主分析中的作用 |
|---|---|---|---|
| `X` | MRI image | 预处理后的 3D T1 MRI | 唯一主分析输入 |
| `D` | demographics | age、sex、education 及其 missingness mask | 仅用于独立扩展轴 |
| `A` | acquisition | field strength、manufacturer、sequence、TR/TE/TI、flip angle、spacing、matrix、slices、site | 不输入模型；仅作审计/分层 |
| `Y` | diagnosis | CN、MCI、AD 标签 | 监督目标 |

“Image-only”在本文中严格指 **模型诊断路径只接收 `X`**。仅仅不使用 `A`、但仍使用 `D` 的模型必须标记为 `X+D`，不得标记为 image-only。

### 1.4 模型与公平对照

#### 主分析轴：严格 image-only

| 冻结名称 | 输入 | 训练期干预 | 推理路径 | 目的 |
|---|---|---|---|---|
| `ce_x` | `X` | 无 | clean image path | 图像基线 |
| `mixstyle_x` | `X` | 浅层随机 channel mean/std mixing | clean image path | 主要风格增强基线 |
| `apic_v3_x` | `X` | image-derived style memory + content gate + bounded low-rank residual | clean image path | 提出方法 |

三者必须使用相同 backbone、分类头、预处理、训练预算、batch size、优化器、source split 和 checkpoint 选择规则。除方法定义所必需的组件外，不允许改变容量或训练预算。

#### 扩展轴：人口学融合

| 冻结名称 | 输入 | 训练期干预 | 推理路径 | 目的 |
|---|---|---|---|---|
| `ce_xd` | `X+D` | 无 | image + demographic fusion | 人口学增量基线 |
| `mixstyle_xd` | `X+D` | MixStyle | image + demographic fusion | 同模态 MixStyle 对照 |
| `apic_v3_xd` | `X+D` | APIC v3 | image + demographic fusion | 检查人口学与 APIC 的交互 |

`X+D` 轴是独立的次要分析，不能用来挽救失败的 `X` 主分析。论文中任何 APIC 模块收益必须在同一模态内比较：`apic_v3_x - ce_x/mixstyle_x` 或 `apic_v3_xd - ce_xd/mixstyle_xd`。

#### 当前不进入两 seed 主矩阵

- `metadata`、`metadata_xda`、bounded scan FiLM 和 APIS v2：它们显式使用 `A`，仅保留为历史或补充实验；
- CDT：本轮关闭，以隔离 APIC v3 的贡献；
- target adaptation、UDA 或 target normalization：禁止；
- APIC v3 内部 ablation：两 seed 主筛选通过后再运行。

### 1.5 APIC v3 干预定义

当前 `v3_style_memory` 从 source 图像特征构造风格表征：

```text
layer1/layer2 channel mean + std
+ low-frequency energy
+ high-frequency energy
        -> style encoder
        -> source EMA style memory
        -> alternative style target and delta
        -> anatomy/confidence/entropy gate
        -> bounded layer1/layer2 low-rank residual
```

干预只在训练期启用；validation、source test 和 external target 均走 clean path。由此能检验的是“训练期风格稳健化”，不是测试时适配。

### 1.6 实验范围

```text
tasks      = CN_vs_AD, MCI_vs_AD
directions = ADNI_to_NACC, NACC_to_ADNI
seeds      = 42, 43

primary variants   = ce_x, mixstyle_x, apic_v3_x
secondary variants = ce_xd, mixstyle_xd, apic_v3_xd
```

标签映射冻结为：

| Task | Negative | Positive |
|---|---|---|
| CN vs AD | CN (`1 -> 0`) | AD (`3 -> 1`) |
| MCI vs AD | MCI (`2 -> 0`) | AD (`3 -> 1`) |

主分析共 `2 tasks × 2 directions × 2 seeds × 3 variants = 24 runs`。只有主分析通过 Gate S1 后，才启动 `X+D` 的另外 24 runs。

### 1.7 数据划分与纳排

- source 内执行 subject-level `6/2/2` train/validation/test hold-out；不做交叉验证；
- `split_seed=42` 固定 subject partition；训练 seeds 42/43 只改变初始化、采样和 APIC memory 轨迹；
- 同一 task×direction 的所有变体复用完全相同的 frozen split manifest；
- CN vs AD 与 MCI vs AD 因纳入标签不同，分别生成 manifest，不得交叉复用；
- 所有归一化、人口学编码和 checkpoint 选择只使用 source-train/source-validation；
- external target 不用于超参数、阈值、epoch 或 checkpoint 选择；
- ADNI 中冻结的 73 名同人 1.5T/3T subjects 不进入 train/validation/source-test 或普通 external-target 指标，单独保留给 Phase P；
- Phase P 主配对分析预先定义最近日期配对，并单列 `<=30 days` 与 `<=7 days` 敏感性集合；
- 在生成 manifest 后记录 subject 数、类别数、scan 数、SHA-256 和配对排除列表 SHA-256。

若排除 73 人后任一 source split 出现类别不足或明显不可训练，必须停止并书面报告；不得在查看 target 结果后缩小配对排除集。

### 1.8 精简 Phase D-image

在远程训练前只执行与 image-only 结论直接相关的审计：

1. subject、标签、重复扫描和路径匹配审计；
2. 两队列图像 shape、orientation、voxel spacing、强度归一化和预处理版本审计；
3. 按 task/split 报告类别、年龄、性别、教育及缺失率；
4. 使用冻结的 image embedding 运行 ADNI/NACC domain probe，仅用于测量队列可分性；
5. scan 参数只生成评价分层，不作为模型输入或训练 Gate。

Phase D-image 未通过时不得启动完整 24-run 主矩阵。

### 1.9 主要终点与输出

主要终点：

```text
external target + subject_mean + balanced_accuracy
```

每个 task×direction 分别报告，不先合并方向。辅助指标包括 AUC、macro-F1、accuracy、sensitivity、specificity、Brier、ECE 和逐类 recall。

必须输出：

- source validation、source test、`target_full` 的 subject-level 指标；
- `target_common_support_3T` 与 `target_unsupported_protocol`，仅作解释性分层；
- 每个 subject 的预测概率、标签、scan 数和聚合结果；
- 每个 epoch 的 source-validation 选择分数与最终 best epoch；
- APIC v3 的 memory occupancy/counts、`valid_intervention_frac`、style confidence、style entropy、style delta、gate 和 realized intervention strength；
- split/config/data/Git/environment fingerprint；
- 失败、重启、OOM、NaN、单类预测和缺失产物清单。

### 1.10 两 seed Gate S1

Gate 在运行前冻结，`X` 主分析独立判定：

1. 四个 task×direction 单元中，`apic_v3_x` 相对 `ce_x` 和 `mixstyle_x` 的两 seed 平均 `delta BA` 均为正的单元不少于 3 个；
2. 8 个 seed×task×direction 单元中，APIC v3 同时胜过两个基线的单元不少于 5 个；
3. 最差 task×direction 的两 seed 平均退化不超过 `0.02 BA`；
4. sensitivity、specificity 和任一 class recall 均不得低于 `0.15`；
5. 不得出现系统性单类预测、NaN、memory 未建立或 intervention 长期无效；
6. MCI vs AD 必须独立满足可解释性要求，不能被 CN vs AD 的改善替代。

两 seed 不运行显著性检验，不使用其置信区间宣称优效。Gate S1 未通过时停止 APIC v3 性能扩展，只保留失败分析；不得通过修改 Gate、筛选 seed 或只报告单一方向继续。

`X+D` 次要轴通过条件同样按同模态比较，但其结果不改变 `X` 主分析的 Go/No-Go。

## 2. 可复现记录

### 2.1 实现门槛

`ffc57b3` 之后已补齐本计划的实验基础设施。当前状态为：

- [x] DualShift 显式 `use_demographics` 开关，`X` 路径绕过 demographic encoder/fusion；
- [x] 六个冻结变体名称及逐变体模态解析；
- [x] 独立 APIC v3 screening launcher、CN vs AD/MCI vs AD YAML 和输出树；
- [x] APIC memory/gate 审计量写入 epoch history 和最终 JSON；
- [x] checkpoint reload、无 acquisition metadata、X/XD 隔离和合成 DataLoader smoke；
- [x] evaluation 使用 clean path，target 不更新 style memory；
- [ ] 远程机器数据路径、manifest SHA-256 和真实数据 one-epoch smoke；
- [ ] 远程工作区 clean commit 与环境指纹归档。

最后两项未完成前不得启动正式 24-run 主矩阵。

### 2.2 冻结配置

- 配置文件：`journal_dual_shift_apic_v3_screen_cn_ad.yaml` 与 `journal_dual_shift_apic_v3_screen_mci_ad.yaml`
- 数据与划分版本：待生成 task-specific frozen manifests 与 SHA-256
- 随机种子：`split_seed=42`；`training_seeds=[42,43]`
- 训练预算：所有同模态方法固定相同 epochs、optimizer、learning rate、weight decay、batch size 和 sampler
- 模型选择：仅使用 subject-aggregated source-validation composite score 和统一 collapse guard
- 环境与硬件：远程 3090/5090；每张 GPU 同时最多一个训练 worker
- 工作区状态：必须 clean；记录完整 Git commit、CUDA、cuDNN、PyTorch、GPU 和驱动版本
- 产物位置：`outputs/journal/apic_v3_screening/<task>/<seed>/<direction>/<variant>/`

### 2.3 启动顺序

1. 单元测试与配置校验；
2. `CN_vs_AD × ADNI_to_NACC × seed42` 的六变体 one-epoch smoke；
3. 双方向 `apic_v3_x` checkpoint reload 与 clean inference 检查；
4. 执行 24-run `X` 主矩阵；
5. 生成冻结的 Gate S1 报告；
6. 仅在 Gate S1 通过后执行 24-run `X+D` 矩阵；
7. 不在本轮启动 seeds 44--46。

启动命令与回传要求见 `review/operations/21_apic_v3_remote_screening_handoff_2026-08-03.md`；不得误调用旧 r2 launcher。

## 3. 分析与结果

### 3.1 结果

待运行。正式结果必须由结构化 predictions/metrics 文件生成，不在本计划中手工填写或覆盖预注册条件。

| 方法/条件 | 主要指标 | 辅助指标 | 判定 | 备注 |
|---|---:|---:|---|---|
| `ce_x` | 待运行 | 待运行 | 待判定 | X 基线 |
| `mixstyle_x` | 待运行 | 待运行 | 待判定 | X 主对照 |
| `apic_v3_x` | 待运行 | 待运行 | 待判定 | X 提出方法 |
| `ce_xd` | 待运行 | 待运行 | 待判定 | 条件性扩展 |
| `mixstyle_xd` | 待运行 | 待运行 | 待判定 | 条件性扩展 |
| `apic_v3_xd` | 待运行 | 待运行 | 待判定 | 条件性扩展 |

### 3.2 分析

- 相对基线：待运行后报告逐 seed、逐 task、逐 direction 的绝对 `delta BA` 和胜负计数；
- 异常与局限：两 seed 不能估计稳定显著性；ADNI/NACC 的 cohort 差异不能解释为单一扫描协议效应；
- 结果结论：仅允许输出 Gate S1 Go/No-Go，不允许输出最终优效结论。

## 4. 建议下一步实验指导

- 建议动作：完成实现门槛和真实数据 smoke 后，先运行严格 `X` 的 24-run 两 seed 主矩阵；
- 建议依据：该矩阵以最少计算量直接回答 APIC v3 是否在公平 image-only 条件下优于 CE 和 MixStyle；
- 固定条件：模态、split、标签映射、训练预算、基线、主终点和 Gate S1 全部保持不变；
- 进入条件：Gate S1 通过后，先运行 Phase P 配对/协议簇机制实验，再冻结唯一候选进入 seeds 42--46 确认性实验；
- 禁止事项：不看 target 调参；不以 `X+D` 对比 `X` 归因 APIC；不把两 seed 写成显著优效；不把 scan 分层写成因果机制；不覆盖 APIS v2 r2 产物。
