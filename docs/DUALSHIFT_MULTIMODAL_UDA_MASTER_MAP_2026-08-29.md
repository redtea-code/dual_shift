# DualShift 多模态 UDA 研究主地图

更新时间：2026-08-29  
状态：后续论文与实验的方向约束；不是已验证的方法结论  
主任务：scan-filtered MRI，MCI vs AD  
主方向：`ADNI_to_NACC`

当前实验登记：**DS-042**，详见 [`DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md`](DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md)。

CMRP-UDA v0 的模型、label-blind target loader、subject-disjoint adaptation split 和 journal runner 已实现；当前缺口是服务器端正式数据运行与多 seed 证据，而不是另起新的批次效应实验线。

## 1. 研究主线

我们的课题是：

> 在 target 主任务标签不可见、但 target MRI 与部署时可获得的 table 可见的条件下，研究 MRI+table 的多模态域适应是否能够提高目标域疾病判别，并保持跨模态的疾病相关关系。

批次效应、scanner、site、field strength 和 cohort shift 只是解释模型为什么有效或失效的候选机制。它们不是研究问题本身，也不能作为单独的成功标准。

```text
MRI + table 任务
        -> 多模态 source-only 基线
        -> 匹配的 MRI 域适应基线
        -> 多模态 UDA
        -> 跨模态关系保持
        -> target subject-level 任务结果
        -> batch/domain 解释分析
```

## 2. 三个核心问题

### Q1：多模态是否有任务价值

在相同 MRI backbone、subject split、训练预算和 selector 下，MRI+table 是否优于 MRI-only？pooling 前的空间交互是否优于 late fusion？

### Q2：无标签 target 是否能帮助多模态泛化

在严格 target-label-blind 条件下，target 无标签 MRI+table 是否能通过普通 alignment 或 relation-preserving UDA 改善 target 任务终点？

### Q3：跨模态关系是否是关键

如果整体对齐会损伤诊断信息，显式保护 MRI 与 table 的 shared relation 是否能降低负迁移？这是主要方法假说。

### Batch 假说的层级

“模型减少了部分批次相关变化”只能是结果解释。只有当 target 任务、跨模态关系和域诊断共同支持时，才可说结果与该假说一致；不能称为已经去除了 batch effect，更不能推导 scanner/field-strength 因果效应。

## 3. 默认 UDA 协议

第一版只使用部署时可获得且不包含诊断标签的 table：

```text
t = [age, sex, education]
```

| 集合 | 可访问信息 | 用途 |
|---|---|---|
| `S_train` | MRI、table、source `y` | 任务训练和 source relation constraint |
| `S_val` | MRI、table、source `y` | checkpoint、权重和停止点选择 |
| `S_test` | MRI、table、source `y` | checkpoint 固定后的 source 报告 |
| `T_adapt` | 无标签 MRI、table | UDA 更新与无标签关系约束 |
| `T_test` | MRI、table，最终读取 `y` | 冻结协议后的 target 评估 |

协议硬约束：

1. 所有集合按 subject 划分；纵向 scan 不能跨集合。
2. `T_adapt` 与 `T_test` 必须 subject-disjoint。
3. target label、prediction、metric 和模型排名不进入 adaptation/selector。
4. `ADNI_to_NACC` 是主方向；`NACC_to_ADNI` 是 unsupported-protocol stress test，单独报告。
5. `field_strength`、site、manufacturer 和 sequence 用于支持集/解释分析，不作为默认模型输入。
6. DG、strict UDA、target-metadata-supervised adaptation 必须分开命名和比较。

## 4. 模型主线：CMRP-UDA

工作标签：**Cross-Modal Relation-Preserving UDA（CMRP-UDA）**。

```text
x_MRI -> MRI encoder   -> h_M_shared, h_M_specific
t     -> table encoder -> h_T_shared, h_T_specific

(h_M_shared, h_T_shared, h_M_specific, h_T_specific)
                       -> relation/fusion
                       -> z_joint -> classifier -> y_hat
```

这里的 `shared/specific` 是可测试的结构分解，不等于“shared=纯 biology、specific=纯 batch”。第一版优先使用容量受控的 projection + gated bilinear/elementwise fusion；现有空间条件化模型可作为工程起点，但不能把其输出直接重新命名为生物因果表示。

### 第一版损失族

```text
L_task  = source classification CE
L_rel_s = source MRI-table paired relation preservation
L_rel_t = target unlabeled paired relation / augmentation consistency
L_align = CORAL 或 linear-kernel MMD，分别作用于 modality-specific features
L_prox  = source checkpoint proximal penalty
L_id    = bounded identity-preservation penalty
```

第一版不以 target pseudo-label、target entropy 或多头 GRL 作为默认主机制。它们可以作为后续消融，但不能让研究问题重新变成“哪种 discrepancy 最容易下降”。

## 5. 论文主表与消融表

以下不是“只做审计的测试”，而是论文实验的直接组成部分。主表优先展示任务比较，消融表解释各组件贡献。

### 5.1 主比较表

| 组别 | 实验行 | 论文作用 |
|---|---|---|
| MRI baseline | MRI-only source-only | 基础任务参考 |
| multimodal baseline | MRI+table late fusion | 证明 table 的独立任务价值 |
| spatial multimodal | MRI+table pooling 前空间交互 | 检验交互位置假说 |
| image-only UDA | FMM、amplitude/phase transport、GRL、residual、source-free residual | 与多模态方法比较的 MRI UDA 基线 |
| multimodal UDA | joint alignment、MRI-specific alignment | 普通多模态 UDA 对照 |
| proposed | CMRP-UDA | 跨模态关系保持的主方法候选 |

DS-034 至 DS-041 的已有方向都可在论文主比较表中保留相应位置。若历史数值与新 protocol 不完全一致，应统一 protocol 后重跑；不能因为进入主表就改写其历史结论。

### 5.2 主消融表

CMRP-UDA 至少包含：

1. MRI-only vs MRI+table；
2. late fusion vs pooling 前空间融合；
3. joint alignment vs MRI-specific alignment；
4. 去掉 source relation preservation；
5. 去掉 target unlabeled relation consistency；
6. 去掉 proximal/identity constraint；
7. 既有 FMM/transport/GRL/residual 方向的匹配消融。

### 5.3 必要负控

`shuffled-tabular`、`missingness-only` 和 `table-only` 直接进入主消融或补充表，用来判断真实 MRI-table 配对是否必要，以及是否存在 table shortcut。它们不是额外的研究方向。

## 6. 结果判定

只保留三条判断规则：

1. **任务优先**：target subject-level BA/AUROC 是主终点，source diagnosis、calibration 和 sensitivity/specificity 用于检查负迁移。
2. **关系保持**：真实 MRI-table 配对必须优于 shuffled-tabular，且 relation constraint 不能以表示塌缩为代价。
3. **协议有效**：target label、prediction 和 metric 在 adaptation/selector 中不可见；否则从 strict UDA 表中移出并单独标注。

domain discrepancy、domain-probe、adapter activity 和 CAPM anchor 只作为解释列。discrepancy 下降而 target 任务下降，报告为负迁移；target 任务改善但 discrepancy 不支持时，保留任务结果但不强行使用 batch 解释。

## 7. 现有实验的正确位置

| 实验 | 新主地图中的定位 |
|---|---|
| DS-034/C4 | 频率/CAPM 历史 exploratory reference，可作为 image/CAPM 消融 |
| DS-035/FMM | image-only UDA baseline |
| DS-036 | target-style CAPM UDA 消融 |
| DS-037 | amplitude/phase transport 消融 |
| DS-038 | domain/intensity GRL 消融 |
| DS-039 | residual adaptation 消融 |
| DS-040 | frequency-GRL 消融 |
| DS-041 | source-free residual UDA 参考 |

这些实验不需要被丢弃，也不需要全部升级为新方法。它们直接为论文主表提供 MRI UDA 对照；新方法贡献集中在“多模态输入 + 跨模态关系保持 + strict UDA”。

## 8. 最短实施路线

1. 固定服务器端 multimodal manifest、table contract 和 subject splits。
2. 统一实现 MRI-only、MRI+table late fusion、空间融合和既有 MRI UDA baselines。
3. 在同一 protocol 下运行主比较表和主消融表。
4. 使用已实现的 DS-042 CMRP-UDA relation preservation、modality-specific alignment 和 identity constraint 入口运行正式实验。
5. 统一报告 target task、跨模态关系、负迁移和 provenance；batch/scanner 只作为解释列。

在主表和必要消融完成前，不再为频率、GRL、GMM、residual rank 或 scanner-specific gate 单独扩展复杂矩阵。

## 9. 论文主张边界

当前工作标签：

> We study label-blind multimodal UDA for MRI and deployable tabular covariates, with cross-modal relation preservation as a protection mechanism against negative transfer.

只有在统一 protocol、多 seed 和 target-label-blind 证据完成后，才可进一步声称方法改善了特定目标域的 subject-level discrimination。

仍然禁止以下表述：

- 我们已经去除了 scanner/batch effect；
- CAPM/shared representation 是纯 biological information；
- residual/specific representation 是纯 batch information；
- domain discrepancy 下降证明 scanner 因果校正；
- 单个 seed、source gain 或 target point estimate 证明稳定有效。

## 10. 依据与边界

本地图吸收多模态 MRI+table 阅读汇总、MRI 域适应论文和现有 DualShift 交接/结果文档。它保留既有实验作为可复用的基线与消融，不改变历史 protocol、结果数值或 target access 事实。
