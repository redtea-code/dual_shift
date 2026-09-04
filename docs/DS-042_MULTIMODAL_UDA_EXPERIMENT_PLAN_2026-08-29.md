# DS-042：MRI+table 跨模态关系保持的无监督域适应

更新时间：2026-08-29
状态：**PROPOSED / CODE CHECKED / CMRP-UDA IMPLEMENTED / NO REAL-DATA RESULTS**
主任务：scan-filtered MCI vs AD
主方向：`ADNI_to_NACC`
方法工作标签：**Cross-Modal Relation-Preserving UDA（CMRP-UDA）**

## 1. 实验问题

DS-042 研究：在 target 主任务标签不可见、但 target MRI 与部署时可获得的 table 可见的条件下，MRI+table 多模态 UDA 是否能改善目标域疾病判别，并保持 MRI 与 table 之间的疾病相关关系。

批次效应、scanner、site、field strength 和 cohort shift 只是解释模型有效或失效的候选机制。它们不是本实验的主问题，也不能作为单独的成功标准。

主问题拆成三步：

1. MRI+table 是否比 MRI-only 有任务价值？
2. 无标签 target MRI+table 是否比 source-only 多模态模型更有利于 target 任务？
3. 跨模态关系保持是否比无条件的整体对齐更能减少负迁移？

## 2. 可复用灵感与 DS-042 的差异

### 2.1 MRI+table 方法提供的结构灵感

- 独立的 MRI 分支和 table 分支，而不是把 table 直接拼在最终分类器上；
- 使用 MRI-table 配对关系作为任务相关约束；
- 在空间语义聚合前允许 table 条件参与 MRI 表示交互；
- 将跨模态关系和域对齐作为两个不同功能，分别做消融。

MedPro-DG 和 LGMDG 属于 DG，不访问 target；它们的跨域语义对比和配对一致性可以作为 source relation / target unlabeled relation 的设计灵感，但不能把 DG 结果直接当作 UDA 证据。Embracing the Disharmony 支持“适配而不是全不变”的原则，但其 target age/sex 辅助标签属于 target-metadata-supervised adaptation，不能照搬到 strict UDA。

### 2.2 MRI 域适应方法提供的比较基线

FMM、amplitude/phase transport、domain/intensity GRL、residual adaptation、source-free residual alignment、CORAL/MMD 等都属于 DS-042 的 image-only 或 matched UDA 比较线。它们不被删除，也不需要重新命名为 CMRP-UDA。

DS-042 的新意不在于再增加一个 batch correction loss，而在于：**多模态输入、跨模态关系保持和无标签 target adaptation 被放在同一个严格协议中。**

## 3. 固定数据合同

第一版 table 固定为部署时可获得且不包含诊断标签的：

```text
t = [age, sex, education]
```

| 集合 | 可访问信息 | 用途 |
|---|---|---|
| `S_train` | MRI、table、source `y` | 任务训练与 source relation constraint |
| `S_val` | MRI、table、source `y` | checkpoint、loss 权重和停止点选择 |
| `S_test` | MRI、table、source `y` | checkpoint 固定后的 source 报告 |
| `T_adapt` | 无标签 MRI、table | UDA 更新和 target 无标签关系约束 |
| `T_test` | MRI、table，最终才读取 `y` | 冻结协议后的 target 评估 |

固定规则：

1. 所有划分按 subject，纵向扫描不能跨 split；
2. `T_adapt` 与 `T_test` subject-disjoint；
3. source imputation/scaling 只在 `S_train` 拟合；
4. target label、prediction、metric 和模型排名不进入 adaptation/selector；
5. `ADNI_to_NACC` 是主方向；`NACC_to_ADNI` 只作为 unsupported-protocol stress test；
6. `field_strength`、site、manufacturer 和 sequence 只用于支持集分层或解释，不进入默认模型输入；
7. DG、strict UDA 和 target-metadata-supervised adaptation 分开标记。

第一版不加入 MMSE、CDR、FAQ、直接 diagnosis 字段、APOE、PET/CSF、MRI-derived ROI 或 scan metadata。若将来加入，必须作为额外模态或独立消融注册。

## 4. CMRP-UDA v0

### 4.1 表示结构

```text
x_MRI -> E_M -> h_M_shared, h_M_specific
t     -> E_T -> h_T_shared, h_T_specific

(h_M_shared, h_T_shared, h_M_specific, h_T_specific)
                       -> relation/fusion R
                       -> z_joint -> classifier -> y_hat
```

`shared/specific` 是待验证的结构分解，不等于 shared 是纯 biological information，也不等于 specific 是纯 batch information。

第一版采用容量受控的 projection + gated bilinear/elementwise fusion。MRI 的空间 feature map 保留到 `layer4`，table 在空间语义聚合前参与一次条件化交互；不同时引入大型 CLIP、多个 cross-attention、频域 gate 和多头 GRL。

### 4.2 损失

对 source batch `(x_s, t_s, y_s)` 和 target batch `(x_t, t_t)`：

```text
L_task  = CE(classifier(z_joint_s), y_s)
L_rel_s = paired_relation(h_M_shared_s, h_T_shared_s)
L_rel_t = paired_relation(h_M_shared_t, h_T_shared_t)
          + MRI_augmentation_consistency(h_M_shared_t)
L_align = CORAL(h_M_specific_s, h_M_specific_t)
        + beta_tab * CORAL(h_T_specific_s, h_T_specific_t)
L_prox  = ||theta_adapt - theta_source||_2^2
L_id    = ||z_joint_adapt - z_joint_clean||_2^2

L_total = L_task + lambda_s L_rel_s + lambda_t L_rel_t
        + lambda_a L_align + lambda_p L_prox + lambda_i L_id
```

其中：

- `L_task` 是唯一的 source 主任务监督；
- `L_rel_s` 使用合法的 source MRI-table 配对；
- `L_rel_t` 只使用 target 无标签的同 subject 配对和 MRI augmentation；
- `L_align` 先选择一个简单的 CORAL 或 linear-kernel MMD，分别观察 MRI-specific 和 table-specific shift；
- `L_prox/L_id` 限制适配不要任意改写 source 决策。

第一版不以 target pseudo-label CE、target entropy minimization 或多头 confusion loss 作为默认主机制。

## 5. 论文主表和主消融

### 5.1 主比较表

| 组别 | 实验行 | 论文作用 |
|---|---|---|
| MRI baseline | MRI-only source-only | 基础任务参考 |
| multimodal baseline | MRI+table late fusion | 检验 table 的独立贡献 |
| spatial multimodal | pooling 前空间交互 | 检验交互位置 |
| image-only UDA | FMM、amplitude/phase transport、GRL、residual、source-free residual | MRI UDA 匹配基线 |
| multimodal UDA | joint alignment、MRI-specific alignment | 普通多模态 UDA 对照 |
| proposed | CMRP-UDA | 主方法候选 |

DS-034 至 DS-041 的既有方向可直接占据主比较表或主消融表中的相应位置。历史数值若与 DS-042 的 protocol、split、selector 或 target access 不完全一致，应统一 protocol 后重跑；表格位置变化不能改变原始 claim boundary。

### 5.2 CMRP-UDA 必要消融

1. MRI-only vs MRI+table；
2. late fusion vs pooling 前空间融合；
3. joint alignment vs MRI-specific alignment；
4. 去掉 source relation preservation；
5. 去掉 target unlabeled relation consistency；
6. 去掉 proximal/identity constraint；
7. shuffled-tabular、missingness-only 和 table-only shortcut control；
8. 与 FMM、transport、GRL、residual 和 source-free residual 的匹配比较。

这些实验直接属于论文主表或消融表，不需要先被拆成一套独立的 batch-effect 审计实验。

## 6. 结果终点

### 6.1 主任务

每个方向、task 和 seed 分开报告：subject-level BA、AUROC、Macro-F1、MCC、sensitivity、specificity、Brier/ECE，以及 paired seed difference 和 bootstrap CI。

### 6.2 多模态关系

- real MRI-table pair 与 shuffled-tabular 的差；
- MRI/table 缺失时的退化；
- source/target cross-modal relation consistency；
- joint 与 modality-specific representation drift。

### 6.3 解释列

- MRI-specific、table-specific 和 joint discrepancy；
- domain-probe accuracy/AUROC；
- adapter activity、identity 比例和 source-anchor preservation。

这些解释列不能单独替代 target 任务终点。若 discrepancy 下降但 target 任务下降，报告为负迁移；若 target 任务改善但 discrepancy 不支持，只保留任务结论，不强行使用 batch 解释。

## 7. 当前代码实现检查

### 7.1 已实现、可直接复用

| 代码位置 | 当前能力 | DS-042 用途 |
|---|---|---|
| `Model/ablation/scale_table_transformer.py` | MRI+age/sex/education 的 CAPM、`transformer_self`、`transformer_cross` 和 pooling 前条件化 | B1/B2 多模态 source-only baseline |
| `Model/backbone/journal_resnet.py` | table-conditioned spatial modulation 的基础 ResNet 路径 | MRI+table 工程参考 |
| `Model/ablation/frequency_uda.py` | label-free target frequency prior 和 bounded frequency gate | image-only UDA 消融 |
| `Model/ablation/residual_adaptation.py` | CAPM residual statistics、bounded residual correction 和 source-free pilot 接口 | residual/source-free UDA 消融 |
| `experiments/train_journal.py` | 统一 source split、subject-level aggregation、checkpoint selector 和既有 variant registry | 主表运行骨架 |
| `tests/test_scale_table_transformer_ablation.py`、`tests/test_frequency_uda.py`、`tests/test_capm_residual_adaptation.py` | 既有模型、频域和 residual 接口测试 | 回归保障 |

### 7.2 已实现的 DS-042 部分

DS-042 的最小可运行 CMRP-UDA v0 已接入现有 journal runner：

1. `Model/ablation/cmrp_uda.py` 提供独立 MRI/table encoder、shared/specific projection、显式 relation/fusion 表示、bounded residual adapter、CORAL 和 paired-relation loss；
2. `data/journal_dataset.py` 的 `UnlabeledJournalSubset` 只返回 target MRI、table、missingness 和 subject metadata，不 materialize target label；
3. `training/cmrp_uda_loop.py` 实现 source CE、source/target relation、modality-specific/joint CORAL、identity/proximal 约束和 target augmentation consistency；
4. `experiments/cmrp_uda.py` 实现 subject-disjoint `T_adapt/T_test` split 和 label-blind split manifest；
5. `experiments/train_journal.py` 注册 CMRP variants，统一 checkpoint selector、预测输出、subject-level metrics 和 bootstrap CI；
6. `tests/test_cmrp_uda.py` 覆盖表示、loss、split、label-blind loader 和训练接口。

代码检查结论为：CMRP-UDA 主方法和必要消融已经具备运行入口；尚未有服务器端 ADNI/NACC 正式训练结果。现有 `transformer_cross` 仍只作为 table-conditioned CAPM baseline，不能改称 CMRP-UDA。

## 8. DS-042 代码实施顺序

1. 服务器端准备 multimodal manifest、table contract 和固定 subject split；
2. 用 `journal_ds042_multimodal_uda_scan_filtered_1p5t_mci_ad.yaml` 运行 CMRP variants 的短 run；
3. 在同一 protocol 下将 FMM、transport、GRL、residual、source-free residual 和 CMRP-UDA 汇总到论文主表；
4. 按正式 seeds 运行，并检查 target-label-blind manifest、checkpoint 和 predictions provenance。

## 9. 代码检查记录

本次检查覆盖：

- 全部现有及 DS-042 测试：`62 passed`；
- `Model`、`experiments`、`data`、`training`、`utils` 和 `tests` 的 Python bytecode compile：通过；
- DS-042 专属 model/loader/split/training tests：通过；尚无真实数据结果。

上述通过项只说明既有接口和回归测试正常，不等于 DS-042 已完成，也不等于 ADNI/NACC UDA 证据。

## 10. 主张边界

DS-042 完成后，最强的合理表述是：

> Under a specific scan-filtered MRI+tabular UDA protocol, cross-modal relation-preserving adaptation improves subject-level target discrimination relative to matched source-only and image-only UDA baselines.

在结果完成前，不得声称：

- 已经去除了 scanner/batch effect；
- shared 表示是纯 biological information；
- specific 表示是纯 batch information；
- discrepancy 下降证明 scanner 因果校正；
- 单个 seed、source-side gain 或 target point estimate 证明稳定有效。

本计划把既有 MRI 域适应方法保留为论文主表/消融，并把已实现的 CMRP-UDA v0 作为新增主方法候选；不修改 DS-034--041 的历史结果和协议。
