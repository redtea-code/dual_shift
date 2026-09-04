# DS-042 多模态 UDA Phase 1 执行方案（工作草案）

更新时间：2026-08-29
状态：PROPOSAL；用于推进实验设计，不代表已验证结果
主方向：`ADNI_to_NACC`，scan-filtered MCI vs AD
工作标签：**cross-modal relation-preserving multimodal UDA**

正式实验计划：[`DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md`](DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md)。本文件保留为较详细的设计草案；实验登记、代码状态和主表口径以 DS-042 计划为准。

## 0. 决策先行

本项目下一阶段的核心不是“去除批次效应”，而是：

> 在 target 主任务标签不可见、但 target MRI 与部署时可获得的 table 可见的条件下，研究 MRI+table 的多模态域适应是否能够提高目标域疾病判别，并保持 MRI 与 table 之间的疾病相关关系。

批次效应、scanner、site、field strength 和 cohort shift 只作为**解释模型有效或失效的候选机制**。它们必须在任务终点成立之后接受审计，不能成为先验事实，也不能成为单独优化的目标。

因此，后续工作分成三层：

```text
第一层：多模态任务是否真的有价值？
第二层：无标签 target 信息能否改善多模态 target 任务？
第三层：如果改善，是否与“部分域相关变化被减少且跨模态关系保留”一致？
```

没有通过上一层，不进入下一层。

## 1. 从相关论文借什么，不能照搬什么

### 1.1 MedPro-DG：借“任务条件的跨域关系”，不借 target-label access

MedPro-DG 使用 source 标签构造同类、跨机构的语义对比。对本课题最有用的不是 prompt 形式，而是：**关系约束应服务于临床任务，而不是把全部表示强行做域不变。**

迁移到当前 UDA 的方式：

- source 侧可以使用 `y` 约束 MRI/table 的疾病相关表示；
- target 侧不能用 `y` 构造正负对；
- target 侧只使用同一 subject 的 MRI-table 配对、增强一致性和无标签统计；
- 不把 target 预测伪标签作为第一版主机制，以免把伪标签噪声误写成跨模态关系。

### 1.2 LGMDG：借“独立 table 分支 + 高阶融合 + 配对一致性”

LGMDG 的结构启发是：临床变量先进入独立语义分支，再与图像表示交互；域对抗和跨模态配对约束是两个不同功能。当前 MRI 版本不必使用 CLIP 或自然语言 prompt，但应保留三个接口：

1. MRI encoder：保留空间 feature map，并输出 MRI embedding；
2. table encoder：逐变量处理数值、类别和缺失标志；
3. relation/fusion module：显式生成 joint representation，而不是只在最后 concat。

不能照搬的部分：LGMDG 是 leave-one-domain-out DG，训练时不访问 target；其结果不能被当作 UDA 证据。

### 1.3 Embracing the Disharmony：借“适配而非全不变”，改成 strict UDA

该工作最重要的原则是：不要为了域不变而删除所有可能有用的变化；proximal/anchor 约束可以限制小样本 target adaptation 的破坏性。

当前只借以下思想：

- 保留 source checkpoint 作为 clean anchor；
- UDA 更新使用 bounded adapter 或 proximal penalty；
- source diagnosis preservation 与 target adaptation 分开审计。

不能照搬 target age/sex 辅助标签微调。若 target age/sex 被当作输入，是 target-unlabeled covariate access；若用它们的标签训练辅助任务，则必须单独注册为 target-metadata-supervised adaptation。

### 1.4 MRI 域适应方法：作为匹配基线，不作为主张来源

DANN/ADDA、CORAL/MMD、MixStyle、consistency adaptation 和 source-free feature statistics 都可提供 baseline 或组件候选。第一版只选择一个简单、可解释的无标签 alignment（优先 CORAL 或 linear-kernel MMD），用于回答“普通 UDA 是否有效”。

频域 enhancement、GRL、GMM transport 和 residual adaptation 不应被删除。它们应作为**匹配的 MRI 域适应消融**保留，并直接预留在论文主比较表或主消融表中；只有在协议、split、selector 和 target access 不匹配时，才单独标注为历史 exploratory 结果，不能与新模型数值混排。

## 2. 数据合同：第一版只用低风险 table

### 2.1 默认 table contract

第一版固定为：

```text
t = [age, sex, education]
```

原因：现有 CAPM/数据加载器已经有这三个变量的 source-fitted preprocessing；它们在部署时通常可获得，且不会直接读取 MCI/AD 标签。服务器端 manifest 只需将原始字段映射到统一语义，不绑定某一台机器或某个文件名。

统一语义字段为：

```text
AGE_YEARS
PTDEMOG__PTGENDER
PTDEMOG__PTEDUCAT
```

正式实验前需要在服务器端生成 frozen multimodal manifest，确认 subject/date 对齐、缺失标志和 subject split；这些是运行条件，不是额外的方法实验。

### 2.2 暂不进入默认输入的变量

以下变量可能有研究价值，但第一版不放进主模型：

- `MMSE`、`CDRSB`、`FAQ` 等可能接近诊断标签的临床量表；
- `DXSUM__DIAGNOSIS`、`DXSUM__DXMCI`、`DXSUM__DXAD` 等直接诊断字段；
- `APOE`、PET/CSF、生物标志物和 MRI-derived ROI；
- `SITE`、厂家、field strength、sequence、cohort ID 等 acquisition/domain 字段。

它们要么可能构成标签代理/泄漏，要么属于额外模态/域变量，应该单独注册，不能悄悄混进 MRI+table 主实验。

### 2.3 访问协议

| 集合 | 可访问信息 | 用途 |
|---|---|---|
| `S_train` | MRI、`t`、source `y` | 训练 MRI encoder、table encoder、fusion 和 source relation loss |
| `S_val` | MRI、`t`、source `y` | checkpoint、UDA 权重和停止点选择 |
| `S_test` | MRI、`t`、source `y` | checkpoint 固定后的 source 诊断 |
| `T_adapt` | 无标签 MRI、`t` | target 无标签统计、配对关系一致性和 UDA 更新 |
| `T_val`（可选） | 无标签 MRI、`t` | label-blind 稳定性/停止判据 |
| `T_test` | MRI、`t`，最终才读取 `y` | 最终 exploratory target 评估 |

硬约束：所有划分按 subject；`T_adapt` 与 `T_test` subject-disjoint；target `y`、prediction、metric、confusion matrix 和模型排名不能进入 adaptation/selector。

## 3. 最小候选模型：CMRP-UDA v0

工作标签为 **Cross-Modal Relation-Preserving UDA**（CMRP-UDA）。这是一个便于审计的第一版，不预先宣称是最终方法名。

### 3.1 表示结构

```text
x_MRI  -> E_M -> h_M_shared, h_M_specific
t      -> E_T -> h_T_shared, h_T_specific

(h_M_shared, h_T_shared, h_M_specific, h_T_specific)
                    -> relation/fusion R
                    -> z_joint -> classifier -> y_hat
```

实现上先采用容量受控的 MLP projection 和 gated bilinear/elementwise fusion；不要第一版同时引入大型 cross-attention、CLIP、频域 gate 和多层 GRL。现有 `ScaleTableInteractionAblation3D` 的 `layer4_pixel + transformer_cross` 可作为空间融合的工程起点，但必须增加可返回 `h_M`、`h_T`、`z_joint` 的明确接口，不能从 attention map 反推表示。

### 3.2 为什么要 shared/specific

这不是把 shared 命名为“纯 biology”、specific 命名为“纯 batch”。它只是一个可测试的结构分解：

- `shared`：MRI 与 table 都能支持的联合信息；
- `specific`：某一模态独有、可能有用也可能受域影响的信息；
- `z_joint`：真正交给疾病分类器的多模态表示。

批次解释只能在结果审计后讨论：如果 specific alignment 伴随 target 任务改善、关系保持和 domain discrepancy 的一致变化，才可说与该假说相容。

### 3.3 第一版损失

令 source batch 为 `(x_s, t_s, y_s)`，target adaptation batch 为 `(x_t, t_t)`，则第一版只使用以下损失族：

```text
L_task = CE(classifier(z_joint_s), y_s)

L_rel_s = 1 - cosine(P_M(h_M_shared_s), P_T(h_T_shared_s))

L_rel_t = 1 - cosine(P_M(h_M_shared_t), P_T(h_T_shared_t))
         + consistency(h_M_shared_t(x), h_M_shared_t(aug(x)))

L_align = CORAL(h_M_specific_s, h_M_specific_t)
        + beta_tab * CORAL(h_T_specific_s, h_T_specific_t)

L_prox = ||theta_adapt - theta_source||_2^2
L_id   = ||z_joint_adapt - z_joint_clean||_2^2 / (||z_joint_clean||_2^2 + eps)

L_total = L_task + lambda_s L_rel_s
        + lambda_t L_rel_t + lambda_a L_align
        + lambda_p L_prox + lambda_i L_id
```

解释：

- `L_task` 是唯一的 source 主任务监督；
- `L_rel_s` 保持 source 已知的 MRI-table 配对关系；
- `L_rel_t` 只使用 target 无标签的同 subject 配对和 MRI augmentation；
- `L_align` 先作为普通 UDA 锚点，分别观察 MRI-specific 与 table-specific shift；
- `L_prox/L_id` 防止 adaptation 把 source 任务表示任意改写。

第一轮不加入 target pseudo-label CE，不加入 target entropy minimization，不加入 source/target confusion loss 的多头组合。若后续需要加入，必须作为独立 ablation，并证明不会诱发预测塌缩。

### 3.4 bounded adapter 与 identity fallback

CMRP-UDA v0 使用联合 source/target 优化器，并通过 bounded residual adapter、`L_prox` 和 `L_id` 限制适配幅度；当前实现不预设 backbone 冻结或 target support fallback，后续可把 staged freeze 作为独立消融注册。适配强度由 source validation 或 label-blind `T_val` 选择，并固定上限。

这一步的意义不是证明 batch correction，而是提供一个可测量的“适配是否真的发生、是否过度改变”的工程边界。

## 4. 论文主表与消融矩阵

下面的行不是“只做审计的测试”，而是论文主表和主消融表的候选实验。所有行共享相同 subject split、预处理、optimizer、训练预算、checkpoint selector 和 target-test rows。先用一个短 run 检查实现，再按正式 seed 运行；短 run 不写入论文结果。

| ID | 设置 | 回答的问题 |
|---|---|---|
| `B0` | MRI-only source-only | 图像基线 |
| `B1` | MRI + table，late fusion，source-only | table 是否有独立任务贡献 |
| `B2` | MRI + table，layer4/空间融合，source-only | pooling 前交互是否值得研究 |
| `N1` | `B1` + shuffled-tabular | 真实 subject 配对是否必要（主消融/补充表） |
| `N2` | `B1` + missingness-only | 缺失模式是否成为 shortcut（主消融/补充表） |
| `N3` | table-only | table 是否单独携带过强诊断/队列 shortcut（主消融/补充表） |
| `U0` | `B1` 的 source-only matching control | UDA 的严格锚点 |
| `U1` | `U0` + joint CORAL/MMD | 普通多模态 UDA 是否有收益 |
| `U2` | `U0` + MRI-specific alignment | MRI shift 是否主要限制迁移 |
| `U3` | `U0` + CMRP-UDA v0 | relation preservation 是否降低负迁移（主模型候选） |

### 4.1 既有 MRI 域适应结果在论文中的位置

以下既有方向应作为 image-only 或 CAPM-based UDA 的匹配消融，直接纳入论文实验规划，而不是只放在“审计记录”：

| 既有方向 | 论文表格位置 | 作用 |
|---|---|---|
| DS-035 FMM | 主比较表：image-only UDA baseline | 与多模态 UDA 比较图像域适应基线 |
| DS-036 target-style CAPM | 主比较表或 UDA 消融 | 检查 target-style 条件 transport 的作用 |
| DS-037 amplitude/phase transport | 主消融表 | 检查频域 transport 的强度/phase 选择 |
| DS-038 domain/intensity GRL | 主消融表 | 检查普通 GRL 与多模态关系保持的差异 |
| DS-039 residual adaptation | 主消融表 | 检查受限 residual 更新是否减少负迁移 |
| DS-040 frequency GRL | 主消融表 | 检查频率引导与 GRL placement |
| DS-041 source-free residual alignment | 主比较表或补充表 | 作为 source-free UDA 参考 |

这些行应在同一篇论文的统一 protocol 下重跑或严格对齐后再比较。已有历史数字可以保留为背景结果，但不能因表格位置改变而被重新解释为 CMRP-UDA 证据。

### 4.2 CMRP-UDA 的必要消融

CMRP-UDA 至少保留以下主消融：

1. 去掉 table 分支：MRI-only；
2. late fusion 对比 pooling 前空间融合；
3. 去掉 source relation preservation；
4. 去掉 target unlabeled relation consistency；
5. joint alignment 对比 MRI-specific alignment；
6. 去掉 proximal/identity constraint；
7. shuffled-tabular、missingness-only 和 table-only shortcut controls。

因此，UDA 主线不是“只训练一个新模型”，而是由 MRI-only、MRI+table、既有 MRI UDA、普通 multimodal UDA、CMRP-UDA 及其必要消融组成一张完整的主表。

## 5. 最小判定规则

实验表可以一次性包含上述 baseline 和消融，不需要为每一个组件建立独立的审计阶段。只保留三条决定规则：

1. **主任务优先**：以 target subject-level BA/AUROC 为主终点，source diagnosis、calibration 和 sensitivity/specificity 作为负迁移检查。
2. **关系保持**：CMRP-UDA 只有在真实 MRI-table 配对不劣于 shuffled-tabular，且 relation preservation 没有造成表示塌缩时，才可解释为跨模态关系候选。
3. **协议有效**：target label、prediction 和 metric 不进入 adaptation/selector；若违反，整行从 strict UDA 主表移出并单独标注协议。

domain discrepancy、domain-probe、adapter RMS 和 CAPM anchor 只作为结果解释列。它们下降而 target 任务下降时，报告为负迁移；它们不变而 target 任务改善时，保留任务结果但不强行使用 batch 解释。

## 6. 结果报告口径

### 6.1 主任务终点

每个方向、task、seed 单独报告：

- subject-level balanced accuracy；
- AUROC、Macro-F1、MCC；
- sensitivity、specificity 和 confusion matrix；
- Brier/ECE 等 calibration；
- paired seed difference、bootstrap CI 和聚合单位。

### 6.2 多模态终点

- `B1/B2` 相对 `B0` 的增益；
- real-pair 与 shuffled-tabular 的差；
- MRI missing/table missing/both missing 的退化；
- source/target cross-modal relation consistency；
- modality-specific 与 joint representation 的 drift。

### 6.3 机制解释终点

- MRI-specific、table-specific 和 joint discrepancy；
- domain-probe accuracy/AUROC（仅作诊断）；
- adapter RMS、有效强度、identity/fallback 比例；
- CAPM/source-anchor prediction preservation；
- protocol support、paired 1.5T/3T consistency 和 ROI bias（如有合法配对数据）。

机制终点不能单独替代 target task endpoint。

## 7. 实施顺序与仓库落点

现有 `ScaleTableInteractionAblation3D` 提供 MRI+age/sex/education baseline；DS-042 的 CMRP-UDA runner、label-blind target loader 和 subject-disjoint split 已补全。下一步是在服务器端数据合同下统一运行主表和消融：

### Work package 1：manifest/schema（代码接口已完成，待服务器数据落地）

新增独立的 multimodal manifest schema，至少包含：

```text
subject_id, scan_id, image_path, date, label_or_hidden_label,
age, sex, education, age_missing, sex_missing, education_missing,
cohort, protocol_version, split, split_digest
```

target adaptation loader 必须只返回 `image`、合法 `table`、`subject_id` 和缺失标志；不要复用会 materialize target label 的普通 dataset 路径。

### Work package 2：representation API（已完成）

在新模块中提供显式接口：

```text
encode_mri(image) -> h_mri / spatial_map
encode_table(table, missingness) -> h_table
fuse(h_mri, h_table) -> z_joint
forward_with_repr(...) -> logits, representation_audit
```

旧模型可以作为 baseline factory，但不要把现有 CAPM output 直接重命名为 `z_bio`。

### Work package 3：UDA loss/adapter（已完成）

先实现 CORAL 或 linear MMD、source relation loss、target pair consistency、proximal/identity regularization 和 label-blind access assertions。每个 loss 都要有 shape、zero-weight、identity 和 no-target-label 单元测试。

### Work package 4：journal runner（已完成）

新增独立 runner 或在 `train_journal.py` 外建立清晰入口，支持：

- `B0-B2/N1-N3/U0-U3` 变体；
- source-only 与 target-adapt 两种 loader；
- source-val selector；
- frozen `T_test` final evaluation；
- resolved config、manifest digest、checkpoint hash、target access audit 和 predictions artifact。

### Work package 5：统一跑主表与消融

服务器端 manifest 和 loader 就绪后，统一运行主表、既有 MRI UDA 消融和 CMRP-UDA 消融。短 run 只用于发现实现错误；正式结果直接进入同一论文结果目录，并记录 protocol、seed 和 provenance。

## 8. 当前不应做的事情

- 不把 DS-034/C4、DS-035 FMM 或 DS-041 residual pilot 直接写成 CMRP-UDA 先验结果；
- 不在没有服务器端 MRI-table manifest 的情况下开始新的正式训练；
- 不因为 batch/discrepancy 指标下降就把某个消融升级为主方法；
- 不把 age/sex/education 当作 domain-invariant biology；
- 不把 target metadata-supervised adaptation 结果与 strict UDA 合并；
- 不把 `NACC_to_ADNI` 与主方向平均；
- 不用 target test 选择 table 变量、loss 权重、停止点、adapter strength 或论文主结果。

## 9. 论文方向的暂定表述

在证据完成前，只使用：

> We study label-blind multimodal UDA for MRI and deployable tabular covariates, with cross-modal relation preservation as a protection mechanism against negative transfer.

若 Gate C 和稳健性审计均通过，可以进一步表述为：

> Under a specific scan-filtered MRI+tabular UDA protocol, relation-preserving selective adaptation improves subject-level target discrimination while reducing negative transfer.

仍然不能写成：

- 我们去除了 scanner/batch effect；
- CAPM/shared representation 是纯 biological information；
- residual/specific representation 是纯 batch information；
- domain discrepancy 下降证明了 scanner 因果校正；
- 一个 seed 或一个 target point estimate 证明方法稳定有效。

## 10. 立即执行的三件事

1. **冻结服务器端 table contract 和 manifest schema**：先只用 age/sex/education，明确 target 是否可见及缺失规则。
2. **把已有 MRI UDA 方向整理成统一主表消融**：FMM、transport、GRL、residual 和 source-free 结果不再孤立审计。
3. **运行 B0-B2/N1-N3/U0-U3 与 CMRP-UDA 消融**：代码已具备统一入口，服务器端正式 run 用同一 protocol 回答多模态输入、普通 UDA 和关系保持的相对贡献。

Phase 1 的目标不是一次找到最终模型，而是把研究问题从“哪个批次模块有效”重新拉回“无标签 target 下的多模态疾病判别和跨模态关系泛化”。

## 11. 依据与边界

本方案吸收：

- 多模态 MRI+table 域适应阅读汇总及其三篇论文笔记；
- `docs/DUALSHIFT_MULTIMODAL_UDA_MASTER_MAP_2026-08-29.md`；
- `docs/DUALSHIFT_NEW_MEMBER_HANDOVER_2026-08-17.md`；
- `docs/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`；
- `docs/PROPOSAL_CAPM_CONDITIONED_BIOLOGY_PRESERVING_BATCH_DECOUPLING_2026-08-25.md`。

本方案是下一阶段的执行合同，不把 proposal 变成结果，不改变 DS-034--041 的历史 protocol，也不授权在现有 dirty worktree 中混用旧 artifacts。
