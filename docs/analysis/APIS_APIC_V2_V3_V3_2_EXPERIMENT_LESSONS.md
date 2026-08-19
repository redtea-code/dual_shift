# APIS v2、APIC v3 与 APIC v3_2 的实验教训

日期：2026-08-07
状态：研究复盘；不是新的性能主张或实验协议
范围：ADNI/NACC 的 image-only 跨队列筛选、APIS v2 Claim E1 与 APIC v3/v3_2 机制开发记录。

## 1. 结论先行

截至本文件引用的记录，三个版本均**没有形成可发表的、普遍有效的跨域性能主张**。它们提供的证据强度不同，不能混为“同一种失败”。

| 版本 | 可支持的结论 | 不可支持的结论 | 当前研究处置 |
| --- | --- | --- | --- |
| APIS v2 | CN vs AD 出现过任务/方向依赖的正向趋势；MCI vs AD 的已完成比较未显示相对 MixStyle 的优势。 | 稳定优于 MixStyle、跨任务普遍有效、扫描参数的因果收益。 | 保留为历史基线与不完整 Claim E1；不以其当前结果作正向结论。 |
| APIC v3 | 在 CN 与 MCI 已诊断单元中，特征干预近恒等，且存在 style-memory 塌缩与基础过拟合。 | 观测到的个别 BA 胜出由 APIC 干预造成。 | 冻结为负对照；不扩 seed 或模态。 |
| APIC v3_2 | 初版的零干预尺度错误已定位并修复；修复后可出现有界、可审计的非零干预。 | 修复已带来稳定跨域收益，或 M0 已关闭。 | 停止把 v3_2 作为正式候选；保留诊断与实现资产，v4 从新的可证伪契约开始。 |

这里的“未形成主张”不等于证明任何方法在所有数据、任务和实现中无效；它只表示在当前数据、协议、完整度和诊断证据下，正向泛化或机制结论没有被支持。

## 2. 证据范围与术语

- **APIS v2** 是早期的 scan/protocol-aware 方法系列；**APIC v3/v3_2** 是后续 image-only style/prototype 干预系列。名称相近，不应把它们的结果、配置或主张门槛混合。
- 主性能终点应为 target split 的 subject-mean balanced accuracy（BA），并按任务和方向分别报告。AUC、scan-level 指标或单一 seed 不能替代该终点。
- 本文将“机制有效”限定为：在可支持样本上，干预真实执行、其反事实输出可审计，且不会破坏 clean 决策；这本身仍不等于性能收益或因果识别。
- ADNI->NACC 与 NACC->ADNI 是不同的问题。NACC 缺少 1.5T 且协议变化有限，NACC->ADNI 包含 source 支持集外的 1.5T/厂家变化；双向平均不能掩盖该不对称性。

数据可识别性和允许的主张边界以 `docs/decisions/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md` 为准。

## 3. APIS v2：先验证完整度与可复现性，再解释趋势

### 已观察到的事实

1. 早期 CN vs AD 三随机种子 Gate A 未通过：两个方向均为 2/3 通过，且 Windows 与 Linux 在 seed43 NACC->ADNI 上给出方向相反的大幅差异。Windows 的 six-cell Gate A 因此为 No-Go。
2. Claim E1 的后续结果仍不完整。CN vs AD 中存在相对 MixStyle 的小幅、seed 间不稳定趋势；MCI vs AD 已完成的 seed x direction 比较中，APIS v2 主终点均低于 MixStyle。
3. metadata 与 metadata_xda 在多个单元接近 BA=0.5 或呈类别塌缩。因此，APIS 相对这些基线的差值不能解释为 scan/protocol mechanism 的优势。

### 从中得到的教训

**性能趋势不是机制证据。** 即使 APIS 在某些单元提高 AUC 或 BA，也不能据此断言模型学习了扫描参数响应；当前采集参数与 cohort、厂家、site 和 diagnosis 的混杂无法支持该因果解释。

**未完成矩阵不能被“已有均值”替代。** 缺失 seed、变体或方向必须在主表中保留为未完成/失败，而不是从汇总中消失。结果应同时报告计划单元数、完成单元数和失败原因。

**跨主机不一致首先是复现问题。** 在同一 seed/方向中出现相反结论时，首要工作是冻结并记录代码 commit、配置/manifest hash、依赖版本、GPU、确定性设置、checkpoint 和原始日志；不能把不同环境的结果拼成一个确认性均值。

**弱基线不构成比较优势。** metadata 分支的类别塌缩提示应先审计标签映射、选模、标准化、缺失处理与特征融合。任何方法都不应以失败的 metadata 基线作为主要机制参照。

## 4. APIC v3：必须从“参数非零”升级到“反事实输出非恒等”

### 已观察到的事实

1. MCI vs AD 的完整 primary 矩阵（2 seeds x 2 directions）中，APIC v3 target BA 平均为 0.6439，低于 CE 的 0.6552 与 MixStyle 的 0.6741；四个单元均未超过 MixStyle。
2. CN vs AD 的 4 个可用单元仅有 1 个同时超过 CE 和 MixStyle；两个方向的两-seed 平均均未同时超过两种基线。
3. CN 和 MCI 的 checkpoint 反事实诊断一致显示：虽然 `valid_intervention` 可为真，但 gate 约为 0.03，layer1/2 RMS、JS divergence 与 prediction flip 接近零，clean 与 shifted 指标几乎相同。
4. style memory 经常高度集中于单一 slot，例如 MCI 的最大 slot share 达到约 92%--99.5%，部分诊断样本的 prototype assignment 退化为单一取值。与此同时，训练集接近拟合而 validation/target 明显变差。

### 从中得到的教训

**有效标志不等于有效作用。** `valid_intervention_frac=1`、非空 memory、非零的内部损失或系数，都不能证明分类网络实际感受到了干预。必须在 checkpoint 上报告逐样本 layer RMS、embedding distance、概率 JS、flip 和 clean/shift 指标。

**个别性能胜出在近恒等机制下不可归因。** 当 clean 与 shifted 几乎相同，某次 APIC 胜出更可能是常规训练、初始化或选模波动；它不能作为 style transport 的证据。

**原型数量不是原型多样性。** `K=8` 且所有 slot 标记有效并不排除赋值几乎集中在一个 slot。后续原型方法必须审计 subject occupancy、最大占比、slot 间距离、assignment 的稳定性以及与标签/采集变量的关联；若分配退化，应停止而非把空 slot 人为视为有效。

**干预退化与基础过拟合须分开诊断。** APIC v3 的主要机制失败是近恒等；同时存在 train/source 与 validation/target 的明显裂缝。前者说明干预没有发挥作用，后者说明即使 clean 网络也缺乏稳健泛化；修复其中一个不能自动修复另一个。

## 5. APIC v3_2：修复零干预后，暴露出更深的训练与选择问题

### 5.1 修复前：尺度不一致导致确定性的零干预

v3_2 初版的 K=4 bank 表面上有有效 slot 和非空计数，但 prototype 的 PCA 欧氏距离约为十几到几十，而合法替代的 `delta_max` 约为 0.5。所有替代 target 因尺度不匹配而被拒绝，最终 target 等于 source，gate/RMS/JS/flip 均为零。此阶段的性能差异是 clean-path 波动，不能归因于 APIC。

教训是：**任何有阈值的几何门控，都必须在其实际表征坐标和尺度上做合成单元测试与真实 checkpoint 审计。** 仅检查 K-means 成功、slot 计数或代码不报错不足以发现“全样本被门控关闭”。

### 5.2 修复后：机制路径可执行，但并未建立性能收益

相对 prototype distance（用 calibration radius 归一化）、严格 fit/calibration 检查、全覆盖 bank-build loader、设备对齐和 `last_checkpoint` 审计之后：

- 3090 的 CN 验收显示，四个检查单元均有 in-band 替代、非零 supported fraction 和 bounded layer RMS；unsupported 样本严格回退至 clean 输出，clean reload 与导出预测可在 `1e-5` 内匹配。
- 5090 的 MCI ADNI->NACC post-bank checkpoint 也具有 4 个有效 slot、非零支持与非零 RMS，但 target clean/shift BA 为 0.6175/0.6178，flip 为 0.0046，JS 约 `1.6e-5`。这说明路径可运行，**不说明它改善了分类**。
- 5090 的 MCI NACC->ADNI 在 post-bank epoch 7 中 clean 和 shifted BA 都为 0.5、JS 为 0，预测全部为 class 0；持续训练到 epoch 50 后多数预测转为 class 1，但所有 split 的 BA 仍约 0.48--0.52。该塌缩发生在 clean 与 shifted 路径中，不能仅归因于一次特征平移。

### 5.3 v3_2 的关键教训

**“RMS 非零”只是必要条件。** 有界特征变化、band hit 和严格 fallback 证明实现没有退化为恒等，但不能保证分类器的决策产生有意义或有益的变化。机制 gate 必须同时包含 clean 稳定性、预测分布、分类塌缩和目标性能的检查。

**性能选择器必须知道机制阶段。** NACC->ADNI 的 published best 停留在 clean warmup epoch 3/4，`valid_slots=0`；真正含 bank 的 epoch 7/50 反而塌缩。因而该 published BA 不能解释为 APIC v3_2 效果。主结果的 checkpoint 必须至少满足 bank 已 finalize、所有审计门通过和 clean 分支未塌缩；也应并列保存 performance-best 与 post-bank/last checkpoint，禁止用前者掩盖后者。

**bank 成功不等于训练目标稳定。** 设备不一致曾使真实 B0 bank finalize 崩溃；WeightedRandomSampler 也会在小 cohort 中漏掉 calibration slot。修复后虽可建立 bank，但 NACC->ADNI 的 clean/shift 共同塌缩表明，bank 构建、阶段切换、损失权重和优化状态仍需要作为一个整体被验证。

**强制 calibration floor 是工程折衷，不能被误写为自然数据支持。** 它可在样本数足够时给每个有效 slot 分配最邻近的 calibration 样本，而不移动 k-means fit 成员。后续报告应区分自然 nearest assignment 与 floor 强制分配数，并报告每 slot radius/距离分布；否则“4/4 slot 支持”可能高估稳定的风格覆盖。

**诊断导出也属于研究方法。** clean checkpoint 可能没有 teacher state，snapshot 可能没有对应的正式预测 CSV；若加载与 reference 对齐不完整，`1e-5` 一致性不能宣称通过。自动化监控脚本的失败也不能靠人工补跑后忽略，必须被记录为工程失败并在正式实验前关闭。

## 6. 跨版本的共同原则

1. **分离三类结论。** 代码运行、机制路径开启、性能改善和采集机制解释是四个不同层次；每层都需要自己的预先定义指标与否决条件。
2. **先证伪，再扩矩阵。** 新模块先在单方向、单 seed、冻结 checkpoint 的最小试验中证明：干预会发生、unsupported 回退、clean 不塌缩、反事实差异可测且有方向性。未通过前不扩 seed、任务或模态。
3. **以 subject 为独立单位。** ADNI 的重复扫描不能被当作独立样本；性能、bootstrap 和配对比较应使用 subject-level 聚合，并明确 label-conflict 规则。
4. **方向与支持集分层。** 报告 `target_full`、共同支持子集、unsupported-protocol stress test 和 source/target support fraction。不得用双向平均、共同支持结果或 target 外的 validation 结果替代主终点。
5. **每个 checkpoint 都必须可重建。** 归档完整 commit、配置和 manifest hash、环境、模型/teacher/bank 状态、selector history、预测 CSV 和诊断输出；不要只保留最终汇总数字。
6. **负对照和失败基线必须保留。** scan-only、missingness-only、acquisition shuffle、image+scan concat、clean image-only 以及 collapse 的 metadata 结果应随主结果报告。它们限制可作出的解释，而不是应被删除的噪声。
7. **停止规则保护研究效度。** 当机制近恒等、clean/shift 共同塌缩、checkpoint 与机制阶段脱钩，或复现资产不完整时，应停止性能扩展并新开 protocol revision，而不是在 target 指标上反复调参。

## 7. 对 v4 的最低研究要求

本节只定义进入 v4 的最低证据要求，不预先指定网络结构。

1. **明确可证伪目标。** v4 首先主张“在 source-observed support 内降低同一 subject 跨协议预测不稳定性，同时不降低诊断判别力”；不主张从 NACC 单一协议外推出 1.5T 或学习采集参数的因果响应。
2. **冻结两阶段职责。** 预测器与风格/协议模块应有明确训练边界；若干预器训练会改变 clean 分类器，必须额外证明 clean 路径没有退化。
3. **机制感知的选择与停止。** 候选 checkpoint 至少须满足 finalized bank/support、clean 非塌缩、严格 fallback、可重建预测和预定义的机制审计。任何一项失败均不能参加性能排名。
4. **独立的机制 fit/calibration/test。** 机制的 prototype/radius 拟合、校准和最终诊断需 subject-disjoint；报告自然覆盖、校准覆盖和 unsupported 比例，而非只报告 K。
5. **正式扩展前的最小验收。** 对预先选定的一个方向运行 clean、干预、shuffle/identity 等对照，并在源验证和目标上共同报告 BA、AUC、SEN/SPE、Brier/ECE、JS/flip、支持率和 paired consistency。只有机制与性能门都通过，才可扩 seed 或比较更多模态。

## 8. 可用与禁用的论文表述

当前可用的内部/论文表述：

> 在当前 ADNI/NACC 数据和冻结协议下，APIS/APIC 系列未显示跨任务、跨方向稳定的性能收益。机制诊断表明，APIC v3 主要退化为近恒等干预；APIC v3_2 的尺度修复使干预可观测，但发现了 post-bank 训练塌缩与 checkpoint 选择脱钩，因此不进入正式性能比较。

当前禁止的表述：

- “APIS/APIC 已证明扫描参数或风格建模普遍提升域泛化”；
- “个别 seed 的 target BA 提升证明了机制有效”；
- “v3_2 修复后已通过机制验收并可进入正式 E3”；
- “NACC->ADNI 的结果证明模型学会了 1.5T 或厂家校正”；
- “metadata 基线失败证明 image-only 方法具有因果优势”。

## 9. 主要证据索引

除当前 `docs/` 中的文档外，下列 `review/...` 均是历史证据路径，保存在 Git ref
`archive/pre-plan34-cleanup-20260809`，不是当前 `main` 的工作文档。

- APIS v2 三种子 Gate A：`review/analysis/08_dual_shift_apis_3seed_analysis_2026-07-30.md`。
- APIS v2 跨任务中期判断：`review/analysis/18_apis_v2_interim_cross_task_assessment_2026-08-03.md`。
- 数据现实与主张边界：`docs/decisions/SCAN_AWARE_DATA_REALITY_AND_CLAIM_BOUNDARY.md`。
- APIC v3 CN 诊断：`review/records/apic_v3/3090/25_apic_v3_cn_failure_diagnosis_2026-08-04.md`。
- APIC v3 MCI 诊断：`review/records/apic_v3/5090/25_apic_v3_failure_diagnostics_mci_ad_2026-08-04.md`。
- APIC v3_2 实现复审与修复复审：`review/analysis/27_apic_v3_2_implementation_review_2026-08-04.md`、`review/analysis/28_apic_v3_2_defect_remediation_review_2026-08-04.md`。
- APIC v3_2 修复前机制诊断：`review/analysis/35_apic_v3_2_mci_mechanism_diagnosis_report_2026-08-05.md`。
- APIC v3_2 M0 修复后验收：`review/analysis/37_apic_v3_2_m0_scale_repair_experiment_report_2026-08-06.md`、`review/records/apic_v3_2/5090/apic_v3_2_m0_acceptance_2026-08-06/GO_NO_GO.md`、`review/records/apic_v3_2/3090/37_apic_v3_2_m0_cn_scale_repair_acceptance_2026-08-06.md`。
