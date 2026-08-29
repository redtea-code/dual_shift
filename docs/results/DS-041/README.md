# DS-041：CAPM 条件化的 source-free residual distribution alignment

Status: **COMPLETED STAGE B / EXPLORATORY PILOT / NO STAGE C**

本报告基于 DS-041 的完整 real-data pilot 矩阵，汇总 2 个迁移方向、2 个随机种子和 C0–C4 五个注册变体，共 **20/20 cells**。报告同时收集分类性能、source preservation、分布 discrepancy、projector 与 C4 synthetic perturbation audit。所有 target adaptation statistics 均在不读取 target labels 和 target metrics 的条件下生成；target labels 仅用于最终 exploratory evaluation。

## 1. 执行摘要

- C0–C4 的 20 个 pilot cell 均已生成 `report.json`、`audit.json`、`predictions.json` 和 `best.pt`；4 组方向/seed 的 projector 与 16 个 statistics artifact 亦齐全。
- residual transport 的统计目标确实被压缩：相对于 full-scope K=1，residual-scope 的 discrepancy 整体更低；但 discrepancy 降低没有转化为跨方向、跨 seed 稳定的 target 分类增益。
- target BA、AUROC、Macro-F1、Sensitivity 与 Specificity 具有明显方向和 seed 依赖；部分 adapted cell 接近随机分类或表现出极端 operating-point 偏置。
- C4 的输出 finite、预测一致性保持，但 logit recovery fraction 未显示可靠的正向恢复，因此 **synthetic recovery gate 不通过**。
- 结论限于 exploratory pilot：不能据此宣称稳定 domain adaptation、harmonization superiority 或 biological preservation。按注册决策规则，暂不进入 Stage C，也不新增 C5/C6、rank、strength 或 seed。

## 2. 实验协议

| 项目 | 固定值 |
|---|---|
| 任务 | Scan-filtered MCI vs AD |
| 主方向 | ADNI→NACC |
| 扩展方向 | NACC→ADNI |
| Backbone | Frozen `original_capm` |
| Feature preset | `layer4_pixel` |
| ResNet10 | `layers=[1,1,1,1]` |
| CAPM variables | age, sex, education |
| Seeds | 42, 43 |
| Projector | Source-only TaskSupportProjector, rank 32 |
| Transport | Full K=1；residual K=1；residual K=2 |
| 最大 transport strength | 0.25 |
| Adaptation | Frozen backbone；target image-only、subject-disjoint summary；no retraining |
| Checkpoint selection | Source-validation-only |
| Target evaluation | Post-selection exploratory report |

### 注册变体

| ID | 名称 | 作用 |
|---|---|---|
| C0 | `capm_control` | Frozen original-CAPM control |
| C1 | `full_diag_transport` | Full feature、K=1 diagonal transport control |
| C2 | `residual_diag_transport` | Residual、K=1 diagonal transport |
| C3 | `residual_gmm_transport` | Residual、K=2 GMM transport，主要候选 |
| C4 | `residual_gmm_perturbation` | C3 + source-only synthetic perturbation audit |

## 3. 评价指标定义

分类指标均按 subject-level aggregation 后计算：

- **Accuracy**：总体分类正确率。
- **BA (Balanced Accuracy)**：Sensitivity 与 Specificity 的平均，适用于类别不平衡。
- **AUROC**：概率排序层面的区分能力。
- **Macro-F1**：各类别 F1 的非加权平均。
- **Sensitivity / Recall**：正类召回率。
- **Specificity**：负类召回率。
- **Precision**：正类预测中的正确比例；本轮由 `predictions.json` 结合正类定义重算。
- **MCC**：基于 TP/TN/FP/FN 的相关系数，作为不平衡数据下的补充指标。
- **Brier score**：概率预测的均方误差，越低越好。
- **ECE**：Expected Calibration Error，越低表示校准更好。
- **Group gap / worst-group AUROC**：按 environment/group 的性能差异，用于检查是否由单一环境主导。

除 target 指标外，报告还检查 source control 与 source adapted diagnostic 的差异，以及 transport 的 finite、gate activity、correction RMS、anchor drift 和 identity loss。

## 4. Target-test 逐 cell 结果

以下是每个 cell 的 target exploratory evaluation。`n` 为 subject-level 样本数；`BA/AUROC/Macro-F1/Sens/Spec/Acc/Precision/MCC/Brier/ECE` 均为直接或由报告预测重算的指标。

| Cell | n | BA | AUROC | Macro-F1 | Acc | Precision | Sensitivity | Specificity | MCC | Brier | ECE | Confusion matrix (TN,FP;FN,TP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed42 C0 | 263 | 0.6461 | 0.7483 | 0.6561 | 0.7414 | 0.6441 | 0.3765 | 0.9157 | 0.2997 | 0.2089 | 0.1602 | [192,21; 73,38] |
| seed42 C1 | 263 | 0.6461 | 0.7483 | 0.6561 | 0.7414 | 0.6441 | 0.3765 | 0.9157 | 0.2997 | 0.2089 | 0.1602 | [192,21; 73,38] |
| seed42 C2 | 263 | 0.6223 | 0.7486 | 0.6275 | 0.7300 | 0.6200 | 0.3176 | 0.9270 | 0.2497 | 0.2101 | 0.1646 | [194,19; 80,31] |
| seed42 C3 | 263 | 0.6223 | 0.7486 | 0.6275 | 0.7300 | 0.6200 | 0.3176 | 0.9270 | 0.2497 | 0.2101 | 0.1646 | [194,19; 80,31] |
| seed42 C4 | 263 | 0.6223 | 0.7486 | 0.6275 | 0.7300 | 0.6200 | 0.3176 | 0.9270 | 0.2497 | 0.2101 | 0.1646 | [194,19; 80,31] |
| seed43 C0 | 263 | 0.6735 | 0.7726 | 0.6837 | 0.7452 | 0.6538 | 0.4706 | 0.8764 | 0.3346 | 0.2103 | 0.1912 | [174,27; 67,51] |
| seed43 C1 | 263 | 0.6735 | 0.7726 | 0.6837 | 0.7452 | 0.6538 | 0.4706 | 0.8764 | 0.3346 | 0.2103 | 0.1912 | [174,27; 67,51] |
| seed43 C2 | 263 | 0.6410 | 0.7740 | 0.6496 | 0.7262 | 0.6338 | 0.4000 | 0.8820 | 0.2925 | 0.2082 | 0.1913 | [175,26; 73,45] |
| seed43 C3 | 263 | 0.6410 | 0.7740 | 0.6496 | 0.7262 | 0.6338 | 0.4000 | 0.8820 | 0.2925 | 0.2082 | 0.1913 | [175,26; 73,45] |
| seed43 C4 | 263 | 0.6410 | 0.7740 | 0.6496 | 0.7262 | 0.6338 | 0.4000 | 0.8820 | 0.2925 | 0.2082 | 0.1913 | [175,26; 73,45] |

### NACC→ADNI（逐 cell）

| Cell | n | BA | AUROC | Macro-F1 | Acc | Precision | Sensitivity | Specificity | MCC | Brier | ECE | Confusion matrix (TN,FP;FN,TP) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed42 C0 | 121 | 0.6649 | 0.7223 | 0.6691 | 0.7190 | 0.6642 | 0.5128 | 0.8171 | 0.3099 | 0.2439 | 0.2538 | [178,46; 91,91] |
| seed42 C1 | 121 | 0.6649 | 0.7223 | 0.6691 | 0.7190 | 0.6642 | 0.5128 | 0.8171 | 0.3099 | 0.2439 | 0.2538 | [178,46; 91,91] |
| seed42 C2 | 121 | 0.6778 | 0.7154 | 0.6812 | 0.7273 | 0.6667 | 0.5385 | 0.8171 | 0.3099 | 0.2399 | 0.2164 | [179,45; 92,90] |
| seed42 C3 | 121 | 0.6778 | 0.7154 | 0.6812 | 0.7273 | 0.6667 | 0.5385 | 0.8171 | 0.3099 | 0.2399 | 0.2164 | [179,45; 92,90] |
| seed42 C4 | 121 | 0.6778 | 0.7154 | 0.6812 | 0.7273 | 0.6667 | 0.5385 | 0.8171 | 0.3099 | 0.2399 | 0.2164 | [179,45; 92,90] |
| seed43 C0 | 121 | 0.5664 | 0.6477 | 0.5671 | 0.6033 | 0.5704 | 0.4222 | 0.7105 | 0.1793 | 0.2864 | 0.2619 | [157,61; 98,81] |
| seed43 C1 | 121 | 0.5664 | 0.6477 | 0.5671 | 0.6033 | 0.5704 | 0.4222 | 0.7105 | 0.1793 | 0.2864 | 0.2619 | [157,61; 98,81] |
| seed43 C2 | 121 | 0.5882 | 0.6480 | 0.5892 | 0.6364 | 0.5887 | 0.4000 | 0.7763 | 0.1867 | 0.2758 | 0.2172 | [167,51; 106,73] |
| seed43 C3 | 121 | 0.5882 | 0.6480 | 0.5892 | 0.6364 | 0.5887 | 0.4000 | 0.7763 | 0.1867 | 0.2758 | 0.2172 | [167,51; 106,73] |
| seed43 C4 | 121 | 0.5882 | 0.6480 | 0.5892 | 0.6364 | 0.5887 | 0.4000 | 0.7763 | 0.1867 | 0.2758 | 0.2172 | [167,51; 106,73] |

## 5. Target 结果：双 seed 汇总

表中为 seed 42/43 的 mean ± sample SD。双 seed 仅用于描述稳定性，不能替代正式统计推断。

### ADNI→NACC（primary）

| Variant | BA | AUROC | Macro-F1 | Acc | Precision | Sensitivity | Specificity | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.6598 ± 0.0194 | 0.7605 ± 0.0172 | 0.6699 ± 0.0195 | 0.7433 ± 0.0027 | 0.6490 ± 0.0069 | 0.4235 ± 0.0666 | 0.8961 ± 0.0278 | 0.3172 ± 0.0246 | 0.2096 ± 0.0010 | 0.1757 ± 0.0220 |
| C1 | 0.6598 ± 0.0194 | 0.7605 ± 0.0172 | 0.6699 ± 0.0195 | 0.7433 ± 0.0027 | 0.6490 ± 0.0069 | 0.4235 ± 0.0666 | 0.8961 ± 0.0278 | 0.3172 ± 0.0246 | 0.2096 ± 0.0010 | 0.1757 ± 0.0220 |
| C2 | 0.6317 ± 0.0132 | 0.7613 ± 0.0179 | 0.6385 ± 0.0156 | 0.7281 ± 0.0027 | 0.6269 ± 0.0098 | 0.3588 ± 0.0582 | 0.9045 ± 0.0318 | 0.2711 ± 0.0302 | 0.2092 ± 0.0014 | 0.1779 ± 0.0189 |
| C3 | 0.6317 ± 0.0132 | 0.7613 ± 0.0179 | 0.6385 ± 0.0156 | 0.7281 ± 0.0027 | 0.6269 ± 0.0098 | 0.3588 ± 0.0582 | 0.9045 ± 0.0318 | 0.2711 ± 0.0302 | 0.2092 ± 0.0014 | 0.1779 ± 0.0189 |
| C4 | 0.6317 ± 0.0132 | 0.7613 ± 0.0179 | 0.6385 ± 0.0156 | 0.7281 ± 0.0027 | 0.6269 ± 0.0098 | 0.3588 ± 0.0582 | 0.9045 ± 0.0318 | 0.2711 ± 0.0302 | 0.2092 ± 0.0014 | 0.1779 ± 0.0189 |

### NACC→ADNI（prespecified extension）

| Variant | BA | AUROC | Macro-F1 | Acc | Precision | Sensitivity | Specificity | MCC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.6157 ± 0.0697 | 0.6850 ± 0.0528 | 0.6181 ± 0.0721 | 0.6612 ± 0.0818 | 0.6173 ± 0.0663 | 0.4675 ± 0.0641 | 0.7638 ± 0.0753 | 0.2446 ± 0.0924 | 0.2652 ± 0.0301 | 0.2579 ± 0.0057 |
| C1 | 0.6157 ± 0.0697 | 0.6850 ± 0.0528 | 0.6181 ± 0.0721 | 0.6612 ± 0.0818 | 0.6173 ± 0.0663 | 0.4675 ± 0.0641 | 0.7638 ± 0.0753 | 0.2446 ± 0.0924 | 0.2652 ± 0.0301 | 0.2579 ± 0.0057 |
| C2 | 0.6330 ± 0.0634 | 0.6817 ± 0.0477 | 0.6352 ± 0.0651 | 0.6818 ± 0.0643 | 0.6277 ± 0.0551 | 0.4692 ± 0.0979 | 0.7967 ± 0.0288 | 0.2483 ± 0.0872 | 0.2578 ± 0.0254 | 0.2168 ± 0.0006 |
| C3 | 0.6330 ± 0.0634 | 0.6817 ± 0.0477 | 0.6352 ± 0.0651 | 0.6818 ± 0.0643 | 0.6277 ± 0.0551 | 0.4692 ± 0.0979 | 0.7967 ± 0.0288 | 0.2483 ± 0.0872 | 0.2578 ± 0.0254 | 0.2168 ± 0.0006 |
| C4 | 0.6330 ± 0.0634 | 0.6817 ± 0.0477 | 0.6352 ± 0.0651 | 0.6818 ± 0.0643 | 0.6277 ± 0.0551 | 0.4692 ± 0.0979 | 0.7967 ± 0.0288 | 0.2483 ± 0.0872 | 0.2578 ± 0.0254 | 0.2168 ± 0.0006 |

## 6. Source-side preservation

Source-side control 与 adapted diagnostic 用于检查适配是否破坏冻结 source task representation。它们不是 target 变体选择器。

| Direction | Variant | Source control BA | Source adapted BA | Δ BA | Source control AUROC | Source adapted AUROC | Δ AUROC | Source control MCC | Source adapted MCC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ADNI→to→NACC | C0 | 0.6984 | 0.6908 | -0.0076 | 0.8087 | 0.8078 | -0.0009 | 0.3183 | 0.3038 |
| ADNI→to→NACC | C1 | 0.6984 | 0.6908 | -0.0076 | 0.8087 | 0.8078 | -0.0009 | 0.3183 | 0.3038 |
| ADNI→to→NACC | C2 | 0.6984 | 0.6984 | +0.0000 | 0.8087 | 0.8087 | +0.0000 | 0.3183 | 0.3183 |
| ADNI→to→NACC | C3 | 0.6984 | 0.6984 | +0.0000 | 0.8087 | 0.8087 | +0.0000 | 0.3183 | 0.3183 |
| ADNI→to→NACC | C4 | 0.6984 | 0.6984 | +0.0000 | 0.8087 | 0.8087 | +0.0000 | 0.3183 | 0.3183 |
| NACC→to→ADNI | C0 | 0.7355 | 0.7355 | +0.0000 | 0.8395 | 0.8405 | +0.0010 | 0.4101 | 0.3952 |
| NACC→to→ADNI | C1 | 0.7355 | 0.7355 | +0.0000 | 0.8395 | 0.8405 | +0.0010 | 0.4101 | 0.3952 |
| NACC→to→ADNI | C2 | 0.7355 | 0.7355 | +0.0000 | 0.8395 | 0.8395 | +0.0000 | 0.4101 | 0.4101 |
| NACC→to→ADNI | C3 | 0.7355 | 0.7355 | +0.0000 | 0.8395 | 0.8395 | +0.0000 | 0.4101 | 0.4101 |
| NACC→to→ADNI | C4 | 0.7355 | 0.7355 | +0.0000 | 0.8395 | 0.8395 | +0.0000 | 0.4101 | 0.4101 |

重点解释：若 source adapted 的 BA/AUROC 与 source control 基本一致，只能说明在该 source diagnostic 上没有明显破坏；不能推出 target 泛化或 biology preservation。若出现变化，则应结合 sensitivity、specificity、Brier、ECE 和 group gap 判断，而不能只看 BA。

## 7. Distribution discrepancy

每个 statistics artifact 在 512 个 feature channels 上记录 source-target discrepancy。下表报告 mean、P50、P90、maximum；数值越低表示统计摘要在该定义下越接近。residual 结果不等价于完整 feature distribution 已被对齐。

| Direction | Seed | Scope / K | Mean | P50 | P90 | Max | Source subjects | Target-adapt subjects |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| ADNI→to→NACC | 42 | full / K=1 | 0.3243 | 0.3147 | 0.5094 | 1.0000 | 144 | 264 |
| ADNI→to→NACC | 42 | residual / K=1 | 0.2758 | 0.2577 | 0.5121 | 1.0000 | 144 | 264 |
| ADNI→to→NACC | 42 | residual / K=2 | 0.2377 | 0.1820 | 0.5021 | 1.0000 | 144 | 264 |
| ADNI→to→NACC | 43 | full / K=1 | 0.4123 | 0.3525 | 0.6974 | 1.0000 | 144 | 264 |
| ADNI→to→NACC | 43 | residual / K=1 | 0.2360 | 0.1930 | 0.4585 | 1.0000 | 144 | 264 |
| ADNI→to→NACC | 43 | residual / K=2 | 0.2698 | 0.2357 | 0.5157 | 1.0000 | 144 | 264 |
| NACC→to→ADNI | 42 | full / K=1 | 0.4268 | 0.3569 | 0.6932 | 1.0000 | 316 | 120 |
| NACC→to→ADNI | 42 | residual / K=1 | 0.3519 | 0.2868 | 0.6659 | 1.0000 | 316 | 120 |
| NACC→to→ADNI | 42 | residual / K=2 | 0.2761 | 0.2176 | 0.5972 | 1.0000 | 316 | 120 |
| NACC→to→ADNI | 43 | full / K=1 | 0.5216 | 0.5312 | 0.6900 | 1.0000 | 316 | 120 |
| NACC→to→ADNI | 43 | residual / K=1 | 0.3795 | 0.3967 | 0.5809 | 1.0000 | 316 | 120 |
| NACC→to→ADNI | 43 | residual / K=2 | 0.0890 | 0.0593 | 0.2067 | 1.0000 | 316 | 120 |

比较原则：

1. C1 与 C2/C3 的比较用于判断把 transport 限制在 residual branch 是否减少统计差异。
2. C2 与 C3 的比较用于判断 K=2 GMM 是否相对 K=1 diagonal moments 改善统计描述；它不自动证明分类性能更好。
3. discrepancy 是 adaptation statistics 的机制诊断，不是 target label-based performance metric。

## 8. Projector 与 provenance

| Direction | Seed | Channels | Rank | Source count (fit) | Source train subjects | Top eigenvalue mass |
|---|---:|---:|---:|---:|---:|---:|
| ADNI→to→NACC | 42 | 512 | 32 | 490 | 144 | 1.000000 |
| ADNI→to→NACC | 43 | 512 | 32 | 490 | 144 | 0.999999 |
| NACC→to→ADNI | 42 | 512 | 32 | 389 | 316 | 0.999999 |
| NACC→to→ADNI | 43 | 512 | 32 | 389 | 316 | 0.999999 |

Projector 由 source-only CE-gradient covariance 拟合，rank 固定为 32，channels 为 512。projector metadata 记录 source checkpoint、source split、source subject digest、config hash，以及 `target_labels_read=false` 和 `target_metrics_read=false`。所有报告均验证：

- `preset=layer4_pixel`，输入形状为 `[160,196,160]`；
- source checkpoint 与 DS-041 模型结构匹配；
- checkpoint 选择为 source-validation-only；
- target labels 未用于 adaptation；
- target metrics 只出现在 final exploratory report；
- C4 audit finite 且使用 batch size 2。

## 9. C4 Synthetic perturbation audit

C4 对 source-only clean image 施加 bounded intensity 与 Fourier amplitude perturbation，比较 transport 前后 clean/perturbed logits 的距离，并检查预测一致性。`recovery fraction = 1 - after / before`，正值才表示扰动后的 logit 距离下降。

| Direction | Seed | Before | After | Recovery fraction | Clean agreement | Perturbed agreement | Finite | n scans | Audit batch |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| ADNI→to→NACC | 42 | 0.058491 | 0.058492 | -0.000020 | 1.0000 | 1.0000 | True | 149 | 2 |
| ADNI→to→NACC | 43 | 0.087015 | 0.087407 | -0.004497 | 1.0000 | 1.0000 | True | 149 | 2 |
| NACC→to→ADNI | 42 | 0.348207 | 0.348216 | -0.000026 | 1.0000 | 1.0000 | True | 132 | 2 |
| NACC→to→ADNI | 43 | 0.574528 | 0.574537 | -0.000015 | 1.0000 | 1.0000 | True | 132 | 2 |

本轮 C4 的 finite 与 prediction-agreement 检查通过，但 recovery fraction 未形成一致的正向恢复证据，因此 C4 mechanism gate 不通过。该结果也说明“输出有限”和“预测标签未改变”不能替代真正的 logit recovery。

## 10. 决策规则判定

| 决策规则 | 结果 | 解释 |
|---|---|---|
| Stage A synthetic/interface gate | 通过 | 统计 determinism、zero-strength identity、residual anchor、label-blind loader 与 batched audit interface 已验证 |
| Stage B real-data execution | 通过 | 20/20 pilot cells 与支持 artifact 完成 |
| residual discrepancy reduction | 支持 | residual-scope discrepancy 相对 full-scope 控制降低；属于机制层证据 |
| target performance superiority | 不支持 | 方向/seed 依赖明显，未观察到稳定跨 cell 增益 |
| source anchor/preservation | 部分支持 | finite、anchor 与 source diagnostic 需结合具体 cell 解读，不能外推 biology preservation |
| C4 synthetic recovery | 不通过 | recovery fraction 未显示稳定正向恢复 |
| source-free compliance | 通过 | adaptation statistics 未读取 target labels/metrics；target labels 仅用于最终报告 |
| 进入 Stage C | 否 | C4 gate 未通过且分类增益不稳定 |

## 11. 主要限制

1. 只有两个 seeds，SD 主要用于描述，不足以提供稳健的置信区间或显著性结论。
2. target-test 结果属于 exploratory post-selection reporting，不应视为完全独立的确认性测试。
3. 统计 discrepancy 是按预定义 channel-wise summary 计算的距离，不等价于所有高阶特征分布、scanner effect 或 biological nuisance 已消除。
4. C4 使用合成强度/Fourier 扰动，不能替代真实 scanner perturbation 或生物学保持性验证。
5. 结果中若 sensitivity 与 specificity 高度不平衡，BA 可能掩盖 operating-point collapse，因此必须联合阅读 AUROC、Macro-F1、MCC、Brier、ECE 和混淆矩阵。
6. seed 43 的部分旧报告生成于 C4 OOM 修复前，历史 `pilot_code_sha256` 与当前代码不一致；这属于 provenance 版本差异，不改变已记录的 artifact 内容，但后续正式研究应绑定统一代码版本。

## 12. Artifact inventory

```text
ds041_runs/
├── assets/{ADNI_to_NACC,NACC_to_ADNI}/seed{42,43}/projector.json
├── stats/{ADNI_to_NACC,NACC_to_ADNI}/seed{42,43}/
│   ├── target_split.json
│   ├── full_k1.json
│   ├── residual_k1.json
│   └── residual_k2.json
└── pilots/{ADNI_to_NACC,NACC_to_ADNI}/seed{42,43}/
    └── {C0_capm_control,C1_full_diag_transport,C2_residual_diag_transport,
        C3_residual_gmm_transport,C4_residual_gmm_perturbation}/
        ├── report.json
        ├── audit.json
        ├── predictions.json
        └── best.pt
```

原始 artifacts 位于 `/zjs/AD_Project/ds041_runs`；本报告位于 `dual_shift_github/docs/results/DS-041/README.md`。本报告不复制大体积 checkpoint 和 prediction 文件。
