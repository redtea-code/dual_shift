# ResNet10 双向 ADNI/NACC 频域与冻结泛化对照（seed 43/44）

## 1. 实验范围

- Backbone：ResNet10，`layers=[1,1,1,1]`。
- Seeds：43、44；presets：`layer3_patch2`、`layer4_pixel`、`layer5_pixel`。
- 当前报告的新增方向：ADNI→NACC，`image_only` 与 `original_capm`；对照方向：既有 NACC→ADNI `image_only` 特征频谱。
- 特征频域分析使用 task-filtered 数据，并按 `subject_id` 进行 `earliest_visit` 去重：ADNI→NACC 每个 checkpoint 为 ADNI source-train 144、NACC target 527；NACC→ADNI 为 NACC source-train 316、ADNI target 241。
- 去重规则：按 `subject_id`、有效 `scan_date`（最早优先）、路径排序；缺少日期的记录排在有日期记录之后。频域表、比较统计和 domain classifier 均仅使用该去重后的样本。
- 所有 checkpoint 均先按 source validation BA + collapse guard 冻结；target 只做一次盲评估。

## 2. ADNI→NACC source-test 结果

每个 checkpoint 已在独立 ADNI held-out source-test split（49 subjects）上评估。下表是 subject-mean 结果及 subject-level bootstrap 95% CI。source-test 是 checkpoint 冻结后的独立审计集，未参与 source-validation BA + collapse guard 的选择，也未参与任何 target 相关决策。

| Preset | Seed | Variant | N | BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `layer3_patch2` | 43 | `image_only` | 49 | 0.548 [0.455,0.642] | 0.752 | 0.188 | 0.909 | 0.531 | 0.246 | 0.270 |
| `layer3_patch2` | 43 | `original_capm` | 49 | 0.782 [0.657,0.892] | 0.890 | 0.625 | 0.939 | 0.800 | 0.142 | 0.089 |
| `layer3_patch2` | 44 | `image_only` | 49 | 0.644 [0.504,0.797] | 0.769 | 0.500 | 0.788 | 0.646 | 0.229 | 0.199 |
| `layer3_patch2` | 44 | `original_capm` | 49 | 0.598 [0.457,0.753] | 0.722 | 0.438 | 0.758 | 0.599 | 0.253 | 0.229 |
| `layer4_pixel` | 43 | `image_only` | 49 | 0.647 [0.492,0.770] | 0.761 | 0.688 | 0.606 | 0.620 | 0.242 | 0.244 |
| `layer4_pixel` | 43 | `original_capm` | 49 | 0.644 [0.519,0.770] | 0.763 | 0.500 | 0.788 | 0.646 | 0.240 | 0.211 |
| `layer4_pixel` | 44 | `image_only` | 49 | 0.565 [0.455,0.720] | 0.759 | 0.312 | 0.818 | 0.565 | 0.197 | 0.135 |
| `layer4_pixel` | 44 | `original_capm` | 49 | 0.691 [0.536,0.830] | 0.775 | 0.625 | 0.758 | 0.685 | 0.235 | 0.204 |
| `layer5_pixel` | 43 | `image_only` | 49 | 0.643 [0.505,0.752] | 0.752 | 0.438 | 0.848 | 0.650 | 0.220 | 0.176 |
| `layer5_pixel` | 43 | `original_capm` | 49 | 0.630 [0.489,0.754] | 0.763 | 0.562 | 0.697 | 0.622 | 0.263 | 0.263 |
| `layer5_pixel` | 44 | `image_only` | 49 | 0.501 [0.439,0.579] | 0.795 | 0.062 | 0.939 | 0.445 | 0.236 | 0.236 |
| `layer5_pixel` | 44 | `original_capm` | 49 | 0.753 [0.598,0.860] | 0.850 | 0.688 | 0.818 | 0.749 | 0.151 | 0.036 |

## 3. ADNI→NACC target 泛化结果

| Preset | Seed | Variant | BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `layer3_patch2` | 43 | `image_only` | 0.519 [0.503,0.534] | 0.723 | 0.046 | 0.991 | 0.445 | 0.291 | 0.284 |
| `layer3_patch2` | 43 | `original_capm` | 0.604 [0.570,0.644] | 0.777 | 0.234 | 0.974 | 0.596 | 0.208 | 0.175 |
| `layer3_patch2` | 44 | `image_only` | 0.623 [0.591,0.657] | 0.758 | 0.280 | 0.966 | 0.623 | 0.226 | 0.193 |
| `layer3_patch2` | 44 | `original_capm` | 0.566 [0.540,0.594] | 0.755 | 0.154 | 0.977 | 0.536 | 0.251 | 0.231 |
| `layer4_pixel` | 43 | `image_only` | 0.645 [0.603,0.686] | 0.752 | 0.389 | 0.901 | 0.653 | 0.217 | 0.180 |
| `layer4_pixel` | 43 | `original_capm` | 0.643 [0.605,0.684] | 0.755 | 0.394 | 0.892 | 0.651 | 0.215 | 0.187 |
| `layer4_pixel` | 44 | `image_only` | 0.580 [0.553,0.607] | 0.766 | 0.189 | 0.972 | 0.560 | 0.211 | 0.154 |
| `layer4_pixel` | 44 | `original_capm` | 0.606 [0.579,0.633] | 0.746 | 0.280 | 0.932 | 0.605 | 0.239 | 0.221 |
| `layer5_pixel` | 43 | `image_only` | 0.554 [0.529,0.581] | 0.780 | 0.114 | 0.994 | 0.510 | 0.244 | 0.229 |
| `layer5_pixel` | 43 | `original_capm` | 0.673 [0.632,0.708] | 0.788 | 0.469 | 0.878 | 0.683 | 0.199 | 0.157 |
| `layer5_pixel` | 44 | `image_only` | 0.527 [0.510,0.550] | 0.806 | 0.057 | 0.997 | 0.458 | 0.265 | 0.261 |
| `layer5_pixel` | 44 | `original_capm` | 0.662 [0.622,0.699] | 0.760 | 0.469 | 0.855 | 0.670 | 0.185 | 0.080 |

## 4. ADNI→NACC CAPM 增益

| Preset | Seed43 ΔBA | Seed44 ΔBA | Mean ΔBA |
|---|---:|---:|---:|
| `layer3_patch2` | +0.086 | -0.057 | +0.014 |
| `layer4_pixel` | -0.001 | +0.026 | +0.012 |
| `layer5_pixel` | +0.119 | +0.135 | +0.127 |

## 5. Feature-domain frequency classification

仅使用固定的 9 个频谱摘要、标准化 logistic regression、5-fold subject-level CV。该结果衡量 source-train 与 target feature 的可分性，不使用诊断标签或 target 任务指标。下表为修正 `earliest_visit` 去重后结果；每个 checkpoint 都包含一个 subject 一次。

| Direction | Preset | Seed | Source N | Target N | Domain BA | Domain AUROC |
|---|---|---:|---:|---:|---:|---:|
| `NACC_to_ADNI` | `layer3_patch2` | 43 | 316 | 241 | 0.854 | 0.924 |
| `NACC_to_ADNI` | `layer3_patch2` | 44 | 316 | 241 | 0.745 | 0.807 |
| `NACC_to_ADNI` | `layer4_pixel` | 43 | 316 | 241 | 0.795 | 0.867 |
| `NACC_to_ADNI` | `layer4_pixel` | 44 | 316 | 241 | 0.876 | 0.952 |
| `NACC_to_ADNI` | `layer5_pixel` | 43 | 316 | 241 | 0.707 | 0.793 |
| `NACC_to_ADNI` | `layer5_pixel` | 44 | 316 | 241 | 0.857 | 0.930 |
| `ADNI_to_NACC` | `layer3_patch2` | 43 | 144 | 527 | 0.790 | 0.924 |
| `ADNI_to_NACC` | `layer3_patch2` | 44 | 144 | 527 | 0.770 | 0.918 |
| `ADNI_to_NACC` | `layer4_pixel` | 43 | 144 | 527 | 0.729 | 0.901 |
| `ADNI_to_NACC` | `layer4_pixel` | 44 | 144 | 527 | 0.718 | 0.863 |
| `ADNI_to_NACC` | `layer5_pixel` | 43 | 144 | 527 | 0.743 | 0.896 |
| `ADNI_to_NACC` | `layer5_pixel` | 44 | 144 | 527 | 0.807 | 0.932 |

按 seed 汇总，domain AUROC 介于 0.793–0.952（NACC→ADNI）和 0.863–0.932（ADNI→NACC），因此两种训练方向的 image-only 特征均保留显著的 cohort/domain 频谱信息。该现象并非由同一 subject 的重复扫描驱动。

## 6. 结论与边界

1. ADNI→NACC 的 `original_capm` 增益具有 preset 依赖：layer5 两个 seed 均为正（约 +0.12、+0.13），layer4 一正一近零，layer3 seed43 为正而 seed44 为负。因此不能称为跨 preset 普遍稳定增益。
2. ADNI→NACC 的 layer5 CAPM target BA 最高，但这只是两个 seed、一个方向下的结果；不能直接推导出 layer5 对所有方向最优。
3. 经 `earliest_visit` 去重后，双向 feature-domain AUROC 仍明显高于随机水平（0.793–0.952），说明 ResNet10 image-only 深层特征仍保留 cohort/domain 频谱信号；方向和 preset 影响很大。
4. 频域可分性与 target BA 不是简单一一对应关系。CAPM 是否降低域频谱差异，必须结合 CAPM 特征频谱进行下一步分析；当前报告只覆盖 image-only 特征，CAPM feature extraction 尚未完成。
5. 1.5T/3T 与 cohort、site、设备、构成和预处理残差混杂，因此结果支持域频谱关联，不证明磁场强度因果。

## 7. 审计输出

- ADNI→NACC source/target feature tables：`outputs/frequency_audit/resnet10_adni_to_nacc_features/`
- NACC→ADNI feature tables：`outputs/frequency_audit/resnet10_image_only_features/`
- 双向频域统计：`outputs/frequency_audit/resnet10_bidirectional_feature_frequency_statistics.json`
- 双向 domain summary：`outputs/frequency_audit/resnet10_bidirectional_feature_domain_summary.csv`
- ADNI→NACC frozen target：`outputs/resnet10_adni_to_nacc_seed43_44_validation/target_eval/`
