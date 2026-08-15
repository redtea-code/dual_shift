# ResNet10 双向 CAPM 与 Image-Only 特征频域对照（seed 43/44）

## 1. 范围与锁定规则

- Backbone：ResNet10（`layers=[1,1,1,1]`）；variant：`image_only` 与 `original_capm`；presets：`layer3_patch2`、`layer4_pixel`、`layer5_pixel`；seeds：43、44。
- 两个方向均使用已冻结 checkpoint：NACC→ADNI 与 ADNI→NACC。source-validation BA 加 collapse guard 选择 checkpoint；target 指标没有参与 epoch、variant、preset、阈值或配置选择。
- 特征从 preset 指定的 backbone layer 提取（layer3/layer4/layer5）；对每个 feature channel 进行 3D FFT，并平均其频率 bin power。domain classifier 只使用九个预设频谱摘要，采用标准化 logistic regression、5-fold subject-level CV。
- subject 范围为 task-filtered source-train 和完整 frozen target；各 cohort 内按 `subject_id`、最早有效 `scan_date`、路径进行 `earliest_visit` 去重。ADNI→NACC 每项为 144 source + 527 target subjects；NACC→ADNI 为 316 source + 241 target subjects。
- 完整性：12/12 CAPM checkpoints、7,368 条 CAPM subject-feature rows；无失败 batch、cohort-role 内 subject 唯一、数值有限，且低/中/高频比例之和为 1。

## 2. 冻结 target 泛化与 domain-frequency 配对

| Direction | Preset | Seed | Image BA [95% CI] | CAPM BA [95% CI] | Δ Target BA | Image Domain BA/AUROC | CAPM Domain BA/AUROC | Δ Domain BA | Δ Domain AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `NACC_to_ADNI` | `layer3_patch2` | 43 | 0.550 [0.509, 0.597] | 0.659 [0.601, 0.716] | +0.109 | 0.854/0.924 | 0.884/0.944 | +0.029 | +0.020 |
| `NACC_to_ADNI` | `layer3_patch2` | 44 | 0.680 [0.611, 0.750] | 0.638 [0.581, 0.699] | -0.042 | 0.745/0.807 | 0.853/0.930 | +0.108 | +0.123 |
| `NACC_to_ADNI` | `layer4_pixel` | 43 | 0.605 [0.551, 0.660] | 0.646 [0.581, 0.704] | +0.041 | 0.795/0.867 | 0.762/0.833 | -0.033 | -0.035 |
| `NACC_to_ADNI` | `layer4_pixel` | 44 | 0.660 [0.599, 0.724] | 0.679 [0.624, 0.734] | +0.019 | 0.876/0.952 | 0.795/0.853 | -0.081 | -0.099 |
| `NACC_to_ADNI` | `layer5_pixel` | 43 | 0.566 [0.530, 0.615] | 0.604 [0.549, 0.652] | +0.038 | 0.707/0.793 | 0.720/0.804 | +0.013 | +0.011 |
| `NACC_to_ADNI` | `layer5_pixel` | 44 | 0.614 [0.565, 0.679] | 0.634 [0.576, 0.687] | +0.020 | 0.857/0.930 | 0.652/0.729 | -0.205 | -0.200 |
| `ADNI_to_NACC` | `layer3_patch2` | 43 | 0.519 [0.503, 0.534] | 0.604 [0.570, 0.644] | +0.086 | 0.790/0.924 | 0.821/0.942 | +0.032 | +0.018 |
| `ADNI_to_NACC` | `layer3_patch2` | 44 | 0.623 [0.591, 0.657] | 0.566 [0.540, 0.594] | -0.057 | 0.770/0.918 | 0.804/0.943 | +0.034 | +0.024 |
| `ADNI_to_NACC` | `layer4_pixel` | 43 | 0.645 [0.603, 0.686] | 0.643 [0.605, 0.684] | -0.001 | 0.729/0.901 | 0.648/0.817 | -0.080 | -0.083 |
| `ADNI_to_NACC` | `layer4_pixel` | 44 | 0.580 [0.553, 0.607] | 0.606 [0.579, 0.633] | +0.026 | 0.718/0.863 | 0.734/0.879 | +0.015 | +0.015 |
| `ADNI_to_NACC` | `layer5_pixel` | 43 | 0.554 [0.529, 0.581] | 0.673 [0.632, 0.708] | +0.119 | 0.743/0.896 | 0.737/0.895 | -0.005 | -0.001 |
| `ADNI_to_NACC` | `layer5_pixel` | 44 | 0.527 [0.510, 0.550] | 0.662 [0.622, 0.699] | +0.135 | 0.807/0.932 | 0.827/0.937 | +0.020 | +0.005 |

## 3. 结果解读

1. CAPM 的 target BA 改变具有方向、preset 与 seed 依赖，不能概括为普遍增益。ADNI→NACC 的 layer5 两个 seed 均提高 target BA（+0.119、+0.135）；NACC→ADNI 的 layer4 两个 seed 均提高 target BA（+0.041、+0.019）。
2. “CAPM 降低 cohort/domain 频谱可分性”同样不具有普遍性：NACC→ADNI layer4 的两个 seed 均降低 domain BA/AUROC（BA −0.033、−0.081；AUROC −0.035、−0.099），但 NACC→ADNI layer3 两个 seed 均提高可分性；ADNI→NACC layer3 也在两个 seed 均提高可分性。
3. 在全部 12 个配对中，Δ target BA 与 Δ domain BA 的 Pearson r=+0.034，和 Δ domain AUROC 的 r=−0.007。这是描述性关联（n=12），接近零，不能支持“减少频率 domain separation 导致 target BA 增益”的机制性结论。
4. 可分性变化和泛化变化的局部一致性并不稳定。例如 NACC→ADNI layer4 的两个 seed 同时出现 target BA 上升与 domain separation 下降；ADNI→NACC layer5 则 target BA 均上升而 domain separation 在两个 seed 间方向不同。

## 4. 结论与下一步

- 当前证据支持：ResNet10 深层 image-only 和 CAPM 表征都保留可被频谱摘要识别的 cohort/domain 差异；CAPM 可在个别 preset/direction 重新塑造这种差异。
- 当前证据不支持：CAPM 一致消除 domain-frequency shift，或这种消除是其泛化增益的唯一解释。
- 下一步应预注册一个更窄的验证：以 NACC→ADNI `layer4_pixel` 和 ADNI→NACC `layer5_pixel` 为独立候选，在新增 seed 上同时检验 target BA 与 CAPM-image domain separation；报告全体新增 seed，不据结果进一步选择 preset。
- 磁场强度仍和 cohort、site、manufacturer、队列构成及预处理残余混杂；上述结果只表明 cohort/domain 表征频谱关联，不能证明 1.5T/3T 的因果作用。

## 5. 审计输出

- 汇总统计：`outputs/frequency_audit/resnet10_bidirectional_image_capm_feature_frequency_statistics.json`
- Domain 汇总：`outputs/frequency_audit/resnet10_bidirectional_image_capm_domain_summary.csv`
- CAPM-image domain 差异：`outputs/frequency_audit/resnet10_bidirectional_capm_minus_image_domain_delta.csv`
- Target-frequency 配对表：`outputs/frequency_audit/resnet10_bidirectional_capm_image_target_frequency_pairing.csv`
- CAPM 特征表：`outputs/frequency_audit/resnet10_capm_features/`
