# Plan 34 E2 完整阶段结果（MCI vs AD）

- 更新：2026-08-11 UTC+8
- 执行计划：`docs/plans/DS-034.md`
- 协议：`scan_filtered_v1_2026-08-08`（ADNI 1.5T scan-filtered；NACC 3T）
- 冻结代码：`plan34-scan-filtered-v2+local-selector-fix`；commit `a07c882f4ad3c7aeb907438a0dd64d1c5c178249`
- 任务：MCI vs AD；seed=42；source split seed=42；训练上限 50 epoch。
- 选模：仅使用 source-validation BA，并应用 collapse guard。本文所有指标均来自 source validation，**不包含 target/external 性能**。

## 1. 完成状态

| preset | ADNI → NACC | NACC → ADNI | 可审计报告 | 状态 |
|---|---:|---:|---:|---|
| `layer3_patch2` | 6/6 | 6/6 | 12/12 | 完成 |
| `layer4_pixel` | 6/6 | 6/6 | 12/12 | 完成 |
| `layer5_pixel` | 5/6 | 6/6 | 11/12 | **缺 1 项** |
| **合计** | **17/18** | **18/18** | **35/36** | **E2 尚未完全闭合** |

缺失项：`layer5_pixel / ADNI → NACC / transformer_self`。其日志曾达到 `epoch 50/50`，但没有生成 `journal_metrics.json`，所以不将该运行当作正式可审计指标。

所有已落盘的 35 个最佳 checkpoint 均通过 collapse guard；这只说明 source-validation checkpoint 未触发最低类别召回阈值，不代表外部泛化成立。

## 2.1 `layer3_patch2` 详细结果
### ADNI → NACC（source=ADNI）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_only | 24 | 0.594 | 0.695 | 0.438 | 0.750 | 0.595 | 0.247 | +0.000 | 通过 |
| capm | 14 | 0.641 | 0.693 | 0.562 | 0.719 | 0.636 | 0.262 | +0.047 | 通过 |
| conv_gate | 41 | 0.625 | 0.729 | 0.562 | 0.688 | 0.618 | 0.255 | +0.031 | 通过 |
| original_capm | 18 | 0.672 | 0.734 | 0.562 | 0.781 | 0.672 | 0.232 | +0.078 | 通过 |
| transformer_self | 13 | 0.578 | 0.693 | 0.375 | 0.781 | 0.580 | 0.221 | -0.016 | 通过 |
| transformer_cross | 25 | 0.672 | 0.715 | 0.688 | 0.656 | 0.652 | 0.272 | +0.078 | 通过 |
### NACC → ADNI（source=NACC）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_only | 24 | 0.700 | 0.830 | 0.457 | 0.943 | 0.717 | 0.180 | +0.000 | 通过 |
| capm | 6 | 0.779 | 0.844 | 0.857 | 0.700 | 0.744 | 0.176 | +0.079 | 通过 |
| conv_gate | 30 | 0.721 | 0.823 | 0.629 | 0.814 | 0.721 | 0.195 | +0.021 | 通过 |
| original_capm | 29 | 0.757 | 0.848 | 0.771 | 0.743 | 0.738 | 0.192 | +0.057 | 通过 |
| transformer_self | 38 | 0.714 | 0.833 | 0.543 | 0.886 | 0.725 | 0.179 | +0.014 | 通过 |
| transformer_cross | 40 | 0.750 | 0.828 | 0.714 | 0.786 | 0.741 | 0.200 | +0.050 | 通过 |

### `layer3_patch2` 跨方向 BA 摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.594 | 0.700 | 0.647 | 0.000 |
| capm | 0.641 | 0.779 | 0.710 | 0.063 |
| conv_gate | 0.625 | 0.721 | 0.673 | 0.026 |
| original_capm | 0.672 | 0.757 | 0.715 | 0.068 |
| transformer_self | 0.578 | 0.714 | 0.646 | -0.001 |
| transformer_cross | 0.672 | 0.750 | 0.711 | 0.064 |

## 2.2 `layer4_pixel` 详细结果
### ADNI → NACC（source=ADNI）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_only | 12 | 0.594 | 0.730 | 0.750 | 0.438 | 0.541 | 0.269 | +0.000 | 通过 |
| capm | 6 | 0.656 | 0.707 | 0.812 | 0.500 | 0.603 | 0.256 | +0.062 | 通过 |
| conv_gate | 43 | 0.688 | 0.705 | 0.750 | 0.625 | 0.657 | 0.285 | +0.094 | 通过 |
| original_capm | 17 | 0.625 | 0.672 | 0.500 | 0.750 | 0.625 | 0.238 | +0.031 | 通过 |
| transformer_self | 14 | 0.672 | 0.727 | 0.812 | 0.531 | 0.622 | 0.313 | +0.078 | 通过 |
| transformer_cross | 6 | 0.625 | 0.695 | 0.562 | 0.688 | 0.618 | 0.214 | +0.031 | 通过 |
### NACC → ADNI（source=NACC）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_only | 15 | 0.786 | 0.838 | 0.714 | 0.857 | 0.786 | 0.167 | +0.000 | 通过 |
| capm | 25 | 0.771 | 0.842 | 0.714 | 0.829 | 0.768 | 0.184 | -0.014 | 通过 |
| conv_gate | 44 | 0.757 | 0.818 | 0.743 | 0.771 | 0.744 | 0.211 | -0.029 | 通过 |
| original_capm | 22 | 0.729 | 0.870 | 0.514 | 0.943 | 0.747 | 0.160 | -0.057 | 通过 |
| transformer_self | 41 | 0.721 | 0.850 | 0.600 | 0.843 | 0.726 | 0.190 | -0.064 | 通过 |
| transformer_cross | 44 | 0.736 | 0.818 | 0.657 | 0.814 | 0.734 | 0.202 | -0.050 | 通过 |

### `layer4_pixel` 跨方向 BA 摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.594 | 0.786 | 0.690 | 0.000 |
| capm | 0.656 | 0.771 | 0.714 | 0.024 |
| conv_gate | 0.688 | 0.757 | 0.722 | 0.033 |
| original_capm | 0.625 | 0.729 | 0.677 | -0.013 |
| transformer_self | 0.672 | 0.721 | 0.697 | 0.007 |
| transformer_cross | 0.625 | 0.736 | 0.680 | -0.009 |

## 2.3 `layer5_pixel` 详细结果
### ADNI → NACC（source=ADNI）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_only | 14 | 0.672 | 0.740 | 0.688 | 0.656 | 0.652 | 0.251 | +0.000 | 通过 |
| capm | 28 | 0.672 | 0.748 | 0.750 | 0.594 | 0.638 | 0.279 | +0.000 | 通过 |
| conv_gate | 47 | 0.641 | 0.729 | 0.625 | 0.656 | 0.626 | 0.260 | -0.031 | 通过 |
| original_capm | 2 | 0.672 | 0.756 | 0.438 | 0.906 | 0.684 | 0.182 | +0.000 | 通过 |
| transformer_self | — | — | — | — | — | — | — | 未落盘 |
| transformer_cross | 13 | 0.656 | 0.768 | 0.438 | 0.875 | 0.665 | 0.185 | -0.016 | 通过 |
### NACC → ADNI（source=NACC）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA | Guard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| image_only | 11 | 0.757 | 0.836 | 0.571 | 0.943 | 0.776 | 0.152 | +0.000 | 通过 |
| capm | 24 | 0.750 | 0.847 | 0.657 | 0.843 | 0.752 | 0.161 | -0.007 | 通过 |
| conv_gate | 41 | 0.736 | 0.836 | 0.600 | 0.871 | 0.744 | 0.189 | -0.021 | 通过 |
| original_capm | 16 | 0.771 | 0.838 | 0.657 | 0.886 | 0.779 | 0.145 | +0.014 | 通过 |
| transformer_self | 11 | 0.714 | 0.837 | 0.457 | 0.971 | 0.735 | 0.159 | -0.043 | 通过 |
| transformer_cross | 1 | 0.736 | 0.827 | 0.829 | 0.643 | 0.698 | 0.186 | -0.021 | 通过 |

### `layer5_pixel` 跨方向 BA 摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.672 | 0.757 | 0.715 | 0.000 |
| capm | 0.672 | 0.750 | 0.711 | -0.004 |
| conv_gate | 0.641 | 0.736 | 0.688 | -0.026 |
| original_capm | 0.672 | 0.771 | 0.722 | 0.007 |
| transformer_self | — | — | — | — | 数据不完整 |
| transformer_cross | 0.656 | 0.736 | 0.696 | -0.019 |

## 3. 跨尺度、跨方向分析

### 3.1 按 E2 规则选出的跨 preset 结果（source-validation）

以下每行先在对应 preset/方向内按 **BA + collapse guard** 选出 checkpoint，再同时报告该 checkpoint 的其他指标。AUROC、Sensitivity、Specificity、Macro-F1 和 Brier 仅用于描述，不参与 E2 checkpoint 选择。

| preset | 方向 | BA 选中变体（epoch） | BA | AUROC | Sensitivity | Specificity | Macro-F1 | Brier |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` (epoch 25) | 0.672 | 0.715 | 0.688 | 0.656 | 0.652 | 0.272 |
| `layer3_patch2` | NACC → ADNI | `capm` (epoch 6) | 0.779 | 0.844 | 0.857 | 0.700 | 0.744 | 0.176 |
| `layer4_pixel` | ADNI → NACC | `conv_gate` (epoch 43) | 0.688 | 0.705 | 0.750 | 0.625 | 0.657 | 0.285 |
| `layer4_pixel` | NACC → ADNI | `image_only` (epoch 15) | 0.786 | 0.838 | 0.714 | 0.857 | 0.786 | 0.167 |
| `layer5_pixel` | ADNI → NACC | `original_capm` (epoch 2) | 0.672 | 0.756 | 0.438 | 0.906 | 0.684 | 0.182 |
| `layer5_pixel` | NACC → ADNI | `original_capm` (epoch 16) | 0.771 | 0.838 | 0.657 | 0.886 | 0.779 | 0.145 |

### 3.2 各指标在跨 preset 对比中的最高值（描述性）

下表分别对每个 preset/方向寻找各指标的最高值；Brier 是越低越好。该表用于描述指标间排序差异，**不改变 E2 按 BA 选模的结果**。

#### BA 最高值

| preset | 方向 | 变体 | BA |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.672 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.779 |
| `layer4_pixel` | ADNI → NACC | `conv_gate` | 0.688 |
| `layer4_pixel` | NACC → ADNI | `image_only` | 0.786 |
| `layer5_pixel` | ADNI → NACC | `original_capm` | 0.672 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.771 |

#### AUROC 最高值

| preset | 方向 | 变体 | AUROC |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `original_capm` | 0.734 |
| `layer3_patch2` | NACC → ADNI | `original_capm` | 0.848 |
| `layer4_pixel` | ADNI → NACC | `image_only` | 0.730 |
| `layer4_pixel` | NACC → ADNI | `original_capm` | 0.870 |
| `layer5_pixel` | ADNI → NACC | `transformer_cross` | 0.768 |
| `layer5_pixel` | NACC → ADNI | `capm` | 0.847 |

#### Sensitivity 最高值

| preset | 方向 | 变体 | Sensitivity |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.688 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.857 |
| `layer4_pixel` | ADNI → NACC | `transformer_self` | 0.812 |
| `layer4_pixel` | NACC → ADNI | `conv_gate` | 0.743 |
| `layer5_pixel` | ADNI → NACC | `capm` | 0.750 |
| `layer5_pixel` | NACC → ADNI | `transformer_cross` | 0.829 |

#### Specificity 最高值

| preset | 方向 | 变体 | Specificity |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_self` | 0.781 |
| `layer3_patch2` | NACC → ADNI | `image_only` | 0.943 |
| `layer4_pixel` | ADNI → NACC | `original_capm` | 0.750 |
| `layer4_pixel` | NACC → ADNI | `original_capm` | 0.943 |
| `layer5_pixel` | ADNI → NACC | `original_capm` | 0.906 |
| `layer5_pixel` | NACC → ADNI | `transformer_self` | 0.971 |

#### Macro-F1 最高值

| preset | 方向 | 变体 | Macro-F1 |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `original_capm` | 0.672 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.744 |
| `layer4_pixel` | ADNI → NACC | `conv_gate` | 0.657 |
| `layer4_pixel` | NACC → ADNI | `image_only` | 0.786 |
| `layer5_pixel` | ADNI → NACC | `original_capm` | 0.684 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.779 |

#### Brier 最低值

| preset | 方向 | 变体 | Brier |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_self` | 0.221 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.176 |
| `layer4_pixel` | ADNI → NACC | `transformer_cross` | 0.214 |
| `layer4_pixel` | NACC → ADNI | `original_capm` | 0.160 |
| `layer5_pixel` | ADNI → NACC | `original_capm` | 0.182 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.145 |

### 3.3 主要观察

1. **没有稳定的跨方向、跨尺度唯一胜者。** `layer3_patch2` 的两方向均值最高是 `original_capm`（0.715）；`layer4_pixel` 是 `conv_gate`（0.722）；`layer5_pixel` 在当前已落盘结果中 `original_capm` 的两方向均值最高（0.721），但 ADNI→NACC 的 `transformer_self` 缺失，因此不能据此完成最终全矩阵排序。
2. **source cohort 影响明显。** 三种 preset 中，NACC→ADNI 的 source-validation BA 普遍高于 ADNI→NACC；该现象不能解释为 target 方向的外部泛化优劣，因为两方向的 source cohort、样本量和验证难度不同。
3. **cross 相对 self 没有一致优势。** `layer3_patch2` 中 cross 两方向均高于 self；`layer4_pixel` 中差异为 ADNI→NACC `-0.047`、NACC→ADNI `+0.014`；`layer5_pixel` 的 ADNI→NACC self 缺失，NACC→ADNI cross=0.736，高于 self=0.714，尚不足以支持跨尺度结论。
4. **BA、AUROC 与校准指标可能给出不同排序。** 例如 `layer4_pixel / NACC→ADNI / original_capm` 的 AUROC=0.870、Brier=0.160，但 BA=0.729，主要受 sensitivity=0.514 限制；因此 E2 按计划以 BA+collapse guard 选模，不能用单一 AUROC 或 Brier 替代。
5. **collapse guard 结果。** 35 个已落盘 checkpoint 全部通过 guard；缺失项没有可审计 checkpoint 结果，不能推断其是否通过。

## 4. E2 结论与后续动作

- E2 目前为 **35/36 个结果可审计**，尚不能宣布三尺度矩阵完全闭合。
- 应优先补跑或恢复 `layer5_pixel / ADNI→NACC / transformer_self`，并确认生成 `journal_metrics.json`、`best_checkpoint.pt`、`last_checkpoint.pt` 及最终 `completed` 标记。
- 在该缺失项补齐前，不应冻结 E3 的唯一候选。补齐后，按预先定义的 source-only BA + collapse guard 规则完成候选冻结，再进行 seeds 42/43、两个方向的 E3 `external target + subject_mean + balanced accuracy` 评估。
- 当前结果只能支持“固定 scan-filtered 协议下的 source-validation 预测差异”，不能支持场强因果效应、超分辨率能力或普适域泛化结论。

## 5. 可审计来源

- 指标：`outputs/e2_plan34/{layer3_patch2,layer4_pixel,layer5_pixel}/{direction}_seed42/{variant}/journal_metrics.json`
- 日志：对应方向目录下的 `train.log`、`resume.log` 或 `final_transformer_cross.log`
- 配置与 provenance：`resolved_e2/mci_ad_{preset}_seed42.yaml` 及同名 `.provenance.json`
- 规范：`docs/plans/DS-034.md`
- 表中数值直接来自各报告的 `best_checkpoint_selection.collapse_guard.metrics`；缺失项不作数值填充。
