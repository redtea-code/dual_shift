# Plan 34 E2 阶段性实验结果表（MCI vs AD）

- 更新：2026-08-11 UTC+8
- 执行计划：`docs/plans/DS-034.md`
- 协议：`scan_filtered_v1_2026-08-08`（ADNI 1.5T scan-filtered；NACC 3T）
- 代码冻结：`plan34-scan-filtered-v2+local-selector-fix`；commit `a07c882f4ad3c7aeb907438a0dd64d1c5c178249`
- 任务：MCI vs AD；E2 seed=42；source split seed=42；训练上限 50 epoch。
- E2 选模规则：仅按 source validation balanced accuracy（BA）并应用 collapse guard 选择 checkpoint。**本文没有 target/external 指标，所有性能数值均不是外部泛化结果。**

## 1. 完成状态

| 尺度 preset | ADNI → NACC | NACC → ADNI | 状态 |
|---|---:|---:|---|
| `layer3_patch2` | 6/6 | 6/6 | 完成 |
| `layer4_pixel` | 6/6 | 6/6 | 完成 |
| `layer5_pixel` | 0/6 | 0/6 | 未启动 |

已完成的 24 个最佳 checkpoint 均通过 collapse guard（选中 checkpoint 的 sensitivity 与 specificity 均 ≥ 0.15）。因此 `layer3_patch2` 和 `layer4_pixel` 的 source-only 矩阵已完整，但全 E2 仍未完成，不能冻结 E3 的唯一候选。

## 2. `layer3_patch2` 结果

### ADNI → NACC（source=ADNI）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 24 | 0.594 | 0.695 | 0.438 | 0.750 | 0.595 | 0.247 | 0.000 |
| capm | 14 | 0.641 | 0.693 | 0.563 | 0.719 | 0.636 | 0.262 | +0.047 |
| conv_gate | 41 | 0.625 | 0.729 | 0.563 | 0.688 | 0.618 | 0.255 | +0.031 |
| original_capm | 18 | **0.672** | **0.734** | 0.563 | **0.781** | **0.672** | 0.232 | **+0.078** |
| transformer_self | 13 | 0.578 | 0.693 | 0.375 | **0.781** | 0.580 | **0.221** | -0.016 |
| transformer_cross | 25 | **0.672** | 0.715 | **0.688** | 0.656 | 0.652 | 0.272 | **+0.078** |

### NACC → ADNI（source=NACC）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 24 | 0.700 | 0.830 | 0.457 | **0.943** | 0.717 | 0.180 | 0.000 |
| capm | 6 | **0.779** | 0.844 | **0.857** | 0.700 | **0.744** | **0.176** | **+0.079** |
| conv_gate | 30 | 0.721 | 0.823 | 0.629 | 0.814 | 0.721 | 0.195 | +0.021 |
| original_capm | 29 | 0.757 | **0.848** | 0.771 | 0.743 | 0.738 | 0.192 | +0.057 |
| transformer_self | 38 | 0.714 | 0.833 | 0.543 | 0.886 | 0.725 | 0.179 | +0.014 |
| transformer_cross | 40 | 0.750 | 0.828 | 0.714 | 0.786 | 0.741 | 0.200 | +0.050 |

### 跨方向摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向简单均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.594 | 0.700 | 0.647 | 0.000 |
| capm | 0.641 | **0.779** | 0.710 | +0.063 |
| conv_gate | 0.625 | 0.721 | 0.673 | +0.026 |
| original_capm | **0.672** | 0.757 | **0.715** | **+0.068** |
| transformer_self | 0.578 | 0.714 | 0.646 | -0.001 |
| transformer_cross | **0.672** | 0.750 | 0.711 | +0.064 |

## 3. `layer4_pixel` 结果

### ADNI → NACC（source=ADNI）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 12 | 0.594 | **0.730** | 0.750 | 0.438 | 0.541 | 0.269 | 0.000 |
| capm | 6 | 0.656 | 0.707 | **0.813** | 0.500 | 0.603 | 0.256 | +0.063 |
| conv_gate | 43 | **0.688** | 0.705 | 0.750 | 0.625 | **0.657** | 0.285 | **+0.094** |
| original_capm | 17 | 0.625 | 0.672 | 0.500 | **0.750** | 0.625 | 0.238 | +0.031 |
| transformer_self | 14 | 0.672 | 0.727 | **0.813** | 0.531 | 0.622 | 0.313 | +0.078 |
| transformer_cross | 6 | 0.625 | 0.695 | 0.563 | 0.688 | 0.618 | **0.214** | +0.031 |

### NACC → ADNI（source=NACC）

| 变体 | 最佳 epoch | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 15 | **0.786** | 0.838 | 0.714 | 0.857 | **0.786** | 0.167 | 0.000 |
| capm | 25 | 0.771 | 0.842 | 0.714 | 0.829 | 0.768 | 0.184 | -0.014 |
| conv_gate | 44 | 0.757 | 0.818 | **0.743** | 0.771 | 0.744 | 0.211 | -0.029 |
| original_capm | 22 | 0.729 | **0.870** | 0.514 | **0.943** | 0.747 | **0.160** | -0.057 |
| transformer_self | 41 | 0.721 | 0.850 | 0.600 | 0.843 | 0.726 | 0.190 | -0.064 |
| transformer_cross | 44 | 0.736 | 0.818 | 0.657 | 0.814 | 0.734 | 0.202 | -0.050 |

### 跨方向摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向简单均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.594 | **0.786** | 0.690 | 0.000 |
| capm | 0.656 | 0.771 | **0.714** | +0.024 |
| conv_gate | **0.688** | 0.757 | 0.722 | +0.032 |
| original_capm | 0.625 | 0.729 | 0.677 | -0.013 |
| transformer_self | 0.672 | 0.721 | 0.697 | +0.007 |
| transformer_cross | 0.625 | 0.736 | 0.680 | -0.010 |

## 4. 结构比较与当前解释

| 比较 | `layer3_patch2`：ADNI → NACC / NACC → ADNI | `layer4_pixel`：ADNI → NACC / NACC → ADNI | 解读 |
|---|---:|---:|---|
| 最佳 BA 变体 | original_capm / transformer_cross（0.672）；capm（0.779） | conv_gate（0.688）；image_only（0.786） | 最优结构随 source cohort 与尺度改变，尚无跨方向、跨尺度的唯一胜者。 |
| cross − self BA | +0.094 / +0.036 | -0.047 / +0.014 | cross interaction 仅在 `layer3_patch2` 两方向稳定优于 self；在 `layer4_pixel` 不具有一致优势。 |
| 相对 image-only 的最佳提升 | +0.078 / +0.079 | +0.094 / 0.000 | `layer4_pixel` 的方向结果发生分歧：ADNI 源队列中 conv_gate 获益，NACC 源队列中 image-only 最优。 |

1. `layer3_patch2`：`original_capm`、`transformer_cross` 与 `capm` 的两方向均值领先（0.715、0.711、0.710），但单一 seed 的 source validation 不足以确认唯一候选。
2. `layer4_pixel`：`conv_gate` 的两方向 BA 均值最高（0.722），但其 NACC → ADNI BA 仍低于 image-only；它不是可直接冻结的方向一致胜者。
3. 所有变体在 NACC → ADNI 的 source validation BA 普遍高于 ADNI → NACC。因为 source cohort、样本规模和验证难度不同，不能将其解释为 target 方向的外部泛化优劣。
4. 低 Brier 或高 AUROC 不自动等价于最高 BA。例如 `layer4_pixel` 的 NACC → ADNI `original_capm` 有最高 AUROC（0.870）及最低 Brier（0.160），但 BA=0.729，主要由较低 sensitivity（0.514）限制。

## 5. 边界、下一步与可解释范围

- `layer5_pixel` 的 12 个 E2 运行尚未开始。因此不能执行跨尺度最终排序，不能冻结 E3 唯一候选，也不能报告 E3 外部 target 性能。
- 本阶段所有 checkpoint 仅通过 source validation BA 与 collapse guard 选取；没有 target 指标参与训练、early stopping、checkpoint、阈值或结构选择。
- E3 前须完成 `layer5_pixel` 的两个方向六变体矩阵，再按预先定义的 source-only 规则冻结每个 task 的唯一候选。随后才可在 seeds 42/43 的两个方向上运行确认性 `external target + subject_mean + balanced accuracy` 评估。
- 当前结果仅可描述为：固定 scan-filtered 协议下，各候选结构在 source-validation 上的预测差异。不得据此主张场强因果效应、超分辨率能力或普适域泛化。

## 6. 数据来源与可审计性

- 结果文件：`outputs/e2_plan34/{layer3_patch2,layer4_pixel}/{direction}_seed42/{variant}/journal_metrics.json`
- Resolved config / provenance：`resolved_e2/mci_ad_{preset}_seed42.yaml` 及同名 `.provenance.json`
- E2 规范：`docs/plans/DS-034.md`
- `layer3_patch2` 与 `layer4_pixel` 各自使用独立 resolved config hash；同一 preset 的两个方向和全部变体使用相同 hash。报告中的每行数值直接读取对应 `journal_metrics.json` 的 `best_checkpoint_selection.collapse_guard.metrics`。
