# Plan 34 Target 盲测结果表（MCI vs AD）

- 更新：2026-08-11 UTC+8
- 输入 checkpoint：E2 source-validation BA + collapse guard 冻结的 `best_checkpoint.pt`
- 评估方式：冻结 checkpoint 后只读 target 推理；不重新训练、不调阈值、不用 target 选择 checkpoint 或 preset。
- 协议：`scan_filtered_v1_2026-08-08`；ADNI 1.5T scan-filtered；NACC 3T。
- 任务：MCI vs AD；seed=42；target 指标为 subject-mean 聚合。
- 输出来源：`outputs/e3_target_plan34/{preset}/{direction}_seed42/{variant}/target_metrics.json`

## 1. 完成状态

| preset | ADNI → NACC | NACC → ADNI | target 报告 | 状态 |
|---|---:|---:|---:|---|
| `layer3_patch2` | 6/6 | 6/6 | 12/12 | 完成 |
| `layer4_pixel` | 6/6 | 6/6 | 12/12 | 完成 |
| `layer5_pixel` | 6/6 | 6/6 | 12/12 | 完成 |
| **合计** | **18/18** | **18/18** | **36/36** | **完成** |

## 2. Target 结果总览

### 2.1 各 preset 的平均 target 指标（36 个组合按 preset 分组）

| preset | 平均 BA | 平均 AUROC | 平均 Macro-F1 | 平均 Brier | 平均 ECE |
|---|---:|---:|---:|---:|---:|
| `layer3_patch2` | 0.614 | 0.719 | 0.608 | 0.238 | 0.192 |
| `layer4_pixel` | 0.643 | 0.723 | 0.644 | 0.223 | 0.153 |
| `layer5_pixel` | 0.610 | 0.741 | 0.598 | 0.226 | 0.180 |

### 2.2 每个 preset/方向内的 target BA 最优变体

| preset | 方向 | Target BA 最优变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.649 | 0.732 | 0.457 | 0.841 | 0.656 | 0.224 | 0.175 |
| `layer3_patch2` | NACC → ADNI | `original_capm` | 0.651 | 0.696 | 0.558 | 0.744 | 0.647 | 0.247 | 0.174 |
| `layer4_pixel` | ADNI → NACC | `transformer_self` | 0.692 | 0.759 | 0.566 | 0.818 | 0.695 | 0.199 | 0.133 |
| `layer4_pixel` | NACC → ADNI | `transformer_cross` | 0.663 | 0.698 | 0.558 | 0.768 | 0.661 | 0.249 | 0.205 |
| `layer5_pixel` | ADNI → NACC | `capm` | 0.658 | 0.754 | 0.480 | 0.835 | 0.664 | 0.229 | 0.205 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.677 | 0.741 | 0.494 | 0.860 | 0.685 | 0.208 | 0.155 |

## 3.1 `layer3_patch2` target 详细结果

### ADNI → NACC

| 变体 | E2 epoch | Target BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 24 | 0.620 | 0.719 | 0.309 | 0.932 | 0.623 | 0.227 | 0.185 | +0.000 |
| capm | 14 | 0.603 | 0.748 | 0.251 | 0.955 | 0.597 | 0.236 | 0.205 | -0.017 |
| conv_gate | 41 | 0.623 | 0.732 | 0.337 | 0.909 | 0.628 | 0.231 | 0.198 | +0.003 |
| original_capm | 18 | 0.590 | 0.779 | 0.194 | 0.986 | 0.572 | 0.243 | 0.226 | -0.030 |
| transformer_self | 13 | 0.563 | 0.740 | 0.149 | 0.977 | 0.532 | 0.232 | 0.202 | -0.057 |
| transformer_cross | 25 | 0.649 | 0.732 | 0.457 | 0.841 | 0.656 | 0.224 | 0.175 | +0.029 |

### NACC → ADNI

| 变体 | E2 epoch | Target BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 24 | 0.537 | 0.727 | 0.130 | 0.945 | 0.506 | 0.257 | 0.247 | +0.000 |
| capm | 6 | 0.631 | 0.678 | 0.584 | 0.677 | 0.619 | 0.228 | 0.093 | +0.093 |
| conv_gate | 30 | 0.638 | 0.705 | 0.442 | 0.835 | 0.645 | 0.242 | 0.199 | +0.101 |
| original_capm | 29 | 0.651 | 0.696 | 0.558 | 0.744 | 0.647 | 0.247 | 0.174 | +0.114 |
| transformer_self | 38 | 0.617 | 0.673 | 0.351 | 0.884 | 0.623 | 0.243 | 0.207 | +0.080 |
| transformer_cross | 40 | 0.650 | 0.697 | 0.519 | 0.780 | 0.650 | 0.244 | 0.186 | +0.112 |

### `layer3_patch2` 跨方向 target BA 摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.620 | 0.537 | 0.579 | +0.000 |
| capm | 0.603 | 0.631 | 0.617 | +0.038 |
| conv_gate | 0.623 | 0.638 | 0.631 | +0.052 |
| original_capm | 0.590 | 0.651 | 0.621 | +0.042 |
| transformer_self | 0.563 | 0.617 | 0.590 | +0.011 |
| transformer_cross | 0.649 | 0.650 | 0.650 | +0.071 |


## 3.2 `layer4_pixel` target 详细结果

### ADNI → NACC

| 变体 | E2 epoch | Target BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 12 | 0.659 | 0.732 | 0.566 | 0.753 | 0.657 | 0.203 | 0.062 | +0.000 |
| capm | 6 | 0.638 | 0.721 | 0.423 | 0.852 | 0.644 | 0.192 | 0.024 | -0.022 |
| conv_gate | 43 | 0.642 | 0.722 | 0.451 | 0.832 | 0.648 | 0.232 | 0.194 | -0.017 |
| original_capm | 17 | 0.632 | 0.773 | 0.303 | 0.960 | 0.635 | 0.204 | 0.156 | -0.028 |
| transformer_self | 14 | 0.692 | 0.759 | 0.566 | 0.818 | 0.695 | 0.199 | 0.133 | +0.033 |
| transformer_cross | 6 | 0.580 | 0.729 | 0.223 | 0.938 | 0.569 | 0.200 | 0.094 | -0.079 |

### NACC → ADNI

| 变体 | E2 epoch | Target BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 15 | 0.663 | 0.705 | 0.455 | 0.872 | 0.673 | 0.225 | 0.189 | +0.000 |
| capm | 25 | 0.646 | 0.722 | 0.481 | 0.811 | 0.650 | 0.218 | 0.144 | -0.018 |
| conv_gate | 44 | 0.639 | 0.690 | 0.571 | 0.707 | 0.631 | 0.271 | 0.228 | -0.024 |
| original_capm | 22 | 0.610 | 0.739 | 0.312 | 0.909 | 0.614 | 0.229 | 0.207 | -0.053 |
| transformer_self | 41 | 0.650 | 0.684 | 0.506 | 0.793 | 0.652 | 0.247 | 0.200 | -0.014 |
| transformer_cross | 44 | 0.663 | 0.698 | 0.558 | 0.768 | 0.661 | 0.249 | 0.205 | +0.000 |

### `layer4_pixel` 跨方向 target BA 摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.659 | 0.663 | 0.661 | +0.000 |
| capm | 0.638 | 0.646 | 0.642 | -0.020 |
| conv_gate | 0.642 | 0.639 | 0.641 | -0.021 |
| original_capm | 0.632 | 0.610 | 0.621 | -0.040 |
| transformer_self | 0.692 | 0.650 | 0.671 | +0.010 |
| transformer_cross | 0.580 | 0.663 | 0.622 | -0.039 |


## 3.3 `layer5_pixel` target 详细结果

### ADNI → NACC

| 变体 | E2 epoch | Target BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 14 | 0.637 | 0.775 | 0.349 | 0.926 | 0.644 | 0.202 | 0.151 | +0.000 |
| capm | 28 | 0.658 | 0.754 | 0.480 | 0.835 | 0.664 | 0.229 | 0.205 | +0.020 |
| conv_gate | 47 | 0.652 | 0.778 | 0.434 | 0.869 | 0.660 | 0.218 | 0.198 | +0.014 |
| original_capm | 2 | 0.544 | 0.674 | 0.120 | 0.969 | 0.504 | 0.214 | 0.099 | -0.093 |
| transformer_self | 13 | 0.544 | 0.772 | 0.103 | 0.986 | 0.496 | 0.241 | 0.220 | -0.093 |
| transformer_cross | 13 | 0.566 | 0.807 | 0.149 | 0.983 | 0.534 | 0.228 | 0.209 | -0.072 |

### NACC → ADNI

| 变体 | E2 epoch | Target BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | 相对 image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| image_only | 11 | 0.602 | 0.740 | 0.234 | 0.970 | 0.596 | 0.227 | 0.206 | +0.000 |
| capm | 24 | 0.665 | 0.735 | 0.506 | 0.823 | 0.670 | 0.227 | 0.189 | +0.063 |
| conv_gate | 41 | 0.627 | 0.722 | 0.351 | 0.902 | 0.634 | 0.241 | 0.219 | +0.025 |
| original_capm | 16 | 0.677 | 0.741 | 0.494 | 0.860 | 0.685 | 0.208 | 0.155 | +0.075 |
| transformer_self | 11 | 0.539 | 0.726 | 0.091 | 0.988 | 0.490 | 0.261 | 0.248 | -0.062 |
| transformer_cross | 1 | 0.607 | 0.672 | 0.519 | 0.695 | 0.601 | 0.216 | 0.059 | +0.006 |

### `layer5_pixel` 跨方向 target BA 摘要

| 变体 | ADNI → NACC BA | NACC → ADNI BA | 两方向均值 BA | 相对 image-only 均值 |
|---|---:|---:|---:|---:|
| image_only | 0.637 | 0.602 | 0.619 | +0.000 |
| capm | 0.658 | 0.665 | 0.661 | +0.042 |
| conv_gate | 0.652 | 0.627 | 0.639 | +0.020 |
| original_capm | 0.544 | 0.677 | 0.611 | -0.009 |
| transformer_self | 0.544 | 0.539 | 0.542 | -0.078 |
| transformer_cross | 0.566 | 0.607 | 0.587 | -0.033 |

## 4. 各指标最高值/最低值（描述性）

以下表格用于判断迁移表现的不同侧面。Brier 和 ECE 越低越好；其他指标越高越好。该表仍不改变“target 不参与调参”的原则。

### Target BA 最高值

| preset | 方向 | 变体 | BA |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.649 |
| `layer3_patch2` | NACC → ADNI | `original_capm` | 0.651 |
| `layer4_pixel` | ADNI → NACC | `transformer_self` | 0.692 |
| `layer4_pixel` | NACC → ADNI | `transformer_cross` | 0.663 |
| `layer5_pixel` | ADNI → NACC | `capm` | 0.658 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.677 |

### AUROC 最高值

| preset | 方向 | 变体 | AUROC |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `original_capm` | 0.779 |
| `layer3_patch2` | NACC → ADNI | `image_only` | 0.727 |
| `layer4_pixel` | ADNI → NACC | `original_capm` | 0.773 |
| `layer4_pixel` | NACC → ADNI | `original_capm` | 0.739 |
| `layer5_pixel` | ADNI → NACC | `transformer_cross` | 0.807 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.741 |

### Sensitivity 最高值

| preset | 方向 | 变体 | Sensitivity |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.457 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.584 |
| `layer4_pixel` | ADNI → NACC | `transformer_self` | 0.566 |
| `layer4_pixel` | NACC → ADNI | `conv_gate` | 0.571 |
| `layer5_pixel` | ADNI → NACC | `capm` | 0.480 |
| `layer5_pixel` | NACC → ADNI | `transformer_cross` | 0.519 |

### Specificity 最高值

| preset | 方向 | 变体 | Specificity |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `original_capm` | 0.986 |
| `layer3_patch2` | NACC → ADNI | `image_only` | 0.945 |
| `layer4_pixel` | ADNI → NACC | `original_capm` | 0.960 |
| `layer4_pixel` | NACC → ADNI | `original_capm` | 0.909 |
| `layer5_pixel` | ADNI → NACC | `transformer_self` | 0.986 |
| `layer5_pixel` | NACC → ADNI | `transformer_self` | 0.988 |

### Macro-F1 最高值

| preset | 方向 | 变体 | Macro-F1 |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.656 |
| `layer3_patch2` | NACC → ADNI | `transformer_cross` | 0.650 |
| `layer4_pixel` | ADNI → NACC | `transformer_self` | 0.695 |
| `layer4_pixel` | NACC → ADNI | `image_only` | 0.673 |
| `layer5_pixel` | ADNI → NACC | `capm` | 0.664 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.685 |

### Brier 最低值

| preset | 方向 | 变体 | Brier |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.224 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.228 |
| `layer4_pixel` | ADNI → NACC | `capm` | 0.192 |
| `layer4_pixel` | NACC → ADNI | `capm` | 0.218 |
| `layer5_pixel` | ADNI → NACC | `image_only` | 0.202 |
| `layer5_pixel` | NACC → ADNI | `original_capm` | 0.208 |

### ECE 最低值

| preset | 方向 | 变体 | ECE |
|---|---|---|---:|
| `layer3_patch2` | ADNI → NACC | `transformer_cross` | 0.175 |
| `layer3_patch2` | NACC → ADNI | `capm` | 0.093 |
| `layer4_pixel` | ADNI → NACC | `capm` | 0.024 |
| `layer4_pixel` | NACC → ADNI | `capm` | 0.144 |
| `layer5_pixel` | ADNI → NACC | `original_capm` | 0.099 |
| `layer5_pixel` | NACC → ADNI | `transformer_cross` | 0.059 |

## 5. 迁移能力初步结论

1. **按全部 12 个组合的平均 target BA，`layer4_pixel` 最高。** 这说明在当前冻结 checkpoint 的全面 target 盲测中，`layer4_pixel` 的整体迁移分类表现最稳；但平均 Brier/ECE 需要结合下表判断校准稳定性。
2. **`layer5_pixel` 的 source-side 高 specificity / F1 / Brier 优势并未自动转化为全面 target 优势。** 它在部分方向和变体上仍可达到较好 AUROC 或校准，但整体 target BA 不应仅凭 source 指标推断。
3. **target 表现存在明显方向性。** ADNI→NACC 与 NACC→ADNI 的最优变体并不完全一致，因此应保留方向分表，避免合并后隐藏某一方向失败。
4. **source 表现不佳的变体仍有必要被 target 盲测。** 本报告完整保留所有 preset/变体，能够观察 source 与 target 排序不一致的情况；这些结果只能用于迁移分析，不可回头调参。
5. **下一步若要正式 E3 确认，应预先冻结唯一候选和 seed 方案。** 当前报告是全面 target test，用于比较迁移能力；若进入正式确认性实验，应遵循 Plan 34 的 seeds 42/43 与预注册候选规则。

## 6. 可审计来源

- Evaluator：`experiments/evaluate_frozen_journal_target.py`
- Target 指标：`outputs/e3_target_plan34/{preset}/{direction}_seed42/{variant}/target_metrics.json`
- Target 预测：`outputs/e3_target_plan34/{preset}/{direction}_seed42/{variant}/target_predictions.csv`
- E2 checkpoint：各 `outputs/e2_plan34/.../{variant}/best_checkpoint.pt`；`layer5_pixel / ADNI→NACC / transformer_self` 使用补跑目录 `outputs/e2_plan34/layer5_pixel/ADNI_to_NACC_seed42_transformer_self_rerun/transformer_self/best_checkpoint.pt`
