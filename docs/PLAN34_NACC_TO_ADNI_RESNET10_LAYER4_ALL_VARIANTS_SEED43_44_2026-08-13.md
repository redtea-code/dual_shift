# Plan 34 NACC → ADNI：ResNet10 layer4_pixel 六变体消融结果

- 更新：2026-08-13 UTC+8
- 目的：在 ResNet10、`layer4_pixel` 条件下，完整比较 `image_only`、`capm`、`conv_gate`、`original_capm`、`transformer_self`、`transformer_cross` 六种变体。
- 方向：NACC（source）→ ADNI（target）；target 为 subject-mean 聚合，241 subjects。
- seeds：43、44；输入 `160×196×160`；backbone 深度 `layers=[1,1,1,1]`。
- 训练/选择：source-validation BA + collapse guard 冻结 checkpoint；12/12 checkpoint 均通过 guard。
- Target：冻结 checkpoint 后只读 ADNI 推理；不训练、不调阈值、不用 target 选择 checkpoint 或变体。
- 协议：`scan_filtered_v1_2026-08-08`；ADNI 1.5T scan-filtered，NACC 3T。

## 1. 完成状态

| seed | source 训练 | frozen target 评估 | 变体数 | 状态 |
|---|---:|---:|---:|---|
| 43 | 6/6 | 6/6 | 6 | 完成 |
| 44 | 6/6 | 6/6 | 6 | 完成 |
| **合计** | **12/12** | **12/12** | **6** | **完成** |

## 2. Target BA 总览（均值 ± SD）

| 变体 | Target BA | 相对 image_only BA | AUROC | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.632 ± 0.039 | +0.000 | 0.680 ± 0.032 | 0.633 ± 0.037 | 0.236 ± 0.014 | 0.175 ± 0.005 |
| `capm` | 0.629 ± 0.028 | -0.004 | 0.671 ± 0.004 | 0.626 ± 0.021 | 0.241 ± 0.010 | 0.178 ± 0.013 |
| `conv_gate` | 0.655 ± 0.010 | +0.022 | 0.696 ± 0.008 | 0.660 ± 0.007 | 0.229 ± 0.003 | 0.183 ± 0.010 |
| `original_capm` | 0.662 ± 0.023 | +0.030 | 0.711 ± 0.010 | 0.664 ± 0.021 | 0.221 ± 0.005 | 0.163 ± 0.014 |
| `transformer_self` | 0.656 ± 0.014 | +0.024 | 0.696 ± 0.002 | 0.657 ± 0.013 | 0.236 ± 0.001 | 0.179 ± 0.002 |
| `transformer_cross` | 0.645 ± 0.014 | +0.013 | 0.699 ± 0.012 | 0.650 ± 0.024 | 0.232 ± 0.026 | 0.183 ± 0.028 |

## 3. 六变体详细结果

Target BA 方括号为单次 target 评估的 200 次 subject bootstrap 95% CI；`Δ vs image-only` 为同 seed 的 BA 差。Brier/ECE 越低越好，其余指标越高越好。

### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 18 | 0.757 | 0.605 [0.551, 0.660] | 0.657 | 0.442 | 0.768 | 0.607 | 0.245 | 0.178 | +0.000 |
| `capm` | 33 | 0.764 | 0.649 [0.584, 0.705] | 0.673 | 0.571 | 0.726 | 0.642 | 0.248 | 0.169 | +0.044 |
| `conv_gate` | 36 | 0.779 | 0.662 [0.603, 0.719] | 0.701 | 0.519 | 0.805 | 0.665 | 0.227 | 0.176 | +0.057 |
| `original_capm` | 17 | 0.757 | 0.646 [0.581, 0.704] | 0.704 | 0.494 | 0.799 | 0.649 | 0.224 | 0.173 | +0.041 |
| `transformer_self` | 43 | 0.736 | 0.647 [0.583, 0.709] | 0.697 | 0.506 | 0.787 | 0.648 | 0.236 | 0.177 | +0.042 |
| `transformer_cross` | 25 | 0.736 | 0.655 [0.603, 0.711] | 0.708 | 0.390 | 0.921 | 0.667 | 0.214 | 0.163 | +0.050 |

### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 25 | 0.743 | 0.660 [0.599, 0.724] | 0.702 | 0.545 | 0.774 | 0.659 | 0.226 | 0.171 | +0.000 |
| `capm` | 25 | 0.714 | 0.609 [0.564, 0.662] | 0.668 | 0.286 | 0.933 | 0.611 | 0.234 | 0.187 | -0.051 |
| `conv_gate` | 47 | 0.729 | 0.648 [0.590, 0.717] | 0.690 | 0.442 | 0.854 | 0.655 | 0.231 | 0.191 | -0.012 |
| `original_capm` | 15 | 0.743 | 0.679 [0.624, 0.734] | 0.718 | 0.558 | 0.799 | 0.679 | 0.217 | 0.153 | +0.019 |
| `transformer_self` | 46 | 0.800 | 0.666 [0.608, 0.732] | 0.695 | 0.545 | 0.787 | 0.666 | 0.235 | 0.180 | +0.006 |
| `transformer_cross` | 41 | 0.729 | 0.635 [0.571, 0.693] | 0.690 | 0.519 | 0.750 | 0.633 | 0.251 | 0.203 | -0.025 |

### 跨 seed 指标均值 ± SD

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.632 ± 0.039 | 0.680 ± 0.032 | 0.494 ± 0.073 | 0.771 ± 0.004 | 0.633 ± 0.037 | 0.236 ± 0.014 | 0.175 ± 0.005 |
| `capm` | 0.629 ± 0.028 | 0.671 ± 0.004 | 0.429 ± 0.202 | 0.829 ± 0.147 | 0.626 ± 0.021 | 0.241 ± 0.010 | 0.178 ± 0.013 |
| `conv_gate` | 0.655 ± 0.010 | 0.696 ± 0.008 | 0.481 ± 0.055 | 0.829 ± 0.034 | 0.660 ± 0.007 | 0.229 ± 0.003 | 0.183 ± 0.010 |
| `original_capm` | 0.662 ± 0.023 | 0.711 ± 0.010 | 0.526 ± 0.046 | 0.799 ± 0.000 | 0.664 ± 0.021 | 0.221 ± 0.005 | 0.163 ± 0.014 |
| `transformer_self` | 0.656 ± 0.014 | 0.696 ± 0.002 | 0.526 ± 0.028 | 0.787 ± 0.000 | 0.657 ± 0.013 | 0.236 ± 0.001 | 0.179 ± 0.002 |
| `transformer_cross` | 0.645 ± 0.014 | 0.699 ± 0.012 | 0.455 ± 0.092 | 0.835 ± 0.121 | 0.650 ± 0.024 | 0.232 ± 0.026 | 0.183 ± 0.028 |

## 4. 结果分析

1. **六变体中平均 target BA 最高的是 `original_capm`。** 其 BA 为 `0.662 ± 0.023`；其次为 `transformer_self`（`0.656 ± 0.014`）、`conv_gate`（`0.655 ± 0.010`）和 `transformer_cross`（`0.645 ± 0.014`）。
2. **`original_capm` 是唯一在 seed43 和 seed44 都相对 `image_only` 为正的变体。** 两个 seed 的增益分别为 `+0.041`、`+0.019`，均值为 `+0.030`。这支持在当前 layer4/ResNet10 条件下，original CAPM 的增益具有初步稳定性。
3. **不能把所有表格交互变体都称为普遍增益。** `capm`、`conv_gate`、`transformer_cross` 在 seed43 为正、seed44 为负；`transformer_self` 两个 seed 均为正，但均值增益较小（`+0.024`）。
4. **`original_capm` 同时具有最好的平均 BA、Macro-F1、Brier 和 ECE。** 相对 image_only，平均 BA 从 `0.632` 提升至 `0.662`，Macro-F1 从 `0.633` 提升至 `0.664`，Brier 从 `0.236` 降至 `0.221`，ECE 从 `0.175` 降至 `0.163`。
5. **结果仍是小规模验证。** 仅包含 NACC→ADNI、两个 seeds 和一个 preset；target 全程未参与选择。结论应表述为“在预先固定的 layer4/ResNet10 条件下，original CAPM 获得初步可复现增益”，不应外推为所有架构、方向和层级均成立。

## 5. 可审计来源

- 已有基线训练：`outputs/resnet10_seed43_44_validation/layer4_pixel/NACC_to_ADNI_seed{43,44}/{image_only,original_capm}/journal_metrics.json`
- 新增训练：`outputs/resnet10_layer4_remaining_validation/layer4_pixel/NACC_to_ADNI_seed{43,44}/{capm,conv_gate,transformer_self,transformer_cross}/journal_metrics.json`
- 已有基线 target：`outputs/resnet10_seed43_44_validation/target_eval/layer4_pixel/NACC_to_ADNI_seed{43,44}/{image_only,original_capm}/target_metrics.json`
- 新增 target：`outputs/resnet10_layer4_remaining_validation/target_eval/layer4_pixel/NACC_to_ADNI_seed{43,44}/{capm,conv_gate,transformer_self,transformer_cross}/target_metrics.json`
- Evaluator：`experiments/evaluate_frozen_journal_target.py`
- 配置：`resolved_e2/resnet10_seed43_44_validation/mci_ad_resnet10_layer4_pixel_seed{43,44}.yaml`
