# Plan 34 NACC → ADNI：ResNet10 最小结构消融结果（MCI vs AD）

- 更新：2026-08-12 UTC+8
- 目的：在与 ResNet18 验证完全相同的数据协议与评估流程下，比较 ResNet10 容量对三种 feature-scale preset 及 `original_capm` 的影响。
- 方向：NACC（source）→ ADNI（target）；target 为 subject-mean 聚合，241 subjects。
- 验证 seeds：43、44；此前 ResNet18 seed43/44 作为成对容量对照。
- Backbone：ResNet10，`layers=[1,1,1,1]`；输入 `160×196×160`。
- Preset：`layer3_patch2`、`layer4_pixel`、`layer5_pixel`。
- 变体：`image_only`、`original_capm`。
- 训练/选择：source-validation BA + collapse guard 冻结 checkpoint；所有 12 个 checkpoint 均通过 guard。
- Target：冻结 checkpoint 后只读 ADNI 推理；target 不参与 checkpoint、preset 或变体选择。
- 协议：`scan_filtered_v1_2026-08-08`；ADNI 1.5T scan-filtered，NACC 3T。

## 1. 完成状态

| preset | seed 43 source | seed 43 target | seed 44 source | seed 44 target | 状态 |
|---|---:|---:|---:|---:|---|
| `layer3_patch2` | 2/2 | 2/2 | 2/2 | 2/2 | 完成 |
| `layer4_pixel` | 2/2 | 2/2 | 2/2 | 2/2 | 完成 |
| `layer5_pixel` | 2/2 | 2/2 | 2/2 | 2/2 | 完成 |
| **合计** | **6/6** | **6/6** | **6/6** | **6/6** | **12 个冻结 target 报告完成** |

## 2. ResNet10 target 结果总览

| preset | image_only BA（43/44） | original_capm BA（43/44） | 最优候选 BA（43/44） | 平均最优变体 |
|---|---:|---:|---:|---|
| `layer3_patch2` | 0.615 ± 0.092 | 0.649 ± 0.015 | 0.670 ± 0.015 | `original_capm` |
| `layer4_pixel` | 0.632 ± 0.039 | 0.662 ± 0.023 | 0.662 ± 0.023 | `original_capm` |
| `layer5_pixel` | 0.590 ± 0.034 | 0.619 ± 0.022 | 0.619 ± 0.022 | `original_capm` |

## 3. ResNet10 详细 target 结果

各行的 source BA 是冻结 checkpoint 的 source-validation 选择指标；Target BA 方括号为 200 次 subject bootstrap 95% CI；`Δ vs image-only` 为同 seed、同 preset 的差值。

### `layer3_patch2`

#### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 15 | 0.686 | 0.550 [0.509, 0.597] | 0.678 | 0.143 | 0.957 | 0.521 | 0.242 | 0.222 | +0.000 |
| `original_capm` | 18 | 0.757 | 0.659 [0.601, 0.716] | 0.699 | 0.623 | 0.695 | 0.646 | 0.237 | 0.132 | +0.109 |

#### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 43 | 0.750 | 0.680 [0.611, 0.750] | 0.698 | 0.610 | 0.750 | 0.673 | 0.246 | 0.195 | +0.000 |
| `original_capm` | 16 | 0.757 | 0.638 [0.581, 0.699] | 0.686 | 0.442 | 0.835 | 0.645 | 0.223 | 0.163 | -0.042 |

#### 43/44 target 汇总（均值 ± SD）

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.615 ± 0.092 | 0.688 ± 0.014 | 0.377 ± 0.331 | 0.854 ± 0.147 | 0.597 ± 0.107 | 0.244 ± 0.003 | 0.209 ± 0.019 |
| `original_capm` | 0.649 ± 0.015 | 0.693 ± 0.009 | 0.532 ± 0.129 | 0.765 ± 0.099 | 0.645 ± 0.001 | 0.230 ± 0.010 | 0.148 ± 0.023 |

### `layer4_pixel`

#### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 18 | 0.757 | 0.605 [0.551, 0.660] | 0.657 | 0.442 | 0.768 | 0.607 | 0.245 | 0.178 | +0.000 |
| `original_capm` | 17 | 0.757 | 0.646 [0.581, 0.704] | 0.704 | 0.494 | 0.799 | 0.649 | 0.224 | 0.173 | +0.041 |

#### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 25 | 0.743 | 0.660 [0.599, 0.724] | 0.702 | 0.545 | 0.774 | 0.659 | 0.226 | 0.171 | +0.000 |
| `original_capm` | 15 | 0.743 | 0.679 [0.624, 0.734] | 0.718 | 0.558 | 0.799 | 0.679 | 0.217 | 0.153 | +0.019 |

#### 43/44 target 汇总（均值 ± SD）

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.632 ± 0.039 | 0.680 ± 0.032 | 0.494 ± 0.073 | 0.771 ± 0.004 | 0.633 ± 0.037 | 0.236 ± 0.014 | 0.175 ± 0.005 |
| `original_capm` | 0.662 ± 0.023 | 0.711 ± 0.010 | 0.526 ± 0.046 | 0.799 ± 0.000 | 0.664 ± 0.021 | 0.221 ± 0.005 | 0.163 ± 0.014 |

### `layer5_pixel`

#### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 9 | 0.707 | 0.566 [0.530, 0.615] | 0.740 | 0.156 | 0.976 | 0.540 | 0.238 | 0.204 | +0.000 |
| `original_capm` | 7 | 0.707 | 0.604 [0.549, 0.652] | 0.706 | 0.299 | 0.909 | 0.606 | 0.219 | 0.168 | +0.038 |

#### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 32 | 0.764 | 0.614 [0.565, 0.679] | 0.733 | 0.338 | 0.890 | 0.619 | 0.234 | 0.209 | +0.000 |
| `original_capm` | 32 | 0.779 | 0.634 [0.576, 0.687] | 0.737 | 0.403 | 0.866 | 0.642 | 0.226 | 0.202 | +0.020 |

#### 43/44 target 汇总（均值 ± SD）

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.590 ± 0.034 | 0.737 ± 0.005 | 0.247 ± 0.129 | 0.933 ± 0.060 | 0.580 ± 0.056 | 0.236 ± 0.003 | 0.206 ± 0.004 |
| `original_capm` | 0.619 ± 0.022 | 0.722 ± 0.021 | 0.351 ± 0.073 | 0.887 ± 0.030 | 0.624 ± 0.025 | 0.222 ± 0.005 | 0.185 ± 0.024 |

## 4. ResNet10 vs ResNet18 容量对照

以下比较使用相同 preset、方向、seed、变体和 target 评估标准；数值为 target BA 均值 ± SD。

| preset | variant | ResNet10 | ResNet18 | ResNet10 − ResNet18 |
|---|---|---:|---:|---:|
| `layer3_patch2` | `image_only` | 0.615 ± 0.092 | 0.651 ± 0.005 | -0.036 |
| `layer3_patch2` | `original_capm` | 0.649 ± 0.015 | 0.612 ± 0.013 | +0.037 |
| `layer4_pixel` | `image_only` | 0.632 ± 0.039 | 0.591 ± 0.071 | +0.041 |
| `layer4_pixel` | `original_capm` | 0.662 ± 0.023 | 0.620 ± 0.003 | +0.042 |
| `layer5_pixel` | `image_only` | 0.590 ± 0.034 | 0.614 ± 0.024 | -0.024 |
| `layer5_pixel` | `original_capm` | 0.619 ± 0.022 | 0.647 ± 0.028 | -0.028 |

## 5. CAPM 增益与容量结论

| preset | ResNet10 seed43 Δ | ResNet10 seed44 Δ | ResNet10 平均 Δ | ResNet18 平均 Δ |
|---|---:|---:|---:|---:|
| `layer3_patch2` | +0.109 | -0.042 | +0.034 | -0.039 |
| `layer4_pixel` | +0.041 | +0.019 | +0.030 | +0.029 |
| `layer5_pixel` | +0.038 | +0.020 | +0.029 | +0.033 |

1. **ResNet10 下 layer4 的平均 target BA 最高。** 在两个固定变体的候选集合中，`layer4_pixel` 最优候选均值为 `0.662 ± 0.023`，低于 layer3 的 `0.670 ± 0.015`，高于 layer5 的 `0.619 ± 0.022`。
2. **ResNet10 下 original_capm 在三个 preset 的均值 BA 都高于同层 image_only。** 增益分别为 layer3 `+0.034`、layer4 `+0.030`、layer5 `+0.029`；但仅两个 seed，仍属于小规模验证。
3. **与 ResNet18 相比，ResNet10 的收益不是统一方向。** layer3 的两种变体均低于 ResNet18；layer4 的两种变体均高于 ResNet18；layer5 的两种变体也低于 ResNet18。说明模型容量与 feature scale 存在交互，不能把 layer4 的结果简单归因于“更深层”或“更大模型”。
4. **layer5 未显示优势。** ResNet10 的 layer5 平均 BA 低于 layer3/layer4，且 `original_capm` 虽优于同层 image_only，仍不能支持 layer5 是普遍最优配置。
5. **解释边界。** 本实验固定 split seed=42，仅使用训练 seeds 43/44，且 target 只做冻结盲测；结果适合用于容量-尺度消融，不足以作为最终泛化结论。

## 6. 可审计来源

- ResNet10 训练报告：`outputs/resnet10_seed43_44_validation/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/journal_metrics.json`
- ResNet10 checkpoint：`outputs/resnet10_seed43_44_validation/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/best_checkpoint.pt`
- ResNet10 target：`outputs/resnet10_seed43_44_validation/target_eval/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/target_metrics.json`
- ResNet10 配置：`resolved_e2/resnet10_seed43_44_validation/mci_ad_resnet10_{preset}_seed{43,44}.yaml`
- ResNet18 对照报告：`NACC_TO_ADNI_RESNET10_VALIDATION_2026-08-12.md`
- Evaluator：`experiments/evaluate_frozen_journal_target.py`
