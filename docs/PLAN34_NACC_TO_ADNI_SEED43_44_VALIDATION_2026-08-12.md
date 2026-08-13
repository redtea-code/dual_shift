# Plan 34 NACC → ADNI：seed 43/44 预注册验证结果（MCI vs AD）

- 更新：2026-08-12 UTC+8
- 目的：验证 seed 42 中观察到的“`NACC → ADNI` target 最优指标随层级加深上升”现象，以及 CAPM 是否仅在更深层级出现相对同层 `image_only` 的稳定收益。
- 验证 seeds：43、44；seed 42 为先前发现性分析，未纳入下方验证性均值。
- 方向：NACC（source）→ ADNI（target）；ADNI target 指标为 subject-mean 聚合（241 subjects）。
- 预先固定候选：每个 preset 仅比较 `image_only` 与 `original_capm`，不根据新的 target 结果增加、删除或调换候选。
- 训练/选择：每个 seed 在 source validation 上以 BA + collapse guard 冻结 `best_checkpoint.pt`；所有 12 个 checkpoint 均通过 guard。
- Target：冻结 checkpoint 后只读 ADNI 推理；不训练、不调阈值、不用 target 选择 checkpoint、preset 或变体。
- 协议：`scan_filtered_v1_2026-08-08`；ADNI 1.5T scan-filtered，NACC 3T。

## 1. 完成状态

| preset | seed 43 source checkpoint | seed 43 target | seed 44 source checkpoint | seed 44 target | 状态 |
|---|---:|---:|---:|---:|---|
| `layer3_patch2` | 2/2 | 2/2 | 2/2 | 2/2 | 完成 |
| `layer4_pixel` | 2/2 | 2/2 | 2/2 | 2/2 | 完成 |
| `layer5_pixel` | 2/2 | 2/2 | 2/2 | 2/2 | 完成 |
| **合计** | **6/6** | **6/6** | **6/6** | **6/6** | **12 个冻结 target 报告完成** |

## 2. 验证结论

### 2.1 层级递增的 target 最优 BA 未获稳定复现

在每个 preset 内仅从预先固定的两个候选（`image_only`、`original_capm`）选择较高的 target BA，结果如下。此表用于检验层级趋势，不用于事后选择模型。

| seed | layer3_patch2 最优 BA | layer4_pixel 最优 BA | layer5_pixel 最优 BA | 是否单调上升 |
|---|---:|---:|---:|---|
| 43 | `image_only` 0.647 | `original_capm` 0.618 | `original_capm` 0.666 | 否 |
| 44 | `image_only` 0.655 | `image_only` 0.641 | `image_only` 0.631 | 否 |
| 43/44 均值 ± SD | 0.651 ± 0.005 | 0.630 ± 0.016 | 0.649 ± 0.025 | 否 |

两次验证 seed 都不是单调上升：seed 43 为 `0.647 → 0.618 → 0.666`，seed 44 为 `0.655 → 0.622 → 0.631`。因此，seed 42 中的上升现象目前只能保留为发现性观察，不能作为已确认的层数效应。

### 2.2 CAPM 的平均增益在更深层较大，但符号不跨 seed 稳定

| preset | seed 43: original_capm − image_only BA | seed 44: original_capm − image_only BA | 43/44 均值差 | 可复现性判断 |
|---|---:|---:|---:|---|
| `layer3_patch2` | -0.045 | -0.033 | -0.039 | 方向不一致 |
| `layer4_pixel` | +0.077 | -0.019 | +0.029 | 方向不一致 |
| `layer5_pixel` | +0.070 | -0.004 | +0.033 | 方向不一致 |

`original_capm` 的 43/44 平均 BA 差从 layer3 的 `−0.039` 变为 layer4 的 `+0.029` 和 layer5 的 `+0.033`，与“深层可能更适合交互”的机制假设一致；但 layer4、layer5 的单 seed 差值均一正一负，故尚未构成稳定机制证据。

## 3. Target 详细结果

各行的 source BA 为冻结 checkpoint 在 source validation 上的选择指标；Target BA 的方括号为该单次 target 评估的 200 次 subject bootstrap 95% CI。`Δ vs image-only` 是同 seed、同 preset 的 target BA 差。Brier/ECE 越低越好，其他指标越高越好。

### `layer3_patch2`

#### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 23 | 0.771 | 0.647 [0.578, 0.700] | 0.705 | 0.429 | 0.866 | 0.656 | 0.224 | 0.168 | +0.000 |
| `original_capm` | 9 | 0.707 | 0.602 [0.555, 0.651] | 0.726 | 0.260 | 0.945 | 0.601 | 0.236 | 0.211 | -0.045 |

#### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 17 | 0.757 | 0.655 [0.585, 0.716] | 0.687 | 0.571 | 0.738 | 0.649 | 0.248 | 0.189 | +0.000 |
| `original_capm` | 36 | 0.771 | 0.621 [0.554, 0.673] | 0.699 | 0.377 | 0.866 | 0.628 | 0.229 | 0.202 | -0.033 |

#### 43/44 target 汇总（均值 ± SD）

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.651 ± 0.005 | 0.696 ± 0.012 | 0.500 ± 0.101 | 0.802 ± 0.091 | 0.652 ± 0.005 | 0.236 ± 0.017 | 0.179 ± 0.015 |
| `original_capm` | 0.612 ± 0.013 | 0.713 ± 0.019 | 0.318 ± 0.083 | 0.905 ± 0.056 | 0.614 ± 0.019 | 0.232 ± 0.005 | 0.207 ± 0.006 |

### `layer4_pixel`

#### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 15 | 0.679 | 0.541 [0.499, 0.583] | 0.695 | 0.143 | 0.939 | 0.513 | 0.247 | 0.221 | +0.000 |
| `original_capm` | 21 | 0.771 | 0.618 [0.563, 0.682] | 0.698 | 0.377 | 0.860 | 0.624 | 0.243 | 0.220 | +0.077 |

#### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 35 | 0.707 | 0.641 [0.578, 0.708] | 0.699 | 0.429 | 0.854 | 0.649 | 0.236 | 0.198 | +0.000 |
| `original_capm` | 23 | 0.786 | 0.622 [0.548, 0.680] | 0.682 | 0.506 | 0.738 | 0.620 | 0.262 | 0.236 | -0.019 |

#### 43/44 target 汇总（均值 ± SD）

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.591 ± 0.071 | 0.697 ± 0.002 | 0.286 ± 0.202 | 0.896 ± 0.060 | 0.581 ± 0.096 | 0.241 ± 0.008 | 0.210 ± 0.017 |
| `original_capm` | 0.620 ± 0.003 | 0.690 ± 0.011 | 0.442 ± 0.092 | 0.799 ± 0.086 | 0.622 ± 0.003 | 0.253 ± 0.013 | 0.228 ± 0.012 |

### `layer5_pixel`

#### seed 43

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 10 | 0.771 | 0.596 [0.541, 0.644] | 0.735 | 0.260 | 0.933 | 0.594 | 0.222 | 0.193 | +0.000 |
| `original_capm` | 5 | 0.757 | 0.666 [0.606, 0.722] | 0.712 | 0.753 | 0.579 | 0.626 | 0.272 | 0.201 | +0.070 |

#### seed 44

| 变体 | 冻结 epoch | Source val BA | Target BA [95% CI] | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE | Δ vs image-only BA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 33 | 0.743 | 0.631 [0.572, 0.696] | 0.740 | 0.390 | 0.872 | 0.638 | 0.225 | 0.223 | +0.000 |
| `original_capm` | 17 | 0.764 | 0.627 [0.575, 0.692] | 0.742 | 0.364 | 0.890 | 0.634 | 0.219 | 0.181 | -0.004 |

#### 43/44 target 汇总（均值 ± SD）

| 变体 | BA | AUROC | Sens. | Spec. | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `image_only` | 0.614 ± 0.024 | 0.737 ± 0.004 | 0.325 ± 0.092 | 0.902 ± 0.043 | 0.616 ± 0.031 | 0.224 ± 0.002 | 0.208 ± 0.021 |
| `original_capm` | 0.647 ± 0.028 | 0.727 ± 0.021 | 0.558 ± 0.275 | 0.735 ± 0.220 | 0.630 ± 0.006 | 0.246 ± 0.037 | 0.191 ± 0.014 |

## 4. 跨层比较（seed 43/44）

| preset | image_only BA（均值 ± SD） | original_capm BA（均值 ± SD） | 最优候选 BA（均值 ± SD） | 平均最优变体 |
|---|---:|---:|---:|---|
| `layer3_patch2` | 0.651 ± 0.005 | 0.612 ± 0.013 | 0.651 ± 0.005 | `image_only` |
| `layer4_pixel` | 0.591 ± 0.071 | 0.620 ± 0.003 | 0.630 ± 0.016 | `original_capm` |
| `layer5_pixel` | 0.614 ± 0.024 | 0.647 ± 0.028 | 0.649 ± 0.025 | `original_capm` |

观察：在此有限验证候选集内，layer3 的最优平均 BA（`0.651 ± 0.005`）最高，layer5（`0.647 ± 0.028`）次之，layer4（`0.620 ± 0.003`）最低；这与严格单调随层数增加的假设不一致。

## 5. 解释边界与下一步

1. **未确认的趋势。** seed 43/44 没有复现 seed 42 的层级最优 BA 单调上升，因此不能声称“更深层在 NACC→ADNI 上必然更好”。
2. **仍值得保留的机制线索。** CAPM 相对 image-only 的均值 BA 在 layer4/layer5 为正、layer3 为负，但两个 seed 的符号均不一致。这只能作为下一轮的待检验假设，而非结论。
3. **性能与校准需同时审视。** 例如 layer5/original_capm 的平均 BA 高于同层 image-only（0.647 vs 0.614），但 Brier 更高（0.246 vs 0.224），且 sensitivity/specificity 的 seed 间波动较大；不宜只按 BA 做机制解释。
4. **后续确认实验建议。** 若要继续，应在不接触新 target 选择的前提下扩展独立 seeds（建议至少 seed 45–47），保持相同两个变体、三个 preset、source BA + collapse guard 选点及冻结 target 评估流程。预先指定主问题为“CAPM 相对同层 image-only 的跨 seed 平均 BA 差是否随层级递增”，并报告置信区间/配对 seed 差，而不以单个 best target 作为主要证据。

## 6. 可审计来源

- 训练报告：`outputs/seed43_44_validation/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/journal_metrics.json`
- 冻结 checkpoint：`outputs/seed43_44_validation/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/best_checkpoint.pt`
- Target 报告：`outputs/seed43_44_validation/target_eval/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/target_metrics.json`
- Target 预测：`outputs/seed43_44_validation/target_eval/{preset}/NACC_to_ADNI_seed{43,44}/{variant}/target_predictions.csv`
- Evaluator：`experiments/evaluate_frozen_journal_target.py`
- 冻结评价配置：`resolved_e2/seed43_44_validation/mci_ad_{preset}_seed{43,44}.yaml`（训练 seed 为 43/44，`split_seed=42`）。
