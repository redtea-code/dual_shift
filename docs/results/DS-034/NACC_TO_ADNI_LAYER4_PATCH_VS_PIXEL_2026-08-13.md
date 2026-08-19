# Plan 34 NACC → ADNI：ResNet10 layer4 patch vs pixel 消融结果

- 更新：2026-08-13 UTC+8
- 目的：在同一 ResNet10、layer4、NACC→ADNI、seed43/44 条件下，比较逐 feature-cell pixel 处理（`patch_size=1×1×1`）与 `2×2×2` patch 处理。
- Backbone：ResNet10，`layers=[1,1,1,1]`；输入 `160×196×160`；source split 固定为 `split_seed=42`。
- Pixel contract：`layer4_pixel`，175 tokens；Patch contract：`layer4_patch2`，72 tokens。
- 受 patch 粒度影响并重新训练的变体：`original_capm`、`transformer_self`、`transformer_cross`。`image_only`、`capm`、`conv_gate` 不使用该 patch 切分，复用既有结果、不重跑。
- 训练/选择：source-validation BA + collapse guard 冻结 checkpoint；12/12 checkpoint 均通过 guard。
- Target：冻结 checkpoint 后只读 ADNI 推理；target 不参与 checkpoint、preset 或变体选择。

## 1. 完成状态

| 条件 | seed 43 source/target | seed 44 source/target | 变体数 | 状态 |
|---|---:|---:|---:|---|
| layer4 pixel | 3/3 / 3/3 | 3/3 / 3/3 | 3 | 已有结果复用 |
| layer4 patch2 | 3/3 / 3/3 | 3/3 / 3/3 | 3 | 完成 |
| **合计** | **6/6 / 6/6** | **6/6 / 6/6** | **3** | **12 个 source/target 对照完成** |

## 2. Pixel vs Patch 总览（target 均值 ± SD）

| 变体 | Pixel BA | Patch BA | Patch − Pixel BA | Pixel AUROC | Patch AUROC | Pixel Brier | Patch Brier | Pixel ECE | Patch ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_capm` | 0.662 ± 0.023 | 0.626 ± 0.041 | -0.036 | 0.711 ± 0.010 | 0.675 ± 0.021 | 0.221 ± 0.005 | 0.244 ± 0.024 | 0.163 ± 0.014 | 0.184 ± 0.018 |
| `transformer_self` | 0.656 ± 0.014 | 0.650 ± 0.030 | -0.007 | 0.696 ± 0.002 | 0.674 ± 0.016 | 0.236 ± 0.001 | 0.271 ± 0.054 | 0.179 ± 0.002 | 0.219 ± 0.059 |
| `transformer_cross` | 0.645 ± 0.014 | 0.632 ± 0.005 | -0.013 | 0.699 ± 0.012 | 0.656 ± 0.010 | 0.232 ± 0.026 | 0.238 ± 0.002 | 0.183 ± 0.028 | 0.186 ± 0.016 |

## 3. 每个 seed 的冻结 target 结果

Target BA 方括号为 200 次 subject bootstrap 95% CI；`Patch − Pixel` 为同一 seed、同一变体的 BA 差。Brier/ECE 越低越好。

### seed 43

| 变体 | Pixel epoch | Pixel BA [95% CI] | Patch epoch | Patch BA [95% CI] | Patch − Pixel BA | Pixel AUROC | Patch AUROC | Pixel Macro-F1 | Patch Macro-F1 | Pixel Brier | Patch Brier | Pixel ECE | Patch ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_capm` | 17 | 0.646 [0.581, 0.704] | 8 | 0.597 [0.539, 0.655] | -0.049 | 0.704 | 0.660 | 0.649 | 0.594 | 0.224 | 0.261 | 0.173 | 0.196 |
| `transformer_self` | 43 | 0.647 [0.583, 0.709] | 36 | 0.628 [0.561, 0.689] | -0.018 | 0.697 | 0.663 | 0.648 | 0.589 | 0.236 | 0.309 | 0.177 | 0.261 |
| `transformer_cross` | 25 | 0.655 [0.603, 0.711] | 27 | 0.635 [0.583, 0.702] | -0.020 | 0.708 | 0.662 | 0.667 | 0.641 | 0.214 | 0.237 | 0.163 | 0.174 |

### seed 44

| 变体 | Pixel epoch | Pixel BA [95% CI] | Patch epoch | Patch BA [95% CI] | Patch − Pixel BA | Pixel AUROC | Patch AUROC | Pixel Macro-F1 | Patch Macro-F1 | Pixel Brier | Patch Brier | Pixel ECE | Patch ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `original_capm` | 15 | 0.679 [0.624, 0.734] | 35 | 0.655 [0.586, 0.715] | -0.023 | 0.718 | 0.690 | 0.679 | 0.660 | 0.217 | 0.227 | 0.153 | 0.171 |
| `transformer_self` | 46 | 0.666 [0.608, 0.732] | 47 | 0.671 [0.612, 0.726] | +0.005 | 0.695 | 0.685 | 0.666 | 0.678 | 0.235 | 0.233 | 0.180 | 0.178 |
| `transformer_cross` | 41 | 0.635 [0.571, 0.693] | 21 | 0.628 [0.580, 0.696] | -0.007 | 0.690 | 0.649 | 0.633 | 0.635 | 0.251 | 0.240 | 0.203 | 0.197 |

## 4. 结论与解释边界

1. **在本实验条件下，pixel 处理优于 patch2 处理。** 三个受影响变体的跨 seed 平均 target BA 均由 pixel 条件取得更高值：original CAPM 差 `−0.036`、transformer self 差 `−0.007`、transformer cross 差 `−0.013`（均为 Patch−Pixel）。
2. **original CAPM 的 pixel 优势跨两个 seed 一致。** patch2 相比 pixel 的 BA 在 seed43 为 `−0.049`、seed44 为 `−0.023`，同时 AUROC、Macro-F1、Brier 和 ECE 的跨 seed 平均也均不优于 pixel。
3. **transformer self 的 BA 对粒度敏感但结论较弱。** seed43 patch2 下降 `−0.018`，seed44 小幅上升 `+0.005`；平均 BA 仍低于 pixel，且 patch2 的 Brier/ECE 更差。
4. **transformer cross 同样偏向 pixel。** 两个 seed 的 Patch−Pixel BA 都为负（`−0.020`、`−0.007`）。
5. **结论范围。** 该结果仅支持在固定的 ResNet10/layer4/NACC→ADNI、token 数由 175 降至 72 的对照中，细粒度 pixel 处理更好。它不能单独判定差异来自 patch 汇聚、token 数变化或两者的交互，也不能外推至其他 layer、方向或 backbone。

## 5. 可审计来源

- Patch2 训练：`outputs/resnet10_layer4_patch2_seed43_44/NACC_to_ADNI_seed{43,44}/{original_capm,transformer_self,transformer_cross}/journal_metrics.json`
- Patch2 target：`outputs/resnet10_layer4_patch2_seed43_44/target_eval/NACC_to_ADNI_seed{43,44}/{original_capm,transformer_self,transformer_cross}/target_metrics.json`
- Pixel original CAPM：`outputs/resnet10_seed43_44_validation/{,target_eval/}layer4_pixel/NACC_to_ADNI_seed{43,44}/original_capm/`
- Pixel transformers：`outputs/resnet10_layer4_remaining_validation/{,target_eval/}layer4_pixel/NACC_to_ADNI_seed{43,44}/{transformer_self,transformer_cross}/`
- Patch2 配置：`resolved_e2/layer4_patch2_resnet10_seed43_44/mci_ad_resnet10_layer4_patch2_seed{43,44}.yaml`
- Evaluator：`experiments/evaluate_frozen_journal_target.py`
