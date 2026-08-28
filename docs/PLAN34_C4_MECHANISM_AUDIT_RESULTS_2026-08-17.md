# Plan 34 C4 机制审计结果

- 更新：2026-08-17 UTC+8
- 方向：ADNI 1.5T → NACC 3T；MCI vs AD
- 审计对象：冻结 C4 `frequency_uda` checkpoints，seeds 43、44
- 状态：P0/P1 审计完成，2/2 通过产物与泄漏边界核验

## 1. 审计边界与完成状态

本审计不重训模型，也不改写既有 C4 target 结果。每个 seed 使用：

- `S_train`：先验 source population；144 个 subject
- `S_val`：独立诊断 population；48 个 subject
- `T_adapt`：仅无标签图像和协变量；264 个 subject
- `T_test`：仅使用 split membership 进行 subject-disjoint 验证；不读取标签、指标或预测

两组 seed 均生成全部 9 个要求产物：

`prior_provenance.json`、`source_target_band_summary.csv`、`prior_bootstrap_stability.json`、`gate_activity.csv`、`pre_post_spectrum_summary.csv`、`c4_identity_forward_comparison.json`、`source_environment_audit.csv`、`covariate_support_audit.csv`、`pre_post_discrepancy.json`。

审计产物中的 boundary flags 均为：`target_labels_read=false`、`target_metrics_read=false`、`target_test_accessed=false`。每个 seed 使用 1,000 次 subject-level bootstrap；`T_adapt` 与 `T_test` subject-disjoint。

## 2. 先验频段差异稳定性

频段为 low `[0,0.15)`、mid `[0.15,0.35)`、high `[0.35,∞)`。`raw_d` 使用计划定义的 pooled-standard-deviation 标准化差异，`normalized_discrepancy` 再按最大 band discrepancy 归一化。

| seed | raw d low / mid / high | normalized low / mid / high | 最高 discrepancy 排序概率 |
|---|---|---|---|
| 43 | 0.446 / 0.018 / 0.345 | 1.000 / 0.040 / 0.775 | low 95.8%，high 4.1% |
| 44 | 0.220 / 0.088 / 0.215 | 1.000 / 0.398 / 0.976 | low 48.4%，high 44.8% |

seed43 的排序较明确：low > high > mid；seed44 的 low/high 排序不稳定，low 仅略高于 high。bootstrap CI 也均为正，但 seed44 的 low/high 区间明显重叠。

这与此前 prior JSON 中的 band ordering 反转一致，说明当前 target-spectrum prior 的 band ordering 对 seed 敏感。它支持“存在频谱差异”的描述性证据，但不支持把当前 ordering 当作稳定机制证据。

## 3. Gate 是否实际干预

| seed | population | effective strength | identity loss | attenuation 均值 |
|---|---|---:|---:|---:|
| 43 | S_train | 0.04879 | 0.000356 | 0.98339 |
| 43 | S_val | 0.04879 | 0.000356 | 0.98336 |
| 43 | T_adapt | 0.04879 | 0.000355 | 0.98337 |
| 44 | S_train | 0.04855 | 0.000302 | 0.98347 |
| 44 | S_val | 0.04855 | 0.000303 | 0.98346 |
| 44 | T_adapt | 0.04855 | 0.000302 | 0.98348 |

C4 不是严格 identity：gate 有非零 strength，平均 attenuation 约 0.983，identity loss 非零。但干预幅度较小，且三个 population 的强度几乎一致；因此不能仅凭非零 attenuation 声称存在强机制效应。

## 4. Gate 前后 layer4 频谱

本节使用严格的 gate-level pre/post layer4 feature maps；不把 layer5/CAPM 后特征误作 gate 输出。

### seed43

| population | low pre → post | mid pre → post | high pre → post |
|---|---:|---:|---:|
| S_train | 0.4282 → 0.4375 | 0.2271 → 0.2234 | 0.3447 → 0.3391 |
| S_val | 0.4314 → 0.4407 | 0.2295 → 0.2258 | 0.3391 → 0.3335 |
| T_adapt | 0.4319 → 0.4412 | 0.2270 → 0.2233 | 0.3411 → 0.3355 |

### seed44

| population | low pre → post | mid pre → post | high pre → post |
|---|---:|---:|---:|
| S_train | 0.4192 → 0.4256 | 0.2349 → 0.2305 | 0.3460 → 0.3439 |
| S_val | 0.4215 → 0.4280 | 0.2362 → 0.2318 | 0.3423 → 0.3402 |
| T_adapt | 0.4177 → 0.4241 | 0.2342 → 0.2299 | 0.3480 → 0.3459 |

Gate 的实际方向较一致：提高 low fraction，并降低 mid/high fraction；但变化量小，且没有显示只对 `T_adapt` 产生特殊作用。

## 5. Gate-disabled frozen forward 对照

同一 frozen C4 model、同一输入和同一协变量向量分别执行 enabled/disabled forward；未修改 checkpoint。

| seed | feature absolute difference | logit absolute difference | finite | checkpoint mutated |
|---|---:|---:|---:|---:|
| 43 | 0.00494 | 0.02357 | yes | no |
| 44 | 0.00542 | 0.01915 | yes | no |

这证明 gate 对冻结 forward 的输出有可测影响，但该结果本身不证明 target-specific prior 优于普通 gate capacity；C2/C3 对照和原始 C4 target 结果仍是必要的机制边界。

## 6. Source frequency environment 审计

以下仅使用 source rows；GroupDRO 状态无法从 frozen checkpoint 重建，因此没有伪造权重或 worst-group risk。

| seed | environment | cross-entropy | balanced accuracy | GroupDRO weight |
|---|---|---:|---:|---:|
| 43 | original | 0.0580 | 0.9899 | unavailable |
| 43 | lowpass | 0.0901 | 0.9798 | unavailable |
| 43 | downsample_resample | 0.1015 | 0.9798 | unavailable |
| 43 | mild_blur | 0.0796 | 0.9798 | unavailable |
| 44 | original | 0.0436 | 0.9949 | unavailable |
| 44 | lowpass | 0.0716 | 0.9899 | unavailable |
| 44 | downsample_resample | 0.0825 | 0.9899 | unavailable |
| 44 | mild_blur | 0.0619 | 0.9899 | unavailable |

Original environment 最好，频率扰动使 source CE 增加、BA 轻微下降。这说明 frozen C4 对 source frequency perturbations 有一定鲁棒性，但不能将 source environment 结果直接解释为 target adaptation 增益。

## 7. 协变量支持

性别使用冻结 source preprocessor 编码：female=0、male=1；未再被误报为缺失。两 seed 的 source / `T_adapt` 均为 192 / 264 个记录，age、education、sex 缺失率均为 0。

- seed43：source vs `T_adapt` 的 age SMD = 0.534，education SMD = −0.122，sex encoded SMD = 0.244。
- seed44：age SMD = 0.494，education SMD = −0.129，sex encoded SMD = 0.306。
- source sex female/male：68/124；`T_adapt` female/male：seed43 为 125/139，seed44 为 133/131。

因此 target adaptation cohort 与 source 在人口学构成上并非完全一致，频谱 prior 与协变量差异仍存在混杂可能。该审计不能建立频谱差异的场强因果解释。

## 8. Gate 前后 discrepancy

| seed | pre raw d low / mid / high | post raw d low / mid / high |
|---|---|---|
| 43 | 0.446 / 0.018 / 0.345 | 0.445 / 0.023 / 0.347 |
| 44 | 0.220 / 0.088 / 0.215 | 0.222 / 0.086 / 0.216 |

Gate 后 source–`T_adapt` discrepancy 几乎不变：seed43 的 low/high 差异没有实质降低，seed44 也没有一致的 discrepancy 收缩。这不支持“当前 gate 已将 target spectrum 稳定拉回 source spectrum”的强机制叙述。

## 9. 综合判定

本次审计支持以下较谨慎的结论：

1. C4 gate 确实在冻结 forward 中产生了非零、可测但幅度较小的频谱干预。
2. 先验识别出的 source–`T_adapt` 频谱差异在两个 seed 中存在，但 band ordering 不稳定，尤其 low/high 在 seed44 接近反转。
3. gate 前后 discrepancy 基本不变，且三类 population 的 gate activity 相近；当前证据不足以证明 C4 的 target 增益来自稳定的 target-specific frequency ordering。
4. source frequency environment 结果显示模型对四种频率扰动仍保持较高 source BA，但无法分离普通 environment robustness、gate capacity 与 target prior 作用。
5. 协变量支持存在差异，因此频谱差异仍可能混入 age、sex、education 构成差异。

**机制审计判定：不支持将当前 C4 解释为已确证的 target-specific frequency mechanism。** 这与此前 C4 主矩阵的 NO-GO / negative-inconclusive 结论一致；不建议仅依据当前审计引入 C7/C8、SoftRegion 或第二个新机制。

## 10. 可审计来源

- 审计计划：`docs/PLAN34_C4_MECHANISM_AUDIT_PLAN_2026-08-17.md`
- 审计实现：`experiments/audit_frequency_uda_c4.py`
- 审计测试：`tests/test_frequency_uda_c4_audit.py`
- 审计产物：`outputs/c4_mechanism_audit/ADNI_to_NACC/seed_{43,44}/`
- 既有 C4 最终结果：`docs/PLAN34_FREQUENCY_GUIDED_UDA_FINAL_RESULTS_2026-08-17.md`
- 先验与冻结 checkpoint：`outputs/frequency_uda_prepare/` 与 `outputs/frequency_uda_final_v2/`
