# Plan 34 全局低秩 MRI Skip：Seed45/46 验证结果

- 更新时间：2026-08-18 UTC+8
- 目的：验证 `PLAN34_GLOBAL_LOW_RANK_SKIP_RESULTS_2026-08-17.md` 中关于低秩分支“没有可重复 source gain”的判断
- 方向：ADNI 1.5T → NACC 3T，但本实验严格 source-only
- 变体：`original_capm`、`global_stats_skip`、`global_frequency_skip`
- 新增训练 seeds：45、46；原报告 seeds：43、44
- target 边界：没有构造、读取、推理或选择 NACC target rows

## 1. 协议与 split 核验

新实验完全复用原报告配置：ResNet10 (`layers=[1,1,1,1]`)、`layer5_pixel + original_capm(age, sex, education)`、50 epochs、batch size 4、AdamW learning rate `1e-4`、weight decay `1e-4`、`split_seed=42`、scan-filtered ADNI source protocol。

新生成的 seed45/46 manifest 与既有 seed43/44 的 source subject membership 完全一致：

- source train：144 subjects
- source validation：48 subjects
- source test：49 subjects
- target subjects：0
- `source_only=true`

因此新增 seed 主要验证训练初始化与训练过程变化，而不是改变数据划分。

## 2. Seed45/46 source validation

| seed | variant | CE | BA | AUC | sensitivity | specificity | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 45 | R0 `original_capm` | 0.737 | 0.641 | 0.736 | 0.375 | 0.906 | 0.648 |
| 45 | `global_stats_skip` | 1.013 | 0.625 | 0.711 | 0.875 | 0.375 | 0.541 |
| 45 | `global_frequency_skip` | 1.861 | 0.641 | 0.686 | 1.000 | 0.281 | 0.510 |
| 46 | R0 `original_capm` | 1.066 | 0.594 | 0.676 | 0.188 | 1.000 | 0.573 |
| 46 | `global_stats_skip` | 1.282 | 0.563 | 0.695 | 0.563 | 0.563 | 0.547 |
| 46 | `global_frequency_skip` | 1.189 | 0.609 | 0.699 | 0.500 | 0.719 | 0.608 |

相对同 seed R0，validation BA 差值为：

- `global_stats_skip`：seed45 −0.016；seed46 −0.031
- `global_frequency_skip`：seed45 0.000；seed46 +0.016

global-frequency 在 seed46 的 BA 只获得 +0.016，seed45 没有提升；同时 seed45 CE 明显增大、specificity 降至 0.281。新 seed 没有显示一致的 clean validation gain。

## 3. Seed45/46 source test

source test 只作描述性诊断，不参与 checkpoint 选择。

| seed | variant | CE | BA | AUC | sensitivity | specificity | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 45 | R0 `original_capm` | 0.827 | 0.705 | 0.773 | 0.500 | 0.909 | 0.719 |
| 45 | `global_stats_skip` | 0.925 | 0.697 | 0.790 | 1.000 | 0.394 | 0.590 |
| 45 | `global_frequency_skip` | 1.502 | 0.652 | 0.803 | 1.000 | 0.303 | 0.523 |
| 46 | R0 `original_capm` | 1.233 | 0.594 | 0.858 | 0.188 | 1.000 | 0.576 |
| 46 | `global_stats_skip` | 1.070 | 0.677 | 0.809 | 0.688 | 0.667 | 0.656 |
| 46 | `global_frequency_skip` | 1.139 | 0.690 | 0.811 | 0.563 | 0.818 | 0.693 |

source test BA 相对 R0：

- `global_stats_skip`：seed45 −0.008；seed46 +0.083
- `global_frequency_skip`：seed45 −0.053；seed46 +0.097

新增 seeds 也呈现方向不一致：seed46 两个低秩变体高于 R0，但 seed45 两者都低于 R0。尤其 global-frequency 的 seed45 specificity=0.303，说明 BA 变化伴随明显阈值/类别平衡变化，不能直接解释为稳定性能增益。

## 4. 四 seed 综合验证

原报告 seeds43/44 与本次 seeds45/46 的 validation BA 差值（低秩 − 同 seed R0）如下：

| seed | global_stats_skip | global_frequency_skip |
|---:|---:|---:|
| 43 | 0.000 | 0.000 |
| 44 | −0.063 | +0.016 |
| 45 | −0.016 | 0.000 |
| 46 | −0.031 | +0.016 |

source test BA 差值如下：

| seed | global_stats_skip | global_frequency_skip |
|---:|---:|---:|
| 43 | −0.042 | +0.003 |
| 44 | +0.015 | −0.014 |
| 45 | −0.008 | −0.053 |
| 46 | +0.083 | +0.097 |

四 seed 结果没有形成一致方向：

- validation：global-stats 四次均不优于 R0；global-frequency 仅 seed46 获得 +0.016，seed43/45 持平、seed44 +0.016。
- source test：两个低秩分支均出现正负交替，global-frequency 在 seed45 明显下降，在 seed46 上升。
- sensitivity/specificity 变化较大，部分 BA 变化伴随类别阈值偏移。

## 5. 机制活动与解释

新 seed 的 low-rank branch 均为非恒等：

| seed | variant | effective strength | residual-relative RMS | nonidentity fraction |
|---:|---|---:|---:|---:|
| 45 | global_stats_skip | 0.01446 | 0.01446 | 1.000 |
| 45 | global_frequency_skip | 0.01674 | 0.01674 | 1.000 |
| 46 | global_stats_skip | 0.01436 | 0.01436 | 1.000 |
| 46 | global_frequency_skip | 0.01577 | 0.01577 | 1.000 |

因此新验证没有发现 branch 未接入或 gate 退化为 identity 的问题。没有稳定收益是结果本身，而不是模块没有生效。

## 6. 验证结论

seed45/46 支持并加强原报告的谨慎判断：在相同 source-only 协议下，global mean/std 和 global low/mid/high frequency fractions 都没有显示跨 seed 可重复的 source gain。global-frequency 在少数 seed 上出现小幅 BA 提升，但无法在 validation 与 source test、或不同 seed 之间保持一致，且常伴随 specificity/sensitivity 重分配。

**结论：原报告关于“低秩全局统计分支没有可重复 source gain”的现象在新增 seed45/46 上得到支持。没有理由基于这组 source-only 结果继续进行 target evaluation。**

## 7. 限制

- 本实验未使用 NACC target，不能做跨域或 scanner 因果推断。
- 四个 source frequency environments 的 evaluation CE/BA/worst-environment risk 仍未由 runner 作为独立 artifact 输出。
- source test 未参与 checkpoint 选择，仅用于描述性核验。
- 该验证增加的是训练 seed，不是新的 subject split；这一点由 manifest membership 核验确认。
