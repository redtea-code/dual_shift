# Plan 34 全局低秩 MRI Skip：Source-Only 结果

- 更新：2026-08-17 UTC+8
- 实验范围：ADNI 1.5T source-only，MCI vs AD，固定 scan-filtered source split
- 比较对象：复用 R0 `original_capm`、已完成的高维 raw rFFT R1、`global_stats_skip` 与 `global_frequency_skip`
- 重要边界：没有构造、读取、推理、选择或报告 NACC target rows；结果不是跨域泛化证据。

## 1. 问题与实现

原始 R1 通过 `8×8×8` raw rFFT descriptor 和两个 3D convolution 构造 residual。这可能使结果同时反映频率信息和额外高维空间 encoder 的容量。本实验将 residual 输入严格限制为每个 MRI 的全局标量，并保留相同的 layer3 bounded-RMS residual 注入、`layer5_pixel + original_capm(age, sex, education)` backbone、训练预算和冻结 source split。

| 变体 | 每样本输入 | 输入维数 | 从输入到 layer3 residual 的可训练映射 | 不使用的部分 |
|---|---|---:|---|---|
| `global_stats_skip` | MRI global mean、global std | 2 | `Linear(2→256)` 后空间 broadcast | rFFT、compact grid、3D raw encoder |
| `global_frequency_skip` | raw MRI low/mid/high rFFT band-power fractions | 3 | `Linear(3→256)` 后空间 broadcast | compact grid、3D raw encoder |

两个分支都在 layer3 后注入：`h' = h + a × RMS(h)/RMS(r) × r`，其中 `a=0.15×tanh(theta)`；所以低秩输入不会形成第二个空间 backbone。每个 variant 在 seed43、44 上完成 50 epochs source-only 训练。

## 2. 产物与验证

四项实验均有 `best_checkpoint.pt`、`last_checkpoint.pt`、`journal_metrics.json`、source validation/test predictions。所有 manifest 的 `source_only=true`；model signature 分别记录 stats/2D 与 frequency/3D low-rank input，注入 stage 均为 layer3。

| seed | variant | checkpoint SHA-256 | metrics SHA-256 |
|---|---|---|---|
| 43 | global_stats_skip | `273fb66e58504bc63d06b403e69bd930ffca4ce8a16d36ef175fac0c378ae1c6` | `c627328cd8c885012de797de3801a229a32d0d15ddffd35eac771346173fa782` |
| 43 | global_frequency_skip | `e81bbe28bb507793e57817a9e50e75267525e24d8a12430e703ef75e8c9a6f4a` | `97cf0f32cf6925671f9110e79e225c920d9e85f4843607f15bd73a7115e32270` |
| 44 | global_stats_skip | `766021f3cb9e2701e172d574289a70e55142e314c76e8930ae60c079e38e09c3` | `9a4b2525bc56fa098ec59b52c921d52c47af997f426a9e1b136f71154236fb76` |
| 44 | global_frequency_skip | `1515d33666a58ab07326fc40d7e2e4d451bee806f6597e391ef1a866eecdabfa` | `04a2ae82b3c4fbd2ae1161b6d82459b505680ab98bf41626d92804d64028efa0` |

## 3. Clean source validation（checkpoint selection split）

| seed | variant | CE | BA | AUC | sensitivity | specificity | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 43 | R0 original_capm | 0.630 | 0.609 | 0.680 | 0.625 | 0.594 | 0.590 |
| 43 | high-dimensional R1 raw_frequency_skip | 1.224 | 0.594 | 0.662 | 0.562 | 0.625 | 0.582 |
| 43 | global_stats_skip | 1.254 | 0.609 | 0.674 | 0.938 | 0.281 | 0.492 |
| 43 | global_frequency_skip | 1.081 | 0.609 | 0.668 | 0.812 | 0.406 | 0.542 |
| 44 | R0 original_capm | 0.961 | 0.641 | 0.666 | 0.688 | 0.594 | 0.614 |
| 44 | high-dimensional R1 raw_frequency_skip | 1.896 | 0.547 | 0.689 | 0.938 | 0.156 | 0.390 |
| 44 | global_stats_skip | 1.194 | 0.578 | 0.646 | 0.562 | 0.594 | 0.564 |
| 44 | global_frequency_skip | 1.792 | 0.656 | 0.660 | 0.812 | 0.500 | 0.603 |

相对 R0，global-stats 的 validation BA 变化为 seed43 0.000、seed44 −0.063；global-frequency 为 seed43 0.000、seed44 +0.016。尽管 global-frequency 在 seed44 的 BA 较 R0 高 0.016，两个 seed 没有一致的 BA 或 CE 改善，且 AUC 没有提升。它不满足稳定 source gain 的标准。

## 4. Source test（描述性；未参与 checkpoint 选择）

| seed | variant | CE | BA | AUC | sensitivity | specificity | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 43 | R0 original_capm | 0.596 | 0.739 | 0.809 | 0.750 | 0.727 | 0.718 |
| 43 | high-dimensional R1 raw_frequency_skip | 1.031 | 0.662 | 0.786 | 0.688 | 0.636 | 0.638 |
| 43 | global_stats_skip | 1.053 | 0.697 | 0.792 | 1.000 | 0.394 | 0.590 |
| 43 | global_frequency_skip | 0.870 | 0.741 | 0.794 | 0.938 | 0.545 | 0.672 |
| 44 | R0 original_capm | 0.773 | 0.647 | 0.794 | 0.688 | 0.606 | 0.620 |
| 44 | high-dimensional R1 raw_frequency_skip | 1.700 | 0.606 | 0.780 | 1.000 | 0.212 | 0.451 |
| 44 | global_stats_skip | 1.088 | 0.662 | 0.777 | 0.688 | 0.636 | 0.638 |
| 44 | global_frequency_skip | 1.477 | 0.633 | 0.784 | 0.750 | 0.515 | 0.588 |

source test 上，global-frequency 在 seed43 BA 略高于 R0（+0.0028），但 seed44 低于 R0（−0.0142），AUC 两 seed 均低于 R0。global-stats 的 seed44 BA 略高（+0.0152），但 seed43 低（−0.0417），同样没有一致趋势。

## 5. Low-rank branch activity

| seed | variant | effective strength | residual/feature RMS | nonidentity fraction |
|---|---|---:|---:|---:|
| 43 | global_stats_skip | 0.01429 | 0.01429 | 1.000 |
| 43 | global_frequency_skip | 0.01433 | 0.01433 | 1.000 |
| 44 | global_stats_skip | 0.01455 | 0.01455 | 1.000 |
| 44 | global_frequency_skip | 0.01441 | 0.01441 | 1.000 |

两种低秩 branch 都产生约 1.4% feature RMS 的稳定非恒等干预。因此没有增益不能归因于模块未接入或 gate 被完全关闭。

## 6. 解释与判定

1. 消除高维 descriptor/3D raw encoder 后，2D intensity stats 和 3D global frequency fractions 都仍可驱动真实 residual；这排除了“模块根本不工作”的解释。
2. 没有一致的 source validation 或 source test 改善。global-frequency 的单 seed BA 小幅波动（+0.016 validation seed44，+0.003 test seed43）与另一 seed 的下降并存，不能被解释为稳健收益。
3. 与高维 raw-frequency R1 相比，低秩 versions 在部分指标上较少恶化，但没有恢复到稳定优于 R0 的水平；因此现有数据不支持“高维 CNN capacity 是 R1 无增益的唯一原因”。
4. 两种 global 分支均显示敏感性/特异性权衡变化；尤其 global-stats seed43 test sensitivity=1.0、specificity=0.394，提示阈值/校准问题仍然存在。

**结论：在当前冻结 source-only 协议下，global mean/std 和 global band-power fraction residual 均未提供可重复的 source gain。该结果不支持继续将其推广到 target evaluation，也不支持把 MRI 全局频域统计作为当前 CAPM backbone 的有效增益机制。**

## 7. 限制与未进行的工作

- 未使用 NACC target，不能推出任何跨域、场强或 scanner 因果结论。
- 逐 source-frequency-environment CE/BA 与 worst-environment risk 仍未由 runner 作为 evaluation artifact 输出；因此没有声称正式的 robustness gate 已通过。
- 原 R2 shuffled descriptor 已停止且不纳入任何比较。
- source test 是描述性诊断，未用于选择 checkpoint 或模型结构。

## 8. 可审计来源

- 低秩实现：`Model/ablation/raw_frequency_skip.py`
- journal runner：`experiments/train_journal.py`
- 单元测试：`tests/test_raw_frequency_skip.py`
- 先前 raw descriptor R1：`docs/PLAN34_RAW_FREQUENCY_SKIP_PARTIAL_RESULTS_2026-08-17.md`
- outputs：`outputs/global_low_rank_skip/ADNI_to_NACC/seed_{43,44}/`
