# Plan 34 Raw Frequency Feature Skip：阶段性 Source-Only 结果

- 更新：2026-08-17 UTC+8
- 实验状态：R0 已复用；R1 已完成；R2 尚未实施
- 固定方向：ADNI 1.5T → NACC 3T
- 训练与评价边界：本报告只使用 ADNI source train / validation / test。没有构造、读取、推理或选择 NACC target rows。

## 1. 目的与冻结协议

本阶段检验的是 source-only robustness candidate，而不是 target-domain 结果。R1 在 `layer5_pixel + original_capm(age, sex, education)` 基线上加入原始 MRI 的紧凑 rFFT residual：`log(1+|F|), cos(angle(F)), sin(angle(F))` 被 adaptive pooled 到 `8×8×8`，经小型 encoder 后注入 layer3。残差强度受 `0.15 * tanh(theta)` 约束。

固定条件：ResNet10（`layers=[1,1,1,1]`）、`layer5_pixel` preset、50 epochs、batch size 4、AdamW `1e-4`、weight decay `1e-4`；seeds 43/44 和冻结 scan-filtered ADNI split。R0 直接复用协议一致的既有 `original_capm` checkpoint 与 metrics；R1 使用 layer3、grid `8×8×8`、hidden channels 32、max residual 0.15、gate initialization 0.1。

## 2. 执行状态

| 实验 | seed43 | seed44 | 说明 |
|---|---|---|---|
| R0 `original_capm` | 复用 | 复用 | 既有、协议一致 source baseline |
| R1 `raw_frequency_skip` | 完成 | 完成 | source-only；50 epochs 正常退出 |
| R2 shuffled descriptor | 未实施 | 未实施 | 当前代码不消费 `shuffle_raw_descriptor`；已停止无效重跑，不能记作负对照 |

R1 所有输出均在独立工作区 `dual_shift_raw_frequency_skip` 内，未覆盖既有 UDA artifacts。

## 3. R1 raw skip activity

| seed | split | effective strength | residual / feature RMS | nonidentity fraction | raw low / mid / high fraction |
|---|---|---:|---:|---:|---|
| 43 | S_val | 0.01384 | 0.01384 | 1.000 | 0.9111 / 0.0709 / 0.0180 |
| 44 | S_val | 0.01415 | 0.01415 | 1.000 | 0.9111 / 0.0709 / 0.0180 |

该分支有可测、跨 seed 一致的非恒等影响：相对 feature RMS 约 1.38%–1.41%。这只证明 skip branch 被激活，并不证明其诊断或跨域价值。

## 4. Clean source validation：R0 vs R1

| seed | variant | CE | BA | AUC | sensitivity | specificity | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 43 | R0 original_capm | 0.630 | 0.609 | 0.680 | 0.625 | 0.594 | 0.590 |
| 43 | R1 raw_frequency_skip | 1.224 | 0.594 | 0.662 | 0.562 | 0.625 | 0.582 |
| 44 | R0 original_capm | 0.961 | 0.641 | 0.666 | 0.688 | 0.594 | 0.614 |
| 44 | R1 raw_frequency_skip | 1.896 | 0.547 | 0.689 | 0.938 | 0.156 | 0.390 |

R1 相对 R0 的 clean S_val BA 变化为 seed43 −0.0156、seed44 −0.0938；CE 增加 0.595 与 0.936。特别是 seed44 出现高 sensitivity / 低 specificity 的阈值失衡。因此，尽管 AUC 在 seed44 略升，R1 没有展示稳定的 clean source preservation。

## 5. Source test（描述性，不用于 checkpoint 选择）

| seed | variant | CE | BA | AUC | sensitivity | specificity | macro-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 43 | R0 original_capm | 0.596 | 0.739 | 0.809 | 0.750 | 0.727 | 0.718 |
| 43 | R1 raw_frequency_skip | 1.031 | 0.662 | 0.786 | 0.688 | 0.636 | 0.638 |
| 44 | R0 original_capm | 0.773 | 0.647 | 0.794 | 0.688 | 0.606 | 0.620 |
| 44 | R1 raw_frequency_skip | 1.700 | 0.606 | 0.780 | 1.000 | 0.212 | 0.451 |

R1 在两个 source test seed 上均低于复用 R0 的 BA（−0.0767、−0.0407），同时 CE / calibration 变差。source test 仅作阶段性诊断，不能替代预注册 S_val selector。

## 6. 产物与可追溯性

R1 产物包括每个 seed 的 `best_checkpoint.pt`、`last_checkpoint.pt`、`journal_metrics.json`、source validation/test predictions，以及训练日志。

| seed | config SHA-256 | best checkpoint epoch | checkpoint SHA-256 | metrics SHA-256 |
|---|---|---:|---|---|
| 43 | `d668685238815600c197d5204c0baec327b0746cf6e6ab64b338952821ffa1da` | 42 | `e35fe1bb8d885681772f91a6b38619b3414653537d593d87014ccacec3d6da8c` | `3f7c3a7dee7e3c4269a1c15494d39d0e7c23324964858fe67848479a9a228428` |
| 44 | `dbd7454950210b53f67e7fd9617e054da22b6f92ce3539972c3cb0b3be3397b6` | 50 | `8ec6f8f6c75677edecb82d9d97f4135e150b96fc0babbfbe8586cc586c24f657` | `64146d81028a3fca2cf8bf24a4a0ae0de0164b323997f86e8db9583867bb23c0` |

## 7. 审计缺口与协议判定

设计要求 R1 promotion 前报告四个预注册 source frequency environments（original、lowpass、downsample_resample、mild_blur）的逐环境 CE/BA 与 worst-environment risk。当前 runner 将这些环境用于训练中的随机 GroupDRO assignment，但没有为 evaluation 写出逐环境结果；`journal_metrics.json` 也没有 source manifest hash 字段，且 raw skip 不包含 checkpoint hash。

此外，设计引用“R0 degradation tolerance”，但设计文件没有给出数值。即使忽略这一未量化阈值，R1 的 S_val / source-test BA 和 CE 也没有提供稳定的 source-preservation 信号。

**阶段性判定：R1 已证明 raw residual 是非零且稳定的 feature intervention，但不满足将其解释为 source robustness gain 的条件；R2 尚不能作为已完成的 negative control。因而当前不进入 target evaluation，也不声称 raw frequency skip 的诊断或跨域改进。**

## 8. 后续工作（不自动启动）

继续 R2 前需要先补足可测试实现与协议：

1. 实现并单元测试真正的 per-sample descriptor shuffle，记录 permutation / seed，确保不是普通 R1 重跑；
2. 在 frozen source evaluation 中为四个 frequency environments 分别记录 CE、BA 与 worst-environment risk；
3. 将 source manifest、checkpoint 与 config 的 hashes 写入同一 audit artifact；
4. 明确 R0 clean S_val degradation tolerance，并以该锁定阈值判定 R1/R2。
