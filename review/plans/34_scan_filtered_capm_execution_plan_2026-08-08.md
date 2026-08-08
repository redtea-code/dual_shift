# CAPM-SF-P1：Scan-Filtered 任务化执行计划

日期：2026-08-08
状态：待 E0 manifest gate
协议版本：`scan_filtered_v1_2026-08-08`
适用分支：`codex/scan-filtered-loader`

## 1. 适用范围与历史边界

本计划是后续 CAPM、IE-CAPM、原始 CAPM 对照以及 feature-scale / patch-table
交互消融的唯一执行计划。它取代旧计划中“删除 73 个 paired subject”的未来执行
含义，但不改写那些计划或已有结果。

禁止把本计划产生的结果与旧 `subjects_all_paired` 协议结果混为同一数据版本。
每个输出必须保存 protocol version、git commit、task YAML hash 和两个 manifest hash。

## 2. 冻结数据协议

```text
ADNI_1.5T_scan_filtered: 保留所有合格 field_strength == 1.5T 的 scan
                           删除 3T、unknown、other 的 scan
                           不因 subject 曾有 3T 记录而删除其 1.5T scan
NACC_3T:                 保留所有合格 field_strength == 3T 的 scan
```

过滤在任何 split 前完成。使用 `write_filtered_manifest` 生成独立 frozen manifest，
再由 `ScanFilteredManifestDataset` 加载。它默认不返回 acquisition，因此 field strength、
site、vendor、protocol 和 `paired_origin` 不能进入模型输入。

场强筛选并不使队列其它采集差异可识别为因果效应。本计划只研究固定外部验证协议下的
分类与融合稳定性。

## 3. 任务、方向和 split

| task | label mapping | directions |
|---|---|---|
| `MCI_vs_AD` | `{2: 0, 3: 1}` | `ADNI_to_NACC`, `NACC_to_ADNI` |
| `CN_vs_AD` | `{1: 0, 3: 1}` | `ADNI_to_NACC`, `NACC_to_ADNI` |

每个 task 独立生成 resolved YAML、source split、输出目录和统计表。source 在过滤后按
subject 做 60/20/20 划分，`split_seed=42`；同一 subject 的所有保留 scan 必须落在同一
partition。target 完整保留，绝不用于训练、early stopping、checkpoint、阈值或调参。

所有 table-aware 模型固定使用 `[age, sex, education]`，缺失处理和 source-train 标准化
完全一致。image-only 不使用表格。输入几何固定为经 E0 抽样核验的 `160 x 196 x 160`。

## 4. 分阶段执行

### E0：Manifest 与接口门

1. 运行 loader 单测、CAPM 单测和 `py_compile`。
2. 生成 ADNI 1.5T 与 NACC 3T manifest 及 audit JSON。
3. 验证 ADNI 无 3T/unknown，NACC 无非 3T；保留 paired-origin subject 的 1.5T 行。
4. 验证 manifest protocol marker、hash、subject 数、类别数、NIfTI geometry。
5. 验证 source train/val/test subject 不重叠，且新 YAML 不含 `claim.exclude_subjects_json`。

任何一项失败即 `NO-GO`，不得启动真实训练。

### E1：每 task x direction 的 smoke

从任务 YAML 生成 smoke YAML，只允许改变：`epochs=1`、`bootstrap_samples=10`、输出根目录。
运行 `image_only`、`capm`、`transformer_cross`（若该 launcher 对应 scale ablation）；若当前
journal launcher 仅支持 APIC variants，则运行其预注册 `ce_x`、`mixstyle_x`、`apic_v3_2_x`。

检查 dataloader 样本构成、input shape、source-val checkpoint、NaN/Inf、collapse guard、
预测文件和 target 未参与选模。E1 不作性能主张。

### E2：结构筛选（source-only）

对每个 task 和方向，固定 `seed=42`，仅用 source validation BA + collapse guard 筛选：

| family | variants |
|---|---|
| 基线 | `image_only`, `capm`, `original_capm` |
| 尺度 | `layer3_patch2`, `layer4_pixel`, `layer5_pixel` |
| 交互 | `conv_gate`, `transformer_self`, `transformer_cross` |

严格比较遵循同尺度、同训练预算、同三变量、同 source split。注意力权重不构成因果解释；
`transformer_cross` 的关键比较是其相对同尺度 `transformer_self` 的差值。

### E3：确认性外部评估

对 E2 每个 task 的唯一冻结候选，在两个方向、seeds `42/43` 上重跑。主终点为：

```text
external target + subject_mean + balanced accuracy
```

同时报告 AUROC、sensitivity、specificity、macro-F1、Brier、ECE、200 次 subject-level
bootstrap CI、逐 subject 预测和类别塌缩审计。不得汇总两个方向后隐藏某一方向失败。

## 5. 结果解释和停止条件

结果只支持“在该 scan-filtered 外部验证协议中，某融合结构是否带来可复现的预测差异”。
不得声称学习了 1.5T 到 3T 的超分辨率、场强因果响应或普适域泛化。

以下任一情况停止该 task/direction 的扩展：mixed-strength manifest、paired holdout 配置残留、
subject leakage、target 选模、类别塌缩、seed 间方向相反，或 manifest/YAML/commit 不一致。

## 6. 多机器交付

使用 `docs/SCAN_FILTERED_MULTI_MACHINE_EXPERIMENT_PLAN.md` 分配机器。每个 worker 只交付
audit、resolved YAML、hash、split manifest、metrics、subject-level predictions 和状态记录；不交付
原始影像。主控合并前先验证 task、direction、seed、commit 和 manifest hash 完全一致。
