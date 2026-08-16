# Plan34 C4 机制审计补充计划

状态：补充审计计划，不引入 C7/C8，不改写已冻结的 C4 最终结果。

## 1. 目标与固定边界

本审计回答：C4 的 target 增益是否来自稳定、非恒等、可解释的频率干预，还是仅来自普通 gate 容量或 source 频率环境训练。

- 主方向仍为 `ADNI 1.5T -> NACC 3T`，模型仍为 `layer5_pixel + original_capm`。
- 分别审计 seed 43、44 的冻结 C4 checkpoint。
- 允许使用 `S_train`、`S_val` 和无标签 `T_adapt`。
- `T_test` 保持冻结；不得读取其标签、指标或预测来选择审计结论或新模型。
- 所有 subject 统计均按 subject 聚合，重复扫描不能重复计数。

## 2. 必做审计表

| 优先级 | 审计项目 | 数据与方法 | 必须输出 | 机制用途 |
|---|---|---|---|---|
| P0 | 先验来源与可复现性 | 记录 `S_train`、`T_adapt` 的 subject 数、split/hash、source checkpoint/hash；确认 `target_labels_read=false`、`target_metrics_read=false` | `prior_provenance.json` | 排除先验泄漏和错误 checkpoint/split |
| P0 | 三频段统计 | 分别计算 source 与 `T_adapt` 的 low/mid/high 均值、标准差和频段比例 | `source_target_band_summary.csv` | 复核先验差异的实际来源 |
| P0 | 先验稳定性 | 对 source 与 `T_adapt` 做至少 1000 次 subject-level bootstrap，记录各 `d_k` 的 CI、标准差和频段排序概率 | `prior_bootstrap_stability.json` | 判断 seed 间排序反转是否为抽样波动 |
| P0 | C4 是否实际干预 | 记录 gate 的 `effective_strength`、各频段 attenuation 的均值/中位数/P5/P95 和 identity loss | `gate_activity.csv` | 排除 C4 接近 identity 的情况 |
| P0 | 干预作用位置 | 对 `S_train`、`S_val`、`T_adapt` 统计 gate 前后的频谱比例 | `pre_post_spectrum_summary.csv` | 判断 C4 是否真的改变 feature spectrum |
| P0 | Gate-disabled 对照 | 对同一冻结 C4 checkpoint 做正常 gate forward 与 gate-disabled forward；不重训、不使用 `T_test` 标签 | `c4_identity_forward_comparison.json` | 分离 gate 实际效应与模型容量/初始化效应 |
| P1 | source 频率环境表现 | 对 `original`、`lowpass`、`downsample_resample`、`mild_blur` 分别记录 BA、CE、worst-group risk 和 GroupDRO 权重 | `source_environment_audit.csv` | 判断收益是否主要来自频率环境鲁棒性 |
| P1 | 协变量支持审计 | 统计 source/`T_adapt` 的 age、sex、education 支持范围、缺失率和标准化差异 | `covariate_support_audit.csv` | 判断频谱差异是否混入人口学构成差异 |
| P1 | 频谱差异前后变化 | 在 gate 前后分别计算 source 与 `T_adapt` 的频段差异；只使用 `T_adapt`，不使用 `T_test` 标签 | `pre_post_discrepancy.json` | 判断 gate 是否朝预期的频谱修正方向工作 |

## 3. 先验计算记录

对频段 `k`，记录当前先验使用的标准化差异：

```text
d_k = abs(mu_source_k - mu_target_k)
      / sqrt((sigma_source_k^2 + sigma_target_k^2) / 2)
```

并记录归一化前后的数值、source/target 样本数、bootstrap 分布和频段排序。Gate 的实际衰减、identity loss 和 source/target-adapt 前后频谱必须能够从审计产物复算。

## 4. 判定规则

| 观察结果 | 结论 |
|---|---|
| Gate 接近 identity | 不能把 C4 的 target 增益归因于频率干预 |
| Gate 有明显作用，但行为与 C2 相近 | 增益主要来自普通 spectral gate，而非 target-specific prior |
| 先验排序稳定，gate 按预期改变频谱，且不等同于 C2 | 支持 C4 的频率先验机制 |
| 两个 seed 的先验排序不稳定 | 需要先验 uncertainty shrink/fallback，不应直接增加新模块 |
| 协变量重加权会显著改变 `d_k` | 当前先验可能混入 age/sex/education 构成差异 |

## 5. 严格禁止事项

- 不在本审计中运行 C7/C8 或其他新模型。
- 不用 `T_test` 标签、target 指标或 target 预测选择阈值、频段、强度或新架构。
- 不把 gate activity、非零 attenuation 或单个 BA 增益单独写成机制证据。
- 不将当前 ADNI 1.5T -> NACC 3T 结果解释为 1.5T/3T 因果校正。
