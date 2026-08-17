# Plan 34 全局频率差异图谱设计

## 1. 状态与目的

- 状态：设计与可运行审计工具；不是新的 target 性能结果。
- 主方向：`ADNI 1.5T -> NACC 3T`，任务为 MCI vs AD。
- 目的：把“全局频率差异”从一个直接驱动 gate 的 `d_k`，改为可复现的描述性图谱。图谱只回答差异出现在哪个表示层、频带和 source 支持范围；不直接生成模型强度或场强校正结论。

当前 C4 的问题不是没有任何频率干预，而是 target-specific band ordering 在 seed 间不稳定，且 gate 对 source 与 `T_adapt` 的行为近似相同。因此，下一阶段首先需要测量，而不是用新的 target 全局统计继续驱动每个样本的同一干预。

## 2. 固定协议边界

- 只使用冻结的 scan-filtered manifest：ADNI 1.5T source、NACC 3T target。
- 图谱的 target population 只能是无标签、subject-disjoint 的 `T_adapt`；不得读取 `T_test` 标签、预测或指标。
- source 图谱使用 `S_train`；`S_val` 只作独立诊断，不参与图谱选择。
- 统计独立单位为 subject，每个 subject 只保留 earliest visit。
- 不引入 paired scans、paired loss、pseudo-label、domain adversary 或 target-metric selection。
- 图谱中的跨队列差异可能混合 cohort、site、acquisition、preprocessing、age、sex、education 和 diagnosis 构成；禁止表述为 field-strength causal effect。

## 3. 三种不能混淆的图谱

| 图谱 | 输入 | 回答的问题 | 不能回答的问题 |
|---|---|---|---|
| raw-volume FFT | 预处理后的 MRI volume | 图像层的 acquisition/preprocessing 差异是否集中在某些径向频带 | 模型是否依赖这些频带 |
| frozen feature-map FFT | source `original_capm` 的 spatial map（`layer3`、`layer4`） | 差异是否进入任务 backbone 的空间表示 | 某个频带是否是疾病或 scanner 的因果来源 |
| source-environment FFT | `original`、`lowpass`、`downsample_resample`、`mild_blur` 的 source view | 合成环境覆盖了哪些频谱变化 | 合成环境已经复现真实 target scanner |

`[N, C]` GAP/vector cache 不具备空间维度，不能代替 feature-map FFT。

## 4. P0：原始 prior 的严格复现

先用训练时冻结的 source `original_capm` checkpoint 复算 `frequency_prior.json`，并与原 JSON 逐 band 比较。此步骤必须与“最终 C4 checkpoint 的 pre-gate feature 审计”分开输出：前者复核先验来源，后者只诊断训练后表征。

P0 通过条件：

1. source/target split、checkpoint、config 的 SHA-256 与原产物一致；
2. 复算频段 fractions 与原 prior 在预先声明的数值容差内一致；
3. `target_labels_read=false`、`target_metrics_read=false`；
4. `T_adapt` 与 `T_test` subject digest 不相交。

任何一项失败时，只能修复 provenance/实现，不能解释先验稳定性，更不能训练新模型。

## 5. P1：图谱输出与稳定性分析

审计工具必须输出：

```text
atlas_provenance.json
frequency_prior_recomputed.json
frequency_prior_reproduction.json
per_subject_band_fractions.csv
band_summary.csv
bootstrap_stability.json
source_environment_band_summary.csv
```

所有频段采用 3-D rFFT 的径向、完备划分：low `[0, 0.15)`、mid `[0.15, 0.35)`、high `[0.35, infinity)`。`per_subject_band_fractions.csv` 保存 raw/layer3/layer4 在每个 subject 的 low/mid/high fraction；汇总表只保存均值、标准差和 subject 数。

对每个 layer，使用至少 1,000 次 subject-level bootstrap 记录：标准化绝对差异、95% CI、最高 discrepancy 的 rank probability。额外执行：

1. `T_adapt` 半数 cross-fit：两半分别重算排序；
2. 至少三个独立 source `original_capm` pretrain seed；
3. age/sex/education reweighting 或共同支持子集的敏感性分析；
4. 四个 source frequency environment 与 source original 的同口径图谱。

实现工具提供 P0/P1 的核心 band fraction、bootstrap 与 environment 汇总；多 checkpoint 和 covariate reweighting 由固定清单逐次调用，结果必须并列保存。

## 6. 预注册决策

将一个频带或层位标为“可用于后续 Frequency MixStyle 的候选依据”，必须同时满足：

1. 原始 prior 复现通过；
2. 三个 source pretrain 的最高频带一致；
3. 每个 pretrain 的最高频带 bootstrap rank probability >= 0.80；
4. 两个 `T_adapt` half 的最高频带一致；
5. covariate 敏感性分析不改变该结论；
6. target feature fractions 落在 source frequency-environment 图谱定义的支持范围内，或明确报告 fallback fraction。

若任何条件不满足，结果是 **NO-GO for target-guided band selection**。仍可将图谱用于描述和选择保守的 source-only robustness controls，但不能再次把 target discrepancy 变成全局 gate 系数。

## 7. 运行入口

`experiments/global_spectral_atlas.py` 接受冻结 config、source split、target split、source checkpoint、输出目录和待分析 stages。它只访问 `S_train` 与 `T_adapt` 的图像/subject ID；raw stage 不经过模型，feature stage 从冻结 source model 提取空间 feature map。

示例（真实路径必须来自已核验的冻结产物）：

```text
<PYTHON> experiments/global_spectral_atlas.py \
  --config <RESOLVED_CONFIG> \
  --direction ADNI_to_NACC \
  --source-split <SOURCE_SPLIT_JSON> \
  --target-split <TARGET_SPLIT_JSON> \
  --source-checkpoint <ORIGINAL_CAPM_CHECKPOINT> \
  --reference-prior <ORIGINAL_FREQUENCY_PRIOR_JSON> \
  --output-dir <ATLAS_OUTPUT> \
  --stages raw layer3 layer4 \
  --n-bootstrap 1000
```

该命令的完成只证明图谱工具和产物边界可运行，不证明频率假说、训练收益或跨队列泛化。
