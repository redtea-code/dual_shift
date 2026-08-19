# IE-CAPM / APIC V3_2 修订数据协议：ADNI 1.5T Scan-Filtered

版本：1.0
日期：2026-08-08
状态：待生成 frozen manifest 后执行
适用任务：`MCI_vs_AD`
适用模型：image-only、CAPM、IE-CAPM 及其预注册消融

## 1. 修订目的

本协议替代“排除所有 1.5T/3T 配对 subject”的执行方案，但不修改旧协议及其历史结果。修订的目标是构造方向明确的场强协议：

```text
ADNI 1.5T scan-filtered  ->  NACC 3T
NACC 3T                  ->  ADNI 1.5T scan-filtered
```

对于同时拥有 1.5T 和 3T 扫描的 ADNI subject，只删除其 3T scan，保留其合格的 1.5T scan；不得因为该 subject 的历史记录中出现过 3T 而删除整个 subject。

## 2. 数据事实与命名边界

根据数据认知文档，ADNI 原始记录包含 1.5T、3T 及少量 unknown/other，NACC 为 3T。故本协议的 ADNI 数据集应称为：

> `ADNI_1.5T_scan_filtered`

不能称为“从未接受过 3T 的纯 1.5T subject 队列”。部分保留 subject 曾有 3T 记录，但这些记录不会进入本协议的任何 manifest、split 或模型输入。

本协议研究的是 scan-level acquisition composition 对跨队列分类的影响，不声称识别场强的因果效应，也不声称解决所有样本选择偏差。

## 3. 纳入与排除规则

### 3.1 ADNI

对每一条 scan 记录执行以下规则，顺序固定：

1. 保留 `field_strength == 1.5T` 的合格 MRI scan。
2. 删除 `field_strength == 3T` 的所有 scan。
3. 删除 `field_strength` 缺失、unknown 或 other 的 scan；不得将其映射为 1.5T。
4. 继续执行既有的图像可读性、预处理成功、诊断匹配和表格变量完整性规则。
5. 不使用 `subjects_all_paired` 做 subject-wide 排除。

### 3.2 NACC

保留既有协议中的全部合格 NACC 3T scan，并执行同样的图像质量、诊断匹配、表格变量和预处理成功检查。NACC 不进行伪造的 1.5T 筛选。

### 3.3 记录级决策

每条被保留或删除的 scan 必须在 manifest 中有可审计的 `inclusion_reason`，至少包含：`kept_1p5T`、`dropped_3T`、`dropped_unknown_strength`、`dropped_bad_image`、`dropped_missing_table`、`dropped_label_mismatch` 等。删除 3T 只影响本协议，不得删除原始数据文件。

## 4. Subject-level 划分

### 4.1 划分时机

必须按以下顺序执行：

```text
原始 manifest
  -> scan-level field filter
  -> 图像/标签/表格质量过滤
  -> subject-level 6/2/2 split
  -> frozen source manifest + external target manifest
```

不得先按原始 subject 集合划分，再在 partition 内删除 3T；否则不同 partition 的有效样本构成会被事后改变。

### 4.2 比例与随机性

- `split_seed = 42`。
- 对 source subject 按 subject 划分 train/validation/test = 60%/20%/20%。
- 同一 subject 的所有保留 scan/visit 必须位于同一 partition。
- 6/2/2 是 subject 比例，不是 scan 比例。
- target 完整保留，不参与训练、early stopping、checkpoint 选择、阈值调节或超参数选择。
- 正式结果至少运行 seeds `42` 和 `43`；扩展结果可增加预注册 seed，但不得事后挑选。

方向定义：

| direction | source | target |
|---|---|---|
| `ADNI15_to_NACC3` | ADNI 1.5T scan-filtered | 全部合格 NACC 3T |
| `NACC3_to_ADNI15` | 全部合格 NACC 3T | ADNI 1.5T scan-filtered |

## 5. 表格输入与模型公平性

所有表格模型固定使用三个变量，顺序不可变：

```text
[age, sex, education]
```

缺失处理、归一化、列顺序、编码和 CAPM 接口在四类模型间完全一致。不得将 `field_strength`、manufacturer、site、sequence、cohort 或 paired-origin 标记作为模型输入。image-only 不接收任何表格变量。

## 6. 评估与主要终点

主要终点：

```text
external target + subject_mean aggregation + balanced accuracy
```

同时报告 AUROC、sensitivity、specificity、校准指标和 demographic subgroup gap。重复 scan 先在 subject 内求均值，再进入总体指标；bootstrap 以 subject 为重采样单位。

每个方向、模型和 seed 必须输出：

- target 全体结果；
- 按 1.5T/3T 的场强分层结果（仅用于审计）；
- source validation BA、checkpoint、collapse guard 记录；
- subject-level prediction 和 manifest hash。

不得使用 target 指标挑选 checkpoint 或反向修改筛选规则。

## 7. 配对 subject 的处理与审计

配对 subject 的 1.5T scan可以进入 source 或 target，但其 3T scan不进入本协议。为确认过滤确实生效，manifest 必须记录：

- `n_subjects_with_retained_1p5T_and_dropped_3T`；
- `n_subjects_never_observed_3T`；
- 两类 subject 的 scan 数、诊断构成和人口学描述；
- 每个 subject 是否在 source/target 及其 partition。

这些字段只用于数据审计和外部有效性说明，不进入模型、不用于加权、不用于后处理校正。

特别禁止：

- 将同一 subject 的 1.5T 与被删除的 3T 配成训练约束；
- 用被删除的 3T 选择模型或计算 target 指标；
- 将 paired-origin 当作隐藏环境标签；
- 将本协议结果与旧的 73-subject 排除结果混写为同一实验。

## 8. Frozen manifest 与可复现性

新协议必须生成独立 manifest 和 split 文件，例如：

```text
data/claim/adni_1p5t_scan_filtered_manifest.json
data/claim/adni_1p5t_scan_filtered_split_seed42.json
data/claim/adni_1p5t_scan_filtered_split_seed43.json
```

每次运行保存：原始输入文件 hash、过滤脚本版本、过滤前后 scan/subject 计数、保留/删除原因计数、split hash、resolved config、git commit、模型 checkpoint 和 subject-level predictions。原始旧 manifest 不得覆盖。

## 9. 自我审核

### 9.1 已通过的协议一致性检查

- 场强方向明确：ADNI 仅 1.5T，NACC 为 3T。
- 配对 subject 的 1.5T 被保留，符合当前研究决定。
- 3T 删除发生在 split 前，避免 partition 后样本构成漂移。
- split 仍为 subject-level 6/2/2，避免重复 scan 泄漏。
- 三个表格变量固定，未引入 scan 参数 shortcut。
- target 未参与训练和模型选择。
- 旧协议和旧结果保持可复现，不会被新 manifest 覆盖。

### 9.2 仍需承认的限制

“不把样本选择差异作为模型因素”并不等于该差异不存在。保留 paired-origin subject 可能带来可观测队列构成差异，因此论文中必须报告计数和描述性比较，但不需要将其作为输入或校正变量。

此外，本协议不能单独证明 1.5T/3T 的因果场强效应；它只提供一个清晰、可复现的跨场强外部验证设定。若未来要声称场强机制，应另行设计同 subject 配对或多协议实验。

### 9.3 开放门槛

正式 E2 运行前必须通过：

1. 过滤后 ADNI manifest 中不存在 3T/unknown 行；
2. source train/validation/test subject 集合互不相交；
3. target subject 不与 source subject 重叠；
4. 被删除 3T 行不出现在任何 dataloader、预测文件或指标文件；
5. 过滤前后计数、paired-origin 描述和 manifest hash 可复核；
6. image-only、CAPM、IE-CAPM 使用完全相同的 manifest、split、训练预算和 checkpoint 规则。

任一门槛失败时，实验标记为 `NO-GO`，不得解释性能结果。

## 10. 与旧协议的关系

旧的 `IE_CAPM_APIC_V3_2_ALIGNED_EXPERIMENT_PROTOCOL.md` 仍是历史协议，用于复现已完成或已归档结果；本文件是新的 scan-filtered 研究协议。两者必须使用不同的实验 ID、manifest hash 和结果目录，禁止直接横向比较而不注明数据协议差异。

## 11. 代码接口

训练配置启用新 loader 时使用独立配置段，并删除旧的
`claim.exclude_subjects_json`：

```yaml
scan_filtered_protocol:
  enabled: true
  version: scan_filtered_v1_2026-08-08
  root: ./scan_manifests_filtered
  files:
    ADNI: ADNI_1p5T_scan_filtered_manifest.csv
    NACC: NACC_3T_scan_filtered_manifest.csv
  expose_acquisition: false
```

`data.scan_filtered_loader.write_filtered_manifest` 负责生成 manifest；
`data.journal_dataset.build_journal_dataset` 会根据该配置自动调用
`ScanFilteredManifestDataset`。新 loader 默认不向模型返回 acquisition 字段，
但会在 manifest 和 audit JSON 中保留 paired-origin 供审计。
