# APIS v2 实验主表规范

版本：`1.0`  
适用任务：3090 节点 CN vs AD、5090 节点 MCI vs AD 及后续 APIS v2 Claim E1 实验  
规范状态：强制执行（`MUST`）；本文件与实验协议 r2 配套使用

## 1. 目的与现状

本规范用于统一 3090 与 5090 实验主表的结构、命名、缺失值语义和质量检查。当前两类主表存在结构差异：

- 3090 表是“每行一个 `seed × direction × variant × split`”的长表，指标列直接展开为 `subject_*` 和 `scan_*`。
- 5090 表是“每行一个任务/seed/方向/variant”，将 `target__*`、`source_val__*`、`source_test__*` 和 `fs_*` 拼接为宽表。
- 两者对 CI 的顺序、样本量字段、方向显示名和未完成运行的表示方式不完全一致。

从本规范版本开始，原始产物可以保留，但用于汇总、比较、绘图和论文的表必须遵循下面的规范化结构。未经规范化的表不得直接作为论文主结果来源。

## 2. 文件分层与命名

每个任务必须至少保存以下三类文件：

| 层级 | 文件命名 | 内容 | 要求 |
|---|---|---|---|
| 原始记录 | `metrics_raw_<task>_<date>.csv` | 运行器直接导出的原始字段 | 不改写；保留原始列名和空值 |
| 规范长表 | `metrics_long_<task>_<date>.csv` | 一行一个评估单元 | 供统计、比较和绘图使用 |
| 主表 | `metrics_main_<task>_<date>.md` | 人工可读的摘要表 | 只能从规范长表生成 |

其中 `<task>` 只允许：`cn_ad`、`mci_ad`。日期使用 ISO 格式 `YYYY-MM-DD`。同一批结果的三个文件必须使用同一日期和同一 `run_batch_id`。

推荐目录：

```text
review/records/apis_v2/3090/
review/records/apis_v2/5090/
outputs/journal/dual_shift_apis_v2/<claim_root>/e1/
```

## 3. 规范长表：唯一主数据模型

规范长表每行代表一个唯一评估单元：

```text
task × seed × direction × variant × split × aggregation × field_strength
```

当某个维度不适用时，必须填 `not_applicable`，不能留空。唯一键为：

```text
task, seed, direction, variant, split, aggregation, field_strength
```

### 3.1 标识字段

| 字段 | 类型 | 允许值/格式 | 说明 |
|---|---|---|---|
| `task` | string | `CN_vs_AD` / `MCI_vs_AD` | 任务名，大小写固定 |
| `seed` | integer | `42`, `43`, `44`, `45`, `46` | 训练随机种子 |
| `direction` | string | `ADNI_to_NACC` / `NACC_to_ADNI` | 方向显示名，唯一标准写法 |
| `variant` | string | `ce_only`, `mixstyle`, `metadata`, `metadata_xda`, `apis_v2` | 方法名；不得使用别名 |
| `split` | string | `target`, `source_val`, `source_test` | 评估分区 |
| `aggregation` | string | `subject_mean` / `scan` | 指标聚合层级 |
| `field_strength` | string | `1.5T`, `3T`, `pooled`, `not_applicable` | 场强分层；未分层时用 `pooled` |
| `status` | string | `complete`, `running`, `failed`, `not_started` | 运行状态 |
| `present` | boolean | `true` / `false` | 是否存在可读取产物；仅原始兼容层使用，长表以 `status` 为准 |

方向 slug 只允许在兼容原始字段中使用：`adni_to_nacc`、`nacc_to_adni`。规范主表不得同时出现两种写法。

### 3.2 协议与复现字段

| 字段 | 类型 | 要求 |
|---|---|---|
| `claim_protocol` | string | 例如 `apis_v2_claim_e1`、`apis_v2_claim_e1_mci_ad` |
| `claim_protocol_revision` | integer | 当前确认性实验必须为 `2` |
| `split_seed` | integer | r2 默认 `42`；若改变必须单独注册 |
| `training_seed` | integer | 与 `seed` 一致时仍需显式记录 |
| `initialization_seed` | integer | 必须显式记录，不能由 `seed` 推断 |
| `config_hash` | string | 配置指纹，建议 SHA-256 |
| `metrics_path` | string | 仓库相对路径或稳定外部路径 |
| `git_commit` | string | 产生结果的完整 commit |
| `run_batch_id` | string | 同一批运行的唯一标识 |

### 3.3 指标字段

所有指标列使用统一的基础名，不再使用 `target__accuracy` 这类 split 前缀展开格式。split 已由 `split` 字段表达。

| 基础字段 | 含义 | 范围/单位 |
|---|---|---|
| `accuracy` | Accuracy | `[0,1]` |
| `balanced_accuracy` | Balanced Accuracy，主终点 | `[0,1]` |
| `auc` | ROC-AUC | `[0,1]` |
| `macro_f1` | Macro-F1 | `[0,1]` |
| `sensitivity` | Sensitivity | `[0,1]` |
| `specificity` | Specificity | `[0,1]` |
| `brier` | Brier score | `>=0`，越低越好 |
| `ece` | Expected Calibration Error | `>=0`，越低越好 |
| `group_gap` | 最大组间差异 | `>=0` |
| `worst_group_auc` | 最差组 AUC | `[0,1]` |
| `loss` | 记录的损失 | `>=0`，需注明损失类型 |
| `n_subjects` | subject 数 | 非负整数 |
| `n_scans` | scan 数 | 非负整数 |

所有置信区间统一使用独立列：`<metric>_ci_low`、`<metric>_ci_high`。禁止使用 `ci_lo/ci_hi`、`ci_low/ci_high` 混用，也禁止把 high 放在 low 前面作为唯一顺序。

### 3.4 场强字段

场强统计使用 `field_strength` 维度，而不是把 `fs_1.5T_accuracy`、`fs_3T_accuracy` 展开为大量列。每个场强行必须至少包含：

```text
field_strength, accuracy, balanced_accuracy, auc, macro_f1,
sensitivity, specificity, brier, ece, group_gap,
worst_group_auc, n_subjects, n_scans
```

如果原始运行器仍输出 `fs_1.5T_*` 或 `fs_3T_*`，导入规范长表时必须拆行，并将前缀转换为 `field_strength` 值。

## 4. 空值、失败和未完成运行

空值不能同时承担“未运行”“运行失败”“指标不适用”三种含义。规范如下：

| 情况 | `status` | 指标值 | 备注 |
|---|---|---|---|
| 已完成且指标有效 | `complete` | 数值 | 可进入汇总 |
| 正在运行 | `running` | `NA` | 不得进入均值、CI 或排名 |
| 已运行但失败 | `failed` | `NA` | 必须在 `failure_reason` 记录原因 |
| 计划中尚未运行 | `not_started` | `NA` | 不得解释为随机水平 |
| 指标不适用 | `complete` | `NA` | `metric_applicability=not_applicable` |

推荐使用字符串 `NA` 作为 CSV 中的缺失标记；不得使用空字符串、`0`、`null` 或 `-` 混用。`0.5000` 只有在模型确实输出随机水平结果时才可作为数值记录，不能用来代替缺失。

## 5. 3090 与 5090 的迁移规则

### 5.1 3090 CN vs AD

当前 3090 长表已经接近规范形式，需做以下统一：

1. `direction` 由 `adni_to_nacc`/`nacc_to_adni` 转为 `ADNI_to_NACC`/`NACC_to_ADNI`。
2. `subject_n` 统一为 `n_subjects`，`scan_n` 统一为 `n_scans`。
3. `subject_*` 和 `scan_*` 拆成 `aggregation=subject_mean` 与 `aggregation=scan` 两行。
4. `subject_balanced_accuracy_ci_lo/hi` 转为 `balanced_accuracy_ci_low/high`。
5. 原始 `status` 保留；缺失 seed/variant 不得删除，需生成 `not_started` 行。

### 5.2 5090 MCI vs AD

当前 5090 宽表需转换为规范长表：

1. `task=MCI_vs_AD` 固定写入每行。
2. `direction` 使用 `ADNI_to_NACC` 或 `NACC_to_ADNI`，`direction_slug` 仅保留在兼容原始层。
3. `target__*`、`source_val__*`、`source_test__*` 拆为 `split` 行。
4. `target__fs_1.5T_*` 和 `target__fs_3T_*` 拆为 `field_strength` 行；未带 `fs_` 的指标填 `field_strength=pooled`。
5. `target__balanced_accuracy_ci_high/low` 统一为 `balanced_accuracy_ci_low/high`。
6. `present=false` 的行转为 `status=not_started` 或 `failed`；必须依据运行日志区分，不能只依据布尔值猜测。
7. `target__n` 转为 `n_scans`，`target__n_subjects` 转为 `n_subjects`。

## 6. 汇总与统计规则

1. 论文主终点固定为 `split=target`、`aggregation=subject_mean`、`metric=balanced_accuracy`。
2. 跨 seed 汇总只能对 `status=complete` 的行计算；必须报告完成 seed 数和计划 seed 数，例如 `4/5`。
3. 不得把 `source_val` 或 `source_test` 与 `target` 混合计算主结果。
4. 不得把 `scan` 指标替代 `subject_mean` 主终点。
5. 每个方向分别汇总，再报告双向平均；不得先把两个方向的所有 scan 拼接后计算一个结果。
6. 相对基线差值必须使用同一 `task × seed × direction × split × aggregation` 配对行，至少报告 `delta_mean`、`delta_std`、胜负计数和 subject-level 配对 bootstrap 95% CI。
7. 未完成矩阵必须在主表中显示完成度，不得从表中删除空缺组合后让表格看起来完整。

## 7. 主表 Markdown 规范

主表只呈现规范长表的摘要，不直接复制 55 列或 220 列宽表。推荐列：

```text
task | split | aggregation | direction | variant | n_seeds_complete |
balanced_accuracy_mean | balanced_accuracy_std |
balanced_accuracy_ci_low | balanced_accuracy_ci_high |
auc_mean | macro_f1_mean | sensitivity_mean | specificity_mean |
n_subjects_total | status_summary | source_file
```

每个方向和 split 使用独立小节，表头顺序固定。主表底部必须包含：

- 数据生成 commit、协议 revision、split seed；
- 计划矩阵与完成矩阵；
- 主终点定义；
- 缺失/失败任务清单；
- 原始 CSV 和规范长表路径。

## 8. 自动质控门槛

发布或合并主表前必须通过以下检查：

- [ ] 唯一键无重复；
- [ ] `task`、`direction`、`variant`、`split`、`aggregation` 使用允许值；
- [ ] 所有概率指标位于 `[0,1]`，`n_subjects/n_scans` 为非负整数；
- [ ] `ci_low <= point_estimate <= ci_high`，且 `ci_low <= ci_high`；
- [ ] `status != complete` 的行没有进入均值、标准差或 CI；
- [ ] 每个计划的 seed×方向×variant 组合都有一行；
- [ ] `field_strength` 统计与 `pooled` 统计使用同一 split 和 aggregation；
- [ ] 3090/5090 的主终点均可通过 `task + split + aggregation + metric` 唯一定位；
- [ ] 每个结果都能回溯到 `metrics_path`、`config_hash`、`git_commit`；
- [ ] 主表与 CSV 行数、完成度和关键均值一致；
- [ ] 失败运行有 `failure_reason`，中断运行不得伪装成 `not_started`。

未通过上述任一项时，结果只能标记为 `under_review`，不得进入论文主结果或 claim gate。

## 9. 最小规范示例

```csv
task,seed,direction,variant,split,aggregation,field_strength,status,balanced_accuracy,balanced_accuracy_ci_low,balanced_accuracy_ci_high,n_subjects,n_scans
CN_vs_AD,42,ADNI_to_NACC,apis_v2,target,subject_mean,pooled,complete,0.8232,0.7928,0.8520,960,960
MCI_vs_AD,42,ADNI_to_NACC,mixstyle,target,subject_mean,pooled,complete,0.6363,0.5962,0.6836,28,527
MCI_vs_AD,46,NACC_to_ADNI,apis_v2,target,subject_mean,pooled,not_started,NA,NA,NA,NA,NA
```

示例中的最后一行必须保留：它说明该组合属于计划矩阵但尚未完成，不能被误读为缺失记录或随机水平。
