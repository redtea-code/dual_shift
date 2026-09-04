# DualShift 多模态 UDA Phase 0 数据准备说明

更新时间：2026-08-29
状态：服务器端实验准备要求；不是独立实验结果
主方向：`ADNI_to_NACC`，scan-filtered MCI vs AD

正式实验计划：[`DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md`](DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md)

## 1. 目的

这份说明只保留真实实验必须的数据合同，不把数据准备扩展成一套繁杂的审计流程。正式实验主要在服务器上进行，因此本文不绑定任何本地目录、机器路径或文件名。

核心要求只有一个：为 MRI+table UDA 生成可复现的 frozen multimodal manifest，并让 source-only、UDA 和消融实验使用同一份 split 与聚合规则。

## 2. 默认数据合同

第一版 table 固定为：

```text
t = [age, sex, education]
```

服务器端 manifest 将原始数据映射为统一语义字段：

```text
subject_id, scan_id, image_path, scan_date,
age, sex, education,
age_missing, sex_missing, education_missing,
cohort, field_strength, protocol_version, split
```

`field_strength`、site、manufacturer 和 sequence 只用于协议分层或解释分析，不进入默认模型输入。MMSE、CDR、FAQ、直接 diagnosis 字段、APOE、PET/CSF 和 MRI-derived ROI 暂不进入第一版 table；如需使用，另行注册为额外模态或独立消融。

## 3. UDA 访问规则

| 集合 | 可访问信息 | 用途 |
|---|---|---|
| `S_train` | MRI、table、source `y` | 任务训练和 source relation constraint |
| `S_val` | MRI、table、source `y` | checkpoint、权重和停止点选择 |
| `S_test` | MRI、table、source `y` | checkpoint 固定后的 source 报告 |
| `T_adapt` | 无标签 MRI、table | UDA 更新和无标签关系约束 |
| `T_test` | MRI、table，最终读取 `y` | 冻结协议后的 target 评估 |

必须满足：

1. 所有 split 按 subject，不按 scan row 随机切分；
2. `T_adapt` 与 `T_test` subject-disjoint；
3. source 的 imputation/scaling 只在 `S_train` 拟合；
4. target label、prediction、metric 和模型排名不能进入 adaptation/selector；
5. `ADNI_to_NACC` 为主方向，`NACC_to_ADNI` 单独作为 unsupported stress test，不与主方向平均。

## 4. 最小服务器端检查

在正式训练前只需确认以下项目：

- MRI path、table row 和 subject/date join 可复现；
- age/sex/education 的缺失标志、编码和 source-fitted preprocessing 固定；
- `S_train/S_val/S_test/T_adapt/T_test` 无 subject overlap；
- target adaptation loader 不返回 target `y`、prediction 或 metric；
- manifest digest、resolved config、代码 commit、checkpoint hash 和 prediction artifact 能随结果保存。

这些是实验可复现条件，不是需要单独发表的机制实验。完成后直接进入论文主表和消融矩阵。

## 5. 论文主表中的实验位置

数据准备完成后，以下实验直接作为同一论文的主表或主消融表运行，不再被隔离成“只做审计的测试”：

| 组别 | 代表实验 | 论文用途 |
|---|---|---|
| MRI baseline | MRI-only source-only | 主任务基线 |
| multimodal baseline | MRI+table late fusion、pooling 前空间融合 | 证明 table 与交互位置价值 |
| MRI UDA baseline | FMM、amplitude/phase transport、GRL、residual、source-free residual | 与多模态 UDA 的匹配图像域适应比较 |
| multimodal UDA | joint alignment、MRI-specific alignment | 普通多模态 UDA 对照 |
| proposed | cross-modal relation-preserving UDA | 主方法候选 |
| shortcut controls | shuffled-tabular、missingness-only、table-only | 主消融或补充表的必要负控 |

既有 DS-034 至 DS-041 结果可以作为历史背景；若要把数值放入新的主表，需在统一 protocol、split、selector 和 target access 下重跑或确认严格可比。表格位置可以升级为主表消融，但不能改变历史结果的原始解释边界。

## 6. 与主地图的关系

本说明对应：

- `docs/DUALSHIFT_MULTIMODAL_UDA_MASTER_MAP_2026-08-29.md` 的数据合同；
- `docs/DUALSHIFT_MULTIMODAL_UDA_PHASE1_EXECUTION_PLAN_2026-08-29.md` 的主表和消融矩阵；
- scan-filtered 数据现实与方向不对称约束。

本文不要求先完成一套独立的“batch effect 证明”。batch/scanner 诊断只在 DS-042 主表结果产生后作为解释列或机制分析列报告。
