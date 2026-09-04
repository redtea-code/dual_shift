# DS-042 实验报告：MRI+table 跨模态关系保持的无监督域适应

> 实验编号：DS-042
> 主方向：ADNI → NACC
> 任务：scan-filtered MCI vs AD
> 报告状态：3 个正式 seed 已完成（42 / 43 / 44）
> 数据汇总日期：2026-09-01

## 1. 摘要

DS-042 评估在 target 主任务标签不可见、但 target MRI 与部署时可获得的表格协变量可见条件下，MRI+table 多模态无监督域适应（UDA）是否能够改善 NACC 目标域判别，并检验跨模态关系保持机制。实验使用固定的 `ADNI_to_NACC` 方向、subject-disjoint 的 target adaptation/test 划分、三个随机种子和统一 checkpoint 选择规则。

主要结果应以 target subject-level AUROC 和 balanced accuracy 为主：完整 CMRP-UDA 的三 seed 均值为 AUROC **0.7311 ± 0.0172**、BA **0.6515 ± 0.0487**、Macro-F1 **0.6443 ± 0.0515**。匹配的多模态 source-only 基线为 AUROC **0.7362 ± 0.0131**、BA **0.6763 ± 0.0281**、Macro-F1 **0.6467 ± 0.0386**。因此，在本次协议和实现下，CMRP-UDA 没有相对 source-only 基线显示出 target AUROC 或 BA 的平均提升；不应宣称 UDA 已改善目标域性能。

## 2. 实验协议

| 项目 | 设定 |
|---|---|
| 数据方向 | `ADNI_to_NACC` |
| 任务 | scan-filtered MCI vs AD，二分类 |
| 输入 | MRI + `age, sex, education` |
| target adaptation | 无标签 MRI + table；不读取 target label |
| target 评估 | 冻结 checkpoint 后的 target test，subject-level aggregation |
| target subjects | 263 |
| seeds | 42、43、44 |
| 变体数 | 13 |
| checkpoint | source validation loss / score 选择，按 seed 独立确定 |
| 主要终点 | target AUROC、balanced accuracy |
| 辅助终点 | accuracy、Macro-F1、sensitivity、specificity、Brier、ECE |

协议边界：target label、prediction、metric 和模型排名不进入 adaptation 或 selector；`field_strength`、site、manufacturer 和 sequence 仅用于支持/解释分析，不作为默认模型输入。

## 3. 运行完整性

- seed 42、43、44 的训练日志均记录 `[journal] completed`。
- 每个 seed 均生成 13 个变体的 `journal_metrics.json`，共 **39** 份正式指标文件。
- 每个变体均包含 target prediction 输出；本报告只使用冻结 checkpoint 后的 target `metrics`，不使用 `metrics_scan` 作为主结果。
- target split manifest 显示 adaptation fraction 为 0.5，且 target 总 subjects 为 263。
- 结果来源目录：`ds042_runs/ADNI_to_NACC/seed_{42,43,44}/cmrp_*/`。

## 4. Target 主结果（subject-level）

数值为 3 个 seed 的 `均值 ± 标准差`。

| 变体 | AUROC | BA | ACC | Macro-F1 | Sensitivity | Specificity | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cmrp_both_specific`（MRI/table 双 specific 对齐） | 0.7386 ± 0.0148 | 0.6336 ± 0.0398 | 0.5526 ± 0.0944 | 0.5442 ± 0.0909 | 0.8627 ± 0.1258 | 0.4045 ± 0.1975 | 0.3618 ± 0.0781 | 0.3338 ± 0.0720 |
| `cmrp_joint_uda`（联合对齐消融） | 0.7363 ± 0.0193 | 0.6277 ± 0.0241 | 0.6527 ± 0.1110 | 0.5968 ± 0.0684 | 0.5569 ± 0.3107 | 0.6985 ± 0.3090 | 0.2871 ± 0.1039 | 0.2440 ± 0.1263 |
| `cmrp_missingness_only`（missingness-only 负控） | 0.6953 ± 0.0403 | 0.5962 ± 0.0545 | 0.6337 ± 0.0881 | 0.5569 ± 0.0714 | 0.4902 ± 0.3710 | 0.7022 ± 0.2955 | 0.2793 ± 0.0923 | 0.2207 ± 0.1347 |
| `cmrp_mri_specific`（MRI-specific 对齐消融） | 0.7020 ± 0.0491 | 0.6146 ± 0.0464 | 0.6350 ± 0.1042 | 0.5645 ± 0.0710 | 0.5569 ± 0.3911 | 0.6723 ± 0.3325 | 0.3008 ± 0.0913 | 0.2677 ± 0.1009 |
| `cmrp_no_alignment`（去除 alignment） | 0.6778 ± 0.0298 | 0.6195 ± 0.0397 | 0.7110 ± 0.0212 | 0.6180 ± 0.0533 | 0.3608 ± 0.1494 | 0.8783 ± 0.0822 | 0.2088 ± 0.0085 | 0.1085 ± 0.0155 |
| `cmrp_no_identity`（去除 identity/proximal） | 0.6932 ± 0.0183 | 0.6084 ± 0.0195 | 0.7085 ± 0.0096 | 0.6117 ± 0.0234 | 0.3255 ± 0.0475 | 0.8914 ± 0.0086 | 0.2305 ± 0.0167 | 0.1776 ± 0.0426 |
| `cmrp_no_source_relation`（去除 source relation） | 0.7231 ± 0.0217 | 0.5728 ± 0.0719 | 0.5729 ± 0.1282 | 0.4807 ± 0.1087 | 0.5725 ± 0.4847 | 0.5730 ± 0.4024 | 0.3681 ± 0.0755 | 0.3528 ± 0.0785 |
| `cmrp_no_target_relation`（去除 target relation） | 0.6905 ± 0.0309 | 0.6365 ± 0.0486 | 0.6882 ± 0.0397 | 0.6207 ± 0.0501 | 0.4902 ± 0.2684 | 0.7828 ± 0.1798 | 0.2646 ± 0.0590 | 0.2243 ± 0.0885 |
| `cmrp_shuffled_table`（shuffled table 负控） | 0.7263 ± 0.0181 | 0.6148 ± 0.0395 | 0.5437 ± 0.1233 | 0.5219 ± 0.1112 | 0.8157 ± 0.2480 | 0.4139 ± 0.2952 | 0.3937 ± 0.1670 | 0.3494 ± 0.2308 |
| `cmrp_source_only`（多模态 source-only 基线） | 0.7362 ± 0.0131 | 0.6763 ± 0.0281 | 0.6603 ± 0.0457 | 0.6467 ± 0.0386 | 0.7216 ± 0.0359 | 0.6311 ± 0.0803 | 0.2769 ± 0.0296 | 0.2351 ± 0.0405 |
| `cmrp_table_only`（table-only 负控） | 0.4993 ± 0.0393 | 0.4723 ± 0.0238 | 0.3676 ± 0.0516 | 0.3403 ± 0.0742 | 0.7686 ± 0.2073 | 0.1760 ± 0.1720 | 0.3216 ± 0.0149 | 0.2809 ± 0.0238 |
| `cmrp_table_specific`（table-specific 对齐消融） | 0.7416 ± 0.0118 | 0.6649 ± 0.0088 | 0.6337 ± 0.0557 | 0.6218 ± 0.0438 | 0.7529 ± 0.1273 | 0.5768 ± 0.1428 | 0.2990 ± 0.0624 | 0.2645 ± 0.0601 |
| `cmrp_uda`（CMRP-UDA（完整方法）） | 0.7311 ± 0.0172 | 0.6515 ± 0.0487 | 0.7085 ± 0.0478 | 0.6443 ± 0.0515 | 0.4902 ± 0.2294 | 0.8127 ± 0.1614 | 0.2517 ± 0.0434 | 0.2282 ± 0.0642 |

## 5. 核心比较

### 5.1 完整 CMRP-UDA 与 source-only

| 指标 | source-only | CMRP-UDA | 差值（CMRP-UDA − source-only） |
|---|---:|---:|---:|
| auc | 0.7362 | 0.7311 | -0.0052 |
| balanced_accuracy | 0.6763 | 0.6515 | -0.0249 |
| accuracy | 0.6603 | 0.7085 | +0.0482 |
| macro_f1 | 0.6467 | 0.6443 | -0.0024 |
| sensitivity | 0.7216 | 0.4902 | -0.2314 |
| specificity | 0.6311 | 0.8127 | +0.1816 |
| brier | 0.2769 | 0.2517 | -0.0252 |
| ece | 0.2351 | 0.2282 | -0.0069 |

解释：完整 CMRP-UDA 相比 source-only 的 target AUROC 平均变化为 **−0.0051**，BA 平均变化为 **−0.0248**，Macro-F1 平均变化为 **−0.0024**。三个指标均未显示平均改善，且该结论只适用于本实验的固定协议。

### 5.2 关系保持与负控

`cmrp_shuffled_table` 的 target AUROC 为 **0.7263 ± 0.0181**，低于真实配对的 CMRP-UDA（0.7311 ± 0.0172），但差距较小；其 BA（0.6148 ± 0.0395）和 Macro-F1（0.5219 ± 0.1112）明显低于完整方法。该结果与真实 MRI-table 配对具有价值的方向一致，但不能单独证明 relation loss 是唯一原因。

`cmrp_table_only` 的 target AUROC 为 **0.4993 ± 0.0393**，BA 为 **0.4723 ± 0.0238**，说明仅依赖 age/sex/education 不能提供与 MRI+table 模型相当的判别性能。`cmrp_missingness_only` 的 AUROC 为 **0.6953 ± 0.0403**，低于 source-only 和完整 CMRP-UDA，表明缺失模式本身不能解释主要结果。

### 5.3 组件消融

- 去除 alignment：AUROC **0.6778 ± 0.0298**，BA **0.6195 ± 0.0397**；相比完整方法下降。
- 去除 source relation：AUROC **0.7231 ± 0.0217**，BA **0.5728 ± 0.0719**；BA 和 Macro-F1 明显下降，提示 source paired relation 约束可能重要。
- 去除 target relation：AUROC **0.6905 ± 0.0309**，BA **0.6365 ± 0.0486**；相对完整方法 AUROC 下降。
- 去除 identity/proximal：AUROC **0.6932 ± 0.0183**，BA **0.6084 ± 0.0195**；整体低于完整方法。
- joint alignment：AUROC **0.7363 ± 0.0193**，BA **0.6277 ± 0.0241**；AUROC 接近 source-only，但 BA 低于 source-only。

这些是跨 seed 的描述性比较；正式显著性结论应使用预注册的 paired seed difference / bootstrap CI，并避免以单个指标或单个 seed 排名。

## 6. 每个 seed 的 target 结果

| 变体 | seed 42 AUROC / BA / F1 | seed 43 AUROC / BA / F1 | seed 44 AUROC / BA / F1 |
|---|---:|---:|---:|
| `cmrp_both_specific` | 0.7464 / 0.6335 / 0.5214 | 0.7215 / 0.5939 / 0.4669 | 0.7479 / 0.6734 / 0.6444 |
| `cmrp_joint_uda` | 0.7367 / 0.6212 / 0.5234 | 0.7554 / 0.6075 / 0.6082 | 0.7168 / 0.6544 / 0.6587 |
| `cmrp_missingness_only` | 0.6510 / 0.5338 / 0.5014 | 0.7054 / 0.6342 / 0.6375 | 0.7296 / 0.6207 / 0.5319 |
| `cmrp_mri_specific` | 0.7541 / 0.6217 / 0.5143 | 0.6565 / 0.5650 / 0.5334 | 0.6956 / 0.6570 / 0.6457 |
| `cmrp_no_alignment` | 0.6436 / 0.5773 / 0.5600 | 0.6915 / 0.6252 / 0.6291 | 0.6983 / 0.6561 / 0.6648 |
| `cmrp_no_identity` | 0.6942 / 0.6055 / 0.6086 | 0.6744 / 0.5906 / 0.5900 | 0.7110 / 0.6292 / 0.6365 |
| `cmrp_no_source_relation` | 0.7189 / 0.5061 / 0.4252 | 0.7465 / 0.5633 / 0.4108 | 0.7038 / 0.6489 / 0.6060 |
| `cmrp_no_target_relation` | 0.6549 / 0.5803 / 0.5660 | 0.7070 / 0.6653 / 0.6318 | 0.7097 / 0.6638 / 0.6643 |
| `cmrp_shuffled_table` | 0.7472 / 0.6368 / 0.5158 | 0.7155 / 0.6383 / 0.6361 | 0.7164 / 0.5692 / 0.4139 |
| `cmrp_source_only` | 0.7441 / 0.7007 / 0.6849 | 0.7212 / 0.6456 / 0.6078 | 0.7434 / 0.6827 / 0.6475 |
| `cmrp_table_only` | 0.4707 / 0.4501 / 0.4183 | 0.4830 / 0.4695 / 0.2706 | 0.5442 / 0.4974 / 0.3320 |
| `cmrp_table_specific` | 0.7465 / 0.6577 / 0.5741 | 0.7281 / 0.6747 / 0.6603 | 0.7501 / 0.6622 / 0.6308 |
| `cmrp_uda` | 0.7143 / 0.5957 / 0.5927 | 0.7302 / 0.6853 / 0.6957 | 0.7487 / 0.6734 / 0.6444 |

## 7. 结论

1. DS-042 正式三 seed 运行已完成，运行文件和 target 指标齐全。
2. 在当前 scan-filtered `ADNI_to_NACC` 协议下，完整 CMRP-UDA 没有相对匹配的 multimodal source-only 基线带来 target AUROC 或 BA 的平均提升；因此主方法的“改善 target 判别”假设未被本次结果支持。
3. 去除 source relation、target relation、alignment 或 identity/proximal 后，多个主要指标下降，说明这些组件可能共同影响稳定性；但不能仅凭本实验区分每个损失项的独立因果贡献。
4. shuffled-table 和 table-only 对照总体弱于真实 MRI+table 设置，支持真实配对和 MRI 信息具有任务价值，但不等于证明跨模态关系保持机制已经被充分验证。
5. 结果不支持“已经去除 scanner/batch effect”“shared 表示是纯 biological information”或“specific 表示是纯 batch information”等更强表述。

## 8. 局限与后续工作

- 当前报告覆盖 DS-042 的 `ADNI_to_NACC` 方向，不应外推到 `NACC_to_ADNI`。
- 目标域只有 263 个 subject；按 seed 的点估计波动仍然存在，尤其 sensitivity/specificity。
- CMRP-UDA 的 source-only 对照虽然使用同一模型族和协议，但 target 结果未显示适配收益，后续应检查 adaptation 强度、checkpoint selector 与 relation-loss 权重的预注册敏感性。
- 需要将本报告与同一 protocol 下的 image-only UDA 基线合并，才能完成论文主比较表。
- 所有最终论文数字应保留 seed、split revision、label-blind manifest 和结果文件路径。

## 9. 结果文件索引

- 实验计划：[DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md](../../DS-042_MULTIMODAL_UDA_EXPERIMENT_PLAN_2026-08-29.md)
- 研究主地图：[DUALSHIFT_MULTIMODAL_UDA_MASTER_MAP_2026-08-29.md](../../DUALSHIFT_MULTIMODAL_UDA_MASTER_MAP_2026-08-29.md)
- 运行日志、各变体指标和 split/provenance 文件保留在服务器端运行目录，不纳入 Git。
